"""Numpy-faced sliding-window builder for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-windows-design.md.
"""

from __future__ import annotations

import numpy as np

_BINARY = np.array([0, 1], dtype=np.int8)
_LABEL_DOMAIN = np.array([0, 1, -1], dtype=np.int8)
_DATETIME_NS = np.dtype("datetime64[ns]")
_ZERO_NS = np.timedelta64(0, "ns")


def _validate_core_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> int:
    """Fail-loud validation shared by both public builders.

    Returns the row count ``n`` on success.
    """
    aux = {
        "labels": labels,
        "timestamps": timestamps,
        "partition": partition,
        "feature_valid_mask": feature_valid_mask,
        "target_valid_mask": target_valid_mask,
    }
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be np.ndarray; got {type(features).__name__}")
    for name, arr in aux.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray; got {type(arr).__name__}")

    # bool is a subclass of int -> check bool FIRST.
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError(
            f"window_size must be int (not bool); got {type(window_size).__name__}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0; got {window_size}")

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got ndim={features.ndim}")
    if features.shape[1] < 1:
        raise ValueError(f"features must have >= 1 column; got F={features.shape[1]}")
    for name, arr in aux.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got ndim={arr.ndim}")

    n = features.shape[0]
    for name, arr in aux.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"{name} length {arr.shape[0]} != features rows {n}"
            )

    if features.dtype != np.float64:
        raise ValueError(f"features must be float64; got {features.dtype}")
    if labels.dtype != np.int8:
        raise ValueError(f"labels must be int8; got {labels.dtype}")
    if timestamps.dtype != _DATETIME_NS:
        raise ValueError(f"timestamps must be datetime64[ns]; got {timestamps.dtype}")
    if partition.dtype != np.int8:
        raise ValueError(f"partition must be int8; got {partition.dtype}")
    if feature_valid_mask.dtype != np.bool_:
        raise ValueError(
            f"feature_valid_mask must be bool; got {feature_valid_mask.dtype}"
        )
    if target_valid_mask.dtype != np.bool_:
        raise ValueError(
            f"target_valid_mask must be bool; got {target_valid_mask.dtype}"
        )

    if not np.isin(partition, _BINARY).all():
        raise ValueError("partition values must be in {0, 1}")

    # Label domain: every label must be in {0, 1, -1} (spec 2.2). Without this
    # an invalid-target row could carry an arbitrary int8 (e.g. 99) and slip
    # through the target-valid-only pre-pass below.
    in_domain = np.isin(labels, _LABEL_DOMAIN)
    if not in_domain.all():
        first = int(np.flatnonzero(~in_domain)[0])
        raise ValueError(
            "labels must be in {0, 1, -1}; "
            f"row {first} has label {int(labels[first])}"
        )

    # Label-contract pre-pass: a target-valid row must carry a binary label.
    bad = target_valid_mask & ~np.isin(labels, _BINARY)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            "target_valid_mask=True rows must have labels in {0, 1}; "
            f"row {first} has label {int(labels[first])}"
        )
    return n


def _empty_core_result(window_size: int, n_features: int) -> dict[str, np.ndarray]:
    """Empty output schema with exact dtypes (shared by both builders)."""
    return {
        "X": np.empty((0, window_size, n_features), dtype=np.float64),
        "y": np.empty((0,), dtype=np.int8),
        "target_partition": np.empty((0,), dtype=np.int8),
        "target_timestamps": np.empty((0,), dtype=_DATETIME_NS),
        "target_row_positions": np.empty((0,), dtype=np.int64),
    }


def build_windows_single_ticker(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Build stride-1 same-day sliding windows for a single ticker.

    Args:
        features: ``(n, F)`` float64 feature matrix, ``F >= 1``. Finiteness is
            NOT re-checked here; the caller asserts it via ``feature_valid_mask``.
        labels: ``(n,)`` int8 in ``{0, 1, -1}``; ``-1`` only where
            ``target_valid_mask`` is False.
        timestamps: ``(n,)`` datetime64[ns], tz-naive, nondecreasing.
        partition: ``(n,)`` int8 in ``{0, 1}`` (0=train, 1=validation).
        feature_valid_mask: ``(n,)`` bool; True iff that row's features are usable.
        target_valid_mask: ``(n,)`` bool; True iff that row may be a target.
        window_size: positive int (not bool); window length, stride is 1.

    Returns:
        dict with keys ``X`` (W, window_size, F) float64, ``y`` (W,) int8 in
        ``{0, 1}``, ``target_partition`` (W,) int8, ``target_timestamps`` (W,)
        datetime64[ns], ``target_row_positions`` (W,) int64 (positions into the
        input arrays). ``W`` may be 0.

    Raises:
        TypeError / ValueError: see the spec error-mode table.
    """
    n = _validate_core_inputs(
        features, labels, timestamps, partition,
        feature_valid_mask, target_valid_mask, window_size,
    )
    n_features = features.shape[1]

    if n > 1 and (np.diff(timestamps) < _ZERO_NS).any():
        raise ValueError("timestamps must be nondecreasing")

    if n < window_size:
        return _empty_core_result(window_size, n_features)

    dates = timestamps.astype("datetime64[D]")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.int8] = []
    part_rows: list[np.int8] = []
    ts_rows: list[np.datetime64] = []
    pos_rows: list[int] = []

    for end_pos in range(window_size - 1, n):
        sl = slice(end_pos - window_size + 1, end_pos + 1)
        if not (dates[sl] == dates[end_pos]).all():
            continue  # window crosses a trading-day boundary
        if not (partition[sl] == partition[end_pos]).all():
            raise ValueError(
                "same-day window has non-uniform partition at end_pos "
                f"{end_pos} (timestamp {timestamps[end_pos]}); the caller's "
                "partition is not date-aligned"
            )
        if not feature_valid_mask[sl].all():
            continue
        if not target_valid_mask[end_pos]:
            continue
        x_rows.append(features[sl])
        y_rows.append(labels[end_pos])
        part_rows.append(partition[end_pos])
        ts_rows.append(timestamps[end_pos])
        pos_rows.append(end_pos)

    if not x_rows:
        return _empty_core_result(window_size, n_features)

    return {
        "X": np.stack(x_rows).astype(np.float64, copy=False),
        "y": np.asarray(y_rows, dtype=np.int8),
        "target_partition": np.asarray(part_rows, dtype=np.int8),
        "target_timestamps": np.asarray(ts_rows, dtype=_DATETIME_NS),
        "target_row_positions": np.asarray(pos_rows, dtype=np.int64),
    }


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Pooled multi-ticker wrapper around ``build_windows_single_ticker``.

    Groups rows by ``ticker_ids`` (sorted-unique order), runs the single-ticker
    core per group, remaps ``target_row_positions`` back to positions into the
    pooled input arrays, and concatenates the per-ticker blocks (per-ticker
    block order, block-internal ascending by target timestamp).

    ``ticker_ids`` may be numeric, string, or homogeneous object dtype; a mixed
    incomparable object array is rejected. All other arguments and the output
    schema match the core, with one extra key ``target_ticker_ids`` (dtype
    equal to ``ticker_ids.dtype``).
    """
    n = _validate_core_inputs(
        features, labels, timestamps, partition,
        feature_valid_mask, target_valid_mask, window_size,
    )
    if not isinstance(ticker_ids, np.ndarray):
        raise TypeError(
            f"ticker_ids must be np.ndarray; got {type(ticker_ids).__name__}"
        )
    if ticker_ids.ndim != 1:
        raise ValueError(f"ticker_ids must be 1-D; got ndim={ticker_ids.ndim}")
    if ticker_ids.shape[0] != n:
        raise ValueError(
            f"ticker_ids length {ticker_ids.shape[0]} != features rows {n}"
        )
    n_features = features.shape[1]

    def _empty_pooled() -> dict[str, np.ndarray]:
        res = _empty_core_result(window_size, n_features)
        res["target_ticker_ids"] = np.empty((0,), dtype=ticker_ids.dtype)
        return res

    if n == 0:
        return _empty_pooled()

    # Fail-loud on non-self-comparable ids (e.g. float NaN): grouping by
    # `ticker_ids == ticker` would silently drop such rows (NaN != NaN),
    # losing data and weakening the no-cross-ticker guarantee.
    if not (ticker_ids == ticker_ids).all():
        raise ValueError(
            "ticker_ids contains entries that do not equal themselves "
            "(e.g. NaN); every ticker id must be self-comparable"
        )

    try:
        unique_tickers = np.unique(ticker_ids)
    except TypeError as exc:
        raise ValueError(
            f"ticker_ids must be homogeneous comparable dtype; "
            f"got {ticker_ids.dtype} ({exc})"
        ) from exc

    blocks: list[dict[str, np.ndarray]] = []
    for ticker in unique_tickers:
        global_pos = np.flatnonzero(ticker_ids == ticker)
        try:
            block = build_windows_single_ticker(
                features[global_pos],
                labels[global_pos],
                timestamps[global_pos],
                partition=partition[global_pos],
                feature_valid_mask=feature_valid_mask[global_pos],
                target_valid_mask=target_valid_mask[global_pos],
                window_size=window_size,
            )
        except ValueError as exc:
            raise ValueError(f"ticker {ticker!r}: {exc}") from exc
        w = block["y"].shape[0]
        if w == 0:
            continue
        block["target_row_positions"] = global_pos[block["target_row_positions"]]
        block["target_ticker_ids"] = np.full(w, ticker, dtype=ticker_ids.dtype)
        blocks.append(block)

    if not blocks:
        return _empty_pooled()

    keys = (
        "X", "y", "target_partition", "target_timestamps",
        "target_row_positions", "target_ticker_ids",
    )
    return {key: np.concatenate([b[key] for b in blocks]) for key in keys}
