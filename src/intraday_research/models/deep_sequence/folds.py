"""Train-inner fold builders for N08 section 8.2.

Three allowed fold modes:
  - ``rolling_origin_folds``        expanding-window train / inner-validation
                                    with a label-horizon purge at the fold's
                                    validation boundary (implemented #5B)
  - ``purged_time_series_folds``    same purge applied to every train row whose
                                    label horizon reaches inner-validation
                                    (scaffold)
  - ``embargoed_train_inner_folds``  purge plus an extra embargo gap on both
                                    sides of inner-validation (scaffold)

Responsibility layers (a fold builder owns only layer 1):

1. Fold builder (THIS module): split each ticker's samples chronologically,
   then choose ``(train_inner_fit_idx, train_inner_val_idx)``. It enforces the
   label-horizon PURGE: a train row ``t`` is excluded when its forward label
   horizon ``[t, t + label_horizon_k]`` reaches the inner-validation interval
   (``t + label_horizon_k`` lands at or past the validation start), since that
   label is derived from bars the fold validates on. "Overlap" throughout this
   module means exactly this reach. An EMBARGO is a purge plus an additional
   fixed gap; for a tail validation block (rolling-origin) the two coincide.
2. Window builder (upstream, ``data/windows.py`` #5C-5): no input window may
   cross a trading-day or ticker boundary; ``window_size`` and stride are fixed
   there. Fold indices are positional into that already-windowed, per-ticker
   chronologically-ordered sample frame.
3. Preprocessing (upstream): scaling / statistics are fit on train-inner-fit
   rows only (AGENTS.md section 4.1), never on inner-validation or holdout.

These builders return iterables of ``(train_inner_fit_idx, train_inner_val_idx)``
pairs in chronological order. Indices are positional into the pooled, per-ticker
sorted sample frame; the upstream frame is responsible for honoring the
trading-day / ticker / preprocessing-statistics invariants in layers 2-3.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def rolling_origin_folds(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    n_folds: int,
    inner_validation_size: int,
    label_horizon_k: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Per-ticker chronological rolling-origin folds with label-horizon embargo.

    Within each ticker:
      - samples are sorted chronologically by ``timestamps``;
      - the last ``n_folds * inner_validation_size`` samples are partitioned
        into ``n_folds`` contiguous inner-validation slices;
      - for each fold ``i``, train-inner-fit is every sample chronologically
        before the validation slice MINUS the last ``label_horizon_k`` rows
        (their forward label would land inside the validation interval,
        violating AGENTS.md §4.1 leakage rules).

    Folds are then POOLED across tickers: fold ``i``'s yielded
    ``(train_inner_fit_idx, train_inner_val_idx)`` is the union of each
    ticker's fold ``i`` indices, expressed as positional indices into the
    input ``timestamps`` / ``ticker_ids`` arrays. The two index arrays are
    sorted ascending so downstream consumers can rely on a stable order.

    The fold builder does NOT enforce trading-day or ticker window-boundary
    rules; those are window-construction invariants that the upstream
    sample frame must already honor. The fold builder only enforces the
    per-ticker chronological split and the label-horizon embargo.

    Args:
        timestamps: 1-D array of per-sample timestamps. Any numpy
            orderable dtype (``int64``, ``float64``, ``datetime64``) works.
        ticker_ids: 1-D array of per-sample ticker identifiers, aligned
            with ``timestamps``. Any hashable dtype works.
        n_folds: number of rolling-origin folds (>= 1).
        inner_validation_size: per-fold per-ticker validation-window length
            (>= 1).
        label_horizon_k: label horizon length used by the active Stage 0
            freeze (>= 0). The last ``label_horizon_k`` train-row positions
            before each validation slice are excluded from train-inner-fit.

    Yields:
        ``(train_inner_fit_idx, train_inner_val_idx)`` for fold 0,
        1, ..., ``n_folds - 1`` in chronological order. Both arrays
        contain ``np.int64`` positional indices into the input arrays.

    Raises:
        ValueError: on shape mismatch, parameter bounds, or any ticker
            having fewer samples than
            ``n_folds * inner_validation_size + label_horizon_k + 1``.
    """
    timestamps = np.asarray(timestamps)
    ticker_ids = np.asarray(ticker_ids)
    if timestamps.ndim != 1 or ticker_ids.ndim != 1:
        raise ValueError(
            "timestamps and ticker_ids must be 1-D arrays; got shapes "
            f"{timestamps.shape}, {ticker_ids.shape}."
        )
    if timestamps.shape != ticker_ids.shape:
        raise ValueError(
            "timestamps and ticker_ids must have the same length; got "
            f"{timestamps.shape} and {ticker_ids.shape}."
        )
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1; got {n_folds}.")
    if inner_validation_size < 1:
        raise ValueError(
            f"inner_validation_size must be >= 1; got {inner_validation_size}."
        )
    if label_horizon_k < 0:
        raise ValueError(
            f"label_horizon_k must be >= 0; got {label_horizon_k}."
        )

    # Per-ticker chronological positions: global positional indices ordered
    # by timestamp within each ticker. np.unique returns sorted unique
    # values, so iteration order across tickers is deterministic.
    ticker_positions: dict = {}
    for ticker in np.unique(ticker_ids):
        mask = ticker_ids == ticker
        ticker_global = np.where(mask)[0]
        order = np.argsort(timestamps[ticker_global], kind="stable")
        ticker_positions[ticker.item() if hasattr(ticker, "item") else ticker] = (
            ticker_global[order].astype(np.int64, copy=False)
        )

    min_required = n_folds * inner_validation_size + label_horizon_k + 1
    for ticker, positions in ticker_positions.items():
        if len(positions) < min_required:
            raise ValueError(
                f"ticker {ticker!r} has {len(positions)} samples; "
                f"needs at least {min_required} for n_folds={n_folds}, "
                f"inner_validation_size={inner_validation_size}, "
                f"label_horizon_k={label_horizon_k}."
            )

    # For each fold (chronologically earliest-validation first), gather
    # per-ticker (train, val) slices and pool.
    for fold_i in range(n_folds):
        fold_train_chunks: list[np.ndarray] = []
        fold_val_chunks: list[np.ndarray] = []
        for positions in ticker_positions.values():
            m = len(positions)
            # Validation slice: contiguous samples starting at
            # m - (n_folds - fold_i) * inner_validation_size. Fold 0 is the
            # earliest validation window; fold n_folds-1 is the latest.
            val_start = m - (n_folds - fold_i) * inner_validation_size
            val_end = m - (n_folds - fold_i - 1) * inner_validation_size
            # Train cutoff: exclude the last label_horizon_k samples before
            # val so their forward label does not land inside val.
            train_end = val_start - label_horizon_k
            fold_train_chunks.append(positions[:train_end])
            fold_val_chunks.append(positions[val_start:val_end])
        train_idx = np.sort(np.concatenate(fold_train_chunks))
        val_idx = np.sort(np.concatenate(fold_val_chunks))
        yield train_idx, val_idx


def _per_ticker_positions(timestamps: np.ndarray, ticker_ids: np.ndarray) -> dict:
    """Global positional indices grouped by ticker, each chronologically sorted.

    Shared by the purged / embargoed interior-block builders (rolling_origin
    builds its own inline). Ticker iteration order is deterministic (np.unique
    is sorted).
    """
    timestamps = np.asarray(timestamps)
    ticker_ids = np.asarray(ticker_ids)
    if timestamps.ndim != 1 or ticker_ids.ndim != 1:
        raise ValueError(
            "timestamps and ticker_ids must be 1-D arrays; got shapes "
            f"{timestamps.shape}, {ticker_ids.shape}."
        )
    if timestamps.shape != ticker_ids.shape:
        raise ValueError(
            "timestamps and ticker_ids must have the same length; got "
            f"{timestamps.shape} and {ticker_ids.shape}."
        )
    positions: dict = {}
    for ticker in np.unique(ticker_ids):
        mask = ticker_ids == ticker
        ticker_global = np.where(mask)[0]
        order = np.argsort(timestamps[ticker_global], kind="stable")
        key = ticker.item() if hasattr(ticker, "item") else ticker
        positions[key] = ticker_global[order].astype(np.int64, copy=False)
    return positions


def _interior_block_kfold(
    ticker_positions: dict, *, n_folds: int, gap: int, builder_name: str
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Interior-block K-fold with a SYMMETRIC exclusion gap on both sides of each
    validation block.

    Per ticker the chronological positions are tiled into ``n_folds`` contiguous
    rank blocks (``np.array_split``; earlier blocks take the remainder). For each
    fold, validation = block ``i`` ``[a, b)`` and the half-open interval
    ``[a - gap, b + gap)`` is excluded from train. With ``gap = label_horizon_k``
    this is the López de Prado purge (removes train rows whose label
    ``[t, t+k]`` overlaps the validation labels ``[a, b+k-1]`` on EITHER side);
    ``embargoed`` passes ``gap = label_horizon_k + embargo_size``.

    Raises ``ValueError`` if any ticker has fewer than ``n_folds`` samples, or if
    any ticker x fold yields an empty validation or train array after exclusion
    (checked by simulation, so fold 0 / the last fold / large gaps / uneven
    blocks all fail loud rather than silently producing an empty slice).
    """
    blocks_by_ticker: dict = {}
    for ticker, positions in ticker_positions.items():
        m = len(positions)
        if m < n_folds:
            raise ValueError(
                f"{builder_name}: ticker {ticker!r} has {m} samples; needs at "
                f"least n_folds={n_folds} for one validation block per fold."
            )
        blocks_by_ticker[ticker] = np.array_split(np.arange(m, dtype=np.int64), n_folds)

    # Simulate every ticker x fold; reject any empty train/val after exclusion.
    for ticker, positions in ticker_positions.items():
        m = len(positions)
        for fold_i, block in enumerate(blocks_by_ticker[ticker]):
            a = int(block[0])
            b = int(block[-1]) + 1
            lo = max(0, a - gap)
            hi = min(m, b + gap)
            if (b - a) < 1 or (lo + (m - hi)) < 1:
                empty = "val" if (b - a) < 1 else "train"
                raise ValueError(
                    f"{builder_name}: ticker {ticker!r} fold {fold_i} yields an "
                    f"empty {empty} after the symmetric gap={gap} exclusion "
                    f"(m={m}, n_folds={n_folds}); not enough samples."
                )

    for fold_i in range(n_folds):
        train_chunks: list[np.ndarray] = []
        val_chunks: list[np.ndarray] = []
        for ticker, positions in ticker_positions.items():
            m = len(positions)
            block = blocks_by_ticker[ticker][fold_i]
            a = int(block[0])
            b = int(block[-1]) + 1
            lo = max(0, a - gap)
            hi = min(m, b + gap)
            train_chunks.append(np.concatenate([positions[:lo], positions[hi:]]))
            val_chunks.append(positions[a:b])
        yield np.sort(np.concatenate(train_chunks)), np.sort(np.concatenate(val_chunks))


def purged_time_series_folds(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    n_folds: int,
    label_horizon_k: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Section 8.2 ``purged_time_series_folds`` — interior-block K-fold (train-
    inner exploration) with a symmetric label-horizon purge.

    Per ticker, positions are tiled into ``n_folds`` contiguous chronological
    blocks; each block is the inner-validation slice of one fold, and train is the
    other blocks MINUS the symmetric purge ``[a - k, b + k)`` around the
    validation block ``[a, b)``. This removes every train row whose label
    ``[t, t+k]`` overlaps the validation labels ``[a, b+k-1]`` — both the
    train-BEFORE rows whose forward label reaches the block and the train-AFTER
    rows that fall inside a validation row's forward label horizon (AGENTS.md
    §4.1.4). Folds are pooled across tickers; indices are ``np.int64`` positional,
    sorted ascending.

    Args:
        timestamps: 1-D per-sample timestamps (any orderable dtype).
        ticker_ids: 1-D per-sample ticker ids, aligned with ``timestamps``.
        n_folds: number of interior validation blocks (>= 2 — an interior fold
            needs at least one other block to train on).
        label_horizon_k: label horizon length (>= 0); the symmetric purge width.

    Yields:
        ``(train_inner_fit_idx, train_inner_val_idx)`` for folds 0..n_folds-1.

    Raises:
        ValueError: shape mismatch, ``n_folds < 2``, negative ``label_horizon_k``,
            or any ticker x fold with an empty train/val after the purge.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for interior K-fold; got {n_folds}.")
    if label_horizon_k < 0:
        raise ValueError(f"label_horizon_k must be >= 0; got {label_horizon_k}.")
    ticker_positions = _per_ticker_positions(timestamps, ticker_ids)
    yield from _interior_block_kfold(
        ticker_positions,
        n_folds=n_folds,
        gap=label_horizon_k,
        builder_name="purged_time_series_folds",
    )


def embargoed_train_inner_folds(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    n_folds: int,
    label_horizon_k: int,
    embargo_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Section 8.2 ``embargoed_train_inner_folds`` — purged interior-block K-fold
    plus a symmetric embargo gap on both sides of each validation block.

    Identical to ``purged_time_series_folds`` but the excluded interval widens to
    ``[a - (k + embargo_size), b + (k + embargo_size))``: the ``k`` symmetric
    label-purge plus an ``embargo_size`` serial-correlation gap on each side.

    Window-overlap note: this layer-1 builder has no ``window_size``, so it
    guarantees only the label-horizon purge + the requested ``embargo_size`` band.
    To ALSO exclude input-window overlap with the validation block, the caller
    must size ``embargo_size`` so that ``label_horizon_k + embargo_size >=
    window_size - 1`` (a window-construction invariant owned upstream per this
    module's layer-2/3 responsibility split).

    Raises:
        ValueError: shape mismatch, ``n_folds < 2``, negative ``label_horizon_k``
            or ``embargo_size``, or any ticker x fold with an empty train/val.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for interior K-fold; got {n_folds}.")
    if label_horizon_k < 0:
        raise ValueError(f"label_horizon_k must be >= 0; got {label_horizon_k}.")
    if embargo_size < 0:
        raise ValueError(f"embargo_size must be >= 0; got {embargo_size}.")
    ticker_positions = _per_ticker_positions(timestamps, ticker_ids)
    yield from _interior_block_kfold(
        ticker_positions,
        n_folds=n_folds,
        gap=label_horizon_k + embargo_size,
        builder_name="embargoed_train_inner_folds",
    )
