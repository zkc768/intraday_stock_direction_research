"""Train-inner fold builders for N08 section 8.2.

Three allowed fold modes:
  - ``rolling_origin_folds``        expanding-window train / inner-validation
                                    with label-horizon embargo (implemented #5B)
  - ``purged_time_series_folds``    train rows whose label horizon overlaps
                                    inner-validation are dropped (scaffold)
  - ``embargoed_train_inner_folds``  purged + an additional embargo gap
                                    (scaffold)

Section 8.2 requirements (also AGENTS.md section 4.1):
  - split per ticker chronologically before pooling
  - no fold may train on a label horizon overlapping its inner-validation
  - no input window may cross trading-day boundaries  (upstream, window-build)
  - no input window may cross ticker boundaries       (upstream, window-build)
  - preprocessing statistics fit on train-inner-fit rows only (upstream)

These builders return iterables of ``(train_inner_fit_idx, train_inner_val_idx)``
pairs in chronological order. Idx are positional into the pooled, per-ticker
sorted sample frame; the upstream sample frame is responsible for honoring
the trading-day / ticker / preprocessing-statistics invariants.
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


def purged_time_series_folds(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    n_folds: int,
    label_horizon_k: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Section 8.2 ``purged_time_series_folds`` scaffold.

    Drops train-inner-fit rows whose label horizon overlaps the
    train-inner-validation interval.
    """
    raise NotImplementedError(
        "purged_time_series_folds is a scaffold; N08 task #4 half 2."
    )


def embargoed_train_inner_folds(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    n_folds: int,
    label_horizon_k: int,
    embargo_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Section 8.2 ``embargoed_train_inner_folds`` scaffold.

    Purged + an additional embargo gap on both sides of the inner-validation
    block to remove temporal leakage from window overlap.
    """
    raise NotImplementedError(
        "embargoed_train_inner_folds is a scaffold; N08 task #4 half 2."
    )
