"""Train-inner fold builder scaffolds for N08 section 8.2.

Three allowed fold modes:
  - ``rolling_origin_folds``        sliding-origin train / inner-validation
  - ``purged_time_series_folds``    train rows whose label horizon overlaps
                                    inner-validation are dropped
  - ``embargoed_train_inner_folds``  purged + an additional embargo gap

Section 8.2 requirements (also AGENTS.md section 4.1):
  - split per ticker chronologically before pooling
  - no fold may train on a label horizon overlapping its inner-validation
  - no input window may cross trading-day boundaries
  - no input window may cross ticker boundaries
  - preprocessing statistics fit on train-inner-fit rows only

These builders return iterables of ``(train_inner_fit_idx, train_inner_val_idx)``
pairs in chronological order. Idx are positional into the pooled, per-ticker
sorted sample frame.

Substantive bodies are the second half of N08 task #4.
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
    """Section 8.2 ``rolling_origin_folds`` scaffold."""
    raise NotImplementedError(
        "rolling_origin_folds is a scaffold; N08 task #4 half 2."
    )


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
