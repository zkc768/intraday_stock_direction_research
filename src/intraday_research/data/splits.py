"""Chronological train/validation split markers for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_research.data.raw_bars import VAL_END


# Project-frozen Stage 0 train→validation boundary.
# (CONFIG_SCREENING_FREEZE_2026-06-04.md: TRAIN ends 2013-09-16,
# VALIDATION starts 2013-09-16, VALIDATION ends 2017-01-25 = VAL_END.)
VALIDATION_START: pd.Timestamp = pd.Timestamp("2013-09-16")

# Two-partition int8 codes. PARTITION_TRAIN and PARTITION_VALIDATION
# are exported so downstream filters (e.g. #5C-5 window builder) can
# slice the pooled DataFrame by partition without re-deriving the
# numeric values.
PARTITION_TRAIN: np.int8 = np.int8(0)
PARTITION_VALIDATION: np.int8 = np.int8(1)


def apply_chronological_split(
    timestamps: np.ndarray,
    *,
    validation_start: pd.Timestamp,
    val_end: pd.Timestamp,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row partition codes + cross-split valid_mask.

    Args:
        timestamps: 1-D ``datetime64[ns]`` array, timezone-naive,
            sorted ascending.
        validation_start: train→validation boundary (Timestamp,
            timezone-naive). Rows where ``timestamps < validation_start``
            are classified as ``PARTITION_TRAIN``; rows in
            ``[validation_start, val_end)`` are
            ``PARTITION_VALIDATION``.
        val_end: validation end / holdout boundary (Timestamp,
            timezone-naive). Any ``timestamp >= val_end`` raises a
            ValueError (holdout closure fail-loud).
        horizon_k: positive integer label horizon. ``valid_mask[t]`` is
            True iff row ``t + horizon_k`` exists AND
            ``partition[t] == partition[t + horizon_k]``.

    Returns:
        ``(partition, valid_mask)`` where ``partition`` is ``int8``
        shape ``(n,)`` with values in
        ``{PARTITION_TRAIN, PARTITION_VALIDATION}`` and ``valid_mask``
        is ``bool_`` shape ``(n,)``.
    """
    # ---- Step 1: validate validation_start ----
    if not isinstance(validation_start, pd.Timestamp):
        raise TypeError(
            "validation_start must be pd.Timestamp; "
            f"got {type(validation_start).__name__}"
        )
    if validation_start.tzinfo is not None:
        raise ValueError(
            "validation_start must be timezone-naive; "
            f"got tz={validation_start.tzinfo}"
        )

    # ---- Step 2: validate val_end ----
    if not isinstance(val_end, pd.Timestamp):
        raise TypeError(
            f"val_end must be pd.Timestamp; got {type(val_end).__name__}"
        )
    if val_end.tzinfo is not None:
        raise ValueError(
            f"val_end must be timezone-naive; got tz={val_end.tzinfo}"
        )

    # ---- Step 3: validate boundary ordering ----
    if not validation_start < val_end:
        raise ValueError(
            f"validation_start ({validation_start}) must be "
            f"< val_end ({val_end})"
        )

    # ---- Step 4: validate horizon_k ----
    if (
        isinstance(horizon_k, bool)
        or not isinstance(horizon_k, int)
        or horizon_k <= 0
    ):
        raise ValueError(
            f"horizon_k must be a positive int; got {horizon_k!r}"
        )

    # ---- Step 5: validate timestamps ----
    if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1:
        shape = (
            timestamps.shape if isinstance(timestamps, np.ndarray) else None
        )
        raise ValueError(
            f"timestamps must be a 1-D ndarray; got shape {shape}"
        )
    # Spec requires datetime64[ns] specifically; a coarser precision such as
    # datetime64[s]/[D] would silently change comparison granularity, so reject
    # anything that is not exactly nanosecond datetime64.
    if timestamps.dtype != np.dtype("datetime64[ns]"):
        raise ValueError(
            f"timestamps dtype must be datetime64[ns]; got {timestamps.dtype}"
        )
    # NOTE: do NOT use pd.api.types.is_datetime64tz_dtype(...). It is
    # deprecated and emits a DeprecationWarning; pytest.ini turns
    # Warnings from intraday_research.* into errors. Use the supported
    # isinstance check on pd.DatetimeTZDtype instead (same approach as
    # #5C-2 features.py).
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "timestamps must be timezone-naive; "
            f"got tz={timestamps.dtype.tz}"
        )
    if np.isnat(timestamps).any():
        raise ValueError("timestamps contains NaT")
    if timestamps.size > 1 and (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError("timestamps must be sorted ascending")

    # ---- Step 6: HOLDOUT CLOSURE CHECK (fail-loud) ----
    val_end_np = np.datetime64(val_end.to_datetime64(), "ns")
    contam_mask = timestamps >= val_end_np
    if contam_mask.any():
        n_contam = int(contam_mask.sum())
        n_total = int(timestamps.size)
        first_bad = pd.Timestamp(timestamps[contam_mask][0])
        raise ValueError(
            "holdout closure violated; "
            f"first contaminated timestamp={first_bad}; "
            f"rows={n_contam}/{n_total}"
        )

    # ---- Step 7: n=0 short-circuit ----
    n = int(timestamps.size)
    if n == 0:
        return (
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.bool_),
        )

    # ---- Step 8: compute partition (pure numpy) ----
    validation_start_np = np.datetime64(
        validation_start.to_datetime64(), "ns"
    )
    partition = np.where(
        timestamps < validation_start_np,
        PARTITION_TRAIN,
        PARTITION_VALIDATION,
    ).astype(np.int8)

    # ---- Step 9: compute valid_mask ----
    valid_mask = np.zeros(n, dtype=np.bool_)
    if n > horizon_k:
        same_partition = (
            partition[: n - horizon_k] == partition[horizon_k:]
        )
        valid_mask[: n - horizon_k] = same_partition

    return partition, valid_mask


def apply_stage0_chronological_split(
    timestamps: np.ndarray,
    *,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project-frozen Stage 0 alias.

    Equivalent to::

        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=VAL_END,
            horizon_k=horizon_k,
        )

    where ``VAL_END`` is imported from ``intraday_research.data.raw_bars``
    as the single source of truth.
    """
    return apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=VAL_END,
        horizon_k=horizon_k,
    )
