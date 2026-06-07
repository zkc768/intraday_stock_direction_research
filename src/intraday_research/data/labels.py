"""No-trade-band binary labels for the frozen Stage 0 candidate space.

Wraps ``intraday_research.baseline_v1.make_no_trade_band_labels`` with a
numpy-faced API. ``baseline_v1`` is the single source of truth for the
label semantics; this module's only job is the numpy <-> pandas glue and
input validation. Cross-split invalidation is deferred to #5C-4.

Frozen configurations (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md):

  - ``h03_bps1p5``  horizon_k=3,  threshold_bps=1.5
  - ``h09_bps3p0``  horizon_k=9,  threshold_bps=3.0
  - ``h24_bps7p5``  horizon_k=24, threshold_bps=7.5

See docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from intraday_research.baseline_v1 import make_no_trade_band_labels


H03_BPS1P5: dict = {"horizon_k": 3, "threshold_bps": 1.5}
H09_BPS3P0: dict = {"horizon_k": 9, "threshold_bps": 3.0}
H24_BPS7P5: dict = {"horizon_k": 24, "threshold_bps": 7.5}


def build_no_trade_band_labels(
    close: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon_k: int,
    threshold_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap ``baseline_v1.make_no_trade_band_labels`` with a numpy-faced API.

    Args:
        close: 1-D ``float64`` close prices, per-ticker chronological.
        timestamps: 1-D ``datetime64[ns]`` aligned with ``close``, sorted
            ascending (caller responsibility).
        horizon_k: positive integer bars to look ahead.
        threshold_bps: non-negative finite no-trade-band half-width in bps.

    Returns:
        ``(labels, valid_mask)`` where labels is ``int8`` with values in
        ``{0, 1, -1}`` (``-1`` is the invalid placeholder, never to be
        interpreted as class 0), and valid_mask is ``bool_``. Both have
        ``shape == close.shape`` and position-aligned with the inputs.

    The wrapper does not detect or split multi-ticker input; callers must
    group by ticker before calling.
    """
    close = np.asarray(close)
    timestamps = np.asarray(timestamps)
    if close.ndim != 1 or timestamps.ndim != 1:
        raise ValueError(
            "close and timestamps must be 1-D arrays; got shapes "
            f"{close.shape}, {timestamps.shape}."
        )
    if close.shape != timestamps.shape:
        raise ValueError(
            "close and timestamps must be same length; got "
            f"{close.shape} and {timestamps.shape}."
        )
    if (
        isinstance(horizon_k, bool)
        or not isinstance(horizon_k, int)
        or horizon_k <= 0
    ):
        raise ValueError(
            f"horizon_k must be a positive int; got {horizon_k!r}."
        )
    if isinstance(threshold_bps, bool) or not isinstance(
        threshold_bps, (int, float)
    ):
        raise ValueError(
            f"threshold_bps must be numeric; got {threshold_bps!r}."
        )
    if not math.isfinite(float(threshold_bps)) or threshold_bps < 0:
        raise ValueError(
            "threshold_bps must be non-negative and finite; got "
            f"{threshold_bps!r}."
        )

    n = int(close.shape[0])
    if n == 0:
        return (
            np.array([], dtype=np.int8),
            np.array([], dtype=np.bool_),
        )

    ts = pd.to_datetime(timestamps)
    if pd.Series(ts).isna().any():
        raise ValueError("timestamps contains NaT.")
    if not pd.Series(ts).is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending.")

    frame = pd.DataFrame({
        "ticker": "_anon",
        "timestamp": ts,
        "close": close.astype(float),
    })
    result = make_no_trade_band_labels(
        frame, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )
    label_float = result["label"].to_numpy()
    valid_mask = ~np.isnan(label_float)
    labels = np.where(valid_mask, label_float, -1).astype(np.int8)
    return labels, valid_mask.astype(np.bool_)


def build_h03_bps1p5_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h03_bps1p5 frozen alias (horizon_k=3, threshold_bps=1.5)."""
    return build_no_trade_band_labels(close, timestamps, **H03_BPS1P5)


def build_h09_bps3p0_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h09_bps3p0 frozen alias (horizon_k=9, threshold_bps=3.0)."""
    return build_no_trade_band_labels(close, timestamps, **H09_BPS3P0)


def build_h24_bps7p5_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h24_bps7p5 frozen alias (horizon_k=24, threshold_bps=7.5)."""
    return build_no_trade_band_labels(close, timestamps, **H24_BPS7P5)
