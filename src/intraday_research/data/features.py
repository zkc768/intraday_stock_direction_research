"""Numpy-faced feature builder for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from intraday_research.baseline_v1 import add_baseline_v1_features


# Frozen feature sets from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md.
# Any change here MUST be accompanied by an update to the freeze
# document; tests in Task 7 lock the values verbatim.
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "price_action_core": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
    ),
    "technical_price": (
        "log_return",
        "high_low_range",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
    ),
    "price_volume_time": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
        "rolling_volatility_20",
        "normalized_volume_20",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
        "time_of_day_sin",
        "time_of_day_cos",
    ),
}

_REQUIRED_BASE_COLUMNS: frozenset[str] = frozenset({
    "ticker", "timestamp", "open", "high", "low", "close", "volume",
})


def build_features(
    frame: pd.DataFrame,
    *,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap baseline_v1.add_baseline_v1_features with a numpy-faced API.

    Args:
        frame: single-ticker DataFrame with columns
            ``{ticker, timestamp, open, high, low, close, volume}``;
            ``timestamp`` is ``datetime64[ns]``, timezone-naive, and
            sorted ascending.
        feature_set: one of ``"price_action_core"``, ``"technical_price"``,
            or ``"price_volume_time"``.

    Returns:
        ``(features, valid_mask)`` where:
          - ``features`` is ``float64`` shape
            ``(len(frame), len(FEATURE_SETS[feature_set]))``; column
            order verbatim matches ``FEATURE_SETS[feature_set]`` tuple.
          - ``valid_mask`` is ``bool_`` shape ``(len(frame),)``; True
            iff EVERY column at row ``t`` is finite. Reserved for
            derived-feature NaN (warmup, denominator-zero); raw OHLCV
            NaN/inf is rejected fail-loud by ``baseline_v1`` before
            this mask is computed.
    """
    # ---- Step 1: validate feature_set ----
    if not isinstance(feature_set, str):
        raise TypeError(
            f"feature_set must be a str; got {type(feature_set).__name__}"
        )
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {sorted(FEATURE_SETS)}; "
            f"got {feature_set!r}"
        )

    # ---- Step 2: validate frame ----
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be pd.DataFrame; got {type(frame).__name__}"
        )
    missing = sorted(_REQUIRED_BASE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"frame missing required columns: {missing}"
        )
    ts = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError(
            "frame['timestamp'] must be datetime64; "
            f"got {ts.dtype}"
        )
    # NOTE: do NOT use pd.api.types.is_datetime64tz_dtype(ts). It is
    # deprecated in recent pandas versions and emits a DeprecationWarning;
    # the project's pytest.ini turns Warnings from intraday_research.*
    # into errors, which would break this code at test time. Use the
    # supported isinstance check on pd.DatetimeTZDtype instead.
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"frame['timestamp'] must be timezone-naive; got tz={ts.dt.tz}"
        )
    if ts.isna().any():
        raise ValueError("frame['timestamp'] contains NaT")
    if not ts.is_monotonic_increasing:
        raise ValueError("frame['timestamp'] must be sorted ascending")

    cols = FEATURE_SETS[feature_set]
    k = len(cols)

    # ---- Step 3: n=0 short-circuit (BEFORE delegating to baseline_v1) ----
    # baseline_v1._require_single_ticker_frame would otherwise raise
    # "Expected a single ticker frame." on an empty frame because
    # frame['ticker'].nunique(dropna=True) == 0 != 1. We handle the
    # empty case here so callers get a clean empty-array return.
    if len(frame) == 0:
        return (
            np.empty((0, k), dtype=np.float64),
            np.empty((0,), dtype=np.bool_),
        )

    # ---- Step 4: delegate to baseline_v1 ----
    # Validates single-ticker uniqueness, OHLCV value sanity (high>=low,
    # open/close in [low, high], positive prices, non-negative volume,
    # raw OHLCV NaN/inf fail-loud). Returns the input frame with all 10
    # feature columns appended.
    enriched = add_baseline_v1_features(frame)

    # ---- Step 5: select feature subset in canonical column order ----
    features_df = enriched[list(cols)]

    # ---- Step 6: convert to ndarray ----
    features = features_df.to_numpy(dtype=np.float64)

    # ---- Step 7: compute row-level valid_mask ----
    valid_mask = np.isfinite(features).all(axis=1).astype(np.bool_)

    return features, valid_mask


def build_price_action_core_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """price_action_core frozen alias (3 features)."""
    return build_features(frame, feature_set="price_action_core")


def build_technical_price_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """technical_price frozen alias (5 features)."""
    return build_features(frame, feature_set="technical_price")


def build_price_volume_time_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """price_volume_time frozen alias (10 features; Stage 0 default)."""
    return build_features(frame, feature_set="price_volume_time")
