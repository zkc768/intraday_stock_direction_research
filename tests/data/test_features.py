"""Behavioral tests for ``intraday_research.data.features`` (N08 #5C-2).

Synthetic-data tests only. No raw bar I/O, no fixture files committed
to the repo, no official validation, no holdout. Verifies the section 4
contract documented in
``docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import add_baseline_v1_features
from intraday_research.data.features import (
    FEATURE_SETS,
    build_features,
    build_price_action_core_features,
    build_price_volume_time_features,
    build_technical_price_features,
)


def _synthetic_intraday_session(
    n: int = 80,
    start: str = "2010-01-04 09:30:00",
    ticker: str = "CSCO",
    drift_bps_per_bar: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """5-min OHLCV bars that pass baseline_v1._validated_ohlcv.

    Default `n=80` is large enough that every feature in price_volume_time
    has warmed up by the end (longest warmup is normalized_macd_hist with
    cumulative EWM 12+26+9 ≈ 47 effective lag).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="5min")
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return pd.DataFrame({
        "ticker": ticker,
        "timestamp": timestamps,
        "open": close * 0.9995,
        "high": close * 1.0005,
        "low": close * 0.9990,
        "close": close,
        "volume": rng.integers(1000, 10_000, n),
    })


def test_price_volume_time_matches_baseline_v1_column_by_column():
    """Anti-drift gate: every feature in price_volume_time matches
    baseline_v1.add_baseline_v1_features value-for-value at every valid row."""
    frame = _synthetic_intraday_session(n=80)
    expected_df = add_baseline_v1_features(frame)
    features, valid_mask = build_features(
        frame, feature_set="price_volume_time"
    )

    cols = FEATURE_SETS["price_volume_time"]
    assert features.shape == (len(frame), len(cols))
    assert valid_mask.shape == (len(frame),)

    # Compare each column at every row; NaNs in expected must coincide
    # with NaNs in features (np.isnan equivalence).
    for col_idx, col_name in enumerate(cols):
        expected_col = expected_df[col_name].to_numpy(dtype=np.float64)
        actual_col = features[:, col_idx]
        # Both NaN at same positions, equal values where both finite.
        np.testing.assert_array_equal(
            np.isnan(expected_col), np.isnan(actual_col),
            err_msg=f"NaN positions differ for column {col_name!r}",
        )
        finite_mask = np.isfinite(expected_col) & np.isfinite(actual_col)
        np.testing.assert_allclose(
            expected_col[finite_mask], actual_col[finite_mask],
            rtol=0.0, atol=0.0,
            err_msg=f"Numeric mismatch for column {col_name!r}",
        )


# ---------------------------------------------------------------------------
# Task 2: feature_set subset selection + alias
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature_set,expected_n_cols", [
    ("price_action_core", 3),
    ("technical_price", 5),
    ("price_volume_time", 10),
])
def test_each_feature_set_returns_correct_shape(feature_set, expected_n_cols):
    frame = _synthetic_intraday_session(n=80)
    features, _ = build_features(frame, feature_set=feature_set)
    assert features.shape == (len(frame), expected_n_cols)


@pytest.mark.parametrize("feature_set", [
    "price_action_core", "technical_price", "price_volume_time",
])
def test_column_order_matches_FEATURE_SETS_tuple_verbatim(feature_set):
    """Sanity: cross-check baseline_v1 column-by-column for THIS feature_set."""
    frame = _synthetic_intraday_session(n=80)
    expected_df = add_baseline_v1_features(frame)
    features, _ = build_features(frame, feature_set=feature_set)
    cols = FEATURE_SETS[feature_set]
    for col_idx, col_name in enumerate(cols):
        expected_col = expected_df[col_name].to_numpy(dtype=np.float64)
        actual_col = features[:, col_idx]
        finite_mask = np.isfinite(expected_col) & np.isfinite(actual_col)
        np.testing.assert_allclose(
            expected_col[finite_mask], actual_col[finite_mask],
            rtol=0.0, atol=0.0,
        )


def test_price_action_core_alias_matches_generic():
    frame = _synthetic_intraday_session(n=40)
    via_alias = build_price_action_core_features(frame)
    via_generic = build_features(frame, feature_set="price_action_core")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_technical_price_alias_matches_generic():
    frame = _synthetic_intraday_session(n=80)
    via_alias = build_technical_price_features(frame)
    via_generic = build_features(frame, feature_set="technical_price")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_price_volume_time_alias_matches_generic():
    frame = _synthetic_intraday_session(n=80)
    via_alias = build_price_volume_time_features(frame)
    via_generic = build_features(frame, feature_set="price_volume_time")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


# ---------------------------------------------------------------------------
# Task 3: warmup qualitative ordering
# ---------------------------------------------------------------------------


def test_initial_rows_invalid_then_valid_rows_appear():
    """n=80 frame: at least some early rows are invalid (warmup not done)
    AND at least some late rows are valid (warmup completed)."""
    frame = _synthetic_intraday_session(n=80)
    _, valid_mask = build_features(
        frame, feature_set="price_volume_time"
    )
    assert not valid_mask[0], "row 0 must be invalid (warmup)"
    assert valid_mask.any(), (
        "at least one row should reach valid_mask=True with n=80; "
        "test fixture may be too short"
    )


def test_price_action_core_reaches_valid_no_later_than_technical_price():
    frame = _synthetic_intraday_session(n=80)
    _, pac_mask = build_features(frame, feature_set="price_action_core")
    _, tp_mask = build_features(frame, feature_set="technical_price")
    pac_first_valid = int(np.argmax(pac_mask)) if pac_mask.any() else None
    tp_first_valid = int(np.argmax(tp_mask)) if tp_mask.any() else None
    assert pac_first_valid is not None
    assert tp_first_valid is not None
    assert pac_first_valid <= tp_first_valid


def test_price_action_core_reaches_valid_no_later_than_price_volume_time():
    frame = _synthetic_intraday_session(n=80)
    _, pac_mask = build_features(frame, feature_set="price_action_core")
    _, pvt_mask = build_features(frame, feature_set="price_volume_time")
    pac_first_valid = int(np.argmax(pac_mask)) if pac_mask.any() else None
    pvt_first_valid = int(np.argmax(pvt_mask)) if pvt_mask.any() else None
    assert pac_first_valid is not None
    assert pvt_first_valid is not None
    assert pac_first_valid <= pvt_first_valid


def test_valid_mask_eventually_becomes_true_for_each_feature_set():
    frame = _synthetic_intraday_session(n=80)
    for feature_set in ("price_action_core", "technical_price", "price_volume_time"):
        _, mask = build_features(frame, feature_set=feature_set)
        assert mask.any(), (
            f"feature_set={feature_set!r} produced no valid rows; "
            f"the n=80 fixture may be insufficient"
        )


# ---------------------------------------------------------------------------
# Task 4: output dtype + shape
# ---------------------------------------------------------------------------


def test_features_dtype_is_float64():
    frame = _synthetic_intraday_session(n=30)
    features, _ = build_features(frame, feature_set="price_volume_time")
    assert features.dtype == np.float64


def test_valid_mask_dtype_is_bool():
    frame = _synthetic_intraday_session(n=30)
    _, valid_mask = build_features(frame, feature_set="price_volume_time")
    assert valid_mask.dtype == np.bool_


def test_features_and_valid_mask_shapes_align_with_input_length():
    frame = _synthetic_intraday_session(n=37)
    features, valid_mask = build_features(
        frame, feature_set="technical_price"
    )
    assert features.shape == (37, 5)
    assert valid_mask.shape == (37,)


# ---------------------------------------------------------------------------
# Task 5: wrapper-layer input guards
# ---------------------------------------------------------------------------


def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="frame must be pd.DataFrame"):
        build_features({"ticker": "CSCO"}, feature_set="price_volume_time")


def test_missing_timestamp_column_raises_with_list():
    frame = _synthetic_intraday_session(n=10).drop(columns=["timestamp"])
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "missing required columns" in msg
    assert "'timestamp'" in msg


def test_missing_ticker_column_raises_clean_value_error_not_keyerror():
    """Locks the wrapper's required-column check running BEFORE
    baseline_v1._require_single_ticker_frame; without this, a missing
    ticker would surface as a downstream KeyError or the less-helpful
    'Expected a ticker column ...' message."""
    frame = _synthetic_intraday_session(n=10).drop(columns=["ticker"])
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "missing required columns" in msg
    assert "'ticker'" in msg


def test_missing_multiple_required_columns_lists_all_sorted():
    frame = (
        _synthetic_intraday_session(n=10)
        .drop(columns=["high", "volume"])
    )
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "'high'" in msg
    assert "'volume'" in msg


def test_timestamp_int_dtype_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame["timestamp"] = np.arange(len(frame), dtype=np.int64)
    with pytest.raises(ValueError, match="must be datetime64"):
        build_features(frame, feature_set="price_volume_time")


def test_tz_aware_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="must be timezone-naive"):
        build_features(frame, feature_set="price_volume_time")


def test_nat_in_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="contains NaT"):
        build_features(frame, feature_set="price_volume_time")


def test_unsorted_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="must be sorted ascending"):
        build_features(frame, feature_set="price_volume_time")


def test_invalid_feature_set_name_raises_with_valid_choices():
    frame = _synthetic_intraday_session(n=10)
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="nonexistent_set")
    msg = str(excinfo.value)
    assert "feature_set must be one of" in msg
    assert "'price_volume_time'" in msg


def test_non_str_feature_set_raises_type_error():
    frame = _synthetic_intraday_session(n=10)
    with pytest.raises(TypeError, match="feature_set must be a str"):
        build_features(frame, feature_set=42)


# ---------------------------------------------------------------------------
# Task 6: delegated guards from baseline_v1
# ---------------------------------------------------------------------------


def test_multi_ticker_frame_raises_via_baseline_v1():
    """baseline_v1._require_single_ticker_frame catches multi-ticker."""
    frame_a = _synthetic_intraday_session(n=10, ticker="CSCO")
    frame_b = _synthetic_intraday_session(
        n=10, ticker="JPM", start="2010-01-04 10:20:00",
    )
    pooled = pd.concat([frame_a, frame_b], ignore_index=True)
    with pytest.raises(ValueError, match="single ticker frame"):
        build_features(pooled, feature_set="price_volume_time")


def test_ohlc_high_less_than_low_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "high"] = 50.0
    frame.loc[3, "low"] = 200.0
    with pytest.raises(ValueError, match="high must be >= low"):
        build_features(frame, feature_set="price_volume_time")


def test_non_positive_price_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "open"] = -1.0
    with pytest.raises(ValueError):
        build_features(frame, feature_set="price_volume_time")


def test_negative_volume_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "volume"] = -100
    with pytest.raises(ValueError):
        build_features(frame, feature_set="price_volume_time")


# ---------------------------------------------------------------------------
# Task 7: edge cases + frozen constant lock
# ---------------------------------------------------------------------------


def test_empty_frame_with_full_schema_returns_empty_arrays():
    """n=0 short-circuits BEFORE baseline_v1 delegation; without the
    short-circuit, baseline_v1._require_single_ticker_frame would raise
    'Expected a single ticker frame.' because nunique=0 != 1."""
    empty = _synthetic_intraday_session(n=0)
    features, valid_mask = build_features(
        empty, feature_set="price_volume_time"
    )
    assert features.shape == (0, 10)
    assert valid_mask.shape == (0,)
    assert features.dtype == np.float64
    assert valid_mask.dtype == np.bool_


def test_empty_frame_missing_required_column_still_raises():
    """Schema check runs BEFORE the n=0 short-circuit, so a missing
    column on an empty frame is still a ValueError."""
    empty = _synthetic_intraday_session(n=0).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(empty, feature_set="price_volume_time")


def test_raw_close_nan_is_fail_loud_via_baseline_v1_not_valid_mask():
    """Raw OHLCV NaN/inf does NOT propagate to valid_mask=False; it
    raises via baseline_v1._validated_ohlcv. Locks the
    'raw vs derived NaN' distinction in spec."""
    frame = _synthetic_intraday_session(n=30).copy()
    frame.loc[5, "close"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_features(frame, feature_set="price_volume_time")


def test_constant_close_produces_bollinger_nan_so_valid_mask_false():
    """With constant close, rolling_std_20 = 0 → Bollinger band width = 0
    → bollinger_denom replaced with NaN → bollinger_pctb = NaN for every
    row. price_action_core (no bollinger) reaches valid_mask=True after
    warmup; technical_price and price_volume_time (both contain
    bollinger_pctb) never reach valid_mask=True. Wrapper does NOT raise."""
    n = 80
    timestamps = pd.date_range("2010-01-04 09:30:00", periods=n, freq="5min")
    frame = pd.DataFrame({
        "ticker": "CSCO",
        "timestamp": timestamps,
        "open": 100.0,
        "high": 100.05,
        "low": 99.95,
        "close": 100.0,
        "volume": 5_000,
    })

    _, pac_mask = build_features(frame, feature_set="price_action_core")
    assert pac_mask.any(), (
        "price_action_core (no bollinger_pctb) should reach valid_mask=True "
        "with constant close once warmup completes"
    )

    _, tp_mask = build_features(frame, feature_set="technical_price")
    assert not tp_mask.any(), (
        "technical_price (contains bollinger_pctb) should never reach "
        "valid_mask=True under constant close because the Bollinger "
        "band width is 0 → bollinger_pctb = NaN for every row"
    )

    _, pvt_mask = build_features(frame, feature_set="price_volume_time")
    assert not pvt_mask.any()


def test_FEATURE_SETS_locked_to_CONFIG_SCREENING_FREEZE_verbatim():
    """Locks the FEATURE_SETS constant against drift. If the freeze
    document changes, this test fails LOUD and forces a coordinated
    update to both this module and the freeze document."""
    assert FEATURE_SETS == {
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
