"""Behavioral tests for ``intraday_research.data.labels`` (N08 #5C-1).

Synthetic-data tests only. No raw bar I/O, no fixture files, no official
validation, no holdout. Verifies the section 4 contract documented in
``docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import make_no_trade_band_labels
from intraday_research.data.labels import (
    H03_BPS1P5,
    H09_BPS3P0,
    H24_BPS7P5,
    build_h03_bps1p5_labels,
    build_h09_bps3p0_labels,
    build_h24_bps7p5_labels,
    build_no_trade_band_labels,
)


def _synthetic_intraday_session(
    n: int = 80,
    start: str = "2025-01-02 09:30",
    close_seed: int = 0,
    drift_bps_per_bar: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """5-minute bars within one trading day. Close = 100 * cumulative product
    of (1 + per-bar drift + noise) so labels can be reasoned about by hand."""
    rng = np.random.default_rng(close_seed)
    timestamps = pd.date_range(start, periods=n, freq="5min").to_numpy()
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5  # ~0.1 bps -- below default threshold
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return close.astype(np.float64), timestamps


def test_wrapper_matches_baseline_v1_on_identical_inputs():
    """Anti-drift gate: same numeric output as baseline_v1 directly called."""
    close, timestamps = _synthetic_intraday_session(n=60, drift_bps_per_bar=2.0)
    horizon_k, threshold_bps = 3, 1.5

    # Baseline_v1 path -- the source of truth.
    frame = pd.DataFrame({
        "ticker": "_synthetic",
        "timestamp": pd.to_datetime(timestamps),
        "close": close,
    })
    expected = make_no_trade_band_labels(
        frame, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )
    expected_labels_float = expected["label"].to_numpy()
    expected_valid_mask = ~np.isnan(expected_labels_float)

    # Wrapper path.
    labels, valid_mask = build_no_trade_band_labels(
        close, timestamps, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )

    np.testing.assert_array_equal(valid_mask, expected_valid_mask)
    # Compare valid-position labels (sentinel -1 vs NaN differ by design).
    np.testing.assert_array_equal(
        labels[valid_mask], expected_labels_float[expected_valid_mask].astype(np.int8),
    )


def test_up_only_drift_produces_label_1_then_invalid_tail():
    """Drift > +1.5bps/bar -> every valid sample labels 1, last 3 invalid."""
    close, timestamps = _synthetic_intraday_session(
        n=20, drift_bps_per_bar=10.0,  # 10 bps drift dominates the noise
    )
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # First 17 samples are valid (h=3 lookahead exists); last 3 invalid.
    assert valid_mask[:17].all()
    assert not valid_mask[17:].any()
    assert (labels[:17] == 1).all()
    assert (labels[17:] == -1).all()


def test_down_only_drift_produces_label_0():
    close, timestamps = _synthetic_intraday_session(
        n=20, drift_bps_per_bar=-10.0,
    )
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert (labels[valid_mask] == 0).all()


def test_within_band_returns_yield_no_trade_band_invalid():
    """Constant close -> future_return == 0 -> within [-1.5bps, +1.5bps] band."""
    n = 20
    timestamps = pd.date_range("2025-01-02 09:30", periods=n, freq="5min").to_numpy()
    close = np.full(n, 100.0, dtype=np.float64)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Every position in [0, 17) should be no-trade-band invalid.
    assert not valid_mask[:17].any()
    assert (labels[:17] == -1).all()


def test_cross_day_horizon_invalidates_late_bars_of_day_one():
    """Last 3 bars of trading day 1 have horizon falling into day 2 -> invalid."""
    day1 = pd.date_range("2025-01-02 09:30", periods=8, freq="5min")
    day2 = pd.date_range("2025-01-03 09:30", periods=8, freq="5min")
    timestamps = np.concatenate([day1.to_numpy(), day2.to_numpy()])
    close = np.linspace(100.0, 110.0, 16).astype(np.float64)  # strong up drift
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # day1 indices 5, 6, 7 (last 3) -> horizon falls into day2 -> invalid.
    assert not valid_mask[5:8].any()
    assert (labels[5:8] == -1).all()
    # day1 indices 0..4 should be valid with up-drift labels.
    assert valid_mask[:5].all()
    assert (labels[:5] == 1).all()


def test_threshold_bps_zero_degenerate_strict_sign_labels():
    """threshold_bps=0.0 mirrors baseline_v1: any sign-deterministic return labels."""
    close, timestamps = _synthetic_intraday_session(
        n=15, drift_bps_per_bar=0.5,  # below 1.5 bps but above 0
    )
    labels, valid_mask = build_no_trade_band_labels(
        close, timestamps, horizon_k=3, threshold_bps=0.0,
    )
    # Every valid sample should be label 1 (up-drift) because threshold=0 leaves
    # no no-trade-band; only strictly-zero returns would be invalid.
    assert (labels[valid_mask] == 1).all()
    assert valid_mask[:12].all()  # first 12 have h=3 lookahead


def test_close_nan_propagates_to_invalid_mask():
    close, timestamps = _synthetic_intraday_session(n=20)
    close[5] = np.nan
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Index 5 (NaN close itself) AND indices 2, 3, 4 (whose t+3 = 5, 6, 7)
    # may all be invalid due to NaN propagation through future_cumulative_return.
    assert not valid_mask[5]
    # The wrapper does not raise; sentinel -1 at invalid positions.
    assert labels[5] == -1


# ---------------------------------------------------------------------------
# Task 3: output format
# ---------------------------------------------------------------------------


def test_output_dtypes_and_shapes():
    close, timestamps = _synthetic_intraday_session(n=30, drift_bps_per_bar=3.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert labels.dtype == np.int8
    assert valid_mask.dtype == np.bool_
    assert labels.shape == close.shape
    assert valid_mask.shape == close.shape


def test_sentinel_minus_one_only_at_invalid_positions():
    close, timestamps = _synthetic_intraday_session(n=30, drift_bps_per_bar=3.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Invalid positions carry sentinel -1.
    assert (labels[~valid_mask] == -1).all()
    # Valid positions carry only 0 or 1.
    assert set(labels[valid_mask].tolist()).issubset({0, 1})


def test_empty_input_returns_empty_outputs():
    close = np.array([], dtype=np.float64)
    timestamps = np.array([], dtype="datetime64[ns]")
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert labels.shape == (0,)
    assert valid_mask.shape == (0,)
    assert labels.dtype == np.int8
    assert valid_mask.dtype == np.bool_


def test_n_less_than_horizon_plus_one_all_invalid():
    """h=3 needs at least 4 samples for any to be valid; with 3 every row invalid."""
    close, timestamps = _synthetic_intraday_session(n=3, drift_bps_per_bar=5.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert not valid_mask.any()
    assert (labels == -1).all()


# ---------------------------------------------------------------------------
# Task 4: input-validation guards
# ---------------------------------------------------------------------------


def test_rejects_non_1d_close():
    close = np.zeros((4, 3), dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    with pytest.raises(ValueError, match="1-D arrays"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_non_1d_timestamps():
    close = np.zeros(12, dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    timestamps_2d = timestamps.reshape(4, 3)
    with pytest.raises(ValueError, match="1-D arrays"):
        build_h03_bps1p5_labels(close, timestamps_2d)


def test_rejects_misaligned_lengths():
    close = np.zeros(10, dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    with pytest.raises(ValueError, match="same length"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_unsorted_timestamps():
    close, timestamps = _synthetic_intraday_session(n=10)
    # Swap two adjacent timestamps to break monotonicity.
    timestamps = timestamps.copy()
    timestamps[3], timestamps[5] = timestamps[5], timestamps[3]
    with pytest.raises(ValueError, match="sorted ascending"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_nat_in_timestamps():
    close, timestamps = _synthetic_intraday_session(n=10)
    timestamps = timestamps.copy()
    timestamps[4] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="NaT"):
        build_h03_bps1p5_labels(close, timestamps)


@pytest.mark.parametrize("bad", [0, -1, -5, 1.5, True, False])
def test_rejects_invalid_horizon_k(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="horizon_k must be a positive int"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=bad, threshold_bps=1.5,
        )


@pytest.mark.parametrize("bad", [-0.5, -1.0, math.inf, -math.inf, math.nan])
def test_rejects_invalid_threshold_bps(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="threshold_bps must be non-negative and finite"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=3, threshold_bps=bad,
        )


@pytest.mark.parametrize("bad", ["abc", None, [1.0]])
def test_rejects_non_numeric_threshold_bps(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="threshold_bps must be numeric"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=3, threshold_bps=bad,
        )


# ---------------------------------------------------------------------------
# Task 5: frozen alias equivalence
# ---------------------------------------------------------------------------


def test_h03_bps1p5_alias_matches_generic_call():
    close, timestamps = _synthetic_intraday_session(n=40, drift_bps_per_bar=2.0)
    via_alias = build_h03_bps1p5_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H03_BPS1P5,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_h09_bps3p0_alias_matches_generic_call():
    # h=9 needs >= 9 same-day bars; use one trading day of 78 bars.
    close, timestamps = _synthetic_intraday_session(n=78, drift_bps_per_bar=4.0)
    via_alias = build_h09_bps3p0_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H09_BPS3P0,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_h24_bps7p5_alias_matches_generic_call():
    close, timestamps = _synthetic_intraday_session(n=78, drift_bps_per_bar=10.0)
    via_alias = build_h24_bps7p5_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H24_BPS7P5,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_frozen_config_constants_match_screening_freeze():
    """Lock the frozen config values from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md."""
    assert H03_BPS1P5 == {"horizon_k": 3, "threshold_bps": 1.5}
    assert H09_BPS3P0 == {"horizon_k": 9, "threshold_bps": 3.0}
    assert H24_BPS7P5 == {"horizon_k": 24, "threshold_bps": 7.5}
