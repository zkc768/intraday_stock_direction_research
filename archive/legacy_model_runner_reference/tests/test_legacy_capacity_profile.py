import numpy as np
import pandas as pd
import pytest


BAR_SUMMARY_COLUMNS = [
    "ticker",
    "n_days",
    "min_bars_per_day",
    "p5_bars_per_day",
    "p10_bars_per_day",
    "median_bars_per_day",
    "max_bars_per_day",
]

CAPACITY_COLUMNS = [
    "ticker",
    "k",
    "threshold_bps",
    "n_total",
    "n_tail",
    "n_cross_day",
    "n_neutral",
    "n_up",
    "n_down",
    "n_retained",
    "retained_pct",
    "neutral_dropped_pct",
    "up_pct_retained",
    "down_pct_retained",
    "minority_pct_retained",
    "eff_target_independent",
]

FEASIBILITY_COLUMNS = [
    "window_size",
    "required_bars",
    "p5_bars_per_day",
    "feasible_by_p5",
    "feasibility_status",
]

SCALER_DIAGNOSTIC_COLUMNS = [
    "feature_group",
    "full_train_mean",
    "retained_train_mean",
    "mean_shift",
    "full_train_std",
    "retained_train_std",
    "std_ratio",
    "warning",
]


def _compute_bar_count_summary(*args, **kwargs):
    from runner_utils.profiling import compute_bar_count_summary

    return compute_bar_count_summary(*args, **kwargs)


def _profile_label_capacity(*args, **kwargs):
    from runner_utils.profiling import profile_label_capacity

    return profile_label_capacity(*args, **kwargs)


def _add_intraday_feasibility_flags(*args, **kwargs):
    from runner_utils.profiling import add_intraday_feasibility_flags

    return add_intraday_feasibility_flags(*args, **kwargs)


def _compute_scaler_diagnostic(*args, **kwargs):
    from runner_utils.profiling import compute_scaler_diagnostic

    return compute_scaler_diagnostic(*args, **kwargs)


def _frame_from_day_counts(day_counts):
    timestamps = []
    for day_offset, count in enumerate(day_counts):
        start = pd.Timestamp("2024-01-02 09:30") + pd.Timedelta(days=day_offset)
        timestamps.extend(pd.date_range(start, periods=count, freq="5min"))
    return pd.DataFrame({"timestamp": timestamps, "close": np.arange(len(timestamps), dtype=float) + 100.0})


def _price_frame(closes, timestamps=None):
    if timestamps is None:
        timestamps = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="5min")
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "close": [float(close) for close in closes]})


def _capacity_df():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "k": [12, 24],
            "threshold_bps": [10.0, 10.0],
            "n_total": [100, 120],
            "n_tail": [12, 24],
            "n_cross_day": [0, 0],
            "n_neutral": [20, 30],
            "n_up": [40, 30],
            "n_down": [28, 36],
            "n_retained": [68, 66],
            "retained_pct": [0.68, 0.55],
            "neutral_dropped_pct": [0.20, 0.25],
            "up_pct_retained": [40 / 68, 30 / 66],
            "down_pct_retained": [28 / 68, 36 / 66],
            "minority_pct_retained": [28 / 68, 30 / 66],
            "eff_target_independent": [68 / 12, 66 / 24],
        }
    )


def test_bar_count_summary_computes_per_ticker_day_distribution():
    frames = {
        "AAA": _frame_from_day_counts([3, 5, 4]),
        "BBB": _frame_from_day_counts([2, 6]),
    }

    summary = _compute_bar_count_summary(frames, timestamp_col="timestamp")

    assert list(summary.columns) == BAR_SUMMARY_COLUMNS
    assert summary["ticker"].tolist() == ["AAA", "BBB"]
    aaa = summary.set_index("ticker").loc["AAA"]
    assert aaa["n_days"] == 3
    assert aaa["min_bars_per_day"] == 3
    assert np.isclose(aaa["p5_bars_per_day"], np.percentile([3, 5, 4], 5))
    assert np.isclose(aaa["p10_bars_per_day"], np.percentile([3, 5, 4], 10))
    assert np.isclose(aaa["median_bars_per_day"], 4.0)
    assert aaa["max_bars_per_day"] == 5


def test_bar_count_summary_does_not_hardcode_78():
    frames = {"TINY": _frame_from_day_counts([4, 6])}

    summary = _compute_bar_count_summary(frames, timestamp_col="timestamp")

    row = summary.iloc[0]
    assert row["n_days"] == 2
    assert row["max_bars_per_day"] == 6
    assert row["p5_bars_per_day"] < 78
    assert "short_day_count" not in summary.columns


def test_profile_label_capacity_reports_retained_neutral_and_class_balance():
    frames = {"AAA": _price_frame([100.0, 100.2, 100.25, 100.0])}

    profile = _profile_label_capacity(
        frames,
        price_col="close",
        timestamp_col="timestamp",
        k_values=[1],
        threshold_bps_values=[10.0],
    )

    assert list(profile.columns) == CAPACITY_COLUMNS
    row = profile.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["n_total"] == 4
    assert row["n_tail"] == 1
    assert row["n_cross_day"] == 0
    assert row["n_neutral"] == 1
    assert row["n_up"] == 1
    assert row["n_down"] == 1
    assert row["n_retained"] == 2
    assert np.isclose(row["retained_pct"], 0.5)
    assert np.isclose(row["neutral_dropped_pct"], 0.25)
    assert np.isclose(row["up_pct_retained"], 0.5)
    assert np.isclose(row["down_pct_retained"], 0.5)
    assert np.isclose(row["minority_pct_retained"], 0.5)
    assert np.isclose(row["eff_target_independent"], 2.0)


def test_profile_label_capacity_counts_tail_cross_day_neutral_up_down():
    frames = {
        "AAA": _price_frame(
            [100.0, 101.0, 101.2, 101.25, 101.0, 101.05],
            timestamps=[
                "2024-01-02 15:55",
                "2024-01-03 09:30",
                "2024-01-03 09:35",
                "2024-01-03 09:40",
                "2024-01-03 09:45",
                "2024-01-03 09:50",
            ],
        )
    }

    profile = _profile_label_capacity(
        frames,
        price_col="close",
        timestamp_col="timestamp",
        k_values=[1],
        threshold_bps_values=[10.0],
    )

    row = profile.iloc[0]
    assert row["n_total"] == 6
    assert row["n_tail"] == 1
    assert row["n_cross_day"] == 1
    assert row["n_neutral"] == 2
    assert row["n_up"] == 1
    assert row["n_down"] == 1
    assert row["n_retained"] == 2


def test_profile_label_capacity_handles_zero_retained_without_division_error():
    frames = {"AAA": _price_frame([100.0, 100.0, 100.0])}

    profile = _profile_label_capacity(
        frames,
        price_col="close",
        timestamp_col="timestamp",
        k_values=[1],
        threshold_bps_values=[0.0],
    )

    row = profile.iloc[0]
    assert row["n_retained"] == 0
    assert row["retained_pct"] == 0.0
    assert np.isclose(row["neutral_dropped_pct"], 2 / 3)
    assert np.isnan(row["up_pct_retained"])
    assert np.isnan(row["down_pct_retained"])
    assert np.isnan(row["minority_pct_retained"])
    assert row["eff_target_independent"] == 0.0


def test_profile_label_capacity_preserves_input_frames():
    frames = {
        "AAA": _price_frame([100.0, 100.2, 100.25, 100.0, 100.3]),
        "BBB": _price_frame([50.0, 49.8, 49.75, 50.1, 50.2]),
    }
    originals = {ticker: frame.copy(deep=True) for ticker, frame in frames.items()}

    _profile_label_capacity(
        frames,
        price_col="close",
        timestamp_col="timestamp",
        k_values=[1, 2],
        threshold_bps_values=[0.0, 10.0],
    )

    for ticker, original in originals.items():
        pd.testing.assert_frame_equal(frames[ticker], original)
        assert "future_avg_r" not in frames[ticker].columns
        assert "label" not in frames[ticker].columns


def test_feasibility_flags_use_p5_not_theoretical_78():
    capacity_df = pd.DataFrame({"ticker": ["AAA"], "k": [24], "threshold_bps": [10.0]})
    bar_summary_df = pd.DataFrame({"ticker": ["AAA"], "p5_bars_per_day": [72.0]})

    flagged = _add_intraday_feasibility_flags(
        capacity_df,
        bar_summary_df,
        window_sizes=[48, 54],
    )

    by_window = flagged.set_index("window_size")
    assert by_window.loc[48, "required_bars"] == 72
    assert bool(by_window.loc[48, "feasible_by_p5"]) is True
    assert by_window.loc[48, "feasibility_status"] == "PASS"
    assert by_window.loc[54, "required_bars"] == 78
    assert bool(by_window.loc[54, "feasible_by_p5"]) is False
    assert by_window.loc[54, "feasibility_status"] == "INFEASIBLE_INTRADAY_CAPACITY"


def test_feasibility_flags_expand_each_capacity_row_by_window_size():
    capacity_df = _capacity_df()
    bar_summary_df = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "p5_bars_per_day": [72.0, 50.0],
        }
    )

    flagged = _add_intraday_feasibility_flags(
        capacity_df,
        bar_summary_df,
        window_sizes=[12, 24, 60],
    )

    assert len(flagged) == len(capacity_df) * 3
    assert set(FEASIBILITY_COLUMNS).issubset(flagged.columns)
    assert flagged.groupby(["ticker", "k", "threshold_bps"]).size().tolist() == [3, 3]
    aaa_60 = flagged[(flagged["ticker"] == "AAA") & (flagged["window_size"] == 60)].iloc[0]
    assert aaa_60["required_bars"] == 72
    assert aaa_60["p5_bars_per_day"] == 72.0


def test_capacity_profile_collects_multiple_k_and_threshold_combinations():
    frames = {
        "AAA": _price_frame([100.0, 100.2, 100.25, 100.0, 100.3]),
        "BBB": _price_frame([50.0, 50.1, 49.9, 50.0, 50.2]),
    }

    profile = _profile_label_capacity(
        frames,
        price_col="close",
        timestamp_col="timestamp",
        k_values=[1, 2],
        threshold_bps_values=[0.0, 10.0],
    )

    expected = pd.MultiIndex.from_product(
        [["AAA", "BBB"], [1, 2], [0.0, 10.0]],
        names=["ticker", "k", "threshold_bps"],
    )
    actual = pd.MultiIndex.from_frame(profile[["ticker", "k", "threshold_bps"]])
    assert len(profile) == len(expected)
    assert set(actual) == set(expected)


def test_scaler_diagnostic_reports_group_mean_shift_std_ratio_and_warning():
    full_train_df = pd.DataFrame({"a": [0.0, 2.0], "b": [4.0, 6.0]})
    retained_train_df = pd.DataFrame({"a": [0.0, 4.0], "b": [8.0, 12.0]})

    diagnostic = _compute_scaler_diagnostic(
        full_train_df,
        retained_train_df,
        feature_groups={"wide": ["a", "b"]},
    )

    assert list(diagnostic.columns) == SCALER_DIAGNOSTIC_COLUMNS
    row = diagnostic.iloc[0]
    assert row["feature_group"] == "wide"
    assert np.isclose(row["full_train_mean"], 3.0)
    assert np.isclose(row["retained_train_mean"], 6.0)
    assert np.isclose(row["mean_shift"], 3.0)
    assert row["full_train_std"] > 0.0
    assert row["retained_train_std"] > 0.0
    assert np.isclose(row["std_ratio"], 2.0)
    assert bool(row["warning"]) is True


def test_scaler_diagnostic_no_warning_when_shift_and_std_ratio_are_within_bounds():
    full_train_df = pd.DataFrame({"a": [0.0, 2.0], "b": [4.0, 6.0]})
    retained_train_df = pd.DataFrame({"a": [0.2, 2.2], "b": [4.2, 6.2]})

    diagnostic = _compute_scaler_diagnostic(
        full_train_df,
        retained_train_df,
        feature_groups={"stable": ["a", "b"]},
    )

    row = diagnostic.iloc[0]
    assert np.isclose(row["mean_shift"], 0.2)
    assert np.isclose(row["std_ratio"], 1.0)
    assert bool(row["warning"]) is False


def test_scaler_diagnostic_raises_for_missing_feature_columns():
    full_train_df = pd.DataFrame({"a": [1.0, 2.0]})
    retained_train_df = pd.DataFrame({"a": [1.0, 2.0]})

    with pytest.raises(ValueError):
        _compute_scaler_diagnostic(
            full_train_df,
            retained_train_df,
            feature_groups={"missing": ["a", "b"]},
        )


def test_scaler_diagnostic_group_level_flattening_is_documented_by_behavior():
    full_train_df = pd.DataFrame({"up": [0.0, 0.0], "down": [10.0, 10.0]})
    retained_train_df = pd.DataFrame({"up": [2.0, 2.0], "down": [8.0, 8.0]})

    diagnostic = _compute_scaler_diagnostic(
        full_train_df,
        retained_train_df,
        feature_groups={"paired": ["up", "down"]},
    )

    assert diagnostic["feature_group"].tolist() == ["paired"]
    row = diagnostic.iloc[0]
    assert np.isclose(row["full_train_mean"], 5.0)
    assert np.isclose(row["retained_train_mean"], 5.0)
    assert np.isclose(row["mean_shift"], 0.0)
    assert "up" not in diagnostic["feature_group"].tolist()
    assert "down" not in diagnostic["feature_group"].tolist()
