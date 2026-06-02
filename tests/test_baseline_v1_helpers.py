import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import (
    add_baseline_v1_features,
    add_split_and_invalidate_boundaries,
    assign_calendar_split,
    build_windows_by_ticker_and_split,
    build_windows_for_segment,
    evaluate_stratified_dummy,
    fit_train_only_scaler,
    make_no_trade_band_labels,
    transform_train_and_validation,
)


def make_one_ticker_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * 8,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:30",
                    "2020-01-01 09:35",
                    "2020-01-01 09:40",
                    "2020-01-01 09:45",
                    "2020-01-02 09:30",
                    "2020-01-02 09:35",
                    "2020-01-02 09:40",
                    "2020-01-02 09:45",
                ]
            ),
            "open": [100, 101, 102, 103, 110, 111, 112, 113],
            "high": [101, 102, 103, 104, 111, 112, 113, 114],
            "low": [99, 100, 101, 102, 109, 110, 111, 112],
            "close": [100, 101, 103, 102, 110, 111, 112, 112],
            "volume": [1000, 1100, 1200, 1300, 1000, 1100, 1200, 1300],
        }
    )


def test_no_trade_label_uses_future_cumulative_return():
    frame = make_one_ticker_frame()
    labeled = make_no_trade_band_labels(frame, horizon_k=2, threshold_bps=0.0)

    expected = frame["close"].shift(-2) / frame["close"] - 1.0

    assert labeled.loc[0, "future_cumulative_return"] == expected.loc[0]
    assert labeled.loc[1, "future_cumulative_return"] == expected.loc[1]
    assert pd.isna(labeled.loc[6, "label"])
    assert pd.isna(labeled.loc[7, "label"])


def test_no_trade_band_marks_near_zero_returns_invalid_not_classed():
    frame = make_one_ticker_frame()
    frame["close"] = [
        100,
        100.001,
        100.002,
        100.003,
        100.004,
        100.005,
        100.006,
        100.007,
    ]

    labeled = make_no_trade_band_labels(frame, horizon_k=2, threshold_bps=5.0)

    assert labeled["label"].isna().any()
    assert set(labeled["label"].dropna().unique()).issubset({0, 1})


def test_no_trade_label_audits_irregular_same_day_positional_horizon():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:30",
                    "2020-01-01 09:35",
                    "2020-01-01 09:45",
                    "2020-01-01 09:50",
                ]
            ),
            "open": [100.0, 100.5, 101.0, 101.5],
            "high": [101.0, 101.5, 102.0, 102.5],
            "low": [99.0, 99.5, 100.0, 100.5],
            "close": [100.0, 100.5, 101.0, 101.5],
            "volume": [1000, 1100, 1200, 1300],
        }
    )

    labeled = make_no_trade_band_labels(frame, horizon_k=2, threshold_bps=0.0)

    assert bool(labeled.loc[0, "diagnostic_irregular_horizon"])
    assert labeled.loc[0, "future_horizon_minutes"] == pytest.approx(15.0)
    assert not bool(labeled.loc[0, "invalid_cross_day"])


def test_no_trade_label_rejects_pooled_multi_ticker_frame():
    first = make_one_ticker_frame().head(3)
    second = first.copy()
    second["ticker"] = "BBB"
    pooled = pd.concat([first, second], ignore_index=True).sort_values("timestamp")

    with pytest.raises(ValueError, match="single ticker"):
        make_no_trade_band_labels(pooled, horizon_k=2, threshold_bps=0.0)


def test_calendar_split_boundaries_are_half_open():
    splits = {
        "train": ("2020-01-01", "2020-01-02"),
        "validation": ("2020-01-02", "2020-01-03"),
        "closed_holdout_boundary_only": ("2020-01-03", "2020-01-04"),
    }

    assert assign_calendar_split(pd.Timestamp("2020-01-01 00:00"), splits) == "train"
    assert assign_calendar_split(pd.Timestamp("2020-01-02 00:00"), splits) == "validation"
    assert (
        assign_calendar_split(pd.Timestamp("2020-01-03 00:00"), splits)
        == "closed_holdout_boundary_only"
    )
    assert (
        assign_calendar_split(pd.Timestamp("2020-01-04 00:00"), splits)
        == "outside_defined_calendar"
    )


def test_split_boundary_invalidates_labels_before_next_split():
    frame = make_one_ticker_frame()
    frame["future_cumulative_return"] = 0.01
    frame["label"] = 1.0
    splits = {
        "train": ("2020-01-01 09:30", "2020-01-01 09:45"),
        "validation": ("2020-01-01 09:45", "2020-01-02 09:45"),
        "closed_holdout_boundary_only": ("2020-01-02 09:45", "2020-01-03"),
    }

    checked = add_split_and_invalidate_boundaries(frame, splits=splits, horizon_k=2)

    assert pd.isna(checked.loc[1, "label"])
    assert checked.loc[1, "invalid_cross_split"]
    assert checked.loc[3, "split"] == "validation"


def test_split_boundary_rejects_pooled_multi_ticker_frame():
    first = make_one_ticker_frame().head(3)
    second = first.copy()
    second["ticker"] = "BBB"
    pooled = pd.concat([first, second], ignore_index=True).sort_values("timestamp")
    pooled["future_cumulative_return"] = 0.01
    pooled["label"] = 1.0
    splits = {
        "train": ("2020-01-01 09:30", "2020-01-01 09:45"),
        "validation": ("2020-01-01 09:45", "2020-01-02"),
        "closed_holdout_boundary_only": ("2020-01-02", "2020-01-03"),
    }

    with pytest.raises(ValueError, match="single ticker"):
        add_split_and_invalidate_boundaries(pooled, splits=splits, horizon_k=2)


def test_split_boundary_requires_label_return_column():
    frame = make_one_ticker_frame()
    frame["label"] = 1.0
    splits = {
        "train": ("2020-01-01", "2020-01-02"),
        "validation": ("2020-01-02", "2020-01-03"),
        "closed_holdout_boundary_only": ("2020-01-03", "2020-01-04"),
    }

    with pytest.raises(ValueError, match="future_cumulative_return"):
        add_split_and_invalidate_boundaries(frame, splits=splits, horizon_k=2)


def test_feature_construction_uses_prior_only_volume_within_each_day():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 42,
            "timestamp": pd.to_datetime(
                [f"2020-01-01 09:{30 + i:02d}" for i in range(20)]
                + [f"2020-01-02 09:{30 + i:02d}" for i in range(22)]
            ),
            "open": [100.0] * 42,
            "high": [101.0] * 42,
            "low": [99.0] * 42,
            "close": [100.0 + i * 0.01 for i in range(42)],
            "volume": list(range(100, 120)) + list(range(200, 222)),
        }
    )

    featured = add_baseline_v1_features(frame)

    assert pd.isna(featured.loc[39, "normalized_volume_20"])
    prior_day_two = pd.Series(range(200, 220)).map(np.log1p)
    expected = np.log1p(220) - prior_day_two.mean()
    assert featured.loc[40, "normalized_volume_20"] == pytest.approx(expected)


def test_prior_only_feature_warmup_cost_is_explicit():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 25,
            "timestamp": pd.date_range("2020-01-01 09:30", periods=25, freq="5min"),
            "open": np.linspace(100.0, 102.4, 25),
            "high": np.linspace(100.5, 102.9, 25),
            "low": np.linspace(99.5, 101.9, 25),
            "close": np.linspace(100.0, 102.4, 25),
            "volume": np.arange(100, 125),
        }
    )

    featured = add_baseline_v1_features(frame)

    assert pd.isna(featured.loc[19, "normalized_volume_20"])
    assert pd.notna(featured.loc[20, "normalized_volume_20"])
    assert pd.isna(featured.loc[20, "rolling_volatility_20"])
    assert pd.notna(featured.loc[21, "rolling_volatility_20"])


def test_feature_construction_time_encoding_uses_regular_session_phase():
    frame = make_one_ticker_frame()
    featured = add_baseline_v1_features(frame)

    assert featured.loc[0, "time_of_day_sin"] == pytest.approx(0.0)
    assert featured.loc[0, "time_of_day_cos"] == pytest.approx(1.0)


def test_feature_construction_time_encoding_does_not_wrap_close_to_open():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "timestamp": pd.to_datetime(["2020-01-01 09:30", "2020-01-01 16:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    )

    featured = add_baseline_v1_features(frame)

    open_point = featured.loc[0, ["time_of_day_sin", "time_of_day_cos"]].to_numpy(
        dtype=float
    )
    close_point = featured.loc[1, ["time_of_day_sin", "time_of_day_cos"]].to_numpy(
        dtype=float
    )
    assert not np.allclose(open_point, close_point)


def test_feature_construction_rsi_uses_wilder_alpha():
    close_values = [
        100.0,
        101.0,
        100.5,
        102.0,
        101.2,
        102.5,
        103.0,
        102.6,
        103.4,
        104.0,
        103.6,
        104.2,
        104.8,
        104.1,
        105.0,
        105.4,
        104.9,
        105.6,
        106.0,
        105.5,
        106.4,
        106.8,
        106.2,
        107.0,
        107.5,
        107.1,
        107.9,
        108.2,
        107.7,
        108.5,
    ]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 30,
            "timestamp": pd.date_range("2020-01-01 09:30", periods=30, freq="5min"),
            "open": close_values,
            "high": [value + 0.5 for value in close_values],
            "low": [value - 0.5 for value in close_values],
            "close": close_values,
            "volume": np.arange(100, 130),
        }
    )

    featured = add_baseline_v1_features(frame)
    close_delta = frame["close"].diff()
    gain = close_delta.clip(lower=0.0)
    loss = (-close_delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    expected = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan)))
    expected = expected.mask(avg_loss.eq(0.0) & avg_gain.gt(0.0), 100.0)
    expected = expected.mask(avg_loss.eq(0.0) & avg_gain.eq(0.0), 50.0)

    valid = expected.notna()
    assert featured.loc[valid, "rsi_14"].tolist() == pytest.approx(
        expected.loc[valid].tolist()
    )


def test_feature_construction_macd_ema_is_continuous_per_ticker_across_days():
    day_one = pd.date_range("2020-01-01 09:30", periods=45, freq="5min")
    day_two = pd.date_range("2020-01-02 09:30", periods=5, freq="5min")
    close = np.linspace(100.0, 105.0, len(day_one) + len(day_two))
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(close),
            "timestamp": list(day_one) + list(day_two),
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": np.arange(1000, 1000 + len(close)),
        }
    )

    featured = add_baseline_v1_features(frame)

    assert pd.notna(featured.loc[45, "normalized_macd_hist"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("high", 98.0, "high must be >= low"),
        ("open", 98.0, "open must be within high-low range"),
        ("close", 102.0, "close must be within high-low range"),
        ("low", 0.0, "price columns must be positive"),
        ("volume", -1.0, "volume must be non-negative"),
    ],
)
def test_feature_construction_rejects_invalid_raw_ohlcv_contract(
    column, value, message
):
    frame = make_one_ticker_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        add_baseline_v1_features(frame)


def test_feature_construction_marks_zero_bollinger_width_invalid():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 21,
            "timestamp": pd.date_range("2020-01-01 09:30", periods=21, freq="5min"),
            "open": [100.0] * 21,
            "high": [100.0] * 21,
            "low": [100.0] * 21,
            "close": [100.0] * 21,
            "volume": [1000] * 21,
        }
    )

    featured = add_baseline_v1_features(frame)

    assert pd.isna(featured.loc[19, "bollinger_pctb"])


def test_feature_construction_rejects_null_or_pooled_tickers():
    frame = make_one_ticker_frame()
    frame.loc[0, "ticker"] = pd.NA

    with pytest.raises(ValueError, match="single non-null ticker"):
        add_baseline_v1_features(frame)

    no_ticker = frame.drop(columns=["ticker"])
    with pytest.raises(ValueError, match="ticker column"):
        add_baseline_v1_features(no_ticker)


@pytest.mark.parametrize("horizon_k", [0, -1, 1.5, True])
def test_label_rejects_invalid_horizon_values(horizon_k):
    frame = make_one_ticker_frame()

    with pytest.raises(ValueError, match="horizon_k"):
        make_no_trade_band_labels(frame, horizon_k=horizon_k, threshold_bps=0.0)


def test_scaler_fit_ignores_validation_and_closed_holdout_values():
    feature_columns = ["f1", "f2"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "split": [
                "train",
                "train",
                "train",
                "validation",
                "validation",
                "closed_holdout_boundary_only",
            ],
            "f1": [1.0, 2.0, 3.0, 1000.0, 2000.0, 3000.0],
            "f2": [10.0, 20.0, 30.0, 10000.0, 20000.0, 30000.0],
        }
    )

    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    assert list(scaler.mean_) == [2.0, 20.0]


def test_scaler_fit_ignores_non_finite_train_rows():
    feature_columns = ["f1"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "split": ["train", "train", "train", "validation"],
            "f1": [1.0, 3.0, float("inf"), 1000.0],
        }
    )

    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    assert list(scaler.mean_) == [2.0]


def test_scaler_fit_reports_non_numeric_feature_columns():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 2,
            "split": ["train", "train"],
            "f1": ["bad", "data"],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        fit_train_only_scaler({"AAA": frame}, feature_columns=["f1"])


def test_transform_train_and_validation_does_not_transform_closed_holdout():
    feature_columns = ["f1"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "split": ["train", "train", "validation", "closed_holdout_boundary_only"],
            "f1": [1.0, 2.0, 3.0, 999.0],
        }
    )
    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    transformed = transform_train_and_validation(
        {"AAA": frame}, scaler, feature_columns=feature_columns
    )["AAA"]

    assert "f1_scaled" in transformed.columns
    assert pd.notna(transformed.loc[0, "f1_scaled"])
    assert pd.notna(transformed.loc[2, "f1_scaled"])
    assert pd.isna(transformed.loc[3, "f1_scaled"])


def test_transform_skips_non_finite_feature_rows():
    feature_columns = ["f1"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "split": ["train", "train", "validation", "validation"],
            "f1": [1.0, 2.0, float("inf"), 3.0],
        }
    )
    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    transformed = transform_train_and_validation(
        {"AAA": frame}, scaler, feature_columns=feature_columns
    )["AAA"]

    assert pd.isna(transformed.loc[2, "f1_scaled"])
    assert pd.notna(transformed.loc[3, "f1_scaled"])


def test_windows_do_not_cross_day_or_split_and_skip_invalid_labels():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:30",
                    "2020-01-01 09:35",
                    "2020-01-01 09:40",
                    "2020-01-02 09:30",
                    "2020-01-02 09:35",
                    "2020-01-02 09:40",
                ]
            ),
            "split": ["train", "train", "train", "train", "train", "train"],
            "f1_scaled": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 1, 0, 1, float("nan")],
        }
    )

    result = build_windows_for_segment(
        frame, "train", feature_columns=["f1"], window_size=2
    )

    assert result["X"].shape[1:] == (2, 1)
    assert len(result["y"]) == 3
    assert set(result["metadata"]["target_timestamp"].dt.date) == {
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-02").date(),
    }


def test_model_windows_do_not_fall_back_to_raw_features():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": pd.to_datetime(
                ["2020-01-01 09:30", "2020-01-01 09:35", "2020-01-01 09:40"]
            ),
            "split": ["train", "train", "train"],
            "f1": [1, 2, 3],
            "label": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="scaled"):
        build_windows_for_segment(frame, "train", feature_columns=["f1"], window_size=2)


def test_model_windows_reject_pooled_multi_ticker_frame():
    first = make_one_ticker_frame().head(3)
    first["split"] = "train"
    first["f1_scaled"] = [1, 2, 3]
    first["label"] = [0, 1, 0]
    second = first.copy()
    second["ticker"] = "BBB"
    second["f1_scaled"] = [10, 20, 30]
    pooled = pd.concat([first, second], ignore_index=True).sort_values("timestamp")

    with pytest.raises(ValueError, match="single ticker"):
        build_windows_for_segment(pooled, "train", feature_columns=["f1"], window_size=2)


def test_windows_are_built_per_ticker_not_pooled_across_tickers():
    base = make_one_ticker_frame().head(4)
    base["split"] = "train"
    base["f1_scaled"] = [1, 2, 3, 4]
    base["label"] = [0, 1, 0, 1]
    other = base.copy()
    other["ticker"] = "BBB"
    other["f1_scaled"] = [10, 20, 30, 40]

    windows = build_windows_by_ticker_and_split(
        {"AAA": base, "BBB": other},
        feature_columns=["f1"],
        window_size=2,
    )

    assert set(windows) == {"AAA", "BBB"}
    assert windows["AAA"]["train"]["X"].shape[0] == windows["BBB"]["train"]["X"].shape[0]
    assert windows["AAA"]["train"]["X"][0, :, 0].tolist() == [1, 2]
    assert windows["BBB"]["train"]["X"][0, :, 0].tolist() == [10, 20]


def test_stratified_dummy_uses_train_distribution_and_validation_labels():
    y_train = [0, 0, 0, 1]
    y_validation = [0, 1, 1, 1]

    result = evaluate_stratified_dummy(y_train, y_validation, seeds=(41, 42))

    assert set(result.columns) == {
        "seed",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "validation_n",
    }
    assert result["validation_n"].tolist() == [4, 4]
    assert result[["macro_f1", "balanced_accuracy", "accuracy"]].notna().all().all()


def test_stratified_dummy_does_not_use_validation_distribution():
    y_train = [0, 0, 0, 0]
    y_validation = [1, 1, 1, 1]

    result = evaluate_stratified_dummy(y_train, y_validation, seeds=(41, 42))

    assert result["accuracy"].tolist() == [0.0, 0.0]
    assert result["macro_f1"].tolist() == [0.0, 0.0]
    assert result["balanced_accuracy"].tolist() == [0.0, 0.0]


def test_stratified_dummy_scores_absent_binary_class_explicitly():
    y_train = [1, 1, 1, 1]
    y_validation = [1, 1, 1, 1]

    result = evaluate_stratified_dummy(y_train, y_validation, seeds=(41,))

    assert result.loc[0, "accuracy"] == 1.0
    assert result.loc[0, "balanced_accuracy"] == 1.0
    assert result.loc[0, "macro_f1"] == 0.5


def test_active_helper_imports_do_not_reference_archived_paths():
    tree = ast.parse(Path("intraday_research/baseline_v1.py").read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    forbidden = {
        "archive",
        "legacy_model_runner",
        "ml_utils",
        "runner_utils",
        "train_test_split",
    }
    assert not (imported_names & forbidden)
