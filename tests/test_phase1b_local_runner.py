import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.phase1b_local import local_baseline_matrix as runner


def _args(label_mode, threshold_bps=None):
    return Namespace(label_mode=label_mode, threshold_bps=threshold_bps)


def test_binary_alias_normalizes_to_legacy_binary():
    assert runner.resolve_label_mode(_args("binary")) == "legacy_binary"
    assert runner.resolve_label_mode(_args("legacy_binary")) == "legacy_binary"


def test_cli_default_label_mode_is_canonical_legacy_binary(monkeypatch):
    monkeypatch.setattr("sys.argv", ["local_baseline_matrix.py"])

    args = runner.parse_args()

    assert args.label_mode == "legacy_binary"
    assert args.split_mode == "ratio"


def test_calendar_split_parser_rejects_partial_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--split-mode",
            "calendar",
            "--train-start-ts",
            "2024-01-02 09:30",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_calendar_split_parser_rejects_unordered_windows(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--split-mode",
            "calendar",
            "--train-start-ts",
            "2024-01-02 09:30",
            "--train-end-ts",
            "2024-01-03 09:30",
            "--val-start-ts",
            "2024-01-03 09:25",
            "--val-end-ts",
            "2024-01-04 09:30",
            "--holdout-start-ts",
            "2024-01-04 09:30",
            "--holdout-end-ts",
            "2024-01-05 09:30",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_calendar_split_parser_rejects_max_rows_per_ticker(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--split-mode",
            "calendar",
            "--train-start-ts",
            "2024-01-02 09:30",
            "--train-end-ts",
            "2024-01-03 09:30",
            "--val-start-ts",
            "2024-01-03 09:30",
            "--val-end-ts",
            "2024-01-04 09:30",
            "--holdout-start-ts",
            "2024-01-04 09:30",
            "--holdout-end-ts",
            "2024-01-05 09:30",
            "--max-rows-per-ticker",
            "1000",
        ],
    )

    with pytest.raises(SystemExit):
        runner.parse_args()


def test_legacy_binary_threshold_is_forced_to_zero():
    assert runner.resolve_threshold_bps(_args("legacy_binary"), "legacy_binary") == 0.0


def test_legacy_binary_rejects_explicit_nonzero_threshold():
    with pytest.raises(ValueError, match="threshold-bps"):
        runner.resolve_threshold_bps(_args("legacy_binary", threshold_bps=5.0), "legacy_binary")


def test_no_trade_band_default_threshold_remains_five_bps():
    assert runner.resolve_threshold_bps(_args("no_trade_band"), "no_trade_band") == 5.0


def test_protocol_metadata_locks_mentor_clean_fixed_5bps_route():
    metadata = runner.protocol_metadata_fields(
        "mentor_clean_v1",
        "no_trade_band",
        threshold_bps=5.0,
        cli_threshold_bps=5.0,
    )

    assert metadata == {
        "decision_time_policy": "post_bar_close_completed_bar",
        "scaler_id": "standard_pooled_train_only_v1",
        "scaler_fit_scope": "pooled_train_after_per_ticker_chronological_split",
        "threshold_source": "fixed_pre_registered_5bps",
    }


def test_protocol_metadata_marks_default_no_trade_threshold_separately():
    metadata = runner.protocol_metadata_fields(
        "mentor_clean_v1",
        "no_trade_band",
        threshold_bps=5.0,
        cli_threshold_bps=None,
    )

    assert metadata["threshold_source"] == "default_fixed_5bps"


def test_audit_scope_fields_label_smoke_metrics_as_non_claim():
    fields = runner.audit_scope_fields("smoke")

    assert fields["claim_scope"] == "smoke_observation_not_performance_claim"
    assert fields["diagnostic_scope"] == "bounded_smoke_pipeline_diagnostic"
    assert fields["diagnostic_only"] is True
    assert fields["non_claim"] is True


def test_legacy_binary_metadata_fields_are_canonical():
    metadata = runner.label_metadata_fields("legacy_binary", threshold_bps=0.0)

    assert metadata["label_semantics"] == "canonical_phase1_full_binary"
    assert metadata["zero_return_policy"] == "class_0_non_up"
    assert metadata["no_trade_band_enabled"] is False
    assert metadata["neutral_policy"] == "not_applicable"


def test_compute_baselines_reports_balanced_accuracy_and_confusion_matrices():
    y_train = np.asarray([0, 0, 1, 1, 1, 1])
    y_eval = np.asarray([0, 1, 1, 0])

    metrics = runner.compute_baselines(y_train, y_eval)

    assert "dummy_stratified_balanced_accuracy_mean" in metrics
    assert "dummy_stratified_balanced_accuracy_std" in metrics
    assert "dummy_stratified_confusion_matrix_mean" in metrics
    assert "dummy_prior_balanced_accuracy" in metrics
    assert "dummy_prior_confusion_matrix" in metrics
    assert "always_up_balanced_accuracy" in metrics
    assert "always_up_confusion_matrix" in metrics
    assert "always_down_balanced_accuracy" in metrics
    assert "always_down_confusion_matrix" in metrics
    assert np.asarray(json.loads(metrics["dummy_stratified_confusion_matrix_mean"])).shape == (2, 2)
    assert json.loads(metrics["dummy_prior_confusion_matrix"]) == [[0, 2], [0, 2]]
    assert json.loads(metrics["always_up_confusion_matrix"]) == [[0, 2], [0, 2]]
    assert json.loads(metrics["always_down_confusion_matrix"]) == [[2, 0], [2, 0]]


def test_ticker_baseline_fields_preserve_balanced_accuracy_and_confusion_matrices():
    y_train = np.asarray([0, 0, 1, 1, 1, 1])
    y_eval = np.asarray([0, 1, 1, 0])
    metrics = runner.compute_baselines(y_train, y_eval)

    ticker_fields = runner.ticker_baseline_fields(metrics)

    assert ticker_fields["ticker_dummy_stratified_balanced_accuracy_mean"] == metrics[
        "dummy_stratified_balanced_accuracy_mean"
    ]
    assert ticker_fields["ticker_dummy_stratified_confusion_matrix_mean"] == metrics[
        "dummy_stratified_confusion_matrix_mean"
    ]
    assert ticker_fields["ticker_dummy_prior_confusion_matrix"] == metrics[
        "dummy_prior_confusion_matrix"
    ]
    assert ticker_fields["ticker_always_up_confusion_matrix"] == metrics[
        "always_up_confusion_matrix"
    ]
    assert ticker_fields["ticker_always_down_confusion_matrix"] == metrics[
        "always_down_confusion_matrix"
    ]


def test_legacy_binary_diagnostics_keep_exact_zero_in_class_zero():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=4, freq="5min"),
            "close": [100.0, 125.0, 93.75, 100.0],
        }
    )

    labeled_df, diagnostics = runner.make_legacy_binary_labels_with_diagnostics(
        df,
        price_col="close",
        k=2,
        timestamp_col="timestamp",
    )

    assert labeled_df.loc[0, "future_avg_r"] == 0.0
    assert labeled_df.loc[0, "label"] == 0.0
    assert diagnostics["n_zero_return"] == 1
    assert diagnostics["n_neutral"] == 0
    assert diagnostics["n_down"] == 2


def test_legacy_binary_diagnostics_mark_cross_day_horizon_invalid():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 15:50",
                    "2024-01-02 15:55",
                    "2024-01-03 09:30",
                    "2024-01-03 09:35",
                    "2024-01-03 09:40",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    labeled_df, diagnostics = runner.make_legacy_binary_labels_with_diagnostics(
        df,
        price_col="close",
        k=2,
        timestamp_col="timestamp",
    )

    assert labeled_df.loc[[0, 1], "label"].isna().all()
    assert labeled_df.loc[2, "label"] == 1.0
    assert diagnostics["n_cross_day"] == 2
    assert diagnostics["n_tail"] == 2
    assert diagnostics["n_neutral"] == 0


def test_stationary_v1_core_features_are_expected_and_finite():
    df = _stationary_raw_frame(days=1, bars_per_day=10)
    same_price_time = pd.Timestamp("2024-01-02 10:00")
    same_price_mask = df["timestamp"].eq(same_price_time)
    df.loc[same_price_mask, ["open", "high", "low", "close"]] = 106.0

    feature_df = runner.add_feature_set(df, "stationary_v1_core")

    assert tuple(runner.FEATURE_SETS["stationary_v1_core"]) == runner.STATIONARY_V1_CORE_FEATURES
    assert set(runner.STATIONARY_V1_CORE_FEATURES).issubset(feature_df.columns)
    assert np.isfinite(feature_df.loc[:, runner.STATIONARY_V1_CORE_FEATURES].to_numpy()).all()
    assert feature_df.loc[feature_df["timestamp"].eq(same_price_time), "body_to_range"].item() == 0.0
    assert runner.feature_diagnostics(df, feature_df)["feature_drop_count"] == 6


def test_stationary_v1_core_exact_values_for_hand_checked_row():
    df = _stationary_raw_frame(days=1, bars_per_day=10)

    feature_df = runner.add_feature_set(df, "stationary_v1_core")

    row = feature_df.loc[feature_df["timestamp"].eq(pd.Timestamp("2024-01-02 10:00"))].iloc[0]
    log_ret_1_values = [
        np.log(df.loc[index, "close"] / df.loc[index - 1, "close"])
        for index in range(1, 7)
    ]
    expected = {
        "log_ret_1": np.log(106.0 / 105.0),
        "log_ret_3": np.log(106.0 / 103.0),
        "log_ret_6": np.log(106.0 / 100.0),
        "oc_log_ret": np.log(106.0 / 105.75),
        "hl_log_range": np.log(106.5 / 105.5),
        "body_to_range": 0.25,
        "rv_6": np.std(log_ret_1_values, ddof=1),
        "log_volume_chg_1": np.log1p(1006.0) - np.log1p(1005.0),
    }
    for feature_name, expected_value in expected.items():
        assert row[feature_name] == pytest.approx(expected_value)


def test_stationary_v1_core_resets_lags_at_trading_day_boundary():
    df = _stationary_raw_frame(days=2, bars_per_day=8)

    feature_df = runner.add_feature_set(df, "stationary_v1_core")

    second_day_first_retained = pd.Timestamp("2024-01-03 10:00")
    observed = feature_df.loc[
        feature_df["timestamp"].eq(second_day_first_retained),
        "log_ret_6",
    ].item()
    day_mask = df["timestamp"].dt.date == second_day_first_retained.date()
    day_rows = df.loc[day_mask].reset_index(drop=True)
    expected = np.log(day_rows.loc[6, "close"] / day_rows.loc[0, "close"])
    assert observed == pytest.approx(expected)


def test_stationary_v1_core_groups_pooled_input_by_ticker_and_trading_day():
    aaa = _stationary_raw_frame(days=1, bars_per_day=10)
    aaa["ticker"] = "AAA"
    bbb = _stationary_raw_frame(days=1, bars_per_day=10)
    bbb["ticker"] = "BBB"
    bbb[["open", "high", "low", "close"]] += 100.0
    pooled = pd.concat([aaa, bbb], ignore_index=True).sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)

    feature_df = runner.add_feature_set(pooled, "stationary_v1_core")

    assert runner.feature_diagnostics(pooled, feature_df)["feature_drop_count"] == 12
    for ticker, offset in [("AAA", 0.0), ("BBB", 100.0)]:
        row = feature_df.loc[
            feature_df["ticker"].eq(ticker)
            & feature_df["timestamp"].eq(pd.Timestamp("2024-01-02 10:00"))
        ].iloc[0]
        assert row["log_ret_6"] == pytest.approx(np.log((106.0 + offset) / (100.0 + offset)))


def test_stationary_v1_core_rejects_duplicate_timestamp_inside_warmup():
    df = _stationary_raw_frame(days=1, bars_per_day=10)
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]

    with pytest.raises(ValueError) as exc_info:
        runner.add_feature_set(df, "stationary_v1_core")

    message = str(exc_info.value)
    assert "stationary_v1_core" in message
    assert "duplicate timestamp" in message
    assert "row/index 0" in message


def test_stationary_v1_core_rejects_cross_day_interleaved_rows_before_warmup_drop():
    df = _stationary_raw_frame(days=2, bars_per_day=10)
    interleaved = pd.concat(
        [df.iloc[:5], df.iloc[10:], df.iloc[5:10]],
        ignore_index=True,
    )

    with pytest.raises(ValueError) as exc_info:
        runner.add_feature_set(interleaved, "stationary_v1_core")

    message = str(exc_info.value)
    assert "stationary_v1_core" in message
    assert "raw dataframe" in message
    assert "strict monotonically increasing" in message
    assert "row/index 15" in message
    assert "previous timestamp" in message


def test_stationary_v1_core_rejects_reversed_trading_dates_before_labeling():
    df = _stationary_raw_frame(days=2, bars_per_day=10)
    reversed_days = pd.concat([df.iloc[10:], df.iloc[:10]], ignore_index=True)

    with pytest.raises(ValueError) as exc_info:
        runner.add_feature_set(reversed_days, "stationary_v1_core")

    message = str(exc_info.value)
    assert "stationary_v1_core" in message
    assert "raw dataframe" in message
    assert "strict monotonically increasing" in message
    assert "row/index 10" in message
    assert "previous timestamp" in message


def test_stationary_v1_core_prepare_data_rejects_out_of_order_csv_before_sorting(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = _stationary_raw_frame(days=5, bars_per_day=100)
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"] - pd.Timedelta(minutes=5)
    df.to_csv(data_dir / "AAA.csv", index=False)
    candidate = runner.CandidateSpec(
        "A",
        window_size=12,
        label_horizon_k=12,
        label_mode="legacy_binary",
        threshold_bps=0.0,
    )

    with pytest.raises(ValueError) as exc_info:
        runner.prepare_data(
            data_dir=data_dir,
            tickers=["AAA"],
            feature_set_id="stationary_v1_core",
            feature_cols=list(runner.STATIONARY_V1_CORE_FEATURES),
            candidate=candidate,
            max_rows_per_ticker=None,
            shuffle_train_labels=False,
            shuffle_seed=42,
        )

    message = str(exc_info.value)
    assert "strict monotonically increasing" in message
    assert "row/index 1" in message
    assert "previous timestamp" in message


def test_calendar_time_split_uses_half_open_intervals():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=6, freq="5min"),
            "label": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    spec = runner.CalendarSplitSpec(
        train_start_ts=pd.Timestamp("2024-01-02 09:30"),
        train_end_ts=pd.Timestamp("2024-01-02 09:40"),
        val_start_ts=pd.Timestamp("2024-01-02 09:40"),
        val_end_ts=pd.Timestamp("2024-01-02 09:50"),
        holdout_start_ts=pd.Timestamp("2024-01-02 09:50"),
        holdout_end_ts=pd.Timestamp("2024-01-02 10:00"),
    )

    train_df, val_df, holdout_df = runner.make_calendar_time_splits(
        df,
        spec,
        timestamp_col="timestamp",
        ticker="AAA",
    )

    assert train_df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 09:30"),
        pd.Timestamp("2024-01-02 09:35"),
    ]
    assert val_df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 09:40"),
        pd.Timestamp("2024-01-02 09:45"),
    ]
    assert holdout_df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 09:50"),
        pd.Timestamp("2024-01-02 09:55"),
    ]


def test_stationary_v1_core_valid_multi_day_multi_ticker_still_works():
    aaa = _stationary_raw_frame(days=2, bars_per_day=10)
    aaa["ticker"] = "AAA"
    bbb = _stationary_raw_frame(days=2, bars_per_day=10)
    bbb["ticker"] = "BBB"
    bbb[["open", "high", "low", "close"]] += 100.0
    pooled = pd.concat([aaa, bbb], ignore_index=True).sort_values(
        ["timestamp", "ticker"]
    ).reset_index(drop=True)

    feature_df = runner.add_feature_set(pooled, "stationary_v1_core")

    assert runner.feature_diagnostics(pooled, feature_df)["feature_drop_count"] == 24
    for ticker in ["AAA", "BBB"]:
        row = feature_df.loc[
            feature_df["ticker"].eq(ticker)
            & feature_df["timestamp"].eq(pd.Timestamp("2024-01-03 10:00"))
        ]
        assert len(row) == 1
        assert np.isfinite(row.loc[:, runner.STATIONARY_V1_CORE_FEATURES].to_numpy()).all()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.drop(columns=["close"]), "missing columns"),
        (lambda df: df.assign(open=np.nan), "open"),
        (lambda df: df.assign(close=0.0), "close"),
        (lambda df: df.assign(volume=-1.0), "volume"),
        (lambda df: df.assign(high=df["low"] - 0.01), "high"),
        (lambda df: df.assign(open=df["high"] + 0.01), "open"),
        (lambda df: df.assign(close=df["low"] - 0.01), "close"),
    ],
)
def test_stationary_v1_core_rejects_invalid_raw_input(mutate, message):
    df = mutate(_stationary_raw_frame(days=1, bars_per_day=10))

    with pytest.raises(ValueError, match=message):
        runner.add_feature_set(df, "stationary_v1_core")


def test_technical_v1_feature_construction_still_uses_technical_columns():
    df = _technical_raw_frame(rows=80)

    feature_df = runner.add_feature_set(df, "technical_v1")

    assert tuple(runner.FEATURE_SETS["technical_v1"]) == runner.TECHNICAL_FEATURES
    assert set(runner.TECHNICAL_FEATURES).issubset(feature_df.columns)
    assert "log_ret_1" not in feature_df.columns
    assert not feature_df.empty
    assert np.isfinite(feature_df.loc[:, runner.TECHNICAL_FEATURES].to_numpy()).all()


def test_default_feature_set_resolves_to_mentor_clean_v1():
    args = Namespace(feature_set=None)

    assert runner.resolve_feature_set(args, "smoke") == "mentor_clean_v1"
    assert runner.resolve_feature_set(args, "full") == "mentor_clean_v1"


def test_mentor_clean_v1_feature_set_excludes_raw_price_volume_and_raw_macd():
    df = _technical_raw_frame(rows=80)

    feature_df = runner.add_feature_set(df, "mentor_clean_v1")

    expected_features = {
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
    }
    forbidden_features = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "macd",
        "macd_signal",
        "macd_hist",
    }
    assert tuple(runner.FEATURE_SETS["mentor_clean_v1"]) == runner.MENTOR_CLEAN_V1_FEATURES
    assert set(runner.MENTOR_CLEAN_V1_FEATURES) == expected_features
    assert set(runner.MENTOR_CLEAN_V1_FEATURES).isdisjoint(forbidden_features)
    assert expected_features.issubset(feature_df.columns)
    assert np.isfinite(feature_df.loc[:, runner.MENTOR_CLEAN_V1_FEATURES].to_numpy()).all()


def test_mentor_clean_v1_exact_values_for_hand_checked_row():
    df = _technical_raw_frame(rows=80)

    feature_df = runner.add_feature_set(df, "mentor_clean_v1")

    timestamp = pd.Timestamp("2024-01-02 12:50")
    row = feature_df.loc[feature_df["timestamp"].eq(timestamp)].iloc[0]
    source_idx = df.index[df["timestamp"].eq(timestamp)][0]
    close = df["close"].astype(float)
    open_price = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    log_close = np.log(close)
    log_return = log_close.diff()
    log_volume = np.log1p(volume)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rolling_mean = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std()
    upper_band = rolling_mean + 2.0 * rolling_std
    lower_band = rolling_mean - 2.0 * rolling_std
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd - macd_signal
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    time_angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    expected = {
        "log_return": log_return.iloc[source_idx],
        "close_to_open_return": close.iloc[source_idx] / open_price.iloc[source_idx] - 1.0,
        "high_low_range": np.log(high.iloc[source_idx] / low.iloc[source_idx]),
        "rolling_volatility_20": log_return.rolling(window=20, min_periods=20).std().iloc[source_idx],
        "normalized_volume_20": (
            log_volume.iloc[source_idx]
            - log_volume.rolling(window=20, min_periods=20).mean().iloc[source_idx]
        ),
        "rsi_14": 100.0 - (100.0 / (1.0 + rs.iloc[source_idx])),
        "bollinger_pctb": (
            (close.iloc[source_idx] - lower_band.iloc[source_idx])
            / (upper_band.iloc[source_idx] - lower_band.iloc[source_idx])
        ),
        "normalized_macd_hist": macd_hist.iloc[source_idx] / close.iloc[source_idx],
        "time_of_day_sin": np.sin(time_angle),
        "time_of_day_cos": np.cos(time_angle),
    }
    for feature_name, expected_value in expected.items():
        assert row[feature_name] == pytest.approx(expected_value)


def test_mentor_clean_v1_resets_lagged_features_at_trading_day_boundary():
    df = _stationary_raw_frame(days=2, bars_per_day=60)

    feature_df = runner.add_feature_set(df, "mentor_clean_v1")

    second_day = pd.Timestamp("2024-01-03").date()
    second_day_features = feature_df.loc[feature_df["timestamp"].dt.date.eq(second_day)]
    first_second_day_row = second_day_features.iloc[0]
    source_idx = df.index[df["timestamp"].eq(first_second_day_row["timestamp"])][0]
    previous_source_idx = source_idx - 1
    assert df.loc[previous_source_idx, "timestamp"].date() == second_day
    assert first_second_day_row["log_return"] == pytest.approx(
        np.log(df.loc[source_idx, "close"] / df.loc[previous_source_idx, "close"])
    )
    assert first_second_day_row["rolling_volatility_20"] == pytest.approx(
        np.log(
            df.loc[source_idx - 19:source_idx, "close"].to_numpy(dtype=float)
            / df.loc[source_idx - 20:source_idx - 1, "close"].to_numpy(dtype=float)
        ).std(ddof=1)
    )


def test_sklearn_feature_views_keep_expected_shapes():
    dataset = _toy_window_dataset(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        [0, 1],
    )

    last_step = runner.dataset_features(dataset, "last_step")
    flattened = runner.dataset_features(dataset, "flatten_window")

    np.testing.assert_allclose(last_step, np.asarray([[5.0, 6.0], [11.0, 12.0]]))
    np.testing.assert_allclose(
        flattened,
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            ]
        ),
    )


def test_sklearn_logreg_selection_uses_validation_macro_f1_not_test_metrics():
    candidates = [
        {
            "C": 0.01,
            "class_weight": None,
            "val_metrics": {"macro_f1": 0.40},
            "test_metrics": {"macro_f1": 1.00},
        },
        {
            "C": 10.0,
            "class_weight": "balanced",
            "val_metrics": {"macro_f1": 0.60},
            "test_metrics": {"macro_f1": 0.00},
        },
    ]

    selected = runner.select_sklearn_logreg_candidate(candidates)

    assert selected["C"] == 10.0
    assert selected["class_weight"] == "balanced"


def test_sklearn_logreg_selection_breaks_ties_by_regularization_and_class_weight():
    candidates = [
        {"C": 1.0, "class_weight": None, "val_metrics": {"macro_f1": 0.50}},
        {"C": 0.1, "class_weight": "balanced", "val_metrics": {"macro_f1": 0.50}},
        {"C": 0.1, "class_weight": None, "val_metrics": {"macro_f1": 0.50}},
    ]

    selected = runner.select_sklearn_logreg_candidate(candidates)

    assert selected["C"] == 0.1
    assert selected["class_weight"] is None


def test_parse_logreg_c_grid_accepts_comma_separated_positive_values():
    values = runner.parse_logreg_c_grid(["0.001,0.003,0.01,0.03,0.1"])

    assert values == (0.001, 0.003, 0.01, 0.03, 0.1)


def test_parse_logreg_c_grid_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive"):
        runner.parse_logreg_c_grid(["0.1,0"])


def test_parse_logreg_class_weights_accepts_none_and_balanced():
    values = runner.parse_logreg_class_weights(["balanced,none"])

    assert values == ("balanced", None)


def test_parse_logreg_class_weights_rejects_unknown_values():
    with pytest.raises(ValueError, match="none/null or balanced"):
        runner.parse_logreg_class_weights(["prior"])


def test_sklearn_logreg_baseline_reports_finite_metrics_and_fixed_confusion_labels():
    prepared = _toy_prepared_data()

    rows = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=prepared,
        feature_view="last_step",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["model_name"] == "sklearn_logreg_l2"
    assert row["feature_view"] == "last_step"
    assert row["C"] in runner.SKLEARN_LOGREG_C_GRID
    assert row["class_weight"] in runner.SKLEARN_LOGREG_CLASS_WEIGHTS
    assert row["n_train_windows"] == len(prepared.train_dataset)
    assert row["n_val_windows"] == len(prepared.val_dataset)
    assert row["n_test_windows"] == len(prepared.test_dataset)
    assert json.loads(row["confusion_matrix_labels"]) == [0, 1]
    assert np.asarray(json.loads(row["confusion_matrix"])).shape == (2, 2)
    assert isinstance(row["n_iter"], int)
    assert isinstance(json.loads(row["warnings"]), list)
    assert row["claim_scope"] == "smoke_observation_not_performance_claim"
    assert row["diagnostic_only"] is True
    assert row["non_claim"] is True
    assert row["split"] == "test"
    assert "report_scope" not in row
    assert "test_metrics_embargoed" not in row
    for key in [
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_delta_macro_f1_vs_dummy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "delta_macro_f1_vs_dummy",
    ]:
        assert np.isfinite(row[key])


def test_sklearn_logreg_custom_c_grid_is_used_for_selection():
    row = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=_toy_prepared_data(),
        feature_view="last_step",
        c_grid=(0.003,),
    )[0]

    assert row["C"] == pytest.approx(0.003)


def test_sklearn_logreg_custom_class_weights_are_used():
    row = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=_toy_prepared_data(),
        feature_view="last_step",
        class_weights=("balanced",),
    )[0]

    assert row["class_weight"] == "balanced"


def test_sklearn_validation_only_report_embargoes_test_metric_fields():
    prepared = _toy_prepared_data()

    rows = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=prepared,
        feature_view="last_step",
        validation_only_report=True,
    )

    row = rows[0]
    assert row["report_scope"] == "validation_only"
    assert row["selection_scope"] == "validation_only"
    assert row["test_metrics_embargoed"] is True
    assert row["test_metrics_used"] is False
    assert row["split"] == "validation"
    _assert_validation_only_no_test_exposure(row)
    for key in [
        "model_macro_f1",
        "model_balanced_accuracy",
        "model_precision_macro",
        "model_recall_macro",
        "n_test_windows",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_up_pct",
        "delta_macro_f1_vs_dummy",
        "confusion_matrix_labels",
        "confusion_matrix",
        "classification_report",
        "dummy_stratified_macro_f1_mean",
        "ticker_dummy_stratified_macro_f1_mean",
    ]:
        assert key not in row


def test_sklearn_validation_only_report_keeps_validation_and_convergence_fields():
    prepared = _toy_prepared_data()

    row = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=prepared,
        feature_view="flatten_window",
        validation_only_report=True,
    )[0]

    assert row["feature_view"] == "flatten_window"
    assert row["C"] in runner.SKLEARN_LOGREG_C_GRID
    assert row["class_weight"] in runner.SKLEARN_LOGREG_CLASS_WEIGHTS
    assert row["n_val_windows"] == len(prepared.val_dataset)
    assert np.isfinite(row["val_up_pct"])
    for key in [
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_delta_macro_f1_vs_dummy",
        "val_dummy_stratified_macro_f1_mean",
    ]:
        assert np.isfinite(row[key])
    assert isinstance(row["converged"], bool)
    assert isinstance(row["n_iter"], int)
    assert isinstance(json.loads(row["warnings"]), list)
    assert row["suspicious_status"] == "ok"


def test_sklearn_validation_only_report_does_not_score_test_split(monkeypatch):
    prepared = _toy_prepared_data()
    original_dataset_features = runner.dataset_features
    original_compute_metrics = runner.compute_classification_metrics
    original_compute_baselines = runner.compute_baselines

    def guarded_dataset_features(dataset, feature_view):
        if dataset is prepared.test_dataset:
            pytest.fail("validation-only report should not featurize test split")
        return original_dataset_features(dataset, feature_view)

    def guarded_compute_metrics(y_true, y_pred):
        if y_true is prepared.y_test:
            pytest.fail("validation-only report should not compute test metrics")
        return original_compute_metrics(y_true, y_pred)

    def guarded_compute_baselines(y_train, y_eval):
        if y_eval is prepared.y_test:
            pytest.fail("validation-only report should not compute test baselines")
        return original_compute_baselines(y_train, y_eval)

    monkeypatch.setattr(runner, "dataset_features", guarded_dataset_features)
    monkeypatch.setattr(runner, "compute_classification_metrics", guarded_compute_metrics)
    monkeypatch.setattr(runner, "compute_baselines", guarded_compute_baselines)

    row = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=prepared,
        feature_view="last_step",
        validation_only_report=True,
    )[0]

    assert row["test_metrics_used"] is False


def test_sklearn_validation_only_per_ticker_outputs_pooled_and_ticker_rows():
    prepared = _toy_prepared_data()

    rows = runner.run_sklearn_logreg_baseline(
        metadata=_sklearn_metadata(),
        candidate=_candidate(),
        prepared=prepared,
        feature_view="last_step",
        validation_only_report=True,
        validation_only_per_ticker=True,
    )

    assert [row["ticker"] for row in rows] == ["pooled", "AAA"]
    ticker_row = rows[1]
    assert ticker_row["report_scope"] == "validation_only"
    assert ticker_row["selection_scope"] == "validation_only"
    assert ticker_row["n_val_windows"] == len(prepared.val_datasets_by_ticker["AAA"])
    assert np.isfinite(ticker_row["val_macro_f1"])
    assert np.isfinite(ticker_row["val_balanced_accuracy"])
    assert np.isfinite(ticker_row["val_delta_macro_f1_vs_dummy"])
    assert np.isfinite(ticker_row["val_dummy_stratified_macro_f1_mean"])
    assert ticker_row["C"] in runner.SKLEARN_LOGREG_C_GRID
    assert ticker_row["class_weight"] in runner.SKLEARN_LOGREG_CLASS_WEIGHTS
    assert isinstance(ticker_row["converged"], bool)
    assert isinstance(ticker_row["n_iter"], int)
    for row in rows:
        _assert_validation_only_no_test_exposure(row)


def test_sklearn_validation_only_keeps_shuffle_metadata_without_test_exposure():
    metadata = _sklearn_metadata()
    metadata["shuffle_train_labels"] = True

    row = runner.run_sklearn_logreg_baseline(
        metadata=metadata,
        candidate=_candidate(),
        prepared=_toy_prepared_data(),
        feature_view="last_step",
        validation_only_report=True,
    )[0]

    assert row["shuffle_train_labels"] is True
    _assert_validation_only_no_test_exposure(row)


def test_sklearn_baseline_cli_does_not_enter_torch_training_path(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()
    observed = {}

    def fake_prepare_data(**kwargs):
        observed["max_rows_per_ticker"] = kwargs["max_rows_per_ticker"]
        return prepared

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(
        runner,
        "run_model_once",
        lambda *args, **kwargs: pytest.fail("sklearn baseline should not train torch models"),
    )

    def fake_sklearn_baseline(
        metadata,
        candidate,
        prepared,
        feature_view,
        validation_only_report=False,
        validation_only_per_ticker=False,
        c_grid=runner.SKLEARN_LOGREG_C_GRID,
        class_weights=runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
    ):
        observed["feature_view"] = feature_view
        observed["model_family"] = metadata["model_family"]
        observed["validation_only_report"] = validation_only_report
        observed["validation_only_per_ticker"] = validation_only_per_ticker
        observed["c_grid"] = c_grid
        observed["class_weights"] = class_weights
        return [{"model_name": "sklearn_logreg_l2", "feature_view": feature_view}]

    monkeypatch.setattr(runner, "run_sklearn_logreg_baseline", fake_sklearn_baseline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--feature-view",
            "flatten_window",
            "--feature-set",
            "technical_v1",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
            "--max-rows-per-ticker",
            "1000",
        ],
    )

    runner.main()

    assert observed == {
        "feature_view": "flatten_window",
        "model_family": "sklearn_logreg",
        "max_rows_per_ticker": 1000,
        "validation_only_report": False,
        "validation_only_per_ticker": False,
        "c_grid": runner.SKLEARN_LOGREG_C_GRID,
        "class_weights": runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
    }
    metadata_path = next((tmp_path / "out").rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["max_rows_per_ticker"] == 1000
    assert metadata["effective_max_rows_per_ticker"] == 1000
    assert metadata["claim_scope"] == "smoke_observation_not_performance_claim"
    assert metadata["diagnostic_only"] is True
    assert metadata["non_claim"] is True
    results_path = next((tmp_path / "out").rglob("results.csv"))
    results = pd.read_csv(results_path)
    assert results.loc[0, "model_name"] == "sklearn_logreg_l2"


def test_sklearn_validation_only_cli_marks_metadata_and_result_scope(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()
    observed = {}

    def fake_prepare_data(**kwargs):
        return prepared

    def fake_sklearn_baseline(
        metadata,
        candidate,
        prepared,
        feature_view,
        validation_only_report=False,
        validation_only_per_ticker=False,
        c_grid=runner.SKLEARN_LOGREG_C_GRID,
        class_weights=runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
    ):
        observed["validation_only_report"] = validation_only_report
        observed["validation_only_per_ticker"] = validation_only_per_ticker
        observed["c_grid"] = c_grid
        observed["class_weights"] = class_weights
        observed["metadata_report_scope"] = metadata["report_scope"]
        observed["decision_time_policy"] = metadata["decision_time_policy"]
        observed["scaler_id"] = metadata["scaler_id"]
        observed["scaler_fit_scope"] = metadata["scaler_fit_scope"]
        observed["threshold_source"] = metadata["threshold_source"]
        return [
            {
                **runner.base_result_fields(metadata, candidate),
                **runner.validation_only_report_fields(),
                "model_name": "sklearn_logreg_l2",
            }
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(runner, "run_sklearn_logreg_baseline", fake_sklearn_baseline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--validation-only-report",
            "--feature-set",
            "mentor_clean_v1",
            "--label-mode",
            "no_trade_band",
            "--threshold-bps",
            "5.0",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    runner.main()

    assert observed == {
        "validation_only_report": True,
        "validation_only_per_ticker": False,
        "c_grid": runner.SKLEARN_LOGREG_C_GRID,
        "class_weights": runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
        "metadata_report_scope": "validation_only",
        "decision_time_policy": "post_bar_close_completed_bar",
        "scaler_id": "standard_pooled_train_only_v1",
        "scaler_fit_scope": "pooled_train_after_per_ticker_chronological_split",
        "threshold_source": "fixed_pre_registered_5bps",
    }
    metadata_path = next((tmp_path / "out").rglob("metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_set_id"] == "mentor_clean_v1"
    assert metadata["label_mode"] == "no_trade_band"
    assert metadata["threshold_bps"] == 5.0
    assert metadata["report_scope"] == "validation_only"
    assert metadata["selection_scope"] == "validation_only"
    assert metadata["test_metrics_embargoed"] is True
    assert metadata["test_metrics_used"] is False
    assert metadata["decision_time_policy"] == "post_bar_close_completed_bar"
    assert metadata["scaler_id"] == "standard_pooled_train_only_v1"
    assert metadata["scaler_fit_scope"] == "pooled_train_after_per_ticker_chronological_split"
    assert metadata["threshold_source"] == "fixed_pre_registered_5bps"
    results = pd.read_csv(next((tmp_path / "out").rglob("results.csv")))
    assert results.loc[0, "report_scope"] == "validation_only"
    assert bool(results.loc[0, "test_metrics_embargoed"]) is True
    assert results.loc[0, "decision_time_policy"] == "post_bar_close_completed_bar"
    assert results.loc[0, "scaler_id"] == "standard_pooled_train_only_v1"
    assert results.loc[0, "threshold_source"] == "fixed_pre_registered_5bps"


def test_validation_only_manifest_omits_holdout_and_test_exposure(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()

    def fake_prepare_data(**kwargs):
        return prepared

    def fake_sklearn_baseline(
        metadata,
        candidate,
        prepared,
        feature_view,
        validation_only_report=False,
        validation_only_per_ticker=False,
        c_grid=runner.SKLEARN_LOGREG_C_GRID,
        class_weights=runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
    ):
        return [
            {
                **runner.base_result_fields(metadata, candidate),
                **runner.validation_only_report_fields(),
                "model_name": "sklearn_logreg_l2",
            }
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(runner, "run_sklearn_logreg_baseline", fake_sklearn_baseline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--validation-only-report",
            "--feature-set",
            "mentor_clean_v1",
            "--label-mode",
            "no_trade_band",
            "--threshold-bps",
            "5.0",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    runner.main()

    manifest = pd.read_csv(next((tmp_path / "out").rglob("manifest.csv")))
    _assert_validation_only_manifest_no_holdout_test_exposure(manifest.columns)


def test_torch_validation_only_ms_dlinear_tcn_cli_uses_safe_branch(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()
    observed = {}

    def fake_prepare_data(**kwargs):
        observed["candidate"] = kwargs["candidate"]
        return prepared

    def fake_run_model_once(
        model_name,
        seed,
        max_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        early_stop_patience,
        output_dir,
        metadata,
        candidate,
        prepared,
        feature_cols,
        validation_only_report=False,
        validation_only_per_ticker=False,
    ):
        observed["model_name"] = model_name
        observed["metadata"] = metadata
        observed["feature_cols"] = feature_cols
        observed["validation_only_report"] = validation_only_report
        observed["validation_only_per_ticker"] = validation_only_per_ticker
        assert validation_only_report is True
        assert validation_only_per_ticker is True
        return [
            {
                **runner.base_result_fields(metadata, candidate),
                **runner.validation_only_report_fields(),
                "model_name": model_name,
                "model_family": "torch",
                "ticker": "pooled",
                "seed": seed,
                "split": "validation",
                "n_train_windows": len(prepared.train_dataset),
                "n_val_windows": len(prepared.val_dataset),
                "val_macro_f1": 0.5,
                "val_balanced_accuracy": 0.5,
                "val_delta_macro_f1_vs_dummy": 0.0,
                "best_epoch": 1,
                "best_val_macro_f1": 0.5,
                "val_dummy_stratified_macro_f1_mean": 0.5,
            },
            {
                **runner.base_result_fields(metadata, candidate),
                **runner.validation_only_report_fields(),
                "model_name": model_name,
                "model_family": "torch",
                "ticker": "AAA",
                "seed": seed,
                "split": "validation",
                "n_train_windows": len(prepared.train_dataset),
                "n_val_windows": len(prepared.val_datasets_by_ticker["AAA"]),
                "val_macro_f1": 0.5,
                "val_balanced_accuracy": 0.5,
                "val_delta_macro_f1_vs_dummy": 0.0,
                "best_epoch": 1,
                "best_val_macro_f1": 0.5,
                "val_dummy_stratified_macro_f1_mean": 0.5,
            },
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(runner, "run_model_once", fake_run_model_once)
    monkeypatch.setattr(
        runner,
        "evaluate_scope",
        lambda *args, **kwargs: pytest.fail(
            "torch validation-only route must not enter test evaluation"
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        _torch_validation_only_cli_args(tmp_path / "out"),
    )

    runner.main()

    assert observed["model_name"] == "ms_dlinear_tcn"
    assert observed["candidate"].label_mode == "no_trade_band"
    assert observed["candidate"].threshold_bps == 5.0
    assert observed["feature_cols"] == list(runner.FEATURE_SETS["mentor_clean_v1"])
    assert observed["metadata"]["models"] == ["ms_dlinear_tcn"]
    assert observed["metadata"]["model_family"] == "torch"
    _assert_mentor_clean_validation_only_metadata(observed["metadata"])
    results = pd.read_csv(next((tmp_path / "out").rglob("results.csv")))
    assert set(results["ticker"]) == {"pooled", "AAA"}
    assert set(results["split"]) == {"validation"}
    assert set(results["report_scope"]) == {"validation_only"}
    assert set(results["selection_scope"]) == {"validation_only"}
    assert set(results["test_metrics_embargoed"]) == {True}
    assert set(results["test_metrics_used"]) == {False}
    for row in results.to_dict(orient="records"):
        _assert_validation_only_no_test_exposure(row)
    manifest = pd.read_csv(next((tmp_path / "out").rglob("manifest.csv")))
    _assert_validation_only_manifest_no_holdout_test_exposure(manifest.columns)


@pytest.mark.parametrize(
    ("route_overrides", "expected_message"),
    [
        ({"feature_set": "ohlcv_only_v1"}, "mentor_clean_v1"),
        ({"label_mode": "legacy_binary"}, "no_trade_band"),
        ({"threshold_bps": None}, "threshold-bps 5.0"),
        ({"threshold_bps": "1.0"}, "threshold-bps 5.0"),
        ({"models": ["lstm"]}, "ms_dlinear_tcn"),
        ({"models": ["ms_dlinear_tcn", "lstm"]}, "ms_dlinear_tcn"),
        ({"full_run": True}, "full-run"),
    ],
)
def test_torch_validation_only_ms_dlinear_tcn_rejects_unlocked_route_before_training(
    tmp_path,
    monkeypatch,
    route_overrides,
    expected_message,
):
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda **kwargs: pytest.fail("invalid torch validation route should stop early"),
    )
    monkeypatch.setattr(
        runner,
        "run_model_once",
        lambda *args, **kwargs: pytest.fail(
            "invalid torch validation route should not train"
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        _torch_validation_only_cli_args(tmp_path / "out", **route_overrides),
    )

    with pytest.raises(ValueError, match=expected_message):
        runner.main()


def test_write_outputs_rejects_existing_run_directory(tmp_path):
    output_dir = tmp_path / "out"
    run_dir = output_dir / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "sentinel.txt").write_text("existing artifact", encoding="utf-8")

    with pytest.raises(FileExistsError, match="run_a"):
        runner.write_outputs(
            output_dir,
            "manifest",
            [{"ticker": "AAA", "candidate_id": "A"}],
            {"run_id": "run_a"},
        )


def test_validation_only_per_ticker_requires_validation_only_report(monkeypatch):
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda **kwargs: pytest.fail("invalid per-ticker mode should stop early"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--validation-only-per-ticker",
        ],
    )

    with pytest.raises(ValueError, match="validation-only-per-ticker"):
        runner.main()


def test_lightgbm_requires_validation_only_report(monkeypatch):
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda **kwargs: pytest.fail("invalid LightGBM mode should stop early"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--model-family",
            "lightgbm",
        ],
    )

    with pytest.raises(ValueError, match="lightgbm requires --validation-only-report"):
        runner.main()


def test_lightgbm_rejects_full_run_before_prepare_data(monkeypatch):
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda **kwargs: pytest.fail("invalid LightGBM full-run should stop early"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--model-family",
            "lightgbm",
            "--validation-only-report",
            "--full-run",
            "--feature-set",
            "mentor_clean_v1",
            "--label-mode",
            "no_trade_band",
            "--threshold-bps",
            "5.0",
        ],
    )

    with pytest.raises(ValueError, match="lightgbm.*--full-run"):
        runner.main()


@pytest.mark.parametrize(
    ("route_args", "message"),
    [
        (
            [
                "--feature-set",
                "technical_v1",
                "--label-mode",
                "no_trade_band",
                "--threshold-bps",
                "5.0",
            ],
            "lightgbm.*--feature-set mentor_clean_v1",
        ),
        (
            [
                "--feature-set",
                "mentor_clean_v1",
                "--label-mode",
                "legacy_binary",
            ],
            "lightgbm.*--label-mode no_trade_band",
        ),
        (
            [
                "--feature-set",
                "mentor_clean_v1",
                "--label-mode",
                "no_trade_band",
            ],
            "lightgbm.*--threshold-bps 5.0",
        ),
        (
            [
                "--feature-set",
                "mentor_clean_v1",
                "--label-mode",
                "no_trade_band",
                "--threshold-bps",
                "3.0",
            ],
            "lightgbm.*--threshold-bps 5.0",
        ),
    ],
)
def test_lightgbm_requires_pm_route_args_before_prepare_data(
    monkeypatch,
    route_args,
    message,
):
    monkeypatch.setattr(
        runner,
        "prepare_data",
        lambda **kwargs: pytest.fail("invalid LightGBM PM route should stop early"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--model-family",
            "lightgbm",
            "--validation-only-report",
            *route_args,
        ],
    )

    with pytest.raises(ValueError, match=message):
        runner.main()


def test_lightgbm_validation_only_cli_dispatches_without_torch_or_logreg(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()
    observed = {}

    def fake_prepare_data(**kwargs):
        return prepared

    def fake_lightgbm_baseline(
        metadata,
        candidate,
        prepared,
        feature_view,
        validation_only_per_ticker=False,
    ):
        observed["model_family"] = metadata["model_family"]
        observed["models"] = metadata["models"]
        observed["feature_view"] = feature_view
        observed["validation_only_per_ticker"] = validation_only_per_ticker
        return [
            {
                **runner.base_result_fields(metadata, candidate),
                **runner.validation_only_report_fields(),
                "model_name": runner.LIGHTGBM_MODEL_NAME,
                "model_family": "lightgbm",
            }
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(
        runner,
        "run_model_once",
        lambda *args, **kwargs: pytest.fail("LightGBM validation should not train torch"),
    )
    monkeypatch.setattr(
        runner,
        "run_sklearn_logreg_baseline",
        lambda *args, **kwargs: pytest.fail("LightGBM should not run sklearn_logreg"),
    )
    monkeypatch.setattr(
        runner,
        "run_lightgbm_validation_only_baseline",
        fake_lightgbm_baseline,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--model-family",
            "lightgbm",
            "--validation-only-report",
            "--feature-set",
            "mentor_clean_v1",
            "--label-mode",
            "no_trade_band",
            "--threshold-bps",
            "5.0",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    runner.main()

    assert observed == {
        "model_family": "lightgbm",
        "models": [runner.LIGHTGBM_MODEL_NAME],
        "feature_view": "last_step",
        "validation_only_per_ticker": False,
    }
    metadata = json.loads(next((tmp_path / "out").rglob("metadata.json")).read_text())
    assert metadata["model_family"] == "lightgbm"
    assert metadata["models"] == [runner.LIGHTGBM_MODEL_NAME]
    assert metadata["threshold_source"] == "fixed_pre_registered_5bps"
    results = pd.read_csv(next((tmp_path / "out").rglob("results.csv")))
    assert results.loc[0, "model_name"] == runner.LIGHTGBM_MODEL_NAME
    assert results.loc[0, "model_family"] == "lightgbm"


def test_lightgbm_validation_only_report_embargoes_test_metric_fields(monkeypatch):
    monkeypatch.setitem(runner.sys.modules, "lightgbm", _FakeLightGBMModule)

    rows = runner.run_lightgbm_validation_only_baseline(
        metadata={**_sklearn_metadata(), "model_family": "lightgbm"},
        candidate=_candidate(),
        prepared=_toy_prepared_data(),
        feature_view="last_step",
        validation_only_per_ticker=True,
    )

    assert [row["ticker"] for row in rows] == ["pooled", "AAA"]
    for row in rows:
        assert row["model_name"] == runner.LIGHTGBM_MODEL_NAME
        assert row["model_family"] == "lightgbm"
        assert row["split"] == "validation"
        assert row["report_scope"] == "validation_only"
        assert row["test_metrics_used"] is False
        assert np.isfinite(row["val_macro_f1"])
        _assert_validation_only_no_test_exposure(row)


def test_lightgbm_validation_only_report_does_not_score_test_split(monkeypatch):
    prepared = _toy_prepared_data()
    original_dataset_features = runner.dataset_features
    original_compute_metrics = runner.compute_classification_metrics
    original_compute_baselines = runner.compute_baselines

    def guarded_dataset_features(dataset, feature_view):
        if dataset is prepared.test_dataset:
            pytest.fail("LightGBM validation-only report should not featurize test split")
        return original_dataset_features(dataset, feature_view)

    def guarded_compute_metrics(y_true, y_pred):
        if y_true is prepared.y_test:
            pytest.fail("LightGBM validation-only report should not compute test metrics")
        return original_compute_metrics(y_true, y_pred)

    def guarded_compute_baselines(y_train, y_eval):
        if y_eval is prepared.y_test:
            pytest.fail("LightGBM validation-only report should not compute test baselines")
        return original_compute_baselines(y_train, y_eval)

    monkeypatch.setitem(runner.sys.modules, "lightgbm", _FakeLightGBMModule)
    monkeypatch.setattr(runner, "dataset_features", guarded_dataset_features)
    monkeypatch.setattr(runner, "compute_classification_metrics", guarded_compute_metrics)
    monkeypatch.setattr(runner, "compute_baselines", guarded_compute_baselines)

    rows = runner.run_lightgbm_validation_only_baseline(
        metadata={**_sklearn_metadata(), "model_family": "lightgbm"},
        candidate=_candidate(),
        prepared=prepared,
        feature_view="last_step",
        validation_only_per_ticker=True,
    )

    assert all(row["test_metrics_used"] is False for row in rows)


def test_fit_lightgbm_classifier_uses_locked_default_params(monkeypatch):
    _FakeLGBMClassifier.created = []
    monkeypatch.setitem(runner.sys.modules, "lightgbm", _FakeLightGBMModule)

    model, fit_info = runner.fit_lightgbm_classifier(
        np.asarray([[-1.0], [1.0]]),
        np.asarray([0, 1]),
    )

    assert model.fit_called is True
    assert _FakeLGBMClassifier.created[0].params == runner.default_lightgbm_params()
    assert fit_info["objective"] == "binary"
    assert fit_info["n_estimators"] == 100
    assert fit_info["learning_rate"] == 0.05
    assert fit_info["num_leaves"] == 31
    assert fit_info["random_state"] == 42
    assert fit_info["n_jobs"] == 1
    assert fit_info["verbosity"] == -1


def test_load_lightgbm_module_missing_dependency_message(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ModuleNotFoundError("missing", name="lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="lightgbm==4.6.0"):
        runner.load_lightgbm_module()


def test_load_lightgbm_module_preserves_internal_import_errors(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ModuleNotFoundError("missing dependency", name="lightgbm_dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        runner.load_lightgbm_module()


def test_sklearn_cli_passes_custom_logreg_and_window_controls(
    tmp_path,
    monkeypatch,
):
    observed = {}

    def fake_prepare_data(**kwargs):
        observed["window_size"] = kwargs["candidate"].window_size
        observed["label_horizon_k"] = kwargs["candidate"].label_horizon_k
        return _toy_prepared_data()

    def fake_sklearn_baseline(
        metadata,
        candidate,
        prepared,
        feature_view,
        validation_only_report=False,
        validation_only_per_ticker=False,
        c_grid=runner.SKLEARN_LOGREG_C_GRID,
        class_weights=runner.SKLEARN_LOGREG_CLASS_WEIGHTS,
    ):
        observed["metadata_window_size"] = metadata["window_size"]
        observed["metadata_label_horizon_k"] = metadata["label_horizon_k"]
        observed["metadata_logreg_c_grid"] = metadata["logreg_c_grid"]
        observed["metadata_class_weights"] = metadata["logreg_class_weights"]
        observed["c_grid"] = c_grid
        observed["class_weights"] = class_weights
        return [
            {
                "model_name": "sklearn_logreg_l2",
                "window_size": candidate.window_size,
                "label_horizon_k": candidate.label_horizon_k,
            }
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(runner, "run_sklearn_logreg_baseline", fake_sklearn_baseline)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--feature-set",
            "technical_v1",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
            "--window-size",
            "24",
            "--logreg-c-grid",
            "0.001,0.003,0.01,0.03,0.1",
            "--logreg-class-weights",
            "balanced,none",
        ],
    )

    runner.main()

    assert observed == {
        "window_size": 24,
        "label_horizon_k": 12,
        "metadata_window_size": 24,
        "metadata_label_horizon_k": 12,
        "metadata_logreg_c_grid": [0.001, 0.003, 0.01, 0.03, 0.1],
        "metadata_class_weights": ["balanced", "none"],
        "c_grid": (0.001, 0.003, 0.01, 0.03, 0.1),
        "class_weights": ("balanced", None),
    }
    metadata = json.loads(
        next((tmp_path / "out").rglob("metadata.json")).read_text(encoding="utf-8")
    )
    assert metadata["window_size"] == 24
    assert metadata["label_horizon_k"] == 12
    results = pd.read_csv(next((tmp_path / "out").rglob("results.csv")))
    assert results.loc[0, "window_size"] == 24
    assert results.loc[0, "label_horizon_k"] == 12


def test_sklearn_model_family_switch_selects_logreg_without_alias():
    args = _args("legacy_binary")
    args.model_family = "sklearn_logreg"
    args.sklearn_baseline = False

    assert runner.resolve_model_family(args) == "sklearn_logreg"


def test_build_model_constructs_ms_dlinear_tcn_with_seq_len_and_input_size():
    model = runner.build_model("ms_dlinear_tcn", seq_len=12, input_size=10)
    x = torch.randn(2, 12, 10)

    logits = model(x)

    assert model.__class__.__name__ == "MultiScaleDLinearTCNClassifier"
    assert model.seq_len == 12
    assert model.input_size == 10
    assert logits.shape == (2, 2)


def test_build_model_rejects_unknown_model_name_still():
    with pytest.raises(ValueError, match="unknown model name"):
        runner.build_model("missing_model", seq_len=12, input_size=10)


def test_torch_cli_can_select_ms_dlinear_tcn_without_real_training(
    tmp_path,
    monkeypatch,
):
    observed = {}

    def fake_prepare_data(**kwargs):
        observed["candidate"] = kwargs["candidate"]
        return _toy_prepared_data()

    def fake_run_model_once(
        model_name,
        seed,
        max_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        early_stop_patience,
        output_dir,
        metadata,
        candidate,
        prepared,
        feature_cols,
        validation_only_report=False,
        validation_only_per_ticker=False,
    ):
        observed["model_name"] = model_name
        observed["metadata"] = metadata
        observed["feature_cols"] = feature_cols
        observed["validation_only_report"] = validation_only_report
        observed["validation_only_per_ticker"] = validation_only_per_ticker
        return [
            {
                **runner.base_result_fields(metadata, candidate),
                "model_name": model_name,
                "seed": seed,
                "report_scope": "dispatch_only",
            }
        ]

    monkeypatch.setattr(runner, "prepare_data", fake_prepare_data)
    monkeypatch.setattr(runner, "run_model_once", fake_run_model_once)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--models",
            "ms_dlinear_tcn",
            "--feature-set",
            "mentor_clean_v1",
            "--label-mode",
            "no_trade_band",
            "--threshold-bps",
            "5.0",
            "--tickers",
            "AAA",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    runner.main()

    assert observed["model_name"] == "ms_dlinear_tcn"
    assert observed["candidate"].label_mode == "no_trade_band"
    assert observed["candidate"].threshold_bps == 5.0
    assert observed["metadata"]["models"] == ["ms_dlinear_tcn"]
    assert observed["metadata"]["decision_time_policy"] == "post_bar_close_completed_bar"
    assert observed["metadata"]["scaler_id"] == "standard_pooled_train_only_v1"
    assert observed["metadata"]["threshold_source"] == "fixed_pre_registered_5bps"
    assert observed["validation_only_report"] is False
    assert observed["validation_only_per_ticker"] is False
    results = pd.read_csv(next((tmp_path / "out").rglob("results.csv")))
    assert results.loc[0, "model_name"] == "ms_dlinear_tcn"


def test_run_model_once_torch_validation_only_uses_train_and_validation_not_test(
    tmp_path,
    monkeypatch,
):
    prepared = _toy_prepared_data()
    prepared.test_datasets_by_ticker["AAA"] = prepared.test_dataset
    observed = {}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.classifier(x[:, -1, :])

    class FakeTrainer:
        def __init__(self, **kwargs):
            observed["checkpoint_dir"] = kwargs["checkpoint_dir"]

        def fit(self, train_loader, val_loader, num_epochs):
            observed["fit_loaders"] = (train_loader, val_loader)
            observed["num_epochs"] = num_epochs
            return {"best_epoch": 1, "best_metric": 0.5}

    def fake_make_loader(dataset, batch_size, shuffle, seed):
        if dataset is prepared.train_dataset:
            observed["train_shuffle"] = shuffle
            return "train_loader"
        if dataset is prepared.val_dataset:
            validation_loaders = observed.setdefault("validation_loaders", [])
            scope_name = "pooled" if "pooled" not in validation_loaders else "AAA"
            validation_loaders.append(scope_name)
            return f"{scope_name}_validation_loader"
        if dataset is prepared.val_datasets_by_ticker["AAA"]:
            observed.setdefault("validation_loaders", []).append("AAA")
            return "AAA_validation_loader"
        if dataset is prepared.test_dataset:
            pytest.fail("torch validation-only route should not build a test loader")
        if any(dataset is value for value in prepared.test_datasets_by_ticker.values()):
            pytest.fail("torch validation-only route should not build ticker test loaders")
        raise AssertionError(f"unexpected dataset: {dataset!r}")

    def fake_evaluate(model, loader, criterion, device):
        assert loader in {"pooled_validation_loader", "AAA_validation_loader"}
        return (
            {
                "macro_f1": 0.5,
                "balanced_accuracy": 0.5,
                "precision_macro": 0.5,
                "recall_macro": 0.5,
                "confusion_matrix": np.asarray([[1, 1], [1, 1]]),
                "classification_report": {},
            },
            None,
            None,
        )

    def guarded_compute_baselines(y_train, y_eval):
        if y_eval is prepared.y_test:
            pytest.fail("torch validation-only route should not baseline pooled test labels")
        if any(y_eval is value for value in prepared.y_test_by_ticker.values()):
            pytest.fail("torch validation-only route should not baseline ticker test labels")
        return {
            "dummy_stratified_macro_f1_mean": 0.5,
            "dummy_stratified_macro_f1_std": 0.0,
            "dummy_prior_macro_f1": 0.5,
            "always_up_macro_f1": 0.5,
            "always_down_macro_f1": 0.5,
        }

    metadata = {
        **_sklearn_metadata(),
        **runner.validation_only_report_fields(),
        "feature_set_id": "mentor_clean_v1",
        "label_mode": "no_trade_band",
        "threshold_bps": 5.0,
        "threshold_source": "fixed_pre_registered_5bps",
        "decision_time_policy": "post_bar_close_completed_bar",
        "scaler_id": "standard_pooled_train_only_v1",
        "scaler_fit_scope": "pooled_train_after_per_ticker_chronological_split",
        "model_family": "torch",
        "models": ["ms_dlinear_tcn"],
    }
    candidate = runner.CandidateSpec(
        "A",
        window_size=2,
        label_horizon_k=2,
        label_mode="no_trade_band",
        threshold_bps=5.0,
    )

    monkeypatch.setattr(runner, "build_model", lambda *args, **kwargs: TinyModel())
    monkeypatch.setattr(runner, "Trainer", FakeTrainer)
    monkeypatch.setattr(runner, "make_loader", fake_make_loader)
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner, "compute_baselines", guarded_compute_baselines)

    rows = runner.run_model_once(
        model_name="ms_dlinear_tcn",
        seed=42,
        max_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        early_stop_patience=1,
        output_dir=tmp_path / "out",
        metadata=metadata,
        candidate=candidate,
        prepared=prepared,
        feature_cols=["feature_a", "feature_b"],
        validation_only_report=True,
        validation_only_per_ticker=True,
    )

    assert observed["fit_loaders"] == ("train_loader", "pooled_validation_loader")
    assert observed["train_shuffle"] is True
    assert set(observed["validation_loaders"]) == {"pooled", "AAA"}
    assert [row["ticker"] for row in rows] == ["pooled", "AAA"]
    for row in rows:
        assert row["split"] == "validation"
        assert row["report_scope"] == "validation_only"
        assert row["selection_scope"] == "validation_only"
        assert row["test_metrics_embargoed"] is True
        assert row["test_metrics_used"] is False
        _assert_validation_only_no_test_exposure(row)


def test_stationary_v1_core_manifest_only_writes_feature_metadata_without_training(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _stationary_raw_frame(days=5, bars_per_day=100).to_csv(data_dir / "AAA.csv", index=False)
    monkeypatch.setattr(
        runner,
        "run_model_once",
        lambda *args, **kwargs: pytest.fail("manifest-only should not train"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--manifest-only",
            "--feature-set",
            "stationary_v1_core",
            "--tickers",
            "AAA",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    runner.main()

    manifest_path = next(output_dir.rglob("manifest.csv"))
    manifest = pd.read_csv(manifest_path)
    metadata = json.loads(next(output_dir.rglob("metadata.json")).read_text())
    pooled = manifest.loc[manifest["ticker"].eq("pooled")].iloc[0]
    assert metadata["split_date_ranges_available"] is True
    assert metadata["split_date_range_timestamp_col"] == "timestamp"
    assert (
        metadata["split_date_range_source"]
        == "prepared_split_frames_after_feature_label_filtering"
    )
    assert pooled["feature_set_id"] == "stationary_v1_core"
    assert json.loads(pooled["feature_columns"]) == list(runner.STATIONARY_V1_CORE_FEATURES)
    assert pooled["feature_drop_count"] == 30
    assert pooled["feature_drop_pct"] == pytest.approx(30 / 500)
    assert not list(output_dir.rglob("results.csv"))


def test_calendar_manifest_records_filtered_date_ranges(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _calendar_wave_frame(days=3, bars_per_day=90).to_csv(data_dir / "AAA.csv", index=False)
    monkeypatch.setattr(
        runner,
        "run_model_once",
        lambda *args, **kwargs: pytest.fail("manifest-only should not train"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--manifest-only",
            "--feature-set",
            "ohlcv_only_v1",
            "--tickers",
            "AAA",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--split-mode",
            "calendar",
            "--train-start-ts",
            "2024-01-02 09:30",
            "--train-end-ts",
            "2024-01-02 17:00",
            "--val-start-ts",
            "2024-01-03 09:30",
            "--val-end-ts",
            "2024-01-03 17:00",
            "--holdout-start-ts",
            "2024-01-04 09:30",
            "--holdout-end-ts",
            "2024-01-04 17:00",
        ],
    )

    runner.main()

    metadata = json.loads(next(output_dir.rglob("metadata.json")).read_text())
    manifest = pd.read_csv(next(output_dir.rglob("manifest.csv")))
    pooled = manifest.loc[manifest["ticker"].eq("pooled")].iloc[0]
    assert metadata["split_mode"] == "calendar"
    assert (
        metadata["calendar_interval_convention"]
        == "half_open_start_inclusive_end_exclusive"
    )
    assert metadata["calendar_train_start_ts"] == "2024-01-02T09:30:00"
    assert metadata["calendar_holdout_end_ts"] == "2024-01-04T17:00:00"
    assert pooled["train_start_ts"] == "2024-01-02T09:30:00"
    assert pooled["train_end_ts"] == "2024-01-02T16:55:00"
    assert pooled["val_start_ts"] == "2024-01-03T09:30:00"
    assert pooled["val_end_ts"] == "2024-01-03T16:55:00"
    assert pooled["holdout_start_ts"] == "2024-01-04T09:30:00"
    assert pooled["holdout_end_ts"] == "2024-01-04T16:55:00"


def test_calendar_validation_only_output_embargoes_test_fields(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    _calendar_wave_frame(days=3, bars_per_day=90).to_csv(data_dir / "AAA.csv", index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "local_baseline_matrix.py",
            "--sklearn-baseline",
            "--validation-only-report",
            "--validation-only-per-ticker",
            "--feature-set",
            "ohlcv_only_v1",
            "--feature-view",
            "flatten_window",
            "--window-size",
            "6",
            "--tickers",
            "AAA",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--split-mode",
            "calendar",
            "--train-start-ts",
            "2024-01-02 09:30",
            "--train-end-ts",
            "2024-01-02 17:00",
            "--val-start-ts",
            "2024-01-03 09:30",
            "--val-end-ts",
            "2024-01-03 17:00",
            "--holdout-start-ts",
            "2024-01-04 09:30",
            "--holdout-end-ts",
            "2024-01-04 17:00",
        ],
    )

    runner.main()

    metadata = json.loads(next(output_dir.rglob("metadata.json")).read_text())
    results = pd.read_csv(next(output_dir.rglob("results.csv")))
    assert metadata["split_mode"] == "calendar"
    assert set(results["split"]) == {"validation"}
    assert set(results["report_scope"]) == {"validation_only"}
    assert set(results["selection_scope"]) == {"validation_only"}
    for row in results.to_dict(orient="records"):
        _assert_validation_only_no_test_exposure(row)
    manifest = pd.read_csv(next(output_dir.rglob("manifest.csv")))
    _assert_validation_only_manifest_no_holdout_test_exposure(manifest.columns)


def test_manifest_rows_include_pooled_validation_observability():
    metadata = {
        "run_id": "run_a",
        "git_commit_hash": "abc123",
        "data_source": "synthetic",
        "feature_set_id": "ohlcv_only_v1",
        "label_mode": "legacy_binary",
        **runner.protocol_metadata_fields("ohlcv_only_v1", "legacy_binary", 0.0, None),
        "label_semantics": "canonical_phase1_full_binary",
        "label_formula": "label = 1 if future_avg_r > 0 else 0",
        "class_0_name": "non_up",
        "class_1_name": "up",
        "zero_return_policy": "class_0_non_up",
        "no_trade_band_enabled": False,
        "neutral_policy": "not_applicable",
        "timestamp_col": "timestamp",
        "price_col": "close",
        "shuffle_train_labels": False,
        "shuffle_seed": 42,
        "checkpoint_policy": "best_val_macro_f1",
        "training_scope": "pooled",
        "baseline_scope": "pooled_train",
        "primary_baseline_scope": "pooled_train",
        "dummy_stratified_random_states": list(range(10)),
        "max_rows_per_ticker": 1000,
        "effective_max_rows_per_ticker": 1000,
        "claim_scope": "smoke_observation_not_performance_claim",
        "diagnostic_scope": "bounded_smoke_pipeline_diagnostic",
        "diagnostic_only": True,
        "non_claim": True,
        "tickers": ["CSCO", "MSFT"],
    }
    candidate = runner.CandidateSpec(
        "A",
        window_size=12,
        label_horizon_k=12,
        label_mode="legacy_binary",
        threshold_bps=0.0,
    )
    prepared = runner.PreparedData(
        train_df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        val_datasets_by_ticker={},
        test_datasets_by_ticker={},
        y_train=np.asarray([0, 1, 1, 0, 1]),
        y_val=np.asarray([0, 1, 1]),
        y_test=np.asarray([1, 0, 1, 0]),
        y_train_by_ticker={
            "CSCO": np.asarray([0, 1, 1]),
            "MSFT": np.asarray([0, 1]),
        },
        y_val_by_ticker={
            "CSCO": np.asarray([0, 1]),
            "MSFT": np.asarray([1]),
        },
        y_test_by_ticker={
            "CSCO": np.asarray([1, 0]),
            "MSFT": np.asarray([1, 0]),
        },
        diagnostics_by_ticker={
            "CSCO_label": _label_diag(n_total=100, n_up=40, n_down=50),
            "MSFT_label": _label_diag(n_total=80, n_up=30, n_down=35),
            "CSCO_train": _split_diag(n_rows=60, n_retained=54, n_nan=6),
            "CSCO_val": _split_diag(n_rows=20, n_retained=18, n_nan=2),
            "CSCO_test": _split_diag(n_rows=20, n_retained=18, n_nan=2),
            "MSFT_train": _split_diag(n_rows=48, n_retained=42, n_nan=6),
            "MSFT_val": _split_diag(n_rows=16, n_retained=14, n_nan=2),
            "MSFT_test": _split_diag(n_rows=16, n_retained=14, n_nan=2),
        },
    )

    rows = runner.build_manifest_rows(metadata, candidate, prepared)

    csco_row = next(row for row in rows if row["ticker"] == "CSCO")
    pooled_row = next(row for row in rows if row["ticker"] == "pooled")
    assert csco_row["n_val_windows"] == 2
    assert csco_row["max_rows_per_ticker"] == 1000
    assert csco_row["effective_max_rows_per_ticker"] == 1000
    assert csco_row["claim_scope"] == "smoke_observation_not_performance_claim"
    assert csco_row["diagnostic_only"] is True
    assert csco_row["non_claim"] is True
    assert csco_row["val_up_pct"] == pytest.approx(0.5)
    assert csco_row["val_retained_labels"] == 18
    assert csco_row["val_nan_labels"] == 2
    assert pooled_row["train_rows"] == 108
    assert pooled_row["val_rows"] == 36
    assert pooled_row["test_rows"] == 36
    assert pooled_row["train_retained_labels"] == 96
    assert pooled_row["val_retained_labels"] == 32
    assert pooled_row["test_retained_labels"] == 32
    assert pooled_row["train_nan_labels"] == 12
    assert pooled_row["val_nan_labels"] == 4
    assert pooled_row["test_nan_labels"] == 4
    assert pooled_row["n_val_windows"] == 3
    assert pooled_row["val_up_pct"] == pytest.approx(2 / 3)


def test_manifest_rows_include_split_date_ranges_for_tickers_and_pooled():
    metadata = _manifest_metadata(["CSCO", "MSFT"])
    candidate = _candidate()
    prepared = runner.PreparedData(
        train_df=pd.concat(
            [
                _split_frame("CSCO", ["2024-01-02 09:30", "2024-01-02 09:35"]),
                _split_frame("MSFT", ["2024-01-02 09:45", "2024-01-02 09:50"]),
            ],
            ignore_index=True,
        ),
        val_df=pd.concat(
            [
                _split_frame("CSCO", ["2024-01-03 09:30", "2024-01-03 09:35"]),
                _split_frame("MSFT", ["2024-01-03 10:00", "2024-01-03 10:05"]),
            ],
            ignore_index=True,
        ),
        test_df=pd.concat(
            [
                _split_frame("CSCO", ["2024-01-04 09:30", "2024-01-04 09:35"]),
                _split_frame("MSFT", ["2024-01-04 09:55", "2024-01-04 10:00"]),
            ],
            ignore_index=True,
        ),
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        val_datasets_by_ticker={},
        test_datasets_by_ticker={},
        y_train=np.asarray([0, 1, 0, 1]),
        y_val=np.asarray([0, 1, 1, 0]),
        y_test=np.asarray([1, 0, 0, 1]),
        y_train_by_ticker={
            "CSCO": np.asarray([0, 1]),
            "MSFT": np.asarray([0, 1]),
        },
        y_val_by_ticker={
            "CSCO": np.asarray([0, 1]),
            "MSFT": np.asarray([1, 0]),
        },
        y_test_by_ticker={
            "CSCO": np.asarray([1, 0]),
            "MSFT": np.asarray([0, 1]),
        },
        diagnostics_by_ticker={
            "CSCO_label": _label_diag(n_total=12, n_up=4, n_down=4),
            "MSFT_label": _label_diag(n_total=12, n_up=4, n_down=4),
            "CSCO_train": _split_diag(n_rows=2, n_retained=2, n_nan=0),
            "CSCO_val": _split_diag(n_rows=2, n_retained=2, n_nan=0),
            "CSCO_test": _split_diag(n_rows=2, n_retained=2, n_nan=0),
            "MSFT_train": _split_diag(n_rows=2, n_retained=2, n_nan=0),
            "MSFT_val": _split_diag(n_rows=2, n_retained=2, n_nan=0),
            "MSFT_test": _split_diag(n_rows=2, n_retained=2, n_nan=0),
        },
    )

    rows = runner.build_manifest_rows(metadata, candidate, prepared)

    csco_row = next(row for row in rows if row["ticker"] == "CSCO")
    msft_row = next(row for row in rows if row["ticker"] == "MSFT")
    pooled_row = next(row for row in rows if row["ticker"] == "pooled")
    for row in rows:
        for key in [
            "train_start_ts",
            "train_end_ts",
            "val_start_ts",
            "val_end_ts",
            "holdout_start_ts",
            "holdout_end_ts",
        ]:
            assert key in row
    assert csco_row["train_start_ts"] == "2024-01-02T09:30:00"
    assert csco_row["train_end_ts"] == "2024-01-02T09:35:00"
    assert csco_row["val_start_ts"] == "2024-01-03T09:30:00"
    assert csco_row["val_end_ts"] == "2024-01-03T09:35:00"
    assert csco_row["holdout_start_ts"] == "2024-01-04T09:30:00"
    assert csco_row["holdout_end_ts"] == "2024-01-04T09:35:00"
    assert msft_row["train_start_ts"] == "2024-01-02T09:45:00"
    assert msft_row["train_end_ts"] == "2024-01-02T09:50:00"
    assert pooled_row["train_start_ts"] == "2024-01-02T09:30:00"
    assert pooled_row["train_end_ts"] == "2024-01-02T09:50:00"
    assert pooled_row["val_start_ts"] == "2024-01-03T09:30:00"
    assert pooled_row["val_end_ts"] == "2024-01-03T10:05:00"
    assert pooled_row["holdout_start_ts"] == "2024-01-04T09:30:00"
    assert pooled_row["holdout_end_ts"] == "2024-01-04T10:00:00"


def _assert_validation_only_no_test_exposure(row):
    allowed_test_prefix_fields = {"test_metrics_embargoed", "test_metrics_used"}
    forbidden = [
        "model_macro_f1",
        "model_balanced_accuracy",
        "model_precision_macro",
        "model_recall_macro",
        "n_test_windows",
        "test_up_pct",
        "delta_macro_f1_vs_dummy",
        "confusion_matrix_labels",
        "confusion_matrix",
        "classification_report",
        "dummy_stratified_macro_f1_mean",
        "ticker_dummy_stratified_macro_f1_mean",
    ]
    for key in forbidden:
        assert key not in row
    leaked_test_fields = [
        key for key in row if key.startswith("test_") and key not in allowed_test_prefix_fields
    ]
    assert leaked_test_fields == []


def _assert_validation_only_manifest_no_holdout_test_exposure(columns):
    forbidden = {
        "test_rows",
        "test_retained_labels",
        "test_nan_labels",
        "n_test_windows",
        "test_up_pct",
        "holdout_start_ts",
        "holdout_end_ts",
    }
    assert sorted(forbidden.intersection(columns)) == []


def _assert_mentor_clean_validation_only_metadata(metadata):
    assert metadata["feature_set_id"] == "mentor_clean_v1"
    assert metadata["label_mode"] == "no_trade_band"
    assert metadata["threshold_bps"] == 5.0
    assert metadata["report_scope"] == "validation_only"
    assert metadata["selection_scope"] == "validation_only"
    assert metadata["test_metrics_embargoed"] is True
    assert metadata["test_metrics_used"] is False
    assert metadata["decision_time_policy"] == "post_bar_close_completed_bar"
    assert metadata["scaler_id"] == "standard_pooled_train_only_v1"
    assert (
        metadata["scaler_fit_scope"]
        == "pooled_train_after_per_ticker_chronological_split"
    )
    assert metadata["threshold_source"] == "fixed_pre_registered_5bps"


def _torch_validation_only_cli_args(
    output_dir,
    *,
    feature_set="mentor_clean_v1",
    label_mode="no_trade_band",
    threshold_bps="5.0",
    models=("ms_dlinear_tcn",),
    full_run=False,
):
    args = [
        "local_baseline_matrix.py",
        "--model-family",
        "torch",
        "--models",
        *models,
        "--validation-only-report",
        "--validation-only-per-ticker",
        "--feature-set",
        feature_set,
        "--label-mode",
        label_mode,
        "--tickers",
        "AAA",
        "--output-dir",
        str(output_dir),
    ]
    if threshold_bps is not None:
        args.extend(["--threshold-bps", threshold_bps])
    if full_run:
        args.append("--full-run")
    return args


def _label_diag(n_total, n_up, n_down):
    return {
        "n_total": n_total,
        "n_tail": 2,
        "n_cross_day": 3,
        "n_neutral": 5,
        "n_up": n_up,
        "n_down": n_down,
        "n_zero_return": 1,
    }


def _split_diag(n_rows, n_retained, n_nan):
    return {
        "n_rows": n_rows,
        "n_retained_labels": n_retained,
        "n_nan_labels": n_nan,
        "up_pct": 0.5,
    }


def _manifest_metadata(tickers):
    metadata = _sklearn_metadata()
    metadata["tickers"] = tickers
    return metadata


def _split_frame(ticker, timestamps):
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "ticker": ticker,
            "label": [0.0] * len(timestamps),
        }
    )


class _ToyWindowDataset:
    def __init__(self, windows, labels):
        self._windows = [
            torch.as_tensor(window, dtype=torch.float32)
            for window in windows
        ]
        self._labels = [
            torch.as_tensor(label, dtype=torch.long)
            for label in labels
        ]

    def __len__(self):
        return len(self._labels)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        return self._windows[index], self._labels[index]


class _FakeLGBMClassifier:
    created = []

    def __init__(self, **params):
        self.params = params
        self.fit_called = False
        _FakeLGBMClassifier.created.append(self)

    def fit(self, x_train, y_train):
        self.fit_called = True
        self.x_train_shape = x_train.shape
        self.y_train_shape = y_train.shape
        return self

    def predict(self, x_eval):
        return (x_eval[:, 0] >= 0.0).astype(int)


class _FakeLightGBMModule:
    LGBMClassifier = _FakeLGBMClassifier


def _toy_window_dataset(windows, labels):
    return _ToyWindowDataset(windows, labels)


def _toy_prepared_data():
    train_dataset = _toy_window_dataset(
        [
            [[-2.0, 0.0], [-2.0, 0.0]],
            [[-1.5, 0.1], [-1.5, 0.1]],
            [[-1.0, 0.2], [-1.0, 0.2]],
            [[-0.5, 0.3], [-0.5, 0.3]],
            [[0.5, 0.4], [0.5, 0.4]],
            [[1.0, 0.5], [1.0, 0.5]],
            [[1.5, 0.6], [1.5, 0.6]],
            [[2.0, 0.7], [2.0, 0.7]],
        ],
        [0, 0, 0, 0, 1, 1, 1, 1],
    )
    val_dataset = _toy_window_dataset(
        [
            [[-1.2, 0.0], [-1.2, 0.0]],
            [[-0.8, 0.1], [-0.8, 0.1]],
            [[0.8, 0.2], [0.8, 0.2]],
            [[1.2, 0.3], [1.2, 0.3]],
        ],
        [0, 0, 1, 1],
    )
    test_dataset = _toy_window_dataset(
        [
            [[-1.4, 0.0], [-1.4, 0.0]],
            [[-0.6, 0.1], [-0.6, 0.1]],
            [[0.6, 0.2], [0.6, 0.2]],
            [[1.4, 0.3], [1.4, 0.3]],
        ],
        [0, 0, 1, 1],
    )
    return runner.PreparedData(
        train_df=pd.DataFrame(),
        val_df=pd.DataFrame(),
        test_df=pd.DataFrame(),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        val_datasets_by_ticker={"AAA": val_dataset},
        test_datasets_by_ticker={},
        y_train=runner.dataset_labels(train_dataset),
        y_val=runner.dataset_labels(val_dataset),
        y_test=runner.dataset_labels(test_dataset),
        y_train_by_ticker={"AAA": runner.dataset_labels(train_dataset)},
        y_val_by_ticker={"AAA": runner.dataset_labels(val_dataset)},
        y_test_by_ticker={"AAA": runner.dataset_labels(test_dataset)},
        diagnostics_by_ticker={
            "AAA_label": _label_diag(n_total=20, n_up=8, n_down=8),
            "AAA_train": _split_diag(n_rows=8, n_retained=8, n_nan=0),
            "AAA_val": _split_diag(n_rows=4, n_retained=4, n_nan=0),
            "AAA_test": _split_diag(n_rows=4, n_retained=4, n_nan=0),
        },
    )


def _candidate():
    return runner.CandidateSpec(
        "A",
        window_size=2,
        label_horizon_k=2,
        label_mode="legacy_binary",
        threshold_bps=0.0,
    )


def _sklearn_metadata():
    return {
        "run_id": "sklearn_toy",
        "git_commit_hash": "abc123",
        "data_source": "synthetic",
        "feature_set_id": "technical_v1",
        "feature_columns": list(runner.TECHNICAL_FEATURES),
        "label_mode": "legacy_binary",
        **runner.protocol_metadata_fields("technical_v1", "legacy_binary", 0.0, None),
        "label_semantics": "canonical_phase1_full_binary",
        "label_formula": "label = 1 if future_avg_r > 0 else 0",
        "class_0_name": "non_up",
        "class_1_name": "up",
        "zero_return_policy": "class_0_non_up",
        "no_trade_band_enabled": False,
        "neutral_policy": "not_applicable",
        "timestamp_col": "timestamp",
        "price_col": "close",
        "shuffle_train_labels": False,
        "shuffle_seed": 42,
        "checkpoint_policy": "best_val_macro_f1",
        "training_scope": "pooled",
        "baseline_scope": "pooled_train",
        "primary_baseline_scope": "pooled_train",
        "dummy_stratified_random_states": list(range(10)),
        "max_rows_per_ticker": 1000,
        "effective_max_rows_per_ticker": 1000,
        "claim_scope": "smoke_observation_not_performance_claim",
        "diagnostic_scope": "bounded_smoke_pipeline_diagnostic",
        "diagnostic_only": True,
        "non_claim": True,
        "tickers": ["AAA"],
        "model_family": "sklearn_logreg",
        "feature_view": "last_step",
    }


def _stationary_raw_frame(days, bars_per_day):
    rows = []
    for day_index, day in enumerate(pd.bdate_range("2024-01-02", periods=days)):
        for bar_index in range(bars_per_day):
            timestamp = day + pd.Timedelta(hours=9, minutes=30 + 5 * bar_index)
            close = 100.0 + 50.0 * day_index + float(bar_index)
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": close - 0.25,
                    "high": close + 0.50,
                    "low": close - 0.50,
                    "close": close,
                    "volume": 1_000 + 10 * day_index + bar_index,
                }
            )
    return pd.DataFrame(rows)


def _calendar_wave_frame(days, bars_per_day):
    rows = []
    for day_index, day in enumerate(pd.bdate_range("2024-01-02", periods=days)):
        for bar_index in range(bars_per_day):
            timestamp = day + pd.Timedelta(hours=9, minutes=30 + 5 * bar_index)
            close = 100.0 + 4.0 * np.sin(bar_index / 4.0) + 0.2 * day_index
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": close - 0.10,
                    "high": close + 0.75,
                    "low": close - 0.75,
                    "close": close,
                    "volume": 1_000 + 100 * day_index + bar_index,
                }
            )
    return pd.DataFrame(rows)


def _technical_raw_frame(rows):
    index = np.arange(rows, dtype=float)
    close = 100.0 + np.sin(index / 2.0) + index * 0.01
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=rows, freq="5min"),
            "open": close - 0.10,
            "high": close + 0.75,
            "low": close - 0.75,
            "close": close,
            "volume": 1_000.0 + index,
        }
    )
