import numpy as np
import pandas as pd
import pytest


DIAGNOSTIC_KEYS = {
    "n_total",
    "n_tail",
    "n_cross_day",
    "n_neutral",
    "n_up",
    "n_down",
}


def _make_no_trade_band_labels(*args, **kwargs):
    from runner_utils.dataset import make_no_trade_band_labels

    return make_no_trade_band_labels(*args, **kwargs)


def _price_df(closes, timestamps=None):
    data = {"close": [float(close) for close in closes]}
    if timestamps is not None:
        data["timestamp"] = pd.to_datetime(timestamps)
    return pd.DataFrame(data)


def _assert_diagnostics_sum(diagnostics):
    counted = (
        diagnostics["n_tail"]
        + diagnostics["n_cross_day"]
        + diagnostics["n_neutral"]
        + diagnostics["n_up"]
        + diagnostics["n_down"]
    )
    assert counted == diagnostics["n_total"]


def _cross_day_price_df():
    return _price_df(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        timestamps=[
            "2024-01-02 15:50",
            "2024-01-02 15:55",
            "2024-01-03 09:30",
            "2024-01-03 09:35",
            "2024-01-03 09:40",
        ],
    )


def test_neutral_band_marks_inside_threshold_as_nan():
    df = _price_df([100.0, 100.05, 100.20])

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=10.0,
    )

    expected_return = (100.05 - 100.0) / 100.0
    assert abs(expected_return) < 10.0 / 10_000
    assert pd.isna(out_df.loc[0, "label"])
    assert len(out_df) == len(df)
    assert diagnostics["n_neutral"] == 1
    _assert_diagnostics_sum(diagnostics)


def test_exact_positive_and_negative_threshold_are_neutral():
    positive_df = _price_df([100.0, 100.1])
    negative_df = _price_df([100.0, 99.9])

    positive_out, positive_diagnostics = _make_no_trade_band_labels(
        positive_df,
        price_col="close",
        k=1,
        threshold_bps=10.0,
    )
    negative_out, negative_diagnostics = _make_no_trade_band_labels(
        negative_df,
        price_col="close",
        k=1,
        threshold_bps=10.0,
    )

    threshold = 10.0 / 10_000
    assert np.isclose(positive_out.loc[0, "future_avg_r"], threshold)
    assert np.isclose(negative_out.loc[0, "future_avg_r"], -threshold)
    assert pd.isna(positive_out.loc[0, "label"])
    assert pd.isna(negative_out.loc[0, "label"])
    assert positive_diagnostics["n_neutral"] == 1
    assert negative_diagnostics["n_neutral"] == 1


def test_values_just_outside_threshold_map_to_up_and_down():
    first_next = 100.0 * (1.0 + 0.0011)
    second_next = first_next * (1.0 - 0.0011)
    df = _price_df([100.0, first_next, second_next])

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=10.0,
    )

    threshold = 10.0 / 10_000
    assert out_df.loc[0, "future_avg_r"] > threshold
    assert out_df.loc[0, "label"] == 1.0
    assert out_df.loc[1, "future_avg_r"] < -threshold
    assert out_df.loc[1, "label"] == 0.0
    assert diagnostics["n_up"] == 1
    assert diagnostics["n_down"] == 1
    _assert_diagnostics_sum(diagnostics)


def test_tail_rows_are_nan_and_counted_in_diagnostics():
    k = 3
    df = _price_df([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=k,
        threshold_bps=0.0,
    )

    assert out_df["label"].tail(k).isna().all()
    assert out_df["future_avg_r"].tail(k).isna().all()
    assert diagnostics["n_tail"] == k
    assert len(out_df) == len(df)
    _assert_diagnostics_sum(diagnostics)


def test_cross_day_horizon_is_nan_when_timestamp_col_is_provided():
    df = _cross_day_price_df()

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=2,
        threshold_bps=0.0,
        timestamp_col="timestamp",
    )

    assert out_df.loc[[0, 1], "label"].isna().all()
    assert out_df.loc[2, "label"] == 1.0
    assert out_df["label"].tail(2).isna().all()
    assert diagnostics["n_cross_day"] == 2
    assert diagnostics["n_tail"] == 2
    assert len(out_df) == len(df)
    _assert_diagnostics_sum(diagnostics)


def test_no_cross_day_filter_when_timestamp_col_is_none():
    df = _cross_day_price_df()

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=2,
        threshold_bps=0.0,
    )

    assert out_df.loc[[0, 1], "label"].notna().all()
    assert out_df.loc[[0, 1], "label"].eq(1.0).all()
    assert diagnostics["n_cross_day"] == 0
    assert len(out_df) == len(df)
    _assert_diagnostics_sum(diagnostics)


def test_threshold_zero_matches_legacy_on_nonzero_same_day_returns():
    from runner_utils.dataset import make_binary_labels_from_future_avg_return

    df = _price_df(
        [100.0, 101.0, 100.0, 102.0, 101.0, 103.0],
        timestamps=pd.date_range("2024-01-02 09:30", periods=6, freq="5min"),
    )

    legacy_df = make_binary_labels_from_future_avg_return(df, price_col="close", k=1)
    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=0.0,
        timestamp_col="timestamp",
    )

    comparable = legacy_df["future_avg_r"].notna()
    assert not np.isclose(legacy_df.loc[comparable, "future_avg_r"], 0.0).any()
    np.testing.assert_allclose(
        out_df.loc[comparable, "future_avg_r"].to_numpy(),
        legacy_df.loc[comparable, "future_avg_r"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_series_equal(
        out_df.loc[comparable, "label"],
        legacy_df.loc[comparable, "label"],
        check_names=False,
    )
    assert diagnostics["n_cross_day"] == 0


def test_threshold_zero_exact_zero_behavior_is_documented():
    from runner_utils.dataset import make_binary_labels_from_future_avg_return

    df = _price_df([100.0, 100.0, 101.0])

    legacy_df = make_binary_labels_from_future_avg_return(df, price_col="close", k=1)
    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=0.0,
    )

    assert legacy_df.loc[0, "future_avg_r"] == 0.0
    assert legacy_df.loc[0, "label"] == 0.0
    assert out_df.loc[0, "future_avg_r"] == 0.0
    assert pd.isna(out_df.loc[0, "label"])
    assert diagnostics["n_neutral"] == 1


def test_negative_threshold_bps_raises_value_error():
    df = _price_df([100.0, 101.0, 102.0])

    with pytest.raises(ValueError):
        _make_no_trade_band_labels(
            df,
            price_col="close",
            k=1,
            threshold_bps=-1.0,
        )


def test_output_preserves_row_count_order_and_input_is_not_mutated():
    df = _price_df(
        [100.0, 101.0, 100.5, 102.0],
        timestamps=pd.date_range("2024-01-02 09:30", periods=4, freq="5min"),
    )
    original = df.copy(deep=True)

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=5.0,
        timestamp_col="timestamp",
    )

    assert len(out_df) == len(df)
    pd.testing.assert_series_equal(out_df["timestamp"], original["timestamp"])
    pd.testing.assert_frame_equal(df, original)
    assert "future_avg_r" in out_df.columns
    assert "label" in out_df.columns
    # The legacy route plan used the draft name "future_avg_return", but current runner_utils convention is "future_avg_r".
    assert "future_avg_return" not in out_df.columns
    assert "future_avg_r" not in df.columns
    assert "label" not in df.columns
    assert diagnostics["n_total"] == len(out_df)


def test_diagnostics_keys_and_sum_are_consistent():
    df = _price_df([100.0, 100.2, 100.25, 100.0])

    out_df, diagnostics = _make_no_trade_band_labels(
        df,
        price_col="close",
        k=1,
        threshold_bps=10.0,
    )

    assert set(diagnostics) == DIAGNOSTIC_KEYS
    assert all(isinstance(value, int) for value in diagnostics.values())
    assert diagnostics["n_total"] == len(out_df)
    assert diagnostics["n_up"] == 1
    assert diagnostics["n_neutral"] == 1
    assert diagnostics["n_down"] == 1
    assert diagnostics["n_tail"] == 1
    assert diagnostics["n_cross_day"] == 0
    _assert_diagnostics_sum(diagnostics)
