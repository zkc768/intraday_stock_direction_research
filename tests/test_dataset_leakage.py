import numpy as np
import pandas as pd
import pytest


FEATURE_COLS = ["close", "volume"]


def _add_ticker(frame, ticker):
    result = frame.copy(deep=True)
    result["ticker"] = ticker
    return result


def test_make_time_splits_sorts_and_preserves_non_overlapping_rows(raw_multi_ticker_dict):
    from ml_utils.dataset import make_time_splits

    for ticker, frame in raw_multi_ticker_dict.items():
        shuffled = frame.iloc[[3, 0, 2, 1, 5, 4, 7, 6, 9, 8, 11, 10]].copy()

        train, val, test = make_time_splits(
            shuffled,
            train_ratio=0.5,
            val_ratio=0.25,
            timestamp_col="timestamp",
        )

        assert train["timestamp"].is_monotonic_increasing, ticker
        assert val["timestamp"].is_monotonic_increasing, ticker
        assert test["timestamp"].is_monotonic_increasing, ticker
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

        split_timestamps = [
            set(train["timestamp"]),
            set(val["timestamp"]),
            set(test["timestamp"]),
        ]
        assert split_timestamps[0].isdisjoint(split_timestamps[1])
        assert split_timestamps[0].isdisjoint(split_timestamps[2])
        assert split_timestamps[1].isdisjoint(split_timestamps[2])
        assert len(train) + len(val) + len(test) == len(frame)
        assert set().union(*split_timestamps) == set(frame["timestamp"])


def test_fit_scaler_on_train_uses_train_statistics_only():
    from ml_utils.dataset import fit_scaler_on_train

    train = pd.DataFrame({"close": [10.0, 20.0, 30.0], "volume": [100.0, 200.0, 300.0]})
    val = pd.DataFrame({"close": [1000.0, 1100.0], "volume": [10000.0, 11000.0]})
    test = pd.DataFrame({"close": [2000.0, 2100.0], "volume": [20000.0, 21000.0]})
    full = pd.concat([train, val, test], ignore_index=True)

    scaler = fit_scaler_on_train(train, feature_cols=FEATURE_COLS, scaler_type="standard")

    np.testing.assert_allclose(scaler.mean_, train[FEATURE_COLS].mean().to_numpy(), rtol=1e-12)
    assert not np.allclose(scaler.mean_, full[FEATURE_COLS].mean().to_numpy())


def test_transform_split_returns_new_dataframe_and_preserves_input_values():
    from ml_utils.dataset import fit_scaler_on_train, transform_split

    train = pd.DataFrame({"close": [10.0, 20.0, 30.0], "volume": [100.0, 200.0, 300.0]})
    split = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 09:30", periods=3, freq="5min"),
            "close": [40.0, 50.0, 60.0],
            "volume": [400.0, 500.0, 600.0],
            "label": [1.0, 0.0, np.nan],
        }
    )
    original = split.copy(deep=True)
    scaler = fit_scaler_on_train(train, feature_cols=FEATURE_COLS, scaler_type="standard")

    transformed = transform_split(split, scaler=scaler, feature_cols=FEATURE_COLS)

    assert transformed is not split
    pd.testing.assert_frame_equal(split, original)
    assert transformed["timestamp"].equals(split["timestamp"])
    assert transformed["label"].equals(split["label"])
    assert not np.allclose(transformed[FEATURE_COLS].to_numpy(), split[FEATURE_COLS].to_numpy())


@pytest.mark.parametrize("scaler_type", ["standard", "minmax"])
def test_fit_scaler_on_train_supports_declared_scaler_types(scaler_type):
    from ml_utils.dataset import fit_scaler_on_train

    train = pd.DataFrame({"close": [10.0, 20.0, 30.0], "volume": [100.0, 200.0, 300.0]})

    scaler = fit_scaler_on_train(train, feature_cols=FEATURE_COLS, scaler_type=scaler_type)

    assert hasattr(scaler, "transform")
    assert hasattr(scaler, "fit")


def test_invalid_scaler_type_raises_value_error():
    from ml_utils.dataset import fit_scaler_on_train

    train = pd.DataFrame({"close": [10.0, 20.0], "volume": [100.0, 200.0]})

    with pytest.raises(ValueError):
        fit_scaler_on_train(train, feature_cols=FEATURE_COLS, scaler_type="robust")


def test_windowed_dataset_does_not_create_cross_ticker_windows(raw_multi_ticker_dict):
    from ml_utils.dataset import WindowedClassificationDataset

    frames = []
    for ticker, frame in raw_multi_ticker_dict.items():
        labeled = _add_ticker(frame, ticker)
        labeled["label"] = [1.0, 0.0, 1.0, 0.0, np.nan, np.nan, 1.0, 0.0, 1.0, 0.0, np.nan, np.nan]
        frames.append(labeled)
    merged = pd.concat(frames, ignore_index=True)

    dataset = WindowedClassificationDataset(
        merged,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=3,
        label_horizon_k=2,
        stride=1,
    )

    assert len(dataset) > 0
    for ticker, local_start_idx in dataset.valid_starts:
        ticker_frame = merged[merged["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        window_tickers = ticker_frame.loc[local_start_idx : local_start_idx + 2, "ticker"]
        assert window_tickers.eq(ticker).all()
