import numpy as np
import pandas as pd


FEATURE_COLS = ["close"]


def _windowed_dataset(*args, **kwargs):
    from runner_utils.dataset import WindowedClassificationDataset

    return WindowedClassificationDataset(*args, **kwargs)


def _single_ticker_frame(closes, labels, timestamps=None):
    if timestamps is None:
        timestamps = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="5min")
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * len(closes),
            "timestamp": pd.to_datetime(timestamps),
            "close": [float(close) for close in closes],
            "label": [float(label) if not pd.isna(label) else np.nan for label in labels],
        }
    )


def _make_dataset(frame, window_size=3, label_horizon_k=2):
    return _windowed_dataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=window_size,
        label_horizon_k=label_horizon_k,
        stride=1,
    )


def test_getitem_returns_label_from_window_end_not_window_start():
    frame = _single_ticker_frame(
        closes=[10.0, 11.0, 12.0, 13.0, 14.0],
        labels=[0.0, 0.0, 1.0, 0.0, 0.0],
    )

    dataset = _make_dataset(frame, window_size=3, label_horizon_k=2)
    x, y = dataset[0]

    np.testing.assert_allclose(x.squeeze(-1).numpy(), [10.0, 11.0, 12.0])
    assert y.item() == 1
    assert y.item() != int(frame.loc[0, "label"])


def test_start_label_nan_does_not_reject_window_when_end_label_is_valid():
    frame = _single_ticker_frame(
        closes=[10.0, 11.0, 12.0, 13.0, 14.0],
        labels=[np.nan, 0.0, 1.0, 0.0, 0.0],
    )

    dataset = _make_dataset(frame, window_size=3, label_horizon_k=2)

    assert dataset.valid_starts == [("AAA", 0)]
    _, y = dataset[0]
    assert y.item() == 1


def test_end_label_nan_rejects_window_even_when_start_label_is_valid():
    frame = _single_ticker_frame(
        closes=[10.0, 11.0, 12.0, 13.0, 14.0],
        labels=[1.0, 0.0, np.nan, 0.0, 0.0],
    )

    dataset = _make_dataset(frame, window_size=3, label_horizon_k=2)

    assert dataset.valid_starts == []
    assert len(dataset) == 0


def test_horizon_capacity_is_checked_from_window_end():
    frame = _single_ticker_frame(
        closes=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        labels=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    )

    dataset = _make_dataset(frame, window_size=3, label_horizon_k=2)

    assert dataset.valid_starts == [("AAA", 0), ("AAA", 1)]
    assert ("AAA", 2) not in dataset.valid_starts


def test_horizon_same_day_check_starts_from_window_end():
    frame = _single_ticker_frame(
        closes=[10.0, 11.0, 12.0, 20.0],
        labels=[1.0, 0.0, 1.0, 0.0],
        timestamps=[
            "2024-01-02 09:30",
            "2024-01-02 09:35",
            "2024-01-02 09:40",
            "2024-01-03 09:30",
        ],
    )

    dataset = _make_dataset(frame, window_size=3, label_horizon_k=1)

    assert dataset.valid_starts == []
    assert len(dataset) == 0
