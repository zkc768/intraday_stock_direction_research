import numpy as np
import pandas as pd
import pytest


FEATURE_COLS = ["close", "volume"]


def _with_ticker(frame, ticker="AAA"):
    result = frame.copy(deep=True)
    result["ticker"] = ticker
    return result


def _two_day_labeled_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * 8,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30",
                    "2024-01-02 09:35",
                    "2024-01-02 09:40",
                    "2024-01-02 09:45",
                    "2024-01-03 09:30",
                    "2024-01-03 09:35",
                    "2024-01-03 09:40",
                    "2024-01-03 09:45",
                ]
            ),
            "close": [10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
            "volume": [100.0, 110.0, 120.0, 130.0, 200.0, 210.0, 220.0, 230.0],
            "label": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, np.nan, np.nan],
        }
    )


def _two_ticker_two_day_labeled_frame():
    rows = []
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30",
            "2024-01-02 09:35",
            "2024-01-02 09:40",
            "2024-01-02 09:45",
            "2024-01-03 09:30",
            "2024-01-03 09:35",
            "2024-01-03 09:40",
            "2024-01-03 09:45",
        ]
    )
    for ticker, price_offset, label in [("AAA", 0.0, 1.0), ("BBB", 100.0, 0.0)]:
        for local_idx, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "close": 10.0 + price_offset + local_idx,
                    "volume": 100.0 + price_offset + local_idx,
                    "label": label,
                }
            )
    return pd.DataFrame(rows).sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def test_trim_labels_at_split_boundary_marks_tail_invalid_without_deleting_rows(labeled_df_with_tail_nan):
    from ml_utils.dataset import trim_labels_at_split_boundary

    original = labeled_df_with_tail_nan.copy(deep=True)

    trimmed = trim_labels_at_split_boundary(labeled_df_with_tail_nan, label_horizon_k=2)

    assert len(trimmed) == len(original)
    assert trimmed["label"].tail(2).isna().all()
    assert trimmed.drop(columns=["label"]).equals(original.drop(columns=["label"]))
    pd.testing.assert_frame_equal(labeled_df_with_tail_nan, original)


def test_trim_labels_at_split_boundary_groups_interleaved_tickers_without_deleting_rows():
    from ml_utils.dataset import trim_labels_at_split_boundary

    timestamps = pd.date_range("2024-01-02 09:30", periods=5, freq="5min")
    rows = []
    for local_idx, timestamp in enumerate(timestamps):
        for ticker, price_offset, label in [("AAA", 0.0, 1.0), ("BBB", 100.0, 0.0)]:
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "close": 10.0 + price_offset + local_idx,
                    "volume": 100.0 + price_offset + local_idx,
                    "label": label,
                }
            )
    frame = pd.DataFrame(rows)
    original = frame.copy(deep=True)

    trimmed = trim_labels_at_split_boundary(
        frame,
        label_horizon_k=2,
        ticker_col="ticker",
        timestamp_col="timestamp",
    )

    assert len(trimmed) == len(original)
    assert trimmed.drop(columns=["label"]).equals(original.drop(columns=["label"]))
    pd.testing.assert_frame_equal(frame, original)
    for ticker in ["AAA", "BBB"]:
        ordered = trimmed[trimmed["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        assert ordered["label"].isna().tolist() == [False, False, False, True, True]


def test_trim_labels_at_split_boundary_rejects_out_of_order_timestamp_within_ticker():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_ticker_two_day_labeled_frame()
    frame.loc[3, "timestamp"] = pd.Timestamp("2024-01-02 09:25")

    with pytest.raises(ValueError) as exc_info:
        trim_labels_at_split_boundary(
            frame,
            label_horizon_k=2,
            ticker_col="ticker",
            timestamp_col="timestamp",
        )
    message = str(exc_info.value)
    assert "trim_labels_at_split_boundary" in message
    assert "BBB" in message
    assert "timestamp" in message
    assert "row/index 3" in message
    assert "position 1" in message
    assert "current timestamp" in message
    assert "previous timestamp" in message
    assert "2024-01-02 09:25" in message
    assert "2024-01-02 09:30" in message


def test_trim_labels_at_split_boundary_rejects_duplicate_timestamp_within_ticker():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_ticker_two_day_labeled_frame()
    frame.loc[3, "timestamp"] = frame.loc[1, "timestamp"]

    with pytest.raises(ValueError) as exc_info:
        trim_labels_at_split_boundary(
            frame,
            label_horizon_k=2,
            ticker_col="ticker",
            timestamp_col="timestamp",
        )
    message = str(exc_info.value)
    assert "trim_labels_at_split_boundary" in message
    assert "BBB" in message
    assert "timestamp" in message
    assert "duplicates" in message
    assert "rows" in message
    assert "1" in message
    assert "3" in message


def test_trim_labels_at_split_boundary_rejects_invalid_binary_label_with_ticker_context():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_ticker_two_day_labeled_frame()
    frame.loc[5, "label"] = 2.0

    with pytest.raises(ValueError) as exc_info:
        trim_labels_at_split_boundary(
            frame,
            label_horizon_k=2,
            ticker_col="ticker",
            timestamp_col="timestamp",
        )
    message = str(exc_info.value)
    assert "trim_labels_at_split_boundary" in message
    assert "BBB" in message
    assert "label" in message
    assert "row/index 5" in message
    assert "2.0" in message


def test_trim_labels_marks_cross_trading_day_horizon_invalid_without_deleting_rows():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_day_labeled_frame()

    trimmed = trim_labels_at_split_boundary(frame, label_horizon_k=2)

    assert len(trimmed) == len(frame)
    assert pd.isna(trimmed.loc[2, "label"])
    assert pd.isna(trimmed.loc[3, "label"])
    assert not trimmed["label"].isna().equals(frame["label"].isna())


def test_no_label_horizon_crosses_trading_day():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_ticker_two_day_labeled_frame()
    original = frame.copy(deep=True)

    trimmed = trim_labels_at_split_boundary(
        frame,
        label_horizon_k=2,
        ticker_col="ticker",
        timestamp_col="timestamp",
    )

    assert len(trimmed) == len(original)
    pd.testing.assert_frame_equal(frame, original)
    for ticker in ["AAA", "BBB"]:
        ordered = trimmed[trimmed["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        nan_starts = set(np.flatnonzero(ordered["label"].isna().to_numpy()))
        assert nan_starts == {2, 3, 6, 7}


def test_windowed_dataset_does_not_create_input_windows_across_trading_days():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_day_labeled_frame()

    dataset = WindowedClassificationDataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=3,
        label_horizon_k=1,
        stride=1,
    )

    for ticker, local_start_idx in dataset.valid_starts:
        ticker_frame = frame[frame["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        dates = ticker_frame.loc[local_start_idx : local_start_idx + 2, "timestamp"].dt.date
        assert dates.nunique() == 1


def test_no_window_crosses_trading_day():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_ticker_two_day_labeled_frame()

    dataset = WindowedClassificationDataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=3,
        label_horizon_k=1,
        stride=1,
    )

    assert set(dataset.valid_starts) == {("AAA", 0), ("AAA", 4), ("BBB", 0), ("BBB", 4)}


def test_windowed_dataset_validates_label_at_window_end(split_df_after_trim):
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _with_ticker(split_df_after_trim)
    window_size = 2
    label_horizon_k = 1

    dataset = WindowedClassificationDataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=window_size,
        label_horizon_k=label_horizon_k,
        stride=1,
    )

    assert len(dataset) > 0
    saw_valid_window_with_nan_start_label = False
    for ticker, local_start_idx in dataset.valid_starts:
        ticker_frame = frame[frame["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        target_idx = local_start_idx + window_size - 1
        assert not pd.isna(ticker_frame.loc[target_idx, "label"])
        if pd.isna(ticker_frame.loc[local_start_idx, "label"]):
            saw_valid_window_with_nan_start_label = True
    assert saw_valid_window_with_nan_start_label

    valid_starts = set(dataset.valid_starts)
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        ordered = ticker_frame.sort_values("timestamp").reset_index(drop=True)
        max_start = len(ordered) - window_size - label_horizon_k
        for local_start_idx in range(max_start + 1):
            target_idx = local_start_idx + window_size - 1
            if pd.isna(ordered.loc[target_idx, "label"]):
                assert (ticker, local_start_idx) not in valid_starts
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        if hasattr(y, "detach"):
            assert not bool(y.detach().cpu().isnan().item())
        else:
            assert not pd.isna(y)


def test_windowed_dataset_returns_expected_tensor_shapes_and_label_dtype(split_df_after_trim):
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _with_ticker(split_df_after_trim)

    dataset = WindowedClassificationDataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=2,
        label_horizon_k=1,
        stride=1,
    )

    x, y = dataset[0]

    assert tuple(x.shape) == (2, len(FEATURE_COLS))
    assert tuple(y.shape) == ()
    assert y.item() in (0, 1)
    if hasattr(x, "dtype"):
        import torch

        assert x.dtype == torch.float32
        assert y.dtype == torch.long


def test_windowed_dataset_rejects_invalid_binary_label():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_ticker_two_day_labeled_frame()
    frame.loc[5, "label"] = 2.0

    with pytest.raises(ValueError) as exc_info:
        WindowedClassificationDataset(
            frame,
            feature_cols=FEATURE_COLS,
            label_col="label",
            ticker_col="ticker",
            timestamp_col="timestamp",
            window_size=2,
            label_horizon_k=1,
            stride=1,
        )
    message = str(exc_info.value)
    assert "WindowedClassificationDataset" in message
    assert "BBB" in message
    assert "label" in message
    assert "row/index 5" in message
    assert "2.0" in message


def test_windowed_dataset_rejects_duplicate_timestamp_within_ticker():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_day_labeled_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    with pytest.raises(ValueError) as exc_info:
        WindowedClassificationDataset(
            frame,
            feature_cols=FEATURE_COLS,
            label_col="label",
            ticker_col="ticker",
            timestamp_col="timestamp",
            window_size=2,
            label_horizon_k=1,
            stride=1,
        )
    message = str(exc_info.value)
    assert "AAA" in message
    assert "timestamp" in message
    assert "rows" in message
    assert "0" in message
    assert "1" in message


def test_windowed_dataset_rejects_out_of_order_timestamp_within_ticker():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_ticker_two_day_labeled_frame()
    frame.loc[3, "timestamp"] = pd.Timestamp("2024-01-02 09:25")

    with pytest.raises(ValueError) as exc_info:
        WindowedClassificationDataset(
            frame,
            feature_cols=FEATURE_COLS,
            label_col="label",
            ticker_col="ticker",
            timestamp_col="timestamp",
            window_size=2,
            label_horizon_k=1,
            stride=1,
        )
    message = str(exc_info.value)
    assert "BBB" in message
    assert "timestamp" in message
    assert "row/index 3" in message
    assert "position 1" in message
    assert "current timestamp" in message
    assert "previous timestamp" in message
    assert "2024-01-02 09:25" in message
    assert "2024-01-02 09:30" in message


def test_windowed_dataset_returns_features_from_correct_interleaved_ticker_group():
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _two_ticker_two_day_labeled_frame()

    dataset = WindowedClassificationDataset(
        frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=3,
        label_horizon_k=1,
        stride=1,
    )

    item_idx = dataset.valid_starts.index(("BBB", 0))
    x, y = dataset[item_idx]

    np.testing.assert_allclose(
        x.numpy(),
        np.asarray(
            [
                [110.0, 200.0],
                [111.0, 201.0],
                [112.0, 202.0],
            ],
            dtype=np.float32,
        ),
    )
    assert y.item() == 0


def test_stride_controls_window_count(split_df_after_trim):
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _with_ticker(split_df_after_trim)
    kwargs = dict(
        df=frame,
        feature_cols=FEATURE_COLS,
        label_col="label",
        ticker_col="ticker",
        timestamp_col="timestamp",
        window_size=2,
        label_horizon_k=1,
    )

    stride_one = WindowedClassificationDataset(stride=1, **kwargs)
    stride_two = WindowedClassificationDataset(stride=2, **kwargs)

    assert len(stride_one) >= len(stride_two)
    assert len(stride_two) == (len(stride_one) + 1) // 2


@pytest.mark.parametrize("bad_stride", [0, -1])
def test_stride_must_be_positive(split_df_after_trim, bad_stride):
    from ml_utils.dataset import WindowedClassificationDataset

    frame = _with_ticker(split_df_after_trim)

    with pytest.raises(ValueError):
        WindowedClassificationDataset(
            frame,
            feature_cols=FEATURE_COLS,
            label_col="label",
            ticker_col="ticker",
            timestamp_col="timestamp",
            window_size=2,
            label_horizon_k=1,
            stride=bad_stride,
        )
