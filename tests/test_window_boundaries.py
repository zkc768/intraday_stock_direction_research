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


def test_trim_labels_at_split_boundary_marks_tail_invalid_without_deleting_rows(labeled_df_with_tail_nan):
    from ml_utils.dataset import trim_labels_at_split_boundary

    original = labeled_df_with_tail_nan.copy(deep=True)

    trimmed = trim_labels_at_split_boundary(labeled_df_with_tail_nan, label_horizon_k=2)

    assert len(trimmed) == len(original)
    assert trimmed["label"].tail(2).isna().all()
    assert trimmed.drop(columns=["label"]).equals(original.drop(columns=["label"]))
    pd.testing.assert_frame_equal(labeled_df_with_tail_nan, original)


def test_trim_labels_marks_cross_trading_day_horizon_invalid_without_deleting_rows():
    from ml_utils.dataset import trim_labels_at_split_boundary

    frame = _two_day_labeled_frame()

    trimmed = trim_labels_at_split_boundary(frame, label_horizon_k=2)

    assert len(trimmed) == len(frame)
    assert pd.isna(trimmed.loc[2, "label"])
    assert pd.isna(trimmed.loc[3, "label"])
    assert not trimmed["label"].isna().equals(frame["label"].isna())


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


def test_windowed_dataset_skips_nan_label_starts(split_df_after_trim):
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

    assert len(dataset) > 0
    for ticker, local_start_idx in dataset.valid_starts:
        ticker_frame = frame[frame["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        assert not pd.isna(ticker_frame.loc[local_start_idx, "label"])
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
