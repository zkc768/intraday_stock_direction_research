"""Dataset utilities for time-ordered stock direction classification.

Reference ideas from reference_excerpts/ltsf_data_loader.py:
- fit scalers on the training segment only;
- use explicit chronological split boundaries;
- build windows from local positions inside one ordered series.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


def _check_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


def _require_numeric_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    for column in columns:
        if not is_numeric_dtype(df[column]):
            raise ValueError(f"{context} column {column!r} must be numeric")
        if df[column].isna().any():
            bad_rows = df.index[df[column].isna()].tolist()
            raise ValueError(f"{context} column {column!r} contains NaN at rows {bad_rows}")


def _validate_timezone_policy(timezone_policy: str, context: str) -> None:
    if timezone_policy not in {"naive", "utc"}:
        raise ValueError(f"{context} timezone_policy must be 'naive' or 'utc', got {timezone_policy!r}")


def _validate_timestamp_dtype_and_timezone(
    df: pd.DataFrame,
    timestamp_col: str,
    timezone_policy: str,
    context: str,
) -> None:
    _validate_timezone_policy(timezone_policy, context)
    _require_columns(df, [timestamp_col], context)
    if not is_datetime64_any_dtype(df[timestamp_col]):
        raise ValueError(f"{context} timestamp column {timestamp_col!r} must be datetime dtype")

    timezone = df[timestamp_col].dt.tz
    if timezone_policy == "naive" and timezone is not None:
        raise ValueError(f"{context} timestamp column {timestamp_col!r} must be timezone-naive")
    if timezone_policy == "utc":
        timezone_name = getattr(timezone, "key", None) or getattr(timezone, "zone", None) or str(timezone)
        if timezone is None or timezone_name != "UTC":
            raise ValueError(f"{context} timestamp column {timestamp_col!r} must be UTC timezone-aware")


def _validate_strict_timestamp_order(df: pd.DataFrame, timestamp_col: str, context: str) -> None:
    if df[timestamp_col].duplicated().any():
        duplicate_rows = df.index[df[timestamp_col].duplicated(keep=False)].tolist()
        raise ValueError(f"{context} timestamp column {timestamp_col!r} has duplicates at rows {duplicate_rows}")
    if not df[timestamp_col].is_monotonic_increasing:
        timestamps = df[timestamp_col]
        for position in range(1, len(timestamps)):
            previous_timestamp = timestamps.iloc[position - 1]
            current_timestamp = timestamps.iloc[position]
            if pd.isna(previous_timestamp) or pd.isna(current_timestamp) or current_timestamp <= previous_timestamp:
                previous_row = timestamps.index[position - 1]
                current_row = timestamps.index[position]
                raise ValueError(
                    f"{context} timestamp column {timestamp_col!r} must be strictly increasing; "
                    f"offending row/index {current_row!r} at position {position} has current timestamp "
                    f"{current_timestamp!r} <= previous timestamp {previous_timestamp!r} "
                    f"at row/index {previous_row!r} position {position - 1}"
                )
        raise ValueError(f"{context} timestamp column {timestamp_col!r} must be strictly increasing")


def _validate_binary_label_values(
    df: pd.DataFrame,
    label_col: str,
    context: str,
    ticker_col: str | None = None,
) -> None:
    _require_columns(df, [label_col], context)
    if not is_numeric_dtype(df[label_col]):
        raise ValueError(f"{context} label column {label_col!r} must be numeric")
    non_na_labels = df[label_col][df[label_col].notna()]
    invalid_mask = ~non_na_labels.astype(float).isin([0.0, 1.0])
    if invalid_mask.any():
        bad_index = invalid_mask[invalid_mask].index[0]
        invalid_value = df.loc[bad_index, label_col]
        ticker_message = ""
        if ticker_col is not None:
            ticker_value = df.loc[bad_index, ticker_col]
            ticker_message = f" ticker {ticker_value!r}"
        raise ValueError(
            f"{context}{ticker_message} label column {label_col!r} contains invalid value "
            f"{invalid_value} at offending row/index {bad_index!r}; expected one of {{0, 1, NaN}}"
        )


def _infer_datetime_column(df: pd.DataFrame) -> str | None:
    datetime_columns = [column for column in df.columns if is_datetime64_any_dtype(df[column])]
    if not datetime_columns:
        return None
    if len(datetime_columns) != 1:
        raise ValueError(f"expected one datetime column, found {datetime_columns}")
    return datetime_columns[0]


def make_binary_labels_from_future_avg_return(
    df: pd.DataFrame,
    price_col: str,
    k: int,
) -> pd.DataFrame:
    """Return a new DataFrame with future-average return and binary labels.

    For each row t, label uses k bar-to-bar returns from t to t+k.
    The final k rows receive NaN in both derived columns and are preserved.
    """

    _check_positive_int(k, "k")
    _require_columns(df, [price_col], "make_binary_labels_from_future_avg_return")
    _require_numeric_columns(df, [price_col], "make_binary_labels_from_future_avg_return")
    if (df[price_col] <= 0).any():
        bad_rows = df.index[df[price_col] <= 0].tolist()
        raise ValueError(f"price column {price_col!r} must be positive at rows {bad_rows}")

    result = df.copy(deep=True)
    prices = result[price_col].astype(float)
    next_returns = prices.pct_change().shift(-1)
    future_return_columns = [next_returns.shift(-offset) for offset in range(k)]
    future_avg = pd.concat(future_return_columns, axis=1).mean(axis=1, skipna=False)

    labels = pd.Series(np.nan, index=result.index, name="label", dtype="float64")
    valid_mask = future_avg.notna()
    labels.loc[valid_mask] = np.where(future_avg.loc[valid_mask] > 0.0, 1.0, 0.0)

    result["future_avg_r"] = future_avg
    result["label"] = labels
    return result


def make_no_trade_band_labels(
    df: pd.DataFrame,
    price_col: str,
    k: int,
    threshold_bps: float,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return no-trade-band labels from the existing future-average return.

    The labeled subset estimates P(sign(r) | X, |r| > threshold).
    """

    _check_positive_int(k, "k")
    if threshold_bps < 0:
        raise ValueError(f"threshold_bps must be >= 0, got {threshold_bps}")
    _require_columns(df, [price_col], "make_no_trade_band_labels")
    _require_numeric_columns(df, [price_col], "make_no_trade_band_labels")
    if (df[price_col] <= 0).any():
        bad_rows = df.index[df[price_col] <= 0].tolist()
        raise ValueError(f"price column {price_col!r} must be positive at rows {bad_rows}")
    if timestamp_col is not None:
        _require_columns(df, [timestamp_col], "make_no_trade_band_labels")
        if not is_datetime64_any_dtype(df[timestamp_col]):
            raise ValueError(f"timestamp column {timestamp_col!r} must be datetime dtype")
        _validate_strict_timestamp_order(df, timestamp_col, "make_no_trade_band_labels")

    result = df.copy(deep=True)
    prices = result[price_col].astype(float)
    next_returns = prices.pct_change().shift(-1)
    future_return_columns = [next_returns.shift(-offset) for offset in range(k)]
    future_avg = pd.concat(future_return_columns, axis=1).mean(axis=1, skipna=False)

    threshold = threshold_bps / 10_000
    labels = pd.Series(np.nan, index=result.index, name="label", dtype="float64")
    valid_mask = future_avg.notna()
    up_mask = valid_mask & (future_avg > threshold)
    down_mask = valid_mask & (future_avg < -threshold)
    labels.loc[up_mask] = 1.0
    labels.loc[down_mask] = 0.0

    cross_day_mask = pd.Series(False, index=result.index, dtype=bool)
    if timestamp_col is not None:
        dates = result[timestamp_col].dt.date
        horizon_dates = dates.shift(-k)
        cross_day_mask = valid_mask & horizon_dates.notna() & (dates != horizon_dates)
        labels.loc[cross_day_mask] = np.nan

    result["future_avg_r"] = future_avg
    result["label"] = labels
    diagnostics = {
        "n_total": int(len(result)),
        "n_tail": int(future_avg.isna().sum()),
        "n_cross_day": int(cross_day_mask.sum()),
        "n_neutral": int((valid_mask & ~cross_day_mask & labels.isna()).sum()),
        "n_up": int((labels == 1.0).sum()),
        "n_down": int((labels == 0.0).sum()),
    }
    return result, diagnostics


def make_time_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    timestamp_col: str,
    timezone_policy: str = "naive",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one ticker DataFrame into chronological train, validation, and test frames."""

    _validate_timestamp_dtype_and_timezone(df, timestamp_col, timezone_policy, "make_time_splits")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    _validate_strict_timestamp_order(ordered, timestamp_col, "make_time_splits")
    row_count = len(ordered)
    train_end = int(row_count * train_ratio)
    val_end = train_end + int(row_count * val_ratio)

    train = ordered.iloc[:train_end].copy(deep=True)
    val = ordered.iloc[train_end:val_end].copy(deep=True)
    test = ordered.iloc[val_end:].copy(deep=True)
    return train, val, test


def fit_scaler_on_train(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    scaler_type: str = "standard",
) -> Any:
    """Fit a feature scaler using only the provided training frame."""

    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    _require_columns(train_df, feature_cols, "fit_scaler_on_train")
    _require_numeric_columns(train_df, feature_cols, "fit_scaler_on_train")

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"scaler_type must be 'standard' or 'minmax', got {scaler_type!r}")
    return scaler.fit(train_df[feature_cols].to_numpy(dtype=float))


def transform_split(
    df: pd.DataFrame,
    scaler: Any,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return a new frame with feature columns transformed by a fitted scaler."""

    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    _require_columns(df, feature_cols, "transform_split")
    _require_numeric_columns(df, feature_cols, "transform_split")

    result = df.copy(deep=True)
    transformed_features = pd.DataFrame(
        scaler.transform(result[feature_cols].to_numpy(dtype=float)),
        columns=feature_cols,
        index=result.index,
    )
    result[feature_cols] = transformed_features
    return result


def _trim_groups(
    df: pd.DataFrame,
    ticker_col: str | None,
    context: str,
) -> list[tuple[Any | None, pd.DataFrame]]:
    if ticker_col is None:
        return [(None, df)]
    _require_columns(df, [ticker_col], context)
    return [(ticker, group) for ticker, group in df.groupby(ticker_col, sort=False)]


def _ordered_trim_group(
    group: pd.DataFrame,
    timestamp_col: str | None,
    context: str,
) -> pd.DataFrame:
    if timestamp_col is None:
        return group
    _validate_strict_timestamp_order(group, timestamp_col, context)
    return group


def _mark_cross_day_horizon_labels(
    result: pd.DataFrame,
    positions: list[Any],
    dates: pd.Series,
    label_horizon_k: int,
    label_col: str,
) -> None:
    for local_idx in range(max(len(positions) - label_horizon_k, 0)):
        horizon_idx = local_idx + label_horizon_k
        if dates.iloc[local_idx] != dates.iloc[horizon_idx]:
            result.loc[positions[local_idx], label_col] = np.nan


def trim_labels_at_split_boundary(
    df: pd.DataFrame,
    label_horizon_k: int,
    label_col: str = "label",
    timestamp_col: str | None = None,
    ticker_col: str | None = None,
    timezone_policy: str = "naive",
) -> pd.DataFrame:
    """Mark split-tail and cross-day label horizons as invalid without deleting rows."""

    _check_positive_int(label_horizon_k, "label_horizon_k")
    result = df.copy(deep=True)
    resolved_label_col = label_col
    _validate_binary_label_values(
        result,
        resolved_label_col,
        "trim_labels_at_split_boundary",
        ticker_col,
    )
    resolved_timestamp_col = timestamp_col if timestamp_col is not None else _infer_datetime_column(result)
    if resolved_timestamp_col is not None:
        _validate_timestamp_dtype_and_timezone(
            result,
            resolved_timestamp_col,
            timezone_policy,
            "trim_labels_at_split_boundary",
        )

    for ticker, group in _trim_groups(result, ticker_col, "trim_labels_at_split_boundary"):
        context = "trim_labels_at_split_boundary"
        if ticker_col is not None:
            context = f"{context} ticker {ticker!r}"
        ordered_group = _ordered_trim_group(
            group,
            resolved_timestamp_col,
            context,
        )
        positions = list(ordered_group.index)
        tail_positions = positions[-label_horizon_k:]
        result.loc[tail_positions, resolved_label_col] = np.nan
        if resolved_timestamp_col is None:
            continue
        dates = ordered_group[resolved_timestamp_col].dt.date.reset_index(drop=True)
        _mark_cross_day_horizon_labels(result, positions, dates, label_horizon_k, resolved_label_col)
    return result


def _validate_windowed_dataset_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    ticker_col: str,
    timestamp_col: str,
    window_size: int,
    label_horizon_k: int,
    stride: int,
) -> None:
    _check_positive_int(window_size, "window_size")
    _check_positive_int(label_horizon_k, "label_horizon_k")
    _check_positive_int(stride, "stride")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    required_columns = [*feature_cols, label_col, ticker_col, timestamp_col]
    _require_columns(df, required_columns, "WindowedClassificationDataset")
    _require_numeric_columns(df, feature_cols, "WindowedClassificationDataset")
    _validate_binary_label_values(df, label_col, "WindowedClassificationDataset", ticker_col)
    if not is_datetime64_any_dtype(df[timestamp_col]):
        raise ValueError(f"timestamp column {timestamp_col!r} must be datetime dtype")
    for ticker, ticker_df in df.groupby(ticker_col, sort=False):
        _validate_strict_timestamp_order(
            ticker_df,
            timestamp_col,
            f"WindowedClassificationDataset ticker {ticker!r}",
        )


def _window_start_is_valid(
    labels: np.ndarray,
    dates: pd.Series,
    local_start_idx: int,
    window_size: int,
    label_horizon_k: int,
) -> bool:
    target_idx = local_start_idx + window_size - 1
    if pd.isna(labels[target_idx]):
        return False
    horizon_end_idx = target_idx + label_horizon_k
    return (
        dates.iloc[local_start_idx] == dates.iloc[target_idx]
        and dates.iloc[target_idx] == dates.iloc[horizon_end_idx]
    )


def _valid_window_starts_for_ticker(
    labels: np.ndarray,
    dates: pd.Series,
    window_size: int,
    label_horizon_k: int,
    stride: int,
) -> list[int]:
    max_start = len(labels) - window_size - label_horizon_k
    if max_start < 0:
        return []
    return [
        local_start_idx
        for local_start_idx in range(0, max_start + 1, stride)
        if _window_start_is_valid(labels, dates, local_start_idx, window_size, label_horizon_k)
    ]


class WindowedClassificationDataset(Dataset):
    """Torch Dataset for fixed-length classification windows.

    __getitem__ returns:
    x: torch.Tensor of shape (window_size, num_features)
    y: torch.Tensor scalar containing the class id
    A DataLoader batches x to shape (batch, window_size, num_features).
    The label is aligned to the last row of the input window to avoid overlap
    between the input window and the future label horizon.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        ticker_col: str,
        timestamp_col: str,
        window_size: int,
        label_horizon_k: int,
        stride: int = 1,
    ) -> None:
        _validate_windowed_dataset_inputs(
            df,
            feature_cols,
            label_col,
            ticker_col,
            timestamp_col,
            window_size,
            label_horizon_k,
            stride,
        )

        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.ticker_col = ticker_col
        self.timestamp_col = timestamp_col
        self.window_size = window_size
        self.label_horizon_k = label_horizon_k
        self.stride = stride
        self.valid_starts: list[tuple[Any, int]] = []
        self._features_by_ticker: dict[Any, np.ndarray] = {}
        self._labels_by_ticker: dict[Any, np.ndarray] = {}

        for ticker, ticker_df in df.groupby(ticker_col, sort=False):
            ordered = ticker_df.sort_values(timestamp_col).reset_index(drop=True)
            features = ordered[feature_cols].to_numpy(dtype=np.float32)
            labels = ordered[label_col].to_numpy(dtype=float)
            dates = ordered[timestamp_col].dt.date.reset_index(drop=True)

            self._features_by_ticker[ticker] = features
            self._labels_by_ticker[ticker] = labels
            valid_starts = _valid_window_starts_for_ticker(
                labels,
                dates,
                window_size,
                label_horizon_k,
                stride,
            )
            self.valid_starts.extend((ticker, local_start_idx) for local_start_idx in valid_starts)

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ticker, local_start_idx = self.valid_starts[idx]
        local_end_idx = local_start_idx + self.window_size
        target_idx = local_end_idx - 1
        features = self._features_by_ticker[ticker][local_start_idx:local_end_idx]
        label = self._labels_by_ticker[ticker][target_idx]
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y
