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
        raise ValueError(f"{context} timestamp column {timestamp_col!r} must be strictly increasing")


def _validate_binary_label_values(df: pd.DataFrame, label_col: str, context: str) -> None:
    _require_columns(df, [label_col], context)
    if not is_numeric_dtype(df[label_col]):
        raise ValueError(f"{context} label column {label_col!r} must be numeric")
    values = set(df[label_col].dropna().astype(float).unique())
    invalid_values = values.difference({0.0, 1.0})
    if invalid_values:
        raise ValueError(f"{context} label column {label_col!r} contains values outside {{0, 1, NaN}}")


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
    result.loc[:, feature_cols] = scaler.transform(result[feature_cols].to_numpy(dtype=float))
    return result


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
    _validate_binary_label_values(result, resolved_label_col, "trim_labels_at_split_boundary")
    resolved_timestamp_col = timestamp_col if timestamp_col is not None else _infer_datetime_column(result)
    if resolved_timestamp_col is not None:
        _validate_timestamp_dtype_and_timezone(
            result,
            resolved_timestamp_col,
            timezone_policy,
            "trim_labels_at_split_boundary",
        )

    if ticker_col is not None:
        _require_columns(result, [ticker_col], "trim_labels_at_split_boundary")
        groups = [group for _, group in result.groupby(ticker_col, sort=False)]
    else:
        groups = [result]

    for group in groups:
        if resolved_timestamp_col is not None:
            ordered_group = group.sort_values(resolved_timestamp_col)
            _validate_strict_timestamp_order(
                ordered_group,
                resolved_timestamp_col,
                "trim_labels_at_split_boundary",
            )
        else:
            ordered_group = group
        positions = list(ordered_group.index)
        tail_positions = positions[-label_horizon_k:]
        result.loc[tail_positions, resolved_label_col] = np.nan
        if resolved_timestamp_col is None:
            continue
        dates = ordered_group[resolved_timestamp_col].dt.date.reset_index(drop=True)
        for local_idx in range(max(len(ordered_group) - label_horizon_k, 0)):
            horizon_idx = local_idx + label_horizon_k
            if dates.iloc[local_idx] != dates.iloc[horizon_idx]:
                result.loc[positions[local_idx], resolved_label_col] = np.nan
    return result


class WindowedClassificationDataset(Dataset):
    """Torch Dataset for fixed-length classification windows.

    __getitem__ returns:
    x: torch.Tensor of shape (window_size, num_features)
    y: torch.Tensor scalar containing the class id
    A DataLoader batches x to shape (batch, window_size, num_features).
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
        _check_positive_int(window_size, "window_size")
        _check_positive_int(label_horizon_k, "label_horizon_k")
        _check_positive_int(stride, "stride")
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty")
        required_columns = [*feature_cols, label_col, ticker_col, timestamp_col]
        _require_columns(df, required_columns, "WindowedClassificationDataset")
        _require_numeric_columns(df, feature_cols, "WindowedClassificationDataset")
        if not is_datetime64_any_dtype(df[timestamp_col]):
            raise ValueError(f"timestamp column {timestamp_col!r} must be datetime dtype")

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
            max_start = len(ordered) - window_size - label_horizon_k
            if max_start < 0:
                continue
            for local_start_idx in range(0, max_start + 1, stride):
                if pd.isna(labels[local_start_idx]):
                    continue
                window_end_idx = local_start_idx + window_size - 1
                horizon_end_idx = window_end_idx + label_horizon_k
                window_dates = dates.iloc[local_start_idx : window_end_idx + 1]
                if window_dates.nunique() != 1:
                    continue
                if dates.iloc[local_start_idx] != dates.iloc[horizon_end_idx]:
                    continue
                self.valid_starts.append((ticker, local_start_idx))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ticker, local_start_idx = self.valid_starts[idx]
        local_end_idx = local_start_idx + self.window_size
        features = self._features_by_ticker[ticker][local_start_idx:local_end_idx]
        label = self._labels_by_ticker[ticker][local_start_idx]
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y
