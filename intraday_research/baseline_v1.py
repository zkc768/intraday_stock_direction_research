"""Baseline v1 helper contracts for validation-only research."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

BPS_TO_DECIMAL = 10000.0
MARKET_OPEN_MINUTE = 9 * 60 + 30
TRADING_DAY_MINUTES = 390
BAR_INTERVAL_MINUTES = 5
TIME_OF_DAY_ENCODING_PERIOD_MINUTES = TRADING_DAY_MINUTES + BAR_INTERVAL_MINUTES


def _require_single_ticker_frame(frame):
    if "ticker" not in frame.columns:
        raise ValueError("Expected a ticker column for a single ticker frame.")
    if frame["ticker"].isna().any():
        raise ValueError("Expected a single non-null ticker frame.")
    if frame["ticker"].nunique(dropna=True) != 1:
        raise ValueError("Expected a single ticker frame.")


def _positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _finite_rows(frame: pd.DataFrame, columns) -> pd.Series:
    """Return a boolean mask for rows whose numeric feature values are finite."""
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature columns must be numeric for finite-row checks.") from exc
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def grouped_shift(series, group_key, periods=1):
    return series.groupby(group_key, group_keys=False).shift(periods)


def grouped_rolling(series, group_key, window, reducer):
    window = _positive_int(window, "window")
    return series.groupby(group_key, group_keys=False).apply(
        lambda part: getattr(part.rolling(window=window, min_periods=window), reducer)()
    )


def grouped_ewm(series, group_key, span):
    span = _positive_int(span, "span")
    return series.groupby(group_key, group_keys=False).apply(
        lambda part: part.ewm(span=span, adjust=False, min_periods=span).mean()
    )


def grouped_wilder_ewm(series, group_key, period):
    period = _positive_int(period, "period")
    return series.groupby(group_key, group_keys=False).apply(
        lambda part: part.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
    )


def add_baseline_v1_features(frame):
    _require_single_ticker_frame(frame)
    current = frame.sort_values("timestamp").copy()
    day = current["timestamp"].dt.date
    close = current["close"].astype(float)
    open_ = current["open"].astype(float)
    high = current["high"].astype(float)
    low = current["low"].astype(float)
    volume = current["volume"].astype(float)

    log_close = np.log(close)
    current["log_return"] = log_close.groupby(day, group_keys=False).diff()
    current["close_to_open_return"] = close / open_ - 1.0
    current["high_low_range"] = np.log(high / low)
    current["rolling_volatility_20"] = grouped_rolling(
        grouped_shift(current["log_return"], day, 1), day, 20, "std"
    )

    log_volume = np.log1p(volume)
    volume_mean_20 = grouped_rolling(grouped_shift(log_volume, day, 1), day, 20, "mean")
    current["normalized_volume_20"] = log_volume - volume_mean_20

    close_delta = close.groupby(day, group_keys=False).diff()
    gain = close_delta.clip(lower=0.0)
    loss = (-close_delta).clip(lower=0.0)
    avg_gain = grouped_wilder_ewm(gain, day, 14)
    avg_loss = grouped_wilder_ewm(loss, day, 14)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask(avg_loss.eq(0.0) & avg_gain.gt(0.0), 100.0)
    rsi = rsi.mask(avg_loss.eq(0.0) & avg_gain.eq(0.0), 50.0)
    current["rsi_14"] = rsi

    rolling_mean_20 = grouped_rolling(close, day, 20, "mean")
    rolling_std_20 = grouped_rolling(close, day, 20, "std")
    lower_band = rolling_mean_20 - 2.0 * rolling_std_20
    upper_band = rolling_mean_20 + 2.0 * rolling_std_20
    bollinger_denom = (upper_band - lower_band).replace(0.0, np.nan)
    current["bollinger_pctb"] = (close - lower_band) / bollinger_denom

    ema_12 = grouped_ewm(close, day, 12)
    ema_26 = grouped_ewm(close, day, 26)
    macd = ema_12 - ema_26
    signal = grouped_ewm(macd, day, 9)
    current["normalized_macd_hist"] = (macd - signal) / ema_26.replace(0.0, np.nan)

    minute_of_day = current["timestamp"].dt.hour * 60 + current["timestamp"].dt.minute
    minutes_since_open = minute_of_day - MARKET_OPEN_MINUTE
    current["time_of_day_sin"] = np.sin(
        2.0 * np.pi * minutes_since_open / TIME_OF_DAY_ENCODING_PERIOD_MINUTES
    )
    current["time_of_day_cos"] = np.cos(
        2.0 * np.pi * minutes_since_open / TIME_OF_DAY_ENCODING_PERIOD_MINUTES
    )
    return current


def make_no_trade_band_labels(frame, horizon_k, threshold_bps):
    _require_single_ticker_frame(frame)
    current = frame.sort_values("timestamp").copy()
    horizon_k = _positive_int(horizon_k, "horizon_k")
    threshold = threshold_bps / BPS_TO_DECIMAL

    close = current["close"].astype(float)
    future_timestamp = current["timestamp"].shift(-horizon_k)
    current["future_cumulative_return"] = close.shift(-horizon_k) / close - 1.0

    same_day = pd.Series(True, index=current.index)
    current_day = current["timestamp"].dt.date
    for offset in range(1, horizon_k + 1):
        same_day &= current_day.shift(-offset).eq(current_day)

    actual_horizon = future_timestamp - current["timestamp"]
    expected_horizon = pd.Timedelta(minutes=BAR_INTERVAL_MINUTES * horizon_k)
    current["future_horizon_minutes"] = (
        actual_horizon.dt.total_seconds() / 60.0
    )
    current["diagnostic_irregular_horizon"] = (
        future_timestamp.notna() & same_day & actual_horizon.ne(expected_horizon)
    )
    current["invalid_missing_future"] = current["future_cumulative_return"].isna()
    current["invalid_cross_day"] = ~same_day
    current["label"] = np.nan

    valid = ~(current["invalid_missing_future"] | current["invalid_cross_day"])
    current.loc[valid & (current["future_cumulative_return"] > threshold), "label"] = 1
    current.loc[valid & (current["future_cumulative_return"] < -threshold), "label"] = 0
    return current


def assign_calendar_split(timestamp, splits):
    ts = pd.Timestamp(timestamp)
    for split_name in ("train", "validation", "closed_holdout_boundary_only"):
        start, end = map(pd.Timestamp, splits[split_name])
        if start <= ts < end:
            return split_name
    return "outside_defined_calendar"


def add_split_and_invalidate_boundaries(frame, splits, horizon_k):
    _require_single_ticker_frame(frame)
    if "future_cumulative_return" not in frame.columns:
        raise ValueError("Missing future_cumulative_return; build labels before split checks.")
    current = frame.sort_values("timestamp").copy()
    current["split"] = current["timestamp"].map(
        lambda value: assign_calendar_split(value, splits)
    )
    horizon_k = _positive_int(horizon_k, "horizon_k")
    horizon_split = current["split"].shift(-horizon_k)
    current["invalid_cross_split"] = current["future_cumulative_return"].notna() & (
        current["split"] != horizon_split
    )
    current.loc[current["invalid_cross_split"], "label"] = np.nan
    return current


def fit_train_only_scaler(split_frames_by_ticker, feature_columns):
    train_parts = []
    for frame in split_frames_by_ticker.values():
        train = frame.loc[frame["split"] == "train", list(feature_columns)]
        train_parts.append(train.loc[_finite_rows(train, feature_columns)])
    train_matrix = pd.concat(train_parts, axis=0)
    if train_matrix.empty:
        raise ValueError("No train rows available for scaler fit.")
    scaler = StandardScaler()
    scaler.fit(train_matrix)
    return scaler


def transform_train_and_validation(split_frames_by_ticker, scaler, feature_columns):
    transformed = {}
    feature_columns = list(feature_columns)
    scaled_columns = [f"{name}_scaled" for name in feature_columns]
    for ticker, frame in split_frames_by_ticker.items():
        current = frame.copy()
        for column in scaled_columns:
            if column not in current.columns:
                current[column] = np.nan
        eligible = current["split"].isin(["train", "validation"])
        complete = _finite_rows(current, feature_columns)
        rows = eligible & complete
        if rows.any():
            current.loc[rows, scaled_columns] = scaler.transform(
                current.loc[rows, feature_columns]
            )
        transformed[ticker] = current
    return transformed


def build_windows_for_segment(frame, split_name, feature_columns, window_size):
    _require_single_ticker_frame(frame)
    feature_columns = list(feature_columns)
    scaled_columns = [f"{name}_scaled" for name in feature_columns]
    missing_scaled = [name for name in scaled_columns if name not in frame.columns]
    if missing_scaled:
        raise ValueError(f"Missing scaled feature columns: {missing_scaled}")

    segment = frame.loc[frame["split"] == split_name].sort_values("timestamp").copy()
    window_size = _positive_int(window_size, "window_size")
    x_values = []
    y_values = []
    metadata = []

    for _, day_frame in segment.groupby(segment["timestamp"].dt.date, sort=True):
        day_frame = day_frame.sort_values("timestamp")
        for end_pos in range(window_size - 1, len(day_frame)):
            window = day_frame.iloc[end_pos - window_size + 1 : end_pos + 1]
            target = day_frame.iloc[end_pos]
            if pd.isna(target["label"]):
                continue
            if window[scaled_columns].isna().any().any():
                continue
            x_values.append(window[scaled_columns].to_numpy(dtype=float))
            y_values.append(int(target["label"]))
            metadata.append(
                {
                    "ticker": target.get("ticker"),
                    "split": split_name,
                    "target_timestamp": target["timestamp"],
                }
            )

    if x_values:
        x_array = np.stack(x_values)
    else:
        x_array = np.empty((0, window_size, len(scaled_columns)), dtype=float)
    return {
        "X": x_array,
        "y": np.asarray(y_values, dtype=int),
        "metadata": pd.DataFrame(
            metadata, columns=["ticker", "split", "target_timestamp"]
        ),
    }


def build_windows_by_ticker_and_split(
    split_frames_by_ticker,
    feature_columns,
    window_size,
    split_names=("train", "validation"),
):
    windows = {}
    for ticker, frame in split_frames_by_ticker.items():
        windows[ticker] = {}
        for split_name in split_names:
            if split_name in set(frame["split"].dropna()):
                windows[ticker][split_name] = build_windows_for_segment(
                    frame, split_name, feature_columns, window_size
                )
    return windows


def _balanced_accuracy(y_true, predictions):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A single label was found in 'y_true' and 'y_pred'.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        return balanced_accuracy_score(y_true, predictions)


def evaluate_stratified_dummy(y_train, y_validation, seeds=(41, 42, 43, 44, 45)):
    y_train = np.asarray(y_train)
    y_validation = np.asarray(y_validation)
    y_train = y_train[~pd.isna(y_train)]
    y_validation = y_validation[~pd.isna(y_validation)]
    if len(y_train) == 0:
        raise ValueError("No train labels available for stratified dummy.")
    if len(y_validation) == 0:
        raise ValueError("No validation labels available for stratified dummy.")

    rows = []
    constant_x_train = np.zeros((len(y_train), 1))
    constant_x_validation = np.zeros((len(y_validation), 1))
    for seed in seeds:
        dummy = DummyClassifier(strategy="stratified", random_state=int(seed))
        dummy.fit(constant_x_train, y_train.astype(int))
        predictions = dummy.predict(constant_x_validation)
        rows.append(
            {
                "seed": int(seed),
                "macro_f1": f1_score(
                    y_validation.astype(int),
                    predictions,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": _balanced_accuracy(
                    y_validation.astype(int), predictions
                ),
                "accuracy": accuracy_score(y_validation.astype(int), predictions),
                "validation_n": int(len(y_validation)),
            }
        )
    return pd.DataFrame(rows)
