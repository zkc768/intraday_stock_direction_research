"""Validation-only runnable pipeline for baseline_v1 research checks."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from intraday_research.baseline_v1 import (
    add_baseline_v1_features,
    add_split_and_invalidate_boundaries,
    build_windows_by_ticker_and_split,
    evaluate_stratified_dummy,
    fit_train_only_scaler,
    make_no_trade_band_labels,
    transform_train_and_validation,
)

DEFAULT_TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
FEATURE_COLUMNS = (
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
)
CALENDAR_SPLITS = {
    "train": ("1998-01-02", "2013-09-16"),
    "validation": ("2013-09-16", "2017-01-25"),
    "closed_holdout_boundary_only": ("2017-01-25", "2020-06-06"),
}
EXPECTED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_DIAGNOSTIC_TRAIN_CAP = 20000
ROW_SUBSAMPLE_STRATEGY = "uniform_index_stride_across_concatenated_ticker_windows"


def find_timestamp_column(columns) -> str:
    for candidate in ("timestamp", "datetime", "date", "time"):
        for column in columns:
            if str(column).lower() == candidate:
                return column
    raise ValueError(f"No timestamp-like column found in columns: {list(columns)}")


def load_ticker_csv(ticker: str, data_dir: str | Path = "data") -> pd.DataFrame:
    data_path = Path(data_dir) / f"{ticker}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing raw ticker file: {data_path}")

    frame = pd.read_csv(data_path)
    timestamp_column = find_timestamp_column(frame.columns)
    frame = frame.rename(columns={timestamp_column: "timestamp"}).copy()
    missing_columns = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"{data_path} missing required columns: {missing_columns}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["ticker"] = ticker
    return frame.sort_values("timestamp").reset_index(drop=True)


def audit_ticker_frame(ticker: str, frame: pd.DataFrame, data_dir: str | Path) -> dict:
    path = Path(data_dir) / f"{ticker}.csv"
    return {
        "ticker": ticker,
        "path": str(path),
        "exists": path.exists(),
        "n_rows": int(len(frame)),
        "start_ts": str(frame["timestamp"].min()),
        "end_ts": str(frame["timestamp"].max()),
        "missing_total": int(frame[list(EXPECTED_COLUMNS)].isna().sum().sum()),
        "duplicate_timestamp_count": int(frame["timestamp"].duplicated().sum()),
    }


def prepare_split_frames(
    raw_frames_by_ticker: dict[str, pd.DataFrame],
    *,
    splits: dict[str, tuple[str, str]] = CALENDAR_SPLITS,
    horizon_k: int = 12,
    threshold_bps: float = 5.0,
) -> dict[str, pd.DataFrame]:
    split_frames = {}
    for ticker, frame in raw_frames_by_ticker.items():
        featured = add_baseline_v1_features(frame)
        labeled = make_no_trade_band_labels(
            featured,
            horizon_k=horizon_k,
            threshold_bps=threshold_bps,
        )
        split_frames[ticker] = add_split_and_invalidate_boundaries(
            labeled,
            splits=splits,
            horizon_k=horizon_k,
        )
    return split_frames


def summarize_split_labels(split_frames_by_ticker: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for ticker, frame in split_frames_by_ticker.items():
        for split_name in ("train", "validation"):
            part = frame.loc[frame["split"] == split_name]
            rows.append(
                {
                    "ticker": ticker,
                    "split": split_name,
                    "n_rows": int(len(part)),
                    "valid_labels": int(part["label"].notna().sum()),
                    "invalid_cross_split": int(part["invalid_cross_split"].sum()),
                    "invalid_cross_day": int(part["invalid_cross_day"].sum()),
                    "invalid_missing_future": int(part["invalid_missing_future"].sum()),
                    "diagnostic_irregular_horizon": int(
                        part["diagnostic_irregular_horizon"].sum()
                    ),
                }
            )
    return rows


def summarize_window_class_balance(windows_by_ticker: dict[str, dict]) -> list[dict]:
    rows = []
    for ticker, split_map in windows_by_ticker.items():
        for split_name in ("train", "validation"):
            bundle = split_map.get(split_name)
            if bundle is None:
                continue
            labels = bundle["y"]
            rows.append(
                {
                    "ticker": ticker,
                    "split": split_name,
                    "shape": list(bundle["X"].shape),
                    "n_windows": int(len(labels)),
                    "class_0": int((labels == 0).sum()),
                    "class_1": int((labels == 1).sum()),
                    "scope": f"{split_name}_window_class_balance_diagnostic",
                }
            )
    return rows


def pooled_train_validation_labels(windows_by_ticker: dict[str, dict]) -> tuple[np.ndarray, np.ndarray]:
    train_labels = []
    validation_labels = []
    for ticker, split_map in windows_by_ticker.items():
        if "train" not in split_map or "validation" not in split_map:
            raise ValueError(f"Missing train/validation windows for ticker: {ticker}")
        train_labels.append(split_map["train"]["y"])
        validation_labels.append(split_map["validation"]["y"])
    return np.concatenate(train_labels), np.concatenate(validation_labels)


def collect_last_step_xy(
    windows_by_ticker: dict[str, dict],
    split_name: str,
    feature_indices=None,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    for split_map in windows_by_ticker.values():
        bundle = split_map.get(split_name)
        if bundle is None:
            continue
        x_parts.append(bundle["X"][:, -1, :])
        y_parts.append(bundle["y"])
    if not x_parts:
        raise ValueError(f"No {split_name} windows available.")
    x_values = np.concatenate(x_parts, axis=0)
    if feature_indices is not None:
        x_values = x_values[:, list(feature_indices)]
    return x_values, np.concatenate(y_parts)


def subsample_rows_uniformly(x_values, y_values, max_rows: int | None):
    if max_rows is None or len(y_values) <= max_rows:
        return x_values, y_values
    max_rows = int(max_rows)
    if max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    selected = np.linspace(0, len(y_values) - 1, num=max_rows, dtype=int)
    return x_values[selected], y_values[selected]


def summarize_dummy_baseline(dummy_rows: pd.DataFrame) -> dict:
    return {
        "model": "stratified_dummy",
        "ticker_or_pooled": "pooled",
        "dummy_strategy": "stratified",
        "macro_f1_mean": float(dummy_rows["macro_f1"].mean()),
        "macro_f1_std": float(dummy_rows["macro_f1"].std(ddof=0)),
        "balanced_accuracy_mean": float(dummy_rows["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(dummy_rows["balanced_accuracy"].std(ddof=0)),
        "accuracy_mean": float(dummy_rows["accuracy"].mean()),
        "n": int(dummy_rows["validation_n"].iloc[0]),
        "scope": "validation_only",
    }


def summarize_feature_signal(
    windows_by_ticker: dict[str, dict],
    feature_columns,
) -> list[dict]:
    rows = []
    feature_columns = list(feature_columns)
    for split_name in ("train", "validation"):
        x_parts = []
        y_parts = []
        for split_map in windows_by_ticker.values():
            bundle = split_map.get(split_name)
            if bundle is None:
                continue
            x_parts.append(bundle["X"][:, -1, :])
            y_parts.append(bundle["y"])
        x_values = np.concatenate(x_parts, axis=0)
        y_values = np.concatenate(y_parts, axis=0)
        for feature_index, feature_name in enumerate(feature_columns):
            class_0 = x_values[y_values == 0, feature_index]
            class_1 = x_values[y_values == 1, feature_index]
            rows.append(
                {
                    "split": split_name,
                    "feature": feature_name,
                    "class_0_mean": float(class_0.mean()) if len(class_0) else np.nan,
                    "class_1_mean": float(class_1.mean()) if len(class_1) else np.nan,
                    "class_1_minus_class_0": (
                        float(class_1.mean() - class_0.mean())
                        if len(class_0) and len(class_1)
                        else np.nan
                    ),
                    "scope": "validation_only_diagnostic",
                }
            )
    return rows


def compute_mutual_information_diagnostic(
    windows_by_ticker: dict[str, dict],
    feature_columns,
    *,
    max_rows: int = DEFAULT_DIAGNOSTIC_TRAIN_CAP,
    random_state: int = 42,
) -> list[dict]:
    rows = []
    for split_name in ("train", "validation"):
        x_values, y_values = collect_last_step_xy(windows_by_ticker, split_name)
        x_values, y_values = subsample_rows_uniformly(x_values, y_values, max_rows)
        mi_values = mutual_info_classif(
            x_values,
            y_values,
            discrete_features=False,
            random_state=random_state,
        )
        for feature_name, mi_value in zip(feature_columns, mi_values):
            rows.append(
                {
                    "split": split_name,
                    "feature": feature_name,
                    "mutual_information": float(mi_value),
                    "n": int(len(y_values)),
                    "row_subsample_strategy": ROW_SUBSAMPLE_STRATEGY,
                    "scope": "validation_only_signal_diagnostic_not_selection",
                }
            )
    return rows


def evaluate_predictions(y_true, predictions) -> dict:
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
        return {
            "macro_f1": float(
                f1_score(
                    y_true,
                    predictions,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "accuracy": float(accuracy_score(y_true, predictions)),
        }


def evaluate_sklearn_logreg_last_step(
    windows_by_ticker: dict[str, dict],
    *,
    feature_indices=None,
    max_train_rows: int = DEFAULT_DIAGNOSTIC_TRAIN_CAP,
) -> dict:
    x_train, y_train = collect_last_step_xy(
        windows_by_ticker,
        "train",
        feature_indices=feature_indices,
    )
    x_validation, y_validation = collect_last_step_xy(
        windows_by_ticker,
        "validation",
        feature_indices=feature_indices,
    )
    x_train, y_train = subsample_rows_uniformly(x_train, y_train, max_train_rows)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    metrics = evaluate_predictions(y_validation, predictions)
    metrics.update(
        {
            "model": "sklearn_logreg_last_step",
            "ticker_or_pooled": "pooled",
            "train_n": int(len(y_train)),
            "validation_n": int(len(y_validation)),
            "train_row_subsample_strategy": ROW_SUBSAMPLE_STRATEGY,
            "scope": "validation_only_diagnostic_not_selection",
        }
    )
    return metrics


def run_feature_ablation_diagnostic(
    windows_by_ticker: dict[str, dict],
    feature_columns,
    *,
    max_train_rows: int = DEFAULT_DIAGNOSTIC_TRAIN_CAP,
) -> list[dict]:
    feature_columns = list(feature_columns)
    full = evaluate_sklearn_logreg_last_step(
        windows_by_ticker,
        max_train_rows=max_train_rows,
    )
    rows = [
        {
            "ablation": "all_features",
            "removed_feature": None,
            **full,
        }
    ]
    all_indices = list(range(len(feature_columns)))
    for removed_index, removed_feature in enumerate(feature_columns):
        kept_indices = [index for index in all_indices if index != removed_index]
        result = evaluate_sklearn_logreg_last_step(
            windows_by_ticker,
            feature_indices=kept_indices,
            max_train_rows=max_train_rows,
        )
        result["delta_macro_f1_vs_all_features"] = (
            result["macro_f1"] - full["macro_f1"]
        )
        rows.append(
            {
                "ablation": "leave_one_feature_out",
                "removed_feature": removed_feature,
                **result,
            }
        )
    rows[0]["delta_macro_f1_vs_all_features"] = 0.0
    return rows


def precheck_lightgbm_dependency() -> dict:
    if importlib.util.find_spec("lightgbm") is None:
        return {
            "adapter": "lightgbm",
            "available": False,
            "blocker": "Missing Python dependency: lightgbm",
            "scope": "validation_only_adapter_precheck",
        }
    return {
        "adapter": "lightgbm",
        "available": True,
        "blocker": None,
        "scope": "validation_only_adapter_precheck",
    }


def evaluate_lightgbm_last_step_adapter(
    windows_by_ticker: dict[str, dict],
    *,
    max_train_rows: int = DEFAULT_DIAGNOSTIC_TRAIN_CAP,
) -> dict:
    dependency = precheck_lightgbm_dependency()
    if not dependency["available"]:
        return dependency

    from lightgbm import LGBMClassifier

    x_train, y_train = collect_last_step_xy(windows_by_ticker, "train")
    x_validation, y_validation = collect_last_step_xy(windows_by_ticker, "validation")
    x_train, y_train = subsample_rows_uniformly(x_train, y_train, max_train_rows)
    model = LGBMClassifier(
        n_estimators=25,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_validation)
    metrics = evaluate_predictions(y_validation, predictions)
    metrics.update(
        {
            "adapter": "lightgbm",
            "available": True,
            "model": "lightgbm_lgbmclassifier_last_step_tiny",
            "ticker_or_pooled": "pooled",
            "train_n": int(len(y_train)),
            "validation_n": int(len(y_validation)),
            "train_row_subsample_strategy": ROW_SUBSAMPLE_STRATEGY,
            "scope": "validation_only_tiny_adapter_diagnostic_not_selection",
        }
    )
    return metrics


def build_walk_forward_fold_specs(
    split_frames_by_ticker: dict[str, pd.DataFrame],
    *,
    n_folds: int = 3,
) -> list[dict]:
    n_folds = int(n_folds)
    if n_folds <= 0:
        raise ValueError("n_folds must be positive.")

    rows = []
    for ticker, frame in split_frames_by_ticker.items():
        eligible = frame.loc[frame["split"].isin(["train", "validation"])].copy()
        unique_days = pd.Series(eligible["timestamp"].dt.normalize().unique()).sort_values()
        if len(unique_days) < n_folds + 1:
            raise ValueError(f"Not enough train/validation days for walk-forward folds: {ticker}")
        day_chunks = np.array_split(np.arange(len(unique_days)), n_folds + 1)
        for fold_index in range(1, len(day_chunks)):
            train_positions = np.concatenate(day_chunks[:fold_index])
            validation_positions = day_chunks[fold_index]
            if len(train_positions) == 0 or len(validation_positions) == 0:
                continue
            train_end_pos = int(train_positions[-1])
            validation_start_pos = int(validation_positions[0])
            validation_end_pos = int(validation_positions[-1])
            train_start = unique_days.iloc[0]
            train_end = unique_days.iloc[train_end_pos]
            validation_start = unique_days.iloc[validation_start_pos]
            validation_end = unique_days.iloc[validation_end_pos]
            rows.append(
                {
                    "ticker": ticker,
                    "fold": fold_index,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "validation_start": str(validation_start.date()),
                    "validation_end": str(validation_end.date()),
                    "train_n_rows": int(
                        (
                            (eligible["timestamp"].dt.normalize() >= train_start)
                            & (eligible["timestamp"].dt.normalize() <= train_end)
                        ).sum()
                    ),
                    "validation_n_rows": int(
                        (
                            (eligible["timestamp"].dt.normalize() >= validation_start)
                            & (eligible["timestamp"].dt.normalize() <= validation_end)
                        ).sum()
                    ),
                    "chronological": bool(train_end < validation_start),
                    "scope": "train_validation_only_walk_forward_contract",
                }
            )
    return rows


def build_validation_only_report(
    *,
    data_dir: str | Path = "data",
    tickers=DEFAULT_TICKERS,
    feature_columns=FEATURE_COLUMNS,
    splits: dict[str, tuple[str, str]] = CALENDAR_SPLITS,
    horizon_k: int = 12,
    threshold_bps: float = 5.0,
    window_size: int = 12,
    dummy_seeds=(41, 42, 43, 44, 45),
    diagnostic_max_train_rows: int = DEFAULT_DIAGNOSTIC_TRAIN_CAP,
    walk_forward_folds: int = 3,
) -> dict:
    raw_frames = {
        ticker: load_ticker_csv(ticker, data_dir=data_dir) for ticker in tickers
    }
    split_frames = prepare_split_frames(
        raw_frames,
        splits=splits,
        horizon_k=horizon_k,
        threshold_bps=threshold_bps,
    )
    scaler = fit_train_only_scaler(split_frames, feature_columns=feature_columns)
    scaled_frames = transform_train_and_validation(
        split_frames,
        scaler,
        feature_columns=feature_columns,
    )
    windows = build_windows_by_ticker_and_split(
        scaled_frames,
        feature_columns=feature_columns,
        window_size=window_size,
        split_names=("train", "validation"),
    )
    y_train, y_validation = pooled_train_validation_labels(windows)
    dummy_rows = evaluate_stratified_dummy(
        y_train,
        y_validation,
        seeds=dummy_seeds,
    )
    dummy_summary = summarize_dummy_baseline(dummy_rows)
    logreg_diagnostic = evaluate_sklearn_logreg_last_step(
        windows,
        max_train_rows=diagnostic_max_train_rows,
    )
    logreg_diagnostic["dummy_macro_f1"] = dummy_summary["macro_f1_mean"]
    logreg_diagnostic["delta_macro_f1_vs_dummy"] = (
        logreg_diagnostic["macro_f1"] - dummy_summary["macro_f1_mean"]
    )

    return {
        "metadata": {
            "feature_set_id": "baseline_v1",
            "label_policy": "no_trade_band",
            "threshold_source": "fixed_pre_registered_5bps",
            "threshold_bps": float(threshold_bps),
            "label_horizon_k": int(horizon_k),
            "window_size": int(window_size),
            "decision_time_policy": "post_bar_close_completed_bar",
            "scaler_id": "standard_pooled_train_only_v1",
            "scaler_fit_scope": "pooled_train_after_per_ticker_chronological_split",
            "transformed_splits": ["train", "validation"],
            "diagnostic_row_subsample": {
                "strategy": ROW_SUBSAMPLE_STRATEGY,
                "max_train_rows": int(diagnostic_max_train_rows),
                "applies_to": [
                    "mutual_information_diagnostic",
                    "sklearn_logreg_last_step",
                    "lightgbm_tiny_adapter",
                    "feature_ablation_diagnostic",
                ],
            },
            "closed_holdout_policy": "boundary_invalidation_only_not_transformed_not_windowed_not_scored",
            "scope": "validation_only",
        },
        "coverage": [
            audit_ticker_frame(ticker, frame, data_dir) for ticker, frame in raw_frames.items()
        ],
        "split_label_summary": summarize_split_labels(split_frames),
        "window_class_balance": summarize_window_class_balance(windows),
        "dummy_baseline_by_seed": dummy_rows.to_dict(orient="records"),
        "dummy_baseline_summary": dummy_summary,
        "last_step_feature_signal_diagnostic": summarize_feature_signal(
            windows,
            feature_columns,
        ),
        "mutual_information_diagnostic": compute_mutual_information_diagnostic(
            windows,
            feature_columns,
            max_rows=diagnostic_max_train_rows,
        ),
        "model_adapter_precheck": {
            "lightgbm": precheck_lightgbm_dependency(),
            "dependency_free_diagnostic": logreg_diagnostic,
        },
        "lightgbm_tiny_adapter_result": evaluate_lightgbm_last_step_adapter(
            windows,
            max_train_rows=diagnostic_max_train_rows,
        ),
        "feature_ablation_diagnostic": run_feature_ablation_diagnostic(
            windows,
            feature_columns,
            max_train_rows=diagnostic_max_train_rows,
        ),
        "walk_forward_contract": build_walk_forward_fold_specs(
            split_frames,
            n_folds=walk_forward_folds,
        ),
    }
