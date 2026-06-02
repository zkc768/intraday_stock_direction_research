"""Run local Phase 1B baseline matrices on the five-stock CSV data."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_utils.checkpoint import load_checkpoint
from ml_utils.config import DataConfig
from ml_utils.dataset import WindowedClassificationDataset
from ml_utils.dataset import fit_scaler_on_train
from ml_utils.dataset import make_binary_labels_from_future_avg_return
from ml_utils.dataset import make_no_trade_band_labels
from ml_utils.dataset import make_time_splits
from ml_utils.dataset import transform_split
from ml_utils.dataset import trim_labels_at_split_boundary
from ml_utils.metrics import always_predict_baseline_metrics
from ml_utils.metrics import compute_classification_metrics
from ml_utils.metrics import dummy_baseline_metrics
from ml_utils.models.dlinear_classifier import DLinearClassifier
from ml_utils.models.lstm_classifier import LSTMClassifier
from ml_utils.models.ms_dlinear_tcn_classifier import MultiScaleDLinearTCNClassifier
from ml_utils.models.tcn_classifier import TCNClassifier
from ml_utils.seed import seed_everything
from ml_utils.trainer import Trainer
from ml_utils.trainer import evaluate


DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "phase1b_local_baseline"
DEFAULT_TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
OHLCV_FEATURES = ("open", "high", "low", "close", "volume")
TECHNICAL_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_pctb",
    "rolling_std_20",
    "obv_roc",
)
STATIONARY_V1_CORE_FEATURES = (
    "log_ret_1",
    "log_ret_3",
    "log_ret_6",
    "oc_log_ret",
    "hl_log_range",
    "body_to_range",
    "rv_6",
    "log_volume_chg_1",
)
MENTOR_CLEAN_V1_FEATURES = (
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
FEATURE_SETS = {
    "ohlcv_only_v1": OHLCV_FEATURES,
    "technical_v1": TECHNICAL_FEATURES,
    "stationary_v1_core": STATIONARY_V1_CORE_FEATURES,
    "mentor_clean_v1": MENTOR_CLEAN_V1_FEATURES,
}
SKLEARN_LOGREG_C_GRID = (0.01, 0.1, 1.0, 10.0)
SKLEARN_LOGREG_CLASS_WEIGHTS = (None, "balanced")
LIGHTGBM_MODEL_NAME = "lightgbm_lgbm_classifier"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    window_size: int
    label_horizon_k: int
    label_mode: str
    threshold_bps: float


@dataclass(frozen=True)
class CalendarSplitSpec:
    train_start_ts: pd.Timestamp
    train_end_ts: pd.Timestamp
    val_start_ts: pd.Timestamp
    val_end_ts: pd.Timestamp
    holdout_start_ts: pd.Timestamp
    holdout_end_ts: pd.Timestamp


@dataclass(frozen=True)
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame | None
    train_dataset: WindowedClassificationDataset
    val_dataset: WindowedClassificationDataset
    test_dataset: WindowedClassificationDataset | None
    val_datasets_by_ticker: dict[str, WindowedClassificationDataset]
    test_datasets_by_ticker: dict[str, WindowedClassificationDataset]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray | None
    y_train_by_ticker: dict[str, np.ndarray]
    y_val_by_ticker: dict[str, np.ndarray]
    y_test_by_ticker: dict[str, np.ndarray]
    diagnostics_by_ticker: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Phase 1B baseline matrices."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", choices=["A"], default="A")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=None)
    parser.add_argument(
        "--label-mode",
        choices=["legacy_binary", "binary", "no_trade_band"],
        default="legacy_binary",
        help=(
            "Default legacy_binary/binary uses the canonical Phase 1 label "
            "future_avg_r > 0 else 0; no_trade_band is an explicit Phase 1B "
            "diagnostic subset."
        ),
    )
    parser.add_argument("--threshold-bps", type=float, default=None)
    parser.add_argument("--models", nargs="+", default=["lstm", "tcn", "dlinear"])
    parser.add_argument(
        "--model-family",
        choices=["torch", "sklearn_logreg", "lightgbm"],
        default="torch",
    )
    parser.add_argument("--sklearn-baseline", action="store_true")
    parser.add_argument("--validation-only-report", action="store_true")
    parser.add_argument("--validation-only-per-ticker", action="store_true")
    parser.add_argument("--logreg-c-grid", nargs="+", default=None)
    parser.add_argument("--logreg-class-weights", nargs="+", default=None)
    parser.add_argument(
        "--feature-view",
        choices=["last_step", "flatten_window"],
        default="last_step",
    )
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--max-rows-per-ticker", type=int, default=None)
    parser.add_argument("--split-mode", choices=["ratio", "calendar"], default="ratio")
    parser.add_argument("--train-start-ts", default=None)
    parser.add_argument("--train-end-ts", default=None)
    parser.add_argument("--val-start-ts", default=None)
    parser.add_argument("--val-end-ts", default=None)
    parser.add_argument("--holdout-start-ts", default=None)
    parser.add_argument("--holdout-end-ts", default=None)
    parser.add_argument("--shuffle-train-labels", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    validate_split_args(parser, args)
    return args


def main() -> None:
    args = parse_args()
    run_mode = resolve_run_mode(args)
    model_family = resolve_model_family(args)
    if args.validation_only_report and model_family not in {"sklearn_logreg", "lightgbm", "torch"}:
        raise ValueError(
            "--validation-only-report requires --sklearn-baseline "
            "or --model-family sklearn_logreg/lightgbm/torch"
        )
    if model_family == "lightgbm" and not args.validation_only_report:
        raise ValueError("--model-family lightgbm requires --validation-only-report")
    if args.validation_only_per_ticker and not args.validation_only_report:
        raise ValueError(
            "--validation-only-per-ticker requires --validation-only-report"
        )
    if args.window_size <= 0:
        raise ValueError(f"window-size must be positive, got {args.window_size}")
    calendar_split = calendar_split_spec_from_args(args)
    logreg_c_grid = parse_logreg_c_grid(args.logreg_c_grid)
    logreg_class_weights = parse_logreg_class_weights(args.logreg_class_weights)
    label_mode = resolve_label_mode(args)
    threshold_bps = resolve_threshold_bps(args, label_mode)
    candidate = CandidateSpec(
        "A",
        window_size=args.window_size,
        label_horizon_k=12,
        label_mode=label_mode,
        threshold_bps=threshold_bps,
    )
    tickers = resolve_tickers(args, run_mode)
    seeds = resolve_seeds(args, run_mode)
    max_epochs = resolve_max_epochs(args, run_mode)
    feature_set_id = resolve_feature_set(args, run_mode)
    if model_family == "lightgbm":
        validate_lightgbm_pm_route(
            args=args,
            run_mode=run_mode,
            feature_set_id=feature_set_id,
            label_mode=label_mode,
            threshold_bps=threshold_bps,
        )
    if model_family == "torch" and args.validation_only_report:
        validate_torch_validation_only_pm_route(
            args=args,
            run_mode=run_mode,
            feature_set_id=feature_set_id,
            label_mode=label_mode,
            threshold_bps=threshold_bps,
        )
    feature_cols = list(FEATURE_SETS[feature_set_id])
    max_rows_per_ticker = None if calendar_split is not None else resolve_max_rows(args, run_mode)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = DataConfig(
        tickers=tickers,
        data_dir=str(args.data_dir),
        label_mode=candidate.label_mode,
        threshold_bps=candidate.threshold_bps,
        feature_cols=feature_cols,
    )

    metadata = {
        "run_id": build_run_id(run_mode, candidate.label_mode),
        "run_mode": run_mode,
        "git_commit_hash": git_output(["rev-parse", "HEAD"]),
        "git_status_short": git_output(["status", "--short"]),
        "data_source": str(args.data_dir.resolve()),
        "feature_set_id": feature_set_id,
        "feature_columns": feature_cols,
        "candidate_id": candidate.candidate_id,
        "label_mode": candidate.label_mode,
        "window_size": candidate.window_size,
        "label_horizon_k": candidate.label_horizon_k,
        "threshold_bps": candidate.threshold_bps,
        **protocol_metadata_fields(
            feature_set_id,
            candidate.label_mode,
            candidate.threshold_bps,
            args.threshold_bps,
        ),
        "timestamp_col": data_config.timestamp_col,
        "split_mode": args.split_mode,
        "split_date_ranges_available": True,
        "split_date_range_timestamp_col": data_config.timestamp_col,
        "split_date_range_source": "prepared_split_frames_after_feature_label_filtering",
        "price_col": data_config.price_col,
        "tickers": tickers,
        "models": model_names_for_family(model_family, args.models),
        "model_family": model_family,
        "feature_view": args.feature_view,
        "logreg_c_grid": list(logreg_c_grid),
        "logreg_class_weights": [
            "none" if value is None else value for value in logreg_class_weights
        ],
        "validation_only_per_ticker": bool(args.validation_only_per_ticker),
        "seeds": seeds,
        "shuffle_seed": seeds[0],
        "shuffle_seed_policy": "first_seed",
        "max_epochs": max_epochs,
        "batch_size": args.batch_size,
        "checkpoint_policy": "best_val_macro_f1",
        "training_scope": "pooled",
        "baseline_scope": "pooled_train",
        "primary_baseline_scope": "pooled_train",
        "secondary_baseline_scope": "per_ticker_train",
        "secondary_baseline_scope_note": (
            "ticker rows use per-ticker train labels; pooled rows use pooled_train"
        ),
        "dummy_stratified_random_states": list(range(10)),
        "shuffle_train_labels": bool(args.shuffle_train_labels),
        "max_rows_per_ticker": args.max_rows_per_ticker,
        "effective_max_rows_per_ticker": max_rows_per_ticker,
        **audit_scope_fields(run_mode),
        **label_metadata_fields(candidate.label_mode, candidate.threshold_bps),
        **calendar_split_metadata_fields(calendar_split),
    }
    if args.validation_only_report:
        metadata.update(validation_only_report_fields())

    prepared = prepare_data(
        data_dir=args.data_dir,
        tickers=tickers,
        feature_set_id=feature_set_id,
        feature_cols=feature_cols,
        candidate=candidate,
        max_rows_per_ticker=max_rows_per_ticker,
        calendar_split=calendar_split,
        shuffle_train_labels=args.shuffle_train_labels,
        shuffle_seed=metadata["shuffle_seed"],
        include_test_data=not args.validation_only_report,
    )
    manifest_rows = build_manifest_rows(metadata, candidate, prepared)
    write_outputs(output_dir, "manifest", manifest_rows, metadata)
    if args.manifest_only:
        print(f"wrote manifest rows: {len(manifest_rows)}")
        return

    result_rows: list[dict[str, Any]] = []
    if model_family == "sklearn_logreg":
        result_rows.extend(
            run_sklearn_logreg_baseline(
                metadata=metadata,
                candidate=candidate,
                prepared=prepared,
                feature_view=args.feature_view,
                validation_only_report=args.validation_only_report,
                validation_only_per_ticker=args.validation_only_per_ticker,
                c_grid=logreg_c_grid,
                class_weights=logreg_class_weights,
            )
        )
        write_outputs(output_dir, "results", result_rows, metadata)
    elif model_family == "lightgbm":
        result_rows.extend(
            run_lightgbm_validation_only_baseline(
                metadata=metadata,
                candidate=candidate,
                prepared=prepared,
                feature_view=args.feature_view,
                validation_only_per_ticker=args.validation_only_per_ticker,
            )
        )
        write_outputs(output_dir, "results", result_rows, metadata)
    else:
        for seed in seeds:
            for model_name in args.models:
                row_group = run_model_once(
                    model_name=model_name,
                    seed=seed,
                    max_epochs=max_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    early_stop_patience=args.early_stop_patience,
                    output_dir=output_dir,
                    metadata=metadata,
                    candidate=candidate,
                    prepared=prepared,
                    feature_cols=feature_cols,
                    validation_only_report=args.validation_only_report,
                    validation_only_per_ticker=args.validation_only_per_ticker,
                )
                result_rows.extend(row_group)
                write_outputs(output_dir, "results", result_rows, metadata)
    print(f"wrote result rows: {len(result_rows)}")


def resolve_run_mode(args: argparse.Namespace) -> str:
    if args.full_run and args.smoke:
        raise ValueError("--full-run and --smoke are mutually exclusive")
    if args.full_run:
        return "full"
    if args.manifest_only:
        return "manifest"
    return "smoke"


def resolve_label_mode(args: argparse.Namespace) -> str:
    if args.label_mode == "binary":
        return "legacy_binary"
    return args.label_mode


def resolve_model_family(args: argparse.Namespace) -> str:
    if args.sklearn_baseline:
        return "sklearn_logreg"
    return args.model_family


def model_names_for_family(model_family: str, torch_models: list[str]) -> list[str]:
    if model_family == "sklearn_logreg":
        return ["sklearn_logreg_l2"]
    if model_family == "lightgbm":
        return [LIGHTGBM_MODEL_NAME]
    return torch_models


def parse_logreg_c_grid(values: list[str] | None) -> tuple[float, ...]:
    if values is None:
        return SKLEARN_LOGREG_C_GRID
    parsed = []
    for raw_value in split_cli_tokens(values):
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"logreg-c-grid value must be numeric: {raw_value}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"logreg-c-grid values must be positive, got {raw_value}")
        parsed.append(value)
    if not parsed:
        raise ValueError("logreg-c-grid must contain at least one value")
    return tuple(parsed)


def parse_logreg_class_weights(values: list[str] | None) -> tuple[str | None, ...]:
    if values is None:
        return SKLEARN_LOGREG_CLASS_WEIGHTS
    parsed = []
    for raw_value in split_cli_tokens(values):
        normalized = raw_value.lower()
        if normalized in {"none", "null"}:
            parsed.append(None)
        elif normalized == "balanced":
            parsed.append("balanced")
        else:
            raise ValueError(
                "logreg-class-weights values must be none/null or balanced, "
                f"got {raw_value}"
            )
    if not parsed:
        raise ValueError("logreg-class-weights must contain at least one value")
    return tuple(parsed)


def split_cli_tokens(values: list[str]) -> list[str]:
    tokens = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(","))
    return [token for token in tokens if token]


def validate_split_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    calendar_arg_names = [
        "train_start_ts",
        "train_end_ts",
        "val_start_ts",
        "val_end_ts",
        "holdout_start_ts",
        "holdout_end_ts",
    ]
    provided = [name for name in calendar_arg_names if getattr(args, name) is not None]
    if args.split_mode == "ratio":
        if provided:
            parser.error("calendar timestamp args require --split-mode calendar")
        return

    missing = [name for name in calendar_arg_names if getattr(args, name) is None]
    if missing:
        parser.error(
            "--split-mode calendar requires all calendar timestamp args: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if args.max_rows_per_ticker is not None:
        parser.error("--split-mode calendar cannot be used with --max-rows-per-ticker")
    try:
        calendar_split_spec_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def calendar_split_spec_from_args(
    args: argparse.Namespace,
) -> CalendarSplitSpec | None:
    if args.split_mode != "calendar":
        return None
    spec = CalendarSplitSpec(
        train_start_ts=parse_calendar_timestamp(args.train_start_ts, "train-start-ts"),
        train_end_ts=parse_calendar_timestamp(args.train_end_ts, "train-end-ts"),
        val_start_ts=parse_calendar_timestamp(args.val_start_ts, "val-start-ts"),
        val_end_ts=parse_calendar_timestamp(args.val_end_ts, "val-end-ts"),
        holdout_start_ts=parse_calendar_timestamp(
            args.holdout_start_ts,
            "holdout-start-ts",
        ),
        holdout_end_ts=parse_calendar_timestamp(args.holdout_end_ts, "holdout-end-ts"),
    )
    if not (
        spec.train_start_ts
        < spec.train_end_ts
        <= spec.val_start_ts
        < spec.val_end_ts
        <= spec.holdout_start_ts
        < spec.holdout_end_ts
    ):
        raise ValueError(
            "calendar split must satisfy "
            "train_start < train_end <= val_start < val_end <= "
            "holdout_start < holdout_end"
        )
    return spec


def parse_calendar_timestamp(raw_value: str, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(raw_value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid timestamp: {raw_value}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid timestamp: {raw_value}")
    return timestamp


def calendar_split_metadata_fields(
    calendar_split: CalendarSplitSpec | None,
) -> dict[str, Any]:
    if calendar_split is None:
        return {}
    return {
        "calendar_interval_convention": "half_open_start_inclusive_end_exclusive",
        "calendar_train_start_ts": calendar_split.train_start_ts.isoformat(),
        "calendar_train_end_ts": calendar_split.train_end_ts.isoformat(),
        "calendar_val_start_ts": calendar_split.val_start_ts.isoformat(),
        "calendar_val_end_ts": calendar_split.val_end_ts.isoformat(),
        "calendar_holdout_start_ts": calendar_split.holdout_start_ts.isoformat(),
        "calendar_holdout_end_ts": calendar_split.holdout_end_ts.isoformat(),
    }


def resolve_threshold_bps(args: argparse.Namespace, label_mode: str) -> float:
    if label_mode == "no_trade_band":
        if args.threshold_bps is None:
            return 5.0
        return args.threshold_bps
    if args.threshold_bps not in (None, 0, 0.0):
        raise ValueError("--threshold-bps is only used for label_mode=no_trade_band")
    return 0.0


def resolve_tickers(args: argparse.Namespace, run_mode: str) -> list[str]:
    if args.tickers is not None:
        return args.tickers
    if run_mode == "smoke":
        return ["CSCO"]
    return list(DEFAULT_TICKERS)


def resolve_seeds(args: argparse.Namespace, run_mode: str) -> list[int]:
    if args.seeds is not None:
        return args.seeds
    if run_mode == "full":
        return [42, 43, 44]
    return [42]


def resolve_max_epochs(args: argparse.Namespace, run_mode: str) -> int:
    if args.max_epochs is not None:
        return args.max_epochs
    if run_mode == "full":
        return 3
    return 1


def resolve_max_rows(args: argparse.Namespace, run_mode: str) -> int | None:
    if args.max_rows_per_ticker is not None:
        return args.max_rows_per_ticker
    if run_mode == "smoke":
        return 20_000
    return None


def audit_scope_fields(run_mode: str) -> dict[str, Any]:
    if run_mode == "full":
        return {
            "claim_scope": "full_run_performance_evaluation",
            "diagnostic_scope": "full_run_candidate_evaluation",
            "diagnostic_only": False,
            "non_claim": False,
        }
    if run_mode == "manifest":
        return {
            "claim_scope": "pipeline_schema_observability_not_performance_claim",
            "diagnostic_scope": "pipeline_schema_and_data_manifest",
            "diagnostic_only": True,
            "non_claim": True,
        }
    return {
        "claim_scope": "smoke_observation_not_performance_claim",
        "diagnostic_scope": "bounded_smoke_pipeline_diagnostic",
        "diagnostic_only": True,
        "non_claim": True,
    }


def resolve_feature_set(args: argparse.Namespace, run_mode: str) -> str:
    if args.feature_set is not None:
        return args.feature_set
    return "mentor_clean_v1"


def validate_lightgbm_pm_route(
    args: argparse.Namespace,
    run_mode: str,
    feature_set_id: str,
    label_mode: str,
    threshold_bps: float,
) -> None:
    if run_mode == "full":
        raise ValueError("--model-family lightgbm does not support --full-run")
    if feature_set_id != "mentor_clean_v1":
        raise ValueError(
            "--model-family lightgbm requires --feature-set mentor_clean_v1"
        )
    if label_mode != "no_trade_band":
        raise ValueError(
            "--model-family lightgbm requires --label-mode no_trade_band"
        )
    if args.threshold_bps is None or threshold_bps != 5.0:
        raise ValueError(
            "--model-family lightgbm requires explicit --threshold-bps 5.0"
        )


def validate_torch_validation_only_pm_route(
    args: argparse.Namespace,
    run_mode: str,
    feature_set_id: str,
    label_mode: str,
    threshold_bps: float,
) -> None:
    if run_mode == "full":
        raise ValueError(
            "--model-family torch validation-only ms_dlinear_tcn does not support --full-run"
        )
    if list(args.models) != ["ms_dlinear_tcn"]:
        raise ValueError(
            "--model-family torch validation-only requires --models ms_dlinear_tcn"
        )
    if feature_set_id != "mentor_clean_v1":
        raise ValueError(
            "--model-family torch validation-only requires --feature-set mentor_clean_v1"
        )
    if label_mode != "no_trade_band":
        raise ValueError(
            "--model-family torch validation-only requires --label-mode no_trade_band"
        )
    if args.threshold_bps is None or threshold_bps != 5.0:
        raise ValueError(
            "--model-family torch validation-only requires explicit --threshold-bps 5.0"
        )


def build_run_id(run_mode: str, label_mode: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"phase1b_local_{label_mode}_{run_mode}_{timestamp}"


def protocol_metadata_fields(
    feature_set_id: str,
    label_mode: str,
    threshold_bps: float,
    cli_threshold_bps: float | None,
) -> dict[str, str]:
    decision_time_policy = "not_locked_for_current_protocol"
    if feature_set_id == "mentor_clean_v1":
        decision_time_policy = "post_bar_close_completed_bar"

    threshold_source = "not_applicable_binary_boundary"
    if label_mode == "no_trade_band":
        if threshold_bps == 5.0 and cli_threshold_bps is not None:
            threshold_source = "fixed_pre_registered_5bps"
        elif threshold_bps == 5.0:
            threshold_source = "default_fixed_5bps"
        else:
            threshold_source = "fixed_cli_bps"

    return {
        "decision_time_policy": decision_time_policy,
        "scaler_id": "standard_pooled_train_only_v1",
        "scaler_fit_scope": "pooled_train_after_per_ticker_chronological_split",
        "threshold_source": threshold_source,
    }


def label_metadata_fields(label_mode: str, threshold_bps: float) -> dict[str, Any]:
    if label_mode == "legacy_binary":
        return {
            "label_semantics": "canonical_phase1_full_binary",
            "label_formula": "label = 1 if future_avg_r > 0 else 0",
            "class_0_name": "non_up",
            "class_1_name": "up",
            "zero_return_policy": "class_0_non_up",
            "no_trade_band_enabled": False,
            "neutral_policy": "not_applicable",
        }
    return {
        "label_semantics": "phase1b_no_trade_band_diagnostic",
        "label_formula": (
            f"label = 1 if future_avg_r > {threshold_bps} bps; "
            f"label = 0 if future_avg_r < -{threshold_bps} bps"
        ),
        "class_0_name": "down",
        "class_1_name": "up",
        "zero_return_policy": "neutral_nan",
        "no_trade_band_enabled": True,
        "neutral_policy": "abs(future_avg_r) <= threshold_bps is NaN/skipped",
    }


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


def prepare_data(
    data_dir: Path,
    tickers: list[str],
    feature_set_id: str,
    feature_cols: list[str],
    candidate: CandidateSpec,
    max_rows_per_ticker: int | None,
    shuffle_train_labels: bool,
    shuffle_seed: int,
    calendar_split: CalendarSplitSpec | None = None,
    include_test_data: bool = True,
) -> PreparedData:
    if calendar_split is not None and max_rows_per_ticker is not None:
        raise ValueError("calendar split cannot be used with max_rows_per_ticker")
    data_config = DataConfig(
        tickers=tickers,
        data_dir=str(data_dir),
        label_mode=candidate.label_mode,
        threshold_bps=candidate.threshold_bps,
        feature_cols=feature_cols,
    )
    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    diagnostics_by_ticker: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        raw_df = read_stock_csv(
            data_dir,
            ticker,
            data_config.timestamp_col,
            sort_rows=feature_set_id not in {"stationary_v1_core", "mentor_clean_v1"},
        )
        if max_rows_per_ticker is not None:
            raw_df = raw_df.head(max_rows_per_ticker).copy(deep=True)
        feature_df = add_feature_set(raw_df, feature_set_id)
        diagnostics_by_ticker[f"{ticker}_feature"] = feature_diagnostics(raw_df, feature_df)
        labeled_df, label_diagnostics = make_labels(
            feature_df,
            data_config=data_config,
            candidate=candidate,
        )
        if calendar_split is None:
            if include_test_data:
                train_df, val_df, test_df = make_time_splits(
                    labeled_df,
                    train_ratio=data_config.train_ratio,
                    val_ratio=data_config.val_ratio,
                    timestamp_col=data_config.timestamp_col,
                    timezone_policy=data_config.timezone_policy,
                )
            else:
                train_df, val_df = make_train_val_time_splits(
                    labeled_df,
                    train_ratio=data_config.train_ratio,
                    val_ratio=data_config.val_ratio,
                    timestamp_col=data_config.timestamp_col,
                )
        elif include_test_data:
            train_df, val_df, test_df = make_calendar_time_splits(
                labeled_df,
                calendar_split,
                timestamp_col=data_config.timestamp_col,
                ticker=ticker,
            )
        else:
            train_df, val_df = make_calendar_train_val_splits(
                labeled_df,
                calendar_split,
                timestamp_col=data_config.timestamp_col,
                ticker=ticker,
            )
        split_frames = {
            "train": trim_for_windows(train_df, candidate, data_config.timestamp_col),
            "val": trim_for_windows(val_df, candidate, data_config.timestamp_col),
        }
        if include_test_data:
            split_frames["test"] = trim_for_windows(
                test_df,
                candidate,
                data_config.timestamp_col,
            )
        if calendar_split is not None:
            for split_name, split_df in split_frames.items():
                display_name = "holdout" if split_name == "test" else split_name
                assert_nonempty_split_frame(ticker, display_name, split_df)
        if shuffle_train_labels:
            split_frames["train"] = shuffle_labels(
                split_frames["train"],
                label_col="label",
                seed=shuffle_seed,
            )
        for split_name, split_df in split_frames.items():
            split_df["ticker"] = ticker
            diagnostics_by_ticker[f"{ticker}_{split_name}"] = split_diagnostics(split_df)
        diagnostics_by_ticker[f"{ticker}_label"] = label_diagnostics
        train_frames.append(split_frames["train"])
        val_frames.append(split_frames["val"])
        if include_test_data:
            test_frames.append(split_frames["test"])

    train_all = pd.concat(train_frames, ignore_index=True)
    val_all = pd.concat(val_frames, ignore_index=True)
    scaler = fit_scaler_on_train(train_all, feature_cols, scaler_type="standard")
    train_scaled = transform_split(train_all, scaler, feature_cols)
    val_scaled = transform_split(val_all, scaler, feature_cols)
    test_scaled = None
    if include_test_data:
        test_all = pd.concat(test_frames, ignore_index=True)
        test_scaled = transform_split(test_all, scaler, feature_cols)
    return build_prepared_data(
        train_df=train_scaled,
        val_df=val_scaled,
        test_df=test_scaled,
        feature_cols=feature_cols,
        candidate=candidate,
        timestamp_col=data_config.timestamp_col,
        diagnostics_by_ticker=diagnostics_by_ticker,
        include_test_data=include_test_data,
    )


def make_calendar_time_splits(
    df: pd.DataFrame,
    calendar_split: CalendarSplitSpec,
    timestamp_col: str,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if timestamp_col not in df.columns:
        raise ValueError(f"{ticker} missing timestamp column {timestamp_col!r}")
    timestamps = pd.to_datetime(df[timestamp_col])
    if timestamps.isna().any():
        bad_index = timestamps[timestamps.isna()].index[0]
        raise ValueError(
            f"{ticker} timestamp column {timestamp_col!r} has missing value "
            f"at row/index {bad_index!r}"
        )
    frame = df.sort_values(timestamp_col).reset_index(drop=True)
    timestamps = pd.to_datetime(frame[timestamp_col])
    train_df = calendar_interval_frame(
        frame,
        timestamps,
        calendar_split.train_start_ts,
        calendar_split.train_end_ts,
    )
    val_df = calendar_interval_frame(
        frame,
        timestamps,
        calendar_split.val_start_ts,
        calendar_split.val_end_ts,
    )
    holdout_df = calendar_interval_frame(
        frame,
        timestamps,
        calendar_split.holdout_start_ts,
        calendar_split.holdout_end_ts,
    )
    assert_nonempty_split_frame(ticker, "train", train_df)
    assert_nonempty_split_frame(ticker, "val", val_df)
    assert_nonempty_split_frame(ticker, "holdout", holdout_df)
    return train_df, val_df, holdout_df


def make_train_val_time_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    timestamp_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1, got {train_ratio + val_ratio}"
        )
    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    timestamps = pd.to_datetime(ordered[timestamp_col])
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("make_train_val_time_splits requires strictly increasing timestamps")
    row_count = len(ordered)
    train_end = int(row_count * train_ratio)
    val_end = train_end + int(row_count * val_ratio)
    train = ordered.iloc[:train_end].copy(deep=True)
    val = ordered.iloc[train_end:val_end].copy(deep=True)
    return train, val


def make_calendar_train_val_splits(
    df: pd.DataFrame,
    calendar_split: CalendarSplitSpec,
    timestamp_col: str,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if timestamp_col not in df.columns:
        raise ValueError(f"{ticker} missing timestamp column {timestamp_col!r}")
    timestamps = pd.to_datetime(df[timestamp_col])
    if timestamps.isna().any():
        bad_index = timestamps[timestamps.isna()].index[0]
        raise ValueError(
            f"{ticker} timestamp column {timestamp_col!r} has missing value "
            f"at row/index {bad_index!r}"
        )
    frame = df.sort_values(timestamp_col).reset_index(drop=True)
    timestamps = pd.to_datetime(frame[timestamp_col])
    train_df = calendar_interval_frame(
        frame,
        timestamps,
        calendar_split.train_start_ts,
        calendar_split.train_end_ts,
    )
    val_df = calendar_interval_frame(
        frame,
        timestamps,
        calendar_split.val_start_ts,
        calendar_split.val_end_ts,
    )
    assert_nonempty_split_frame(ticker, "train", train_df)
    assert_nonempty_split_frame(ticker, "val", val_df)
    return train_df, val_df


def calendar_interval_frame(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    mask = timestamps.ge(start_ts) & timestamps.lt(end_ts)
    return frame.loc[mask].copy(deep=True).reset_index(drop=True)


def assert_nonempty_split_frame(
    ticker: str,
    split_name: str,
    split_df: pd.DataFrame,
) -> None:
    if split_df.empty:
        raise ValueError(f"{ticker} calendar {split_name} split has zero rows")


def make_labels(
    feature_df: pd.DataFrame,
    data_config: DataConfig,
    candidate: CandidateSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if candidate.label_mode == "legacy_binary":
        return make_legacy_binary_labels_with_diagnostics(
            feature_df,
            price_col=data_config.price_col,
            k=candidate.label_horizon_k,
            timestamp_col=data_config.timestamp_col,
        )
    if candidate.label_mode == "no_trade_band":
        return make_no_trade_band_labels(
            feature_df,
            price_col=data_config.price_col,
            k=candidate.label_horizon_k,
            threshold_bps=candidate.threshold_bps,
            timestamp_col=data_config.timestamp_col,
        )
    raise ValueError(f"unknown label_mode: {candidate.label_mode}")


def make_legacy_binary_labels_with_diagnostics(
    df: pd.DataFrame,
    price_col: str,
    k: int,
    timestamp_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    labeled_df = make_binary_labels_from_future_avg_return(
        df,
        price_col=price_col,
        k=k,
    )
    future_avg = labeled_df["future_avg_r"]
    valid_mask = future_avg.notna()
    dates = labeled_df[timestamp_col].dt.date
    horizon_dates = dates.shift(-k)
    cross_day_mask = valid_mask & horizon_dates.notna() & (dates != horizon_dates)
    labeled_df.loc[cross_day_mask, "label"] = np.nan

    return labeled_df, {
        "n_total": int(len(labeled_df)),
        "n_tail": int(future_avg.isna().sum()),
        "n_cross_day": int(cross_day_mask.sum()),
        "n_neutral": 0,
        "n_up": int((labeled_df["label"] == 1.0).sum()),
        "n_down": int((labeled_df["label"] == 0.0).sum()),
        "n_zero_return": int((valid_mask & ~cross_day_mask & future_avg.eq(0.0)).sum()),
    }


def read_stock_csv(
    data_dir: Path,
    ticker: str,
    timestamp_col: str,
    sort_rows: bool = True,
) -> pd.DataFrame:
    path = data_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing ticker CSV: {path}")
    df = pd.read_csv(path)
    required_cols = [timestamp_col, *OHLCV_FEATURES]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"{path} missing columns: {missing_cols}")
    df = df[required_cols].copy(deep=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if not sort_rows:
        return df.reset_index(drop=True)
    return df.sort_values(timestamp_col).reset_index(drop=True)


def add_feature_set(df: pd.DataFrame, feature_set_id: str) -> pd.DataFrame:
    if feature_set_id == "ohlcv_only_v1":
        return df.copy(deep=True)
    if feature_set_id == "mentor_clean_v1":
        return add_mentor_clean_v1_features(df)
    if feature_set_id == "stationary_v1_core":
        return add_stationary_v1_core_features(df)
    if feature_set_id != "technical_v1":
        raise ValueError(f"unknown feature_set_id: {feature_set_id}")

    result = df.copy(deep=True)
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    rolling_mean = close.rolling(window=20, min_periods=20).mean()
    rolling_std = close.rolling(window=20, min_periods=20).std()
    upper_band = rolling_mean + 2.0 * rolling_std
    lower_band = rolling_mean - 2.0 * rolling_std
    result["bb_pctb"] = (close - lower_band) / (upper_band - lower_band)
    result["rolling_std_20"] = close.pct_change().rolling(window=20, min_periods=20).std()

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    result["obv_roc"] = obv.diff(5) / volume.rolling(window=5, min_periods=5).sum()

    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=list(TECHNICAL_FEATURES)).reset_index(drop=True)


def add_mentor_clean_v1_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy(deep=True)
    _validate_mentor_clean_v1_raw_input(result)
    timestamps = pd.to_datetime(result["timestamp"])
    trading_date = timestamps.dt.date
    group_keys: list[pd.Series] = [trading_date]
    if "ticker" in result.columns:
        group_keys = [result["ticker"], trading_date]

    open_price = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    log_close = np.log(close)
    log_volume = np.log1p(volume)

    grouped_log_close = log_close.groupby(group_keys, sort=False)
    result["log_return"] = grouped_log_close.transform(lambda values: values - values.shift(1))
    result["close_to_open_return"] = close / open_price - 1.0
    result["high_low_range"] = np.log(high / low)
    result["rolling_volatility_20"] = result["log_return"].groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=20, min_periods=20).std()
    )
    grouped_log_volume = log_volume.groupby(group_keys, sort=False)
    result["normalized_volume_20"] = grouped_log_volume.transform(
        lambda values: values - values.rolling(window=20, min_periods=20).mean()
    )
    result["rsi_14"] = _rsi_14_by_group(close, group_keys)
    result["bollinger_pctb"] = _bollinger_pctb_by_group(close, group_keys)
    result["normalized_macd_hist"] = _normalized_macd_hist_by_group(close, group_keys)

    minute_of_day = timestamps.dt.hour * 60 + timestamps.dt.minute
    time_angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    result["time_of_day_sin"] = np.sin(time_angle)
    result["time_of_day_cos"] = np.cos(time_angle)

    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=list(MENTOR_CLEAN_V1_FEATURES)).reset_index(drop=True)


def _rsi_14_by_group(close: pd.Series, group_keys: list[pd.Series]) -> pd.Series:
    delta = close.groupby(group_keys, sort=False).transform(lambda values: values.diff())
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=14, min_periods=14).mean()
    )
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return rsi


def _bollinger_pctb_by_group(close: pd.Series, group_keys: list[pd.Series]) -> pd.Series:
    rolling_mean = close.groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=20, min_periods=20).mean()
    )
    rolling_std = close.groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=20, min_periods=20).std()
    )
    upper_band = rolling_mean + 2.0 * rolling_std
    lower_band = rolling_mean - 2.0 * rolling_std
    band_width = upper_band - lower_band
    return pd.Series(
        np.where(band_width.to_numpy() == 0.0, 0.5, (close - lower_band) / band_width),
        index=close.index,
    )


def _normalized_macd_hist_by_group(close: pd.Series, group_keys: list[pd.Series]) -> pd.Series:
    return close.groupby(group_keys, sort=False).transform(_normalized_macd_hist_one_group)


def _normalized_macd_hist_one_group(close: pd.Series) -> pd.Series:
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return (macd - macd_signal) / close


def _validate_mentor_clean_v1_raw_input(df: pd.DataFrame) -> None:
    required_cols = ["timestamp", *OHLCV_FEATURES]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"mentor_clean_v1 missing columns: {missing_cols}")

    timestamps = pd.to_datetime(df["timestamp"])
    if timestamps.isna().any():
        bad_index = timestamps[timestamps.isna()].index[0]
        raise ValueError(f"mentor_clean_v1 timestamp is missing at row/index {bad_index}")
    if "ticker" in df.columns and df["ticker"].isna().any():
        bad_index = df.index[df["ticker"].isna()][0]
        raise ValueError(f"mentor_clean_v1 ticker is missing at row/index {bad_index}")
    _validate_mentor_clean_v1_timestamp_order(df, timestamps)

    numeric = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]
    }
    for column in ["open", "high", "low", "close"]:
        values = numeric[column]
        invalid = ~np.isfinite(values.to_numpy(dtype=float)) | (values <= 0.0)
        if invalid.any():
            bad_index = invalid[invalid].index[0]
            raise ValueError(
                f"mentor_clean_v1 invalid {column}: must be finite and > 0 "
                f"at row/index {bad_index}"
            )
    volume = numeric["volume"]
    invalid_volume = ~np.isfinite(volume.to_numpy(dtype=float)) | (volume < 0.0)
    if invalid_volume.any():
        bad_index = invalid_volume[invalid_volume].index[0]
        raise ValueError(
            "mentor_clean_v1 invalid volume: must be finite and >= 0 "
            f"at row/index {bad_index}"
        )

    high = numeric["high"]
    low = numeric["low"]
    open_price = numeric["open"]
    close = numeric["close"]
    invalid_high_low = high < low
    if invalid_high_low.any():
        bad_index = invalid_high_low[invalid_high_low].index[0]
        raise ValueError(f"mentor_clean_v1 invalid high/low: high must be >= low at row/index {bad_index}")
    invalid_open = (open_price > high) | (open_price < low)
    if invalid_open.any():
        bad_index = invalid_open[invalid_open].index[0]
        raise ValueError(
            "mentor_clean_v1 invalid open: must be between low and high "
            f"at row/index {bad_index}"
        )
    invalid_close = (close > high) | (close < low)
    if invalid_close.any():
        bad_index = invalid_close[invalid_close].index[0]
        raise ValueError(
            "mentor_clean_v1 invalid close: must be between low and high "
            f"at row/index {bad_index}"
        )


def _validate_mentor_clean_v1_timestamp_order(
    df: pd.DataFrame,
    timestamps: pd.Series,
) -> None:
    grouped_timestamps = pd.DataFrame({"timestamp": timestamps}, index=df.index)
    trading_date = timestamps.dt.date
    if "ticker" in df.columns:
        for ticker, group in grouped_timestamps.groupby(df["ticker"], sort=False):
            _validate_mentor_timestamp_sequence(
                group["timestamp"],
                f"ticker={ticker}",
            )
        group_keys = [df["ticker"], trading_date]
    else:
        _validate_mentor_timestamp_sequence(
            grouped_timestamps["timestamp"],
            "raw dataframe",
        )
        group_keys = [trading_date]

    for group_key, group in grouped_timestamps.groupby(group_keys, sort=False):
        _validate_mentor_timestamp_sequence(
            group["timestamp"],
            _stationary_timestamp_group_name(group_key),
        )


def _validate_mentor_timestamp_sequence(
    ordered: pd.Series,
    scope_name: str,
) -> None:
    duplicated = ordered.duplicated(keep=False)
    if duplicated.any():
        bad_index = duplicated[duplicated].index[0]
        raise ValueError(
            "mentor_clean_v1 duplicate timestamp within "
            f"{scope_name} at row/index {bad_index}"
        )

    invalid_order = ordered <= ordered.shift(1)
    if invalid_order.any():
        bad_index = invalid_order[invalid_order].index[0]
        position = ordered.index.get_loc(bad_index)
        previous_index = ordered.index[position - 1]
        raise ValueError(
            "mentor_clean_v1 timestamp must be strict monotonically "
            f"increasing within {scope_name} at row/index {bad_index}; "
            f"current timestamp {ordered.loc[bad_index]}, "
            f"previous timestamp {ordered.loc[previous_index]}"
        )


def add_stationary_v1_core_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy(deep=True)
    _validate_stationary_v1_core_raw_input(result)
    timestamps = pd.to_datetime(result["timestamp"])
    trading_date = timestamps.dt.date
    group_keys: list[pd.Series] = [trading_date]
    if "ticker" in result.columns:
        group_keys = [result["ticker"], trading_date]

    open_price = result["open"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    close = result["close"].astype(float)
    volume = result["volume"].astype(float)
    log_close = np.log(close.where(close > 0.0))
    log_volume = np.log1p(volume.where(volume >= 0.0))

    grouped_log_close = log_close.groupby(group_keys, sort=False)
    result["log_ret_1"] = grouped_log_close.transform(lambda values: values - values.shift(1))
    result["log_ret_3"] = grouped_log_close.transform(lambda values: values - values.shift(3))
    result["log_ret_6"] = grouped_log_close.transform(lambda values: values - values.shift(6))
    result["oc_log_ret"] = np.log(close.where(close > 0.0) / open_price.where(open_price > 0.0))
    result["hl_log_range"] = np.log(high.where(high > 0.0) / low.where(low > 0.0))

    price_range = high - low
    body = close - open_price
    result["body_to_range"] = np.where(price_range.to_numpy() == 0.0, 0.0, body / price_range)
    result["rv_6"] = result["log_ret_1"].groupby(group_keys, sort=False).transform(
        lambda values: values.rolling(window=6, min_periods=6).std()
    )
    result["log_volume_chg_1"] = log_volume.groupby(group_keys, sort=False).transform(
        lambda values: values - values.shift(1)
    )

    warmup_mask = result.groupby(group_keys, sort=False).cumcount() < 6
    feature_values = result.loc[:, STATIONARY_V1_CORE_FEATURES].to_numpy(dtype=float)
    nonfinite_mask = ~np.isfinite(feature_values).all(axis=1)
    invalid_mask = pd.Series(nonfinite_mask, index=result.index) & ~warmup_mask
    if invalid_mask.any():
        bad_index = invalid_mask[invalid_mask].index[0]
        bad_features = [
            feature
            for feature in STATIONARY_V1_CORE_FEATURES
            if not np.isfinite(float(result.loc[bad_index, feature]))
        ]
        raise ValueError(
            "stationary_v1_core produced non-finite feature values outside "
            f"expected lag/rolling warmup at row/index {bad_index}: {bad_features}"
        )
    return result.loc[~warmup_mask].reset_index(drop=True)


def _validate_stationary_v1_core_raw_input(df: pd.DataFrame) -> None:
    required_cols = ["timestamp", *OHLCV_FEATURES]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"stationary_v1_core missing columns: {missing_cols}")

    timestamps = pd.to_datetime(df["timestamp"])
    if timestamps.isna().any():
        bad_index = timestamps[timestamps.isna()].index[0]
        raise ValueError(f"stationary_v1_core timestamp is missing at row/index {bad_index}")
    if "ticker" in df.columns and df["ticker"].isna().any():
        bad_index = df.index[df["ticker"].isna()][0]
        raise ValueError(f"stationary_v1_core ticker is missing at row/index {bad_index}")
    _validate_stationary_v1_core_timestamp_order(df, timestamps)

    numeric = {
        column: pd.to_numeric(df[column], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]
    }
    for column in ["open", "high", "low", "close"]:
        values = numeric[column]
        invalid = ~np.isfinite(values.to_numpy(dtype=float)) | (values <= 0.0)
        if invalid.any():
            bad_index = invalid[invalid].index[0]
            raise ValueError(
                f"stationary_v1_core invalid {column}: must be finite and > 0 "
                f"at row/index {bad_index}"
            )
    volume = numeric["volume"]
    invalid_volume = ~np.isfinite(volume.to_numpy(dtype=float)) | (volume < 0.0)
    if invalid_volume.any():
        bad_index = invalid_volume[invalid_volume].index[0]
        raise ValueError(
            "stationary_v1_core invalid volume: must be finite and >= 0 "
            f"at row/index {bad_index}"
        )

    high = numeric["high"]
    low = numeric["low"]
    open_price = numeric["open"]
    close = numeric["close"]
    invalid_high_low = high < low
    if invalid_high_low.any():
        bad_index = invalid_high_low[invalid_high_low].index[0]
        raise ValueError(
            f"stationary_v1_core invalid high/low: high must be >= low at row/index {bad_index}"
        )
    invalid_open = (open_price > high) | (open_price < low)
    if invalid_open.any():
        bad_index = invalid_open[invalid_open].index[0]
        raise ValueError(
            "stationary_v1_core invalid open: must be between low and high "
            f"at row/index {bad_index}"
        )
    invalid_close = (close > high) | (close < low)
    if invalid_close.any():
        bad_index = invalid_close[invalid_close].index[0]
        raise ValueError(
            "stationary_v1_core invalid close: must be between low and high "
            f"at row/index {bad_index}"
        )


def _validate_stationary_v1_core_timestamp_order(
    df: pd.DataFrame,
    timestamps: pd.Series,
) -> None:
    grouped_timestamps = pd.DataFrame({"timestamp": timestamps}, index=df.index)
    trading_date = timestamps.dt.date
    if "ticker" in df.columns:
        for ticker, group in grouped_timestamps.groupby(df["ticker"], sort=False):
            _validate_stationary_timestamp_sequence(
                group["timestamp"],
                f"ticker={ticker}",
            )
        group_keys = [df["ticker"], trading_date]
    else:
        _validate_stationary_timestamp_sequence(
            grouped_timestamps["timestamp"],
            "raw dataframe",
        )
        group_keys = [trading_date]

    for group_key, group in grouped_timestamps.groupby(group_keys, sort=False):
        _validate_stationary_timestamp_sequence(
            group["timestamp"],
            _stationary_timestamp_group_name(group_key),
        )


def _validate_stationary_timestamp_sequence(
    ordered: pd.Series,
    scope_name: str,
) -> None:
    duplicated = ordered.duplicated(keep=False)
    if duplicated.any():
        bad_index = duplicated[duplicated].index[0]
        raise ValueError(
            "stationary_v1_core duplicate timestamp within "
            f"{scope_name} at row/index {bad_index}"
        )

    invalid_order = ordered <= ordered.shift(1)
    if invalid_order.any():
        bad_index = invalid_order[invalid_order].index[0]
        position = ordered.index.get_loc(bad_index)
        previous_index = ordered.index[position - 1]
        raise ValueError(
            "stationary_v1_core timestamp must be strict monotonically "
            f"increasing within {scope_name} at row/index {bad_index}; "
            f"current timestamp {ordered.loc[bad_index]}, "
            f"previous timestamp {ordered.loc[previous_index]}"
        )


def _stationary_timestamp_group_name(group_key: Any) -> str:
    if isinstance(group_key, tuple) and len(group_key) == 2:
        return f"ticker={group_key[0]}, trading_date={group_key[1]}"
    if isinstance(group_key, tuple) and len(group_key) == 1:
        return f"trading_date={group_key[0]}"
    return f"trading_date={group_key}"


def feature_diagnostics(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> dict[str, Any]:
    n_raw = int(len(raw_df))
    n_feature = int(len(feature_df))
    n_dropped = n_raw - n_feature
    return {
        "n_raw_rows": n_raw,
        "n_feature_rows": n_feature,
        "feature_drop_count": n_dropped,
        "feature_drop_pct": n_dropped / n_raw if n_raw else None,
    }


def trim_for_windows(
    df: pd.DataFrame,
    candidate: CandidateSpec,
    timestamp_col: str,
) -> pd.DataFrame:
    return trim_labels_at_split_boundary(
        df,
        label_horizon_k=candidate.label_horizon_k,
        timestamp_col=timestamp_col,
    )


def split_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    labels = df["label"]
    retained = labels.notna()
    return {
        "n_rows": int(len(df)),
        "n_retained_labels": int(retained.sum()),
        "n_nan_labels": int(labels.isna().sum()),
        "up_pct": safe_mean(labels.loc[retained].to_numpy(dtype=float)),
    }


def shuffle_labels(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    result = df.copy(deep=True)
    label_mask = result[label_col].notna()
    labels = result.loc[label_mask, label_col].to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    result.loc[label_mask, label_col] = labels
    return result


def build_prepared_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: list[str],
    candidate: CandidateSpec,
    timestamp_col: str,
    diagnostics_by_ticker: dict[str, dict[str, Any]],
    include_test_data: bool = True,
) -> PreparedData:
    train_dataset = make_dataset(train_df, feature_cols, candidate, timestamp_col)
    val_dataset = make_dataset(val_df, feature_cols, candidate, timestamp_col)
    assert_nonempty_dataset("train", train_dataset)
    assert_nonempty_dataset("val", val_dataset)

    val_datasets_by_ticker = {}
    test_datasets_by_ticker = {}
    y_train_by_ticker = {}
    y_val_by_ticker = {}
    y_test_by_ticker = {}
    tickers = sorted(train_df["ticker"].unique())
    for ticker in tickers:
        train_ticker_df = train_df.loc[train_df["ticker"] == ticker].copy(deep=True)
        val_ticker_df = val_df.loc[val_df["ticker"] == ticker].copy(deep=True)
        train_ticker_dataset = make_dataset(train_ticker_df, feature_cols, candidate, timestamp_col)
        val_ticker_dataset = make_dataset(val_ticker_df, feature_cols, candidate, timestamp_col)
        assert_nonempty_dataset(f"{ticker} train", train_ticker_dataset)
        assert_nonempty_dataset(f"{ticker} val", val_ticker_dataset)
        y_train_by_ticker[ticker] = dataset_labels(train_ticker_dataset)
        y_val_by_ticker[ticker] = dataset_labels(val_ticker_dataset)
        val_datasets_by_ticker[ticker] = val_ticker_dataset

    test_dataset = None
    y_test = None
    if include_test_data:
        if test_df is None:
            raise ValueError("test data is required for non-validation evaluation")
        test_dataset = make_dataset(test_df, feature_cols, candidate, timestamp_col)
        assert_nonempty_dataset("test", test_dataset)
        for ticker in tickers:
            test_ticker_df = test_df.loc[test_df["ticker"] == ticker].copy(deep=True)
            test_ticker_dataset = make_dataset(
                test_ticker_df,
                feature_cols,
                candidate,
                timestamp_col,
            )
            assert_nonempty_dataset(f"{ticker} test", test_ticker_dataset)
            y_test_by_ticker[ticker] = dataset_labels(test_ticker_dataset)
            test_datasets_by_ticker[ticker] = test_ticker_dataset
        y_test = dataset_labels(test_dataset)

    return PreparedData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        val_datasets_by_ticker=val_datasets_by_ticker,
        test_datasets_by_ticker=test_datasets_by_ticker,
        y_train=dataset_labels(train_dataset),
        y_val=dataset_labels(val_dataset),
        y_test=y_test,
        y_train_by_ticker=y_train_by_ticker,
        y_val_by_ticker=y_val_by_ticker,
        y_test_by_ticker=y_test_by_ticker,
        diagnostics_by_ticker=diagnostics_by_ticker,
    )


def make_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    candidate: CandidateSpec,
    timestamp_col: str,
) -> WindowedClassificationDataset:
    return WindowedClassificationDataset(
        df=df,
        feature_cols=feature_cols,
        label_col="label",
        ticker_col="ticker",
        timestamp_col=timestamp_col,
        window_size=candidate.window_size,
        label_horizon_k=candidate.label_horizon_k,
        stride=1,
    )


def assert_nonempty_dataset(name: str, dataset: WindowedClassificationDataset) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset has zero valid windows")


def dataset_labels(dataset: WindowedClassificationDataset) -> np.ndarray:
    labels = [int(y.item()) for _, y in dataset]
    return np.asarray(labels, dtype=int)


def dataset_features(
    dataset: WindowedClassificationDataset,
    feature_view: str,
) -> np.ndarray:
    features = []
    for x, _ in dataset:
        values = x.detach().cpu().numpy()
        if values.ndim != 2:
            raise ValueError(f"expected 2D window features, got shape {values.shape}")
        if feature_view == "last_step":
            features.append(values[-1, :])
        elif feature_view == "flatten_window":
            features.append(values.reshape(-1))
        else:
            raise ValueError(f"unknown feature_view: {feature_view}")
    return np.asarray(features, dtype=np.float64)


def safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def make_loader(
    dataset: WindowedClassificationDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def run_sklearn_logreg_baseline(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    validation_only_report: bool = False,
    validation_only_per_ticker: bool = False,
    c_grid: tuple[float, ...] = SKLEARN_LOGREG_C_GRID,
    class_weights: tuple[str | None, ...] = SKLEARN_LOGREG_CLASS_WEIGHTS,
) -> list[dict[str, Any]]:
    x_train = dataset_features(prepared.train_dataset, feature_view)
    x_val = dataset_features(prepared.val_dataset, feature_view)
    val_baseline_metrics = compute_baselines(prepared.y_train, prepared.y_val)

    candidates = []
    for c_value in c_grid:
        for class_weight in class_weights:
            model, fit_info = fit_sklearn_logreg(
                x_train,
                prepared.y_train,
                c_value=c_value,
                class_weight=class_weight,
            )
            val_pred = model.predict(x_val)
            val_metrics = compute_classification_metrics(prepared.y_val, val_pred)
            candidates.append(
                {
                    "C": c_value,
                    "class_weight": class_weight,
                    "model": model,
                    "fit_info": fit_info,
                    "val_metrics": val_metrics,
                }
            )

    best = select_sklearn_logreg_candidate(candidates)
    val_metrics = best["val_metrics"]
    val_delta = (
        val_metrics["macro_f1"]
        - val_baseline_metrics["dummy_stratified_macro_f1_mean"]
    )
    if validation_only_report:
        rows = [
            sklearn_validation_only_result_row(
                metadata=metadata,
                candidate=candidate,
                prepared=prepared,
                feature_view=feature_view,
                best=best,
                val_metrics=val_metrics,
                val_baseline_metrics=val_baseline_metrics,
                val_delta=val_delta,
            )
        ]
        if validation_only_per_ticker:
            rows.extend(
                sklearn_validation_only_ticker_rows(
                    metadata=metadata,
                    candidate=candidate,
                    prepared=prepared,
                    feature_view=feature_view,
                    best=best,
                )
            )
        return rows

    require_test_data(prepared, require_per_ticker=False)
    x_test = dataset_features(prepared.test_dataset, feature_view)
    test_pred = best["model"].predict(x_test)
    test_metrics = compute_classification_metrics(prepared.y_test, test_pred)
    test_baseline_metrics = compute_baselines(prepared.y_train, prepared.y_test)
    test_delta = (
        test_metrics["macro_f1"]
        - test_baseline_metrics["dummy_stratified_macro_f1_mean"]
    )

    return [
        {
            **base_result_fields(metadata, candidate),
            "model_name": "sklearn_logreg_l2",
            "model_family": "sklearn_logreg",
            "ticker": "pooled",
            "seed": None,
            "split": "test",
            "feature_view": feature_view,
            "C": float(best["C"]),
            "class_weight": best["class_weight"],
            "solver": "lbfgs",
            "max_iter": 500,
            **secondary_baseline_scope_fields("pooled"),
            "n_train_windows": int(len(prepared.train_dataset)),
            "n_val_windows": int(len(prepared.val_dataset)),
            "n_test_windows": int(len(prepared.test_dataset)),
            "train_up_pct": safe_mean(prepared.y_train.astype(float)),
            "val_up_pct": safe_mean(prepared.y_val.astype(float)),
            "test_up_pct": safe_mean(prepared.y_test.astype(float)),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_delta_macro_f1_vs_dummy": float(val_delta),
            "model_macro_f1": float(test_metrics["macro_f1"]),
            "model_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "model_precision_macro": float(test_metrics["precision_macro"]),
            "model_recall_macro": float(test_metrics["recall_macro"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "test_precision_macro": float(test_metrics["precision_macro"]),
            "test_recall_macro": float(test_metrics["recall_macro"]),
            **test_baseline_metrics,
            "delta_macro_f1_vs_dummy": float(test_delta),
            **ticker_baseline_fields(test_baseline_metrics),
            "delta_macro_f1_vs_ticker_dummy": float(test_delta),
            **scope_diagnostics_fields("pooled", prepared, metadata),
            "confusion_matrix_labels": json.dumps([0, 1]),
            "confusion_matrix": json.dumps(test_metrics["confusion_matrix"].tolist()),
            "classification_report": json.dumps(test_metrics["classification_report"]),
            "converged": bool(best["fit_info"]["converged"]),
            "n_iter": int(best["fit_info"]["n_iter"]),
            "warnings": json.dumps(best["fit_info"]["warnings"]),
            "best_epoch": None,
            "best_val_macro_f1": float(val_metrics["macro_f1"]),
            "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
                "dummy_stratified_macro_f1_mean"
            ],
            "training_time_seconds": float(best["fit_info"]["fit_time_seconds"]),
            "suspicious_status": "ok",
        }
    ]


def sklearn_validation_only_result_row(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    best: dict[str, Any],
    val_metrics: dict[str, Any],
    val_baseline_metrics: dict[str, Any],
    val_delta: float,
) -> dict[str, Any]:
    return {
        **base_result_fields(metadata, candidate),
        **validation_only_report_fields(),
        "model_name": "sklearn_logreg_l2",
        "model_family": "sklearn_logreg",
        "ticker": "pooled",
        "seed": None,
        "split": "validation",
        "feature_view": feature_view,
        "C": float(best["C"]),
        "class_weight": best["class_weight"],
        "solver": "lbfgs",
        "max_iter": 500,
        **secondary_baseline_scope_fields("pooled"),
        "n_train_windows": int(len(prepared.train_dataset)),
        "n_val_windows": int(len(prepared.val_dataset)),
        "train_up_pct": safe_mean(prepared.y_train.astype(float)),
        "val_up_pct": safe_mean(prepared.y_val.astype(float)),
        **validation_only_coverage_fields(prepared.y_val),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        "val_delta_macro_f1_vs_dummy": float(val_delta),
        **scope_diagnostics_fields("pooled", prepared, metadata),
        "converged": bool(best["fit_info"]["converged"]),
        "n_iter": int(best["fit_info"]["n_iter"]),
        "warnings": json.dumps(best["fit_info"]["warnings"]),
        "best_epoch": None,
        "best_val_macro_f1": float(val_metrics["macro_f1"]),
        "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
            "dummy_stratified_macro_f1_mean"
        ],
        "training_time_seconds": float(best["fit_info"]["fit_time_seconds"]),
        "suspicious_status": "ok",
    }


def sklearn_validation_only_ticker_rows(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    best: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for ticker in sorted(prepared.val_datasets_by_ticker):
        val_dataset = prepared.val_datasets_by_ticker[ticker]
        if len(val_dataset) == 0:
            raise ValueError(f"{ticker} validation dataset is empty")
        y_val = prepared.y_val_by_ticker[ticker]
        x_val = dataset_features(val_dataset, feature_view)
        val_pred = best["model"].predict(x_val)
        val_metrics = compute_classification_metrics(y_val, val_pred)
        val_baseline_metrics = compute_baselines(
            prepared.y_train_by_ticker[ticker],
            y_val,
        )
        val_delta = (
            val_metrics["macro_f1"]
            - val_baseline_metrics["dummy_stratified_macro_f1_mean"]
        )
        rows.append(
            {
                **base_result_fields(metadata, candidate),
                **validation_only_report_fields(),
                "model_name": "sklearn_logreg_l2",
                "model_family": "sklearn_logreg",
                "ticker": ticker,
                "seed": None,
                "split": "validation",
                "feature_view": feature_view,
                "C": float(best["C"]),
                "class_weight": best["class_weight"],
                "solver": "lbfgs",
                "max_iter": 500,
                **secondary_baseline_scope_fields(ticker),
                "n_train_windows": int(len(prepared.train_dataset)),
                "n_val_windows": int(len(val_dataset)),
                "train_up_pct": safe_mean(prepared.y_train.astype(float)),
                "val_up_pct": safe_mean(y_val.astype(float)),
                **validation_only_coverage_fields(y_val),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_delta_macro_f1_vs_dummy": float(val_delta),
                **scope_diagnostics_fields(ticker, prepared, metadata),
                "converged": bool(best["fit_info"]["converged"]),
                "n_iter": int(best["fit_info"]["n_iter"]),
                "warnings": json.dumps(best["fit_info"]["warnings"]),
                "best_epoch": None,
                "best_val_macro_f1": float(best["val_metrics"]["macro_f1"]),
                "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
                    "dummy_stratified_macro_f1_mean"
                ],
                "training_time_seconds": float(best["fit_info"]["fit_time_seconds"]),
                "suspicious_status": "ok",
            }
        )
    return rows


def fit_sklearn_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    class_weight: str | None,
) -> tuple[LogisticRegression, dict[str, Any]]:
    model = LogisticRegression(
        penalty="l2",
        C=c_value,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=500,
    )
    started_at = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_time_seconds = time.perf_counter() - started_at
    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    return model, {
        "converged": len(convergence_warnings) == 0,
        "n_iter": int(np.max(model.n_iter_)),
        "warnings": convergence_warnings,
        "fit_time_seconds": fit_time_seconds,
    }


def select_sklearn_logreg_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no sklearn logistic regression candidates were evaluated")
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate["val_metrics"]["macro_f1"]),
            float(candidate["C"]),
            0 if candidate["class_weight"] is None else 1,
        ),
    )


def run_lightgbm_validation_only_baseline(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    validation_only_per_ticker: bool = False,
) -> list[dict[str, Any]]:
    x_train = dataset_features(prepared.train_dataset, feature_view)
    x_val = dataset_features(prepared.val_dataset, feature_view)
    model, fit_info = fit_lightgbm_classifier(x_train, prepared.y_train)
    val_pred = model.predict(x_val)
    val_metrics = compute_classification_metrics(prepared.y_val, val_pred)
    val_baseline_metrics = compute_baselines(prepared.y_train, prepared.y_val)
    val_delta = (
        val_metrics["macro_f1"]
        - val_baseline_metrics["dummy_stratified_macro_f1_mean"]
    )
    rows = [
        lightgbm_validation_only_result_row(
            metadata=metadata,
            candidate=candidate,
            prepared=prepared,
            feature_view=feature_view,
            fit_info=fit_info,
            val_metrics=val_metrics,
            val_baseline_metrics=val_baseline_metrics,
            val_delta=val_delta,
        )
    ]
    if validation_only_per_ticker:
        rows.extend(
            lightgbm_validation_only_ticker_rows(
                metadata=metadata,
                candidate=candidate,
                prepared=prepared,
                feature_view=feature_view,
                model=model,
                fit_info=fit_info,
            )
        )
    return rows


def lightgbm_validation_only_result_row(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    fit_info: dict[str, Any],
    val_metrics: dict[str, Any],
    val_baseline_metrics: dict[str, Any],
    val_delta: float,
) -> dict[str, Any]:
    return {
        **base_result_fields(metadata, candidate),
        **validation_only_report_fields(),
        "model_name": LIGHTGBM_MODEL_NAME,
        "model_family": "lightgbm",
        "ticker": "pooled",
        "seed": None,
        "split": "validation",
        "feature_view": feature_view,
        **lightgbm_result_fields(fit_info),
        **secondary_baseline_scope_fields("pooled"),
        "n_train_windows": int(len(prepared.train_dataset)),
        "n_val_windows": int(len(prepared.val_dataset)),
        "train_up_pct": safe_mean(prepared.y_train.astype(float)),
        "val_up_pct": safe_mean(prepared.y_val.astype(float)),
        **validation_only_coverage_fields(prepared.y_val),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        "val_delta_macro_f1_vs_dummy": float(val_delta),
        **scope_diagnostics_fields("pooled", prepared, metadata),
        "best_epoch": None,
        "best_val_macro_f1": float(val_metrics["macro_f1"]),
        "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
            "dummy_stratified_macro_f1_mean"
        ],
        "training_time_seconds": float(fit_info["fit_time_seconds"]),
        "suspicious_status": "ok",
    }


def lightgbm_validation_only_ticker_rows(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_view: str,
    model: Any,
    fit_info: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for ticker in sorted(prepared.val_datasets_by_ticker):
        val_dataset = prepared.val_datasets_by_ticker[ticker]
        if len(val_dataset) == 0:
            raise ValueError(f"{ticker} validation dataset is empty")
        y_val = prepared.y_val_by_ticker[ticker]
        x_val = dataset_features(val_dataset, feature_view)
        val_pred = model.predict(x_val)
        val_metrics = compute_classification_metrics(y_val, val_pred)
        val_baseline_metrics = compute_baselines(
            prepared.y_train_by_ticker[ticker],
            y_val,
        )
        val_delta = (
            val_metrics["macro_f1"]
            - val_baseline_metrics["dummy_stratified_macro_f1_mean"]
        )
        rows.append(
            {
                **base_result_fields(metadata, candidate),
                **validation_only_report_fields(),
                "model_name": LIGHTGBM_MODEL_NAME,
                "model_family": "lightgbm",
                "ticker": ticker,
                "seed": None,
                "split": "validation",
                "feature_view": feature_view,
                **lightgbm_result_fields(fit_info),
                **secondary_baseline_scope_fields(ticker),
                "n_train_windows": int(len(prepared.train_dataset)),
                "n_val_windows": int(len(val_dataset)),
                "train_up_pct": safe_mean(prepared.y_train.astype(float)),
                "val_up_pct": safe_mean(y_val.astype(float)),
                **validation_only_coverage_fields(y_val),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_delta_macro_f1_vs_dummy": float(val_delta),
                **scope_diagnostics_fields(ticker, prepared, metadata),
                "best_epoch": None,
                "best_val_macro_f1": float(val_metrics["macro_f1"]),
                "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
                    "dummy_stratified_macro_f1_mean"
                ],
                "training_time_seconds": float(fit_info["fit_time_seconds"]),
                "suspicious_status": "ok",
            }
        )
    return rows


def lightgbm_result_fields(fit_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": fit_info["objective"],
        "n_estimators": int(fit_info["n_estimators"]),
        "learning_rate": float(fit_info["learning_rate"]),
        "num_leaves": int(fit_info["num_leaves"]),
        "random_state": int(fit_info["random_state"]),
        "n_jobs": int(fit_info["n_jobs"]),
        "verbosity": int(fit_info["verbosity"]),
        "warnings": json.dumps(fit_info["warnings"]),
    }


def fit_lightgbm_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    lightgbm = load_lightgbm_module()
    params = default_lightgbm_params()
    model = lightgbm.LGBMClassifier(**params)
    started_at = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time_seconds = time.perf_counter() - started_at
    return model, {
        **params,
        "warnings": [],
        "fit_time_seconds": fit_time_seconds,
    }


def default_lightgbm_params() -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": -1,
    }


def load_lightgbm_module() -> Any:
    try:
        return __import__("lightgbm")
    except ModuleNotFoundError as exc:
        if exc.name != "lightgbm":
            raise
        raise ImportError(
            "LightGBM support requires the pinned dependency lightgbm==4.6.0. "
            "Install the project requirements before running --model-family lightgbm."
        ) from exc


def run_model_once(
    model_name: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    early_stop_patience: int,
    output_dir: Path,
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
    feature_cols: list[str],
    validation_only_report: bool = False,
    validation_only_per_ticker: bool = False,
) -> list[dict[str, Any]]:
    if not validation_only_report:
        require_test_data(prepared)
    seed_everything(seed, deterministic=False)
    model = build_model(model_name, seq_len=candidate.window_size, input_size=len(feature_cols))
    train_loader = make_loader(prepared.train_dataset, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(prepared.val_dataset, batch_size, shuffle=False, seed=seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    run_dir = output_dir / metadata["run_id"] / f"{model_name}_seed_{seed}"
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        device="cpu",
        checkpoint_dir=str(run_dir),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=early_stop_patience,
        grad_clip=1.0,
        verbose=False,
    )
    started_at = time.perf_counter()
    history = trainer.fit(train_loader, val_loader, num_epochs=max_epochs)
    training_time_seconds = time.perf_counter() - started_at
    load_checkpoint(str(run_dir / "best.pt"), model=model, device="cpu", weights_only=True)

    if validation_only_report:
        rows = [
            evaluate_validation_scope(
                scope_name="pooled",
                model=model,
                dataset=prepared.val_dataset,
                y_train=prepared.y_train,
                y_eval=prepared.y_val,
                batch_size=batch_size,
                seed=seed,
                metadata=metadata,
                candidate=candidate,
                model_name=model_name,
                history=history,
                training_time_seconds=training_time_seconds,
                prepared=prepared,
            )
        ]
        if validation_only_per_ticker:
            for ticker, dataset in prepared.val_datasets_by_ticker.items():
                rows.append(
                    evaluate_validation_scope(
                        scope_name=ticker,
                        model=model,
                        dataset=dataset,
                        y_train=prepared.y_train_by_ticker[ticker],
                        y_eval=prepared.y_val_by_ticker[ticker],
                        batch_size=batch_size,
                        seed=seed,
                        metadata=metadata,
                        candidate=candidate,
                        model_name=model_name,
                        history=history,
                        training_time_seconds=training_time_seconds,
                        prepared=prepared,
                    )
                )
        return rows

    rows = [
        evaluate_scope(
            scope_name="pooled",
            model=model,
            dataset=prepared.test_dataset,
            y_train=prepared.y_train,
            y_eval=prepared.y_test,
            batch_size=batch_size,
            seed=seed,
            metadata=metadata,
            candidate=candidate,
            model_name=model_name,
            history=history,
            training_time_seconds=training_time_seconds,
            prepared=prepared,
        )
    ]
    for ticker, dataset in prepared.test_datasets_by_ticker.items():
        rows.append(
            evaluate_scope(
                scope_name=ticker,
                model=model,
                dataset=dataset,
                y_train=prepared.y_train,
                y_eval=prepared.y_test_by_ticker[ticker],
                batch_size=batch_size,
                seed=seed,
                metadata=metadata,
                candidate=candidate,
                model_name=model_name,
                history=history,
                training_time_seconds=training_time_seconds,
                prepared=prepared,
            )
        )
    return rows


def require_test_data(
    prepared: PreparedData,
    require_per_ticker: bool = True,
) -> None:
    if (
        prepared.test_df is None
        or prepared.test_dataset is None
        or prepared.y_test is None
    ):
        raise ValueError("test data is required for non-validation evaluation")
    if require_per_ticker and (
        not prepared.test_datasets_by_ticker
        or not prepared.y_test_by_ticker
    ):
        raise ValueError("test data is required for non-validation evaluation")


def require_test_label_data(prepared: PreparedData) -> None:
    if (
        prepared.test_df is None
        or prepared.y_test is None
        or not prepared.y_test_by_ticker
    ):
        raise ValueError("test data is required for non-validation evaluation")


def build_model(model_name: str, seq_len: int, input_size: int) -> torch.nn.Module:
    if model_name == "lstm":
        return LSTMClassifier(input_size=input_size, hidden_size=32, num_layers=1)
    if model_name == "tcn":
        return TCNClassifier(input_size=input_size, num_channels=[16, 16], kernel_size=3)
    if model_name == "dlinear":
        return DLinearClassifier(seq_len=seq_len, input_size=input_size, moving_avg_kernel=5)
    if model_name == "ms_dlinear_tcn":
        return MultiScaleDLinearTCNClassifier(
            seq_len=seq_len,
            input_size=input_size,
            moving_avg_kernels=(3, 5, 9),
            tcn_channels=(16, 16),
            tcn_kernel_size=3,
        )
    raise ValueError(f"unknown model name: {model_name}")


def evaluate_scope(
    scope_name: str,
    model: torch.nn.Module,
    dataset: WindowedClassificationDataset,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
    seed: int,
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    model_name: str,
    history: dict[str, Any],
    training_time_seconds: float,
    prepared: PreparedData,
) -> dict[str, Any]:
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    metrics, _, _ = evaluate(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
    )
    baseline_metrics = compute_baselines(y_train, y_eval)
    val_baseline_metrics = compute_baselines(prepared.y_train, prepared.y_val)
    delta = metrics["macro_f1"] - baseline_metrics["dummy_stratified_macro_f1_mean"]
    val_delta = (
        history["best_metric"]
        - val_baseline_metrics["dummy_stratified_macro_f1_mean"]
    )
    suspicious_status = "ok"
    if metrics["macro_f1"] > 0.70:
        suspicious_status = "macro_f1_gt_0.70_check_leakage"
    scope_y_train = prepared.y_train
    if scope_name != "pooled":
        scope_y_train = prepared.y_train_by_ticker[scope_name]
    ticker_baseline_metrics = compute_baselines(scope_y_train, y_eval)
    delta_vs_ticker_dummy = (
        metrics["macro_f1"] - ticker_baseline_metrics["dummy_stratified_macro_f1_mean"]
    )

    return {
        **base_result_fields(metadata, candidate),
        "model_name": model_name,
        "ticker": scope_name,
        "seed": seed,
        "split": "test",
        **secondary_baseline_scope_fields(scope_name),
        "n_train_windows": int(len(prepared.train_dataset)),
        "n_val_windows": int(len(prepared.val_dataset)),
        "n_test_windows": int(len(dataset)),
        "train_up_pct": safe_mean(scope_y_train.astype(float)),
        "baseline_train_up_pct": safe_mean(y_train.astype(float)),
        "val_up_pct": safe_mean(prepared.y_val.astype(float)),
        "test_up_pct": safe_mean(y_eval.astype(float)),
        "model_macro_f1": float(metrics["macro_f1"]),
        "model_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "model_precision_macro": float(metrics["precision_macro"]),
        "model_recall_macro": float(metrics["recall_macro"]),
        **baseline_metrics,
        "delta_macro_f1_vs_dummy": float(delta),
        **ticker_baseline_fields(ticker_baseline_metrics),
        "delta_macro_f1_vs_ticker_dummy": float(delta_vs_ticker_dummy),
        **scope_diagnostics_fields(scope_name, prepared, metadata),
        "confusion_matrix_labels": json.dumps([0, 1]),
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
        "classification_report": json.dumps(metrics["classification_report"]),
        "best_epoch": history["best_epoch"],
        "best_val_macro_f1": history["best_metric"],
        "val_dummy_stratified_macro_f1_mean": val_baseline_metrics[
            "dummy_stratified_macro_f1_mean"
        ],
        "val_delta_macro_f1_vs_dummy": float(val_delta),
        "training_time_seconds": float(training_time_seconds),
        "suspicious_status": suspicious_status,
    }


def evaluate_validation_scope(
    scope_name: str,
    model: torch.nn.Module,
    dataset: WindowedClassificationDataset,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int,
    seed: int,
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    model_name: str,
    history: dict[str, Any],
    training_time_seconds: float,
    prepared: PreparedData,
) -> dict[str, Any]:
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    metrics, _, _ = evaluate(
        model=model,
        loader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
    )
    baseline_metrics = compute_baselines(y_train, y_eval)
    val_delta = (
        metrics["macro_f1"]
        - baseline_metrics["dummy_stratified_macro_f1_mean"]
    )
    suspicious_status = "ok"
    if metrics["macro_f1"] > 0.70:
        suspicious_status = "macro_f1_gt_0.70_check_leakage"

    return {
        **base_result_fields(metadata, candidate),
        **validation_only_report_fields(),
        "model_name": model_name,
        "model_family": metadata["model_family"],
        "ticker": scope_name,
        "seed": seed,
        "split": "validation",
        **secondary_baseline_scope_fields(scope_name),
        "n_train_windows": int(len(prepared.train_dataset)),
        "n_val_windows": int(len(dataset)),
        "train_up_pct": safe_mean(y_train.astype(float)),
        "val_up_pct": safe_mean(y_eval.astype(float)),
        **validation_only_coverage_fields(y_eval),
        "val_macro_f1": float(metrics["macro_f1"]),
        "val_balanced_accuracy": float(metrics["balanced_accuracy"]),
        **scope_diagnostics_fields(scope_name, prepared, metadata),
        "best_epoch": history["best_epoch"],
        "best_val_macro_f1": history["best_metric"],
        "val_dummy_stratified_macro_f1_mean": baseline_metrics[
            "dummy_stratified_macro_f1_mean"
        ],
        "val_delta_macro_f1_vs_dummy": float(val_delta),
        "training_time_seconds": float(training_time_seconds),
        "suspicious_status": suspicious_status,
    }


def scope_diagnostics_fields(
    scope_name: str,
    prepared: PreparedData,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if scope_name == "pooled":
        tickers = metadata["tickers"]
        label_n_total = sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_total"])
            for ticker in tickers
        )
        label_n_retained = sum(
            int(
                prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_up"]
                + prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_down"]
            )
            for ticker in tickers
        )
        return {
            "label_n_total": label_n_total,
            "label_n_retained": label_n_retained,
            "label_n_neutral": sum(
                int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_neutral"])
                for ticker in tickers
            ),
            "label_n_cross_day": sum(
                int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_cross_day"])
                for ticker in tickers
            ),
            "label_n_tail": sum(
                int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_tail"])
                for ticker in tickers
            ),
            "label_n_zero_return": sum(
                int(prepared.diagnostics_by_ticker[f"{ticker}_label"].get("n_zero_return", 0))
                for ticker in tickers
            ),
            "retained_pct": label_n_retained / label_n_total,
            "val_up_pct": safe_mean(prepared.y_val.astype(float)),
        }

    label_diag = prepared.diagnostics_by_ticker[f"{scope_name}_label"]
    label_n_total = int(label_diag["n_total"])
    label_n_retained = int(label_diag["n_up"] + label_diag["n_down"])
    return {
        "label_n_total": label_n_total,
        "label_n_retained": label_n_retained,
        "label_n_neutral": int(label_diag["n_neutral"]),
        "label_n_cross_day": int(label_diag["n_cross_day"]),
        "label_n_tail": int(label_diag["n_tail"]),
        "label_n_zero_return": int(label_diag.get("n_zero_return", 0)),
        "retained_pct": label_n_retained / label_n_total,
        "val_up_pct": safe_mean(prepared.y_val_by_ticker[scope_name].astype(float)),
    }


def base_result_fields(metadata: dict[str, Any], candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "run_id": metadata["run_id"],
        "git_commit_hash": metadata["git_commit_hash"],
        "data_source": metadata["data_source"],
        "feature_set_id": metadata["feature_set_id"],
        "feature_columns": json.dumps(metadata.get("feature_columns", [])),
        "candidate_id": candidate.candidate_id,
        "label_mode": metadata["label_mode"],
        "label_semantics": metadata["label_semantics"],
        "label_formula": metadata["label_formula"],
        "class_0_name": metadata["class_0_name"],
        "class_1_name": metadata["class_1_name"],
        "zero_return_policy": metadata["zero_return_policy"],
        "no_trade_band_enabled": metadata["no_trade_band_enabled"],
        "neutral_policy": metadata["neutral_policy"],
        "window_size": candidate.window_size,
        "label_horizon_k": candidate.label_horizon_k,
        "threshold_bps": candidate.threshold_bps,
        "threshold_source": metadata["threshold_source"],
        "decision_time_policy": metadata["decision_time_policy"],
        "scaler_id": metadata["scaler_id"],
        "scaler_fit_scope": metadata["scaler_fit_scope"],
        "timestamp_col": metadata["timestamp_col"],
        "price_col": metadata["price_col"],
        "shuffle_train_labels": metadata["shuffle_train_labels"],
        "max_rows_per_ticker": metadata["max_rows_per_ticker"],
        "effective_max_rows_per_ticker": metadata["effective_max_rows_per_ticker"],
        "claim_scope": metadata["claim_scope"],
        "diagnostic_scope": metadata["diagnostic_scope"],
        "diagnostic_only": metadata["diagnostic_only"],
        "non_claim": metadata["non_claim"],
        "shuffle_seed": metadata["shuffle_seed"],
        "checkpoint_policy": metadata["checkpoint_policy"],
        "training_scope": metadata["training_scope"],
        "baseline_scope": metadata["baseline_scope"],
        "primary_baseline_scope": metadata["primary_baseline_scope"],
        "dummy_stratified_random_states": json.dumps(
            metadata["dummy_stratified_random_states"]
        ),
    }


def validation_only_report_fields() -> dict[str, Any]:
    return {
        "report_scope": "validation_only",
        "selection_scope": "validation_only",
        "test_metrics_embargoed": True,
        "test_metrics_used": False,
    }


def validation_only_coverage_fields(y_eval: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y_eval, dtype=int)
    n_windows = int(labels.size)
    class_0_count = int(np.sum(labels == 0))
    class_1_count = int(np.sum(labels == 1))
    if n_windows == 0:
        class_0_pct = float("nan")
        class_1_pct = float("nan")
        trade_coverage_rate = float("nan")
        no_trade_rate = float("nan")
    else:
        class_0_pct = class_0_count / n_windows
        class_1_pct = class_1_count / n_windows
        trade_coverage_rate = 1.0
        no_trade_rate = 0.0
    return {
        "validation_coverage_scope": "post_filter_validation_windows",
        "validation_coverage_note": (
            "post_filter_counts_only_no_prefilter_no_trade_reconstruction"
        ),
        "validation_n_windows_post_filter": n_windows,
        "validation_n_trade_windows_post_filter": n_windows,
        "validation_n_no_trade_windows_post_filter": 0,
        "validation_trade_coverage_rate_post_filter": trade_coverage_rate,
        "validation_no_trade_rate_post_filter": no_trade_rate,
        "validation_class_0_count_post_filter": class_0_count,
        "validation_class_1_count_post_filter": class_1_count,
        "validation_class_0_pct_post_filter": class_0_pct,
        "validation_class_1_pct_post_filter": class_1_pct,
    }


def feature_drop_fields(
    prepared: PreparedData,
    scope_name: str,
    tickers: list[str],
) -> dict[str, Any]:
    if scope_name == "pooled":
        diagnostics = [
            prepared.diagnostics_by_ticker.get(f"{ticker}_feature", {})
            for ticker in tickers
        ]
        n_raw = sum(int(values.get("n_raw_rows", 0)) for values in diagnostics)
        n_feature = sum(int(values.get("n_feature_rows", 0)) for values in diagnostics)
        n_dropped = n_raw - n_feature
        return {
            "feature_drop_count": n_dropped,
            "feature_drop_pct": n_dropped / n_raw if n_raw else None,
        }

    diagnostics = prepared.diagnostics_by_ticker.get(f"{scope_name}_feature", {})
    return {
        "feature_drop_count": int(diagnostics.get("feature_drop_count", 0)),
        "feature_drop_pct": diagnostics.get("feature_drop_pct"),
    }


def secondary_baseline_scope_fields(scope_name: str) -> dict[str, str]:
    if scope_name == "pooled":
        return {
            "secondary_baseline_scope": "per_ticker_train",
            "secondary_baseline_scope_effective": "pooled_train",
            "secondary_baseline_scope_note": "not_applicable_to_pooled_row",
        }
    return {
        "secondary_baseline_scope": "per_ticker_train",
        "secondary_baseline_scope_effective": "per_ticker_train",
        "secondary_baseline_scope_note": "",
    }


def split_date_range_fields(
    prepared: PreparedData,
    scope_name: str,
    timestamp_col: str,
    include_holdout: bool = True,
) -> dict[str, Any]:
    fields = {
        **split_frame_date_range_fields(
            prepared.train_df,
            scope_name,
            timestamp_col,
            "train",
        ),
        **split_frame_date_range_fields(
            prepared.val_df,
            scope_name,
            timestamp_col,
            "val",
        ),
    }
    if include_holdout:
        fields.update(
            split_frame_date_range_fields(
                prepared.test_df,
                scope_name,
                timestamp_col,
                "holdout",
            )
        )
    return fields


def split_frame_date_range_fields(
    split_df: pd.DataFrame,
    scope_name: str,
    timestamp_col: str,
    output_prefix: str,
) -> dict[str, Any]:
    if split_df.empty:
        return {
            f"{output_prefix}_start_ts": None,
            f"{output_prefix}_end_ts": None,
        }
    if timestamp_col not in split_df.columns:
        raise ValueError(f"split frame missing timestamp column {timestamp_col!r}")

    frame = split_df
    if scope_name != "pooled":
        if "ticker" not in split_df.columns:
            raise ValueError("ticker split date ranges require ticker column")
        frame = split_df.loc[split_df["ticker"] == scope_name]
        if frame.empty:
            return {
                f"{output_prefix}_start_ts": None,
                f"{output_prefix}_end_ts": None,
            }

    timestamps = pd.to_datetime(frame[timestamp_col])
    if timestamps.isna().any():
        bad_index = timestamps[timestamps.isna()].index[0]
        raise ValueError(
            f"split frame timestamp column {timestamp_col!r} has missing value "
            f"at row/index {bad_index!r}"
        )
    return {
        f"{output_prefix}_start_ts": timestamps.min().isoformat(),
        f"{output_prefix}_end_ts": timestamps.max().isoformat(),
    }


def compute_baselines(y_train: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
    stratified_values = [
        dummy_baseline_metrics(y_train, y_eval, strategy="stratified", random_state=random_state)
        for random_state in range(10)
    ]
    stratified_macro_f1 = np.asarray(
        [values["macro_f1"] for values in stratified_values],
        dtype=float,
    )
    stratified_balanced_accuracy = np.asarray(
        [values["balanced_accuracy"] for values in stratified_values],
        dtype=float,
    )
    stratified_confusion = np.asarray(
        [values["confusion_matrix"] for values in stratified_values],
        dtype=float,
    )
    prior = dummy_baseline_metrics(y_train, y_eval, strategy="prior", random_state=0)
    always_up = always_predict_baseline_metrics(y_eval, constant_label=1)
    always_down = always_predict_baseline_metrics(y_eval, constant_label=0)
    return {
        "dummy_stratified_macro_f1_mean": float(stratified_macro_f1.mean()),
        "dummy_stratified_macro_f1_std": float(stratified_macro_f1.std(ddof=0)),
        "dummy_stratified_balanced_accuracy_mean": float(
            stratified_balanced_accuracy.mean()
        ),
        "dummy_stratified_balanced_accuracy_std": float(
            stratified_balanced_accuracy.std(ddof=0)
        ),
        "dummy_stratified_confusion_matrix_mean": json.dumps(
            stratified_confusion.mean(axis=0).tolist()
        ),
        "dummy_prior_macro_f1": float(prior["macro_f1"]),
        "dummy_prior_balanced_accuracy": float(prior["balanced_accuracy"]),
        "dummy_prior_confusion_matrix": json.dumps(prior["confusion_matrix"].tolist()),
        "always_up_macro_f1": float(always_up["macro_f1"]),
        "always_up_balanced_accuracy": float(always_up["balanced_accuracy"]),
        "always_up_confusion_matrix": json.dumps(always_up["confusion_matrix"].tolist()),
        "always_down_macro_f1": float(always_down["macro_f1"]),
        "always_down_balanced_accuracy": float(always_down["balanced_accuracy"]),
        "always_down_confusion_matrix": json.dumps(
            always_down["confusion_matrix"].tolist()
        ),
    }


def ticker_baseline_fields(baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker_dummy_stratified_macro_f1_mean": baseline_metrics[
            "dummy_stratified_macro_f1_mean"
        ],
        "ticker_dummy_stratified_macro_f1_std": baseline_metrics[
            "dummy_stratified_macro_f1_std"
        ],
        "ticker_dummy_stratified_balanced_accuracy_mean": baseline_metrics[
            "dummy_stratified_balanced_accuracy_mean"
        ],
        "ticker_dummy_stratified_balanced_accuracy_std": baseline_metrics[
            "dummy_stratified_balanced_accuracy_std"
        ],
        "ticker_dummy_stratified_confusion_matrix_mean": baseline_metrics[
            "dummy_stratified_confusion_matrix_mean"
        ],
        "ticker_dummy_prior_macro_f1": baseline_metrics["dummy_prior_macro_f1"],
        "ticker_dummy_prior_balanced_accuracy": baseline_metrics[
            "dummy_prior_balanced_accuracy"
        ],
        "ticker_dummy_prior_confusion_matrix": baseline_metrics[
            "dummy_prior_confusion_matrix"
        ],
        "ticker_always_up_macro_f1": baseline_metrics["always_up_macro_f1"],
        "ticker_always_up_balanced_accuracy": baseline_metrics[
            "always_up_balanced_accuracy"
        ],
        "ticker_always_up_confusion_matrix": baseline_metrics[
            "always_up_confusion_matrix"
        ],
        "ticker_always_down_macro_f1": baseline_metrics["always_down_macro_f1"],
        "ticker_always_down_balanced_accuracy": baseline_metrics[
            "always_down_balanced_accuracy"
        ],
        "ticker_always_down_confusion_matrix": baseline_metrics[
            "always_down_confusion_matrix"
        ],
    }


def build_manifest_rows(
    metadata: dict[str, Any],
    candidate: CandidateSpec,
    prepared: PreparedData,
) -> list[dict[str, Any]]:
    rows = []
    validation_only_report = metadata.get("report_scope") == "validation_only"
    if not validation_only_report:
        require_test_label_data(prepared)
    for ticker in metadata["tickers"]:
        label_diag = prepared.diagnostics_by_ticker[f"{ticker}_label"]
        train_diag = prepared.diagnostics_by_ticker[f"{ticker}_train"]
        val_diag = prepared.diagnostics_by_ticker[f"{ticker}_val"]
        y_train = prepared.y_train_by_ticker[ticker]
        y_val = prepared.y_val_by_ticker[ticker]
        retained = int(label_diag["n_up"] + label_diag["n_down"])
        row = {
            **base_result_fields(metadata, candidate),
            "ticker": ticker,
            **secondary_baseline_scope_fields(ticker),
            **feature_drop_fields(prepared, ticker, metadata["tickers"]),
            **split_date_range_fields(
                prepared,
                ticker,
                metadata["timestamp_col"],
                include_holdout=not validation_only_report,
            ),
            "label_n_total": int(label_diag["n_total"]),
            "label_n_retained": retained,
            "label_n_neutral": int(label_diag["n_neutral"]),
            "label_n_cross_day": int(label_diag["n_cross_day"]),
            "label_n_tail": int(label_diag["n_tail"]),
            "label_n_zero_return": int(label_diag.get("n_zero_return", 0)),
            "retained_pct": retained / int(label_diag["n_total"]),
            "train_rows": train_diag["n_rows"],
            "val_rows": val_diag["n_rows"],
            "train_retained_labels": train_diag["n_retained_labels"],
            "val_retained_labels": val_diag["n_retained_labels"],
            "train_nan_labels": train_diag["n_nan_labels"],
            "val_nan_labels": val_diag["n_nan_labels"],
            "n_train_windows": int(y_train.shape[0]),
            "n_val_windows": int(y_val.shape[0]),
            "train_up_pct": safe_mean(y_train.astype(float)),
            "val_up_pct": safe_mean(y_val.astype(float)),
        }
        if not validation_only_report:
            test_diag = prepared.diagnostics_by_ticker[f"{ticker}_test"]
            y_test = prepared.y_test_by_ticker[ticker]
            row.update(
                {
                    "test_rows": test_diag["n_rows"],
                    "test_retained_labels": test_diag["n_retained_labels"],
                    "test_nan_labels": test_diag["n_nan_labels"],
                    "n_test_windows": int(y_test.shape[0]),
                    "test_up_pct": safe_mean(y_test.astype(float)),
                }
            )
        rows.append(row)
    pooled_label_n_total = sum(
        int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_total"])
        for ticker in metadata["tickers"]
    )
    pooled_label_n_retained = sum(
        int(
            prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_up"]
            + prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_down"]
        )
        for ticker in metadata["tickers"]
    )
    pooled_row = {
        **base_result_fields(metadata, candidate),
        "ticker": "pooled",
        **secondary_baseline_scope_fields("pooled"),
        **feature_drop_fields(prepared, "pooled", metadata["tickers"]),
        **split_date_range_fields(
            prepared,
            "pooled",
            metadata["timestamp_col"],
            include_holdout=not validation_only_report,
        ),
        "label_n_total": pooled_label_n_total,
        "label_n_retained": pooled_label_n_retained,
        "label_n_neutral": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_neutral"])
            for ticker in metadata["tickers"]
        ),
        "label_n_cross_day": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_cross_day"])
            for ticker in metadata["tickers"]
        ),
        "label_n_tail": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_label"]["n_tail"])
            for ticker in metadata["tickers"]
        ),
        "label_n_zero_return": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_label"].get("n_zero_return", 0))
            for ticker in metadata["tickers"]
        ),
        "retained_pct": pooled_label_n_retained / pooled_label_n_total,
        "train_rows": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_train"]["n_rows"])
            for ticker in metadata["tickers"]
        ),
        "val_rows": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_val"]["n_rows"])
            for ticker in metadata["tickers"]
        ),
        "train_retained_labels": sum(
            int(
                prepared.diagnostics_by_ticker[f"{ticker}_train"][
                    "n_retained_labels"
                ]
            )
            for ticker in metadata["tickers"]
        ),
        "val_retained_labels": sum(
            int(
                prepared.diagnostics_by_ticker[f"{ticker}_val"][
                    "n_retained_labels"
                ]
            )
            for ticker in metadata["tickers"]
        ),
        "train_nan_labels": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_train"]["n_nan_labels"])
            for ticker in metadata["tickers"]
        ),
        "val_nan_labels": sum(
            int(prepared.diagnostics_by_ticker[f"{ticker}_val"]["n_nan_labels"])
            for ticker in metadata["tickers"]
        ),
        "n_train_windows": int(prepared.y_train.shape[0]),
        "n_val_windows": int(prepared.y_val.shape[0]),
        "train_up_pct": safe_mean(prepared.y_train.astype(float)),
        "val_up_pct": safe_mean(prepared.y_val.astype(float)),
    }
    if not validation_only_report:
        pooled_row.update(
            {
                "test_rows": sum(
                    int(prepared.diagnostics_by_ticker[f"{ticker}_test"]["n_rows"])
                    for ticker in metadata["tickers"]
                ),
                "test_retained_labels": sum(
                    int(
                        prepared.diagnostics_by_ticker[f"{ticker}_test"][
                            "n_retained_labels"
                        ]
                    )
                    for ticker in metadata["tickers"]
                ),
                "test_nan_labels": sum(
                    int(
                        prepared.diagnostics_by_ticker[f"{ticker}_test"][
                            "n_nan_labels"
                        ]
                    )
                    for ticker in metadata["tickers"]
                ),
                "n_test_windows": int(prepared.y_test.shape[0]),
                "test_up_pct": safe_mean(prepared.y_test.astype(float)),
            }
        )
    rows.append(pooled_row)
    return rows


def write_outputs(
    output_dir: Path,
    stem: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    run_dir = output_dir / metadata["run_id"]
    if "_run_output_dir_initialized" not in metadata:
        if run_dir.exists():
            raise FileExistsError(f"run artifact directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        metadata["_run_output_dir_initialized"] = True
    elif not run_dir.exists():
        raise FileNotFoundError(f"run artifact directory missing: {run_dir}")
    pd.DataFrame(rows).to_csv(run_dir / f"{stem}.csv", index=False)
    persisted_metadata = {
        key: value for key, value in metadata.items() if not key.startswith("_")
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(persisted_metadata, handle, indent=2)


if __name__ == "__main__":
    main()
