"""Summarize completed Phase 1 local run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "phase1b_local_reports"
MODEL_EXPANSION_DELTA_THRESHOLD = 0.01

REQUIRED_RESULT_COLUMNS = (
    "run_id",
    "feature_set_id",
    "label_mode",
    "threshold_bps",
    "model_name",
    "ticker",
    "seed",
    "n_test_windows",
    "test_up_pct",
    "model_macro_f1",
    "model_balanced_accuracy",
    "delta_macro_f1_vs_dummy",
    "dummy_stratified_macro_f1_mean",
    "dummy_stratified_macro_f1_std",
    "delta_macro_f1_vs_ticker_dummy",
    "ticker_dummy_stratified_macro_f1_mean",
    "ticker_dummy_stratified_macro_f1_std",
    "label_n_neutral",
    "retained_pct",
    "training_time_seconds",
    "suspicious_status",
)
REQUIRED_MANIFEST_COLUMNS = (
    "run_id",
    "feature_set_id",
    "label_mode",
    "threshold_bps",
    "ticker",
    "label_n_total",
    "label_n_retained",
    "label_n_neutral",
    "label_n_cross_day",
    "label_n_tail",
    "retained_pct",
    "n_train_windows",
    "n_test_windows",
    "train_up_pct",
    "test_up_pct",
)
OPTIONAL_PROTOCOL_COLUMNS = (
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "neutral_policy",
)
OPTIONAL_COUNT_COLUMNS = ("label_n_zero_return",)
OPTIONAL_GATE_COLUMNS = (
    "best_val_macro_f1",
    "val_dummy_stratified_macro_f1_mean",
    "val_delta_macro_f1_vs_dummy",
    "n_val_windows",
    "val_up_pct",
)
OPTIONAL_MANIFEST_OBSERVABILITY_COLUMNS = (
    "train_rows",
    "val_rows",
    "test_rows",
    "train_retained_labels",
    "val_retained_labels",
    "test_retained_labels",
    "train_nan_labels",
    "val_nan_labels",
    "test_nan_labels",
    "n_val_windows",
    "val_up_pct",
)
OPTIONAL_BASELINE_COLUMNS = (
    "dummy_stratified_balanced_accuracy_mean",
    "dummy_stratified_balanced_accuracy_std",
    "dummy_stratified_confusion_matrix_mean",
    "dummy_prior_balanced_accuracy",
    "dummy_prior_confusion_matrix",
    "always_up_balanced_accuracy",
    "always_up_confusion_matrix",
    "always_down_balanced_accuracy",
    "always_down_confusion_matrix",
    "ticker_dummy_stratified_balanced_accuracy_mean",
    "ticker_dummy_stratified_balanced_accuracy_std",
    "ticker_dummy_stratified_confusion_matrix_mean",
    "ticker_dummy_prior_balanced_accuracy",
    "ticker_dummy_prior_confusion_matrix",
    "ticker_always_up_balanced_accuracy",
    "ticker_always_up_confusion_matrix",
    "ticker_always_down_balanced_accuracy",
    "ticker_always_down_confusion_matrix",
)

RUN_SUMMARY_COLUMNS = (
    "run_dir",
    "run_id",
    "label_mode",
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "neutral_policy",
    "threshold_bps",
    "feature_set_id",
    "models",
    "seeds",
    "n_result_rows",
    "n_manifest_rows",
    "n_suspicious_rows",
    "pooled_retained_pct",
    "pooled_test_windows",
    "pooled_zero_return_rows",
    "gate_split",
    "best_pooled_model",
    "best_pooled_delta_macro_f1_vs_dummy",
    "best_pooled_test_report_model",
    "best_pooled_test_delta_macro_f1_vs_dummy",
    "model_expansion_gate",
    "model_expansion_gate_reason",
)
SUMMARY_COLUMNS = (
    "run_id",
    "label_mode",
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "neutral_policy",
    "threshold_bps",
    "feature_set_id",
    "model_name",
    "ticker",
    "report_split",
    "n_rows",
    "seeds",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "dummy_stratified_macro_f1_mean",
    "dummy_stratified_macro_f1_std",
    "dummy_stratified_balanced_accuracy_mean",
    "dummy_stratified_balanced_accuracy_std",
    "delta_macro_f1_vs_dummy_mean",
    "delta_macro_f1_vs_dummy_std",
    "ticker_dummy_stratified_macro_f1_mean",
    "ticker_dummy_stratified_macro_f1_std",
    "ticker_dummy_stratified_balanced_accuracy_mean",
    "ticker_dummy_stratified_balanced_accuracy_std",
    "delta_macro_f1_vs_ticker_dummy_mean",
    "delta_macro_f1_vs_ticker_dummy_std",
    "best_val_macro_f1_mean",
    "val_dummy_stratified_macro_f1_mean",
    "val_delta_macro_f1_vs_dummy_mean",
    "n_val_windows",
    "n_test_windows",
    "val_up_pct",
    "retained_pct",
    "label_n_neutral",
    "label_n_zero_return",
    "training_time_seconds_mean",
)
COVERAGE_COLUMNS = (
    "run_id",
    "label_mode",
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "neutral_policy",
    "threshold_bps",
    "feature_set_id",
    "ticker",
    "label_n_total",
    "label_n_retained",
    "label_n_neutral",
    "label_n_cross_day",
    "label_n_tail",
    "label_n_zero_return",
    "retained_pct",
    "train_rows",
    "val_rows",
    "test_rows",
    "train_retained_labels",
    "val_retained_labels",
    "test_retained_labels",
    "train_nan_labels",
    "val_nan_labels",
    "test_nan_labels",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
    "train_up_pct",
    "val_up_pct",
    "test_up_pct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write markdown and CSV summaries for completed Phase 1 run dirs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Completed run directory containing results.csv and manifest.csv.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Directory whose immediate children are completed run directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = collect_run_dirs(args.run_dir, args.run_root)
    summarize_run_dirs(run_dirs, args.output_dir)
    print(f"wrote report tables for {len(run_dirs)} run dirs to {args.output_dir}")


def collect_run_dirs(run_dirs: list[Path], run_roots: list[Path]) -> list[Path]:
    candidates = [path.resolve() for path in run_dirs]
    for root in run_roots:
        root = root.resolve()
        if is_run_dir(root):
            candidates.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(f"run root does not exist: {root}")
        candidates.extend(path for path in root.iterdir() if is_run_dir(path))

    unique_dirs = sorted({path for path in candidates})
    if not unique_dirs:
        raise ValueError("provide at least one --run-dir or --run-root with run outputs")
    for run_dir in unique_dirs:
        validate_run_files(run_dir)
    return unique_dirs


def summarize_run_dirs(run_dirs: list[Path], output_dir: Path) -> dict[str, pd.DataFrame]:
    run_summaries: list[dict[str, Any]] = []
    pooled_frames: list[pd.DataFrame] = []
    ticker_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        tables = read_run_tables(run_dir)
        tables["manifest"] = backfill_manifest_from_results(
            tables["manifest"], tables["results"]
        )
        pooled_summary = summarize_results(
            tables["results"].loc[tables["results"]["ticker"] == "pooled"]
        )
        ticker_summary = summarize_results(
            tables["results"].loc[tables["results"]["ticker"] != "pooled"]
        )
        run_summaries.append(summarize_run(run_dir, tables, pooled_summary))
        pooled_frames.append(pooled_summary)
        ticker_frames.append(ticker_summary)
        coverage_frames.append(tables["manifest"].loc[:, COVERAGE_COLUMNS])

    outputs = {
        "run_summary": pd.DataFrame(run_summaries, columns=RUN_SUMMARY_COLUMNS),
        "pooled_by_model": concat_or_empty(pooled_frames, SUMMARY_COLUMNS),
        "by_model_ticker": concat_or_empty(ticker_frames, SUMMARY_COLUMNS),
        "coverage_by_ticker": concat_or_empty(coverage_frames, COVERAGE_COLUMNS),
    }
    write_summary_outputs(output_dir, outputs)
    return outputs


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "results.csv").exists() and (path / "manifest.csv").exists()


def validate_run_files(run_dir: Path) -> None:
    for filename in ("results.csv", "manifest.csv", "metadata.json"):
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing required run file: {path}")


def read_run_tables(run_dir: Path) -> dict[str, Any]:
    validate_run_files(run_dir)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    results = normalize_optional_columns(pd.read_csv(run_dir / "results.csv"), metadata)
    manifest = normalize_optional_columns(pd.read_csv(run_dir / "manifest.csv"), metadata)
    validate_columns(run_dir / "results.csv", results, REQUIRED_RESULT_COLUMNS)
    validate_columns(run_dir / "manifest.csv", manifest, REQUIRED_MANIFEST_COLUMNS)
    return {"metadata": metadata, "results": results, "manifest": manifest}


def validate_columns(path: Path, frame: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def backfill_manifest_from_results(
    manifest: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    backfilled = manifest.copy()
    for row_index, row in backfilled.iterrows():
        result_rows = results.loc[results["ticker"] == row["ticker"]]
        if result_rows.empty:
            continue
        for column in ("retained_pct", "label_n_zero_return"):
            if pd.isna(row[column]) and column in result_rows.columns:
                value = result_rows[column].dropna()
                if not value.empty:
                    backfilled.at[row_index, column] = value.iloc[0]
    return backfilled


def normalize_optional_columns(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in OPTIONAL_PROTOCOL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = infer_protocol_value(column, normalized, metadata)
    for column in OPTIONAL_COUNT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column in OPTIONAL_GATE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column in OPTIONAL_MANIFEST_OBSERVABILITY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column in OPTIONAL_BASELINE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized


def infer_protocol_value(column: str, frame: pd.DataFrame, metadata: dict[str, Any]) -> Any:
    if column in metadata:
        return metadata[column]
    label_mode = metadata.get("label_mode", first_value(frame, "label_mode"))
    if column == "label_semantics":
        if label_mode == "legacy_binary":
            return "canonical_phase1_full_binary"
        return "phase1b_no_trade_band_diagnostic"
    if column == "zero_return_policy":
        if label_mode == "legacy_binary":
            return "class_0_non_up"
        return "neutral_nan"
    if column == "no_trade_band_enabled":
        return label_mode != "legacy_binary"
    if column == "neutral_policy":
        if label_mode == "legacy_binary":
            return "not_applicable"
        return "abs(future_avg_r) <= threshold_bps is NaN/skipped"
    raise ValueError(f"unknown protocol column: {column}")


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "run_id",
        "label_mode",
        "label_semantics",
        "zero_return_policy",
        "no_trade_band_enabled",
        "neutral_policy",
        "threshold_bps",
        "feature_set_id",
        "model_name",
        "ticker",
    ]
    for group_values, group in results.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "report_split": "final_test_exploratory",
                "n_rows": int(len(group)),
                "seeds": join_unique_values(group["seed"]),
                "macro_f1_mean": mean_value(group, "model_macro_f1"),
                "macro_f1_std": std_value(group, "model_macro_f1"),
                "balanced_accuracy_mean": mean_value(group, "model_balanced_accuracy"),
                "dummy_stratified_macro_f1_mean": mean_value(
                    group, "dummy_stratified_macro_f1_mean"
                ),
                "dummy_stratified_macro_f1_std": mean_value(
                    group, "dummy_stratified_macro_f1_std"
                ),
                "dummy_stratified_balanced_accuracy_mean": nullable_mean_value(
                    group, "dummy_stratified_balanced_accuracy_mean"
                ),
                "dummy_stratified_balanced_accuracy_std": nullable_mean_value(
                    group, "dummy_stratified_balanced_accuracy_std"
                ),
                "delta_macro_f1_vs_dummy_mean": mean_value(
                    group, "delta_macro_f1_vs_dummy"
                ),
                "delta_macro_f1_vs_dummy_std": std_value(
                    group, "delta_macro_f1_vs_dummy"
                ),
                "ticker_dummy_stratified_macro_f1_mean": mean_value(
                    group, "ticker_dummy_stratified_macro_f1_mean"
                ),
                "ticker_dummy_stratified_macro_f1_std": mean_value(
                    group, "ticker_dummy_stratified_macro_f1_std"
                ),
                "ticker_dummy_stratified_balanced_accuracy_mean": nullable_mean_value(
                    group, "ticker_dummy_stratified_balanced_accuracy_mean"
                ),
                "ticker_dummy_stratified_balanced_accuracy_std": nullable_mean_value(
                    group, "ticker_dummy_stratified_balanced_accuracy_std"
                ),
                "delta_macro_f1_vs_ticker_dummy_mean": mean_value(
                    group, "delta_macro_f1_vs_ticker_dummy"
                ),
                "delta_macro_f1_vs_ticker_dummy_std": std_value(
                    group, "delta_macro_f1_vs_ticker_dummy"
                ),
                "best_val_macro_f1_mean": nullable_mean_value(
                    group, "best_val_macro_f1"
                ),
                "val_dummy_stratified_macro_f1_mean": nullable_mean_value(
                    group, "val_dummy_stratified_macro_f1_mean"
                ),
                "val_delta_macro_f1_vs_dummy_mean": nullable_mean_value(
                    group, "val_delta_macro_f1_vs_dummy"
                ),
                "n_val_windows": first_non_null_value(group, "n_val_windows"),
                "n_test_windows": first_value(group, "n_test_windows"),
                "val_up_pct": first_non_null_value(group, "val_up_pct"),
                "retained_pct": first_value(group, "retained_pct"),
                "label_n_neutral": first_value(group, "label_n_neutral"),
                "label_n_zero_return": first_value(group, "label_n_zero_return"),
                "training_time_seconds_mean": mean_value(
                    group, "training_time_seconds"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize_run(
    run_dir: Path,
    tables: dict[str, Any],
    pooled_summary: pd.DataFrame,
) -> dict[str, Any]:
    results = tables["results"]
    manifest = tables["manifest"]
    metadata = tables["metadata"]
    gate_row = best_validation_gate_row(pooled_summary)
    test_report_row = pooled_summary.sort_values(
        "delta_macro_f1_vs_dummy_mean", ascending=False
    ).iloc[0]
    pooled_manifest = manifest.loc[manifest["ticker"] == "pooled"]
    if pooled_manifest.empty:
        raise ValueError(f"{run_dir / 'manifest.csv'} missing pooled coverage row")
    pooled_row = pooled_manifest.iloc[0]
    gate_delta = None
    gate_reason = "insufficient_validation_evidence"
    best_model = pd.NA
    if gate_row is not None:
        gate_delta = float(gate_row["val_delta_macro_f1_vs_dummy_mean"])
        gate_reason = "validation_delta_available"
        best_model = gate_row["model_name"]
    test_delta = float(test_report_row["delta_macro_f1_vs_dummy_mean"])
    return {
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id", first_value(results, "run_id")),
        "label_mode": metadata.get("label_mode", first_value(results, "label_mode")),
        "label_semantics": metadata.get(
            "label_semantics", first_value(results, "label_semantics")
        ),
        "zero_return_policy": metadata.get(
            "zero_return_policy", first_value(results, "zero_return_policy")
        ),
        "no_trade_band_enabled": metadata.get(
            "no_trade_band_enabled", first_value(results, "no_trade_band_enabled")
        ),
        "neutral_policy": metadata.get(
            "neutral_policy", first_value(results, "neutral_policy")
        ),
        "threshold_bps": metadata.get(
            "threshold_bps", first_value(results, "threshold_bps")
        ),
        "feature_set_id": metadata.get(
            "feature_set_id", first_value(results, "feature_set_id")
        ),
        "models": join_unique_values(results["model_name"]),
        "seeds": join_unique_values(results["seed"]),
        "n_result_rows": int(len(results)),
        "n_manifest_rows": int(len(manifest)),
        "n_suspicious_rows": int((results["suspicious_status"] != "ok").sum()),
        "pooled_retained_pct": pooled_row["retained_pct"],
        "pooled_test_windows": pooled_row["n_test_windows"],
        "pooled_zero_return_rows": pooled_row["label_n_zero_return"],
        "gate_split": "validation",
        "best_pooled_model": best_model,
        "best_pooled_delta_macro_f1_vs_dummy": gate_delta,
        "best_pooled_test_report_model": test_report_row["model_name"],
        "best_pooled_test_delta_macro_f1_vs_dummy": test_delta,
        "model_expansion_gate": expansion_gate(gate_delta),
        "model_expansion_gate_reason": gate_reason,
    }


def best_validation_gate_row(pooled_summary: pd.DataFrame) -> pd.Series | None:
    if "val_delta_macro_f1_vs_dummy_mean" not in pooled_summary.columns:
        return None
    candidates = pooled_summary.dropna(subset=["val_delta_macro_f1_vs_dummy_mean"])
    if candidates.empty:
        return None
    return candidates.sort_values(
        "val_delta_macro_f1_vs_dummy_mean", ascending=False
    ).iloc[0]


def expansion_gate(best_delta: float | None) -> str:
    if best_delta is None:
        return "closed_insufficient_validation_evidence"
    if best_delta >= MODEL_EXPANSION_DELTA_THRESHOLD:
        return "review_required_delta_ge_0.01"
    return "blocked_delta_lt_0.01"


def mean_value(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="raise").mean())


def nullable_mean_value(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="raise").dropna()
    if values.empty:
        return None
    return float(values.mean())


def std_value(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="raise").std())


def first_value(frame: pd.DataFrame, column: str) -> Any:
    value = frame[column].iloc[0]
    if hasattr(value, "item"):
        return value.item()
    return value


def first_non_null_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].dropna()
    if values.empty:
        return None
    value = values.iloc[0]
    if hasattr(value, "item"):
        return value.item()
    return value


def join_unique_values(values: pd.Series) -> str:
    unique_values = sorted(values.dropna().unique().tolist())
    return ",".join(str(value) for value in unique_values)


def concat_or_empty(frames: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).loc[:, columns]


def write_summary_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_markdown_report(output_dir / "report.md", outputs)


def write_markdown_report(path: Path, outputs: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Phase 1 Local Run Summary",
        "",
        "## Run Gate Summary",
        "",
        dataframe_to_markdown(
            outputs["run_summary"],
            [
                "run_id",
                "label_semantics",
                "zero_return_policy",
                "no_trade_band_enabled",
                "threshold_bps",
                "best_pooled_model",
                "best_pooled_delta_macro_f1_vs_dummy",
                "best_pooled_test_report_model",
                "best_pooled_test_delta_macro_f1_vs_dummy",
                "model_expansion_gate",
                "model_expansion_gate_reason",
                "n_suspicious_rows",
            ],
        ),
        "",
        "## Pooled Final Test Report",
        "",
        dataframe_to_markdown(
            outputs["pooled_by_model"],
            [
                "run_id",
                "label_semantics",
                "report_split",
                "model_name",
                "val_delta_macro_f1_vs_dummy_mean",
                "dummy_stratified_macro_f1_mean",
                "dummy_stratified_balanced_accuracy_mean",
                "macro_f1_mean",
                "delta_macro_f1_vs_dummy_mean",
                "balanced_accuracy_mean",
                "n_test_windows",
                "retained_pct",
            ],
        ),
        "",
        "## Per-Ticker Dummy Delta",
        "",
        dataframe_to_markdown(
            outputs["by_model_ticker"],
            [
                "run_id",
                "model_name",
                "ticker",
                "ticker_dummy_stratified_macro_f1_mean",
                "ticker_dummy_stratified_balanced_accuracy_mean",
                "delta_macro_f1_vs_ticker_dummy_mean",
                "n_test_windows",
                "retained_pct",
            ],
        ),
        "",
        "## Coverage",
        "",
        dataframe_to_markdown(
            outputs["coverage_by_ticker"],
            [
                "run_id",
                "ticker",
                "retained_pct",
                "label_n_neutral",
                "label_n_zero_return",
                "n_val_windows",
                "n_test_windows",
                "val_up_pct",
                "test_up_pct",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.loc[:, columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(format_markdown_value(row[column]) for column in columns) + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
