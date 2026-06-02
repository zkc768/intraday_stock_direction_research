"""Build paper-ready tables from consolidated Phase 1 report CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "checkpoints" / "local_runner_reference_reports" / "table_records_20260525"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "legacy_runner_paper_tables"

INPUT_FILES = {
    "run_summary": "run_summary.csv",
    "pooled_by_model": "pooled_by_model.csv",
    "by_model_ticker": "by_model_ticker.csv",
    "coverage_by_ticker": "coverage_by_ticker.csv",
}

RUN_SUMMARY_COLUMNS = (
    "run_dir",
    "run_id",
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "threshold_bps",
    "feature_set_id",
    "seeds",
    "best_pooled_model",
    "best_pooled_delta_macro_f1_vs_dummy",
    "model_expansion_gate",
    "n_suspicious_rows",
    "pooled_retained_pct",
    "pooled_test_windows",
    "pooled_zero_return_rows",
)
POOLED_COLUMNS = (
    "run_id",
    "label_semantics",
    "threshold_bps",
    "model_name",
    "macro_f1_mean",
    "macro_f1_std",
    "balanced_accuracy_mean",
    "dummy_stratified_macro_f1_mean",
    "dummy_stratified_macro_f1_std",
    "delta_macro_f1_vs_dummy_mean",
    "delta_macro_f1_vs_dummy_std",
    "n_test_windows",
    "retained_pct",
)
TICKER_COLUMNS = (
    "run_id",
    "label_semantics",
    "threshold_bps",
    "model_name",
    "ticker",
    "macro_f1_mean",
    "ticker_dummy_stratified_macro_f1_mean",
    "delta_macro_f1_vs_ticker_dummy_mean",
    "delta_macro_f1_vs_ticker_dummy_std",
    "n_test_windows",
    "retained_pct",
)
COVERAGE_COLUMNS = (
    "run_id",
    "label_semantics",
    "zero_return_policy",
    "no_trade_band_enabled",
    "threshold_bps",
    "ticker",
    "retained_pct",
    "label_n_neutral",
    "label_n_zero_return",
    "n_test_windows",
    "test_up_pct",
)
SEED_RESULT_COLUMNS = (
    "run_id",
    "model_name",
    "ticker",
    "seed",
    "model_macro_f1",
    "delta_macro_f1_vs_ticker_dummy",
    "n_test_windows",
    "retained_pct",
    "suspicious_status",
)

OUTPUT_ORDER = (
    "paper_table_1_run_gate_summary",
    "paper_table_2_pooled_model_vs_dummy",
    "paper_table_3_canonical_ticker_delta",
    "paper_table_4_coverage_label_semantics",
    "paper_table_5_ticker_delta_counts",
    "paper_table_6_seed_ticker_stability",
    "paper_table_7_regime_shift_by_ticker",
    "paper_table_8_coverage_fragility_flags",
    "figure_delta_vs_coverage",
    "figure_ticker_delta_heatmap",
    "figure_threshold_retention_proxy",
)

MODEL_ORDER = {"dlinear": 0, "lstm": 1, "tcn": 2}
ALLOWED_LABEL_SEMANTICS = {
    "canonical_phase1_full_binary",
    "legacy_runner_no_trade_band_diagnostic",
}
LOW_COVERAGE_THRESHOLD = 0.2
LOW_TEST_WINDOWS_THRESHOLD = 5000
CLASS_BALANCE_LOWER_BOUND = 0.4
CLASS_BALANCE_UPPER_BOUND = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper tables from consolidated Phase 1 report CSVs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_paper_tables(args.input_dir, args.output_dir)
    print(f"wrote {len(outputs)} paper table artifacts to {args.output_dir}")


def build_paper_tables(input_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = read_report_tables(input_dir)
    seed_results = read_seed_result_rows(tables["run_summary"])
    outputs = {
        "paper_table_1_run_gate_summary": build_run_gate_summary(
            tables["run_summary"]
        ),
        "paper_table_2_pooled_model_vs_dummy": build_pooled_model_table(
            tables["pooled_by_model"]
        ),
        "paper_table_3_canonical_ticker_delta": build_canonical_ticker_table(
            tables["by_model_ticker"]
        ),
        "paper_table_4_coverage_label_semantics": build_coverage_table(
            tables["coverage_by_ticker"]
        ),
        "paper_table_5_ticker_delta_counts": build_ticker_delta_counts(
            tables["by_model_ticker"]
        ),
        "paper_table_6_seed_ticker_stability": build_seed_ticker_stability(
            seed_results
        ),
        "paper_table_7_regime_shift_by_ticker": build_regime_shift_by_ticker(
            tables["by_model_ticker"]
        ),
        "paper_table_8_coverage_fragility_flags": build_coverage_fragility_flags(
            tables["coverage_by_ticker"]
        ),
        "figure_delta_vs_coverage": build_delta_coverage_data(
            tables["pooled_by_model"]
        ),
        "figure_ticker_delta_heatmap": build_ticker_heatmap_data(
            tables["by_model_ticker"]
        ),
        "figure_threshold_retention_proxy": build_threshold_retention_proxy(
            tables["run_summary"]
        ),
    }
    write_outputs(output_dir, outputs)
    return outputs


def read_report_tables(input_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for table_name, filename in INPUT_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing required input file: {path}")
        frames[table_name] = pd.read_csv(path)
    validate_columns(frames["run_summary"], RUN_SUMMARY_COLUMNS, input_dir / "run_summary.csv")
    validate_columns(frames["pooled_by_model"], POOLED_COLUMNS, input_dir / "pooled_by_model.csv")
    validate_columns(frames["by_model_ticker"], TICKER_COLUMNS, input_dir / "by_model_ticker.csv")
    validate_columns(frames["coverage_by_ticker"], COVERAGE_COLUMNS, input_dir / "coverage_by_ticker.csv")
    validate_report_values(frames)
    return frames


def validate_report_values(frames: dict[str, pd.DataFrame]) -> None:
    for table_name, frame in frames.items():
        if "label_semantics" in frame.columns:
            unknown = sorted(set(frame["label_semantics"]) - ALLOWED_LABEL_SEMANTICS)
            if unknown:
                raise ValueError(f"{table_name} has unknown label_semantics: {unknown}")
        if "threshold_bps" in frame.columns:
            thresholds = pd.to_numeric(frame["threshold_bps"], errors="raise")
            if (thresholds < 0).any():
                raise ValueError(f"{table_name} has negative threshold_bps")
        probability_columns = [
            column
            for column in ("retained_pct", "test_up_pct")
            if column in frame.columns
        ]
        for column in probability_columns:
            values = pd.to_numeric(frame[column], errors="raise")
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{table_name} has {column} outside [0, 1]")
        if "n_test_windows" in frame.columns:
            values = pd.to_numeric(frame["n_test_windows"], errors="raise")
            if (values < 0).any():
                raise ValueError(f"{table_name} has negative n_test_windows")
        if table_name == "by_model_ticker" and (frame["ticker"] == "pooled").any():
            raise ValueError("by_model_ticker must not contain pooled rows")


def read_seed_result_rows(run_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, run in run_summary.iterrows():
        run_dir_value = run["run_dir"]
        if pd.isna(run_dir_value):
            raise ValueError(f"{run['run_id']} has missing run_dir")
        path = Path(str(run_dir_value)) / "results.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing required seed result file: {path}")
        rows = pd.read_csv(path)
        validate_columns(rows, SEED_RESULT_COLUMNS, path)
        rows = rows.loc[rows["run_id"] == run["run_id"]].copy()
        if rows.empty:
            raise ValueError(f"{path} has no rows for run_id {run['run_id']}")
        for column in (
            "label_semantics",
            "zero_return_policy",
            "no_trade_band_enabled",
            "threshold_bps",
        ):
            rows[column] = run[column]
        rows["expected_seeds"] = ",".join(
            str(seed) for seed in normalize_seed_list(run["seeds"])
        )
        frames.append(rows)
    if not frames:
        raise ValueError("run_summary must contain at least one run for seed tables")
    return pd.concat(frames, ignore_index=True)


def validate_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], path: Path
) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def build_run_gate_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows = run_summary.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "run_id"])
    output = pd.DataFrame(
        {
            "regime": rows["regime"],
            "run_id": rows["run_id"],
            "label_semantics": rows["label_semantics"],
            "zero_return_policy": rows["zero_return_policy"],
            "threshold_bps": rows["threshold_bps"],
            "retained_pct": rows["pooled_retained_pct"],
            "n_test_windows": rows["pooled_test_windows"],
            "label_n_zero_return": rows["pooled_zero_return_rows"],
            "best_model": rows["best_pooled_model"],
            "best_delta_macro_f1_vs_dummy": rows[
                "best_pooled_delta_macro_f1_vs_dummy"
            ],
            "model_expansion_gate": rows["model_expansion_gate"],
            "n_suspicious_rows": rows["n_suspicious_rows"],
        }
    )
    return output.reset_index(drop=True)


def build_pooled_model_table(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = pooled.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "_model_order", "model_name"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "model_name",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "dummy_stratified_macro_f1_mean",
            "dummy_stratified_macro_f1_std",
            "delta_macro_f1_vs_dummy_mean",
            "delta_macro_f1_vs_dummy_std",
            "n_test_windows",
            "retained_pct",
        ],
    ].reset_index(drop=True)


def build_canonical_ticker_table(by_model_ticker: pd.DataFrame) -> pd.DataFrame:
    rows = by_model_ticker.loc[
        by_model_ticker["label_semantics"] == "canonical_phase1_full_binary"
    ].copy()
    if rows.empty:
        raise ValueError("canonical ticker rows are required for paper tables")
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows["positive_delta"] = rows["delta_macro_f1_vs_ticker_dummy_mean"] > 0
    rows = rows.sort_values(["_model_order", "model_name", "ticker"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "model_name",
            "ticker",
            "macro_f1_mean",
            "ticker_dummy_stratified_macro_f1_mean",
            "delta_macro_f1_vs_ticker_dummy_mean",
            "delta_macro_f1_vs_ticker_dummy_std",
            "positive_delta",
            "n_test_windows",
            "retained_pct",
        ],
    ].reset_index(drop=True)


def build_ticker_delta_counts(by_model_ticker: pd.DataFrame) -> pd.DataFrame:
    rows = by_model_ticker.copy()
    if rows.empty:
        raise ValueError("ticker rows are required for ticker delta counts")
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows["positive_delta"] = rows["delta_macro_f1_vs_ticker_dummy_mean"] > 0

    records = []
    group_columns = [
        "_regime_order",
        "_model_order",
        "regime",
        "run_id",
        "model_name",
        "threshold_bps",
    ]
    for group_values, group in rows.groupby(group_columns, sort=True):
        group_keys = dict(zip(group_columns, group_values))
        deltas = pd.to_numeric(
            group["delta_macro_f1_vs_ticker_dummy_mean"], errors="raise"
        )
        n_positive = int(group["positive_delta"].sum())
        n_tickers = int(group["ticker"].nunique())
        records.append(
            {
                "regime": group_keys["regime"],
                "run_id": group_keys["run_id"],
                "model_name": group_keys["model_name"],
                "threshold_bps": group_keys["threshold_bps"],
                "n_tickers": n_tickers,
                "n_positive_delta": n_positive,
                "n_non_positive_delta": int(n_tickers - n_positive),
                "all_non_positive_delta": bool(n_positive == 0),
                "mean_delta_macro_f1_vs_ticker_dummy": float(deltas.mean()),
                "min_delta_macro_f1_vs_ticker_dummy": float(deltas.min()),
                "max_delta_macro_f1_vs_ticker_dummy": float(deltas.max()),
                "retained_pct_mean": float(
                    pd.to_numeric(group["retained_pct"], errors="raise").mean()
                ),
                "n_test_windows_sum": int(
                    pd.to_numeric(group["n_test_windows"], errors="raise").sum()
                ),
            }
        )
    return pd.DataFrame(records).reset_index(drop=True)


def build_seed_ticker_stability(seed_results: pd.DataFrame) -> pd.DataFrame:
    rows = seed_results.loc[seed_results["ticker"] != "pooled"].copy()
    if rows.empty:
        raise ValueError("per-ticker seed rows are required for seed stability")
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    rows["delta_macro_f1_vs_ticker_dummy"] = pd.to_numeric(
        rows["delta_macro_f1_vs_ticker_dummy"], errors="raise"
    )
    rows["model_macro_f1"] = pd.to_numeric(rows["model_macro_f1"], errors="raise")
    rows["positive_seed_delta"] = rows["delta_macro_f1_vs_ticker_dummy"] > 0
    duplicate_mask = rows.duplicated(["run_id", "model_name", "ticker", "seed"], keep=False)
    if duplicate_mask.any():
        duplicate = rows.loc[duplicate_mask, ["run_id", "model_name", "ticker", "seed"]].iloc[0]
        raise ValueError(
            "duplicate seed row in results.csv: "
            f"{duplicate['run_id']} {duplicate['model_name']} "
            f"{duplicate['ticker']} seed={duplicate['seed']}"
        )

    records = []
    group_columns = [
        "_regime_order",
        "_model_order",
        "regime",
        "run_id",
        "model_name",
        "ticker",
        "threshold_bps",
    ]
    for group_values, group in rows.groupby(group_columns, sort=True):
        group_keys = dict(zip(group_columns, group_values))
        context = (
            f"{group_keys['run_id']} {group_keys['model_name']} "
            f"{group_keys['ticker']}"
        )
        deltas = group["delta_macro_f1_vs_ticker_dummy"]
        seeds = sorted(group["seed"].unique().tolist())
        expected_seeds = normalize_seed_list(group["expected_seeds"].iloc[0])
        if seeds != expected_seeds:
            raise ValueError(
                f"{context} has seed list {seeds}, expected {expected_seeds}"
            )
        if len(seeds) < 2:
            raise ValueError(f"{context} needs at least two seeds for stability")
        n_positive = int(group["positive_seed_delta"].sum())
        records.append(
            {
                "regime": group_keys["regime"],
                "run_id": group_keys["run_id"],
                "model_name": group_keys["model_name"],
                "ticker": group_keys["ticker"],
                "threshold_bps": group_keys["threshold_bps"],
                "seeds": ",".join(str(seed) for seed in seeds),
                "n_seeds": len(seeds),
                "delta_mean": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
                "n_positive_seed_delta": n_positive,
                "positive_seed_rate": float(n_positive / len(seeds)),
                "all_seeds_positive": bool(n_positive == len(seeds)),
                "all_seeds_non_positive": bool(n_positive == 0),
                "retained_pct": require_single_numeric_value(
                    group, "retained_pct", context
                ),
                "n_test_windows": int(
                    require_single_numeric_value(group, "n_test_windows", context)
                ),
                "n_suspicious_seed_rows": int(
                    (group["suspicious_status"] != "ok").sum()
                ),
            }
        )
    return pd.DataFrame(records).reset_index(drop=True)


def build_regime_shift_by_ticker(by_model_ticker: pd.DataFrame) -> pd.DataFrame:
    rows = by_model_ticker.copy()
    if rows.empty:
        raise ValueError("ticker rows are required for regime shift table")
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    duplicate_mask = rows.duplicated(["regime", "model_name", "ticker"], keep=False)
    if duplicate_mask.any():
        duplicate = rows.loc[duplicate_mask, ["regime", "model_name", "ticker"]].iloc[0]
        raise ValueError(
            "duplicate regime/model/ticker rows in by_model_ticker: "
            f"{duplicate['regime']} {duplicate['model_name']} {duplicate['ticker']}"
        )

    records = []
    for (model_name, ticker), group in rows.groupby(["model_name", "ticker"], sort=True):
        canonical_delta = regime_value(
            group, "canonical_full_binary", "delta_macro_f1_vs_ticker_dummy_mean"
        )
        if pd.isna(canonical_delta):
            continue
        diagnostic_0bps_delta = regime_value(
            group,
            "0bps_no_trade_band_diagnostic",
            "delta_macro_f1_vs_ticker_dummy_mean",
        )
        diagnostic_5bps_delta = regime_value(
            group,
            "5bps_no_trade_band_diagnostic",
            "delta_macro_f1_vs_ticker_dummy_mean",
        )
        canonical_retained_pct = regime_value(
            group, "canonical_full_binary", "retained_pct"
        )
        retained_pct_0bps = regime_value(
            group, "0bps_no_trade_band_diagnostic", "retained_pct"
        )
        retained_pct_5bps = regime_value(
            group, "5bps_no_trade_band_diagnostic", "retained_pct"
        )
        records.append(
            {
                "model_name": model_name,
                "ticker": ticker,
                "canonical_delta": canonical_delta,
                "diagnostic_0bps_delta": diagnostic_0bps_delta,
                "diagnostic_5bps_delta": diagnostic_5bps_delta,
                "delta_0bps_minus_canonical": numeric_difference(
                    diagnostic_0bps_delta, canonical_delta
                ),
                "delta_5bps_minus_canonical": numeric_difference(
                    diagnostic_5bps_delta, canonical_delta
                ),
                "canonical_retained_pct": canonical_retained_pct,
                "retained_pct_0bps": retained_pct_0bps,
                "retained_pct_5bps": retained_pct_5bps,
                "coverage_drop_5bps": numeric_difference(
                    canonical_retained_pct, retained_pct_5bps
                ),
            }
        )
    if not records:
        raise ValueError("canonical ticker rows are required for regime shift table")
    output = pd.DataFrame(records)
    output["_model_order"] = output["model_name"].map(MODEL_ORDER).fillna(99)
    output = output.sort_values(["_model_order", "model_name", "ticker"])
    return output.drop(columns=["_model_order"]).reset_index(drop=True)


def build_coverage_fragility_flags(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = coverage.copy()
    if rows.empty:
        raise ValueError("coverage rows are required for fragility flags")
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_ticker_order"] = rows["ticker"].apply(ticker_order)
    rows["retained_pct"] = pd.to_numeric(rows["retained_pct"], errors="raise")
    rows["n_test_windows"] = pd.to_numeric(rows["n_test_windows"], errors="raise")
    rows["test_up_pct"] = pd.to_numeric(rows["test_up_pct"], errors="raise")
    rows["low_coverage_flag"] = rows["retained_pct"] < LOW_COVERAGE_THRESHOLD
    rows["low_test_window_flag"] = rows["n_test_windows"] < LOW_TEST_WINDOWS_THRESHOLD
    rows["class_balance_edge_flag"] = (
        (rows["test_up_pct"] < CLASS_BALANCE_LOWER_BOUND)
        | (rows["test_up_pct"] > CLASS_BALANCE_UPPER_BOUND)
    )
    rows["diagnostic_only"] = (
        rows["label_semantics"] != "canonical_phase1_full_binary"
    )
    rows["claim_scope"] = rows.apply(coverage_claim_scope, axis=1)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "_ticker_order", "ticker"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "ticker",
            "threshold_bps",
            "retained_pct",
            "n_test_windows",
            "test_up_pct",
            "label_n_neutral",
            "label_n_zero_return",
            "low_coverage_flag",
            "low_test_window_flag",
            "class_balance_edge_flag",
            "diagnostic_only",
            "claim_scope",
        ],
    ].reset_index(drop=True)


def build_coverage_table(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = coverage.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_ticker_order"] = rows["ticker"].apply(ticker_order)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "_ticker_order", "ticker"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "ticker",
            "label_semantics",
            "zero_return_policy",
            "threshold_bps",
            "retained_pct",
            "label_n_neutral",
            "label_n_zero_return",
            "n_test_windows",
            "test_up_pct",
        ],
    ].reset_index(drop=True)


def build_threshold_retention_proxy(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows = run_summary.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "run_id"])
    return pd.DataFrame(
        {
            "proxy_kind": "threshold_retention_not_confidence",
            "regime": rows["regime"],
            "run_id": rows["run_id"],
            "threshold_bps": rows["threshold_bps"],
            "retained_pct": rows["pooled_retained_pct"],
            "n_test_windows": rows["pooled_test_windows"],
            "best_model": rows["best_pooled_model"],
            "best_delta_macro_f1_vs_dummy": rows[
                "best_pooled_delta_macro_f1_vs_dummy"
            ],
            "model_expansion_gate": rows["model_expansion_gate"],
        }
    ).reset_index(drop=True)


def build_delta_coverage_data(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = pooled.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows = rows.sort_values(["_regime_order", "threshold_bps", "_model_order", "model_name"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "model_name",
            "threshold_bps",
            "retained_pct",
            "macro_f1_mean",
            "balanced_accuracy_mean",
            "delta_macro_f1_vs_dummy_mean",
        ],
    ].reset_index(drop=True)


def build_ticker_heatmap_data(by_model_ticker: pd.DataFrame) -> pd.DataFrame:
    rows = by_model_ticker.copy()
    rows["regime"] = rows.apply(regime_label, axis=1)
    rows["_regime_order"] = rows.apply(regime_order, axis=1)
    rows["_model_order"] = rows["model_name"].map(MODEL_ORDER).fillna(99)
    rows["positive_delta"] = rows["delta_macro_f1_vs_ticker_dummy_mean"] > 0
    rows = rows.sort_values(["_regime_order", "threshold_bps", "_model_order", "ticker"])
    return rows.loc[
        :,
        [
            "regime",
            "run_id",
            "model_name",
            "ticker",
            "threshold_bps",
            "delta_macro_f1_vs_ticker_dummy_mean",
            "positive_delta",
            "retained_pct",
            "n_test_windows",
        ],
    ].reset_index(drop=True)


def regime_label(row: pd.Series) -> str:
    if row["label_semantics"] == "canonical_phase1_full_binary":
        return "canonical_full_binary"
    if row["label_semantics"] != "legacy_runner_no_trade_band_diagnostic":
        raise ValueError(f"unknown label_semantics: {row['label_semantics']}")
    threshold = format_threshold(row["threshold_bps"])
    return f"{threshold}bps_no_trade_band_diagnostic"


def regime_order(row: pd.Series) -> tuple[int, float]:
    if row["label_semantics"] == "canonical_phase1_full_binary":
        return (0, 0.0)
    return (1, float(row["threshold_bps"]))


def format_threshold(value: Any) -> str:
    numeric = float(value)
    if numeric < 0:
        raise ValueError(f"threshold_bps must be non-negative, got {value}")
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def ticker_order(ticker: str) -> tuple[int, str]:
    if ticker == "pooled":
        return (0, ticker)
    return (1, ticker)


def require_single_numeric_value(
    group: pd.DataFrame, column: str, context: str
) -> float:
    values = pd.to_numeric(group[column], errors="raise").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{context} has inconsistent {column} values")
    return float(values[0])


def regime_value(group: pd.DataFrame, regime: str, column: str) -> Any:
    values = group.loc[group["regime"] == regime, column]
    if values.empty:
        return pd.NA
    return values.iloc[0]


def numeric_difference(left: Any, right: Any) -> Any:
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return float(left) - float(right)


def coverage_claim_scope(row: pd.Series) -> str:
    if row["diagnostic_only"] and row["low_coverage_flag"]:
        return "diagnostic_low_coverage_descriptive_only"
    if row["diagnostic_only"]:
        return "diagnostic_descriptive_only"
    if row["low_coverage_flag"]:
        return "canonical_low_coverage_descriptive_only"
    return "canonical_descriptive"


def normalize_seed_list(value: Any) -> list[int]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    seeds = sorted({int(str(seed).strip()) for seed in raw_values if str(seed).strip()})
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    return seeds


def write_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_ORDER:
        outputs[name].to_csv(output_dir / f"{name}.csv", index=False)
    write_markdown_report(output_dir / "paper_tables.md", outputs)


def write_markdown_report(path: Path, outputs: dict[str, pd.DataFrame]) -> None:
    sections = [
        ("Table 1 - Run Gate Summary", "paper_table_1_run_gate_summary"),
        ("Table 2 - Pooled Model Vs Dummy", "paper_table_2_pooled_model_vs_dummy"),
        ("Table 3 - Canonical Per-Ticker Delta", "paper_table_3_canonical_ticker_delta"),
        ("Table 4 - Coverage And Label Semantics", "paper_table_4_coverage_label_semantics"),
        ("Table 5 - Ticker Delta Counts", "paper_table_5_ticker_delta_counts"),
        ("Table 6 - Seed Ticker Stability", "paper_table_6_seed_ticker_stability"),
        ("Table 7 - Regime Shift By Ticker", "paper_table_7_regime_shift_by_ticker"),
        ("Table 8 - Coverage Fragility Flags", "paper_table_8_coverage_fragility_flags"),
        ("Figure Data - Delta Vs Coverage", "figure_delta_vs_coverage"),
        ("Figure Data - Ticker Delta Heatmap", "figure_ticker_delta_heatmap"),
        ("Figure Data - Threshold Retention Proxy", "figure_threshold_retention_proxy"),
    ]
    lines = ["# Protocol-Safe Paper Tables", ""]
    for title, key in sections:
        lines.extend([f"## {title}", "", dataframe_to_markdown(outputs[key]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(format_markdown_value(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
