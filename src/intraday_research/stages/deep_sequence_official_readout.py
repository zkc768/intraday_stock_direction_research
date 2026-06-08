"""08O official-validation readout artifact builders.

This module consumes already-frozen official-validation prediction rows. It
does not train models, choose candidates, tune thresholds, or read holdout/test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    check_08o_real_readout_completeness,
)
from intraday_research.models.deep_sequence.metrics import compute_trial_metrics


OFFICIAL_READOUT_INPUT_COLUMNS: tuple[str, ...] = (
    "seed",
    "ticker",
    "y_true",
    "y_pred",
)
VALIDATION_READOUT_COLUMNS: tuple[str, ...] = (
    "seed",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "delta_macro_f1_vs_stratified_dummy_same_rows",
    "delta_balanced_accuracy_vs_stratified_dummy_same_rows",
    "validation_n",
    "class0_pred_rate",
    "class1_pred_rate",
)
PER_TICKER_COLUMNS: tuple[str, ...] = (
    "ticker",
    "macro_f1",
    "delta_macro_f1_vs_dummy",
    "n_rows",
)
SEED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "metric",
    "seed_mean",
    "seed_std",
    "seed_lcb_95",
)
SAME_ROW_BASELINES_COLUMNS: tuple[str, ...] = (
    "baseline",
    "macro_f1_mean",
    "macro_f1_std",
)
CONCENTRATION_GUARDRAILS_COLUMNS: tuple[str, ...] = (
    "guardrail",
    "value",
    "threshold",
    "downgrade_triggered",
)
FAILURE_ROWS_COLUMNS: tuple[str, ...] = (
    "seed",
    "failure_type",
    "failure_message",
)
_FORBIDDEN_INPUT_COLUMN_TOKENS = ("holdout", "test", "selection")


def build_08o_readout_frames(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the 08O CSV artifact frames from frozen prediction rows."""
    frame = _validate_prediction_frame(predictions)
    seed_rows = []
    for seed, group in frame.groupby("seed", sort=True):
        metrics = compute_trial_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            group["ticker"].astype(str).to_numpy(),
        )
        seed_rows.append(
            {
                "seed": int(seed),
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "delta_macro_f1_vs_stratified_dummy_same_rows": (
                    metrics["delta_macro_f1_vs_dummy"]
                ),
                "delta_balanced_accuracy_vs_stratified_dummy_same_rows": (
                    metrics["balanced_accuracy"] - 0.5
                ),
                "validation_n": int(len(group)),
                "class0_pred_rate": metrics["class0_pred_rate"],
                "class1_pred_rate": metrics["class1_pred_rate"],
            }
        )
    validation_readout = pd.DataFrame(seed_rows, columns=VALIDATION_READOUT_COLUMNS)

    ticker_rows = []
    for ticker, group in frame.groupby("ticker", sort=True):
        metrics = compute_trial_metrics(
            group["y_true"].to_numpy(),
            group["y_pred"].to_numpy(),
            group["ticker"].astype(str).to_numpy(),
        )
        ticker_rows.append(
            {
                "ticker": str(ticker),
                "macro_f1": metrics["macro_f1"],
                "delta_macro_f1_vs_dummy": metrics["delta_macro_f1_vs_dummy"],
                "n_rows": int(len(group)),
            }
        )
    per_ticker = pd.DataFrame(ticker_rows, columns=PER_TICKER_COLUMNS)

    seed_summary = _build_seed_summary(validation_readout)
    same_row_baselines = _build_same_row_baselines(validation_readout)
    concentration = _build_concentration_guardrails(frame, per_ticker)
    failures = pd.DataFrame(columns=FAILURE_ROWS_COLUMNS)

    return {
        "08o_validation_readout.csv": validation_readout,
        "08o_validation_per_ticker.csv": per_ticker,
        "08o_seed_summary.csv": seed_summary,
        "08o_same_row_baselines.csv": same_row_baselines,
        "08o_concentration_guardrails.csv": concentration,
        "08o_failure_rows.csv": failures,
    }


def write_08o_readout_artifacts(
    output_dir: Path | str,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Write 08O readout CSV artifacts and return the strict completeness verdict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = build_08o_readout_frames(predictions)
    for filename, frame in frames.items():
        frame.to_csv(out / filename, index=False, lineterminator="\n")
    verdict = check_08o_real_readout_completeness(out)
    if not verdict["is_real_readout"]:
        raise AssertionError(f"08O readout artifacts did not pass real-mode gate: {verdict}")
    return verdict


def _validate_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = set(OFFICIAL_READOUT_INPUT_COLUMNS) - set(predictions.columns)
    if missing:
        raise ValueError(f"08O prediction rows missing columns: {sorted(missing)}")
    forbidden = [
        col for col in predictions.columns
        if any(token in str(col).lower() for token in _FORBIDDEN_INPUT_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"08O prediction rows contain forbidden columns: {forbidden}")
    if predictions.empty:
        raise ValueError("08O prediction rows must be non-empty")

    frame = predictions.loc[:, OFFICIAL_READOUT_INPUT_COLUMNS].copy()
    for column in OFFICIAL_READOUT_INPUT_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"08O prediction rows contain missing {column!r}")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="raise").astype(int)
    frame["ticker"] = frame["ticker"].astype(str)
    if (frame["ticker"].str.strip() == "").any():
        raise ValueError("08O prediction rows contain empty ticker")
    for column in ("y_true", "y_pred"):
        bad = sorted(set(frame[column].astype(int).unique()) - {0, 1})
        if bad:
            raise ValueError(f"08O {column} must be in {{0, 1}}; got {bad}")
    for seed, group in frame.groupby("seed", sort=True):
        classes = set(group["y_true"].astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(
                f"08O seed {seed} y_true must contain both classes 0 and 1; "
                f"got {sorted(classes)}"
            )
    return frame


def _build_seed_summary(validation_readout: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "delta_macro_f1_vs_stratified_dummy_same_rows",
        "delta_balanced_accuracy_vs_stratified_dummy_same_rows",
    )
    rows = []
    for metric in metrics:
        values = validation_readout[metric].astype(float)
        std = _sample_std(values)
        rows.append(
            {
                "metric": metric,
                "seed_mean": float(values.mean()),
                "seed_std": std,
                "seed_lcb_95": float(values.mean() - 1.96 * std / math.sqrt(len(values))),
            }
        )
    return pd.DataFrame(rows, columns=SEED_SUMMARY_COLUMNS)


def _build_same_row_baselines(validation_readout: pd.DataFrame) -> pd.DataFrame:
    dummy_values = (
        validation_readout["macro_f1"]
        - validation_readout["delta_macro_f1_vs_stratified_dummy_same_rows"]
    )
    return pd.DataFrame(
        [
            {
                "baseline": "stratified_dummy_same_rows",
                "macro_f1_mean": float(dummy_values.mean()),
                "macro_f1_std": _sample_std(dummy_values),
            }
        ],
        columns=SAME_ROW_BASELINES_COLUMNS,
    )


def _build_concentration_guardrails(
    predictions: pd.DataFrame,
    per_ticker: pd.DataFrame,
) -> pd.DataFrame:
    ticker_share = predictions["ticker"].value_counts(normalize=True)
    positive_ticker_count = int((per_ticker["delta_macro_f1_vs_dummy"] > 0).sum())
    return pd.DataFrame(
        [
            {
                "guardrail": "ticker_max_share",
                "value": float(ticker_share.max()),
                "threshold": 0.5,
                "downgrade_triggered": bool(ticker_share.max() > 0.5),
            },
            {
                "guardrail": "positive_ticker_count",
                "value": positive_ticker_count,
                "threshold": 4,
                "downgrade_triggered": bool(positive_ticker_count < 4),
            },
        ],
        columns=CONCENTRATION_GUARDRAILS_COLUMNS,
    )


def _sample_std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.astype(float).std(ddof=1))


def reject_holdout_test_filename(path: Path | str, *, field_name: str) -> None:
    """Fail closed if an input filename advertises holdout/test contact."""
    name = Path(path).name.lower()
    forbidden = [token for token in ("holdout", "test") if token in name]
    if forbidden:
        raise ValueError(
            f"{field_name} filename must not contain holdout/test tokens: {Path(path).name!r}"
        )


def resolve_08o_readout_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve package-stage 08O readout inputs from config."""
    inputs = config.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("config['inputs'] must be a mapping for 08O readout")
    policy = config.get("policy", {})
    ledger_cfg = (
        policy.get("validation_budget_ledger", {})
        if isinstance(policy, Mapping)
        else {}
    )
    predictions = inputs.get("official_validation_predictions_csv")
    decision_record = inputs.get("08o_decision_record")
    ledger = inputs.get("validation_budget_ledger") or ledger_cfg.get("path")
    missing = [
        name for name, value in (
            ("inputs.official_validation_predictions_csv", predictions),
            ("inputs.08o_decision_record", decision_record),
            ("validation_budget_ledger.path", ledger),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"08O readout missing required config path(s): {missing}")
    paths = {
        "predictions": Path(str(predictions)),
        "decision_record": Path(str(decision_record)),
        "ledger": Path(str(ledger)),
    }
    reject_holdout_test_filename(paths["predictions"], field_name="official predictions")
    return paths
