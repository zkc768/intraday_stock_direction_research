"""08O official-validation readout artifact builders.

This module consumes already-frozen official-validation prediction rows. It
does not train models, choose candidates, tune thresholds, or read holdout/test.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    check_08o_real_readout_completeness,
    validate_08o_run_manifest,
    validate_freeze_record,
)
from intraday_research.models.deep_sequence.metrics import compute_trial_metrics
from intraday_research.stages.io_helpers import sha256_bytes, sha256_file
from intraday_research.stages.run_manifest import (
    write_run_manifest as write_run_manifest_json,
)


OFFICIAL_READOUT_INPUT_COLUMNS: tuple[str, ...] = (
    "seed",
    "ticker",
    "candidate_id",
    "official_validation_row_id",
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


def build_08o_readout_frames(
    predictions: pd.DataFrame,
    *,
    primary_candidate_id: str | None = None,
    expected_seeds: tuple[int, ...] | None = None,
    expected_tickers: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the 08O CSV artifact frames from frozen prediction rows."""
    frame = _validate_prediction_frame(
        predictions,
        primary_candidate_id=primary_candidate_id,
        expected_seeds=expected_seeds,
        expected_tickers=expected_tickers,
    )
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
    *,
    primary_candidate_id: str | None = None,
    expected_seeds: tuple[int, ...] | None = None,
    expected_tickers: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write 08O readout CSV artifacts and return the strict completeness verdict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = build_08o_readout_frames(
        predictions,
        primary_candidate_id=primary_candidate_id,
        expected_seeds=expected_seeds,
        expected_tickers=expected_tickers,
    )
    for filename, frame in frames.items():
        frame.to_csv(out / filename, index=False, lineterminator="\n")
    verdict = check_08o_real_readout_completeness(out)
    if not verdict["is_real_readout"]:
        raise AssertionError(f"08O readout artifacts did not pass real-mode gate: {verdict}")
    return verdict


def preflight_08o_static_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Validate non-metric 08O inputs before official prediction rows are read."""
    freeze_record_path = paths["freeze_record"]
    decision_record_path = paths["decision_record"]
    freeze_record = _read_json(freeze_record_path, field_name="08F freeze record")
    validate_freeze_record(freeze_record)
    decision_record = _read_json(decision_record_path, field_name="08O decision record")
    _validate_08o_decision_record(decision_record, freeze_record_path)
    freeze_sha = sha256_file(freeze_record_path)
    recorded_freeze_sha = decision_record.get("freeze_record_sha256")
    if recorded_freeze_sha and str(recorded_freeze_sha) != freeze_sha:
        raise ValueError(
            "08O decision record freeze_record_sha256 does not match "
            f"{freeze_record_path}: {recorded_freeze_sha!r} != {freeze_sha!r}"
        )
    return {
        "freeze_record": freeze_record,
        "decision_record": decision_record,
        "static_input_provenance": {
            "freeze_record_path": str(freeze_record_path),
            "freeze_record_sha256": freeze_sha,
            "decision_record_path": str(decision_record_path),
            "decision_record_sha256": sha256_file(decision_record_path),
        },
    }


def preflight_08o_prediction_rows(
    predictions: pd.DataFrame,
    *,
    freeze_record: Mapping[str, Any],
    expected_tickers: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate official-readout prediction provenance and row completeness."""
    frame = _validate_prediction_frame(
        predictions,
        primary_candidate_id=str(freeze_record["primary_candidate_id"]),
        expected_seeds=_expected_seeds_from_freeze_record(freeze_record),
        expected_tickers=expected_tickers,
    )
    row_ids_by_seed = {
        int(seed): sorted(group["official_validation_row_id"].astype(str).tolist())
        for seed, group in frame.groupby("seed", sort=True)
    }
    row_id_hash_by_seed = {
        seed: sha256_bytes("\n".join(row_ids).encode("utf-8"))
        for seed, row_ids in row_ids_by_seed.items()
    }
    return {
        "candidate_id": str(freeze_record["primary_candidate_id"]),
        "prediction_row_count": int(len(frame)),
        "official_validation_row_id_count": int(
            frame["official_validation_row_id"].nunique()
        ),
        "seeds": sorted(int(seed) for seed in frame["seed"].unique()),
        "tickers": sorted(str(ticker) for ticker in frame["ticker"].unique()),
        "same_official_rows_for_each_seed": True,
        "row_id_set_sha256_by_seed": row_id_hash_by_seed,
    }


def write_08o_run_manifest(
    output_dir: Path | str,
    *,
    freeze_record: Mapping[str, Any],
    static_input_provenance: Mapping[str, Any],
    prediction_provenance: Mapping[str, Any],
    constants: Mapping[str, Any] | None = None,
    readout_started_at_utc: str | None = None,
) -> dict[str, Any]:
    """Write the package-side 08O run manifest from current artifact state."""
    out = Path(output_dir)
    completeness = check_08o_real_readout_completeness(out)
    schema_only_stub = not bool(completeness["is_real_readout"])
    payload = {
        "stage": "08O",
        "scope": "validation_only",
        "primary_candidate_id": str(freeze_record["primary_candidate_id"]),
        "freeze_record_sha256": str(static_input_provenance["freeze_record_sha256"]),
        "official_validation_readout_started_at": (
            readout_started_at_utc or _utc_now_iso()
        ),
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
        "same_row_dummy_present": _artifact_passed(
            completeness, "08o_same_row_baselines.csv"
        ),
        "per_ticker_present": _artifact_passed(
            completeness, "08o_validation_per_ticker.csv"
        ),
        "seed_summary_present": _artifact_passed(
            completeness, "08o_seed_summary.csv"
        ),
        "allowed_wording_bucket": _derive_allowed_wording_bucket(
            out,
            freeze_record=freeze_record,
            completeness=completeness,
            constants=constants or {},
        ),
        "schema_only_stub": schema_only_stub,
        "input_provenance": {
            **dict(static_input_provenance),
            "predictions": dict(prediction_provenance),
        },
        "completeness_verdict": completeness,
        "manifest_written_at_utc": _utc_now_iso(),
    }
    return write_run_manifest_json(
        out / "08o_run_manifest.json",
        payload,
        validator=validate_08o_run_manifest,
        stage="08O",
        scope="validation_only",
        false_fields=(
            "official_validation_used_for_selection",
            "holdout_test_authorized",
        ),
    )


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    primary_candidate_id: str | None = None,
    expected_seeds: tuple[int, ...] | None = None,
    expected_tickers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
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
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["official_validation_row_id"] = frame["official_validation_row_id"].astype(
        str
    )
    if (frame["ticker"].str.strip() == "").any():
        raise ValueError("08O prediction rows contain empty ticker")
    if (frame["candidate_id"].str.strip() == "").any():
        raise ValueError("08O prediction rows contain empty candidate_id")
    if (frame["official_validation_row_id"].str.strip() == "").any():
        raise ValueError("08O prediction rows contain empty official_validation_row_id")
    candidate_ids = sorted(frame["candidate_id"].unique())
    if len(candidate_ids) != 1:
        raise ValueError(
            "08O prediction rows must contain exactly one candidate_id: "
            f"{candidate_ids}"
        )
    if primary_candidate_id is not None and candidate_ids[0] != primary_candidate_id:
        raise ValueError(
            "08O prediction candidate_id must match frozen primary_candidate_id: "
            f"{candidate_ids[0]!r} != {primary_candidate_id!r}"
        )
    duplicate_row_ids = frame.duplicated(["seed", "official_validation_row_id"])
    if duplicate_row_ids.any():
        raise ValueError(
            "08O prediction rows contain duplicate official_validation_row_id within a seed"
        )
    for column in ("y_true", "y_pred"):
        bad = sorted(set(frame[column].astype(int).unique()) - {0, 1})
        if bad:
            raise ValueError(f"08O {column} must be in {{0, 1}}; got {bad}")
    observed_seeds = tuple(sorted(int(seed) for seed in frame["seed"].unique()))
    if expected_seeds is not None and observed_seeds != tuple(sorted(expected_seeds)):
        raise ValueError(
            f"08O prediction seeds must match frozen_seed_list: {observed_seeds} "
            f"!= {tuple(sorted(expected_seeds))}"
        )
    observed_tickers = tuple(sorted(str(ticker) for ticker in frame["ticker"].unique()))
    if expected_tickers is not None and observed_tickers != tuple(
        sorted(expected_tickers)
    ):
        raise ValueError(
            f"08O prediction tickers must match expected tickers: {observed_tickers} "
            f"!= {tuple(sorted(expected_tickers))}"
        )
    row_id_sets = [
        set(group["official_validation_row_id"].astype(str))
        for _, group in frame.groupby("seed", sort=True)
    ]
    first_row_ids = row_id_sets[0]
    if any(row_ids != first_row_ids for row_ids in row_id_sets[1:]):
        raise ValueError(
            "08O prediction rows must cover the same official_validation_row_id "
            "set for every frozen seed"
        )
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


def _read_json(path: Path, *, field_name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{field_name} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"{field_name} is not valid JSON: {path}") from err
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a JSON object: {path}")
    return payload


def _validate_08o_decision_record(
    decision_record: Mapping[str, Any],
    freeze_record_path: Path,
) -> None:
    if bool(decision_record.get("official_validation_used_for_selection", False)):
        raise ValueError(
            "08O decision record says official validation was used for selection"
        )
    if bool(decision_record.get("holdout_test_authorized", False)):
        raise ValueError("08O decision record authorizes holdout/test contact")
    recorded_path = decision_record.get("freeze_record_path")
    if recorded_path and Path(str(recorded_path)).name != freeze_record_path.name:
        raise ValueError(
            "08O decision record freeze_record_path does not match configured "
            f"freeze record: {recorded_path!r} != {str(freeze_record_path)!r}"
        )


def _expected_seeds_from_freeze_record(
    freeze_record: Mapping[str, Any],
) -> tuple[int, ...]:
    seeds = freeze_record.get("frozen_seed_list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("08F freeze record must carry a non-empty frozen_seed_list")
    return tuple(sorted(int(seed) for seed in seeds))


def _artifact_passed(completeness: Mapping[str, Any], filename: str) -> bool:
    verdict = completeness["per_artifact"][filename]
    return (
        bool(verdict["present"])
        and bool(verdict["non_empty"])
        and bool(verdict["schema_complete"])
    )


def _derive_allowed_wording_bucket(
    output_dir: Path,
    *,
    freeze_record: Mapping[str, Any],
    completeness: Mapping[str, Any],
    constants: Mapping[str, Any],
) -> str:
    if not bool(completeness["is_real_readout"]):
        return "no_candidate_freezable"
    if bool(freeze_record.get("low_compute_baseline", False)):
        return "weak_mixed"
    seed_summary = pd.read_csv(output_dir / "08o_seed_summary.csv")
    concentration = pd.read_csv(output_dir / "08o_concentration_guardrails.csv")
    delta_lcb = _metric_value(
        seed_summary,
        metric="delta_macro_f1_vs_stratified_dummy_same_rows",
        column="seed_lcb_95",
    )
    macro_f1_seed_std = _metric_value(
        seed_summary,
        metric="macro_f1",
        column="seed_std",
    )
    positive_ticker_count = _guardrail_value(concentration, "positive_ticker_count")
    seed_std_max = float(
        constants.get("tier_escalation_medium_to_aggressive_seed_std_max", 0.01)
    )
    if macro_f1_seed_std > seed_std_max:
        return "unstable"
    delta_threshold = float(
        constants.get("improvement_threshold_delta_macro_f1_lcb_95", 0.005)
    )
    ticker_threshold = int(
        constants.get("improvement_threshold_positive_ticker_count_min", 4)
    )
    if delta_lcb >= delta_threshold and positive_ticker_count >= ticker_threshold:
        return "improvement"
    return "weak_mixed"


def _metric_value(frame: pd.DataFrame, *, metric: str, column: str) -> float:
    matches = frame.loc[frame["metric"] == metric, column]
    if len(matches) != 1:
        raise ValueError(f"08O seed summary missing unique {metric!r} {column!r}")
    return float(matches.iloc[0])


def _guardrail_value(frame: pd.DataFrame, guardrail: str) -> float:
    matches = frame.loc[frame["guardrail"] == guardrail, "value"]
    if len(matches) != 1:
        raise ValueError(f"08O concentration guardrails missing unique {guardrail!r}")
    return float(matches.iloc[0])


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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
    if not isinstance(policy, Mapping):
        raise ValueError("config['policy'] must be a mapping for 08O readout")
    ledger_cfg = policy.get("validation_budget_ledger", {})
    if ledger_cfg is None:
        ledger_cfg = {}
    if not isinstance(ledger_cfg, Mapping):
        raise ValueError(
            "config['policy']['validation_budget_ledger'] must be a mapping "
            "for 08O readout"
        )
    predictions = inputs.get("official_validation_predictions_csv")
    decision_record = inputs.get("08o_decision_record")
    freeze_record = (
        inputs.get("08f_candidate_freeze_record")
        or inputs.get("08f_candidate_freeze_record_json")
    )
    ledger = inputs.get("validation_budget_ledger") or ledger_cfg.get("path")
    missing = [
        name for name, value in (
            ("inputs.official_validation_predictions_csv", predictions),
            ("inputs.08o_decision_record", decision_record),
            ("inputs.08f_candidate_freeze_record", freeze_record),
            ("validation_budget_ledger.path", ledger),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"08O readout missing required config path(s): {missing}")
    paths = {
        "predictions": Path(str(predictions)),
        "decision_record": Path(str(decision_record)),
        "freeze_record": Path(str(freeze_record)),
        "ledger": Path(str(ledger)),
    }
    reject_holdout_test_filename(paths["predictions"], field_name="official predictions")
    return paths
