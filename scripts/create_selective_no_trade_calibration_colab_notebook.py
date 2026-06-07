from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_MODULE = (
    PROJECT_ROOT
    / "src"
    / "intraday_research"
    / "contracts"
    / "selective_no_trade_calibration.py"
)
# Phase 3 migration: canonical contract code now lives under
# src/intraday_research/contracts/. The legacy `scripts/notebook06_contract.py`
# path is a thin re-export shim; reading it here would inline only ~21 lines
# of imports and break the generated notebook's downstream cells. Always read
# CONTRACT_MODULE from the canonical src path.
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "06_selective_no_trade_calibration_colab.ipynb"


CONFIG_SOURCE = r'''
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK06_SCOPE = "validation_only"
NOTEBOOK05_RESULTS_DIR = Path("/content/notebook05_lightgbm_tuning_results")
OUTPUT_DIR = Path("/content/notebook06_selective_no_trade_calibration_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_06A_ARTIFACT_GATE = False
RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS = False
RUN_06C_FIXED_COVERAGE_GRID = False
RUN_06D_AGGREGATE_AND_RISK_COVERAGE = False
RUN_06E_CONCENTRATION_GUARDRAILS = False
RUN_06F_DECISION_RECORD = False
RUN_06G_BACKUP_TO_GOOGLE_DRIVE = False

BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE = False
DRIVE_BACKUP_FOLDER_ID = ""
DRIVE_BACKUP_PREFIX = "notebook06_selective_no_trade_calibration"

OPERATOR_ACKNOWLEDGES_VALIDATION_ONLY_SCOPE = False
OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False
OPERATOR_ACKNOWLEDGES_NO_SELECTIVE_THRESHOLD = False

OUTPUT_FILES = {
    "artifact_contract": OUTPUT_DIR / "notebook06_artifact_contract_check.json",
    "prediction_frame_manifest": OUTPUT_DIR / "notebook06_prediction_frame_manifest.csv",
    "probability_diagnostics": OUTPUT_DIR / "notebook06_probability_diagnostics.csv",
    "reliability_bins": OUTPUT_DIR / "notebook06_reliability_bins.csv",
    "coverage_grid": OUTPUT_DIR / "notebook06_coverage_grid.csv",
    "same_row_baselines": OUTPUT_DIR / "notebook06_same_row_baselines.csv",
    "random_abstention_baselines": OUTPUT_DIR / "notebook06_random_abstention_baselines.csv",
    "risk_coverage_summary": OUTPUT_DIR / "notebook06_risk_coverage_summary.csv",
    "concentration_guardrails": OUTPUT_DIR / "notebook06_concentration_guardrails.csv",
    "per_ticker_coverage": OUTPUT_DIR / "notebook06_per_ticker_coverage.csv",
    "decision_record": OUTPUT_DIR / "notebook06_decision_record.json",
    "run_manifest": OUTPUT_DIR / "notebook06_run_manifest.json",
    "drive_backup_manifest": OUTPUT_DIR / "notebook06_drive_backup_manifest.json",
}

RUN_SWITCHES = {
    "RUN_06A_ARTIFACT_GATE": RUN_06A_ARTIFACT_GATE,
    "RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS": RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS,
    "RUN_06C_FIXED_COVERAGE_GRID": RUN_06C_FIXED_COVERAGE_GRID,
    "RUN_06D_AGGREGATE_AND_RISK_COVERAGE": RUN_06D_AGGREGATE_AND_RISK_COVERAGE,
    "RUN_06E_CONCENTRATION_GUARDRAILS": RUN_06E_CONCENTRATION_GUARDRAILS,
    "RUN_06F_DECISION_RECORD": RUN_06F_DECISION_RECORD,
    "RUN_06G_BACKUP_TO_GOOGLE_DRIVE": RUN_06G_BACKUP_TO_GOOGLE_DRIVE,
}

print("Notebook 06 scope:", NOTEBOOK06_SCOPE)
print("Notebook 05 results dir:", NOTEBOOK05_RESULTS_DIR)
print("Notebook 06 output dir:", OUTPUT_DIR)
print("Run switches:", RUN_SWITCHES)
'''


RUNTIME_HELPERS_SOURCE = r'''
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def macro_f1_binary(y_true, y_pred):
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    scores = []
    for label in (0, 1):
        tp = int(((y == label) & (pred == label)).sum())
        fp = int(((y != label) & (pred == label)).sum())
        fn = int(((y == label) & (pred != label)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return float(np.mean(scores))


def balanced_accuracy_binary(y_true, y_pred):
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    recalls = []
    for label in (0, 1):
        support = int((y == label).sum())
        if support == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(((y == label) & (pred == label)).sum() / support))
    return float(np.mean(recalls))


def metric_row_06(y_true, y_pred):
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    return {
        "macro_f1": macro_f1_binary(y, pred),
        "balanced_accuracy": balanced_accuracy_binary(y, pred),
        "accuracy": float((y == pred).mean()) if len(y) else np.nan,
    }


def brier_score_binary(y_true, prob_up):
    y = np.asarray(y_true).astype(float)
    prob = np.asarray(prob_up).astype(float)
    return float(np.mean(np.square(prob - y))) if len(y) else np.nan


def log_loss_binary(y_true, prob_up):
    y = np.asarray(y_true).astype(float)
    prob = np.clip(np.asarray(prob_up).astype(float), FLOAT_TOLERANCE, 1.0 - FLOAT_TOLERANCE)
    if len(y) == 0:
        return np.nan
    return float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))


def require_artifact_contract():
    global artifact_contract
    if "artifact_contract" not in globals():
        artifact_contract = assert_notebook06_artifact_contract(NOTEBOOK05_RESULTS_DIR)
    return artifact_contract


def load_primary_prediction_frames():
    contract = require_artifact_contract()
    pooled = pd.read_csv(NOTEBOOK05_RESULTS_DIR / "notebook05_official_validation_pooled.csv")
    rows = pooled[
        (pooled["profile_id"].astype(str) == str(contract["primary_profile_id"]))
        & pooled["prediction_artifact"].astype(str).str.strip().astype(bool)
    ].copy()
    frames = []
    manifest_rows = []
    for _, row in rows.iterrows():
        artifact_path = _resolve_prediction_artifact(NOTEBOOK05_RESULTS_DIR, row["prediction_artifact"])
        payload = load_notebook06_prediction_artifact(artifact_path)
        frame = build_canonical_prediction_frame(payload, row.to_dict())
        frame.attrs["metadata"] = row.to_dict()
        frames.append(frame)
        manifest_rows.append(
            {
                "profile_id": str(row["profile_id"]),
                "profile_role": str(row["profile_role"]),
                "seed": int(row["seed"]),
                "prediction_artifact": str(artifact_path),
                "validation_n": int(row["validation_n"]),
                "sample_id_hash": _stable_hash(payload["validation_sample_id"]),
            }
        )
    if not frames:
        raise ValueError("No primary prediction frames were loaded.")
    manifest = pd.DataFrame(manifest_rows)
    return frames, manifest


def retained_class_and_prediction_counts(frame):
    y_true = frame["y_true"].astype(int)
    y_pred = frame["y_pred"].astype(int)
    return {
        "retained_class0_n": int((y_true == 0).sum()),
        "retained_class1_n": int((y_true == 1).sum()),
        "retained_positive_rate": float((y_true == 1).mean()) if len(y_true) else np.nan,
        "retained_pred0_n": int((y_pred == 0).sum()),
        "retained_pred1_n": int((y_pred == 1).sum()),
        "retained_pred1_rate": float((y_pred == 1).mean()) if len(y_pred) else np.nan,
    }


def per_ticker_delta_rows(retained, dummy_pred, key_fields):
    rows = []
    retained = retained.copy()
    retained["same_row_dummy_pred"] = np.asarray(dummy_pred).astype(int)
    positive_deltas = []
    for ticker, group in retained.groupby("ticker", sort=True):
        model_metrics = metric_row_06(group["y_true"], group["y_pred"])
        dummy_metrics = metric_row_06(group["y_true"], group["same_row_dummy_pred"])
        delta = model_metrics["macro_f1"] - dummy_metrics["macro_f1"]
        if delta > 0:
            positive_deltas.append(delta)
        rows.append(
            {
                **key_fields,
                "ticker": str(ticker),
                "retained_n": int(len(group)),
                "macro_f1": model_metrics["macro_f1"],
                "same_row_dummy_macro_f1": dummy_metrics["macro_f1"],
                "delta_macro_f1_vs_same_row_stratified_dummy": delta,
            }
        )
    positive_ticker_count = int(len(positive_deltas))
    top_ticker_gain_share = float(max(positive_deltas) / sum(positive_deltas)) if positive_deltas else 0.0
    for row in rows:
        row["positive_ticker_count"] = positive_ticker_count
        row["top_ticker_gain_share"] = top_ticker_gain_share
    return rows, positive_ticker_count, top_ticker_gain_share


def write_run_manifest(stage_name, extra=None):
    payload = {
        "stage": stage_name,
        "scope": NOTEBOOK06_SCOPE,
        "created_utc": utc_now_iso(),
        "notebook05_result_dir": str(NOTEBOOK05_RESULTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "run_switches": RUN_SWITCHES,
        "constants": {
            "COVERAGE_GRID": list(COVERAGE_GRID),
            "DECISION_COVERAGE_GRID": list(DECISION_COVERAGE_GRID),
            "MIN_DECISION_DELTA_MACRO_F1": MIN_DECISION_DELTA_MACRO_F1,
            "MIN_POSITIVE_SEED_COUNT": MIN_POSITIVE_SEED_COUNT,
            "MIN_POSITIVE_DECISION_COVERAGE_COUNT": MIN_POSITIVE_DECISION_COVERAGE_COUNT,
            "RANDOM_ABSTENTION_REPEATS": RANDOM_ABSTENTION_REPEATS,
            "RANDOM_ABSTENTION_BASE_SEED": RANDOM_ABSTENTION_BASE_SEED,
            "FLOAT_TOLERANCE": FLOAT_TOLERANCE,
        },
        "output_files": {key: str(value) for key, value in OUTPUT_FILES.items()},
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "operator_acknowledgements": {
            "validation_only_scope": bool(OPERATOR_ACKNOWLEDGES_VALIDATION_ONLY_SCOPE),
            "no_holdout_test": bool(OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST),
            "no_selective_threshold": bool(OPERATOR_ACKNOWLEDGES_NO_SELECTIVE_THRESHOLD),
        },
    }
    if extra:
        payload.update(extra)
    write_json(OUTPUT_FILES["run_manifest"], payload)
    return payload


def build_drive_service():
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Google Drive backup is only available in Colab with google APIs.") from exc
    auth.authenticate_user()
    return build("drive", "v3")


def upload_existing_outputs_to_drive():
    if not BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE:
        raise ValueError("BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE is False.")
    service = build_drive_service()
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError("googleapiclient.http.MediaFileUpload is unavailable.") from exc
    uploaded = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for key, path in OUTPUT_FILES.items():
        if key == "drive_backup_manifest" or not path.exists():
            continue
        metadata = {"name": f"{DRIVE_BACKUP_PREFIX}_{timestamp}_{path.name}"}
        if DRIVE_BACKUP_FOLDER_ID:
            metadata["parents"] = [DRIVE_BACKUP_FOLDER_ID]
        media = MediaFileUpload(str(path), resumable=False)
        uploaded.append(
            {
                "key": key,
                "local_path": str(path),
                "drive_file": service.files()
                .create(body=metadata, media_body=media, fields="id,name,webViewLink")
                .execute(),
            }
        )
    manifest = {
        "scope": NOTEBOOK06_SCOPE,
        "created_utc": utc_now_iso(),
        "backup_prefix": DRIVE_BACKUP_PREFIX,
        "uploaded": uploaded,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
    }
    write_json(OUTPUT_FILES["drive_backup_manifest"], manifest)
    return manifest
'''


CELL_06A = r'''
if RUN_06A_ARTIFACT_GATE:
    artifact_contract = assert_notebook06_artifact_contract(NOTEBOOK05_RESULTS_DIR)
    write_json(OUTPUT_FILES["artifact_contract"], artifact_contract)
    write_run_manifest(
        "06A_artifact_gate",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
        },
    )
    print("06A artifact contract passed:", artifact_contract)
else:
    print("RUN_06A_ARTIFACT_GATE is False; artifact gate not run.")
'''


CELL_06B = r'''
if RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS:
    artifact_contract = require_artifact_contract()
    prediction_frames, prediction_manifest = load_primary_prediction_frames()
    prediction_manifest.to_csv(OUTPUT_FILES["prediction_frame_manifest"], index=False)

    diagnostic_rows = []
    reliability_rows = []
    for frame in prediction_frames:
        metadata = frame.attrs["metadata"]
        key_fields = {
            "profile_id": str(metadata["profile_id"]),
            "profile_role": str(metadata["profile_role"]),
            "seed": int(metadata["seed"]),
        }
        quantile_bins = calibration_bins(
            frame["prob_up"].to_numpy(),
            frame["y_true"].to_numpy(),
            CALIBRATION_BIN_COUNT,
            CALIBRATION_PRIMARY_BINNING,
        )
        uniform_bins = calibration_bins(
            frame["prob_up"].to_numpy(),
            frame["y_true"].to_numpy(),
            CALIBRATION_BIN_COUNT,
            CALIBRATION_SENSITIVITY_BINNING,
        )
        for strategy, rows in (("quantile", quantile_bins), ("uniform", uniform_bins)):
            for row in rows:
                reliability_rows.append({**key_fields, **row})
        diagnostic_rows.append(
            {
                **key_fields,
                "n": int(len(frame)),
                "brier_score": brier_score_binary(frame["y_true"], frame["prob_up"]),
                "log_loss": log_loss_binary(frame["y_true"], frame["prob_up"]),
                "ece_quantile": ece_from_bins(quantile_bins),
                "ece_uniform": ece_from_bins(uniform_bins),
            }
        )

    pd.DataFrame(diagnostic_rows).to_csv(OUTPUT_FILES["probability_diagnostics"], index=False)
    pd.DataFrame(reliability_rows).to_csv(OUTPUT_FILES["reliability_bins"], index=False)
    write_run_manifest(
        "06B_prediction_frame_and_probability_diagnostics",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
        },
    )
    print("06B wrote:", OUTPUT_FILES["prediction_frame_manifest"], OUTPUT_FILES["probability_diagnostics"])
else:
    print("RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS is False; diagnostics not run.")
'''


CELL_06C = r'''
if RUN_06C_FIXED_COVERAGE_GRID:
    artifact_contract = require_artifact_contract()
    prediction_frames, prediction_manifest = load_primary_prediction_frames()
    coverage_rows = []
    same_row_rows = []
    random_rows = []
    per_ticker_rows = []

    for frame in prediction_frames:
        metadata = frame.attrs["metadata"]
        seed = int(metadata["seed"])
        key_base = {
            "profile_id": str(metadata["profile_id"]),
            "profile_role": str(metadata["profile_role"]),
            "seed": seed,
        }
        train_class0_n = int(metadata["train_class0_n"])
        train_class1_n = int(metadata["train_class1_n"])
        eligible_n = int(len(frame))
        for coverage_target in COVERAGE_GRID:
            retained_index = selective_retained_indices(
                frame["confidence"].to_numpy(),
                frame["validation_sample_id"].to_numpy(),
                coverage_target,
            )
            retained = frame.iloc[retained_index].copy()
            key_fields = {**key_base, "coverage_target": float(coverage_target)}
            model_metrics = metric_row_06(retained["y_true"], retained["y_pred"])
            dummy_pred = same_row_stratified_dummy_predict(
                train_class0_n,
                train_class1_n,
                len(retained),
                RANDOM_ABSTENTION_BASE_SEED + seed + int(round(float(coverage_target) * 1000)),
            )
            dummy_metrics = metric_row_06(retained["y_true"], dummy_pred)
            per_ticker_delta, positive_ticker_count, top_ticker_gain_share = per_ticker_delta_rows(
                retained,
                dummy_pred,
                key_fields,
            )
            per_ticker_rows.extend(per_ticker_delta)

            retained_count_by_ticker = retained["ticker"].astype(str).value_counts().to_dict()
            if float(coverage_target) == 1.0:
                random_macro_f1_values = [model_metrics["macro_f1"]]
                repeat_count = 1
            else:
                masks = ticker_stratified_random_abstention(
                    retained_count_by_ticker,
                    frame["ticker"].to_numpy(),
                    RANDOM_ABSTENTION_BASE_SEED + seed + int(round(float(coverage_target) * 1000)),
                    RANDOM_ABSTENTION_REPEATS,
                )
                random_macro_f1_values = [
                    metric_row_06(frame.loc[mask, "y_true"], frame.loc[mask, "y_pred"])["macro_f1"]
                    for mask in masks
                ]
                repeat_count = RANDOM_ABSTENTION_REPEATS
            random_mean = float(np.mean(random_macro_f1_values))
            random_std = float(np.std(random_macro_f1_values, ddof=1)) if len(random_macro_f1_values) > 1 else 0.0
            random_rows.append(
                {
                    **key_fields,
                    "random_abstention_repeat_count": repeat_count,
                    "random_abstention_macro_f1_mean": random_mean,
                    "random_abstention_macro_f1_std": random_std,
                }
            )
            same_row_rows.append(
                {
                    **key_fields,
                    "same_row_stratified_dummy_macro_f1": dummy_metrics["macro_f1"],
                    "same_row_stratified_dummy_balanced_accuracy": dummy_metrics["balanced_accuracy"],
                    "same_row_stratified_dummy_accuracy": dummy_metrics["accuracy"],
                    "train_class0_n": train_class0_n,
                    "train_class1_n": train_class1_n,
                    "train_positive_rate": float(metadata["train_positive_rate"]),
                }
            )
            coverage_rows.append(
                {
                    "profile_id": key_base["profile_id"],
                    "profile_role": key_base["profile_role"],
                    "seed": key_base["seed"],
                    "coverage_target": float(coverage_target),
                    "coverage_actual": float(len(retained) / eligible_n),
                    "eligible_n": eligible_n,
                    "retained_n": int(len(retained)),
                    "abstained_n": int(eligible_n - len(retained)),
                    **retained_class_and_prediction_counts(retained),
                    **model_metrics,
                    "same_row_stratified_dummy_macro_f1": dummy_metrics["macro_f1"],
                    "delta_macro_f1_vs_same_row_stratified_dummy": model_metrics["macro_f1"] - dummy_metrics["macro_f1"],
                    "random_abstention_macro_f1_mean": random_mean,
                    "random_abstention_macro_f1_std": random_std,
                    "delta_macro_f1_vs_random_abstention": model_metrics["macro_f1"] - random_mean,
                    "positive_ticker_count": positive_ticker_count,
                    "top_ticker_gain_share": top_ticker_gain_share,
                    "min_retained_confidence": float(retained["confidence"].min()) if len(retained) else np.nan,
                    "selective_threshold_selected": False,
                    "scope": NOTEBOOK06_SCOPE,
                }
            )

    pd.DataFrame(coverage_rows).to_csv(OUTPUT_FILES["coverage_grid"], index=False)
    pd.DataFrame(same_row_rows).to_csv(OUTPUT_FILES["same_row_baselines"], index=False)
    pd.DataFrame(random_rows).to_csv(OUTPUT_FILES["random_abstention_baselines"], index=False)
    pd.DataFrame(per_ticker_rows).to_csv(OUTPUT_FILES["per_ticker_coverage"], index=False)
    write_run_manifest(
        "06C_fixed_coverage_grid",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
        },
    )
    print("06C wrote:", OUTPUT_FILES["coverage_grid"], OUTPUT_FILES["random_abstention_baselines"])
else:
    print("RUN_06C_FIXED_COVERAGE_GRID is False; coverage grid not run.")
'''


CELL_06D = r'''
if RUN_06D_AGGREGATE_AND_RISK_COVERAGE:
    artifact_contract = require_artifact_contract()
    prediction_frames, prediction_manifest = load_primary_prediction_frames()
    risk_rows = []
    for frame in prediction_frames:
        metadata = frame.attrs["metadata"]
        curve = risk_coverage_curve(frame["y_true"], frame["y_pred"], frame["confidence"])
        oracle_curve = risk_coverage_curve(frame["y_true"], frame["y_pred"], frame["correct"])
        risk_rows.append(
            {
                "profile_id": str(metadata["profile_id"]),
                "profile_role": str(metadata["profile_role"]),
                "seed": int(metadata["seed"]),
                "aurc": aurc_from_curve(curve),
                "oracle_aurc": aurc_from_curve(oracle_curve),
                "e_aurc": aurc_from_curve(curve) - aurc_from_curve(oracle_curve),
                "scope": NOTEBOOK06_SCOPE,
            }
        )

    if OUTPUT_FILES["coverage_grid"].exists():
        coverage_grid = pd.read_csv(OUTPUT_FILES["coverage_grid"])
        coverage_summary = aggregate_across_seeds(
            coverage_grid,
            [
                "delta_macro_f1_vs_same_row_stratified_dummy",
                "delta_macro_f1_vs_random_abstention",
                "macro_f1",
                "balanced_accuracy",
            ],
        )
        coverage_summary_path = OUTPUT_DIR / "notebook06_coverage_summary.csv"
        coverage_summary.to_csv(coverage_summary_path, index=False)
    else:
        coverage_summary_path = None

    pd.DataFrame(risk_rows).to_csv(OUTPUT_FILES["risk_coverage_summary"], index=False)
    write_run_manifest(
        "06D_aggregate_and_risk_coverage",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
            "coverage_summary_path": str(coverage_summary_path) if coverage_summary_path else "",
        },
    )
    print("06D wrote:", OUTPUT_FILES["risk_coverage_summary"])
else:
    print("RUN_06D_AGGREGATE_AND_RISK_COVERAGE is False; aggregate/risk coverage not run.")
'''


CELL_06E = r'''
if RUN_06E_CONCENTRATION_GUARDRAILS:
    artifact_contract = require_artifact_contract()
    prediction_frames, prediction_manifest = load_primary_prediction_frames()
    concentration_rows = []

    for frame in prediction_frames:
        metadata = frame.attrs["metadata"]
        key_base = {
            "profile_id": str(metadata["profile_id"]),
            "profile_role": str(metadata["profile_role"]),
            "seed": int(metadata["seed"]),
        }
        for coverage_target in COVERAGE_GRID:
            retained_index = selective_retained_indices(
                frame["confidence"].to_numpy(),
                frame["validation_sample_id"].to_numpy(),
                coverage_target,
            )
            retained = frame.iloc[retained_index].copy()
            metrics = concentration_metrics(retained, frame)
            warning = bool(
                metrics.get("top_ticker_retained_share", 0.0) > 0.50
                or metrics.get("ticker_entropy_normalized", 1.0) < 0.70
            )
            severe = bool(
                metrics.get("top_ticker_retained_share", 0.0) > 0.65
                or metrics.get("ticker_entropy_normalized", 1.0) < 0.50
            )
            concentration_rows.append(
                {
                    **key_base,
                    "coverage_target": float(coverage_target),
                    **metrics,
                    "warning_guardrail_triggered": warning,
                    "severe_downgrade_triggered": severe,
                    "guardrail_pass": not severe,
                    "scope": NOTEBOOK06_SCOPE,
                }
            )

    pd.DataFrame(concentration_rows).to_csv(OUTPUT_FILES["concentration_guardrails"], index=False)
    write_run_manifest(
        "06E_concentration_guardrails",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
        },
    )
    print("06E wrote:", OUTPUT_FILES["concentration_guardrails"])
else:
    print("RUN_06E_CONCENTRATION_GUARDRAILS is False; concentration guardrails not run.")
'''


CELL_06F = r'''
if RUN_06F_DECISION_RECORD:
    artifact_contract = require_artifact_contract()
    if not OUTPUT_FILES["coverage_grid"].exists():
        raise FileNotFoundError(f"Missing required coverage grid: {OUTPUT_FILES['coverage_grid']}")
    coverage_grid = pd.read_csv(OUTPUT_FILES["coverage_grid"])
    coverage_summary = aggregate_across_seeds(
        coverage_grid,
        [
            "delta_macro_f1_vs_same_row_stratified_dummy",
            "delta_macro_f1_vs_random_abstention",
            "macro_f1",
            "balanced_accuracy",
        ],
    )
    guardrails = (
        pd.read_csv(OUTPUT_FILES["concentration_guardrails"])
        if OUTPUT_FILES["concentration_guardrails"].exists()
        else pd.DataFrame()
    )
    decision = evaluate_decision_outcome(
        coverage_summary,
        guardrails,
        {
            "decision_delta_mean_column": "delta_macro_f1_vs_random_abstention_mean",
            "MIN_DECISION_DELTA_MACRO_F1": MIN_DECISION_DELTA_MACRO_F1,
            "MIN_POSITIVE_SEED_COUNT": MIN_POSITIVE_SEED_COUNT,
            "MIN_POSITIVE_DECISION_COVERAGE_COUNT": MIN_POSITIVE_DECISION_COVERAGE_COUNT,
            "NOT_SUPPORTED_FAILURE_COVERAGE_COUNT": NOT_SUPPORTED_FAILURE_COVERAGE_COUNT,
        },
    )
    decision_record = {
        "scope": NOTEBOOK06_SCOPE,
        "created_utc": utc_now_iso(),
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "primary_profile_id": artifact_contract["primary_profile_id"],
        "primary_profile_source": artifact_contract["primary_profile_source"],
        "secondary_profiles_diagnostic_only": artifact_contract["secondary_profile_ids"],
        "reported_coverage_grid": list(COVERAGE_GRID),
        "decision_coverage_grid": list(DECISION_COVERAGE_GRID),
        "selective_threshold_selected": False,
        "holdout_test_authorized": False,
        "positive_decision_coverage_count": decision["positive_decision_coverage_count"],
        "failed_guardrail_count": decision["failed_guardrail_count"],
        "allowed_wording": [
            "validation-only selective no-trade diagnostic",
            "fixed coverage-grid high-confidence rows",
            "same-row dummy and ticker-stratified random-abstention comparisons",
            "holdout/test remains closed",
        ],
        "forbidden_wording": [
            "holdout-ready",
            "deployment-safe",
            "profitable strategy",
            "globally superior model family",
            "optimal abstention rate",
            "ECE proves calibration",
            "AURC proves tradability",
        ],
        "input_artifacts": {
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
            "sample_id_hash": artifact_contract["sample_id_hash"],
        },
    }
    write_json(OUTPUT_FILES["decision_record"], decision_record)
    write_run_manifest(
        "06F_decision_record",
        {
            "primary_profile_id": artifact_contract["primary_profile_id"],
            "secondary_profile_ids": artifact_contract["secondary_profile_ids"],
            "notebook05_entry_decision_sha256": artifact_contract["notebook05_entry_decision_sha256"],
            "notebook05_decision_record_sha256": artifact_contract["notebook05_decision_record_sha256"],
            "notebook05_run_manifest_sha256": artifact_contract["notebook05_run_manifest_sha256"],
            "decision": decision_record["decision"],
            "decision_record_path": str(OUTPUT_FILES["decision_record"]),
        },
    )
    print("06F wrote:", OUTPUT_FILES["decision_record"])
else:
    print("RUN_06F_DECISION_RECORD is False; decision record not written.")
'''


CELL_06G = r'''
if RUN_06G_BACKUP_TO_GOOGLE_DRIVE:
    if not BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE:
        raise ValueError("Set BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE=True before running 06G backup.")
    backup_manifest = upload_existing_outputs_to_drive()
    print("06G Drive backup manifest:", backup_manifest)
else:
    print("RUN_06G_BACKUP_TO_GOOGLE_DRIVE is False; Drive backup not run.")
'''


def validate_code_cells(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook06_cell_{index}")


def validate_notebook(nb: nbformat.NotebookNode) -> None:
    nbformat.validate(nb)
    validate_code_cells(nb)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    if any(cell.get("outputs") for cell in code_cells):
        raise AssertionError("Generated notebook must not contain saved outputs.")
    if [cell.get("execution_count") for cell in code_cells] != [None] * len(code_cells):
        raise AssertionError("Generated notebook code cell execution counts must be None.")
    source = "\n".join(cell.source for cell in code_cells)
    required = (
        "RUN_06A_ARTIFACT_GATE",
        "RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS",
        "RUN_06C_FIXED_COVERAGE_GRID",
        "RUN_06D_AGGREGATE_AND_RISK_COVERAGE",
        "RUN_06E_CONCENTRATION_GUARDRAILS",
        "RUN_06F_DECISION_RECORD",
        "RUN_06G_BACKUP_TO_GOOGLE_DRIVE",
        "assert_notebook06_artifact_contract",
        "same_row_stratified_dummy_predict(",
        "ticker_stratified_random_abstention(",
        "notebook06_coverage_grid.csv",
        "notebook06_decision_record.json",
        "notebook06_run_manifest.json",
    )
    for needle in required:
        if needle not in source:
            raise AssertionError(f"Missing required notebook source string: {needle}")
    forbidden = (
        "from intraday_research",
        "baseline_helpers",
        "train_test_split",
        "drive.mount",
        "select_threshold",
        "best_threshold",
        "optimal_threshold",
        "optimal_coverage",
        "confidence_threshold_implied",
    )
    for needle in forbidden:
        if needle in source:
            raise AssertionError(f"Forbidden notebook source string found: {needle}")


def build_notebook() -> nbformat.NotebookNode:
    if not CONTRACT_MODULE.exists():
        raise FileNotFoundError(f"Missing contract module: {CONTRACT_MODULE}")
    contract_source = CONTRACT_MODULE.read_text(encoding="utf-8")
    cells = [
        new_markdown_cell(
            "# Notebook 06 - Selective No-Trade Calibration\n\n"
            "Validation-only fixed coverage diagnostics for the Notebook 05 primary LightGBM profile. "
            "This notebook reads Notebook 05 artifacts, checks the artifact contract, and reports "
            "same-row dummy, random-abstention, calibration, risk-coverage, and concentration diagnostics. "
            "It does not train, does not read holdout/test, and does not select an operating threshold."
        ),
        new_code_cell(CONFIG_SOURCE.strip()),
        new_markdown_cell("## 06 Contract Helpers\n\nInline copy of `scripts/notebook06_contract.py` for Colab portability."),
        new_code_cell(contract_source.strip()),
        new_markdown_cell("## Runtime Helpers\n\nArtifact writing, metrics, Drive backup, and repeated frame loading helpers."),
        new_code_cell(RUNTIME_HELPERS_SOURCE.strip()),
        new_markdown_cell("## 06A - Artifact Gate\n\nStops before metrics if Notebook 05 artifacts violate the 05 -> 06 contract."),
        new_code_cell(CELL_06A.strip()),
        new_markdown_cell("## 06B - Prediction Frames And Probability Diagnostics"),
        new_code_cell(CELL_06B.strip()),
        new_markdown_cell("## 06C - Fixed Coverage Grid, Same-Row Dummy, Random Abstention"),
        new_code_cell(CELL_06C.strip()),
        new_markdown_cell("## 06D - Aggregate Seed Metrics And AURC/E-AURC"),
        new_code_cell(CELL_06D.strip()),
        new_markdown_cell("## 06E - Concentration Guardrails"),
        new_code_cell(CELL_06E.strip()),
        new_markdown_cell("## 06F - Decision Record"),
        new_code_cell(CELL_06F.strip()),
        new_markdown_cell("## 06G - Optional Google Drive Backup\n\nUses Drive file creation with timestamped names; it does not overwrite existing files."),
        new_code_cell(CELL_06G.strip()),
    ]
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    validate_notebook(nb)
    return nb


def main() -> None:
    nb = build_notebook()
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, TARGET_NOTEBOOK)
    print(f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
