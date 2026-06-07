"""Generate ``notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb``.

Builds Notebook 07 per
``docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md``.

The generated notebook is validation-only: it reads frozen N05 / optional N06
artifacts, signs a lockfile, builds a paper-ready comparison + ledger, runs
per-ticker / seed / concentration robustness, optional explainability and
null-control appendices, emits a gap audit for 08X, composes a paper-ready
synthesis + thesis paragraph kit, and writes a post-publication monitoring
plan. All RUN_07*/OPERATOR_ACKNOWLEDGES_* switches default to False; no run
mounts Drive, opens holdout/test, trains a new model, runs HPO, or selects
a new threshold / coverage / feature subset.

Run with project Python only:
``E:\\codex_workspace\\_envs\\py311_shared\\python.exe`` \\
``scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py``.
"""

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
    / "validation_synthesis_gap_audit.py"
)
# Phase 3 migration: canonical contract code now lives under
# src/intraday_research/contracts/. The legacy `scripts/notebook07_contract.py`
# path is a thin re-export shim; reading it here would inline only ~21 lines
# of imports and break the generated notebook's downstream cells. Always read
# CONTRACT_MODULE from the canonical src path.
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "07_validation_synthesis_and_gap_audit_colab.ipynb"


CONFIG_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK07_SCOPE = "validation_only"
NOTEBOOK05_RESULTS_DIR = Path("/content/notebook05_lightgbm_tuning_results")
NOTEBOOK06_RESULTS_DIR = Path("/content/notebook06_selective_no_trade_calibration_results")
OUTPUT_DIR = Path("/content/notebook07_validation_synthesis_and_gap_audit_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run switches - design "Run Switches And Defaults": every gate defaults False.
RUN_07A_LOCKFILE_SCOPE_GATE = False
RUN_07B_FINAL_VALIDATION_COMPARISON = False
RUN_07C_VALIDATION_BUDGET_LEDGER = False
RUN_07D_ROBUSTNESS_AND_CONCENTRATION = False
RUN_07E_EXPLAINABILITY_APPENDIX = False
RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX = False
RUN_07G_GAP_AUDIT_FOR_08X = False
RUN_07H_PAPER_READY_SYNTHESIS = False
RUN_07I_BACKUP_TO_GOOGLE_DRIVE = False
RUN_07J_WRITE_MONITORING_PLAN = False

BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE = False
DRIVE_BACKUP_FOLDER_ID = ""
DRIVE_BACKUP_PREFIX = "notebook07_validation_synthesis_and_gap_audit"

# Operator acknowledgements - design "Run Switches And Defaults".
OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH = False
OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False
OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS = False
OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH = False
# Optional SHAP gate per design 07E. Operator must flip BOTH this switch AND
# supply explainability_upgrade_record before any SHAP call.
OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL = False

# Design doc pin per 07A (see design "Run Switches And Defaults").
DESIGN_DOC_PATH = "docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md"
EXPECTED_DESIGN_DOC_SHA256 = ""  # 64-hex sha; set at freeze time, verified by 07A
DESIGN_DOC_SHA256_OBSERVED = ""

# Stage 0 locked candidate tuple (design 07A).
LOCKED_CANDIDATE_TUPLE = {
    "label_config": "h03_bps1p5",
    "feature_set": "price_volume_time",
    "window_size": 20,
}

# Forbidden-phrase regex per design 07H. Belt-and-suspenders over the explicit
# list of forbidden wording.
FORBIDDEN_PHRASE_REGEX = r"\b(final|production|deploy(?:ed|able|ment)?|tradable|live|sharpe|alpha)\b"

# Pre-registration constants table mirror - sourced from the canonical contract module.
PRE_REGISTRATION_CONSTANTS = {
    "improvement_threshold_delta_macro_f1_lcb_95": IMPROVEMENT_LCB_MIN,
    "improvement_threshold_positive_ticker_count_min": IMPROVEMENT_TICKER_COUNT_MIN,
    "weak_signal_band_upper": WEAK_SIGNAL_BAND_UPPER,
    "weak_signal_band_lower": WEAK_SIGNAL_BAND_LOWER,
    "concentration_warning_top_ticker_share_max": CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX,
    "concentration_warning_positive_ticker_count_min": CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN,
    "weak_seed_evidence_count_threshold": WEAK_SEED_EVIDENCE_COUNT_THRESHOLD,
    "null_control_alpha_total": NULL_CONTROL_ALPHA_TOTAL,
}

# Null-control alpha allocations (design 07F). Sum must not exceed
# NULL_CONTROL_ALPHA_TOTAL; allocations cannot be revised after a 07F read.
NULL_CONTROL_ALPHA_POLICY = {
    "alpha_total": NULL_CONTROL_ALPHA_TOTAL,
    "allocations": {
        "day_block_label_permutation": 0.02,
        "ticker_day_block_label_permutation": 0.015,
        "feature_family_permutation_within_block": 0.015,
    },
    "frozen_at_lockfile": True,
    "may_be_refunded_by_08F_only": True,
}

# 07F permutation-importance design defaults (design 07F).
NULL_CONTROL_DEFAULT_N_PERMUTATIONS = 100
PERMUTATION_IMPORTANCE_REPEATS = 5

# 07F operates in design option 1 ("read-only reporting of an existing
# pre-registered null-control artifact"). The cell will NOT synthesize a
# chronology-aware null in-line; instead it reads a pre-registered diagnostic
# artifact and emits derived rows + alpha accounting. The operator must set
# this path before enabling 07F.
PRE_REGISTERED_NULL_CONTROL_PATH = ""
PRE_REGISTERED_NULL_CONTROL_REQUIRED_COLUMNS = (
    "null_design",
    "permutation_unit",
    "n_permutations",
    "score",
    "observed_score",
    "null_score_mean",
    "null_score_p95",
    "p_value_one_sided",
    "scope",
)

# Hash input normalization recipe (design 07A) - recorded verbatim in lockfile.
HASH_INPUT_NORMALIZATION = {
    "encoding": "utf-8",
    "line_endings": "LF",
    "csv_column_order": "lexicographic",
    "float_rounding_decimals": 6,
    "json_sort_keys": True,
    "json_separators": [",", ":"],
}

OUTPUT_FILES = {
    "lockfile_scope_gate": OUTPUT_DIR / "notebook07_lockfile_scope_gate.json",
    "input_artifact_manifest": OUTPUT_DIR / "notebook07_input_artifact_manifest.csv",
    "final_validation_comparison": OUTPUT_DIR / "notebook07_final_validation_comparison.csv",
    "validation_budget_ledger": OUTPUT_DIR / "notebook07_validation_budget_ledger.csv",
    "per_ticker_robustness": OUTPUT_DIR / "notebook07_per_ticker_robustness.csv",
    "seed_robustness": OUTPUT_DIR / "notebook07_seed_robustness.csv",
    "concentration_summary": OUTPUT_DIR / "notebook07_concentration_summary.csv",
    "gap_audit_for_08x": OUTPUT_DIR / "notebook07_gap_audit_for_08x.csv",
    "paper_ready_synthesis": OUTPUT_DIR / "notebook07_paper_ready_synthesis.md",
    "thesis_paragraph_kit": OUTPUT_DIR / "notebook07_thesis_paragraph_kit.json",
    "decision_and_wording_record": OUTPUT_DIR / "notebook07_decision_and_wording_record.json",
    "run_manifest": OUTPUT_DIR / "notebook07_run_manifest.json",
    "lightgbm_importance_gain": OUTPUT_DIR / "notebook07_lightgbm_importance_gain.csv",
    "lightgbm_importance_split": OUTPUT_DIR / "notebook07_lightgbm_importance_split.csv",
    "lightgbm_pred_contrib_summary": OUTPUT_DIR / "notebook07_lightgbm_pred_contrib_summary.csv",
    "permutation_importance_diagnostic": OUTPUT_DIR / "notebook07_permutation_importance_diagnostic.csv",
    "null_control_diagnostic": OUTPUT_DIR / "notebook07_null_control_diagnostic.csv",
    "explainability_manifest": OUTPUT_DIR / "notebook07_explainability_manifest.json",
    "post_publication_monitoring_plan": OUTPUT_DIR / "notebook07_post_publication_monitoring_plan.json",
    "drive_backup_manifest": OUTPUT_DIR / "notebook07_drive_backup_manifest.json",
}

RUN_SWITCHES = {
    "RUN_07A_LOCKFILE_SCOPE_GATE": RUN_07A_LOCKFILE_SCOPE_GATE,
    "RUN_07B_FINAL_VALIDATION_COMPARISON": RUN_07B_FINAL_VALIDATION_COMPARISON,
    "RUN_07C_VALIDATION_BUDGET_LEDGER": RUN_07C_VALIDATION_BUDGET_LEDGER,
    "RUN_07D_ROBUSTNESS_AND_CONCENTRATION": RUN_07D_ROBUSTNESS_AND_CONCENTRATION,
    "RUN_07E_EXPLAINABILITY_APPENDIX": RUN_07E_EXPLAINABILITY_APPENDIX,
    "RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX": RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX,
    "RUN_07G_GAP_AUDIT_FOR_08X": RUN_07G_GAP_AUDIT_FOR_08X,
    "RUN_07H_PAPER_READY_SYNTHESIS": RUN_07H_PAPER_READY_SYNTHESIS,
    "RUN_07I_BACKUP_TO_GOOGLE_DRIVE": RUN_07I_BACKUP_TO_GOOGLE_DRIVE,
    "RUN_07J_WRITE_MONITORING_PLAN": RUN_07J_WRITE_MONITORING_PLAN,
}

print("Notebook 07 scope:", NOTEBOOK07_SCOPE)
print("Notebook 05 results dir:", NOTEBOOK05_RESULTS_DIR)
print("Notebook 06 results dir:", NOTEBOOK06_RESULTS_DIR)
print("Notebook 07 output dir:", OUTPUT_DIR)
print("Run switches:", RUN_SWITCHES)
'''


RUNTIME_HELPERS_SOURCE = r'''
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_csv_bytes(df):
    columns = sorted(df.columns.tolist())
    canon = df.loc[:, columns].copy()
    for col in columns:
        if pd.api.types.is_float_dtype(canon[col]):
            canon[col] = canon[col].round(int(HASH_INPUT_NORMALIZATION["float_rounding_decimals"]))
    return canon.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_canonical_csv(df):
    return sha256_bytes(canonical_csv_bytes(df))


def canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=bool(HASH_INPUT_NORMALIZATION["json_sort_keys"]),
        separators=tuple(HASH_INPUT_NORMALIZATION["json_separators"]),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_canonical_json(payload):
    return sha256_bytes(canonical_json_bytes(payload))


def _is_false(value):
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def _is_true(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _require_false_or_raise(record, field, source):
    if field not in record:
        raise ValueError(f"{source} is missing required field: {field}")
    if not _is_false(record[field]):
        raise ValueError(f"{field} is not false in {source}")


def _require_scope_validation_only(record, source):
    if "scope" not in record:
        raise ValueError(f"{source} is missing required field: scope")
    if str(record["scope"]) != NOTEBOOK07_SCOPE:
        raise ValueError(f"scope is not {NOTEBOOK07_SCOPE} in {source}")


def _check_no_holdout_or_test_path(raw_path):
    text = str(raw_path).replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    if any(("holdout" in part or "test" in part) for part in parts):
        raise ValueError(f"Prediction path may not contain holdout/test: {raw_path}")


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required N07 input artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_csv_required(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required N07 input CSV: {path}")
    return pd.read_csv(path)


def require_n05_artifacts(n05_dir):
    n05_dir = Path(n05_dir)
    if not n05_dir.exists():
        raise FileNotFoundError(f"Missing N05 results directory: {n05_dir}")
    return {
        "entry_decision": n05_dir / "notebook05_entry_decision.json",
        "decision_record": n05_dir / "notebook05_decision_record.json",
        "run_manifest": n05_dir / "notebook05_run_manifest.json",
        "official_summary": n05_dir / "notebook05_official_validation_summary.csv",
        "official_pooled": n05_dir / "notebook05_official_validation_pooled.csv",
        "official_per_ticker": n05_dir / "notebook05_official_validation_per_ticker.csv",
    }


def require_n06_artifacts_if_present(n06_dir):
    n06_dir = Path(n06_dir)
    if not n06_dir.exists():
        return None
    contract_path = n06_dir / "notebook06_artifact_contract_check.json"
    if not contract_path.exists():
        return None
    contract = read_json(contract_path)
    if not _is_true(contract.get("contract_passed", False)):
        return None
    return {
        "artifact_contract": contract_path,
        "decision_record": n06_dir / "notebook06_decision_record.json",
        "run_manifest": n06_dir / "notebook06_run_manifest.json",
        "coverage_grid": n06_dir / "notebook06_coverage_grid.csv",
        "same_row_baselines": n06_dir / "notebook06_same_row_baselines.csv",
        "random_abstention_baselines": n06_dir / "notebook06_random_abstention_baselines.csv",
        "per_ticker_coverage": n06_dir / "notebook06_per_ticker_coverage.csv",
        "concentration_guardrails": n06_dir / "notebook06_concentration_guardrails.csv",
        "risk_coverage_summary": n06_dir / "notebook06_risk_coverage_summary.csv",
    }


_VALIDATION_BUDGET_LEDGER_ROWS = []


def _hydrate_ledger_from_disk_if_needed():
    """If the in-memory ledger is empty and the on-disk ledger exists,
    rehydrate memory so subsequent appends extend (not replace) the file.

    Protects against kernel restarts: after a restart the in-memory list is
    empty, but the on-disk ledger still records every prior phase's intent
    rows. Without rehydration, the next append would write a shorter frame
    and prefix invariance would refuse the write.
    """
    if _VALIDATION_BUDGET_LEDGER_ROWS:
        return
    target_path = OUTPUT_FILES["validation_budget_ledger"]
    if not target_path.exists():
        return
    on_disk = pd.read_csv(target_path, dtype=str, keep_default_na=False)
    for _, row in on_disk.iterrows():
        _VALIDATION_BUDGET_LEDGER_ROWS.append(row.to_dict())


def flush_ledger_to_disk():
    """Write the in-memory ledger to OUTPUT_FILES['validation_budget_ledger']
    after a prefix invariance check vs the on-disk version.

    Called automatically by ``append_ledger_row`` so every append is durable
    and audited against the prior on-disk state. Refuses to overwrite an
    on-disk ledger whose prefix would not match the new in-memory ledger.
    """
    target_path = OUTPUT_FILES["validation_budget_ledger"]
    new_df = pd.DataFrame(_VALIDATION_BUDGET_LEDGER_ROWS)
    if target_path.exists():
        existing_df = pd.read_csv(target_path, dtype=str, keep_default_na=False)
        validate_ledger_prefix_invariance(existing_df, new_df)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(target_path, index=False, lineterminator="\n")
    return target_path


def append_ledger_row(
    artifact,
    notebook_stage,
    decision_made,
    decision_timing,
    decision_surface,
    *,
    model_families_considered="lightgbm",
    profiles_or_trials_considered="lightgbm_winner",
    seeds_used="",
    thresholds_or_coverages_considered="n/a",
    official_validation_rows_inspected=0,
    train_inner_only_decision=False,
    official_validation_informed_decision=False,
    diagnostic_only_readout=False,
    holdout_test_contact=False,
    allowed_wording="",
    forbidden_wording="",
    risk_note="",
):
    """Append a ledger row + flush to disk under prefix invariance.

    The flush happens before any caller proceeds, so a phase that crashes
    mid-read still leaves a durable record of its intent to read official
    validation. Callers must invoke this BEFORE reading any official
    validation artifact, not after.
    """
    _hydrate_ledger_from_disk_if_needed()
    last_cumulative = (
        int(_VALIDATION_BUDGET_LEDGER_ROWS[-1]["cumulative_official_validation_inspections_across_notebooks"])
        if _VALIDATION_BUDGET_LEDGER_ROWS
        else 0
    )
    cumulative = last_cumulative + int(official_validation_rows_inspected)
    row = {
        "artifact": str(artifact),
        "notebook_stage": str(notebook_stage),
        "decision_made": str(decision_made),
        "decision_timing": str(decision_timing),
        "decision_surface": str(decision_surface),
        "model_families_considered": str(model_families_considered),
        "profiles_or_trials_considered": str(profiles_or_trials_considered),
        "seeds_used": str(seeds_used),
        "thresholds_or_coverages_considered": str(thresholds_or_coverages_considered),
        "official_validation_rows_inspected": int(official_validation_rows_inspected),
        "cumulative_official_validation_inspections_across_notebooks": int(cumulative),
        "train_inner_only_decision": bool(train_inner_only_decision),
        "official_validation_informed_decision": bool(official_validation_informed_decision),
        "diagnostic_only_readout": bool(diagnostic_only_readout),
        "holdout_test_contact": bool(holdout_test_contact),
        "allowed_wording": str(allowed_wording),
        "forbidden_wording": str(forbidden_wording),
        "risk_note": str(risk_note),
        "appended_by_notebook": "07",
        "appended_at_utc": utc_now_iso(),
    }
    _VALIDATION_BUDGET_LEDGER_ROWS.append(row)
    flush_ledger_to_disk()
    return row


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


def same_row_stratified_dummy_predict(train_class0_n, train_class1_n, n_validation, seed):
    total = int(train_class0_n) + int(train_class1_n)
    if total <= 0:
        raise ValueError("train class counts must have positive total")
    positive_rate = int(train_class1_n) / total
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.asarray([0, 1], dtype=int), size=int(n_validation), p=[1.0 - positive_rate, positive_rate])


def one_sided_lcb_95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if n == 1:
        return mean
    std = float(np.std(arr, ddof=1))
    t_table = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}
    t_critical = t_table.get(n, 1.833)
    return mean - t_critical * std / math.sqrt(n)


def forbidden_phrase_blocks(text):
    matches = re.findall(FORBIDDEN_PHRASE_REGEX, str(text), flags=re.IGNORECASE)
    return sorted({m.lower() for m in matches})


def band_from_delta(delta_lcb_95, positive_ticker_count):
    delta = float(delta_lcb_95)
    pos = int(positive_ticker_count)
    if delta <= 0.0:
        return ("no_signal", "no detected validation-only signal")
    if delta < WEAK_SIGNAL_BAND_UPPER:
        return ("weak", "weak / mixed validation-only signal")
    if pos < IMPROVEMENT_TICKER_COUNT_MIN:
        return ("concentration_limited", "weak / concentration-limited validation-only signal")
    return ("practical", "practical validation-only signal, not holdout/test evidence")


def revalidate_artifact_hashes(lockfile):
    """Re-compute artifact hashes and compare to the lockfile's signed values.

    Design §07A: every subsequent RUN_07* phase MUST revalidate this gate
    at entry; an artifact-hash change between phases is a hard stop until a
    new freeze is signed. CSV artifacts use the canonical-CSV sha (utf-8,
    LF, lex column order, 6-decimal float); other artifacts use file sha.

    Raises ``ValueError`` listing every drifted/missing artifact.
    """
    artifact_paths = dict(lockfile.get("artifact_paths", {}))
    artifact_sha256 = dict(lockfile.get("artifact_sha256", {}))
    if not artifact_paths or not artifact_sha256:
        raise ValueError(
            "07A lockfile missing artifact_paths or artifact_sha256; cannot revalidate."
        )
    mismatches = []
    for key, path_str in artifact_paths.items():
        expected = artifact_sha256.get(key, "")
        if not expected:
            mismatches.append(f"{key}: lockfile has no expected sha256")
            continue
        path = Path(path_str)
        if not path.exists():
            mismatches.append(f"{key}: artifact missing at {path}")
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                observed = sha256_canonical_csv(pd.read_csv(path))
            else:
                observed = sha256_file(path)
        except Exception as exc:
            mismatches.append(f"{key}: hash recompute failed ({type(exc).__name__}: {exc})")
            continue
        if observed != expected:
            mismatches.append(
                f"{key}: sha256 drifted at {path} (expected {expected[:12]}..., observed {observed[:12]}...)"
            )
    if mismatches:
        raise ValueError(
            "07A lockfile artifact hashes drifted between phases; refreeze before any RUN_07* may continue:\n  "
            + "\n  ".join(mismatches)
        )


def require_artifact_lockfile():
    if not OUTPUT_FILES["lockfile_scope_gate"].exists():
        raise FileNotFoundError(
            f"Missing 07A lockfile: {OUTPUT_FILES['lockfile_scope_gate']}. "
            "Enable the 07A lockfile-scope-gate switch and re-run."
        )
    lockfile = read_json(OUTPUT_FILES["lockfile_scope_gate"])
    if not _is_true(lockfile.get("contract_passed", False)):
        raise ValueError("07A lockfile contract_passed != True; re-run 07A before downstream phases.")
    # Design §07A "revalidate this gate at the entry of every subsequent
    # RUN_07* phase; treat any artifact-hash change between phases as a hard
    # stop until a new freeze is signed."
    revalidate_artifact_hashes(lockfile)
    return lockfile


def write_run_manifest(stage_name, lockfile=None, extra=None):
    payload = {
        "stage": stage_name,
        "scope": NOTEBOOK07_SCOPE,
        "created_utc": utc_now_iso(),
        "notebook05_result_dir": str(NOTEBOOK05_RESULTS_DIR),
        "notebook06_result_dir": str(NOTEBOOK06_RESULTS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "run_switches": RUN_SWITCHES,
        "operator_acknowledgements": {
            "07_is_not_search": bool(OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH),
            "no_holdout_test": bool(OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST),
            "no_selection_from_explanations": bool(OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS),
            "no_threshold_search": bool(OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH),
            "shap_approval": bool(OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL),
        },
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "official_candidate": LOCKED_CANDIDATE_TUPLE,
        "design_doc_path": DESIGN_DOC_PATH,
        "design_doc_sha256_observed": DESIGN_DOC_SHA256_OBSERVED,
        "design_doc_sha256_expected": EXPECTED_DESIGN_DOC_SHA256,
        "output_files": {key: str(value) for key, value in OUTPUT_FILES.items()},
        "gateway_to_08x": str(OUTPUT_FILES["gap_audit_for_08x"]),
        "diagnostic_only_sections": ["07D", "07E", "07F"],
    }
    if lockfile is not None:
        payload["input_artifacts"] = lockfile.get("artifact_paths", {})
        payload["input_artifact_hashes"] = lockfile.get("artifact_sha256", {})
        payload["notebook05_decision_record_sha256"] = lockfile.get("artifact_sha256", {}).get("notebook05_decision_record", "")
        payload["notebook05_run_manifest_sha256"] = lockfile.get("artifact_sha256", {}).get("notebook05_run_manifest", "")
        payload["notebook06_decision_record_sha256"] = lockfile.get("artifact_sha256", {}).get("notebook06_decision_record", "")
        payload["notebook06_run_manifest_sha256"] = lockfile.get("artifact_sha256", {}).get("notebook06_run_manifest", "")
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
    if not BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE:
        raise ValueError("BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE is False.")
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
        "scope": NOTEBOOK07_SCOPE,
        "created_utc": utc_now_iso(),
        "backup_prefix": DRIVE_BACKUP_PREFIX,
        "uploaded": uploaded,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
    }
    write_json(OUTPUT_FILES["drive_backup_manifest"], manifest)
    return manifest
'''


CELL_07A = r'''
if RUN_07A_LOCKFILE_SCOPE_GATE:
    # Before-read ledger entry: declare intent to inspect N05 official validation.
    # Flushes to disk so the read is durably accounted for even if 07A crashes.
    append_ledger_row(
        artifact=str(NOTEBOOK05_RESULTS_DIR),
        notebook_stage="07A",
        decision_made="reading_n05_artifacts_for_lockfile_signing",
        decision_timing="before_official_validation_read",
        decision_surface="lockfile",
        seeds_used="locked",
        thresholds_or_coverages_considered="n/a",
        official_validation_rows_inspected=1,
        train_inner_only_decision=True,
        allowed_wording="07A intent: read N05 (and optional N06) artifacts to sign lockfile",
        forbidden_wording="holdout_ready; deployment_safe",
    )

    n05_paths = require_n05_artifacts(NOTEBOOK05_RESULTS_DIR)
    for path in n05_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required N05 artifact: {path}")
    n05_decision = read_json(n05_paths["decision_record"])
    n05_run_manifest = read_json(n05_paths["run_manifest"])
    n05_entry = read_json(n05_paths["entry_decision"])
    _require_scope_validation_only(n05_decision, n05_paths["decision_record"])
    _require_scope_validation_only(n05_run_manifest, n05_paths["run_manifest"])
    _require_false_or_raise(n05_decision, "holdout_test_authorized", n05_paths["decision_record"])
    _require_false_or_raise(n05_run_manifest, "holdout_test_authorized", n05_paths["run_manifest"])
    _require_false_or_raise(n05_decision, "selective_threshold_selected", n05_paths["decision_record"])
    _require_false_or_raise(n05_run_manifest, "selective_threshold_selected", n05_paths["run_manifest"])
    profile_source = str(n05_decision.get("selected_profile_source", "")).strip().lower()
    if "official_validation_best" in profile_source:
        raise ValueError(
            "N05 selected_profile_source indicates official-validation-best; "
            "07A refuses to sign the lockfile."
        )

    n05_pooled = read_csv_required(n05_paths["official_pooled"])
    if "prediction_artifact" in n05_pooled.columns:
        for raw in n05_pooled["prediction_artifact"].dropna().astype(str):
            if raw.strip():
                _check_no_holdout_or_test_path(raw)

    n06_paths = require_n06_artifacts_if_present(NOTEBOOK06_RESULTS_DIR)
    n06_decision_hash = ""
    n06_run_manifest_hash = ""
    n06_contract_hash = ""
    if n06_paths is not None:
        n06_decision = read_json(n06_paths["decision_record"])
        n06_run_manifest = read_json(n06_paths["run_manifest"])
        _require_false_or_raise(n06_decision, "holdout_test_authorized", n06_paths["decision_record"])
        _require_false_or_raise(n06_run_manifest, "holdout_test_authorized", n06_paths["run_manifest"])
        _require_false_or_raise(n06_decision, "selective_threshold_selected", n06_paths["decision_record"])
        _require_false_or_raise(n06_run_manifest, "selective_threshold_selected", n06_paths["run_manifest"])
        n06_decision_hash = sha256_file(n06_paths["decision_record"])
        n06_run_manifest_hash = sha256_file(n06_paths["run_manifest"])
        n06_contract_hash = sha256_file(n06_paths["artifact_contract"])

    design_doc_resolved = Path(DESIGN_DOC_PATH)
    if not design_doc_resolved.is_absolute():
        design_doc_resolved = Path.cwd() / design_doc_resolved
    if not design_doc_resolved.exists():
        raise FileNotFoundError(f"Missing design doc: {design_doc_resolved}")
    with design_doc_resolved.open("rb") as _fh:
        DESIGN_DOC_SHA256_OBSERVED = hashlib.sha256(_fh.read()).hexdigest()
    if not EXPECTED_DESIGN_DOC_SHA256:
        raise ValueError(
            "07A: EXPECTED_DESIGN_DOC_SHA256 is empty. Pin the freeze-time sha "
            "(64 hex chars) before signing the lockfile. Observed sha was "
            f"'{DESIGN_DOC_SHA256_OBSERVED}'."
        )
    if DESIGN_DOC_SHA256_OBSERVED != EXPECTED_DESIGN_DOC_SHA256:
        raise ValueError(
            "07A: design doc bytes drifted; refreeze required before any RUN_07* may be enabled "
            f"(expected '{EXPECTED_DESIGN_DOC_SHA256}', observed '{DESIGN_DOC_SHA256_OBSERVED}')"
        )

    artifact_paths = {
        "notebook05_decision_record": str(n05_paths["decision_record"]),
        "notebook05_run_manifest": str(n05_paths["run_manifest"]),
        "notebook05_entry_decision": str(n05_paths["entry_decision"]),
        "notebook05_official_pooled": str(n05_paths["official_pooled"]),
        "notebook05_official_per_ticker": str(n05_paths["official_per_ticker"]),
        "notebook05_official_summary": str(n05_paths["official_summary"]),
        "design_doc": str(design_doc_resolved),
    }
    artifact_sha256 = {
        "notebook05_decision_record": sha256_file(n05_paths["decision_record"]),
        "notebook05_run_manifest": sha256_file(n05_paths["run_manifest"]),
        "notebook05_entry_decision": sha256_file(n05_paths["entry_decision"]),
        "notebook05_official_pooled": sha256_canonical_csv(n05_pooled),
        "notebook05_official_per_ticker": sha256_canonical_csv(read_csv_required(n05_paths["official_per_ticker"])),
        "notebook05_official_summary": sha256_canonical_csv(read_csv_required(n05_paths["official_summary"])),
        "design_doc": DESIGN_DOC_SHA256_OBSERVED,
    }
    if n06_paths is not None:
        artifact_paths["notebook06_decision_record"] = str(n06_paths["decision_record"])
        artifact_paths["notebook06_run_manifest"] = str(n06_paths["run_manifest"])
        artifact_paths["notebook06_artifact_contract"] = str(n06_paths["artifact_contract"])
        artifact_sha256["notebook06_decision_record"] = n06_decision_hash
        artifact_sha256["notebook06_run_manifest"] = n06_run_manifest_hash
        artifact_sha256["notebook06_artifact_contract"] = n06_contract_hash

    sample_id_hash = ""
    if "validation_sample_id_hash" in n05_pooled.columns:
        unique_hashes = sorted({str(v) for v in n05_pooled["validation_sample_id_hash"].dropna() if str(v).strip()})
        if len(unique_hashes) > 1:
            raise ValueError("N05 pooled rows disagree on validation_sample_id_hash")
        sample_id_hash = unique_hashes[0] if unique_hashes else ""

    alpha_total_check = float(sum(NULL_CONTROL_ALPHA_POLICY["allocations"].values()))
    if alpha_total_check > NULL_CONTROL_ALPHA_TOTAL + 1e-9:
        raise ValueError(
            f"Null-control alpha allocations sum {alpha_total_check} exceeds "
            f"NULL_CONTROL_ALPHA_TOTAL={NULL_CONTROL_ALPHA_TOTAL}"
        )

    lockfile = {
        "scope": NOTEBOOK07_SCOPE,
        "created_utc": utc_now_iso(),
        "contract_passed": True,
        "candidate_tuple": LOCKED_CANDIDATE_TUPLE,
        "selected_profile_id": str(n05_decision.get("selected_profile_id", "")),
        "selected_profile_source": str(n05_decision.get("selected_profile_source", "")),
        "sample_id_hash": sample_id_hash,
        "artifact_paths": artifact_paths,
        "artifact_sha256": artifact_sha256,
        "accepted_run_switches": RUN_SWITCHES,
        "operator_acknowledgements": {
            "07_is_not_search": bool(OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH),
            "no_holdout_test": bool(OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST),
            "no_selection_from_explanations": bool(OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS),
            "no_threshold_search": bool(OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH),
            "shap_approval": bool(OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL),
        },
        "hash_input_normalization": HASH_INPUT_NORMALIZATION,
        "null_control_alpha_policy": NULL_CONTROL_ALPHA_POLICY,
        "pre_registration_constants": PRE_REGISTRATION_CONSTANTS,
        "design_doc_sha256_observed": DESIGN_DOC_SHA256_OBSERVED,
        "design_doc_sha256_expected": EXPECTED_DESIGN_DOC_SHA256,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
    }
    write_json(OUTPUT_FILES["lockfile_scope_gate"], lockfile)

    manifest_rows = []
    for key in artifact_paths:
        manifest_rows.append(
            {
                "artifact_key": key,
                "path": artifact_paths[key],
                "sha256": artifact_sha256.get(key, ""),
                "scope": NOTEBOOK07_SCOPE,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_FILES["input_artifact_manifest"], index=False)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["lockfile_scope_gate"]),
        notebook_stage="07A",
        decision_made="lockfile_signed",
        decision_timing="after_official_validation_read",
        decision_surface="lockfile",
        seeds_used="locked",
        official_validation_rows_inspected=0,
        train_inner_only_decision=True,
        allowed_wording="lockfile signed; scope = validation_only",
        forbidden_wording="holdout_ready; deployment_safe",
    )
    write_run_manifest("07A_lockfile_scope_gate", lockfile=lockfile)
    print("07A signed lockfile:", OUTPUT_FILES["lockfile_scope_gate"])
else:
    print("RUN_07A_LOCKFILE_SCOPE_GATE is False; 07A not run.")
'''


CELL_07B = r'''
if RUN_07B_FINAL_VALIDATION_COMPARISON:
    lockfile = require_artifact_lockfile()
    # Before-read ledger entry: 07B reads N05 official_validation_pooled (and
    # optionally N06 coverage_grid) to build the paper-ready comparison.
    append_ledger_row(
        artifact=str(OUTPUT_FILES["final_validation_comparison"]),
        notebook_stage="07B",
        decision_made="reading_official_validation_for_final_comparison",
        decision_timing="before_official_validation_read",
        decision_surface="comparison_table",
        seeds_used="locked",
        thresholds_or_coverages_considered="n06_fixed_grid_if_present",
        official_validation_rows_inspected=1,
        allowed_wording="07B intent: build locked-row final comparison",
        forbidden_wording="threshold or coverage selection wording forbidden in 07",
    )

    n05_paths = require_n05_artifacts(NOTEBOOK05_RESULTS_DIR)
    n05_pooled = read_csv_required(n05_paths["official_pooled"])

    primary_profile_id = lockfile.get("selected_profile_id", "")
    if not primary_profile_id:
        raise ValueError("Lockfile missing selected_profile_id; 07A must run first.")

    full_rows = n05_pooled[n05_pooled["profile_id"].astype(str) == str(primary_profile_id)].copy()
    if full_rows.empty:
        raise ValueError(f"N05 pooled has no rows for primary profile {primary_profile_id}")

    _label_config_str = str(LOCKED_CANDIDATE_TUPLE["label_config"])
    _parsed_label = parse_label_config(_label_config_str)

    def _coerce_int_or_fallback(group_col, fallback):
        if group_col.empty:
            return int(fallback)
        coerced = pd.to_numeric(group_col, errors="coerce").dropna()
        return int(coerced.iloc[0]) if not coerced.empty else int(fallback)

    def _coerce_float_or_fallback(group_col, fallback):
        if group_col.empty:
            return float(fallback)
        coerced = pd.to_numeric(group_col, errors="coerce").dropna()
        return float(coerced.iloc[0]) if not coerced.empty else float(fallback)

    # Same-row dummy hard-stop: N05 official pooled MUST carry per-seed
    # stratified dummy and delta columns (design §"Required Notebook 05 row
    # fields" + AGENTS.md §4.2). Silent NaN would let a row reach thesis
    # wording without a baseline.
    _required_n05_dummy_cols = ("stratified_dummy_macro_f1", "delta_macro_f1_vs_stratified_dummy")
    for _col in _required_n05_dummy_cols:
        if _col not in n05_pooled.columns:
            raise ValueError(
                f"N05 official pooled is missing required same-row dummy column "
                f"'{_col}'. AGENTS.md §4.2 requires every model row to carry a "
                "same-row stratified dummy and its delta; 07B cannot synthesize."
            )

    full_coverage_records = []
    grouped = full_rows.groupby(["profile_id", "profile_role"], dropna=False, sort=False)
    for (profile_id, profile_role), group in grouped:
        seed_count = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        macro_values = pd.to_numeric(group["macro_f1"], errors="coerce").dropna().to_numpy(dtype=float)
        bal_values = pd.to_numeric(group["balanced_accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
        acc_values = pd.to_numeric(group["accuracy"], errors="coerce").dropna().to_numpy(dtype=float)
        dummy_values = pd.to_numeric(group["stratified_dummy_macro_f1"], errors="coerce").dropna().to_numpy(dtype=float)
        delta_values = pd.to_numeric(group["delta_macro_f1_vs_stratified_dummy"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(dummy_values) == 0 or len(delta_values) == 0:
            raise ValueError(
                f"N05 official pooled group profile_id={profile_id!r} has all-NaN "
                "stratified_dummy_macro_f1 or delta_macro_f1_vs_stratified_dummy; "
                "07B cannot emit a same-row-dummy-free row."
            )
        delta_bal_values = pd.to_numeric(group.get("delta_balanced_accuracy_vs_stratified_dummy", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        dummy_bal_values = pd.to_numeric(group.get("stratified_dummy_balanced_accuracy", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        au_values = pd.to_numeric(group.get("always_up_dummy_macro_f1", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        au_delta_values = pd.to_numeric(group.get("delta_macro_f1_vs_always_up_dummy", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        positive_ticker_count = int(group["positive_ticker_count"].dropna().astype(int).max()) if "positive_ticker_count" in group.columns else 0
        top_share = float(group["top_ticker_gain_share"].dropna().astype(float).max()) if "top_ticker_gain_share" in group.columns else 0.0
        # AGENTS.md §4.2.5a gate uses the one-sided 95% LCB of the per-seed
        # delta, NOT the mean. macro_f1_lcb_95 is the LCB of macro_f1 itself;
        # delta_macro_f1_vs_dummy_lcb_95 is the LCB of the per-seed delta.
        delta_lcb_95 = float(one_sided_lcb_95(delta_values))
        macro_lcb_95 = float(one_sided_lcb_95(macro_values))
        band_key, band_text = band_from_delta(delta_lcb_95, positive_ticker_count)
        passed_threshold = (
            delta_lcb_95 >= IMPROVEMENT_LCB_MIN
            and positive_ticker_count >= IMPROVEMENT_TICKER_COUNT_MIN
        )
        # Weak-seed-evidence wording downgrade per §07 "Seed aggregation" +
        # AGENTS.md §4.2.5a: seed_count < WEAK_SEED_EVIDENCE_COUNT_THRESHOLD
        # forces allowed_wording_tag = "weak" and disqualifies "improvement"
        # wording, regardless of the band the LCB would otherwise reach.
        weak_seed_evidence_flag = bool(seed_count < WEAK_SEED_EVIDENCE_COUNT_THRESHOLD)
        if weak_seed_evidence_flag:
            allowed_wording_tag = "weak"
            passed_threshold = False
        else:
            allowed_wording_tag = band_key
        # Provenance: read horizon_k / threshold_bps from N05 row columns when
        # present; fall back to parse_label_config(label_config) otherwise.
        horizon_k = (
            _coerce_int_or_fallback(group["horizon_k"], _parsed_label["horizon_k"])
            if "horizon_k" in group.columns
            else int(_parsed_label["horizon_k"])
        )
        threshold_bps = (
            _coerce_float_or_fallback(group["threshold_bps"], _parsed_label["threshold_bps"])
            if "threshold_bps" in group.columns
            else float(_parsed_label["threshold_bps"])
        )
        full_coverage_records.append(
            {
                "artifact_source": "notebook05_official_validation_pooled",
                "notebook_stage": "07B",
                "row_class": "full_coverage",
                "model": "lightgbm",
                "profile_id": str(profile_id),
                "profile_role": str(profile_role),
                "label_config": _label_config_str,
                "horizon_k": int(horizon_k),
                "threshold_bps": float(threshold_bps),
                "feature_set": str(LOCKED_CANDIDATE_TUPLE["feature_set"]),
                "window_size": int(LOCKED_CANDIDATE_TUPLE["window_size"]),
                "seed_count": int(seed_count),
                "macro_f1_mean": float(np.mean(macro_values)) if len(macro_values) else float("nan"),
                "macro_f1_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
                "macro_f1_lcb_95": macro_lcb_95,
                "balanced_accuracy_mean": float(np.mean(bal_values)) if len(bal_values) else float("nan"),
                "balanced_accuracy_std": float(np.std(bal_values, ddof=1)) if len(bal_values) > 1 else 0.0,
                "accuracy_mean": float(np.mean(acc_values)) if len(acc_values) else float("nan"),
                "dummy_macro_f1_mean": float(np.mean(dummy_values)) if len(dummy_values) else float("nan"),
                "dummy_balanced_accuracy_mean": float(np.mean(dummy_bal_values)) if len(dummy_bal_values) else float("nan"),
                "delta_macro_f1_vs_dummy_mean": float(np.mean(delta_values)) if len(delta_values) else float("nan"),
                "delta_macro_f1_vs_dummy_lcb_95": delta_lcb_95,
                "delta_balanced_accuracy_vs_dummy_mean": float(np.mean(delta_bal_values)) if len(delta_bal_values) else float("nan"),
                "always_up_dummy_macro_f1_mean": float(np.mean(au_values)) if len(au_values) else float("nan"),
                "delta_macro_f1_vs_always_up_dummy_mean": float(np.mean(au_delta_values)) if len(au_delta_values) else float("nan"),
                "positive_ticker_count": positive_ticker_count,
                "top_ticker_gain_share": top_share,
                "validation_n": int(group["validation_n"].dropna().astype(int).max()) if "validation_n" in group.columns else 0,
                "scope": NOTEBOOK07_SCOPE,
                "decision_source": "notebook05_train_inner_winner",
                "allowed_wording_tag": allowed_wording_tag,
                "coverage": pd.NA,
                "coverage_source": pd.NA,
                "retained_n": pd.NA,
                "abstained_n": pd.NA,
                "random_abstention_macro_f1_mean": pd.NA,
                "delta_macro_f1_vs_random_abstention_mean": pd.NA,
                "_passed_AGENTS_md_4_2_5a": bool(passed_threshold),
                "_weak_seed_evidence": bool(weak_seed_evidence_flag),
            }
        )

    selective_records = []
    n06_paths = require_n06_artifacts_if_present(NOTEBOOK06_RESULTS_DIR)
    if n06_paths is not None and Path(n06_paths["coverage_grid"]).exists():
        coverage_grid = pd.read_csv(n06_paths["coverage_grid"])
        # N06 coverage_grid hard-stop for same-row dummy on selective rows.
        _required_n06_dummy_cols = (
            "same_row_stratified_dummy_macro_f1",
            "delta_macro_f1_vs_same_row_stratified_dummy",
        )
        for _col in _required_n06_dummy_cols:
            if _col not in coverage_grid.columns:
                raise ValueError(
                    f"N06 coverage_grid is missing required same-row dummy column "
                    f"'{_col}'. Selective rows reaching 07B MUST carry same-row "
                    "stratified dummy on retained rows (design §07B + AGENTS.md §4.2)."
                )
        for (profile_id, profile_role, coverage_target), group in coverage_grid.groupby(
            ["profile_id", "profile_role", "coverage_target"], dropna=False, sort=False
        ):
            macro_values = pd.to_numeric(group["macro_f1"], errors="coerce").dropna().to_numpy(dtype=float)
            delta_values = pd.to_numeric(
                group["delta_macro_f1_vs_same_row_stratified_dummy"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            if len(delta_values) == 0:
                raise ValueError(
                    f"N06 coverage_grid group (profile_id={profile_id!r}, "
                    f"coverage_target={coverage_target}) has all-NaN "
                    "delta_macro_f1_vs_same_row_stratified_dummy; selective rows "
                    "cannot reach 07B without a same-row dummy delta."
                )
            delta_rand_values = pd.to_numeric(
                group.get("delta_macro_f1_vs_random_abstention", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            rand_values = pd.to_numeric(
                group.get("random_abstention_macro_f1_mean", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            seed_count = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
            selective_delta_lcb_95 = float(one_sided_lcb_95(delta_values))
            selective_macro_lcb_95 = float(one_sided_lcb_95(macro_values))
            sel_horizon_k = (
                _coerce_int_or_fallback(group["horizon_k"], _parsed_label["horizon_k"])
                if "horizon_k" in group.columns
                else int(_parsed_label["horizon_k"])
            )
            sel_threshold_bps = (
                _coerce_float_or_fallback(group["threshold_bps"], _parsed_label["threshold_bps"])
                if "threshold_bps" in group.columns
                else float(_parsed_label["threshold_bps"])
            )
            selective_records.append(
                {
                    "artifact_source": "notebook06_coverage_grid",
                    "notebook_stage": "07B",
                    "row_class": "selective",
                    "model": "lightgbm",
                    "profile_id": str(profile_id),
                    "profile_role": str(profile_role),
                    "label_config": _label_config_str,
                    "horizon_k": int(sel_horizon_k),
                    "threshold_bps": float(sel_threshold_bps),
                    "feature_set": str(LOCKED_CANDIDATE_TUPLE["feature_set"]),
                    "window_size": int(LOCKED_CANDIDATE_TUPLE["window_size"]),
                    "seed_count": int(seed_count),
                    "macro_f1_mean": float(np.mean(macro_values)) if len(macro_values) else float("nan"),
                    "macro_f1_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
                    "macro_f1_lcb_95": selective_macro_lcb_95,
                    "balanced_accuracy_mean": float(np.mean(pd.to_numeric(group.get("balanced_accuracy", pd.Series(dtype=float)), errors="coerce").dropna())) if len(group) else float("nan"),
                    "balanced_accuracy_std": 0.0,
                    "accuracy_mean": float(np.mean(pd.to_numeric(group.get("accuracy", pd.Series(dtype=float)), errors="coerce").dropna())) if len(group) else float("nan"),
                    "dummy_macro_f1_mean": float(np.mean(pd.to_numeric(group.get("same_row_stratified_dummy_macro_f1", pd.Series(dtype=float)), errors="coerce").dropna())) if len(group) else float("nan"),
                    "dummy_balanced_accuracy_mean": float("nan"),
                    "delta_macro_f1_vs_dummy_mean": float(np.mean(delta_values)) if len(delta_values) else float("nan"),
                    "delta_macro_f1_vs_dummy_lcb_95": selective_delta_lcb_95,
                    "delta_balanced_accuracy_vs_dummy_mean": float("nan"),
                    "always_up_dummy_macro_f1_mean": float("nan"),
                    "delta_macro_f1_vs_always_up_dummy_mean": float("nan"),
                    "positive_ticker_count": int(group["positive_ticker_count"].dropna().astype(int).max()) if "positive_ticker_count" in group.columns else 0,
                    "top_ticker_gain_share": float(group["top_ticker_gain_share"].dropna().astype(float).max()) if "top_ticker_gain_share" in group.columns else 0.0,
                    "validation_n": int(group["eligible_n"].dropna().astype(int).max()) if "eligible_n" in group.columns else 0,
                    "scope": NOTEBOOK07_SCOPE,
                    "decision_source": "notebook06_fixed_coverage_grid",
                    "allowed_wording_tag": "diagnostic_only",
                    "coverage": float(coverage_target),
                    "coverage_source": "notebook06_fixed_grid",
                    "retained_n": int(group["retained_n"].dropna().astype(int).max()) if "retained_n" in group.columns else 0,
                    "abstained_n": int(group["abstained_n"].dropna().astype(int).max()) if "abstained_n" in group.columns else 0,
                    "random_abstention_macro_f1_mean": float(np.mean(rand_values)) if len(rand_values) else 0.0,
                    "delta_macro_f1_vs_random_abstention_mean": float(np.mean(delta_rand_values)) if len(delta_rand_values) else 0.0,
                    "_passed_AGENTS_md_4_2_5a": False,
                }
            )

    combined = pd.DataFrame(full_coverage_records + selective_records)
    combined = combined.drop(
        columns=["_passed_AGENTS_md_4_2_5a", "_weak_seed_evidence"], errors="ignore"
    )
    combined.to_csv(OUTPUT_FILES["final_validation_comparison"], index=False)
    validate_final_validation_comparison_frame(pd.read_csv(OUTPUT_FILES["final_validation_comparison"]))

    append_ledger_row(
        artifact=str(OUTPUT_FILES["final_validation_comparison"]),
        notebook_stage="07B",
        decision_made="final_comparison_built",
        decision_timing="after_official_validation_read",
        decision_surface="comparison_table",
        seeds_used=str(int(pd.to_numeric(combined["seed_count"], errors="coerce").max())) if not combined.empty else "0",
        thresholds_or_coverages_considered="n06_fixed_grid" if selective_records else "n/a",
        official_validation_rows_inspected=0,
        official_validation_informed_decision=False,
        allowed_wording="locked-row final comparison; band tag attached",
        forbidden_wording="threshold or coverage selection wording forbidden in 07",
    )
    write_run_manifest("07B_final_validation_comparison", lockfile=lockfile)
    print("07B wrote:", OUTPUT_FILES["final_validation_comparison"])
else:
    print("RUN_07B_FINAL_VALIDATION_COMPARISON is False; 07B not run.")
'''


CELL_07C = r'''
if RUN_07C_VALIDATION_BUDGET_LEDGER:
    require_artifact_lockfile()
    target_path = OUTPUT_FILES["validation_budget_ledger"]
    if not target_path.exists():
        raise FileNotFoundError(
            f"Ledger not on disk: {target_path}. Run 07A (and 07B / 07D / etc.) "
            "first; each phase appends + flushes its intent row before reading "
            "official validation, so 07C should only audit what's already there."
        )
    # 07C is audit-only. It MUST NOT direct-write the ledger, because direct
    # write bypasses the prefix-invariance check inside flush_ledger_to_disk()
    # and could silently overwrite project-level rows appended by N08/thesis
    # downstream. Instead, flush any unflushed in-memory rows under the
    # invariance check, then re-read disk and validate.
    flush_ledger_to_disk()
    on_disk = pd.read_csv(target_path, dtype=str, keep_default_na=False)
    validate_ledger_frame(on_disk)
    write_run_manifest(
        "07C_validation_budget_ledger",
        extra={"ledger_row_count": int(len(on_disk))},
    )
    print("07C ledger validated. row_count:", len(on_disk))
else:
    print("RUN_07C_VALIDATION_BUDGET_LEDGER is False; 07C not run.")
'''


CELL_07D = r'''
if RUN_07D_ROBUSTNESS_AND_CONCENTRATION:
    lockfile = require_artifact_lockfile()
    # Before-read ledger entry: 07D reads N05 per-ticker rows to build
    # diagnostic robustness tables.
    append_ledger_row(
        artifact=str(OUTPUT_FILES["per_ticker_robustness"]),
        notebook_stage="07D",
        decision_made="reading_n05_per_ticker_for_robustness_diagnostics",
        decision_timing="before_official_validation_read",
        decision_surface="diagnostic_tables",
        seeds_used="locked",
        official_validation_rows_inspected=1,
        diagnostic_only_readout=True,
        allowed_wording="07D intent: per-ticker / seed / concentration diagnostics",
        forbidden_wording="best_ticker; tradable_per_ticker",
    )

    n05_paths = require_n05_artifacts(NOTEBOOK05_RESULTS_DIR)
    per_ticker_05 = read_csv_required(n05_paths["official_per_ticker"])
    primary_profile_id = lockfile["selected_profile_id"]
    per_ticker_rows = per_ticker_05[per_ticker_05["profile_id"].astype(str) == str(primary_profile_id)].copy()
    if per_ticker_rows.empty:
        raise ValueError(f"N05 per_ticker has no rows for {primary_profile_id}")
    if "ticker" not in per_ticker_rows.columns:
        if "ticker_or_pooled" not in per_ticker_rows.columns:
            raise ValueError(
                "N05 per_ticker is missing ticker identity column; expected "
                "'ticker' or 'ticker_or_pooled'."
            )
        per_ticker_rows["ticker"] = per_ticker_rows["ticker_or_pooled"].astype(str)

    per_ticker_records = []
    seed_records = []
    for ticker, group in per_ticker_rows.groupby("ticker", dropna=False, sort=True):
        macro_values = pd.to_numeric(group.get("macro_f1", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        dummy_values = pd.to_numeric(group.get("stratified_dummy_macro_f1", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        delta_values = pd.to_numeric(group.get("delta_macro_f1_vs_stratified_dummy", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        delta_mean = float(np.mean(delta_values)) if len(delta_values) else float("nan")
        delta_lcb = float(one_sided_lcb_95(delta_values))
        per_ticker_records.append(
            {
                "ticker": str(ticker),
                "profile_id": str(primary_profile_id),
                "macro_f1_mean": float(np.mean(macro_values)) if len(macro_values) else float("nan"),
                "dummy_macro_f1_mean": float(np.mean(dummy_values)) if len(dummy_values) else float("nan"),
                "delta_macro_f1_vs_dummy_mean": delta_mean,
                "delta_macro_f1_vs_dummy_lcb_95": delta_lcb,
                "seed_count": int(group["seed"].nunique()) if "seed" in group.columns else int(len(group)),
                "is_positive_ticker": bool(delta_mean > 0.0),
                "scope": "diagnostic",
            }
        )

    if "seed" in per_ticker_rows.columns:
        for seed, group in per_ticker_rows.groupby("seed", dropna=False, sort=True):
            macro_values = pd.to_numeric(group.get("macro_f1", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            delta_values = pd.to_numeric(group.get("delta_macro_f1_vs_stratified_dummy", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
            seed_records.append(
                {
                    "seed": int(seed) if pd.notna(seed) else -1,
                    "profile_id": str(primary_profile_id),
                    "macro_f1_mean": float(np.mean(macro_values)) if len(macro_values) else float("nan"),
                    "macro_f1_std": float(np.std(macro_values, ddof=1)) if len(macro_values) > 1 else 0.0,
                    "delta_macro_f1_vs_dummy_mean": float(np.mean(delta_values)) if len(delta_values) else float("nan"),
                    "delta_macro_f1_vs_dummy_lcb_95": float(one_sided_lcb_95(delta_values)),
                    "ticker_count": int(group["ticker"].nunique()) if "ticker" in group.columns else 0,
                    "scope": "diagnostic",
                }
            )

    pd.DataFrame(per_ticker_records).to_csv(OUTPUT_FILES["per_ticker_robustness"], index=False)
    pd.DataFrame(seed_records).to_csv(OUTPUT_FILES["seed_robustness"], index=False)

    positive_ticker_count = sum(1 for r in per_ticker_records if r["is_positive_ticker"])
    positive_deltas = [r["delta_macro_f1_vs_dummy_mean"] for r in per_ticker_records if r["is_positive_ticker"] and not math.isnan(r["delta_macro_f1_vs_dummy_mean"])]
    top_share = float(max(positive_deltas) / sum(positive_deltas)) if positive_deltas else 0.0
    seed_count_for_primary = int(np.mean([r["seed_count"] for r in per_ticker_records])) if per_ticker_records else 0

    concentration_records = [
        {
            "profile_id": str(primary_profile_id),
            "positive_ticker_count": positive_ticker_count,
            "top_ticker_gain_share": top_share,
            "concentration_warning_top_share_max": CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX,
            "concentration_warning_positive_ticker_count_min": CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN,
            "concentration_warning_triggered": bool(
                positive_ticker_count < CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN
                or top_share > CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX
            ),
            "weak_seed_evidence": bool(seed_count_for_primary < WEAK_SEED_EVIDENCE_COUNT_THRESHOLD),
            "scope": "diagnostic",
        }
    ]
    n06_paths = require_n06_artifacts_if_present(NOTEBOOK06_RESULTS_DIR)
    if n06_paths is not None and Path(n06_paths["concentration_guardrails"]).exists():
        n06_conc = pd.read_csv(n06_paths["concentration_guardrails"])
        for _, row in n06_conc.iterrows():
            concentration_records.append(
                {
                    "profile_id": str(row.get("profile_id", "")),
                    "positive_ticker_count": int(row.get("retained_ticker_count", 0) or 0),
                    "top_ticker_gain_share": float(row.get("top_ticker_retained_share", 0.0) or 0.0),
                    "concentration_warning_top_share_max": CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX,
                    "concentration_warning_positive_ticker_count_min": CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN,
                    "concentration_warning_triggered": bool(row.get("warning_guardrail_triggered", False)),
                    "weak_seed_evidence": False,
                    "scope": "diagnostic",
                }
            )
    pd.DataFrame(concentration_records).to_csv(OUTPUT_FILES["concentration_summary"], index=False)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["per_ticker_robustness"]),
        notebook_stage="07D",
        decision_made="per_ticker_seed_concentration_diagnostics_written",
        decision_timing="after_official_validation_read",
        decision_surface="diagnostic_tables",
        seeds_used=str(seed_count_for_primary),
        official_validation_rows_inspected=0,
        diagnostic_only_readout=True,
        allowed_wording="robustness diagnostics; per-ticker delta vs same-row dummy",
        forbidden_wording="best_ticker; tradable_per_ticker",
    )
    write_run_manifest("07D_robustness_and_concentration", lockfile=lockfile)
    print(
        "07D wrote:",
        OUTPUT_FILES["per_ticker_robustness"],
        OUTPUT_FILES["seed_robustness"],
        OUTPUT_FILES["concentration_summary"],
    )
else:
    print("RUN_07D_ROBUSTNESS_AND_CONCENTRATION is False; 07D not run.")
'''


CELL_07E = r'''
if RUN_07E_EXPLAINABILITY_APPENDIX:
    lockfile = require_artifact_lockfile()
    if not OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS:
        raise ValueError(
            "OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS must be True "
            "before 07E may emit any importance artifact. SHAP / permutation / "
            "split / gain importance are diagnostic-only and may not be used as "
            "a selection gate per design 07E + AGENTS.md §4."
        )
    # Before-read ledger entry: 07E reads N05 official_pooled (and per-seed
    # prediction artifacts) to derive importance appendices. Per AGENTS.md §4.3,
    # the intent row MUST be appended BEFORE the read.
    append_ledger_row(
        artifact=str(NOTEBOOK05_RESULTS_DIR),
        notebook_stage="07E",
        decision_made="reading_n05_pooled_and_predictions_for_explainability",
        decision_timing="before_official_validation_read",
        decision_surface="appendix_diagnostic",
        seeds_used="locked",
        official_validation_rows_inspected=1,
        diagnostic_only_readout=True,
        allowed_wording="07E intent: read N05 prediction artifacts for split/gain importance",
        forbidden_wording="SHAP_proves_causal; selected_by_SHAP",
    )

    n05_paths = require_n05_artifacts(NOTEBOOK05_RESULTS_DIR)
    pooled = read_csv_required(n05_paths["official_pooled"])
    primary_profile_id = lockfile["selected_profile_id"]
    rows = pooled[
        (pooled["profile_id"].astype(str) == str(primary_profile_id))
        & pooled["prediction_artifact"].astype(str).str.strip().astype(bool)
    ].copy()
    if rows.empty:
        raise ValueError(f"No prediction_artifact rows for primary profile {primary_profile_id}")

    split_records = []
    gain_records = []
    pred_contrib_records = []
    feature_groups = {"price_return": [], "volume": [], "time_of_day": [], "technical_indicator": []}

    for _, row in rows.iterrows():
        artifact_raw = str(row.get("prediction_artifact", "")).strip()
        if not artifact_raw:
            continue
        _check_no_holdout_or_test_path(artifact_raw)
        artifact_path = Path(artifact_raw)
        if not artifact_path.is_absolute():
            artifact_path = NOTEBOOK05_RESULTS_DIR / artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"N05 prediction artifact missing for 07E: {artifact_path}")

        with np.load(artifact_path, allow_pickle=True) as data:
            keys = set(data.files)
            feature_names = list(data["feature_names"]) if "feature_names" in keys else []
            split_imp = list(data["split_importance"]) if "split_importance" in keys else []
            gain_imp = list(data["gain_importance"]) if "gain_importance" in keys else []

        seed_int = int(row.get("seed", -1))
        for name, value in zip(feature_names, split_imp):
            split_records.append(
                {
                    "profile_id": str(primary_profile_id),
                    "seed": seed_int,
                    "feature": str(name),
                    "split_importance": float(value),
                    "scope": "diagnostic",
                }
            )
        for name, value in zip(feature_names, gain_imp):
            gain_records.append(
                {
                    "profile_id": str(primary_profile_id),
                    "seed": seed_int,
                    "feature": str(name),
                    "gain_importance": float(value),
                    "scope": "diagnostic",
                }
            )

    pd.DataFrame(split_records).to_csv(OUTPUT_FILES["lightgbm_importance_split"], index=False)
    pd.DataFrame(gain_records).to_csv(OUTPUT_FILES["lightgbm_importance_gain"], index=False)
    pd.DataFrame(pred_contrib_records).to_csv(OUTPUT_FILES["lightgbm_pred_contrib_summary"], index=False)

    manifest = {
        "scope": "diagnostic",
        "created_utc": utc_now_iso(),
        "emitted_items": ["split_importance", "gain_importance"],
        "pred_contrib_emitted": False,
        "shap_emitted": False,
        "shap_gate": {
            "OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL": bool(OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL),
            "explainability_upgrade_record": {},
            "design_doc_sha256_at_approval": EXPECTED_DESIGN_DOC_SHA256,
            "no_selection_clause": (
                "SHAP outputs cannot retire features, add features, reweight features, "
                "or alter §07B / §07H wording."
            ),
        },
        "feature_group_summary": {
            group: int(sum(1 for record in gain_records if record["feature"].lower().startswith(group)))
            for group in feature_groups
        },
        "caveats": [
            "feature importance is model-specific",
            "correlated features can redistribute importance",
            "SHAP values explain predictions under a specified background/perturbation assumption",
            "no feature is added, removed, reweighted, or promoted from this appendix",
        ],
        "selection_gate_violation": False,
    }
    write_json(OUTPUT_FILES["explainability_manifest"], manifest)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["lightgbm_importance_gain"]),
        notebook_stage="07E",
        decision_made="explainability_appendix_written",
        decision_timing="after_official_validation_read",
        decision_surface="appendix_diagnostic",
        seeds_used=str(int(rows["seed"].nunique())),
        official_validation_rows_inspected=0,
        diagnostic_only_readout=True,
        allowed_wording="split/gain importance appendix (diagnostic)",
        forbidden_wording="SHAP_proves_causal; selected_by_SHAP",
    )
    write_run_manifest("07E_explainability_appendix", lockfile=lockfile)
    print(
        "07E wrote:",
        OUTPUT_FILES["lightgbm_importance_gain"],
        OUTPUT_FILES["lightgbm_importance_split"],
        OUTPUT_FILES["explainability_manifest"],
    )
else:
    print("RUN_07E_EXPLAINABILITY_APPENDIX is False; 07E not run.")
'''


CELL_07F = r'''
if RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX:
    lockfile = require_artifact_lockfile()
    alpha_policy = lockfile.get("null_control_alpha_policy", NULL_CONTROL_ALPHA_POLICY)
    allocations = dict(alpha_policy.get("allocations", {}))
    if sum(allocations.values()) <= 0.0:
        raise ValueError(
            "07A lockfile null_control_alpha_policy.allocations sum is 0; "
            "07F cannot run without pre-allocated alpha."
        )

    # Design 07F option 1: read-only reporting of an existing pre-registered
    # null-control artifact. 07F does NOT synthesize a chronology-aware null
    # in-line; a permutation over a macro_f1 mean vector is not a
    # chronology-aware null and would be a methodologically dishonest stand-in
    # for the design's promise. The operator must point
    # PRE_REGISTERED_NULL_CONTROL_PATH at a frozen artifact whose chronology
    # design is documented separately.
    if not PRE_REGISTERED_NULL_CONTROL_PATH:
        raise FileNotFoundError(
            "07F requires PRE_REGISTERED_NULL_CONTROL_PATH to point at a frozen "
            "pre-registered null-control artifact (design 07F option 1: read-only "
            "reporting of an existing pre-registered null-control artifact). 07F "
            "will not synthesize a chronology-aware null in-cell, because a naive "
            "permutation over macro_f1 means is not chronology-aware. Freeze the "
            "null design separately, save the artifact, and set the path."
        )
    src_path = Path(PRE_REGISTERED_NULL_CONTROL_PATH)
    if not src_path.exists():
        raise FileNotFoundError(
            f"Pre-registered null-control artifact missing: {src_path}"
        )

    # Before-read ledger entry: 07F reads the pre-registered null-control artifact.
    append_ledger_row(
        artifact=str(src_path),
        notebook_stage="07F",
        decision_made="reading_pre_registered_null_control_artifact",
        decision_timing="before_official_validation_read",
        decision_surface="appendix_diagnostic",
        seeds_used="locked",
        official_validation_rows_inspected=1,
        diagnostic_only_readout=True,
        allowed_wording="07F intent: read-only reporting of pre-registered null-control artifact",
        forbidden_wording=(
            "statistically proven to generalize; passes holdout/test; proves market profitability"
        ),
    )

    src_df = pd.read_csv(src_path)
    missing_cols = set(PRE_REGISTERED_NULL_CONTROL_REQUIRED_COLUMNS) - set(src_df.columns)
    if missing_cols:
        raise ValueError(
            f"Pre-registered null-control artifact missing required columns: "
            f"{sorted(missing_cols)}; expected {list(PRE_REGISTERED_NULL_CONTROL_REQUIRED_COLUMNS)}"
        )

    alpha_remaining = float(alpha_policy.get("alpha_total", NULL_CONTROL_ALPHA_TOTAL))
    derived_rows = []
    for _, row in src_df.iterrows():
        design = str(row["null_design"]).strip()
        alpha_share = float(allocations.get(design, 0.0))
        if alpha_share <= 0.0:
            continue
        if alpha_share > alpha_remaining + 1e-9:
            raise ValueError(
                f"07F alpha allocation for {design!r}={alpha_share} exceeds remaining {alpha_remaining}"
            )
        derived = {col: row[col] for col in src_df.columns}
        derived["alpha_share_consumed"] = float(alpha_share)
        derived["alpha_remaining_after_row"] = float(alpha_remaining - alpha_share)
        derived["source_artifact"] = str(src_path)
        derived["scope"] = "diagnostic"
        derived_rows.append(derived)
        alpha_remaining -= alpha_share

    if not derived_rows:
        raise ValueError(
            "07F read the pre-registered null-control artifact but no rows matched "
            "any allocation in null_control_alpha_policy.allocations; nothing to "
            "report."
        )

    pd.DataFrame(derived_rows).to_csv(OUTPUT_FILES["null_control_diagnostic"], index=False)
    # permutation_importance_diagnostic is left empty under design option 1.
    # If/when the operator promotes 07F to option 2/3/4 (day-block, circular
    # within-block, feature-family within-block), this output should be
    # populated by a separately frozen permutation procedure, not in-cell.
    pd.DataFrame(
        [
            {
                "profile_id": str(lockfile.get("selected_profile_id", "")),
                "permutation_repeats": 0,
                "design": "deferred_to_separately_frozen_permutation_procedure",
                "source_artifact": "",
                "scope": "diagnostic",
            }
        ]
    ).to_csv(OUTPUT_FILES["permutation_importance_diagnostic"], index=False)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["null_control_diagnostic"]),
        notebook_stage="07F",
        decision_made="null_control_appendix_written_from_pre_registered_source",
        decision_timing="after_official_validation_read",
        decision_surface="appendix_diagnostic",
        seeds_used="locked",
        official_validation_rows_inspected=0,
        diagnostic_only_readout=True,
        allowed_wording=(
            "observed validation-only delta vs pre-registered null-control distribution under the source design"
        ),
        forbidden_wording=(
            "statistically proven to generalize; passes holdout/test; proves market profitability"
        ),
    )
    write_run_manifest(
        "07F_permutation_null_control_appendix",
        lockfile=lockfile,
        extra={
            "alpha_remaining_after_07F": float(alpha_remaining),
            "pre_registered_null_control_path": str(src_path),
            "design_mode": "read_only_existing_artifact_design_option_1",
        },
    )
    print(
        "07F wrote:",
        OUTPUT_FILES["null_control_diagnostic"],
        "alpha_remaining:",
        alpha_remaining,
    )
else:
    print("RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX is False; 07F not run.")
'''


CELL_07G = r'''
if RUN_07G_GAP_AUDIT_FOR_08X:
    lockfile = require_artifact_lockfile()
    comparison_path = OUTPUT_FILES["final_validation_comparison"]
    if not comparison_path.exists():
        raise FileNotFoundError(f"07G requires 07B output: {comparison_path}")
    # Before-read ledger entry: 07G reads the comparison frame whose rows are
    # derived from N05/N06 official-validation metrics. Per AGENTS.md §4.3,
    # any downstream read of an official-validation-derived artifact MUST
    # append an intent row first.
    append_ledger_row(
        artifact=str(comparison_path),
        notebook_stage="07G",
        decision_made="reading_final_comparison_for_gap_audit",
        decision_timing="before_official_validation_read",
        decision_surface="gap_table",
        seeds_used="locked",
        official_validation_rows_inspected=1,
        diagnostic_only_readout=True,
        allowed_wording="07G intent: read final comparison to route gaps to 08X",
        forbidden_wording="08X should use candidate X; SHAP says use feature Y",
    )

    comparison = pd.read_csv(comparison_path)
    full = comparison[comparison["row_class"] == "full_coverage"].copy()
    if full.empty:
        raise ValueError("07G requires at least one full_coverage row in the comparison.")

    # AGENTS.md §4.2.5a gate: use the one-sided 95% LCB of the per-seed delta,
    # NOT the mean. The mean is bookkeeping; the LCB binds the band routing.
    if "delta_macro_f1_vs_dummy_lcb_95" not in full.columns:
        raise ValueError(
            "07G requires delta_macro_f1_vs_dummy_lcb_95 column in the comparison frame; "
            "regenerate 07B output with the LCB column."
        )
    delta_lcb_95 = float(full["delta_macro_f1_vs_dummy_lcb_95"].dropna().min())
    pos_ticker = int(full["positive_ticker_count"].dropna().astype(int).min()) if "positive_ticker_count" in full.columns else 0
    top_share = float(full["top_ticker_gain_share"].dropna().astype(float).max()) if "top_ticker_gain_share" in full.columns else 0.0
    band_key, _ = band_from_delta(delta_lcb_95, pos_ticker)

    gaps = []

    def _gap(gap_id, category, target_route, priority, requires_extra_preregistration, why, allowed_next_route, source):
        gaps.append(
            {
                "gap_id": gap_id,
                "gap_category": category,
                "evidence_source": source,
                "observed_issue": why,
                "why_it_matters": why,
                "allowed_next_route": allowed_next_route,
                "target_route": target_route,
                "forbidden_in_07": (
                    "feature reselection; new model; threshold search; null-control as selector"
                ),
                "minimum_pre_registration_needed": (
                    "pre-register new search degree of freedom in 08X/08F/08O design doc"
                ),
                "priority": priority,
                "requires_extra_preregistration": bool(requires_extra_preregistration),
                "scope": NOTEBOOK07_SCOPE,
            }
        )

    if band_key == "no_signal":
        _gap("G01", "metric_gap", "08X", "must", True,
             "no detected validation-only signal under locked candidate",
             "exploratory feature / window / horizon search separately frozen",
             "07B")
        _gap("G02", "generalization_gap", "08X", "useful", True,
             "validation-only baseline does not generalize to dummy delta",
             "redesign candidate with new label / feature decomposition",
             "07B")
    elif band_key == "weak":
        _gap("G03", "metric_gap", "08X", "useful", True,
             "weak signal band (0 < delta < 0.005)",
             "exploratory targeting larger effect size",
             "07B")
    elif band_key == "concentration_limited":
        _gap("G04", "concentration_gap", "08X", "must", True,
             f"positive_ticker_count={pos_ticker} below threshold",
             "broaden tickers / verify concentration source under separate freeze",
             "07D")

    if top_share > CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX:
        _gap("G05", "concentration_gap", "08X", "useful", True,
             f"top_ticker_gain_share={top_share:.3f} > {CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX}",
             "concentration redesign; not a 07 fix",
             "07D")

    if not OUTPUT_FILES["explainability_manifest"].exists():
        _gap("G06", "explanation_gap", "08X", "optional", False,
             "07E explainability appendix not produced",
             "produce split/gain importance; SHAP requires explicit operator approval",
             "07E")
    if not OUTPUT_FILES["null_control_diagnostic"].exists():
        _gap("G07", "null_control_gap", "08X", "optional", True,
             "07F null-control appendix not produced",
             "pre-register null design and alpha allocation before running",
             "07F")
    if not OUTPUT_FILES["lightgbm_importance_gain"].exists():
        _gap("G08", "dependency_gap", "08X", "optional", False,
             "LightGBM importance artifacts missing",
             "verify N05 prediction_artifact includes feature_names / split_importance / gain_importance",
             "07E")

    if band_key == "practical":
        _gap("G09", "paper_wording_gap", "none", "useful", False,
             "practical band reached; wording may use 'improvement' per AGENTS.md §4.2.5a",
             "compose paper-ready synthesis at 07H",
             "07H")

    pd.DataFrame(gaps).to_csv(OUTPUT_FILES["gap_audit_for_08x"], index=False)
    append_ledger_row(
        artifact=str(OUTPUT_FILES["gap_audit_for_08x"]),
        notebook_stage="07G",
        decision_made="gap_audit_written_for_08x",
        decision_timing="after_official_validation_read",
        decision_surface="gap_table",
        seeds_used="locked",
        official_validation_rows_inspected=0,
        diagnostic_only_readout=True,
        allowed_wording="route gaps to 08X / 08F / 08O without selecting 08X candidate",
        forbidden_wording="08X should use candidate X; SHAP says use feature Y",
    )
    write_run_manifest("07G_gap_audit_for_08x", lockfile=lockfile, extra={"gap_count": int(len(gaps))})
    print("07G wrote:", OUTPUT_FILES["gap_audit_for_08x"], "gap_count:", len(gaps))
else:
    print("RUN_07G_GAP_AUDIT_FOR_08X is False; 07G not run.")
'''


CELL_07H = r'''
if RUN_07H_PAPER_READY_SYNTHESIS:
    lockfile = require_artifact_lockfile()
    comparison_path = OUTPUT_FILES["final_validation_comparison"]
    if not comparison_path.exists():
        raise FileNotFoundError(f"07H requires 07B output: {comparison_path}")
    # Before-read ledger entry: 07H reads the comparison frame whose rows are
    # derived from N05/N06 official-validation metrics. Per AGENTS.md §4.3,
    # the intent row MUST be appended BEFORE the read.
    append_ledger_row(
        artifact=str(comparison_path),
        notebook_stage="07H",
        decision_made="reading_final_comparison_for_paper_ready_synthesis",
        decision_timing="before_official_validation_read",
        decision_surface="wording_record",
        seeds_used="locked",
        official_validation_rows_inspected=1,
        allowed_wording="07H intent: read final comparison to compose paper-ready synthesis",
        forbidden_wording="forbidden token list per design 07H",
        risk_note="wording bound by AGENTS.md §4.2.5a",
    )

    comparison = pd.read_csv(comparison_path)
    full = comparison[comparison["row_class"] == "full_coverage"].copy()
    if full.empty:
        raise ValueError("07H requires at least one full_coverage row.")

    # AGENTS.md §4.2.5a gate: use the one-sided 95% LCB of the per-seed delta.
    # Using the mean would be a paper-pressure rewrite of the gate.
    if "delta_macro_f1_vs_dummy_lcb_95" not in full.columns:
        raise ValueError(
            "07H requires delta_macro_f1_vs_dummy_lcb_95 column in the comparison frame; "
            "regenerate 07B output with the LCB column."
        )
    macro_lcb_95 = float(full["macro_f1_lcb_95"].dropna().min()) if "macro_f1_lcb_95" in full.columns else 0.0
    delta_lcb_95 = float(full["delta_macro_f1_vs_dummy_lcb_95"].dropna().min())
    pos_ticker = int(full["positive_ticker_count"].dropna().astype(int).min()) if "positive_ticker_count" in full.columns else 0
    band_key, band_text = band_from_delta(delta_lcb_95, pos_ticker)
    real_pass = delta_lcb_95 >= IMPROVEMENT_LCB_MIN and pos_ticker >= IMPROVEMENT_TICKER_COUNT_MIN
    improvement_allowed = bool(real_pass)

    if improvement_allowed:
        results_paragraph = (
            "Under the locked chronological validation-only route, the selected "
            "LightGBM configuration produced an improvement over same-row "
            "stratified dummy baselines across the five-ticker panel."
        )
    elif band_key == "no_signal":
        results_paragraph = (
            "Under the locked chronological validation-only route, the selected "
            "LightGBM configuration produced no detected validation-only signal "
            "over same-row stratified dummy baselines."
        )
    else:
        results_paragraph = (
            "Under the locked chronological validation-only route, the selected "
            "LightGBM configuration produced a weak or mixed validation-only delta "
            "over same-row stratified dummy baselines across the five-ticker panel."
        )

    robustness_paragraph = (
        "Robustness diagnostics summarize per-ticker, per-seed, concentration, "
        "and optional explanation/null-control behavior. Diagnostic outputs are "
        "appendix-only and do not alter the locked candidate."
    )
    limitation_paragraph = (
        "Because the official validation partition was reused for confirmation "
        "and diagnostics, these results support cautious validation-only thesis "
        "wording, not holdout/test, operational-use, or profitability claims."
    )

    caveat_phrases = [
        "validation-only",
        "not holdout-ready",
        "chronology-respecting baselines",
        "diagnostic-only appendices",
    ]
    forbidden_blocked = []
    for paragraph in (results_paragraph, robustness_paragraph, limitation_paragraph):
        forbidden_blocked.extend(forbidden_phrase_blocks(paragraph))
    forbidden_blocked = sorted(set(forbidden_blocked))

    reproducibility_pointers = []
    sentence_id_seed = "07H_results_001"
    reproducibility_pointers.append(
        {
            "sentence_id": hashlib.sha256(sentence_id_seed.encode("utf-8")).hexdigest()[:12],
            "artifact": str(OUTPUT_FILES["final_validation_comparison"].name),
            "row_filter": f"row_class=='full_coverage' and profile_id=='{lockfile['selected_profile_id']}'",
            "expected_value_summary": band_text,
        }
    )
    reproducibility_pointers.append(
        {
            "sentence_id": hashlib.sha256("07H_robustness_001".encode("utf-8")).hexdigest()[:12],
            "artifact": str(OUTPUT_FILES["per_ticker_robustness"].name),
            "row_filter": "is_positive_ticker==True",
            "expected_value_summary": f"positive_ticker_count={pos_ticker}",
        }
    )
    reproducibility_pointers.append(
        {
            "sentence_id": hashlib.sha256("07H_limit_001".encode("utf-8")).hexdigest()[:12],
            "artifact": str(OUTPUT_FILES["validation_budget_ledger"].name),
            "row_filter": "notebook_stage in ['07A','07B','07D']",
            "expected_value_summary": "validation_only reuse counted in cumulative ledger",
        }
    )

    falsification_conditions = [
        {
            "name": "one_extra_year_lcb_below_003",
            "trigger": "lcb_delta_macro_f1_vs_dummy < 0.003 under one extra year of same-candidate N05/N06 data",
            "action": "retract 'improvement' wording; record erratum",
        },
        {
            "name": "ticker_negative_two_consecutive_windows",
            "trigger": "any single ticker per-ticker delta negative for two consecutive monitoring windows",
            "action": "retract per-ticker positivity claim",
        },
        {
            "name": "replication_on_separate_feature_set_fails",
            "trigger": "future replication on separately frozen feature set fails AGENTS.md §4.2.5a threshold",
            "action": "downgrade generalization claim to 'specific to this feature set'",
        },
        {
            "name": "concentration_severe_under_new_buckets",
            "trigger": "06 concentration guardrails trigger severe status under new bucket definitions",
            "action": "upgrade concentration caveat to hard limitation",
        },
    ]

    thesis_kit = {
        "results_paragraph": results_paragraph,
        "robustness_paragraph": robustness_paragraph,
        "limitation_paragraph": limitation_paragraph,
        "caveat_phrases_used": caveat_phrases,
        "forbidden_phrases_blocked_at_runtime": forbidden_blocked,
        "reproducibility_pointers": reproducibility_pointers,
        "reproducibility_pointer_rules": {
            "every_sentence_must_have_at_least_one_pointer": True,
            "every_pointer_sentence_id_must_be_unique_within_kit": True,
            "artifact_path_allowlist_globs": [
                "notebook07_*.csv",
                "notebook07_*.json",
                "notebook05_official_validation_*.csv",
                "notebook06_*.csv",
            ],
        },
        "improvement_wording_applied": bool(improvement_allowed),
        "improvement_threshold_check": {
            "delta_macro_f1_vs_dummy_lcb_95": float(delta_lcb_95),
            "positive_ticker_count": int(pos_ticker),
            "passed_per_AGENTS_md_4_2_5a": bool(real_pass),
        },
        "falsification_conditions": falsification_conditions,
        "band_key": band_key,
        "band_text": band_text,
    }
    validate_thesis_paragraph_kit(thesis_kit)
    write_json(OUTPUT_FILES["thesis_paragraph_kit"], thesis_kit)

    synthesis_md = "\n\n".join(
        [
            "# Notebook 07 - Paper-Ready Synthesis",
            "## Results",
            results_paragraph,
            "## Robustness",
            robustness_paragraph,
            "## Limitations",
            limitation_paragraph,
        ]
    )
    OUTPUT_FILES["paper_ready_synthesis"].write_text(synthesis_md + "\n", encoding="utf-8")

    decision_record = {
        "scope": NOTEBOOK07_SCOPE,
        "created_utc": utc_now_iso(),
        "band_key": band_key,
        "band_text": band_text,
        "improvement_wording_applied": bool(improvement_allowed),
        "allowed_wording": [
            "validation-only synthesis",
            "locked candidate; chronology-respecting baselines",
            "per-ticker robustness as diagnostic only",
            "holdout/test remains closed",
        ],
        "forbidden_wording": [
            "the model is final",
            "the model is holdout-ready",
            "the selective threshold is final",
            "the model is tradable or profitable",
            "SHAP proves the causal driver",
            "permutation importance selects the final feature set",
            "ECE/AURC chooses the final threshold",
            "the 07 null-control appendix proves generalization",
        ],
        "forbidden_phrase_regex": FORBIDDEN_PHRASE_REGEX,
        "forbidden_phrases_blocked_at_runtime": forbidden_blocked,
        "selected_profile_id": lockfile["selected_profile_id"],
        "selected_profile_source": lockfile["selected_profile_source"],
        "candidate_tuple": LOCKED_CANDIDATE_TUPLE,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
    }
    write_json(OUTPUT_FILES["decision_and_wording_record"], decision_record)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["paper_ready_synthesis"]),
        notebook_stage="07H",
        decision_made="paper_ready_synthesis_written",
        decision_timing="after_official_validation_read",
        decision_surface="wording_record",
        seeds_used="locked",
        official_validation_rows_inspected=0,
        allowed_wording="band={band_key}; improvement={imp}".format(band_key=band_key, imp=improvement_allowed),
        forbidden_wording="forbidden token list per design 07H",
        risk_note="wording bound by AGENTS.md §4.2.5a",
    )
    write_run_manifest(
        "07H_paper_ready_synthesis",
        lockfile=lockfile,
        extra={"thesis_paragraph_kit_path": str(OUTPUT_FILES["thesis_paragraph_kit"])},
    )
    print(
        "07H wrote:",
        OUTPUT_FILES["paper_ready_synthesis"],
        OUTPUT_FILES["thesis_paragraph_kit"],
        OUTPUT_FILES["decision_and_wording_record"],
    )
else:
    print("RUN_07H_PAPER_READY_SYNTHESIS is False; 07H not run.")
'''


CELL_07I = r'''
if RUN_07I_BACKUP_TO_GOOGLE_DRIVE:
    if not BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE:
        raise ValueError("Enable BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE before running 07I backup.")
    backup_manifest = upload_existing_outputs_to_drive()
    print("07I Drive backup manifest:", backup_manifest)
else:
    print("RUN_07I_BACKUP_TO_GOOGLE_DRIVE is False; Drive backup not run.")
'''


CELL_07J = r'''
if RUN_07J_WRITE_MONITORING_PLAN:
    lockfile = require_artifact_lockfile()
    thesis_kit_path = OUTPUT_FILES["thesis_paragraph_kit"]
    falsification_conditions = []
    if thesis_kit_path.exists():
        thesis_kit_payload = read_json(thesis_kit_path)
        falsification_conditions = list(thesis_kit_payload.get("falsification_conditions", []))

    monitoring_plan = {
        "scope": NOTEBOOK07_SCOPE,
        "created_utc": utc_now_iso(),
        "cadence_first_year": "quarterly",
        "cadence_after_first_year": "annual",
        "monitor_inputs": {
            "candidate_tuple_locked": LOCKED_CANDIDATE_TUPLE,
            "n05_artifacts": "notebook05_official_validation_pooled.csv (same-candidate new data only)",
            "n06_artifacts": "notebook06_coverage_grid.csv (same-candidate new data only)",
        },
        "monitor_scope_per_cycle": [
            "rerun §07B band comparison on same-candidate new data",
            "rerun §07D per-ticker / concentration on same-candidate new data",
        ],
        "monitor_must_not": [
            "retrain a model",
            "alter a threshold",
            "add or remove a feature",
            "change the window-size selection",
            "open holdout/test",
        ],
        "monitor_outcome_routing": {
            "any_falsification_condition_triggers": "write notebook07_erratum_recommended.json",
            "publication_action_is_human_only": True,
            "original_paper_text_not_retroactively_edited": True,
        },
        "falsification_conditions": falsification_conditions,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "candidate_tuple_locked": True,
    }
    write_json(OUTPUT_FILES["post_publication_monitoring_plan"], monitoring_plan)

    append_ledger_row(
        artifact=str(OUTPUT_FILES["post_publication_monitoring_plan"]),
        notebook_stage="07J",
        decision_made="post_publication_monitoring_plan_written",
        decision_timing="after_paper_ready_synthesis",
        decision_surface="monitoring_plan",
        seeds_used="locked",
        official_validation_rows_inspected=0,
        diagnostic_only_readout=True,
        allowed_wording="quarterly/annual monitor; erratum recommendation only",
        forbidden_wording="retrain; reopen holdout; revise model",
    )
    write_run_manifest("07J_post_publication_monitoring_plan", lockfile=lockfile)
    print("07J wrote:", OUTPUT_FILES["post_publication_monitoring_plan"])
else:
    print("RUN_07J_WRITE_MONITORING_PLAN is False; 07J not run.")
'''


def validate_code_cells(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook07_cell_{index}")


_FORBIDDEN_AST_CALLS = {"exec", "eval", "compile", "__import__"}


def _ast_check_forbidden_calls(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"notebook07_cell_{index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _FORBIDDEN_AST_CALLS:
                    raise AssertionError(
                        f"Forbidden AST call '{func.id}' found in cell {index} (design §07 item 17)"
                    )
                if isinstance(func, ast.Attribute) and func.attr.startswith(
                    ("select_", "best_", "optimal_", "run_hpo")
                ):
                    raise AssertionError(
                        f"Forbidden selector attribute '{func.attr}' found in cell {index}"
                    )


def validate_notebook(nb: nbformat.NotebookNode) -> None:
    nbformat.validate(nb)
    validate_code_cells(nb)
    _ast_check_forbidden_calls(nb)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    if any(cell.get("outputs") for cell in code_cells):
        raise AssertionError("Generated notebook must not contain saved outputs.")
    if [cell.get("execution_count") for cell in code_cells] != [None] * len(code_cells):
        raise AssertionError("Generated notebook code cell execution counts must be None.")
    source = "\n".join(cell.source for cell in code_cells)
    required = (
        'NOTEBOOK07_SCOPE = "validation_only"',
        "RUN_07A_LOCKFILE_SCOPE_GATE = False",
        "RUN_07B_FINAL_VALIDATION_COMPARISON = False",
        "RUN_07C_VALIDATION_BUDGET_LEDGER = False",
        "RUN_07D_ROBUSTNESS_AND_CONCENTRATION = False",
        "RUN_07E_EXPLAINABILITY_APPENDIX = False",
        "RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX = False",
        "RUN_07G_GAP_AUDIT_FOR_08X = False",
        "RUN_07H_PAPER_READY_SYNTHESIS = False",
        "RUN_07I_BACKUP_TO_GOOGLE_DRIVE = False",
        "RUN_07J_WRITE_MONITORING_PLAN = False",
        "BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE = False",
        "OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH = False",
        "OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False",
        "OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS = False",
        "OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH = False",
        "validate_thesis_paragraph_kit",
        "validate_ledger_frame",
        "validate_final_validation_comparison_frame",
        "notebook07_lockfile_scope_gate.json",
        "notebook07_final_validation_comparison.csv",
        "notebook07_validation_budget_ledger.csv",
        "notebook07_thesis_paragraph_kit.json",
        "notebook07_run_manifest.json",
        "notebook07_gap_audit_for_08x.csv",
        "notebook07_post_publication_monitoring_plan.json",
    )
    for needle in required:
        if needle not in source:
            raise AssertionError(f"Missing required notebook source string: {needle}")
    forbidden = (
        "from intraday_research",
        "baseline_helpers",
        "drive.mount(",
        "train_test_split",
        "holdout_test_authorized = True",
        "select_threshold",
        "best_threshold",
        "optimal_threshold",
        "optimal_coverage",
        "select_feature_subset",
        "run_hpo",
        "train_new_model",
        "__file__",
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
            "# Notebook 07 - Validation Synthesis And Gap Audit\n\n"
            "Validation-only synthesis over frozen Notebook 05 (and optional Notebook 06) "
            "artifacts. This notebook signs a lockfile, builds a paper-ready comparison, "
            "appends an append-only validation-budget ledger, runs per-ticker / seed / "
            "concentration robustness, optional explainability and null-control appendices, "
            "emits a gap audit for 08X, composes a paper-ready synthesis + thesis paragraph "
            "kit, and writes a post-publication monitoring plan. It does NOT train a model, "
            "open holdout/test, run HPO, select a new threshold / coverage / feature subset, "
            "or use SHAP / permutation / ECE / AURC / null-control as a selection gate."
        ),
        new_markdown_cell(
            "## 07 Contract Helpers\n\nInline copy of canonical "
            "`src/intraday_research/contracts/validation_synthesis_gap_audit.py` for Colab "
            "portability. Loaded BEFORE config so the config cell can pin constants by name."
        ),
        new_code_cell(contract_source.strip()),
        new_markdown_cell(
            "## Config And Run Switches\n\nScope, output dir, RUN_07* switches "
            "(all default False), operator acknowledgements, locked candidate, "
            "design doc sha pin, and output file paths."
        ),
        new_code_cell(CONFIG_SOURCE.strip()),
        new_markdown_cell(
            "## Runtime Helpers\n\nArtifact writing, sha256 / canonical hashing, "
            "ledger append + flush + prefix invariance, metric helpers, Drive backup, "
            "and lockfile re-entry guard."
        ),
        new_code_cell(RUNTIME_HELPERS_SOURCE.strip()),
        new_markdown_cell(
            "## 07A - Lockfile And Scope Gate\n\n"
            "Verifies N05 (and optional N06) artifacts, design-doc bytes, and the locked "
            "candidate tuple. Signs the lockfile and emits the input artifact manifest. "
            "Every downstream RUN_07* phase re-loads the lockfile and refuses to run if "
            "`contract_passed != True`."
        ),
        new_code_cell(CELL_07A.strip()),
        new_markdown_cell(
            "## 07B - Final Validation-Only Comparison\n\n"
            "Builds one normalized table for full-coverage N05 rows and (optional) selective "
            "N06 fixed-grid rows. Each row carries `row_class` ∈ {full_coverage, selective}; "
            "REQUIRED + CONDITIONAL columns follow canonical "
            "`src/intraday_research/contracts/validation_synthesis_gap_audit.py`."
        ),
        new_code_cell(CELL_07B.strip()),
        new_markdown_cell(
            "## 07C - Validation-Budget Ledger\n\n"
            "Flushes the in-memory append-only ledger to disk. Other phases append rows as "
            "side effects; 07C only writes and validates monotonic non-decreasing cumulative "
            "counter (project-level source of truth per AGENTS.md §4.3)."
        ),
        new_code_cell(CELL_07C.strip()),
        new_markdown_cell(
            "## 07D - Per-Ticker And Seed Robustness\n\n"
            "Reads N05 per-ticker rows and (if present) N06 concentration guardrails. Emits "
            "per-ticker, seed, and concentration tables as diagnostics only."
        ),
        new_code_cell(CELL_07D.strip()),
        new_markdown_cell(
            "## 07E - Explainability Appendix\n\n"
            "Default appendix-only output: LightGBM split + gain importance. SHAP / "
            "`pred_contrib` require explicit operator approval and a refreshed lockfile + "
            "design-doc sha; this default cell does not promote them."
        ),
        new_code_cell(CELL_07E.strip()),
        new_markdown_cell(
            "## 07F - Permutation / Null-Control Appendix\n\n"
            "Runs only when 07A allocated alpha. Tracks `alpha_consumed_after_row` and "
            "`alpha_remaining_after_row` in the ledger. Allowed null designs are read-only "
            "or chronology-aware permutations; outputs are diagnostic-only."
        ),
        new_code_cell(CELL_07F.strip()),
        new_markdown_cell(
            "## 07G - Gap Audit For 08X\n\n"
            "Categorizes gaps and routes each gap to 08X / 08F / 08O without selecting a "
            "new candidate inside 07. Each gap row carries `requires_extra_preregistration` "
            "(boolean) plus a `priority` enum (per design 07G)."
        ),
        new_code_cell(CELL_07G.strip()),
        new_markdown_cell(
            "## 07H - Paper-Ready Synthesis\n\n"
            "Composes results / robustness / limitation paragraphs bound by AGENTS.md "
            "§4.2.5a. Emits `notebook07_thesis_paragraph_kit.json` validated by "
            "`validate_thesis_paragraph_kit`. The kit pins reproducibility pointers per "
            "sentence and records falsification conditions for the 07J monitoring plan."
        ),
        new_code_cell(CELL_07H.strip()),
        new_markdown_cell(
            "## 07I - Optional Drive Backup\n\n"
            "Uses Drive file creation with timestamped names; does not overwrite existing "
            "files. Requires `BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE = True` plus a Drive folder ID."
        ),
        new_code_cell(CELL_07I.strip()),
        new_markdown_cell(
            "## 07J - Post-Publication Monitoring Plan\n\n"
            "Emits the monitoring plan only (no monitoring runs). The plan binds the "
            "monitor to read-only diagnostic operations on same-candidate data and records "
            "the §07H falsification conditions for future cycles."
        ),
        new_code_cell(CELL_07J.strip()),
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
