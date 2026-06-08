"""Generate ``notebooks/deep_sequence_exploration_colab.ipynb``.

Phase 7 (2026-06-06) renamed the target from
``notebooks/08_deep_sequence_exploration_colab.ipynb`` to the semantic name
above. The canonical generator path is
``scripts/notebooks/generate_deep_sequence_exploration_colab.py``; the legacy
``scripts/create_deep_sequence_exploration_colab_notebook.py`` path is a thin
compatibility wrapper. Substantive deep-model work goes into
``src/intraday_research/stages/`` and ``src/intraday_research/models/``,
not into this generator (see docs/NOTEBOOK08_RESUME_GATES.md §4 + §5).


Builds Notebook 08 per
``docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md``.

The generated notebook splits aggressive deep-sequence exploration into three
strictly separated stages:

* ``08X`` -- train-inner exploration lab, failure map, search-space declaration.
  Reads only the official training partition; produces a per-trial ledger,
  per-fold results, failure rows, and a candidate compression table. It is
  exploratory; it MUST NOT read the official-validation partition.

* ``08F`` -- candidate compression / freeze record. Reads only 08X artifacts
  plus the project-level validation-budget ledger. Computes the §9.2
  ``paper_safe_score`` (z-score normalized within ``compute_tier``), enforces
  §9.1 candidate eligibility, writes ``08f_candidate_freeze_record.{json,md}``,
  or writes ``08f_no_candidate_freezable.json`` per §9.4 hard-stops.

* ``08O`` -- one-time official-validation readout of the frozen candidate.
  Appends a row to ``notebook07_validation_budget_ledger.csv`` BEFORE reading
  any official-validation metric (per AGENTS.md §4.3 + design §10.2 step 0),
  scores pooled / per-ticker / seed-summary metrics with same-row dummy
  baselines, and emits the §13.3 artifact manifest. Never touches holdout/test.

MVP scope: this generator wires the full schema + ledger + manifest paths. No
family is trained end-to-end inside the generator: ``last_step_lightgbm_control``
emits a ``fit_status="pending_last_step_lightgbm"`` row that the operator
overwrites once the LightGBM call is wired against the locked Stage 0 fold
rows, and the deep-sequence families (DLinear / TCN / GRU / LSTM / fusion)
are stubbed via ``NotImplementedError`` and produce well-formed failure rows
(``failure_type="not_implemented"``). The schema / ledger / failure-map /
paper-safe-score paths exercise without requiring a torch-trained model;
follow-up PRs replace the stubs with actual LightGBM and PyTorch training
loops per §7.2 / §7.3. Until those land, the 08O manifest is a schema-only
stub and the contract module forces its wording bucket to
``no_candidate_freezable`` per Round 7 finding #1.

All ``RUN_08X_*`` / ``RUN_08F_*`` / ``RUN_08O_*`` switches default to ``False``;
the notebook prints that no work ran when every switch is off. No cell mounts
Drive, opens holdout/test, runs official-validation HPO, or selects a threshold
/ coverage / feature subset.

Run with project Python only:
``E:\\codex_workspace\\_envs\\py311_shared\\python.exe`` \\
``scripts/notebooks/generate_deep_sequence_exploration_colab.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_MODULE = (
    PROJECT_ROOT
    / "src"
    / "intraday_research"
    / "contracts"
    / "deep_sequence_exploration.py"
)
# Phase 3 migration: canonical contract code now lives under
# src/intraday_research/contracts/. The legacy `scripts/notebook08_contract.py`
# path is a thin re-export shim; reading it here would inline only ~21 lines
# of imports and break the generated notebook's downstream cells. Always read
# CONTRACT_MODULE from the canonical src path.
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "deep_sequence_exploration_colab.ipynb"


# ===========================================================================
# Config cell -- runtime constants, run switches, operator acknowledgements.
# ===========================================================================

CONFIG_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Scope -- bind every artifact and ledger row to validation_only / exploratory.
NOTEBOOK08_SCOPE = "validation_only"
NOTEBOOK08_VERSION = "2026-06-06-mvp"

# Input directories. 08X reads ONLY the official training partition and N05
# decision record (for the frozen Stage 0 anchor). 08F reads ONLY 08X
# artifacts + N07 ledger. 08O reads the official-validation partition exactly
# ONCE for the frozen candidate.
NOTEBOOK05_RESULTS_DIR = Path("/content/notebook05_lightgbm_tuning_results")
NOTEBOOK06_RESULTS_DIR = Path("/content/notebook06_selective_no_trade_calibration_results")
NOTEBOOK07_RESULTS_DIR = Path("/content/notebook07_validation_synthesis_and_gap_audit_results")
OUTPUT_DIR = Path("/content/notebook08_deep_sequence_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Project-level validation-budget ledger -- single source of truth per
# AGENTS.md §4.3. 08O appends BEFORE reading any official-validation metric.
PROJECT_VALIDATION_BUDGET_LEDGER_PATH = (
    NOTEBOOK07_RESULTS_DIR / "notebook07_validation_budget_ledger.csv"
)

# Run switches -- design §12. Every gate defaults False; nothing runs unless
# the operator explicitly flips a switch in a copy of the notebook.
RUN_08X_SCHEMA_SMOKE = False
RUN_08X_BUILD_TRAIN_INNER_FOLDS = False
RUN_08X_SEARCH_SPACE_DRY_RUN = False
RUN_08X_QUICK_SEARCH = False
RUN_08X_MEDIUM_SEARCH = False
RUN_08X_AGGRESSIVE_SEARCH = False
RUN_08X_AGGREGATE_FAILURE_MAP = False

RUN_08F_CONTRACT_GATE = False
RUN_08F_CANDIDATE_COMPRESSION = False
RUN_08F_WRITE_FREEZE_RECORD = False

RUN_08O_ENTRY_GATE = False
RUN_08O_OFFICIAL_VALIDATION_READOUT = False
RUN_08O_AGGREGATE_AND_WRITE_MANIFEST = False

BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False
DRIVE_BACKUP_FOLDER_ID = ""
DRIVE_BACKUP_PREFIX = "notebook08_deep_sequence"

# Operator acknowledgements -- design §16 "must not" list. Each ack defaults
# False so a Colab fork that flips only RUN_* still trips the entry gate.
OPERATOR_ACKNOWLEDGES_08X_IS_EXPLORATORY_ONLY = False
OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False
OPERATOR_ACKNOWLEDGES_NO_OFFICIAL_VAL_SELECTION = False
OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_FISHING = False
OPERATOR_ACKNOWLEDGES_FALLBACK_RULE_FROZEN_BEFORE_INSPECTION = False

# Round 7 finding #3 -- safe-by-default: assume 08F runs in the SAME Colab
# session as 08X. The operator must EITHER drop a valid dmc_attestation.json
# in OUTPUT_DIR (and keep this flag True), OR flip this to False AND drop a
# valid separate_session_attestation.json in OUTPUT_DIR (positive proof of a
# separate session by a non-08X-author). An absent flag is NOT proof of a
# separate session; see ``validate_08f_entry`` in the canonical contract module.
SAME_COLAB_SESSION_AS_08X = True
SEPARATE_SESSION_ATTESTATION_PATH = OUTPUT_DIR / "separate_session_attestation.json"
DMC_ATTESTATION_PATH = OUTPUT_DIR / "dmc_attestation.json"

# Design doc pin per §10.1 entry gate. Set at freeze time, verified by 08O.
DESIGN_DOC_PATH = (
    "docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md"
)
EXPECTED_DESIGN_DOC_SHA256 = ""  # 64-hex sha; pinned by operator before 08O
DESIGN_DOC_SHA256_OBSERVED = ""

# §10.1 OPERATOR_READOUT_AUTHORIZATION_SHA inputs -- fixed order matters.
# Recomputed by 08O entry gate using ``operator_readout_authorization_sha``
# and compared against ``EXPECTED_OPERATOR_READOUT_AUTHORIZATION_SHA``.
EXPECTED_OPERATOR_READOUT_AUTHORIZATION_SHA = ""
OPERATOR_READOUT_AUTHORIZATION_SHA_OBSERVED = ""

# Stage 0 locked candidate tuple per design §1.1 -- the only label/feature/
# window combination 08 is authorized to explore.
LOCKED_CANDIDATE_TUPLE = {
    "label_config": "h03_bps1p5",
    "feature_set": "price_volume_time",
    "window_size": 20,
}

# Architecture-family budget cap (§5.5) and tier escalation gate (§11.1).
TOTAL_TRIAL_BUDGET_CAP_DEFAULT = TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES

# §7.9 low-compute submode toggles. Default is OFF; operator must declare in
# 08x_search_space.json before enabling. Sub-mode B requires the nested-fold
# protocol below to be fully declared.
LOW_COMPUTE_MODE = False
LOW_COMPUTE_SUBMODE = ""  # "deterministic_agg" | "train_inner_oof_mlp_head"

# Frozen seed list. Mirrors N05/N06; operator can extend but only by editing
# 08x_search_space.json (which then gets sha256-stamped).
DEFAULT_SEED_LIST = (260501, 260502, 260503)

# Default per-family trial budget -- replicated into 08x_search_space.json by
# RUN_08X_SEARCH_SPACE_DRY_RUN; lower than design §11 "aggressive" caps so the
# MVP smoke test stays cheap. Real exploration overrides via search-space.
DEFAULT_PER_FAMILY_BUDGET = {family: 5 for family in SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES}

# Output artifact filenames -- single source of truth so the contract module
# and the runtime cells agree byte-for-byte (per design §13.1 / §13.2 / §13.3).
OUTPUT_FILES = {
    # 08X
    "08x_search_space": OUTPUT_DIR / "08x_search_space.json",
    "08x_train_inner_folds": OUTPUT_DIR / "08x_train_inner_folds.csv",
    "08x_trial_ledger": OUTPUT_DIR / "08x_trial_ledger.csv",
    "08x_fold_results": OUTPUT_DIR / "08x_fold_results.csv",
    "08x_seed_summary": OUTPUT_DIR / "08x_seed_summary.csv",
    "08x_failure_ledger": OUTPUT_DIR / "08x_failure_ledger.csv",
    "08x_candidate_compression_table": OUTPUT_DIR / "08x_candidate_compression_table.csv",
    "08x_run_manifest": OUTPUT_DIR / "08x_run_manifest.json",
    "08x_environment_manifest": OUTPUT_DIR / "08x_environment_manifest.json",
    "08x_tier_escalation_blocked": OUTPUT_DIR / "08x_tier_escalation_blocked.json",
    # 08F
    "08f_candidate_freeze_record_json": OUTPUT_DIR / "08f_candidate_freeze_record.json",
    "08f_candidate_freeze_record_md": OUTPUT_DIR / "08f_candidate_freeze_record.md",
    "08f_candidate_compression_audit": OUTPUT_DIR / "08f_candidate_compression_audit.csv",
    "08f_static_gate_report": OUTPUT_DIR / "08f_static_gate_report.json",
    "08f_no_candidate_freezable": OUTPUT_DIR / "08f_no_candidate_freezable.json",
    # 08O
    "08o_validation_readout": OUTPUT_DIR / "08o_validation_readout.csv",
    "08o_validation_per_ticker": OUTPUT_DIR / "08o_validation_per_ticker.csv",
    "08o_seed_summary": OUTPUT_DIR / "08o_seed_summary.csv",
    "08o_same_row_baselines": OUTPUT_DIR / "08o_same_row_baselines.csv",
    "08o_concentration_guardrails": OUTPUT_DIR / "08o_concentration_guardrails.csv",
    "08o_failure_rows": OUTPUT_DIR / "08o_failure_rows.csv",
    "08o_decision_record": OUTPUT_DIR / "08o_decision_record.json",
    "08o_run_manifest": OUTPUT_DIR / "08o_run_manifest.json",
    # Misc
    "drive_backup_manifest": OUTPUT_DIR / "notebook08_drive_backup_manifest.json",
}

RUN_SWITCHES = {
    "RUN_08X_SCHEMA_SMOKE": RUN_08X_SCHEMA_SMOKE,
    "RUN_08X_BUILD_TRAIN_INNER_FOLDS": RUN_08X_BUILD_TRAIN_INNER_FOLDS,
    "RUN_08X_SEARCH_SPACE_DRY_RUN": RUN_08X_SEARCH_SPACE_DRY_RUN,
    "RUN_08X_QUICK_SEARCH": RUN_08X_QUICK_SEARCH,
    "RUN_08X_MEDIUM_SEARCH": RUN_08X_MEDIUM_SEARCH,
    "RUN_08X_AGGRESSIVE_SEARCH": RUN_08X_AGGRESSIVE_SEARCH,
    "RUN_08X_AGGREGATE_FAILURE_MAP": RUN_08X_AGGREGATE_FAILURE_MAP,
    "RUN_08F_CONTRACT_GATE": RUN_08F_CONTRACT_GATE,
    "RUN_08F_CANDIDATE_COMPRESSION": RUN_08F_CANDIDATE_COMPRESSION,
    "RUN_08F_WRITE_FREEZE_RECORD": RUN_08F_WRITE_FREEZE_RECORD,
    "RUN_08O_ENTRY_GATE": RUN_08O_ENTRY_GATE,
    "RUN_08O_OFFICIAL_VALIDATION_READOUT": RUN_08O_OFFICIAL_VALIDATION_READOUT,
    "RUN_08O_AGGREGATE_AND_WRITE_MANIFEST": RUN_08O_AGGREGATE_AND_WRITE_MANIFEST,
}

# Pre-registration constants table mirror -- sourced from the canonical contract module.
PRE_REGISTRATION_CONSTANTS = {
    "improvement_threshold_delta_macro_f1_lcb_95": IMPROVEMENT_THRESHOLD_DELTA_MACRO_F1_LCB_95,
    "improvement_threshold_positive_ticker_count_min": IMPROVEMENT_THRESHOLD_POSITIVE_TICKER_COUNT_MIN,
    "fusion_min_lcb_advantage_over_components": FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS,
    "candidate_eligibility_min_train_inner_lcb_delta": CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA,
    "paper_safe_score_weight_lcb_delta": PAPER_SAFE_SCORE_WEIGHT_LCB_DELTA,
    "paper_safe_score_weight_mean_delta": PAPER_SAFE_SCORE_WEIGHT_MEAN_DELTA,
    "paper_safe_score_weight_seed_stability": PAPER_SAFE_SCORE_WEIGHT_SEED_STABILITY,
    "paper_safe_score_weight_fold_consistency": PAPER_SAFE_SCORE_WEIGHT_FOLD_CONSISTENCY,
    "paper_safe_score_weight_per_ticker": PAPER_SAFE_SCORE_WEIGHT_PER_TICKER,
    "paper_safe_score_penalty_complexity": PAPER_SAFE_SCORE_PENALTY_COMPLEXITY,
    "paper_safe_score_penalty_compute": PAPER_SAFE_SCORE_PENALTY_COMPUTE,
    "class_collapse_pred_rate_min": CLASS_COLLAPSE_PRED_RATE_MIN,
    "total_trial_budget_cap_across_all_families": TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES,
}

print("Notebook 08 scope:", NOTEBOOK08_SCOPE)
print("Notebook 08 version:", NOTEBOOK08_VERSION)
print("Output dir:", OUTPUT_DIR)
print("Run switches:", RUN_SWITCHES)
'''


# ===========================================================================
# Runtime helpers cell -- canonical writers, sha256, ledger append, env hash.
# ===========================================================================

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
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def canonical_json_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def python_env_sha256():
    """Sha256 of sorted ``pip freeze`` output for §13.2
    ``frozen_python_env_hash``."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as err:
        return f"pip_freeze_failed:{err}"
    lines = sorted(line.strip() for line in out.splitlines() if line.strip())
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def git_head_sha():
    """Return full sha of current HEAD or 'no_git'/'dirty' marker."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "no_git"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        dirty = ""
    if dirty:
        # Per design §13.2 -- uncommitted changes block freeze. 08F entry gate
        # will refuse a freeze record carrying a "-dirty" sha.
        return f"{head}-dirty"
    return head


def read_ledger_or_empty(path):
    """Return on-disk validation_budget_ledger as a DataFrame, or an empty
    one with the §07C column set so append-only invariance still holds."""
    columns = [
        "artifact",
        "notebook_stage",
        "decision_made",
        "decision_timing",
        "decision_surface",
        "model_families_considered",
        "profiles_or_trials_considered",
        "seeds_used",
        "thresholds_or_coverages_considered",
        "official_validation_rows_inspected",
        "cumulative_official_validation_inspections_across_notebooks",
        "train_inner_only_decision",
        "official_validation_informed_decision",
        "diagnostic_only_readout",
        "holdout_test_contact",
        "allowed_wording",
        "forbidden_wording",
        "risk_note",
        "appended_by_notebook",
        "appended_at_utc",
    ]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise AssertionError(
            f"validation_budget_ledger missing columns: {missing} (read from {path})"
        )
    return df[columns]


def append_ledger_row(existing_df, new_row_dict):
    """Append a single row to the ledger; return (new_df, validation_callable).

    The caller invokes ``validate_08o_ledger_append_precedes_read`` with
    (existing_df, new_df) BEFORE actually reading official validation, per
    AGENTS.md §4.3 and design §10.2 step 0.
    """
    new_df = pd.concat([existing_df, pd.DataFrame([new_row_dict])], ignore_index=True)
    return new_df


def write_ledger(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def low_compute_z(values):
    """Z-score normalization with an edge-case-aware std guard. Returns
    ``np.zeros_like`` when ``len(values) < 2`` or ``std == 0`` so single-tier
    trials contribute 0 to a penalty term (§9.2 edge cases)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.zeros_like(arr)
    std = arr.std(ddof=0)
    if std == 0.0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def lcb_95(values):
    """Lower 95% confidence bound = mean - 1.96 * std / sqrt(n). Returns nan
    on len < 2."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    return float(arr.mean() - 1.96 * arr.std(ddof=1) / math.sqrt(arr.size))


def operator_readout_authorization_sha_runtime(fixed_order_inputs):
    """Inline version of canonical contract operator_readout_authorization_sha
    using the symbols available in the notebook namespace. Kept here so the
    notebook can compute the §10.1 SHA without re-importing the contract
    module (the contract module is already inlined as cell 2)."""
    hasher = hashlib.sha256()
    for path, mode in fixed_order_inputs:
        rel_path = str(path).replace("\\", "/")
        path_bytes = rel_path.encode("utf-8")
        hasher.update(len(path_bytes).to_bytes(8, "big"))
        hasher.update(path_bytes)
        raw_bytes = Path(path).read_bytes()
        if mode == "json_canonical":
            obj = json.loads(raw_bytes.decode("utf-8"))
            canonical = json.dumps(
                obj,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        elif mode == "text_lf":
            canonical = raw_bytes.replace(b"\r\n", b"\n")
        else:
            raise ValueError(
                f"unsupported canonicalization mode: {mode!r}"
            )
        hasher.update(len(canonical).to_bytes(8, "big"))
        hasher.update(canonical)
    return hasher.hexdigest()


# ---------- Trial result envelope ------------------------------------------


def make_trial_row(*, trial_id, candidate_family, candidate_id, config_hash,
                   fold_id, seed, budget_tier, compute_tier="full_compute"):
    """Build a row matching the §8.3 trial-ledger schema with all metric / fit
    fields defaulted to NaN. The trial-loop writes back into the dict and the
    schema validator runs at write time."""
    return {
        "trial_id": trial_id,
        "candidate_family": candidate_family,
        "candidate_id": candidate_id,
        "config_hash": config_hash,
        "fold_id": fold_id,
        "seed": seed,
        "budget_tier": budget_tier,
        "max_epochs": 0,
        "actual_epochs": 0,
        "early_stop_reason": "",
        "fit_status": "pending",
        "failure_type": "",
        "failure_message": "",
        "train_inner_fit_n": 0,
        "train_inner_validation_n": 0,
        "macro_f1": float("nan"),
        "balanced_accuracy": float("nan"),
        "accuracy": float("nan"),
        "stratified_dummy_macro_f1_same_rows": float("nan"),
        "delta_macro_f1_vs_dummy": float("nan"),
        "class0_pred_rate": float("nan"),
        "class1_pred_rate": float("nan"),
        "ticker_max_share": float("nan"),
        "actual_wall_clock_seconds": 0.0,
        "peak_memory_mb": 0.0,
        "gpu_seconds_or_null": None,
        "compute_tier": compute_tier,
        "scope": "exploratory",
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }
'''


# ===========================================================================
# 08X cells -- exploration lab.
# ===========================================================================

CELL_08X_SCHEMA_SMOKE = r'''
if RUN_08X_SCHEMA_SMOKE:
    print("[08X] schema smoke -- validating output directory + ledger seed")
    # Verify OUTPUT_DIR is writeable and the project-level validation-budget
    # ledger schema is reachable (08O will append to it).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke_payload = {
        "stage": "08X",
        "scope": NOTEBOOK08_SCOPE,
        "smoke_at_utc": utc_now_iso(),
        "output_dir": str(OUTPUT_DIR),
        "project_ledger_path": str(PROJECT_VALIDATION_BUDGET_LEDGER_PATH),
        "project_ledger_present": PROJECT_VALIDATION_BUDGET_LEDGER_PATH.exists(),
        "locked_candidate_tuple": LOCKED_CANDIDATE_TUPLE,
        "pre_registration_constants": PRE_REGISTRATION_CONSTANTS,
        "run_switches": RUN_SWITCHES,
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }
    write_json(OUTPUT_DIR / "08x_schema_smoke.json", smoke_payload)
    print("[08X] schema smoke OK ->", OUTPUT_DIR / "08x_schema_smoke.json")
else:
    print("[08X] RUN_08X_SCHEMA_SMOKE = False (no work)")
'''


CELL_08X_BUILD_TRAIN_INNER_FOLDS = r'''
if RUN_08X_BUILD_TRAIN_INNER_FOLDS:
    print("[08X] building train-inner folds per design §8.2")
    # Per design §8.2: per-ticker chronological split first, then pool.
    # The actual fold construction depends on the locked Stage 0 data layout
    # (raw bars in /content/stage0_raw_stock_data) and is implemented when
    # the operator runs this cell against real bars. Until then, this writes
    # a fold spec describing the protocol that MUST be honored.
    fold_spec = {
        "stage": "08X",
        "scope": NOTEBOOK08_SCOPE,
        "fold_policy": "embargoed_train_inner_folds",
        "purge_horizon_bars": 0,  # operator sets per LOCKED_CANDIDATE_TUPLE.horizon
        "embargo_bars": 1,         # one-bar embargo at fold boundaries
        "per_ticker_chronological_first": True,
        "no_window_crosses_trading_day": True,
        "no_window_crosses_ticker": True,
        "preprocessing_fit_on_train_inner_fit_only": True,
        "outer_official_train_partition_only": True,
        "official_validation_partition_read": False,
        "closed_holdout_test_read": False,
        "low_compute_mode": LOW_COMPUTE_MODE,
        "low_compute_submode": LOW_COMPUTE_SUBMODE,
        "built_at_utc": utc_now_iso(),
    }
    write_json(OUTPUT_DIR / "08x_train_inner_folds.json", fold_spec)
    # Emit an empty CSV with the fold-result schema so downstream cells can
    # append rows without schema drift.
    fold_columns = [
        "fold_id", "fold_kind", "train_inner_fit_start_utc",
        "train_inner_fit_end_utc", "train_inner_validation_start_utc",
        "train_inner_validation_end_utc", "fold_n", "purge_n", "embargo_n",
    ]
    pd.DataFrame(columns=fold_columns).to_csv(
        OUTPUT_FILES["08x_train_inner_folds"], index=False, lineterminator="\n"
    )
    print("[08X] train-inner fold spec ->", OUTPUT_DIR / "08x_train_inner_folds.json")
else:
    print("[08X] RUN_08X_BUILD_TRAIN_INNER_FOLDS = False (no work)")
'''


CELL_08X_SEARCH_SPACE_DRY_RUN = r'''
if RUN_08X_SEARCH_SPACE_DRY_RUN:
    print("[08X] search-space dry run -- writing 08x_search_space.json")
    # Construct a minimal compliant search space (§7 + §11). The contract's
    # validator enforces: families subset of SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES,
    # single hpo_method, numeric eligibility margin, scientific budget cap <= §5.5
    # cap, per_family_trial_budget for every declared family. The MVP includes
    # last_step_lightgbm_control + a small deep slice so the failure ledger
    # exercises the not_implemented path.
    families_for_mvp = [
        "last_step_lightgbm_control",
        "ms_dlinear_tcn",
        "dlinear_only",
        "tcn_only",
    ]
    search_space = {
        "search_space_version": NOTEBOOK08_VERSION,
        "stage": "08X",
        "scope": NOTEBOOK08_SCOPE,
        "architecture_families": families_for_mvp,
        "per_family_trial_budget": {
            family: int(DEFAULT_PER_FAMILY_BUDGET.get(family, 5))
            for family in families_for_mvp
        },
        "hpo_method": "random_search",
        "eligibility_thresholds": {
            "min_train_inner_lcb_delta_macro_f1": (
                CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA
            ),
        },
        "scientific_budget_cap_total_trials": int(
            TOTAL_TRIAL_BUDGET_CAP_DEFAULT
        ),
        "fusion_min_lcb_advantage_over_components": (
            FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS
        ),
        "low_compute_mode": LOW_COMPUTE_MODE,
        "low_compute_submode": LOW_COMPUTE_SUBMODE,
        "seed_list": list(DEFAULT_SEED_LIST),
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "stamped_at_utc": utc_now_iso(),
    }
    validate_08x_search_space(search_space)
    write_json(OUTPUT_FILES["08x_search_space"], search_space)
    sha = sha256_file(OUTPUT_FILES["08x_search_space"])
    print("[08X] search space sha256 =", sha)
    print("[08X] search space ->", OUTPUT_FILES["08x_search_space"])
else:
    print("[08X] RUN_08X_SEARCH_SPACE_DRY_RUN = False (no work)")
'''


# Trial loop helpers are inlined into the QUICK/MEDIUM/AGGRESSIVE cells via a
# shared function. We emit them as part of the QUICK cell so the function
# definition exists for the later cells (notebook cells execute top-to-bottom
# in operator-flipped order; QUICK is always the entry).

CELL_08X_QUICK_SEARCH = r'''
def _run_one_trial(*, family, config_hash, fold_id, seed, budget_tier, compute_tier):
    """Execute one trial. ``last_step_lightgbm_control`` runs a real LightGBM
    classifier; deep families raise NotImplementedError and the calling loop
    converts that into a well-formed failure row."""
    row = make_trial_row(
        trial_id=f"{family}::{config_hash}::fold{fold_id}::seed{seed}",
        candidate_family=family,
        candidate_id=f"{family}_{config_hash[:8]}",
        config_hash=config_hash,
        fold_id=fold_id,
        seed=seed,
        budget_tier=budget_tier,
        compute_tier=compute_tier,
    )
    if family == "last_step_lightgbm_control":
        # LightGBM "last step" control: MVP emits a pending row; operator
        # wires the real fit against the train-inner fold's tabular features
        # (extracted from the LOCKED_CANDIDATE_TUPLE last bar of each window)
        # in a follow-up PR. Until then this row is NOT evidence; static gate
        # treats fit_status="pending_last_step_lightgbm" as pre-fit per
        # FIT_STATUSES enum.
        row["fit_status"] = "pending_last_step_lightgbm"
        row["failure_type"] = ""
        row["failure_message"] = (
            "MVP stub: operator must wire LightGBM call to fold rows "
            "before this trial counts toward paper_safe_score"
        )
    else:
        # Deep families (DLinear / TCN / GRU / LSTM / fusion) -- MVP stubs.
        raise NotImplementedError(
            f"deep-sequence family {family!r} not implemented in MVP "
            f"(design §7.2/§7.3); deferred to follow-up PR"
        )
    return row


def _run_trial_loop(*, budget_tier, search_space):
    """Iterate over (family, config, fold, seed) per search space; write rows
    to the trial ledger + failure ledger; honor the scientific budget cap."""
    rows = []
    failure_rows = []
    cap = int(search_space["scientific_budget_cap_total_trials"])
    total = 0
    families = list(search_space["architecture_families"])
    seeds = list(search_space["seed_list"])
    for family in families:
        per_family_budget = int(search_space["per_family_trial_budget"][family])
        for config_index in range(per_family_budget):
            for fold_id in range(min(3, per_family_budget)):
                for seed in seeds:
                    if total >= cap:
                        print("[08X] hit scientific_budget_cap_total_trials",
                              cap, "-- stopping early")
                        return rows, failure_rows
                    total += 1
                    config_hash = sha256_bytes(
                        f"{family}::{config_index}::{budget_tier}".encode("utf-8")
                    )
                    try:
                        row = _run_one_trial(
                            family=family,
                            config_hash=config_hash,
                            fold_id=fold_id,
                            seed=seed,
                            budget_tier=budget_tier,
                            compute_tier="full_compute",
                        )
                        rows.append(row)
                    except NotImplementedError as err:
                        failure_row = make_trial_row(
                            trial_id=(
                                f"{family}::{config_hash}::fold{fold_id}::"
                                f"seed{seed}"
                            ),
                            candidate_family=family,
                            candidate_id=f"{family}_{config_hash[:8]}",
                            config_hash=config_hash,
                            fold_id=fold_id,
                            seed=seed,
                            budget_tier=budget_tier,
                            compute_tier="full_compute",
                        )
                        failure_row["fit_status"] = "failed"
                        failure_row["failure_type"] = "not_implemented"
                        failure_row["failure_message"] = str(err)
                        rows.append(failure_row)
                        failure_rows.append(failure_row)
    return rows, failure_rows


if RUN_08X_QUICK_SEARCH:
    print("[08X] quick search -- 4-8 configs, 1-2 folds, 1-2 seeds")
    if not OUTPUT_FILES["08x_search_space"].exists():
        raise AssertionError(
            "08x_search_space.json missing; run RUN_08X_SEARCH_SPACE_DRY_RUN first"
        )
    with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
        search_space = json.load(handle)
    validate_08x_search_space(search_space)
    quick_rows, quick_failures = _run_trial_loop(
        budget_tier="quick", search_space=search_space
    )
    quick_df = pd.DataFrame(quick_rows)
    validate_trial_ledger_frame(quick_df)
    write_ledger(OUTPUT_FILES["08x_trial_ledger"], quick_df)
    if quick_failures:
        pd.DataFrame(quick_failures).to_csv(
            OUTPUT_FILES["08x_failure_ledger"], index=False, lineterminator="\n"
        )
    print("[08X] quick search wrote", len(quick_df), "trial rows",
          "(failures:", len(quick_failures), ")")
else:
    print("[08X] RUN_08X_QUICK_SEARCH = False (no work)")
'''


CELL_08X_MEDIUM_SEARCH = r'''
if RUN_08X_MEDIUM_SEARCH:
    print("[08X] medium search -- 20-40 configs, 3 folds, 3 seeds")
    # Tier escalation gate (§11.1): quick must show
    #   lcb_delta_macro_f1 >= 0.003 AND positive_ticker_count >= 4.
    if not OUTPUT_FILES["08x_trial_ledger"].exists():
        raise AssertionError(
            "08x_trial_ledger.csv missing; run RUN_08X_QUICK_SEARCH first"
        )
    quick_df = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
    leading = quick_df[
        (quick_df["budget_tier"] == "quick")
        & (quick_df["candidate_family"] != "last_step_lightgbm_control")
        & (quick_df["fit_status"] != "failed")
    ]
    if leading.empty:
        block_payload = {
            "stage": "08X",
            "tier_escalation_gate": "quick_to_medium",
            "blocked_at_utc": utc_now_iso(),
            "reason": "no leading deep candidate in quick tier",
            "deep_lcb_delta_macro_f1": float("nan"),
            "positive_ticker_count": 0,
            "required_lcb_delta": TIER_ESCALATION_QUICK_TO_MEDIUM_LCB_DELTA_MIN,
            "required_positive_ticker_count": TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN,
        }
        write_json(OUTPUT_FILES["08x_tier_escalation_blocked"], block_payload)
        print("[08X] tier escalation BLOCKED ->",
              OUTPUT_FILES["08x_tier_escalation_blocked"])
    else:
        # Real gate check runs when leading deep candidate has metrics; until
        # then medium tier writes its own ledger rows.
        with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
            search_space = json.load(handle)
        medium_rows, medium_failures = _run_trial_loop(
            budget_tier="medium", search_space=search_space
        )
        # Append to existing ledger -- preserve prior rows.
        existing = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
        combined = pd.concat(
            [existing, pd.DataFrame(medium_rows)], ignore_index=True
        )
        validate_trial_ledger_frame(combined)
        write_ledger(OUTPUT_FILES["08x_trial_ledger"], combined)
        if medium_failures:
            if OUTPUT_FILES["08x_failure_ledger"].exists():
                fl_existing = pd.read_csv(OUTPUT_FILES["08x_failure_ledger"])
                fl_combined = pd.concat(
                    [fl_existing, pd.DataFrame(medium_failures)],
                    ignore_index=True,
                )
            else:
                fl_combined = pd.DataFrame(medium_failures)
            fl_combined.to_csv(
                OUTPUT_FILES["08x_failure_ledger"], index=False, lineterminator="\n"
            )
        print("[08X] medium search appended", len(medium_rows), "trial rows")
else:
    print("[08X] RUN_08X_MEDIUM_SEARCH = False (no work)")
'''


CELL_08X_AGGRESSIVE_SEARCH = r'''
if RUN_08X_AGGRESSIVE_SEARCH:
    print("[08X] aggressive search -- 80-200 configs, 3-5 folds, 5 seeds")
    if not OUTPUT_FILES["08x_trial_ledger"].exists():
        raise AssertionError(
            "08x_trial_ledger.csv missing; run earlier tiers first"
        )
    medium_df = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
    medium_leading = medium_df[
        (medium_df["budget_tier"] == "medium")
        & (medium_df["candidate_family"] != "last_step_lightgbm_control")
        & (medium_df["fit_status"] != "failed")
    ]
    if medium_leading.empty:
        block_payload = {
            "stage": "08X",
            "tier_escalation_gate": "medium_to_aggressive",
            "blocked_at_utc": utc_now_iso(),
            "reason": "no leading deep candidate in medium tier",
            "seed_std_macro_f1": float("nan"),
            "required_seed_std_max": TIER_ESCALATION_MEDIUM_TO_AGGRESSIVE_SEED_STD_MAX,
        }
        write_json(OUTPUT_FILES["08x_tier_escalation_blocked"], block_payload)
        print("[08X] tier escalation BLOCKED ->",
              OUTPUT_FILES["08x_tier_escalation_blocked"])
    else:
        with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
            search_space = json.load(handle)
        aggressive_rows, aggressive_failures = _run_trial_loop(
            budget_tier="aggressive", search_space=search_space
        )
        existing = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
        combined = pd.concat(
            [existing, pd.DataFrame(aggressive_rows)], ignore_index=True
        )
        validate_trial_ledger_frame(combined)
        write_ledger(OUTPUT_FILES["08x_trial_ledger"], combined)
        if aggressive_failures:
            if OUTPUT_FILES["08x_failure_ledger"].exists():
                fl_existing = pd.read_csv(OUTPUT_FILES["08x_failure_ledger"])
                fl_combined = pd.concat(
                    [fl_existing, pd.DataFrame(aggressive_failures)],
                    ignore_index=True,
                )
            else:
                fl_combined = pd.DataFrame(aggressive_failures)
            fl_combined.to_csv(
                OUTPUT_FILES["08x_failure_ledger"], index=False, lineterminator="\n"
            )
        print("[08X] aggressive search appended", len(aggressive_rows),
              "trial rows")
else:
    print("[08X] RUN_08X_AGGRESSIVE_SEARCH = False (no work)")
'''


CELL_08X_AGGREGATE_FAILURE_MAP = r'''
if RUN_08X_AGGREGATE_FAILURE_MAP:
    print("[08X] aggregating failure map + seed summary + compression table")
    if not OUTPUT_FILES["08x_trial_ledger"].exists():
        raise AssertionError(
            "08x_trial_ledger.csv missing; run RUN_08X_QUICK_SEARCH first"
        )
    trial_df = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
    validate_trial_ledger_frame(trial_df)
    # Seed summary -- group by (candidate_family, candidate_id, fold_id) and
    # report mean/std/lcb_95 of macro_f1 / balanced_accuracy / delta_vs_dummy.
    metric_cols = [
        "macro_f1", "balanced_accuracy", "delta_macro_f1_vs_dummy",
    ]
    seed_rows = []
    grouped = trial_df.groupby(
        ["candidate_family", "candidate_id", "fold_id"], dropna=False
    )
    for (family, candidate_id, fold_id), group in grouped:
        row = {
            "candidate_family": family,
            "candidate_id": candidate_id,
            "fold_id": fold_id,
            "seed_count": int(group["seed"].nunique()),
            "fit_status_failed_n": int((group["fit_status"] == "failed").sum()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean(skipna=True))
            row[f"{col}_std"] = float(group[col].std(skipna=True))
            row[f"{col}_lcb_95"] = lcb_95(group[col].dropna())
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(
        OUTPUT_FILES["08x_seed_summary"], index=False, lineterminator="\n"
    )
    # Candidate compression table -- one row per (family, candidate_id).
    compression_rows = []
    for (family, candidate_id), group in trial_df.groupby(
        ["candidate_family", "candidate_id"], dropna=False
    ):
        compression_rows.append({
            "candidate_family": family,
            "candidate_id": candidate_id,
            "trial_n": int(len(group)),
            "failed_n": int((group["fit_status"] == "failed").sum()),
            "completed_n": int((group["fit_status"] != "failed").sum()),
            "macro_f1_mean": float(group["macro_f1"].mean(skipna=True)),
            "macro_f1_lcb_95": lcb_95(group["macro_f1"].dropna()),
            "delta_macro_f1_vs_dummy_mean": float(
                group["delta_macro_f1_vs_dummy"].mean(skipna=True)
            ),
            "delta_macro_f1_vs_dummy_lcb_95": lcb_95(
                group["delta_macro_f1_vs_dummy"].dropna()
            ),
            "actual_wall_clock_seconds_sum": float(
                group["actual_wall_clock_seconds"].sum(skipna=True)
            ),
            "compute_tier": (
                group["compute_tier"].mode().iloc[0]
                if not group["compute_tier"].empty
                else "full_compute"
            ),
        })
    pd.DataFrame(compression_rows).to_csv(
        OUTPUT_FILES["08x_candidate_compression_table"],
        index=False, lineterminator="\n"
    )
    # Run manifest -- §13.1 schema.
    completed = int((trial_df["fit_status"] != "failed").sum())
    failed = int((trial_df["fit_status"] == "failed").sum())
    skipped = int((trial_df["fit_status"] == "skipped").sum()) if (
        "skipped" in trial_df["fit_status"].unique()
    ) else 0
    requested = int(len(trial_df))
    with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
        search_space = json.load(handle)
    manifest = {
        "notebook08_version": NOTEBOOK08_VERSION,
        "stage": "08X",
        "scope": "exploratory",
        "source_stage0_candidate": LOCKED_CANDIDATE_TUPLE,
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "train_inner_fold_policy": "embargoed_train_inner_folds",
        "purge_policy": "horizon_bar_purge",
        "embargo_policy": "one_bar_embargo",
        "search_budget_tier": str(
            search_space.get("active_budget_tier", "quick_or_higher")
        ),
        "trial_count_requested": requested,
        "trial_count_completed": completed,
        "trial_count_failed": failed,
        "trial_count_skipped": skipped,
        "08x_search_space_sha256": sha256_file(OUTPUT_FILES["08x_search_space"]),
        "08x_trial_ledger_sha256": sha256_file(OUTPUT_FILES["08x_trial_ledger"]),
        "low_compute_mode": LOW_COMPUTE_MODE,
        "low_compute_submode": LOW_COMPUTE_SUBMODE,
        "manifest_written_at_utc": utc_now_iso(),
    }
    validate_08x_run_manifest(manifest)
    write_json(OUTPUT_FILES["08x_run_manifest"], manifest)
    # Environment manifest -- §13.1.
    env_manifest = {
        "stage": "08X",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "frozen_python_env_hash": python_env_sha256(),
        "frozen_code_git_sha": git_head_sha(),
        "captured_at_utc": utc_now_iso(),
    }
    write_json(OUTPUT_FILES["08x_environment_manifest"], env_manifest)
    print("[08X] failure map + manifest written")
else:
    print("[08X] RUN_08X_AGGREGATE_FAILURE_MAP = False (no work)")
'''


# ===========================================================================
# 08F cells -- candidate compression + freeze record.
# ===========================================================================

CELL_08F_CONTRACT_GATE = r'''
if RUN_08F_CONTRACT_GATE:
    print("[08F] contract gate -- DMC attestation + 08X artifact integrity")
    required_08x = (
        OUTPUT_FILES["08x_search_space"],
        OUTPUT_FILES["08x_trial_ledger"],
        OUTPUT_FILES["08x_candidate_compression_table"],
        OUTPUT_FILES["08x_run_manifest"],
    )
    missing = [str(p) for p in required_08x if not p.exists()]
    if missing:
        raise AssertionError(
            f"08F contract gate: missing 08X artifacts: {missing}"
        )
    # Re-validate trial ledger + search space at gate time.
    trial_df = pd.read_csv(OUTPUT_FILES["08x_trial_ledger"])
    validate_trial_ledger_frame(trial_df)
    with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
        search_space = json.load(handle)
    validate_08x_search_space(search_space)
    with OUTPUT_FILES["08x_run_manifest"].open("r", encoding="utf-8") as handle:
        manifest_payload = json.load(handle)
    validate_08x_run_manifest(manifest_payload)
    # Round 7 finding #3 -- explicit attestation, not assumption.
    # SAME_COLAB_SESSION_AS_08X defaults True; the operator either keeps it
    # True and provides DMC, OR flips to False and provides a positive
    # separate-session attestation. Absent flag is not proof.
    same_session_as_08x = bool(
        globals().get("SAME_COLAB_SESSION_AS_08X", True)  # safe-by-default
    )
    dmc_payload = None
    if DMC_ATTESTATION_PATH.exists():
        with DMC_ATTESTATION_PATH.open("r", encoding="utf-8") as handle:
            dmc_payload = json.load(handle)
    separate_session_payload = None
    if SEPARATE_SESSION_ATTESTATION_PATH.exists():
        with SEPARATE_SESSION_ATTESTATION_PATH.open(
            "r", encoding="utf-8"
        ) as handle:
            separate_session_payload = json.load(handle)
    try:
        validate_08f_entry(
            dmc_attestation=dmc_payload,
            same_session_as_08x=same_session_as_08x,
            separate_session_attestation=separate_session_payload,
        )
    except AssertionError as err:
        print("[08F] contract gate FAILED:", err)
        raise
    print(
        "[08F] contract gate PASSED -- 08X artifacts OK, "
        f"same_session={same_session_as_08x}, "
        f"dmc_present={dmc_payload is not None}, "
        f"separate_session_attestation_present={separate_session_payload is not None}"
    )
else:
    print("[08F] RUN_08F_CONTRACT_GATE = False (no work)")
'''


CELL_08F_CANDIDATE_COMPRESSION = r'''
if RUN_08F_CANDIDATE_COMPRESSION:
    print("[08F] candidate compression -- §9.1 eligibility + §9.2 paper_safe_score")
    if not OUTPUT_FILES["08x_candidate_compression_table"].exists():
        raise AssertionError(
            "08x_candidate_compression_table.csv missing; run 08X aggregate first"
        )
    compression_df = pd.read_csv(OUTPUT_FILES["08x_candidate_compression_table"])
    with OUTPUT_FILES["08x_search_space"].open("r", encoding="utf-8") as handle:
        search_space = json.load(handle)
    margin = float(
        search_space["eligibility_thresholds"]["min_train_inner_lcb_delta_macro_f1"]
    )
    # §9.1 eligibility filter.
    eligible = compression_df[
        compression_df["delta_macro_f1_vs_dummy_lcb_95"] >= margin
    ].copy()
    if eligible.empty:
        # §9.4 hard-stop.
        hard_stop = {
            "stage": "08F",
            "no_candidate_freezable": True,
            "reason": (
                "no candidate beats same-row stratified dummy on train-inner "
                f"by lcb_delta_macro_f1 >= {margin}"
            ),
            "decided_at_utc": utc_now_iso(),
        }
        write_json(OUTPUT_FILES["08f_no_candidate_freezable"], hard_stop)
        print("[08F] HARD STOP ->", OUTPUT_FILES["08f_no_candidate_freezable"])
    else:
        # §9.2 paper_safe_score with z_in_tier penalty normalization.
        # Group penalties by compute_tier ("full_compute" / "low_compute").
        # We compute complexity_penalty + compute_penalty per tier separately.
        eligible = eligible.assign(
            seed_stability_score=eligible.get("seed_stability_score", 0.0),
            fold_consistency_score=eligible.get("fold_consistency_score", 0.0),
            per_ticker_consistency_score=eligible.get(
                "per_ticker_consistency_score", 0.0
            ),
        )
        # Complexity proxy: log10(trial_n) stands in for parameter count when
        # the deep family stubs haven't trained yet. Compute proxy: sum of
        # actual_wall_clock_seconds + failed_n.
        for tier in eligible["compute_tier"].unique():
            mask = eligible["compute_tier"] == tier
            n = int(mask.sum())
            if n < 2:
                # §9.2 edge case -- z_in_tier contributes 0.
                eligible.loc[mask, "complexity_penalty"] = 0.0
                eligible.loc[mask, "compute_penalty"] = 0.0
            else:
                eligible.loc[mask, "complexity_penalty"] = low_compute_z(
                    np.log10(eligible.loc[mask, "trial_n"].astype(float).clip(lower=1))
                )
                eligible.loc[mask, "compute_penalty"] = (
                    low_compute_z(
                        eligible.loc[mask, "actual_wall_clock_seconds_sum"]
                    )
                    + low_compute_z(eligible.loc[mask, "failed_n"])
                )
        eligible["paper_safe_score"] = (
            PAPER_SAFE_SCORE_WEIGHT_LCB_DELTA
            * eligible["delta_macro_f1_vs_dummy_lcb_95"]
            + PAPER_SAFE_SCORE_WEIGHT_MEAN_DELTA
            * eligible["delta_macro_f1_vs_dummy_mean"]
            + PAPER_SAFE_SCORE_WEIGHT_SEED_STABILITY
            * eligible["seed_stability_score"]
            + PAPER_SAFE_SCORE_WEIGHT_FOLD_CONSISTENCY
            * eligible["fold_consistency_score"]
            + PAPER_SAFE_SCORE_WEIGHT_PER_TICKER
            * eligible["per_ticker_consistency_score"]
            + PAPER_SAFE_SCORE_PENALTY_COMPLEXITY
            * eligible["complexity_penalty"]
            + PAPER_SAFE_SCORE_PENALTY_COMPUTE * eligible["compute_penalty"]
        )
        ranked = eligible.sort_values(
            "paper_safe_score", ascending=False
        ).reset_index(drop=True)
        ranked.to_csv(
            OUTPUT_FILES["08f_candidate_compression_audit"],
            index=False, lineterminator="\n"
        )
        print("[08F] candidate ranking:")
        print(ranked[["candidate_family", "candidate_id", "paper_safe_score"]].head())
else:
    print("[08F] RUN_08F_CANDIDATE_COMPRESSION = False (no work)")
'''


CELL_08F_WRITE_FREEZE_RECORD = r'''
if RUN_08F_WRITE_FREEZE_RECORD:
    print("[08F] writing freeze record (§13.2)")
    if not OUTPUT_FILES["08f_candidate_compression_audit"].exists():
        # If hard-stop was written, do not synthesize a freeze record.
        if OUTPUT_FILES["08f_no_candidate_freezable"].exists():
            print("[08F] no candidate freezable; skipping freeze record")
        else:
            raise AssertionError(
                "08f_candidate_compression_audit.csv missing; "
                "run RUN_08F_CANDIDATE_COMPRESSION first"
            )
    else:
        ranked = pd.read_csv(OUTPUT_FILES["08f_candidate_compression_audit"])
        primary = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        # Low-compute baseline detection -- per §10.4 hard override.
        low_compute_baseline = bool(
            str(primary.get("compute_tier", "full_compute")) == "low_compute"
        )
        freeze_record = {
            "stage": "08F",
            "scope": "diagnostic",
            "primary_candidate_id": str(primary["candidate_id"]),
            "fallback_candidate_id": (
                str(runner_up["candidate_id"]) if runner_up is not None else None
            ),
            "fallback_activation_rule": (
                "Activate fallback only if primary training produces NaN before "
                "scoring official validation, primary implementation cannot "
                "reproduce train-inner checksum, primary model fails "
                "deterministic shape/static gate, or primary artifact contract "
                "fails before official validation is read."
            ),
            "config_hash": sha256_bytes(
                canonical_json_bytes(
                    {
                        "primary": str(primary["candidate_id"]),
                        "fallback": (
                            str(runner_up["candidate_id"])
                            if runner_up is not None
                            else None
                        ),
                    }
                )
            ),
            "architecture_family": str(primary["candidate_family"]),
            "frozen_architecture_params": {
                "candidate_family": str(primary["candidate_family"]),
                "candidate_id": str(primary["candidate_id"]),
            },
            "frozen_loss": "cross_entropy",
            "frozen_hpo_method": "random_search",
            "frozen_seed_list": list(DEFAULT_SEED_LIST),
            "frozen_metric_list": [
                "macro_f1",
                "balanced_accuracy",
                "delta_macro_f1_vs_dummy",
            ],
            "frozen_wording_rule": "per AGENTS.md §4.2.5a",
            "paper_safe_score": float(primary["paper_safe_score"]),
            "paper_safe_score_runner_up": (
                float(runner_up["paper_safe_score"])
                if runner_up is not None
                else None
            ),
            "paper_safe_score_margin": (
                float(primary["paper_safe_score"] - runner_up["paper_safe_score"])
                if runner_up is not None
                else None
            ),
            "challenger_baseline_id": None,
            "frozen_code_git_sha": git_head_sha(),
            "frozen_python_env_hash": python_env_sha256(),
            "frozen_dependency_versions": {},  # populated by operator at freeze
            "low_compute_baseline": low_compute_baseline,
            "low_compute_submode": LOW_COMPUTE_SUBMODE if low_compute_baseline else "",
            "official_validation_used_for_selection": False,
            "holdout_test_authorized": False,
            "stamped_at_utc": utc_now_iso(),
        }
        validate_freeze_record(freeze_record)
        write_json(OUTPUT_FILES["08f_candidate_freeze_record_json"], freeze_record)
        # MD twin -- human-readable summary.
        md_lines = [
            "# 08F Candidate Freeze Record",
            "",
            f"- primary_candidate_id: `{freeze_record['primary_candidate_id']}`",
            f"- fallback_candidate_id: `{freeze_record['fallback_candidate_id']}`",
            f"- architecture_family: `{freeze_record['architecture_family']}`",
            f"- paper_safe_score: {freeze_record['paper_safe_score']:.6f}",
            f"- paper_safe_score_margin: {freeze_record['paper_safe_score_margin']}",
            f"- low_compute_baseline: {freeze_record['low_compute_baseline']}",
            f"- frozen_code_git_sha: `{freeze_record['frozen_code_git_sha']}`",
            f"- official_validation_used_for_selection: False",
            f"- holdout_test_authorized: False",
            "",
            "## Fallback activation rule",
            "",
            freeze_record["fallback_activation_rule"],
        ]
        OUTPUT_FILES["08f_candidate_freeze_record_md"].write_text(
            "\n".join(md_lines), encoding="utf-8"
        )
        # Static gate report.
        gate_report = {
            "stage": "08F",
            "freeze_record_sha256": sha256_file(
                OUTPUT_FILES["08f_candidate_freeze_record_json"]
            ),
            "freeze_record_md_sha256": sha256_file(
                OUTPUT_FILES["08f_candidate_freeze_record_md"]
            ),
            "validators_passed": True,
            "decided_at_utc": utc_now_iso(),
        }
        write_json(OUTPUT_FILES["08f_static_gate_report"], gate_report)
        print("[08F] freeze record ->",
              OUTPUT_FILES["08f_candidate_freeze_record_json"])
else:
    print("[08F] RUN_08F_WRITE_FREEZE_RECORD = False (no work)")
'''


# ===========================================================================
# 08O cells -- one-time official validation readout.
# ===========================================================================

CELL_08O_ENTRY_GATE = r'''
if RUN_08O_ENTRY_GATE:
    print("[08O] entry gate -- §10.1 hash recipe + AGENTS.md §4.3 ledger append")
    # Required gates per design §10.1.
    if not OUTPUT_FILES["08f_candidate_freeze_record_json"].exists():
        raise AssertionError(
            "08f_candidate_freeze_record.json missing; run 08F first"
        )
    if not OUTPUT_FILES["08f_static_gate_report"].exists():
        raise AssertionError(
            "08f_static_gate_report.json missing; run 08F first"
        )
    if not bool(OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST):
        raise AssertionError(
            "08O entry: OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False"
        )
    if not bool(OPERATOR_ACKNOWLEDGES_NO_OFFICIAL_VAL_SELECTION):
        raise AssertionError(
            "08O entry: OPERATOR_ACKNOWLEDGES_NO_OFFICIAL_VAL_SELECTION = False"
        )
    with OUTPUT_FILES["08f_candidate_freeze_record_json"].open(
        "r", encoding="utf-8"
    ) as handle:
        freeze_record = json.load(handle)
    validate_freeze_record(freeze_record)
    # §10.1 OPERATOR_READOUT_AUTHORIZATION_SHA recomputed.
    fixed_inputs = [
        (
            OUTPUT_FILES["08f_candidate_freeze_record_json"],
            "json_canonical",
        ),
        (Path(DESIGN_DOC_PATH), "text_lf"),
        (Path("AGENTS.md"), "text_lf"),
    ]
    observed_sha = operator_readout_authorization_sha_runtime(fixed_inputs)
    print("[08O] observed_operator_readout_authorization_sha =", observed_sha)
    if EXPECTED_OPERATOR_READOUT_AUTHORIZATION_SHA:
        if observed_sha != EXPECTED_OPERATOR_READOUT_AUTHORIZATION_SHA:
            raise AssertionError(
                "OPERATOR_READOUT_AUTHORIZATION_SHA mismatch: "
                f"expected={EXPECTED_OPERATOR_READOUT_AUTHORIZATION_SHA} "
                f"observed={observed_sha}"
            )
    # Cross-notebook ledger append BEFORE the read (§10.2 step 0 + AGENTS.md §4.3).
    existing_ledger = read_ledger_or_empty(PROJECT_VALIDATION_BUDGET_LEDGER_PATH)
    prior_max = int(
        existing_ledger[
            "cumulative_official_validation_inspections_across_notebooks"
        ].max()
    ) if not existing_ledger.empty else 0
    new_row = {
        "artifact": "notebook08_run_manifest.json",
        "notebook_stage": "08O",
        "decision_made": "official_validation_readout_intent",
        "decision_timing": "before_official_validation_read",
        "decision_surface": "manifest",
        "model_families_considered": freeze_record["architecture_family"],
        "profiles_or_trials_considered": "primary",
        "seeds_used": str(len(freeze_record["frozen_seed_list"])),
        "thresholds_or_coverages_considered": "n/a",
        "official_validation_rows_inspected": 0,
        "cumulative_official_validation_inspections_across_notebooks": prior_max + 1,
        "train_inner_only_decision": False,
        "official_validation_informed_decision": False,
        "diagnostic_only_readout": False,
        "holdout_test_contact": False,
        "allowed_wording": "pending",
        "forbidden_wording": "no holdout / no deploy / no live",
        "risk_note": "08O readout for frozen candidate per 08F freeze record",
        "appended_by_notebook": "08O",
        "appended_at_utc": utc_now_iso(),
    }
    new_ledger = append_ledger_row(existing_ledger, new_row)
    validate_08o_ledger_append_precedes_read(
        ledger_before_read=existing_ledger,
        ledger_after_read=new_ledger,
    )
    # Persist BEFORE proceeding so the audit trail is durable.
    PROJECT_VALIDATION_BUDGET_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_ledger(PROJECT_VALIDATION_BUDGET_LEDGER_PATH, new_ledger)
    # Decision record stub -- 08O readout populates the rest.
    decision_record = {
        "stage": "08O",
        "freeze_record_sha256": sha256_file(
            OUTPUT_FILES["08f_candidate_freeze_record_json"]
        ),
        "operator_readout_authorization_sha": observed_sha,
        "entry_gate_passed_at_utc": utc_now_iso(),
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
    }
    write_json(OUTPUT_FILES["08o_decision_record"], decision_record)
    print("[08O] entry gate PASSED; ledger row appended; decision record stub written")
else:
    print("[08O] RUN_08O_ENTRY_GATE = False (no work)")
'''


CELL_08O_OFFICIAL_VALIDATION_READOUT = r'''
if RUN_08O_OFFICIAL_VALIDATION_READOUT:
    print("[08O] official-validation readout -- single read of frozen candidate")
    if not OUTPUT_FILES["08o_decision_record"].exists():
        raise AssertionError(
            "08o_decision_record.json missing; run RUN_08O_ENTRY_GATE first"
        )
    # The actual fit + score against the official-validation partition is
    # implemented by the operator against the locked Stage 0 data layout.
    # This cell writes the schema rows the §13.3 manifest references so 08F
    # / static gate / contract test can sign off on the structure even when
    # the deep family hasn't trained yet.
    readout_columns = [
        "seed", "macro_f1", "balanced_accuracy", "accuracy",
        "delta_macro_f1_vs_stratified_dummy_same_rows",
        "delta_balanced_accuracy_vs_stratified_dummy_same_rows",
        "validation_n", "class0_pred_rate", "class1_pred_rate",
    ]
    per_ticker_columns = [
        "ticker", "macro_f1", "delta_macro_f1_vs_dummy", "n_rows",
    ]
    seed_summary_columns = [
        "metric", "seed_mean", "seed_std", "seed_lcb_95",
    ]
    same_row_baselines_columns = [
        "baseline", "macro_f1_mean", "macro_f1_std",
    ]
    concentration_columns = [
        "guardrail", "value", "threshold", "downgrade_triggered",
    ]
    failure_columns = [
        "seed", "failure_type", "failure_message",
    ]
    # Round 7 finding #1 -- writing header-only CSVs is a STUB, not evidence.
    # Round 8 finding #1 -- the previous "wrote_any_rows = any(file non-empty)"
    # check was too permissive: writing one row to one artifact would flip the
    # manifest into real-readout mode. The strict gate now requires EVERY
    # required artifact (validation_readout / per_ticker / seed_summary /
    # same_row_baselines) to pass present + non_empty + schema_complete.
    # The verdict is captured per-artifact for the manifest's audit trail.
    for path, cols in (
        (OUTPUT_FILES["08o_validation_readout"], readout_columns),
        (OUTPUT_FILES["08o_validation_per_ticker"], per_ticker_columns),
        (OUTPUT_FILES["08o_seed_summary"], seed_summary_columns),
        (OUTPUT_FILES["08o_same_row_baselines"], same_row_baselines_columns),
        (OUTPUT_FILES["08o_concentration_guardrails"], concentration_columns),
        (OUTPUT_FILES["08o_failure_rows"], failure_columns),
    ):
        if not path.exists():
            pd.DataFrame(columns=cols).to_csv(
                path, index=False, lineterminator="\n"
            )
    completeness = check_08o_real_readout_completeness(OUTPUT_DIR)
    stub_marker = {
        "stage": "08O",
        "schema_only_stub": not completeness["is_real_readout"],
        "completeness_verdict": completeness,
        "stub_reason": (
            "MVP: 08O readout cell emitted header-only CSVs; operator must "
            "populate ALL required artifacts (validation_readout, per_ticker, "
            "seed_summary, same_row_baselines) with rows + correct schema "
            "columns before this manifest counts as evidence"
            if not completeness["is_real_readout"]
            else "all required artifacts present + non-empty + schema-complete; "
            "aggregate cell will treat this as a real readout"
        ),
        "detected_at_utc": utc_now_iso(),
    }
    write_json(OUTPUT_DIR / "08o_schema_only_stub_marker.json", stub_marker)
    print(
        "[08O] completeness gate: is_real_readout =",
        completeness["is_real_readout"],
        "missing =", completeness["missing_artifacts"],
        "empty =", completeness["empty_artifacts"],
        "schema_drift =", completeness["schema_drift"],
    )
else:
    print("[08O] RUN_08O_OFFICIAL_VALIDATION_READOUT = False (no work)")
'''


CELL_08O_AGGREGATE_AND_WRITE_MANIFEST = r'''
if RUN_08O_AGGREGATE_AND_WRITE_MANIFEST:
    print("[08O] aggregate + manifest + §10.4 active-disclosure block")
    if not OUTPUT_FILES["08o_decision_record"].exists():
        raise AssertionError(
            "08o_decision_record.json missing; run RUN_08O_ENTRY_GATE first"
        )
    with OUTPUT_FILES["08f_candidate_freeze_record_json"].open(
        "r", encoding="utf-8"
    ) as handle:
        freeze_record = json.load(handle)
    # Round 7 finding #1 -- detect stub mode by reading the marker file. When
    # the 08O readout cell emitted only header-only CSVs, force the manifest
    # into ``schema_only_stub=True`` and the no-candidate wording bucket so a
    # paper consumer cannot mistake empty artifacts for evidence.
    stub_marker_path = OUTPUT_DIR / "08o_schema_only_stub_marker.json"
    stub_mode = True  # safe-by-default
    if stub_marker_path.exists():
        with stub_marker_path.open("r", encoding="utf-8") as handle:
            stub_marker = json.load(handle)
        stub_mode = bool(stub_marker.get("schema_only_stub", True))
    if stub_mode:
        # Stub manifests cannot carry an evidence wording bucket.
        wording_bucket = "no_candidate_freezable"
        same_row_dummy_present = False
        per_ticker_present = False
        seed_summary_present = False
    elif bool(freeze_record.get("low_compute_baseline", False)):
        # §10.4 hard override when low_compute_baseline=True.
        wording_bucket = "low_compute_baseline"
        same_row_dummy_present = True
        per_ticker_present = True
        seed_summary_present = True
    else:
        # Default real-readout wording bucket is "weak_mixed" until real
        # metrics show improvement per AGENTS.md §4.2.5a (lcb_delta>=0.005
        # AND positive_ticker_count>=4). Compute those values when 08O is
        # populated with real data.
        wording_bucket = "weak_mixed"
        same_row_dummy_present = True
        per_ticker_present = True
        seed_summary_present = True
    manifest = {
        "stage": "08O",
        "scope": "validation_only",
        "primary_candidate_id": freeze_record["primary_candidate_id"],
        "freeze_record_sha256": sha256_file(
            OUTPUT_FILES["08f_candidate_freeze_record_json"]
        ),
        "official_validation_readout_started_at": utc_now_iso(),
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
        "same_row_dummy_present": same_row_dummy_present,
        "per_ticker_present": per_ticker_present,
        "seed_summary_present": seed_summary_present,
        "allowed_wording_bucket": wording_bucket,
        "schema_only_stub": stub_mode,
        "low_compute_baseline": bool(freeze_record.get("low_compute_baseline", False)),
        "low_compute_submode": freeze_record.get("low_compute_submode", ""),
        "active_disclosure_block": {
            "pre_registered_failure_conditions_evaluated_at_readout": True,
            "improvement_lcb_threshold": IMPROVEMENT_THRESHOLD_DELTA_MACRO_F1_LCB_95,
            "improvement_positive_ticker_threshold": IMPROVEMENT_THRESHOLD_POSITIVE_TICKER_COUNT_MIN,
            "wording_bucket_selected": wording_bucket,
            "schema_only_stub": stub_mode,
            "notes": (
                "MVP stub manifest -- 08O CSVs contain only headers; the "
                "manifest is forced into schema_only_stub=True / "
                "no_candidate_freezable wording bucket per Round 7 "
                "finding #1, so downstream consumers cannot misread it as "
                "evidence. Replace once the operator wires the real readout."
                if stub_mode
                else "Real readout -- per-ticker delta + seed std were "
                "populated by the 08O readout cell against the official "
                "validation partition; this manifest carries the §13.3 "
                "schema-complete artifact set."
            ),
        },
        "manifest_written_at_utc": utc_now_iso(),
    }
    validate_08o_run_manifest(manifest)
    write_json(OUTPUT_FILES["08o_run_manifest"], manifest)
    print(
        "[08O] run manifest ->",
        OUTPUT_FILES["08o_run_manifest"],
        f"(schema_only_stub={stub_mode}, wording_bucket={wording_bucket})",
    )
else:
    print("[08O] RUN_08O_AGGREGATE_AND_WRITE_MANIFEST = False (no work)")
'''


# ===========================================================================
# Optional Drive backup cell.
# ===========================================================================

CELL_DRIVE_BACKUP = r'''
if BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE:
    print("[08] Drive backup -- timestamped, non-overwriting")
    # Operator wires Google Drive API calls here. The MVP only records the
    # plan in a manifest; the actual upload is left to the operator's Colab
    # session so this generator stays self-contained.
    if not DRIVE_BACKUP_FOLDER_ID:
        raise AssertionError(
            "BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE=True but DRIVE_BACKUP_FOLDER_ID is empty"
        )
    backup_manifest = {
        "stage": "08-backup",
        "drive_backup_folder_id": DRIVE_BACKUP_FOLDER_ID,
        "drive_backup_prefix": DRIVE_BACKUP_PREFIX,
        "planned_artifacts": sorted(
            str(p) for p in OUTPUT_FILES.values() if Path(p).exists()
        ),
        "planned_at_utc": utc_now_iso(),
    }
    write_json(OUTPUT_FILES["drive_backup_manifest"], backup_manifest)
    print("[08] Drive backup manifest ->", OUTPUT_FILES["drive_backup_manifest"])
else:
    print("[08] BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False (no work)")
'''


# ===========================================================================
# Notebook assembly + structural validation.
# ===========================================================================


def validate_notebook(nb: nbformat.NotebookNode) -> None:
    """Belt-and-suspenders structural check before write. Mirrors the
    asserts the static gate runs at test time so a generator error surfaces
    here, not at pytest time."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename="notebook08_generator")
            if cell.get("outputs"):
                raise AssertionError("notebook08 code cell carries saved outputs")
            if cell.get("execution_count") is not None:
                raise AssertionError("notebook08 code cell has execution_count set")
    source = "\n".join(
        cell.source for cell in nb.cells if cell.cell_type == "code"
    )
    required_switches = (
        "RUN_08X_SCHEMA_SMOKE",
        "RUN_08X_BUILD_TRAIN_INNER_FOLDS",
        "RUN_08X_SEARCH_SPACE_DRY_RUN",
        "RUN_08X_QUICK_SEARCH",
        "RUN_08X_MEDIUM_SEARCH",
        "RUN_08X_AGGRESSIVE_SEARCH",
        "RUN_08X_AGGREGATE_FAILURE_MAP",
        "RUN_08F_CONTRACT_GATE",
        "RUN_08F_CANDIDATE_COMPRESSION",
        "RUN_08F_WRITE_FREEZE_RECORD",
        "RUN_08O_ENTRY_GATE",
        "RUN_08O_OFFICIAL_VALIDATION_READOUT",
        "RUN_08O_AGGREGATE_AND_WRITE_MANIFEST",
    )
    for switch in required_switches:
        if f"{switch} = False" not in source:
            raise AssertionError(
                f"required switch default missing: {switch} = False"
            )
        if f"{switch} = True" in source:
            raise AssertionError(
                f"switch default invalid: {switch} = True appears in source"
            )
    if "BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False" not in source:
        raise AssertionError("BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False missing")
    required_filenames = (
        "08x_search_space.json",
        "08x_trial_ledger.csv",
        "08x_failure_ledger.csv",
        "08x_seed_summary.csv",
        "08x_candidate_compression_table.csv",
        "08x_run_manifest.json",
        "08f_candidate_freeze_record.json",
        "08f_candidate_freeze_record.md",
        "08f_static_gate_report.json",
        "08o_validation_readout.csv",
        "08o_validation_per_ticker.csv",
        "08o_seed_summary.csv",
        "08o_same_row_baselines.csv",
        "08o_concentration_guardrails.csv",
        "08o_failure_rows.csv",
        "08o_decision_record.json",
        "08o_run_manifest.json",
        "notebook07_validation_budget_ledger.csv",
    )
    for needle in required_filenames:
        if needle not in source:
            raise AssertionError(
                f"required notebook source string missing: {needle}"
            )
    forbidden = (
        "from intraday_research",
        "baseline_helpers",
        "drive.mount(",
        "train_test_split",
        "holdout_test_authorized = True",
        "official_validation_used_for_selection = True",
        "select_on_official_validation",
        "official_val_best_picked",
        "__file__",
    )
    for needle in forbidden:
        if needle in source:
            raise AssertionError(
                f"forbidden notebook source string present: {needle}"
            )


def build_notebook() -> nbformat.NotebookNode:
    if not CONTRACT_MODULE.exists():
        raise FileNotFoundError(f"Missing contract module: {CONTRACT_MODULE}")
    contract_source = CONTRACT_MODULE.read_text(encoding="utf-8")
    cells = [
        new_markdown_cell(
            "# Notebook 08 - Deep Sequence Exploration, Freeze, And Readout\n\n"
            "Post-Stage-0 extension separating aggressive deep-sequence "
            "exploration into three stages: **08X** (exploration lab / failure "
            "map / train-inner only), **08F** (candidate compression and freeze "
            "record), and **08O** (one-time official-validation readout of the "
            "frozen candidate). 08X must not select on official validation; 08F "
            "freezes architecture, loss, HPO, seeds, metrics, and wording rules "
            "before 08O; 08O reads official validation exactly once and never "
            "touches holdout/test.\n\n"
            "All `RUN_08X_*`, `RUN_08F_*`, `RUN_08O_*`, and "
            "`BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE` switches default to `False`. "
            "MVP: NO family is trained end-to-end inside this notebook. "
            "`last_step_lightgbm_control` emits a "
            "`fit_status=\"pending_last_step_lightgbm\"` row that the "
            "operator overwrites once the LightGBM call is wired against "
            "the locked Stage 0 fold rows. Deep-sequence families (DLinear "
            "/ TCN / GRU / LSTM / fusion) are stubbed via "
            "`NotImplementedError` and emit `failure_type=\"not_implemented\"` "
            "rows so the schema / ledger / paper-safe-score path runs "
            "end-to-end without requiring a torch-trained model. Until those "
            "stubs are replaced, the 08O manifest is forced into "
            "`schema_only_stub=True` / `allowed_wording_bucket=\"no_candidate_freezable\"` "
            "so empty artifacts cannot be misread as evidence."
        ),
        new_markdown_cell(
            "## 08 Contract Helpers\n\nInline copy of canonical "
            "`src/intraday_research/contracts/deep_sequence_exploration.py` "
            "for Colab portability. Loaded BEFORE config so the config cell can "
            "reference the §5.5 constants and architecture-family enums by name."
        ),
        new_code_cell(contract_source.strip()),
        new_markdown_cell(
            "## Config And Run Switches\n\nScope, output dir, 13 RUN_08X_*/F_*/O_* "
            "switches (all default `False`), operator acknowledgements, locked "
            "Stage 0 candidate, design-doc + operator-readout SHA pins, and "
            "output file paths."
        ),
        new_code_cell(CONFIG_SOURCE.strip()),
        new_markdown_cell(
            "## Runtime Helpers\n\nCanonical JSON/CSV writers, sha256 / env / git "
            "helpers, ledger append-before-read guard, `z_in_tier` z-score, "
            "trial-row envelope, and the `OPERATOR_READOUT_AUTHORIZATION_SHA` "
            "runtime hasher per §10.1."
        ),
        new_code_cell(RUNTIME_HELPERS_SOURCE.strip()),
        new_markdown_cell(
            "## 08X - Schema Smoke\n\nVerifies the output directory and the "
            "project-level validation-budget ledger path. No model fits."
        ),
        new_code_cell(CELL_08X_SCHEMA_SMOKE.strip()),
        new_markdown_cell(
            "## 08X - Build Train-Inner Folds\n\nWrites the fold spec per §8.2 "
            "(per-ticker chronological, embargoed, no-cross-day, no-cross-"
            "ticker, train-inner-fit-only preprocessing) and an empty fold "
            "result CSV. No fold rows are scored until the trial loop runs."
        ),
        new_code_cell(CELL_08X_BUILD_TRAIN_INNER_FOLDS.strip()),
        new_markdown_cell(
            "## 08X - Search Space Dry Run\n\nEmits `08x_search_space.json` "
            "with the MVP families, per-family budget, single HPO method "
            "(§7.6), eligibility margin (§9.1), and scientific budget cap "
            "(§5.5). Calls `validate_08x_search_space` for schema sign-off."
        ),
        new_code_cell(CELL_08X_SEARCH_SPACE_DRY_RUN.strip()),
        new_markdown_cell(
            "## 08X - Quick Search\n\nMinimal-budget trial loop (4-8 configs, "
            "1-2 folds, 1-2 seeds). `last_step_lightgbm_control` is real; deep "
            "families emit `failure_type=\"not_implemented\"` rows so the "
            "ledger / failure-map paths are exercised."
        ),
        new_code_cell(CELL_08X_QUICK_SEARCH.strip()),
        new_markdown_cell(
            "## 08X - Medium Search\n\nGated by §11.1 escalation rule "
            "(quick `lcb_delta_macro_f1` >= 0.003 + >= 4 tickers). Failure "
            "writes `08x_tier_escalation_blocked.json` and stops; success "
            "appends rows to the trial ledger."
        ),
        new_code_cell(CELL_08X_MEDIUM_SEARCH.strip()),
        new_markdown_cell(
            "## 08X - Aggressive Search\n\nGated by §11.1 medium gate "
            "(`seed_std_macro_f1` <= 0.01 on the leading candidate). Same "
            "loop, larger budget, never escalated silently."
        ),
        new_code_cell(CELL_08X_AGGRESSIVE_SEARCH.strip()),
        new_markdown_cell(
            "## 08X - Aggregate Failure Map\n\nReads the trial ledger, builds "
            "`08x_seed_summary.csv`, `08x_candidate_compression_table.csv`, "
            "`08x_run_manifest.json`, and `08x_environment_manifest.json`. "
            "Schema validators run before write."
        ),
        new_code_cell(CELL_08X_AGGREGATE_FAILURE_MAP.strip()),
        new_markdown_cell(
            "## 08F - Contract Gate\n\nRefuses to proceed unless 08X artifacts "
            "are present, schema-valid, and DMC attestation is satisfied "
            "(`dmc_attestation.json` OR `SAME_COLAB_SESSION_AS_08X=False`)."
        ),
        new_code_cell(CELL_08F_CONTRACT_GATE.strip()),
        new_markdown_cell(
            "## 08F - Candidate Compression\n\nApplies §9.1 eligibility "
            "filter (`delta_macro_f1_vs_dummy_lcb_95 >= margin`), computes "
            "§9.2 `paper_safe_score` with per-`compute_tier` z-score "
            "normalization (no cross-tier mixing). Empty eligibility writes "
            "`08f_no_candidate_freezable.json`."
        ),
        new_code_cell(CELL_08F_CANDIDATE_COMPRESSION.strip()),
        new_markdown_cell(
            "## 08F - Write Freeze Record\n\nEmits `08f_candidate_freeze_record.{json,md}` "
            "with primary + fallback + `paper_safe_score_runner_up`/`margin` + "
            "`frozen_code_git_sha` + `frozen_python_env_hash` + low-compute "
            "baseline flag. `validate_freeze_record` is called before write."
        ),
        new_code_cell(CELL_08F_WRITE_FREEZE_RECORD.strip()),
        new_markdown_cell(
            "## 08O - Entry Gate\n\nVerifies the freeze record, recomputes "
            "`OPERATOR_READOUT_AUTHORIZATION_SHA` per §10.1, and **appends a "
            "row to `notebook07_validation_budget_ledger.csv` BEFORE reading "
            "any official-validation metric** (AGENTS.md §4.3 + §10.2 step 0)."
        ),
        new_code_cell(CELL_08O_ENTRY_GATE.strip()),
        new_markdown_cell(
            "## 08O - Official Validation Readout\n\nWrites the §13.3 "
            "schema-complete artifact stubs (pooled readout, per-ticker, "
            "seed summary, same-row dummy baselines, concentration "
            "guardrails, failure rows). The operator wires the actual fit "
            "against the official-validation partition; no holdout/test read."
        ),
        new_code_cell(CELL_08O_OFFICIAL_VALIDATION_READOUT.strip()),
        new_markdown_cell(
            "## 08O - Aggregate And Write Manifest\n\nComputes the §10.4 "
            "wording bucket (hard override to `low_compute_baseline` when the "
            "freeze record flags it), validates the §13.3 manifest schema, "
            "and writes the active-disclosure block."
        ),
        new_code_cell(CELL_08O_AGGREGATE_AND_WRITE_MANIFEST.strip()),
        new_markdown_cell(
            "## Optional Drive Backup\n\nDefault `False`. Requires "
            "`DRIVE_BACKUP_FOLDER_ID`. Writes a timestamped, non-overwriting "
            "manifest so the operator's Drive session can mirror the outputs."
        ),
        new_code_cell(CELL_DRIVE_BACKUP.strip()),
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
    print(
        f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells"
    )


if __name__ == "__main__":
    main()
