"""Artifact contract helpers for Notebook 08 outputs.

Lifts the inline validators previously embedded in
``tests/test_notebook08_artifact_contract.py`` into a reusable module so the
N08 colab notebook generator can inline the same logic for Colab portability
(same pattern as ``scripts/notebook07_contract.py``).

The validators here mirror the design rules in
``docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md``:

* §5.5 Pre-registration Constants Table -- 13 frozen numeric thresholds.
* §7.1 architecture families enum.
* §7.9 low-compute submode enum + sub-mode B nested-fold protocol guard.
* §8.3 trial ledger schema and ``compute_tier`` enum.
* §9.1 candidate eligibility margin (numeric).
* §9.2 ``paper_safe_score`` weights/penalties.
* §9.3 fallback activation rule -- forbid official-validation-metric coupling.
* §10.1 ``OPERATOR_READOUT_AUTHORIZATION_SHA`` canonical-bytes recipe.
* §10.2 step 0 -- 08O ledger append BEFORE official-validation read.
* §13.2 freeze record schema.
* §13.3 08O manifest schema.

These helpers are validation-only utilities; they do not load, transform, or
score holdout/test data, do not import project helper packages from earlier
notebooks, and do not mount Drive.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


NOTEBOOK08_SCOPE = "validation_only"


# ---------- §5.5 Pre-registration Constants Table ---------------------------
# Any change here MUST be accompanied by a new freeze document and a new
# 08x_search_space.json sha256 stamp (per design §5.5).

IMPROVEMENT_THRESHOLD_DELTA_MACRO_F1_LCB_95 = 0.005
IMPROVEMENT_THRESHOLD_POSITIVE_TICKER_COUNT_MIN = 4
FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS = 0.003
CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA = 0.005
PAPER_SAFE_SCORE_WEIGHT_LCB_DELTA = 0.35
PAPER_SAFE_SCORE_WEIGHT_MEAN_DELTA = 0.20
PAPER_SAFE_SCORE_WEIGHT_SEED_STABILITY = 0.15
PAPER_SAFE_SCORE_WEIGHT_FOLD_CONSISTENCY = 0.10
PAPER_SAFE_SCORE_WEIGHT_PER_TICKER = 0.10
PAPER_SAFE_SCORE_PENALTY_COMPLEXITY = -0.05
PAPER_SAFE_SCORE_PENALTY_COMPUTE = -0.05
CLASS_COLLAPSE_PRED_RATE_MIN = 0.05
TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES = 250

# Tier escalation thresholds (§11.1) -- quick to medium gate value.
TIER_ESCALATION_QUICK_TO_MEDIUM_LCB_DELTA_MIN = 0.003
TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN = 4
TIER_ESCALATION_MEDIUM_TO_AGGRESSIVE_SEED_STD_MAX = 0.01

# §9.2 fallback scale when N06 seed-std artifact is unavailable.
SEED_STABILITY_SCALE_FALLBACK = 0.02


# ---------- §7.1 Architecture Families --------------------------------------

ARCHITECTURE_FAMILIES = (
    "ms_dlinear_tcn",
    "dlinear_only",
    "tcn_only",
    "shallow_gru",
    "shallow_lstm",
    "last_step_mlp_sequence_ablation",
    "last_step_lightgbm_control",
)

# Families currently allowed inside an 08X search-space payload. GRU/LSTM remain
# section-7.1 candidate families, but they are withheld from 08X until their
# axis blocks are frozen in config/search-space and sha-stamped before trial 0.
SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES = (
    "ms_dlinear_tcn",
    "dlinear_only",
    "tcn_only",
    "last_step_mlp_sequence_ablation",
    "last_step_lightgbm_control",
)


# ---------- §7.6 HPO Methods + §7.9 Low-Compute -----------------------------

HPO_METHODS = (
    "random_search",
    "tpe",
    "successive_halving",
    "asha",
    "hyperband",
    "bohb",
)

LOW_COMPUTE_SUBMODES = ("deterministic_agg", "train_inner_oof_mlp_head")

# Sub-mode B nested-fold protocol -- required search-space fields (§7.9).
LOW_COMPUTE_SUBMODE_B_REQUIRED_FIELDS = (
    "outer_fold_scheme",
    "outer_fold_k",
    "inner_fold_k_for_head",
    "head_train_data_source",
    "head_eval_data_source",
)
LOW_COMPUTE_SUBMODE_B_ALLOWED_OUTER_FOLD_SCHEMES = (
    "rolling_origin_folds",
    "purged_time_series_folds",
    "embargoed_train_inner_folds",
)
LOW_COMPUTE_SUBMODE_B_MIN_OUTER_FOLD_K = 5
LOW_COMPUTE_SUBMODE_B_MIN_INNER_FOLD_K = 5


# ---------- §8.3 Trial Ledger ----------------------------------------------

COMPUTE_TIER_VALUES = ("full_compute", "low_compute")

REQUIRED_TRIAL_LEDGER_COLUMNS = {
    "trial_id",
    "candidate_family",
    "candidate_id",
    "config_hash",
    "fold_id",
    "seed",
    "budget_tier",
    "max_epochs",
    "actual_epochs",
    "early_stop_reason",
    "fit_status",
    "failure_type",
    "failure_message",
    "train_inner_fit_n",
    "train_inner_validation_n",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "stratified_dummy_macro_f1_same_rows",
    "delta_macro_f1_vs_dummy",
    "class0_pred_rate",
    "class1_pred_rate",
    "ticker_max_share",
    "actual_wall_clock_seconds",
    "peak_memory_mb",
    "gpu_seconds_or_null",
    "compute_tier",
    "scope",
    "official_validation_used",
    "holdout_test_authorized",
}


# §8.3 fit_status enum -- consumed by validate_trial_ledger_frame.
# "pending" / "pending_last_step_lightgbm" are pre-fit MVP placeholders that
# the operator overwrites with one of the post-fit statuses before treating a
# row as evidence; "completed" and "failed" are the post-fit terminal states;
# "skipped" covers trials a tier-escalation gate refuses without running.
FIT_STATUSES = (
    "pending",
    "pending_last_step_lightgbm",
    "completed",
    "failed",
    "skipped",
)


# §8.4 failure-map enum (used by 08X failure-row writer).
FAILURE_TYPES = (
    "class_collapse",
    "unstable_seed_variance",
    "ticker_concentration",
    "date_concentration",
    "time_of_day_concentration",
    "training_divergence",
    "timeout",
    "memory_error",
    "artifact_schema_failure",
    "official_validation_boundary_violation",
    "feature_window_leak_detected",
    "insufficient_same_row_dummy",
    "no_improvement_over_last_step_control",
    # MVP-only placeholder; deep families fit via NotImplementedError -> ledger row.
    "not_implemented",
)


# ---------- DMC Attestation (§9.1) ------------------------------------------

REQUIRED_DMC_FIELDS = {
    "dmc_role",
    "reviewer_identifier",
    "reviewed_08x_run_manifest_sha256",
    "reviewed_at_utc",
    "attestation_statement",
}


# ---------- Freeze Record (§13.2) -------------------------------------------

REQUIRED_FREEZE_RECORD_FIELDS = {
    "stage",
    "scope",
    "primary_candidate_id",
    "fallback_candidate_id",
    "fallback_activation_rule",
    "config_hash",
    "architecture_family",
    "frozen_architecture_params",
    "frozen_loss",
    "frozen_hpo_method",
    "frozen_seed_list",
    "frozen_metric_list",
    "frozen_wording_rule",
    "paper_safe_score",
    "official_validation_used_for_selection",
    "holdout_test_authorized",
}

# §13.2 optional but recommended freeze record fields. The contract does NOT
# require these at write time (so the test fixtures stay minimal) but the
# generator writes them by default.
RECOMMENDED_FREEZE_RECORD_FIELDS = {
    "paper_safe_score_runner_up",
    "paper_safe_score_margin",
    "challenger_baseline_id",
    "frozen_code_git_sha",
    "frozen_python_env_hash",
    "frozen_dependency_versions",
    "low_compute_baseline",
    "low_compute_submode",
}


# Belt-and-suspenders: exact-substring layer catches snake_case identifiers
# that survive normalization without separator collapse.
FORBIDDEN_FALLBACK_RULE_SUBSTRINGS = (
    "official_validation_macro_f1",
    "official_validation_delta",
    "official_val_score",
    "official_validation_balanced_accuracy",
    "official_val_metric",
)


# Regex layer catches English-prose abuses across separators. Patterns are
# applied to the normalized rule string (lowercase + [-_\s]+ collapsed to " ").
FORBIDDEN_FALLBACK_PATTERNS_NORMALIZED = (
    # "official val ... <metric>"
    r"\bofficial val(?:idation)?\b[^.]*?\b(?:macro f1|balanced accuracy|delta|f1 score|f1|accuracy|metric|result|performance|auc|loss)\b",
    # "<metric> ... official val"   (note: 'score' alone excluded so "scoring official val" is OK)
    r"\b(?:macro f1|balanced accuracy|delta|f1 score|metric|result|performance|auc|loss)\b[^.]*?\bofficial val(?:idation)?\b",
    # "primary scor(es/ing) ... wors/lower/fail/poorly" -- primary metric comparison
    r"\b(?:primary|fallback|model|deep)\b[^.]*?\b(?:scor|perform)[a-z]*\b[^.]*?\b(?:wors|worse|lower|low|fail|poorly|badly)\b",
    # explicit comparison verbs
    r"\b(?:fails? to beat|cannot beat|does not beat|outperformed by|beaten by|loses to)\b",
)


# ---------- §10.1 OPERATOR_READOUT_AUTHORIZATION_SHA inputs ----------------

OPERATOR_READOUT_AUTHORIZATION_INPUTS = (
    ("08f_candidate_freeze_record.json", "json_canonical"),
    (
        "docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md",
        "text_lf",
    ),
    ("AGENTS.md", "text_lf"),
)


# ---------- §13.1/§13.3 08X / 08O Manifest Fields --------------------------

REQUIRED_08X_RUN_MANIFEST_FIELDS = {
    "notebook08_version",
    "stage",
    "scope",
    "source_stage0_candidate",
    "official_validation_used",
    "holdout_test_authorized",
    "train_inner_fold_policy",
    "purge_policy",
    "embargo_policy",
    "search_budget_tier",
    "trial_count_requested",
    "trial_count_completed",
    "trial_count_failed",
    "trial_count_skipped",
}

REQUIRED_08O_RUN_MANIFEST_FIELDS = {
    "stage",
    "scope",
    "primary_candidate_id",
    "freeze_record_sha256",
    "official_validation_readout_started_at",
    "official_validation_used_for_selection",
    "holdout_test_authorized",
    "same_row_dummy_present",
    "per_ticker_present",
    "seed_summary_present",
    "allowed_wording_bucket",
}


# ---------- §13.3 08O artifact completeness gate (Round 8 #1) --------------
# Round 7 hardened the MANIFEST so a schema_only_stub manifest cannot carry
# evidence wording. Round 8 closes the upstream gap: the generator's previous
# "wrote_any_rows = any(file non-empty)" check was too permissive -- writing
# one row to one artifact would flip the manifest into real-readout mode.
#
# Real-mode is now defined as: every REQUIRED artifact present + non-empty +
# carries the §13.3 schema columns. Anything short of that stays in stub
# mode, which §10.4 forces into the no_candidate_freezable wording bucket.
#
# 08o_concentration_guardrails.csv and 08o_failure_rows.csv are intentionally
# NOT in REQUIRED_08O_REAL_READOUT_ARTIFACTS -- a real readout with zero
# concentration warnings and zero failed seeds is legitimate, so requiring
# non-empty rows there would create a false-positive stub flag. They remain
# in OUTPUT_FILES_08O for write-out but do not gate real-mode promotion.

REQUIRED_08O_REAL_READOUT_ARTIFACTS = {
    "08o_validation_readout.csv": {
        "seed",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "delta_macro_f1_vs_stratified_dummy_same_rows",
        "delta_balanced_accuracy_vs_stratified_dummy_same_rows",
        "validation_n",
        "class0_pred_rate",
        "class1_pred_rate",
    },
    "08o_validation_per_ticker.csv": {
        "ticker",
        "macro_f1",
        "delta_macro_f1_vs_dummy",
        "n_rows",
    },
    "08o_seed_summary.csv": {
        "metric",
        "seed_mean",
        "seed_std",
        "seed_lcb_95",
    },
    "08o_same_row_baselines.csv": {
        "baseline",
        "macro_f1_mean",
        "macro_f1_std",
    },
}


# ---------- §13.2 08F Freeze MD/JSON Output Filenames ----------------------

OUTPUT_FILES_08X = (
    "08x_search_space.json",
    "08x_trial_ledger.csv",
    "08x_fold_results.csv",
    "08x_seed_summary.csv",
    "08x_failure_ledger.csv",
    "08x_candidate_compression_table.csv",
    "08x_run_manifest.json",
    "08x_environment_manifest.json",
)

OUTPUT_FILES_08F = (
    "08f_candidate_freeze_record.json",
    "08f_candidate_freeze_record.md",
    "08f_candidate_compression_audit.csv",
    "08f_static_gate_report.json",
    "08f_no_candidate_freezable.json",
)

OUTPUT_FILES_08O = (
    "08o_validation_readout.csv",
    "08o_validation_per_ticker.csv",
    "08o_seed_summary.csv",
    "08o_same_row_baselines.csv",
    "08o_concentration_guardrails.csv",
    "08o_failure_rows.csv",
    "08o_decision_record.json",
    "08o_run_manifest.json",
)


# ---------- §10.4 Allowed Wording Buckets ----------------------------------

ALLOWED_WORDING_BUCKETS = (
    "improvement",
    "weak_mixed",
    "low_compute_baseline",
    "no_candidate_freezable",
    "unstable",
)


# ===========================================================================
# Helpers
# ===========================================================================


def _normalize_fallback_rule(text: str) -> str:
    """Lowercase + collapse [-_\\s] runs to single space so prose variants
    ('official-validation', 'official_val', 'official validation') normalize
    to the same form before regex matching."""
    return re.sub(r"[\s\-_]+", " ", text.lower())


def operator_readout_authorization_sha(
    fixed_order_inputs: list[tuple[Path, str]],
) -> str:
    """Compute the §10.1 canonical-bytes sha256 over fixed-order inputs.

    Recipe (verbatim from design §10.1):
      for each input in fixed_order_inputs (DO NOT sort, DO NOT change order):
        1. encode the input's relative_path as UTF-8 bytes;
        2. emit `len(path_bytes).to_bytes(8, "big") + path_bytes`;
        3. canonicalize the file bytes:
             - "json_canonical": json.loads(text) then
               json.dumps(obj, sort_keys=True, separators=(",", ":"),
               ensure_ascii=False, allow_nan=False).encode("utf-8");
             - "text_lf": open(path, "rb").read().replace(b"\\r\\n", b"\\n");
        4. emit `len(canonical_bytes).to_bytes(8, "big") + canonical_bytes`.
      return hashlib.sha256(stream).hexdigest()

    ``fixed_order_inputs`` is a list of ``(Path, mode)`` where mode is
    ``"json_canonical"`` or ``"text_lf"``. ``Path`` may be absolute (the
    canonical bytes use ``Path.as_posix()`` relative to the project root,
    so callers must pass already-relative paths; this matches how design
    §10.1 specifies "relative_path").
    """
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
                f"unsupported canonicalization mode: {mode!r} "
                f"(expected 'json_canonical' or 'text_lf')"
            )
        hasher.update(len(canonical).to_bytes(8, "big"))
        hasher.update(canonical)
    return hasher.hexdigest()


# ===========================================================================
# Validators
# ===========================================================================


def validate_dmc_attestation(payload: dict) -> None:
    """§9.1 dmc_attestation.json schema -- required fields, role enum, hex sha256."""
    missing = REQUIRED_DMC_FIELDS - payload.keys()
    if missing:
        raise AssertionError(f"dmc_attestation.json missing fields: {sorted(missing)}")
    if payload["dmc_role"] != "data_monitoring_committee_proxy":
        raise AssertionError(
            f"dmc_role must be 'data_monitoring_committee_proxy'; got {payload['dmc_role']!r}"
        )
    sha = str(payload["reviewed_08x_run_manifest_sha256"])
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise AssertionError("reviewed_08x_run_manifest_sha256 must be hex sha256 (64 chars)")


def validate_08f_entry(
    *,
    dmc_attestation: dict | None,
    same_session_as_08x: bool,
    separate_session_attestation: dict | None = None,
) -> None:
    """§9.1 08F entry gate: separate Colab session by non-08X-author OR
    valid dmc_attestation.json present in 08F input directory.

    Round 7 finding #3: "separate session" must be explicitly attested via
    ``separate_session_attestation``; absence of the flag does not count as
    proof of a separate session. Acceptable inputs:

      - ``same_session_as_08x=True`` + valid ``dmc_attestation``  -> OK
      - ``same_session_as_08x=False`` + valid ``separate_session_attestation``
        + (dmc_attestation is None OR valid) -> OK
      - any other combination -> AssertionError
    """
    if dmc_attestation is not None:
        validate_dmc_attestation(dmc_attestation)
    if same_session_as_08x:
        if dmc_attestation is None:
            raise AssertionError(
                "08F entry violation: same session as 08X AND no dmc_attestation.json"
            )
        return
    # same_session_as_08x is False -> require a positive attestation file.
    if separate_session_attestation is None:
        if dmc_attestation is None:
            raise AssertionError(
                "08F entry violation: SAME_COLAB_SESSION_AS_08X=False requires "
                "separate_session_attestation.json (or a valid dmc_attestation.json "
                "as fallback); neither was provided"
            )
        return  # DMC alone is enough when same_session_as_08x is False
    validate_separate_session_attestation(separate_session_attestation)


REQUIRED_SEPARATE_SESSION_ATTESTATION_FIELDS = {
    "attestation_kind",
    "reviewer_identifier",
    "reviewed_08x_run_manifest_sha256",
    "attested_at_utc",
    "attestation_statement",
}


def validate_separate_session_attestation(payload: dict) -> None:
    """Round 7 finding #3 -- positive attestation file for the
    SAME_COLAB_SESSION_AS_08X=False branch of §9.1.

    Required fields:
      - ``attestation_kind``: must equal ``"separate_colab_session_by_non_08x_author"``
      - ``reviewer_identifier``: free-text name / stable id
      - ``reviewed_08x_run_manifest_sha256``: 64-hex sha256 of the 08X manifest
      - ``attested_at_utc``: ISO 8601 timestamp
      - ``attestation_statement``: short text
    """
    missing = REQUIRED_SEPARATE_SESSION_ATTESTATION_FIELDS - payload.keys()
    if missing:
        raise AssertionError(
            f"separate_session_attestation.json missing fields: {sorted(missing)}"
        )
    if payload["attestation_kind"] != "separate_colab_session_by_non_08x_author":
        raise AssertionError(
            "separate_session_attestation.attestation_kind must be "
            "'separate_colab_session_by_non_08x_author'; "
            f"got {payload['attestation_kind']!r}"
        )
    sha = str(payload["reviewed_08x_run_manifest_sha256"])
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise AssertionError(
            "separate_session_attestation.reviewed_08x_run_manifest_sha256 "
            "must be hex sha256 (64 chars)"
        )


def validate_freeze_record(payload: dict) -> None:
    """§13.2 freeze record schema -- required fields, stage enum, scope enum,
    boolean guards (holdout closed, no official-val selection), fallback rule
    cannot reference official-validation metric (substring + normalized regex)."""
    missing = REQUIRED_FREEZE_RECORD_FIELDS - payload.keys()
    if missing:
        raise AssertionError(f"freeze record missing fields: {sorted(missing)}")
    if payload["stage"] != "08F":
        raise AssertionError(f"freeze record stage must be 08F; got {payload['stage']!r}")
    if payload["scope"] not in {"diagnostic", "validation_only"}:
        raise AssertionError(f"freeze record scope invalid: {payload['scope']!r}")
    if bool(payload["holdout_test_authorized"]):
        raise AssertionError("freeze record marks holdout_test_authorized=True (forbidden)")
    if bool(payload["official_validation_used_for_selection"]):
        raise AssertionError(
            "freeze record marks official_validation_used_for_selection=True (forbidden)"
        )
    rule_raw = str(payload["fallback_activation_rule"])
    rule_lower = rule_raw.lower()
    for tok in FORBIDDEN_FALLBACK_RULE_SUBSTRINGS:
        if tok in rule_lower:
            raise AssertionError(
                f"fallback_activation_rule references official-validation metric: {tok!r}"
            )
    rule_normalized = _normalize_fallback_rule(rule_raw)
    for pat in FORBIDDEN_FALLBACK_PATTERNS_NORMALIZED:
        m = re.search(pat, rule_normalized)
        if m is not None:
            raise AssertionError(
                f"fallback_activation_rule references official-validation metric: {m.group(0)!r}"
            )


def validate_08o_ledger_append_precedes_read(
    *,
    ledger_before_read: pd.DataFrame,
    ledger_after_read: pd.DataFrame,
) -> None:
    """§10.2 step 0 + AGENTS.md §4.3: 08O must append exactly ONE row recording
    its read intent BEFORE actually reading official validation rows. The new
    row must:
      - carry appended_by_notebook == "08O"
      - carry decision_timing == "before_official_validation_read"
      - carry official_validation_rows_inspected == 0 (the read has not happened)
      - increment the cumulative counter by exactly +1 over the prior max

    Pre-existing rows are untouched (append-only invariance: the first n_before
    rows of ledger_after_read MUST equal ledger_before_read row-for-row and
    column-for-column).
    """
    n_before = len(ledger_before_read)
    n_after = len(ledger_after_read)
    if n_after != n_before + 1:
        raise AssertionError(
            f"08O must append exactly 1 row; before={n_before}, after={n_after}"
        )
    # Append-only prefix invariance: existing rows must not be modified, dropped,
    # reordered, or have new columns inserted into them.
    if n_before > 0:
        prefix_after = ledger_after_read.iloc[:n_before].reset_index(drop=True)
        before_reset = ledger_before_read.reset_index(drop=True)
        if list(prefix_after.columns) != list(before_reset.columns):
            raise AssertionError(
                "ledger is not append-only: column set or order changed "
                f"(before={list(before_reset.columns)}, "
                f"prefix_after={list(prefix_after.columns)})"
            )
        if not prefix_after.equals(before_reset):
            raise AssertionError(
                "ledger is not append-only: pre-existing rows were modified "
                "between read and append"
            )
    new_row = ledger_after_read.iloc[-1]
    appended_by = str(new_row.get("appended_by_notebook"))
    if appended_by != "08O":
        raise AssertionError(
            f"appended_by_notebook must be '08O'; got {appended_by!r}"
        )
    timing = str(new_row.get("decision_timing"))
    if timing != "before_official_validation_read":
        raise AssertionError(
            "decision_timing must be 'before_official_validation_read'; "
            f"got {timing!r}"
        )
    inspected_at_intent = int(new_row.get("official_validation_rows_inspected", -1))
    if inspected_at_intent != 0:
        raise AssertionError(
            "official_validation_rows_inspected must be 0 at append time "
            f"(read hasn't happened yet); got {inspected_at_intent}"
        )
    last_before = (
        int(ledger_before_read["cumulative_official_validation_inspections_across_notebooks"].max())
        if not ledger_before_read.empty
        else 0
    )
    last_after = int(new_row["cumulative_official_validation_inspections_across_notebooks"])
    if last_after != last_before + 1:
        raise AssertionError(
            f"cumulative counter must be +1 from previous max; "
            f"before_max={last_before}, after_new_row={last_after}"
        )


def validate_trial_ledger_frame(df: pd.DataFrame) -> None:
    """§8.3 trial ledger schema -- required column set + compute_tier enum +
    scope/holdout/official guards + fit_status enum + failure_type enum
    (Round 7 finding #6).

    Every row must declare ``fit_status`` from ``FIT_STATUSES``. Rows with
    ``fit_status == "failed"`` must additionally carry ``failure_type`` from
    ``FAILURE_TYPES`` (a failure ledger row without a typed failure makes
    08F's failure map unauditable). Rows with ``fit_status != "failed"`` may
    leave ``failure_type`` empty.

    Raises ``AssertionError`` on contract violations.
    """
    missing = REQUIRED_TRIAL_LEDGER_COLUMNS - set(df.columns)
    if missing:
        raise AssertionError(f"08x_trial_ledger missing columns: {sorted(missing)}")
    if df.empty:
        return
    bad_tier = set(df["compute_tier"].astype(str).unique()) - set(COMPUTE_TIER_VALUES)
    if bad_tier:
        raise AssertionError(
            f"compute_tier has invalid values: {sorted(bad_tier)} "
            f"(allowed: {sorted(COMPUTE_TIER_VALUES)})"
        )
    if (df["scope"].astype(str) != "exploratory").any():
        raise AssertionError("08x_trial_ledger scope column must be 'exploratory' for all rows")
    if df["official_validation_used"].astype(bool).any():
        raise AssertionError("08x_trial_ledger contains a row with official_validation_used=True")
    if df["holdout_test_authorized"].astype(bool).any():
        raise AssertionError("08x_trial_ledger contains a row with holdout_test_authorized=True")
    bad_fit = set(df["fit_status"].astype(str).unique()) - set(FIT_STATUSES)
    if bad_fit:
        raise AssertionError(
            f"08x_trial_ledger fit_status has invalid values: {sorted(bad_fit)} "
            f"(allowed: {sorted(FIT_STATUSES)})"
        )
    failed_mask = df["fit_status"].astype(str) == "failed"
    if failed_mask.any():
        failure_values = df.loc[failed_mask, "failure_type"].astype(str)
        bad_failure = set(failure_values.unique()) - set(FAILURE_TYPES)
        if bad_failure:
            raise AssertionError(
                "08x_trial_ledger has failed rows with invalid failure_type: "
                f"{sorted(bad_failure)} (allowed: {sorted(FAILURE_TYPES)})"
            )


def validate_08x_search_space(payload: dict) -> None:
    """§7 + §11 search-space JSON guard.

    The payload MUST declare:
      - `architecture_families`: subset of SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES
      - `hpo_method`: single value from HPO_METHODS (§7.6 - same HPO per run)
      - `eligibility_thresholds.min_train_inner_lcb_delta_macro_f1`: numeric
      - `scientific_budget_cap_total_trials`: int <= TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES
      - `per_family_trial_budget`: dict mapping each declared family to int
      - `official_validation_used`: must be False (08X is train-inner only; §4.1)
      - `holdout_test_authorized`: must be False (08X is train-inner only; §4.1)
    If `low_compute_mode == True`, `low_compute_submode` must be a known enum
    value and (for sub-mode B) the nested-fold protocol MUST be declared.

    Round 7 finding #4: this validator now enforces the no-official /
    no-holdout flags directly, so the contract module owns the guard rather
    than relying on the generator's defaults.
    """
    # Round 7 finding #4: contract-side guard against operator drift.
    if "official_validation_used" not in payload:
        raise AssertionError(
            "08x_search_space.official_validation_used must be declared as False"
        )
    if bool(payload["official_validation_used"]):
        raise AssertionError(
            "08x_search_space.official_validation_used=True is forbidden "
            "(08X is train-inner only per §4.1)"
        )
    if "holdout_test_authorized" not in payload:
        raise AssertionError(
            "08x_search_space.holdout_test_authorized must be declared as False"
        )
    if bool(payload["holdout_test_authorized"]):
        raise AssertionError(
            "08x_search_space.holdout_test_authorized=True is forbidden"
        )
    families = payload.get("architecture_families")
    if not isinstance(families, list) or not families:
        raise AssertionError("08x_search_space.architecture_families must be a non-empty list")
    bad_families = set(families) - set(ARCHITECTURE_FAMILIES)
    if bad_families:
        raise AssertionError(
            f"08x_search_space.architecture_families has unknown entries: {sorted(bad_families)}"
        )
    ineligible_families = set(families) - set(SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES)
    if ineligible_families:
        raise AssertionError(
            "08x_search_space.architecture_families contains families that are "
            "not 08X search-eligible until their axes are frozen in config / "
            f"search-space: {sorted(ineligible_families)}"
        )
    hpo = payload.get("hpo_method")
    if hpo not in HPO_METHODS:
        raise AssertionError(
            f"08x_search_space.hpo_method must be one of {list(HPO_METHODS)}; got {hpo!r}"
        )
    thresholds = payload.get("eligibility_thresholds", {})
    margin = thresholds.get("min_train_inner_lcb_delta_macro_f1")
    if not isinstance(margin, (int, float)):
        raise AssertionError(
            "08x_search_space.eligibility_thresholds.min_train_inner_lcb_delta_macro_f1 "
            "must be a numeric value (§9.1)"
        )
    cap = payload.get("scientific_budget_cap_total_trials")
    if not isinstance(cap, int) or cap <= 0:
        raise AssertionError(
            "08x_search_space.scientific_budget_cap_total_trials must be a positive int"
        )
    if cap > TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES:
        raise AssertionError(
            f"scientific_budget_cap_total_trials={cap} exceeds §5.5 cap "
            f"{TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES}"
        )
    per_family = payload.get("per_family_trial_budget", {})
    if not isinstance(per_family, dict):
        raise AssertionError("08x_search_space.per_family_trial_budget must be a dict")
    missing_family_budgets = set(families) - set(per_family.keys())
    if missing_family_budgets:
        raise AssertionError(
            "08x_search_space.per_family_trial_budget missing entries for: "
            f"{sorted(missing_family_budgets)}"
        )
    if payload.get("low_compute_mode"):
        submode = payload.get("low_compute_submode")
        if submode not in LOW_COMPUTE_SUBMODES:
            raise AssertionError(
                f"low_compute_submode must be one of {list(LOW_COMPUTE_SUBMODES)}; got {submode!r}"
            )
        if submode == "train_inner_oof_mlp_head":
            validate_low_compute_submode_b_protocol(payload)


def validate_low_compute_submode_b_protocol(search_space: dict) -> None:
    """§7.9 sub-mode B: nested-fold protocol must be fully declared.

    Raises AssertionError if any of the five required declarations is missing,
    if the outer fold scheme is not one of the allowed schemes, or if
    outer_fold_k / inner_fold_k_for_head are < 5.
    """
    missing = [f for f in LOW_COMPUTE_SUBMODE_B_REQUIRED_FIELDS if f not in search_space]
    if missing:
        raise AssertionError(
            "low_compute_submode='train_inner_oof_mlp_head' requires fields: "
            f"{missing}"
        )
    scheme = search_space["outer_fold_scheme"]
    if scheme not in LOW_COMPUTE_SUBMODE_B_ALLOWED_OUTER_FOLD_SCHEMES:
        raise AssertionError(
            f"outer_fold_scheme must be one of "
            f"{list(LOW_COMPUTE_SUBMODE_B_ALLOWED_OUTER_FOLD_SCHEMES)}; got {scheme!r}"
        )
    outer_k = search_space["outer_fold_k"]
    inner_k = search_space["inner_fold_k_for_head"]
    if not isinstance(outer_k, int) or outer_k < LOW_COMPUTE_SUBMODE_B_MIN_OUTER_FOLD_K:
        raise AssertionError(
            f"outer_fold_k must be int >= {LOW_COMPUTE_SUBMODE_B_MIN_OUTER_FOLD_K}; got {outer_k!r}"
        )
    if not isinstance(inner_k, int) or inner_k < LOW_COMPUTE_SUBMODE_B_MIN_INNER_FOLD_K:
        raise AssertionError(
            f"inner_fold_k_for_head must be int >= {LOW_COMPUTE_SUBMODE_B_MIN_INNER_FOLD_K}; "
            f"got {inner_k!r}"
        )
    expected_train = "outer_fold_i.oof_predictions_excluding_held_out_inner_fold"
    expected_eval = "outer_fold_i.oof_predictions_from_held_out_inner_fold"
    if search_space["head_train_data_source"] != expected_train:
        raise AssertionError(
            f"head_train_data_source must be {expected_train!r}; "
            f"got {search_space['head_train_data_source']!r}"
        )
    if search_space["head_eval_data_source"] != expected_eval:
        raise AssertionError(
            f"head_eval_data_source must be {expected_eval!r}; "
            f"got {search_space['head_eval_data_source']!r}"
        )


def validate_08x_run_manifest(payload: dict) -> None:
    """§13.1 08X run manifest schema."""
    missing = REQUIRED_08X_RUN_MANIFEST_FIELDS - payload.keys()
    if missing:
        raise AssertionError(f"08x_run_manifest missing fields: {sorted(missing)}")
    if payload["stage"] != "08X":
        raise AssertionError(f"08x_run_manifest stage must be '08X'; got {payload['stage']!r}")
    if payload["scope"] != "exploratory":
        raise AssertionError(
            f"08x_run_manifest scope must be 'exploratory'; got {payload['scope']!r}"
        )
    if bool(payload["official_validation_used"]):
        raise AssertionError(
            "08x_run_manifest official_validation_used=True is forbidden in 08X"
        )
    if bool(payload["holdout_test_authorized"]):
        raise AssertionError(
            "08x_run_manifest holdout_test_authorized=True is forbidden"
        )


def check_08o_real_readout_completeness(output_dir) -> dict:
    """Round 8 #1 -- strict gate on what counts as a real 08O readout.

    Returns a dict describing the per-artifact verdict so the caller can
    write it into the manifest for transparent audit:

    ::

        {
          "is_real_readout": bool,            # True only when every required
                                              # artifact passes ALL three checks
          "per_artifact": {
            "<filename>": {
              "present":         bool,        # file exists on disk
              "non_empty":       bool,        # at least one data row beyond header
              "schema_complete": bool,        # required columns are present
                                              # (extra columns are OK -- additive)
              "missing_columns": [...],       # required columns NOT present
              "row_count":       int,         # number of data rows read
            },
            ...
          },
          "missing_artifacts":  [...],        # files that flunked `present`
          "empty_artifacts":    [...],        # files that flunked `non_empty`
          "schema_drift":       [...],        # files that flunked `schema_complete`
        }

    Real-mode promotion requires EVERY required artifact in
    ``REQUIRED_08O_REAL_READOUT_ARTIFACTS`` to pass present + non_empty +
    schema_complete. Otherwise the manifest stays in stub mode.

    The previous generator-side check (`any(file non_empty)`) was too
    permissive: writing one row to one artifact would flip the manifest into
    real-readout mode (Round 8 review finding). This function returns a
    structured verdict instead of raising so the generator can record it
    on the manifest for the reviewer, but the boolean ``is_real_readout``
    must be honored by the caller.
    """
    from pathlib import Path as _Path  # local import keeps top-level surface clean

    base = _Path(output_dir)
    per_artifact: dict[str, dict] = {}
    missing_artifacts: list[str] = []
    empty_artifacts: list[str] = []
    schema_drift: list[str] = []
    for filename, required_columns in REQUIRED_08O_REAL_READOUT_ARTIFACTS.items():
        path = base / filename
        present = path.exists()
        non_empty = False
        schema_complete = False
        missing_cols: list[str] = []
        row_count = 0
        if present:
            try:
                df = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                # File exists but is unreadable -- treat as not non_empty and
                # not schema_complete so it stays in stub mode.
                df = None
            if df is not None:
                row_count = int(len(df))
                non_empty = row_count > 0
                missing_cols = sorted(set(required_columns) - set(df.columns))
                schema_complete = not missing_cols
        verdict = {
            "present": present,
            "non_empty": non_empty,
            "schema_complete": schema_complete,
            "missing_columns": missing_cols,
            "row_count": row_count,
        }
        per_artifact[filename] = verdict
        if not present:
            missing_artifacts.append(filename)
        if present and not non_empty:
            empty_artifacts.append(filename)
        if present and non_empty and not schema_complete:
            schema_drift.append(filename)
    is_real = (
        not missing_artifacts
        and not empty_artifacts
        and not schema_drift
    )
    return {
        "is_real_readout": is_real,
        "per_artifact": per_artifact,
        "missing_artifacts": missing_artifacts,
        "empty_artifacts": empty_artifacts,
        "schema_drift": schema_drift,
    }


def validate_08o_run_manifest(payload: dict) -> None:
    """§13.3 08O run manifest schema -- required fields, scope guard,
    same-row dummy / per-ticker / seed summary presence flags.

    Round 7 finding #1: the manifest now distinguishes a schema-only stub
    (no real official-validation read happened; the artifact rows have only
    headers) from a true readout. The optional ``schema_only_stub`` field
    governs this:

      - ``schema_only_stub=True``  -> ``*_present`` may be False;
        ``allowed_wording_bucket`` MUST be ``"no_candidate_freezable"`` so
        downstream consumers cannot mistake stubs for evidence.
      - ``schema_only_stub=False`` (or absent) -> the legacy contract:
        all three ``*_present`` flags MUST be True before the manifest is
        considered a real readout.
    """
    missing = REQUIRED_08O_RUN_MANIFEST_FIELDS - payload.keys()
    if missing:
        raise AssertionError(f"08o_run_manifest missing fields: {sorted(missing)}")
    if payload["stage"] != "08O":
        raise AssertionError(f"08o_run_manifest stage must be '08O'; got {payload['stage']!r}")
    if payload["scope"] != "validation_only":
        raise AssertionError(
            f"08o_run_manifest scope must be 'validation_only'; got {payload['scope']!r}"
        )
    if bool(payload["official_validation_used_for_selection"]):
        raise AssertionError(
            "08o_run_manifest official_validation_used_for_selection=True is forbidden"
        )
    if bool(payload["holdout_test_authorized"]):
        raise AssertionError(
            "08o_run_manifest holdout_test_authorized=True is forbidden"
        )
    schema_only_stub = bool(payload.get("schema_only_stub", False))
    if schema_only_stub:
        # Stub mode: stub manifests MUST carry the no-candidate wording bucket
        # so a downstream paper / static gate cannot accidentally treat empty
        # artifacts as evidence (Round 7 finding #1).
        if payload["allowed_wording_bucket"] != "no_candidate_freezable":
            raise AssertionError(
                "08o_run_manifest schema_only_stub=True requires "
                "allowed_wording_bucket='no_candidate_freezable'; "
                f"got {payload['allowed_wording_bucket']!r}"
            )
    else:
        # Real-readout mode: presence flags must be True.
        for flag in (
            "same_row_dummy_present", "per_ticker_present", "seed_summary_present"
        ):
            if not bool(payload[flag]):
                raise AssertionError(
                    f"08o_run_manifest {flag}=False; required artifacts missing per §13.3"
                )
    if payload["allowed_wording_bucket"] not in ALLOWED_WORDING_BUCKETS:
        raise AssertionError(
            f"08o_run_manifest allowed_wording_bucket must be one of "
            f"{list(ALLOWED_WORDING_BUCKETS)}; got {payload['allowed_wording_bucket']!r}"
        )
