"""Artifact contract helpers for Notebook 07 outputs.

Lifts the inline validators previously embedded in
``tests/test_notebook07_artifact_contract.py`` into a reusable module so the
N07 colab notebook generator can inline the same logic for Colab portability.

The validators here mirror the design rules in
``docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md``:

* ``thesis_paragraph_kit.json`` schema + AGENTS.md §4.2.5a gate.
* ``notebook07_validation_budget_ledger.csv`` schema + append-only monotonic
  cumulative counter rule.
* ``notebook07_final_validation_comparison.csv`` schema + row-class-conditional
  NA contract described in §07B.

These helpers are validation-only utilities; they do not load, transform, or
score holdout/test data.
"""

from __future__ import annotations

import re

import pandas as pd


NOTEBOOK07_SCOPE = "validation_only"


# ---------- Pre-registration Constants Table (§N07 design) -----------------
# Mirrors the numeric thresholds enumerated in the design doc. Any change here
# must be accompanied by a refreshed design doc sha256 and a refreshed
# notebook07_lockfile_scope_gate.json (see design §07A).

# AGENTS.md §4.2.5a thresholds.
IMPROVEMENT_LCB_MIN = 0.005
IMPROVEMENT_TICKER_COUNT_MIN = 4

# §07B weak-signal band and §07D concentration thresholds.
WEAK_SIGNAL_BAND_UPPER = 0.005
WEAK_SIGNAL_BAND_LOWER = 0.0
CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX = 0.35
CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN = 4
WEAK_SEED_EVIDENCE_COUNT_THRESHOLD = 5

# §07F alpha-spending policy total budget.
NULL_CONTROL_ALPHA_TOTAL = 0.05


# Forbidden-phrase regex (§07H belt-and-suspenders layer over the explicit
# forbidden-wording list). Pinned here so the same pattern is used by the
# generator's runtime cell, the notebook's 07H wording emission, and this
# helper's thesis-kit validator.
FORBIDDEN_PHRASE_REGEX = r"\b(final|production|deploy(?:ed|able|ment)?|tradable|live|sharpe|alpha)\b"
FORBIDDEN_PHRASE_REGEX_PATTERN = re.compile(FORBIDDEN_PHRASE_REGEX, re.IGNORECASE)


# label_config string pattern (e.g. "h03_bps1p5" -> horizon 3, threshold 1.5).
_LABEL_CONFIG_PATTERN = re.compile(r"^h(\d+)_bps(\d+)p(\d+)$")


def parse_label_config(label_config) -> dict:
    """Parse a label_config string (e.g. ``"h03_bps1p5"``) into structured fields.

    Returns ``{"horizon_k": int, "threshold_bps": float}`` so 07B can emit the
    correct provenance even if the upstream N05 row only carries the encoded
    string.

    Raises ``ValueError`` if the input does not match the canonical pattern.
    """
    text = str(label_config).strip()
    match = _LABEL_CONFIG_PATTERN.match(text)
    if not match:
        raise ValueError(
            f"Cannot parse label_config: {label_config!r} (expected pattern 'hN_bpsMpD')"
        )
    horizon_k = int(match.group(1))
    bps_int = int(match.group(2))
    bps_frac = int(match.group(3))
    threshold_bps = float(f"{bps_int}.{bps_frac}")
    return {"horizon_k": horizon_k, "threshold_bps": threshold_bps}


# ---------- thesis_paragraph_kit.json contract (§07H) ----------------------


REQUIRED_THESIS_KIT_FIELDS = {
    "results_paragraph",
    "robustness_paragraph",
    "limitation_paragraph",
    "caveat_phrases_used",
    "forbidden_phrases_blocked_at_runtime",
    "reproducibility_pointers",
    "improvement_wording_applied",
    "improvement_threshold_check",
}
REQUIRED_THRESHOLD_CHECK_FIELDS = {
    "delta_macro_f1_vs_dummy_lcb_95",
    "positive_ticker_count",
    "passed_per_AGENTS_md_4_2_5a",
}


# ---------- validation_budget_ledger.csv contract (§07C) -------------------


REQUIRED_LEDGER_COLUMNS = {
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
}


# ---------- final_validation_comparison.csv contract (§07B) ----------------


REQUIRED_FINAL_COMPARISON_COLUMNS = {
    "artifact_source",
    "notebook_stage",
    "row_class",
    "model",
    "profile_id",
    "profile_role",
    "label_config",
    "horizon_k",
    "threshold_bps",
    "feature_set",
    "window_size",
    "seed_count",
    "macro_f1_mean",
    "macro_f1_std",
    "macro_f1_lcb_95",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "accuracy_mean",
    "dummy_macro_f1_mean",
    "dummy_balanced_accuracy_mean",
    "delta_macro_f1_vs_dummy_mean",
    "delta_macro_f1_vs_dummy_lcb_95",
    "delta_balanced_accuracy_vs_dummy_mean",
    "always_up_dummy_macro_f1_mean",
    "delta_macro_f1_vs_always_up_dummy_mean",
    "positive_ticker_count",
    "top_ticker_gain_share",
    "validation_n",
    "scope",
    "decision_source",
    "allowed_wording_tag",
}
CONDITIONAL_FINAL_COMPARISON_COLUMNS = {
    "coverage",
    "coverage_source",
    "retained_n",
    "abstained_n",
    "random_abstention_macro_f1_mean",
    "delta_macro_f1_vs_random_abstention_mean",
}
ALLOWED_ROW_CLASS_VALUES = {"full_coverage", "selective"}


# ---------- Validators -----------------------------------------------------


def validate_thesis_paragraph_kit(payload: dict) -> None:
    """Enforce the §07H thesis_paragraph_kit.json schema + AGENTS.md §4.2.5a
    improvement-wording gate + forbidden-phrase scan over the emitted paragraphs.

    Raises ``AssertionError`` on contract violations. Returns ``None`` on
    success.
    """
    missing = REQUIRED_THESIS_KIT_FIELDS - payload.keys()
    if missing:
        raise AssertionError(f"thesis_paragraph_kit missing fields: {sorted(missing)}")
    check = payload["improvement_threshold_check"]
    if not isinstance(check, dict):
        raise AssertionError("improvement_threshold_check must be an object")
    missing_check = REQUIRED_THRESHOLD_CHECK_FIELDS - check.keys()
    if missing_check:
        raise AssertionError(
            f"improvement_threshold_check missing fields: {sorted(missing_check)}"
        )
    passed_claim = bool(check["passed_per_AGENTS_md_4_2_5a"])
    lcb = float(check["delta_macro_f1_vs_dummy_lcb_95"])
    pos_n = int(check["positive_ticker_count"])
    real_pass = lcb >= IMPROVEMENT_LCB_MIN and pos_n >= IMPROVEMENT_TICKER_COUNT_MIN
    if passed_claim != real_pass:
        raise AssertionError(
            "passed_per_AGENTS_md_4_2_5a disagrees with measured "
            f"(lcb={lcb}, pos_ticker_count={pos_n}, claim={passed_claim}, real={real_pass})"
        )
    if bool(payload["improvement_wording_applied"]) and not real_pass:
        raise AssertionError(
            "improvement_wording_applied=True without meeting AGENTS.md §4.2.5a thresholds"
        )
    # Forbidden-phrase scan over the emitted paragraphs. Anything that survives
    # 07H's runtime block must NOT reach the kit; the regex catches close variants
    # (e.g. "production-grade", "near-final", "ready for live").
    leaked = []
    for key in ("results_paragraph", "robustness_paragraph", "limitation_paragraph"):
        text = str(payload.get(key, ""))
        matches = FORBIDDEN_PHRASE_REGEX_PATTERN.findall(text)
        if matches:
            leaked.append({"paragraph": key, "phrases": sorted({m.lower() for m in matches})})
    if leaked:
        raise AssertionError(
            f"thesis_paragraph_kit paragraphs leaked forbidden phrases: {leaked}"
        )


SAME_ROW_DUMMY_REQUIRED_NON_NULL_COLUMNS = (
    "dummy_macro_f1_mean",
    "delta_macro_f1_vs_dummy_mean",
    "delta_macro_f1_vs_dummy_lcb_95",
)


def validate_final_validation_comparison_frame(df: pd.DataFrame) -> None:
    """Enforce §07B REQUIRED + CONDITIONAL REQUIRED contract on the final
    comparison frame.

    - REQUIRED columns must be present.
    - row_class must be in {"full_coverage", "selective"}.
    - For row_class == "selective", CONDITIONAL columns must be non-NA.
    - For row_class == "full_coverage", CONDITIONAL columns must be NA.
    - Every row (both classes) must carry non-NaN ``dummy_macro_f1_mean``,
      ``delta_macro_f1_vs_dummy_mean``, and ``delta_macro_f1_vs_dummy_lcb_95``.
      Same-row stratified dummy is mandatory per AGENTS.md §4.2 and the
      design §"Baseline rules". Silent NaN here would let a row reach
      thesis wording without a baseline.

    Raises ``AssertionError`` on contract violations.
    """
    missing_required = REQUIRED_FINAL_COMPARISON_COLUMNS - set(df.columns)
    if missing_required:
        raise AssertionError(
            f"final_validation_comparison missing REQUIRED columns: {sorted(missing_required)}"
        )
    missing_conditional = CONDITIONAL_FINAL_COMPARISON_COLUMNS - set(df.columns)
    if missing_conditional:
        raise AssertionError(
            "final_validation_comparison missing CONDITIONAL columns "
            f"(must be present in the rectangular CSV even when NA): "
            f"{sorted(missing_conditional)}"
        )
    bad_row_class = set(df["row_class"].unique()) - ALLOWED_ROW_CLASS_VALUES
    if bad_row_class:
        raise AssertionError(
            f"final_validation_comparison row_class has invalid values: {sorted(bad_row_class)}"
        )
    conditional_cols = sorted(CONDITIONAL_FINAL_COMPARISON_COLUMNS)
    selective_mask = df["row_class"] == "selective"
    full_mask = df["row_class"] == "full_coverage"
    if selective_mask.any():
        sel_na = df.loc[selective_mask, conditional_cols].isna()
        if sel_na.any().any():
            offending = sel_na.any(axis=0)
            cols_with_na = offending[offending].index.tolist()
            raise AssertionError(
                "selective rows must have non-NA conditional columns; "
                f"NA found in: {cols_with_na}"
            )
    if full_mask.any():
        full_not_na = df.loc[full_mask, conditional_cols].notna()
        if full_not_na.any().any():
            offending = full_not_na.any(axis=0)
            cols_with_value = offending[offending].index.tolist()
            raise AssertionError(
                "full_coverage rows must have NA in conditional columns; "
                f"non-NA found in: {cols_with_value}"
            )
    # Same-row dummy baseline hard-stop: silent NaN here would let a row
    # reach thesis wording without a baseline, violating AGENTS.md §4.2.
    for col in SAME_ROW_DUMMY_REQUIRED_NON_NULL_COLUMNS:
        if col not in df.columns:
            # already caught by REQUIRED check above; defensive
            continue
        na_mask = pd.to_numeric(df[col], errors="coerce").isna()
        if na_mask.any():
            offending_classes = sorted(
                df.loc[na_mask, "row_class"].astype(str).unique()
            )
            raise AssertionError(
                f"same-row dummy baseline is missing in column '{col}' for "
                f"{int(na_mask.sum())} row(s) (row_class(es)={offending_classes}); "
                "every full_coverage AND selective row MUST carry a non-NaN "
                "same-row dummy baseline (design §07B + AGENTS.md §4.2)"
            )


def validate_ledger_frame(df: pd.DataFrame) -> None:
    """Enforce §07C validation-budget ledger schema + append-only monotonic
    cumulative counter rule.

    Raises ``AssertionError`` on contract violations.
    """
    missing = REQUIRED_LEDGER_COLUMNS - set(df.columns)
    if missing:
        raise AssertionError(f"validation_budget_ledger missing columns: {sorted(missing)}")
    ordered = df.sort_values("appended_at_utc", kind="mergesort").reset_index(drop=True)
    counter = ordered["cumulative_official_validation_inspections_across_notebooks"].astype(int)
    if not counter.is_monotonic_increasing:
        raise AssertionError(
            "cumulative_official_validation_inspections_across_notebooks is not "
            "monotonically non-decreasing"
        )


def validate_ledger_prefix_invariance(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
    """Enforce strict append-only contract on validation_budget_ledger.csv.

    Given the on-disk ledger (``existing_df``) and the in-memory ledger about
    to be written (``new_df``), the new must be a strict superset preserving
    the **on-disk row order AND column order verbatim**:

    - ``new_df`` MUST carry exactly the same column set as ``existing_df``;
      dropping an existing column or adding a new column is forbidden.
    - ``new_df``'s column order MUST match ``existing_df`` exactly;
      reordering columns is forbidden.
    - The first ``len(existing_df)`` rows of ``new_df`` MUST be byte-equal to
      ``existing_df`` row-by-row, in original CSV order.
    - ``new_df`` may only append additional rows after that prefix.

    NOTE: This compares without sorting rows on purpose. AGENTS.md §4.3
    forbids "modified, dropped, or reordered" rows; a pure reorder of
    existing rows (same content, swapped positions) must therefore be
    rejected. An earlier revision sorted both sides on
    (appended_at_utc, notebook_stage, appended_by_notebook) and silently
    accepted pure reorders. A subsequent revision intersected on shared
    columns and silently accepted missing / extra / reordered columns
    (Round 6 P2 finding); both regressions are locked out below.

    Raises ``AssertionError`` on contract violations. Empty ``existing_df``
    is accepted (no prefix to compare).
    """
    if len(new_df) < len(existing_df):
        raise AssertionError(
            f"validation_budget_ledger lost rows: existing={len(existing_df)} new={len(new_df)}"
        )
    if existing_df.empty:
        return
    existing_reset = existing_df.reset_index(drop=True)
    new_prefix = new_df.iloc[: len(existing_reset)].reset_index(drop=True)
    existing_cols = list(existing_reset.columns)
    new_cols = list(new_prefix.columns)
    # Strict column set equality (Round 6 P2 lock).
    if set(existing_cols) != set(new_cols):
        missing_in_new = sorted(set(existing_cols) - set(new_cols))
        extra_in_new = sorted(set(new_cols) - set(existing_cols))
        raise AssertionError(
            "validation_budget_ledger column set changed: "
            f"missing_from_new={missing_in_new}, extra_in_new={extra_in_new}; "
            "existing ledger columns cannot be dropped or added (AGENTS.md §4.3)"
        )
    # Strict column order equality (Round 6 P2 lock).
    if existing_cols != new_cols:
        raise AssertionError(
            "validation_budget_ledger column order changed: "
            f"existing={existing_cols}, new={new_cols}; "
            "existing ledger column order cannot be reordered (AGENTS.md §4.3)"
        )
    # Cell-by-cell prefix check in the now-confirmed identical column order.
    for col in existing_cols:
        existing_vals = existing_reset[col].astype(str).reset_index(drop=True)
        new_vals = new_prefix[col].astype(str).reset_index(drop=True)
        if not existing_vals.equals(new_vals):
            diff = int((existing_vals != new_vals).sum())
            raise AssertionError(
                f"validation_budget_ledger prefix invariance violated in column '{col}' "
                f"({diff} row(s) differ); existing ledger rows cannot be modified, "
                "dropped, or reordered (AGENTS.md §4.3)"
            )
