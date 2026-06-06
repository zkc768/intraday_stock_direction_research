"""Artifact contract for Notebook 07 outputs.

Phase 1 / Task P1-T05..T06 of phased_implementation_plan.md.

Verifies two contracts introduced by the 2026-06-06 top-5 edits:
  P1-T05: notebook07_thesis_paragraph_kit.json schema and AGENTS.md §4.2.5a
          improvement-wording gate.
  P1-T06: notebook07_validation_budget_ledger.csv must carry the new
          cumulative-across-notebooks field; behaviour is append-only.

There is no scripts/notebook07_contract.py helper yet, so the schema validator
is inlined here. When Codex (or a follow-up task) lifts these checks into
scripts/notebook07_contract.py, update the imports below and remove the
inlined validators.
"""

import json
from pathlib import Path

import pandas as pd
import pytest


# ---------- Inline contract validators (replace once helper exists) --------


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

# AGENTS.md §4.2.5a thresholds.
IMPROVEMENT_LCB_MIN = 0.005
IMPROVEMENT_TICKER_COUNT_MIN = 4

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

# Final validation comparison frame schema (§07B). Selective vs full-coverage
# rows are distinguished by row_class; CONDITIONAL columns are present in the
# rectangular CSV but their NA pattern depends on row_class.
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


def validate_thesis_paragraph_kit(payload: dict) -> None:
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
    # Gate behaviour per AGENTS.md §4.2.5a.
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


def validate_final_validation_comparison_frame(df: pd.DataFrame) -> None:
    """Enforces §07B REQUIRED + CONDITIONAL REQUIRED contract on the final
    comparison frame.

    - REQUIRED columns must be present.
    - row_class must be in {"full_coverage", "selective"}.
    - For row_class == "selective", CONDITIONAL columns must be non-NA.
    - For row_class == "full_coverage", CONDITIONAL columns must be NA.
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


def validate_ledger_frame(df: pd.DataFrame) -> None:
    missing = REQUIRED_LEDGER_COLUMNS - set(df.columns)
    if missing:
        raise AssertionError(f"validation_budget_ledger missing columns: {sorted(missing)}")
    # The cumulative counter must be monotonic non-decreasing along the
    # append order (column appended_at_utc).
    ordered = df.sort_values("appended_at_utc", kind="mergesort").reset_index(drop=True)
    counter = ordered["cumulative_official_validation_inspections_across_notebooks"].astype(int)
    if not counter.is_monotonic_increasing:
        raise AssertionError(
            "cumulative_official_validation_inspections_across_notebooks is not "
            "monotonically non-decreasing"
        )


# ---------- Fixtures --------------------------------------------------------


def _base_kit() -> dict:
    return {
        "results_paragraph": "Under the locked chronological validation-only route, ...",
        "robustness_paragraph": "Per-ticker deltas were positive on 5 of 5 tickers; ...",
        "limitation_paragraph": "Concentration warnings did not trigger; ...",
        "caveat_phrases_used": ["validation-only", "not holdout-ready"],
        "forbidden_phrases_blocked_at_runtime": [],
        "reproducibility_pointers": [
            {
                "sentence_id": "results-001",
                "artifact": "notebook07_final_validation_comparison.csv",
                "row_filter": "model=lightgbm_winner",
            }
        ],
        "improvement_wording_applied": True,
        "improvement_threshold_check": {
            "delta_macro_f1_vs_dummy_lcb_95": 0.0072,
            "positive_ticker_count": 5,
            "passed_per_AGENTS_md_4_2_5a": True,
        },
    }


def _base_ledger_rows():
    return [
        {
            "artifact": "notebook07_lockfile_scope_gate.json",
            "notebook_stage": "07A",
            "decision_made": "lockfile_signed",
            "decision_timing": "before_official_validation_read",
            "decision_surface": "lockfile",
            "model_families_considered": "lightgbm",
            "profiles_or_trials_considered": "lightgbm_winner",
            "seeds_used": "5",
            "thresholds_or_coverages_considered": "n/a",
            "official_validation_rows_inspected": 0,
            "cumulative_official_validation_inspections_across_notebooks": 0,
            "train_inner_only_decision": True,
            "official_validation_informed_decision": False,
            "diagnostic_only_readout": False,
            "holdout_test_contact": False,
            "allowed_wording": "n/a",
            "forbidden_wording": "n/a",
            "risk_note": "",
            "appended_by_notebook": "07",
            "appended_at_utc": "2026-06-06T00:00:00Z",
        }
    ]


# ---------- thesis_paragraph_kit.json tests --------------------------------


def test_thesis_kit_valid_payload_passes(tmp_path: Path):
    kit_path = tmp_path / "notebook07_thesis_paragraph_kit.json"
    kit_path.write_text(json.dumps(_base_kit()), encoding="utf-8")
    payload = json.loads(kit_path.read_text(encoding="utf-8"))
    validate_thesis_paragraph_kit(payload)  # must not raise


@pytest.mark.parametrize("dropped", sorted(REQUIRED_THESIS_KIT_FIELDS))
def test_thesis_kit_rejects_missing_top_level_field(dropped: str):
    payload = _base_kit()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="thesis_paragraph_kit missing"):
        validate_thesis_paragraph_kit(payload)


@pytest.mark.parametrize("dropped", sorted(REQUIRED_THRESHOLD_CHECK_FIELDS))
def test_thesis_kit_rejects_missing_threshold_check_field(dropped: str):
    payload = _base_kit()
    payload["improvement_threshold_check"].pop(dropped)
    with pytest.raises(AssertionError, match="improvement_threshold_check missing"):
        validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_improvement_wording_without_lcb():
    """improvement_wording_applied=True but lcb < 0.005 violates AGENTS.md §4.2.5a."""
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.003
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = True
    with pytest.raises(AssertionError, match="improvement_wording_applied"):
        validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_improvement_wording_without_breadth():
    payload = _base_kit()
    payload["improvement_threshold_check"]["positive_ticker_count"] = 3
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = True
    with pytest.raises(AssertionError, match="improvement_wording_applied"):
        validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_claim_disagreement():
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.0072
    payload["improvement_threshold_check"]["positive_ticker_count"] = 5
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False  # wrong claim
    with pytest.raises(AssertionError, match="disagrees with measured"):
        validate_thesis_paragraph_kit(payload)


def test_thesis_kit_weak_signal_allows_no_improvement_wording():
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.001
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = False  # honest
    validate_thesis_paragraph_kit(payload)  # must not raise


# ---------- validation_budget_ledger.csv tests -----------------------------


def test_ledger_valid_frame_passes():
    df = pd.DataFrame(_base_ledger_rows())
    validate_ledger_frame(df)


@pytest.mark.parametrize("dropped", sorted(REQUIRED_LEDGER_COLUMNS))
def test_ledger_rejects_missing_column(dropped: str):
    df = pd.DataFrame(_base_ledger_rows()).drop(columns=[dropped])
    with pytest.raises(AssertionError, match="missing columns"):
        validate_ledger_frame(df)


def test_ledger_append_only_monotonic(tmp_path: Path):
    """P1-T06: ledger is append-only; cumulative counter must not regress."""
    ledger_path = tmp_path / "notebook07_validation_budget_ledger.csv"

    df1 = pd.DataFrame(_base_ledger_rows())
    df1.to_csv(ledger_path, index=False)
    n_before = len(pd.read_csv(ledger_path))

    # Simulate 08O appending one row recording its intent to read official val.
    new_row = _base_ledger_rows()[0] | {
        "artifact": "notebook08_run_manifest.json",
        "notebook_stage": "08O",
        "decision_made": "official_validation_readout_intent",
        "decision_timing": "before_official_validation_read",
        "official_validation_rows_inspected": 0,
        "cumulative_official_validation_inspections_across_notebooks": 1,
        "official_validation_informed_decision": False,
        "appended_by_notebook": "08O",
        "appended_at_utc": "2026-06-07T00:00:00Z",
    }
    df_append = pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)
    df_append.to_csv(ledger_path, index=False)
    n_after = len(pd.read_csv(ledger_path))

    assert n_after == n_before + 1, "ledger append produced wrong row count"
    validate_ledger_frame(pd.read_csv(ledger_path))


# ---------- final_validation_comparison.csv tests (§07B contract) ----------


def _base_full_coverage_row() -> dict:
    base = {col: "x" for col in REQUIRED_FINAL_COMPARISON_COLUMNS}
    base["row_class"] = "full_coverage"
    # numeric-typed fields
    for f in (
        "horizon_k", "threshold_bps", "window_size", "seed_count",
        "macro_f1_mean", "macro_f1_std", "macro_f1_lcb_95",
        "balanced_accuracy_mean", "balanced_accuracy_std", "accuracy_mean",
        "dummy_macro_f1_mean", "dummy_balanced_accuracy_mean",
        "delta_macro_f1_vs_dummy_mean", "delta_balanced_accuracy_vs_dummy_mean",
        "always_up_dummy_macro_f1_mean", "delta_macro_f1_vs_always_up_dummy_mean",
        "positive_ticker_count", "top_ticker_gain_share", "validation_n",
    ):
        base[f] = 0.5
    # conditional fields all NA for full_coverage
    for f in CONDITIONAL_FINAL_COMPARISON_COLUMNS:
        base[f] = pd.NA
    return base


def _base_selective_row() -> dict:
    base = _base_full_coverage_row()
    base["row_class"] = "selective"
    # conditional fields all non-NA for selective
    for f in CONDITIONAL_FINAL_COMPARISON_COLUMNS:
        base[f] = 0.5
    return base


def test_final_comparison_valid_full_coverage_passes():
    df = pd.DataFrame([_base_full_coverage_row()])
    validate_final_validation_comparison_frame(df)


def test_final_comparison_valid_selective_passes():
    df = pd.DataFrame([_base_selective_row()])
    validate_final_validation_comparison_frame(df)


def test_final_comparison_mixed_rows_pass():
    df = pd.DataFrame([_base_full_coverage_row(), _base_selective_row()])
    validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_missing_required_column():
    row = _base_full_coverage_row()
    row.pop("row_class")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="REQUIRED columns"):
        validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_missing_conditional_column():
    row = _base_full_coverage_row()
    row.pop("coverage")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="CONDITIONAL columns"):
        validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_invalid_row_class():
    row = _base_full_coverage_row()
    row["row_class"] = "partial"
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="invalid values"):
        validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_selective_with_na_conditional():
    row = _base_selective_row()
    row["coverage"] = pd.NA  # selective MUST have non-NA
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="selective rows must have non-NA"):
        validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_full_coverage_with_value_in_conditional():
    row = _base_full_coverage_row()
    row["coverage"] = 0.6  # full_coverage MUST have NA
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="must have NA in conditional"):
        validate_final_validation_comparison_frame(df)


def test_ledger_rejects_regressing_cumulative_counter():
    rows = _base_ledger_rows()
    rows.append(
        rows[0]
        | {
            "cumulative_official_validation_inspections_across_notebooks": -1,
            "appended_at_utc": "2026-06-07T00:00:00Z",
        }
    )
    df = pd.DataFrame(rows)
    with pytest.raises(AssertionError, match="monotonically non-decreasing"):
        validate_ledger_frame(df)
