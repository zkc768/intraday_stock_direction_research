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
