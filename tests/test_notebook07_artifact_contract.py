"""Artifact contract for Notebook 07 outputs.

Phase 1 / Task P1-T05..T06 of phased_implementation_plan.md.

Verifies two contracts introduced by the 2026-06-06 top-5 edits:
  P1-T05: notebook07_thesis_paragraph_kit.json schema and AGENTS.md §4.2.5a
          improvement-wording gate.
  P1-T06: notebook07_validation_budget_ledger.csv must carry the new
          cumulative-across-notebooks field; behaviour is append-only.

Inline validators were previously defined in this file. They now live in
``scripts/notebook07_contract.py`` so the N07 colab notebook generator can
inline the same logic for Colab portability.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import notebook07_contract as c


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
    c.validate_thesis_paragraph_kit(payload)  # must not raise


@pytest.mark.parametrize("dropped", sorted(c.REQUIRED_THESIS_KIT_FIELDS))
def test_thesis_kit_rejects_missing_top_level_field(dropped: str):
    payload = _base_kit()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="thesis_paragraph_kit missing"):
        c.validate_thesis_paragraph_kit(payload)


@pytest.mark.parametrize("dropped", sorted(c.REQUIRED_THRESHOLD_CHECK_FIELDS))
def test_thesis_kit_rejects_missing_threshold_check_field(dropped: str):
    payload = _base_kit()
    payload["improvement_threshold_check"].pop(dropped)
    with pytest.raises(AssertionError, match="improvement_threshold_check missing"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_improvement_wording_without_lcb():
    """improvement_wording_applied=True but lcb < 0.005 violates AGENTS.md §4.2.5a."""
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.003
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = True
    with pytest.raises(AssertionError, match="improvement_wording_applied"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_improvement_wording_without_breadth():
    payload = _base_kit()
    payload["improvement_threshold_check"]["positive_ticker_count"] = 3
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = True
    with pytest.raises(AssertionError, match="improvement_wording_applied"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_claim_disagreement():
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.0072
    payload["improvement_threshold_check"]["positive_ticker_count"] = 5
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False  # wrong claim
    with pytest.raises(AssertionError, match="disagrees with measured"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_weak_signal_allows_no_improvement_wording():
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.001
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = False
    payload["improvement_wording_applied"] = False  # honest
    c.validate_thesis_paragraph_kit(payload)  # must not raise


# ---------- validation_budget_ledger.csv tests -----------------------------


def test_ledger_valid_frame_passes():
    df = pd.DataFrame(_base_ledger_rows())
    c.validate_ledger_frame(df)


@pytest.mark.parametrize("dropped", sorted(c.REQUIRED_LEDGER_COLUMNS))
def test_ledger_rejects_missing_column(dropped: str):
    df = pd.DataFrame(_base_ledger_rows()).drop(columns=[dropped])
    with pytest.raises(AssertionError, match="missing columns"):
        c.validate_ledger_frame(df)


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
    c.validate_ledger_frame(pd.read_csv(ledger_path))


# ---------- final_validation_comparison.csv tests (§07B contract) ----------


def _base_full_coverage_row() -> dict:
    base = {col: "x" for col in c.REQUIRED_FINAL_COMPARISON_COLUMNS}
    base["row_class"] = "full_coverage"
    # numeric-typed fields
    for f in (
        "horizon_k", "threshold_bps", "window_size", "seed_count",
        "macro_f1_mean", "macro_f1_std", "macro_f1_lcb_95",
        "balanced_accuracy_mean", "balanced_accuracy_std", "accuracy_mean",
        "dummy_macro_f1_mean", "dummy_balanced_accuracy_mean",
        "delta_macro_f1_vs_dummy_mean", "delta_macro_f1_vs_dummy_lcb_95",
        "delta_balanced_accuracy_vs_dummy_mean",
        "always_up_dummy_macro_f1_mean", "delta_macro_f1_vs_always_up_dummy_mean",
        "positive_ticker_count", "top_ticker_gain_share", "validation_n",
    ):
        base[f] = 0.5
    # conditional fields all NA for full_coverage
    for f in c.CONDITIONAL_FINAL_COMPARISON_COLUMNS:
        base[f] = pd.NA
    return base


def _base_selective_row() -> dict:
    base = _base_full_coverage_row()
    base["row_class"] = "selective"
    # conditional fields all non-NA for selective
    for f in c.CONDITIONAL_FINAL_COMPARISON_COLUMNS:
        base[f] = 0.5
    return base


def test_final_comparison_valid_full_coverage_passes():
    df = pd.DataFrame([_base_full_coverage_row()])
    c.validate_final_validation_comparison_frame(df)


def test_final_comparison_valid_selective_passes():
    df = pd.DataFrame([_base_selective_row()])
    c.validate_final_validation_comparison_frame(df)


def test_final_comparison_mixed_rows_pass():
    df = pd.DataFrame([_base_full_coverage_row(), _base_selective_row()])
    c.validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_missing_required_column():
    row = _base_full_coverage_row()
    row.pop("row_class")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="REQUIRED columns"):
        c.validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_missing_conditional_column():
    row = _base_full_coverage_row()
    row.pop("coverage")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="CONDITIONAL columns"):
        c.validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_invalid_row_class():
    row = _base_full_coverage_row()
    row["row_class"] = "partial"
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="invalid values"):
        c.validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_selective_with_na_conditional():
    row = _base_selective_row()
    row["coverage"] = pd.NA  # selective MUST have non-NA
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="selective rows must have non-NA"):
        c.validate_final_validation_comparison_frame(df)


def test_final_comparison_rejects_full_coverage_with_value_in_conditional():
    row = _base_full_coverage_row()
    row["coverage"] = 0.6  # full_coverage MUST have NA
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="must have NA in conditional"):
        c.validate_final_validation_comparison_frame(df)


# ---------- same-row dummy hard-stop (P1 from Phase B review) --------------


@pytest.mark.parametrize("nan_col", sorted(c.SAME_ROW_DUMMY_REQUIRED_NON_NULL_COLUMNS))
def test_final_comparison_rejects_full_coverage_with_nan_dummy_metric(nan_col: str):
    """AGENTS.md §4.2 requires same-row stratified dummy on every model row.
    Silent NaN would let a row reach thesis wording without a baseline.
    """
    row = _base_full_coverage_row()
    row[nan_col] = float("nan")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="same-row dummy baseline is missing"):
        c.validate_final_validation_comparison_frame(df)


@pytest.mark.parametrize("nan_col", sorted(c.SAME_ROW_DUMMY_REQUIRED_NON_NULL_COLUMNS))
def test_final_comparison_rejects_selective_with_nan_dummy_metric(nan_col: str):
    """Every selective row needs same-row stratified dummy on retained rows."""
    row = _base_selective_row()
    row[nan_col] = float("nan")
    df = pd.DataFrame([row])
    with pytest.raises(AssertionError, match="same-row dummy baseline is missing"):
        c.validate_final_validation_comparison_frame(df)


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
        c.validate_ledger_frame(df)


# ---------- validate_ledger_prefix_invariance tests (append-only contract) -


def _appended_ledger_row(stage: str, when: str, cumulative: int) -> dict:
    base = _base_ledger_rows()[0]
    return base | {
        "notebook_stage": stage,
        "appended_at_utc": when,
        "appended_by_notebook": stage,
        "cumulative_official_validation_inspections_across_notebooks": cumulative,
    }


def test_prefix_invariance_accepts_strict_append():
    existing = pd.DataFrame(_base_ledger_rows())
    new = pd.DataFrame(
        _base_ledger_rows()
        + [_appended_ledger_row("07B", "2026-06-07T00:00:00Z", 1)]
    )
    c.validate_ledger_prefix_invariance(existing, new)  # must not raise


def test_prefix_invariance_accepts_empty_existing():
    new = pd.DataFrame(_base_ledger_rows())
    c.validate_ledger_prefix_invariance(pd.DataFrame(columns=new.columns), new)


def test_prefix_invariance_rejects_modified_existing_row():
    existing = pd.DataFrame(_base_ledger_rows())
    new_rows = _base_ledger_rows()
    new_rows[0] = new_rows[0] | {"risk_note": "RETROACTIVELY EDITED"}
    new_rows.append(_appended_ledger_row("07B", "2026-06-07T00:00:00Z", 1))
    new = pd.DataFrame(new_rows)
    with pytest.raises(AssertionError, match="prefix invariance violated"):
        c.validate_ledger_prefix_invariance(existing, new)


def test_prefix_invariance_rejects_dropped_existing_row():
    existing_rows = _base_ledger_rows() + [_appended_ledger_row("07B", "2026-06-07T00:00:00Z", 1)]
    existing = pd.DataFrame(existing_rows)
    new = pd.DataFrame(_base_ledger_rows())  # dropped the 07B row
    with pytest.raises(AssertionError, match="lost rows"):
        c.validate_ledger_prefix_invariance(existing, new)


def test_prefix_invariance_rejects_reordered_existing_rows():
    base = _base_ledger_rows()[0]
    existing_rows = [
        base | {"notebook_stage": "07A", "appended_at_utc": "2026-06-06T00:00:00Z", "risk_note": "ROW_A"},
        base | {"notebook_stage": "07B", "appended_at_utc": "2026-06-07T00:00:00Z", "risk_note": "ROW_B",
                "cumulative_official_validation_inspections_across_notebooks": 1},
    ]
    existing = pd.DataFrame(existing_rows)
    swapped_rows = [
        base | {"notebook_stage": "07A", "appended_at_utc": "2026-06-06T00:00:00Z", "risk_note": "ROW_B"},
        base | {"notebook_stage": "07B", "appended_at_utc": "2026-06-07T00:00:00Z", "risk_note": "ROW_A",
                "cumulative_official_validation_inspections_across_notebooks": 1},
    ]
    new = pd.DataFrame(swapped_rows)
    with pytest.raises(AssertionError, match="prefix invariance violated"):
        c.validate_ledger_prefix_invariance(existing, new)


def test_prefix_invariance_rejects_pure_row_reorder():
    """Round 5 regression lock: a pure reorder of existing rows (same
    content per row, swapped row positions) must be rejected.

    The earlier validator sorted both sides on (appended_at_utc,
    notebook_stage, appended_by_notebook) before comparing, so a pure
    reorder was silently re-aligned to canonical order and accepted.
    AGENTS.md §4.3 forbids "modified, dropped, or reordered" — reorder
    alone must trip the contract.
    """
    base = _base_ledger_rows()[0]
    row_a = base | {
        "notebook_stage": "07A",
        "appended_at_utc": "2026-06-06T00:00:00Z",
        "decision_made": "lockfile_intent",
        "appended_by_notebook": "07A",
    }
    row_b = base | {
        "notebook_stage": "07B",
        "appended_at_utc": "2026-06-07T00:00:00Z",
        "decision_made": "comparison_intent",
        "appended_by_notebook": "07B",
        "cumulative_official_validation_inspections_across_notebooks": 1,
    }
    existing = pd.DataFrame([row_a, row_b])
    # Same two rows, no content modification, just swapped positions.
    reordered = pd.DataFrame([row_b, row_a])
    with pytest.raises(AssertionError, match="prefix invariance violated"):
        c.validate_ledger_prefix_invariance(existing, reordered)


def test_prefix_invariance_rejects_missing_existing_column():
    """Round 6 P2 regression lock: dropping a column that existed on disk
    must be rejected even if all remaining (shared) cells line up. Earlier
    revision intersected on shared columns and silently accepted column
    drops."""
    base = _base_ledger_rows()[0]
    existing = pd.DataFrame([base])
    new_dropped = existing.drop(columns=["risk_note"]).copy()
    with pytest.raises(AssertionError, match="column set changed"):
        c.validate_ledger_prefix_invariance(existing, new_dropped)


def test_prefix_invariance_rejects_extra_new_column():
    """Round 6 P2 regression lock: adding a column not in the on-disk
    ledger must be rejected. Earlier revision silently accepted extras."""
    base = _base_ledger_rows()[0]
    existing = pd.DataFrame([base])
    new_with_extra = existing.copy()
    new_with_extra["sneaky_new_column"] = "added"
    with pytest.raises(AssertionError, match="column set changed"):
        c.validate_ledger_prefix_invariance(existing, new_with_extra)


def test_prefix_invariance_rejects_column_reorder():
    """Round 6 P2 regression lock: same column set but different column
    order must be rejected. Earlier revision intersected then sorted
    column names and silently re-aligned reorders."""
    base = _base_ledger_rows()[0]
    existing = pd.DataFrame([base])
    cols = list(existing.columns)
    if len(cols) < 2:
        pytest.skip("ledger schema too small to reorder")
    # Swap the first two columns to construct a pure reorder.
    swapped_cols = [cols[1], cols[0]] + cols[2:]
    new_reordered = existing[swapped_cols].copy()
    with pytest.raises(AssertionError, match="column order changed"):
        c.validate_ledger_prefix_invariance(existing, new_reordered)


# ---------- parse_label_config tests ---------------------------------------


def test_parse_label_config_h03_bps1p5():
    parsed = c.parse_label_config("h03_bps1p5")
    assert parsed == {"horizon_k": 3, "threshold_bps": 1.5}


def test_parse_label_config_h12_bps10p25():
    parsed = c.parse_label_config("h12_bps10p25")
    assert parsed["horizon_k"] == 12
    assert parsed["threshold_bps"] == 10.25


@pytest.mark.parametrize("bad", ["h03_bps1", "h03bps1p5", "horizon=3", "", "  "])
def test_parse_label_config_rejects_malformed(bad: str):
    with pytest.raises(ValueError, match="Cannot parse label_config"):
        c.parse_label_config(bad)


# ---------- forbidden-phrase scan in thesis_paragraph_kit ------------------


def test_thesis_kit_rejects_forbidden_phrase_in_results_paragraph():
    payload = _base_kit()
    payload["results_paragraph"] = (
        "Under locked validation, the model is production-grade and tradable."
    )
    with pytest.raises(AssertionError, match="leaked forbidden phrases"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_forbidden_phrase_in_robustness_paragraph():
    payload = _base_kit()
    payload["robustness_paragraph"] = "Sharpe-aligned per-ticker positivity holds."
    with pytest.raises(AssertionError, match="leaked forbidden phrases"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_rejects_forbidden_phrase_in_limitation_paragraph():
    payload = _base_kit()
    payload["limitation_paragraph"] = "The model is final pending deployment readiness."
    with pytest.raises(AssertionError, match="leaked forbidden phrases"):
        c.validate_thesis_paragraph_kit(payload)


def test_thesis_kit_accepts_neutral_paragraphs():
    payload = _base_kit()
    # base kit uses neutral phrasing (no forbidden tokens) -> must not raise
    c.validate_thesis_paragraph_kit(payload)


# ---------- LCB vs mean disagreement gate (P1 #1) --------------------------


def test_thesis_kit_rejects_lcb_below_threshold_even_when_mean_above():
    """A paper-pressure rewrite that uses delta mean instead of LCB must fail.

    AGENTS.md §4.2.5a is unambiguous: only the one-sided 95% LCB counts. Even
    if the mean is 0.006 (above 0.005), an LCB of 0.003 must NOT clear the
    improvement-wording gate.
    """
    payload = _base_kit()
    payload["improvement_threshold_check"]["delta_macro_f1_vs_dummy_lcb_95"] = 0.003
    payload["improvement_threshold_check"]["positive_ticker_count"] = 5
    payload["improvement_threshold_check"]["passed_per_AGENTS_md_4_2_5a"] = True  # paper claim
    payload["improvement_wording_applied"] = True
    with pytest.raises(AssertionError, match="disagrees with measured"):
        c.validate_thesis_paragraph_kit(payload)
