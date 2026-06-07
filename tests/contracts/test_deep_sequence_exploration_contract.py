"""Artifact contract for Notebook 08 outputs.

Phase 1 / Task P1-T10 of phased_implementation_plan.md.

Verifies contracts introduced by the 2026-06-06 top-5 edits:
  - 08F entry gate requires `dmc_attestation.json` OR a recorded
    separate-session attestation (Patch #2, ENHANCE-005).
  - 08O step 0 requires appending to the project-level
    `notebook07_validation_budget_ledger.csv` BEFORE reading any
    official-validation metric (Patch #4b, ENHANCE-012 + CN-003).
  - 08F freeze record schema must carry primary_candidate_id and
    paper_safe_score (existing design §13.2) plus the booleans the
    08O gate inspects.

Validators live in ``src/intraday_research/contracts/deep_sequence_exploration.py``;
``scripts/notebook08_contract.py`` is a re-export shim retained for legacy
generator/notebook callers (see ``docs/LEGACY_NAME_MAPPING.md``).
"""

from pathlib import Path

import pandas as pd
import pytest

from intraday_research.contracts import deep_sequence_exploration as c


# Re-export under the names the test bodies originally used, so the existing
# ~50 test cases continue to read naturally without rewriting their bodies.
REQUIRED_DMC_FIELDS = c.REQUIRED_DMC_FIELDS
REQUIRED_FREEZE_RECORD_FIELDS = c.REQUIRED_FREEZE_RECORD_FIELDS
REQUIRED_SEPARATE_SESSION_ATTESTATION_FIELDS = (
    c.REQUIRED_SEPARATE_SESSION_ATTESTATION_FIELDS
)
FORBIDDEN_FALLBACK_RULE_SUBSTRINGS = c.FORBIDDEN_FALLBACK_RULE_SUBSTRINGS
FORBIDDEN_FALLBACK_PATTERNS_NORMALIZED = c.FORBIDDEN_FALLBACK_PATTERNS_NORMALIZED
_normalize_fallback_rule = c._normalize_fallback_rule
validate_dmc_attestation = c.validate_dmc_attestation
validate_08f_entry = c.validate_08f_entry
validate_freeze_record = c.validate_freeze_record
validate_08o_ledger_append_precedes_read = c.validate_08o_ledger_append_precedes_read
validate_separate_session_attestation = c.validate_separate_session_attestation


# ---------- Fixtures --------------------------------------------------------


def _valid_dmc() -> dict:
    return {
        "dmc_role": "data_monitoring_committee_proxy",
        "reviewer_identifier": "reviewer-A",
        "reviewed_08x_run_manifest_sha256": "a" * 64,
        "reviewed_at_utc": "2026-06-07T12:34:56Z",
        "attestation_statement": "Reviewed 08X output; no official-validation contact detected.",
    }


def _valid_separate_session_attestation() -> dict:
    return {
        "attestation_kind": "separate_colab_session_by_non_08x_author",
        "reviewer_identifier": "reviewer-B",
        "reviewed_08x_run_manifest_sha256": "b" * 64,
        "attested_at_utc": "2026-06-07T13:00:00Z",
        "attestation_statement": (
            "08F runs in a fresh Colab session opened by a reviewer who did "
            "not author 08X; no in-memory 08X state is reused."
        ),
    }


def _valid_freeze_record() -> dict:
    return {
        "stage": "08F",
        "scope": "diagnostic",
        "primary_candidate_id": "ms_dlinear_tcn_v3",
        "fallback_candidate_id": "dlinear_only_v1",
        "fallback_activation_rule": (
            "Activate fallback only if primary training produces NaN before scoring official validation."
        ),
        "config_hash": "deadbeef" * 4,
        "architecture_family": "ms_dlinear_tcn",
        "frozen_architecture_params": {"channels": [32, 32, 32], "kernel_size": 3},
        "frozen_loss": "cross_entropy",
        "frozen_hpo_method": "asha",
        "frozen_seed_list": [260501, 260502, 260503],
        "frozen_metric_list": ["macro_f1", "balanced_accuracy"],
        "frozen_wording_rule": "per AGENTS.md §4.2.5a",
        "paper_safe_score": 0.42,
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
    }


def _ledger_row(stage: str, cumulative: int, utc: str) -> dict:
    return {
        "artifact": f"notebook{stage.lower()}_run_manifest.json",
        "notebook_stage": stage,
        "decision_made": "official_validation_readout_intent" if stage == "08O" else "freeze_signed",
        "decision_timing": "before_official_validation_read",
        "decision_surface": "manifest",
        "model_families_considered": "ms_dlinear_tcn",
        "profiles_or_trials_considered": "primary",
        "seeds_used": "3",
        "thresholds_or_coverages_considered": "n/a",
        "official_validation_rows_inspected": 0,
        "cumulative_official_validation_inspections_across_notebooks": cumulative,
        "train_inner_only_decision": stage != "08O",
        "official_validation_informed_decision": False,
        "diagnostic_only_readout": False,
        "holdout_test_contact": False,
        "allowed_wording": "n/a",
        "forbidden_wording": "n/a",
        "risk_note": "",
        "appended_by_notebook": stage,
        "appended_at_utc": utc,
    }


# ---------- DMC attestation tests ------------------------------------------


def test_dmc_valid_payload_passes():
    validate_dmc_attestation(_valid_dmc())


@pytest.mark.parametrize("dropped", sorted(REQUIRED_DMC_FIELDS))
def test_dmc_rejects_missing_field(dropped: str):
    payload = _valid_dmc()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="missing fields"):
        validate_dmc_attestation(payload)


def test_dmc_rejects_wrong_role():
    payload = _valid_dmc()
    payload["dmc_role"] = "self_attestation"
    with pytest.raises(AssertionError, match="dmc_role"):
        validate_dmc_attestation(payload)


def test_dmc_rejects_non_sha256_manifest_hash():
    payload = _valid_dmc()
    payload["reviewed_08x_run_manifest_sha256"] = "deadbeef"  # too short
    with pytest.raises(AssertionError, match="hex sha256"):
        validate_dmc_attestation(payload)


# ---------- 08F entry gate tests -------------------------------------------
# Round 7 finding #3 (2026-06-06) tightened the gate: separate session WITHOUT
# DMC AND WITHOUT a positive separate_session_attestation now fails. The
# legacy laxer behavior (separate-session-by-omission) is gone. Tests below
# encode the new contract.


def test_08f_entry_accepts_separate_session_with_attestation_and_no_dmc():
    """Round 7 #3 -- separate session is OK when accompanied by a positive
    attestation file; absent flag is no longer proof."""
    validate_08f_entry(
        dmc_attestation=None,
        same_session_as_08x=False,
        separate_session_attestation=_valid_separate_session_attestation(),
    )


def test_08f_entry_accepts_separate_session_with_dmc_only():
    """Round 7 #3 -- when same_session_as_08x=False, a valid DMC alone is
    sufficient (a non-08X author signed off; that is at least as strong as
    a self-attested separate session)."""
    validate_08f_entry(
        dmc_attestation=_valid_dmc(),
        same_session_as_08x=False,
        separate_session_attestation=None,
    )


def test_08f_entry_rejects_separate_session_without_any_attestation():
    """Round 7 #3 -- the new contract: separate-session-by-omission fails."""
    with pytest.raises(AssertionError, match="separate_session_attestation"):
        validate_08f_entry(
            dmc_attestation=None,
            same_session_as_08x=False,
            separate_session_attestation=None,
        )


def test_08f_entry_accepts_dmc_in_same_session():
    validate_08f_entry(
        dmc_attestation=_valid_dmc(),
        same_session_as_08x=True,
    )


def test_08f_entry_rejects_same_session_without_dmc():
    with pytest.raises(AssertionError, match="dmc_attestation"):
        validate_08f_entry(dmc_attestation=None, same_session_as_08x=True)


def test_08f_entry_with_invalid_dmc_fails():
    bad_dmc = _valid_dmc()
    bad_dmc.pop("attestation_statement")
    with pytest.raises(AssertionError, match="missing fields"):
        validate_08f_entry(dmc_attestation=bad_dmc, same_session_as_08x=True)


# ---------- separate_session_attestation tests (Round 7 #3) ----------------


def test_separate_session_attestation_valid_payload_passes():
    validate_separate_session_attestation(_valid_separate_session_attestation())


@pytest.mark.parametrize(
    "dropped", sorted(REQUIRED_SEPARATE_SESSION_ATTESTATION_FIELDS)
)
def test_separate_session_attestation_rejects_missing_field(dropped: str):
    payload = _valid_separate_session_attestation()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="missing fields"):
        validate_separate_session_attestation(payload)


def test_separate_session_attestation_rejects_wrong_kind():
    payload = _valid_separate_session_attestation()
    payload["attestation_kind"] = "self_attested_separate_session"
    with pytest.raises(AssertionError, match="attestation_kind"):
        validate_separate_session_attestation(payload)


def test_separate_session_attestation_rejects_non_sha256_manifest_hash():
    payload = _valid_separate_session_attestation()
    payload["reviewed_08x_run_manifest_sha256"] = "deadbeef"  # too short
    with pytest.raises(AssertionError, match="hex sha256"):
        validate_separate_session_attestation(payload)


def test_08f_entry_with_invalid_separate_session_attestation_fails():
    bad_attestation = _valid_separate_session_attestation()
    bad_attestation.pop("attestation_statement")
    with pytest.raises(AssertionError, match="missing fields"):
        validate_08f_entry(
            dmc_attestation=None,
            same_session_as_08x=False,
            separate_session_attestation=bad_attestation,
        )


# ---------- Freeze record tests --------------------------------------------


def test_freeze_record_valid_payload_passes():
    validate_freeze_record(_valid_freeze_record())


def test_freeze_record_rejects_holdout_authorized():
    payload = _valid_freeze_record()
    payload["holdout_test_authorized"] = True
    with pytest.raises(AssertionError, match="holdout_test_authorized"):
        validate_freeze_record(payload)


def test_freeze_record_rejects_official_validation_selection():
    payload = _valid_freeze_record()
    payload["official_validation_used_for_selection"] = True
    with pytest.raises(AssertionError, match="official_validation_used_for_selection"):
        validate_freeze_record(payload)


def test_freeze_record_rejects_fallback_rule_referencing_official_val():
    payload = _valid_freeze_record()
    payload["fallback_activation_rule"] = (
        "if primary scores worse on official validation, switch to fallback"
    )
    with pytest.raises(AssertionError, match="fallback_activation_rule references"):
        validate_freeze_record(payload)


@pytest.mark.parametrize("dropped", sorted(REQUIRED_FREEZE_RECORD_FIELDS))
def test_freeze_record_rejects_missing_field(dropped: str):
    payload = _valid_freeze_record()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="missing fields"):
        validate_freeze_record(payload)


# ---------- 08O ledger-append-before-read tests ----------------------------


def test_08o_ledger_append_increments_counter(tmp_path: Path):
    """08O must append a new row with an incremented cumulative counter."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    after = pd.concat(
        [before, pd.DataFrame([_ledger_row("08O", 1, "2026-06-07T00:00:00Z")])],
        ignore_index=True,
    )
    validate_08o_ledger_append_precedes_read(
        ledger_before_read=before, ledger_after_read=after
    )


def test_08o_missing_append_is_rejected():
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    after = before.copy()  # no append happened
    with pytest.raises(AssertionError, match="must append exactly 1 row"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_append_without_counter_increment_is_rejected():
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    # Append a row but forget to bump the cumulative counter.
    after = pd.concat(
        [before, pd.DataFrame([_ledger_row("08O", 0, "2026-06-07T00:00:00Z")])],
        ignore_index=True,
    )
    with pytest.raises(AssertionError, match=r"must be \+1 from previous max"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_first_inspection_in_empty_ledger():
    """Edge case: if 07 never ran, 08O is the first appender; counter goes 0→1."""
    before = pd.DataFrame(columns=[
        "cumulative_official_validation_inspections_across_notebooks",
    ])
    after = pd.DataFrame([_ledger_row("08O", 1, "2026-06-07T00:00:00Z")])
    validate_08o_ledger_append_precedes_read(
        ledger_before_read=before, ledger_after_read=after
    )


# ---------- Strengthened 08O ledger tests (review-finding follow-up) -------


def test_08o_rejects_multi_row_append():
    """08O must append exactly 1 row, not multiple silently."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    after = pd.concat(
        [
            before,
            pd.DataFrame(
                [
                    _ledger_row("08O", 1, "2026-06-07T00:00:00Z"),
                    _ledger_row("08O", 2, "2026-06-07T00:01:00Z"),
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(AssertionError, match="must append exactly 1 row"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_wrong_appended_by_notebook():
    """Append tagged as 08F (or anything ≠ 08O) must fail."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    wrong = _ledger_row("08F", 1, "2026-06-07T00:00:00Z")  # wrong stage tag
    after = pd.concat([before, pd.DataFrame([wrong])], ignore_index=True)
    with pytest.raises(AssertionError, match="appended_by_notebook must be '08O'"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_wrong_decision_timing():
    """Append with decision_timing != 'before_official_validation_read' fails."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    new["decision_timing"] = "after_official_validation_read"
    after = pd.concat([before, pd.DataFrame([new])], ignore_index=True)
    with pytest.raises(AssertionError, match="decision_timing must be"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_non_zero_inspected_at_intent():
    """At intent time, official_validation_rows_inspected MUST be 0."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    new["official_validation_rows_inspected"] = 17  # lies about not-yet-read
    after = pd.concat([before, pd.DataFrame([new])], ignore_index=True)
    with pytest.raises(AssertionError, match="must be 0 at append time"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_counter_jump_by_more_than_one():
    """Counter must increment exactly +1, not skip values."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    jumped = _ledger_row("08O", 5, "2026-06-07T00:00:00Z")  # 0 → 5
    after = pd.concat([before, pd.DataFrame([jumped])], ignore_index=True)
    with pytest.raises(AssertionError, match=r"must be \+1 from previous max"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


# ---------- Append-only prefix invariance tests (review-finding v2) --------


def test_08o_rejects_tampered_prefix_row_value():
    """Append-only invariance: a silent edit to an existing row before append
    must be detected even if the new row itself is well-formed."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    tampered_prefix = before.copy()
    # Rewrite an audit-trail field on the prior row to hide history.
    tampered_prefix.iloc[
        0, tampered_prefix.columns.get_loc("official_validation_rows_inspected")
    ] = 999
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    after = pd.concat([tampered_prefix, pd.DataFrame([new])], ignore_index=True)
    with pytest.raises(AssertionError, match="not append-only"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_tampered_prefix_audit_string():
    """Even string-field edits (e.g. risk_note) on prior rows are forbidden."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    tampered_prefix = before.copy()
    tampered_prefix.iloc[
        0, tampered_prefix.columns.get_loc("risk_note")
    ] = "rewritten retroactively"
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    after = pd.concat([tampered_prefix, pd.DataFrame([new])], ignore_index=True)
    with pytest.raises(AssertionError, match="not append-only"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_rejects_prefix_column_added_silently():
    """Inserting a new column into the prefix region (vs. ledger_before_read)
    is a schema drift that must fail the invariance check."""
    before = pd.DataFrame([_ledger_row("07A", 0, "2026-06-06T00:00:00Z")])
    tampered_prefix = before.copy()
    tampered_prefix["sneaky_new_column"] = "added_silently"
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    new["sneaky_new_column"] = "matches"  # make concat shapes work
    after = pd.concat([tampered_prefix, pd.DataFrame([new])], ignore_index=True)
    with pytest.raises(AssertionError, match="column set or order changed"):
        validate_08o_ledger_append_precedes_read(
            ledger_before_read=before, ledger_after_read=after
        )


def test_08o_accepts_multi_row_prefix_unchanged():
    """Positive control: when prior ledger has multiple rows and the append
    leaves all of them byte-identical, validation succeeds."""
    before = pd.DataFrame(
        [
            _ledger_row("07A", 0, "2026-06-06T00:00:00Z"),
            _ledger_row("07C", 0, "2026-06-06T01:00:00Z"),
            _ledger_row("07G", 0, "2026-06-06T02:00:00Z"),
        ]
    )
    new = _ledger_row("08O", 1, "2026-06-07T00:00:00Z")
    after = pd.concat([before, pd.DataFrame([new])], ignore_index=True)
    validate_08o_ledger_append_precedes_read(
        ledger_before_read=before, ledger_after_read=after
    )


# ---------- Strengthened fallback-rule tests (review-finding follow-up) ----


@pytest.mark.parametrize(
    "rule",
    [
        # snake_case identifiers caught by substring layer
        "trigger if official_validation_macro_f1 < 0.5",
        "trigger if official_validation_delta < 0",
        "fallback when official_val_score is bad",
        "fallback when official_validation_balanced_accuracy < 0.5",
        "fallback when official_val_metric < 0.5",
        # prose with various separators caught by normalized regex layer
        "fallback if delta_macro_f1 on official val is < 0.005",
        "fallback when official-validation balanced accuracy is below threshold",
        "fallback if official validation macro_f1 underperforms",
        "fallback if primary scores worse on official validation",
        "fallback if primary performs poorly on official-val",
        "fallback if primary fails to beat last_step",
        "fallback if primary loses to fallback",
        "fallback if primary outperformed by last_step_lightgbm_control",
        "fallback if primary cannot beat the deep baseline",
    ],
)
def test_freeze_record_rejects_various_official_val_fallback_phrasings(rule: str):
    payload = _valid_freeze_record()
    payload["fallback_activation_rule"] = rule
    with pytest.raises(AssertionError, match="fallback_activation_rule"):
        validate_freeze_record(payload)


@pytest.mark.parametrize(
    "rule",
    [
        # Legitimate triggers per N08 design §9.3 — must NOT be flagged.
        "Activate fallback only if primary training produces NaN before scoring official validation.",
        "fallback when primary implementation cannot reproduce train-inner checksum",
        "fallback when primary model fails deterministic shape/static gate",
        "fallback when primary artifact contract fails before official validation is read",
        "fallback when primary serialization is corrupted before any validation read",
    ],
)
def test_freeze_record_accepts_legitimate_fallback_phrasings(rule: str):
    payload = _valid_freeze_record()
    payload["fallback_activation_rule"] = rule
    validate_freeze_record(payload)  # must not raise


# ===========================================================================
# Round 7 follow-up: unit tests for validators introduced in this iteration.
#
# These tests cover the contract surfaces that were only exercised indirectly
# by the generator at notebook-build time. They directly answer the Round 7
# reviewer's "contract self-consistent" concern by proving each validator
# accepts well-formed payloads and rejects each specific failure mode.
# ===========================================================================


# ---------- §7 + §11 08x_search_space validator -----------------------------


def _valid_search_space() -> dict:
    return {
        "search_space_version": "2026-06-06-mvp",
        "stage": "08X",
        "scope": "validation_only",
        "architecture_families": [
            "last_step_lightgbm_control",
            "ms_dlinear_tcn",
        ],
        "per_family_trial_budget": {
            "last_step_lightgbm_control": 5,
            "ms_dlinear_tcn": 5,
        },
        "hpo_method": "random_search",
        "eligibility_thresholds": {
            "min_train_inner_lcb_delta_macro_f1": 0.005,
        },
        "scientific_budget_cap_total_trials": 50,
        "fusion_min_lcb_advantage_over_components": 0.003,
        "low_compute_mode": False,
        "low_compute_submode": "",
        "seed_list": [260501, 260502, 260503],
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }


def test_search_space_valid_passes():
    c.validate_08x_search_space(_valid_search_space())


def test_search_space_rejects_official_validation_used_True():
    payload = _valid_search_space()
    payload["official_validation_used"] = True
    with pytest.raises(AssertionError, match="official_validation_used"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_holdout_test_authorized_True():
    payload = _valid_search_space()
    payload["holdout_test_authorized"] = True
    with pytest.raises(AssertionError, match="holdout_test_authorized"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_missing_official_validation_used_field():
    """Round 7 #4 -- the contract owns the guard; absent flag must fail."""
    payload = _valid_search_space()
    payload.pop("official_validation_used")
    with pytest.raises(AssertionError, match="official_validation_used"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_missing_holdout_test_authorized_field():
    payload = _valid_search_space()
    payload.pop("holdout_test_authorized")
    with pytest.raises(AssertionError, match="holdout_test_authorized"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_unknown_family():
    payload = _valid_search_space()
    payload["architecture_families"] = ["random_forest_classifier"]
    payload["per_family_trial_budget"] = {"random_forest_classifier": 5}
    with pytest.raises(AssertionError, match="architecture_families"):
        c.validate_08x_search_space(payload)


@pytest.mark.parametrize("family", ["shallow_gru", "shallow_lstm"])
def test_search_space_rejects_unfrozen_recurrent_families(family: str):
    """GRU/LSTM are section-7.1 candidates, but not 08X-search eligible yet.

    They have no frozen axis block in the stage config/search-space surface, so
    listing either family before the 08X harness mirrors and sha-stamps those
    axes must fail loud instead of producing a silent no-axis trial.
    """
    payload = _valid_search_space()
    payload["architecture_families"] = [family]
    payload["per_family_trial_budget"] = {family: 5}
    with pytest.raises(AssertionError, match="not 08X search-eligible"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_empty_family_list():
    payload = _valid_search_space()
    payload["architecture_families"] = []
    with pytest.raises(AssertionError, match="non-empty list"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_invalid_hpo_method():
    payload = _valid_search_space()
    payload["hpo_method"] = "grid_search"  # not in HPO_METHODS
    with pytest.raises(AssertionError, match="hpo_method"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_non_numeric_eligibility_margin():
    payload = _valid_search_space()
    payload["eligibility_thresholds"]["min_train_inner_lcb_delta_macro_f1"] = "tbd"
    with pytest.raises(AssertionError, match="min_train_inner_lcb_delta_macro_f1"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_budget_cap_over_design_limit():
    """§5.5 caps total at 250. Exceeding it is a contract violation that
    silently widening the search space cannot bypass."""
    payload = _valid_search_space()
    payload["scientific_budget_cap_total_trials"] = (
        c.TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES + 1
    )
    with pytest.raises(AssertionError, match="exceeds §5.5 cap"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_negative_budget_cap():
    payload = _valid_search_space()
    payload["scientific_budget_cap_total_trials"] = -10
    with pytest.raises(AssertionError, match="positive int"):
        c.validate_08x_search_space(payload)


def test_search_space_rejects_per_family_budget_missing_one_family():
    payload = _valid_search_space()
    payload["architecture_families"] = ["ms_dlinear_tcn", "tcn_only"]
    payload["per_family_trial_budget"] = {"ms_dlinear_tcn": 5}  # tcn_only missing
    with pytest.raises(AssertionError, match="per_family_trial_budget missing"):
        c.validate_08x_search_space(payload)


def test_search_space_low_compute_mode_rejects_unknown_submode():
    payload = _valid_search_space()
    payload["low_compute_mode"] = True
    payload["low_compute_submode"] = "magic_aggregation"  # not in enum
    with pytest.raises(AssertionError, match="low_compute_submode"):
        c.validate_08x_search_space(payload)


# ---------- §7.9 sub-mode B nested-fold protocol ----------------------------


def _valid_submode_b_search_space() -> dict:
    ss = _valid_search_space()
    ss["low_compute_mode"] = True
    ss["low_compute_submode"] = "train_inner_oof_mlp_head"
    ss["outer_fold_scheme"] = "purged_time_series_folds"
    ss["outer_fold_k"] = 5
    ss["inner_fold_k_for_head"] = 5
    ss["head_train_data_source"] = (
        "outer_fold_i.oof_predictions_excluding_held_out_inner_fold"
    )
    ss["head_eval_data_source"] = (
        "outer_fold_i.oof_predictions_from_held_out_inner_fold"
    )
    return ss


def test_low_compute_submode_b_valid_protocol_passes():
    c.validate_08x_search_space(_valid_submode_b_search_space())


def test_low_compute_submode_b_rejects_missing_nested_fold_fields():
    ss = _valid_submode_b_search_space()
    ss.pop("outer_fold_k")
    with pytest.raises(AssertionError, match="outer_fold_k"):
        c.validate_08x_search_space(ss)


def test_low_compute_submode_b_rejects_outer_fold_k_below_5():
    ss = _valid_submode_b_search_space()
    ss["outer_fold_k"] = 3
    with pytest.raises(AssertionError, match="outer_fold_k"):
        c.validate_08x_search_space(ss)


def test_low_compute_submode_b_rejects_inner_fold_k_below_5():
    ss = _valid_submode_b_search_space()
    ss["inner_fold_k_for_head"] = 4
    with pytest.raises(AssertionError, match="inner_fold_k_for_head"):
        c.validate_08x_search_space(ss)


def test_low_compute_submode_b_rejects_wrong_outer_fold_scheme():
    ss = _valid_submode_b_search_space()
    ss["outer_fold_scheme"] = "random_kfold"  # not in allowed list
    with pytest.raises(AssertionError, match="outer_fold_scheme"):
        c.validate_08x_search_space(ss)


def test_low_compute_submode_b_rejects_wrong_head_train_source():
    """Closes the classical stacking-leak path: head must train on the
    OUTER OOF predictions EXCLUDING the held-out inner fold."""
    ss = _valid_submode_b_search_space()
    ss["head_train_data_source"] = "outer_fold_i.oof_predictions_all"  # leaks
    with pytest.raises(AssertionError, match="head_train_data_source"):
        c.validate_08x_search_space(ss)


def test_low_compute_submode_b_rejects_wrong_head_eval_source():
    ss = _valid_submode_b_search_space()
    ss["head_eval_data_source"] = "outer_fold_i.oof_predictions_all"
    with pytest.raises(AssertionError, match="head_eval_data_source"):
        c.validate_08x_search_space(ss)


# ---------- §8.3 trial_ledger_frame validator -------------------------------


def _valid_trial_row(**overrides) -> dict:
    row = {
        "trial_id": "last_step_lightgbm_control::deadbeef::fold0::seed260501",
        "candidate_family": "last_step_lightgbm_control",
        "candidate_id": "last_step_lightgbm_control_deadbeef",
        "config_hash": "deadbeef" * 8,
        "fold_id": 0,
        "seed": 260501,
        "budget_tier": "quick",
        "max_epochs": 100,
        "actual_epochs": 100,
        "early_stop_reason": "",
        "fit_status": "completed",
        "failure_type": "",
        "failure_message": "",
        "train_inner_fit_n": 1000,
        "train_inner_validation_n": 200,
        "macro_f1": 0.55,
        "balanced_accuracy": 0.54,
        "accuracy": 0.55,
        "stratified_dummy_macro_f1_same_rows": 0.50,
        "delta_macro_f1_vs_dummy": 0.05,
        "class0_pred_rate": 0.48,
        "class1_pred_rate": 0.52,
        "ticker_max_share": 0.25,
        "actual_wall_clock_seconds": 12.5,
        "peak_memory_mb": 256.0,
        "gpu_seconds_or_null": None,
        "compute_tier": "full_compute",
        "scope": "exploratory",
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }
    row.update(overrides)
    return row


def test_trial_ledger_valid_frame_passes():
    df = pd.DataFrame([_valid_trial_row()])
    c.validate_trial_ledger_frame(df)


def test_trial_ledger_empty_frame_passes():
    """Empty trial ledger is acceptable (search hasn't run yet); validator
    only checks columns exist on empty frame."""
    df = pd.DataFrame(columns=sorted(c.REQUIRED_TRIAL_LEDGER_COLUMNS))
    c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_missing_required_columns():
    df = pd.DataFrame([_valid_trial_row()]).drop(columns=["compute_tier"])
    with pytest.raises(AssertionError, match="missing columns"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_invalid_compute_tier():
    df = pd.DataFrame([_valid_trial_row(compute_tier="micro_compute")])
    with pytest.raises(AssertionError, match="compute_tier"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_official_validation_used_True_row():
    df = pd.DataFrame([_valid_trial_row(official_validation_used=True)])
    with pytest.raises(AssertionError, match="official_validation_used=True"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_holdout_authorized_True_row():
    df = pd.DataFrame([_valid_trial_row(holdout_test_authorized=True)])
    with pytest.raises(AssertionError, match="holdout_test_authorized=True"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_non_exploratory_scope():
    df = pd.DataFrame([_valid_trial_row(scope="validation_only")])
    with pytest.raises(AssertionError, match="scope column must be 'exploratory'"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_invalid_fit_status():
    """Round 7 #6 -- fit_status enum check."""
    df = pd.DataFrame([_valid_trial_row(fit_status="running")])
    with pytest.raises(AssertionError, match="fit_status has invalid values"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_accepts_pending_last_step_lightgbm():
    """fit_status='pending_last_step_lightgbm' is the MVP control row; it
    MUST validate so the generator's emitted ledger doesn't trip a static
    gate it itself created."""
    df = pd.DataFrame([_valid_trial_row(fit_status="pending_last_step_lightgbm")])
    c.validate_trial_ledger_frame(df)


def test_trial_ledger_rejects_failed_row_with_invalid_failure_type():
    """Round 7 #6 -- failed rows must carry a typed failure_type from
    FAILURE_TYPES; a free-text failure_type makes the failure map unauditable."""
    df = pd.DataFrame([
        _valid_trial_row(
            fit_status="failed", failure_type="weird_oops_thing"
        )
    ])
    with pytest.raises(AssertionError, match="invalid failure_type"):
        c.validate_trial_ledger_frame(df)


def test_trial_ledger_accepts_failed_row_with_known_failure_type():
    df = pd.DataFrame([
        _valid_trial_row(
            fit_status="failed", failure_type="not_implemented"
        )
    ])
    c.validate_trial_ledger_frame(df)


def test_trial_ledger_accepts_completed_row_with_empty_failure_type():
    """Round 7 #6 -- completed rows may leave failure_type empty; only
    failed rows are required to carry a typed failure."""
    df = pd.DataFrame([
        _valid_trial_row(
            fit_status="completed", failure_type=""
        )
    ])
    c.validate_trial_ledger_frame(df)


# ---------- §13.1 08x_run_manifest validator --------------------------------


def _valid_08x_run_manifest() -> dict:
    return {
        "notebook08_version": "2026-06-06-mvp",
        "stage": "08X",
        "scope": "exploratory",
        "source_stage0_candidate": {
            "label_config": "h03_bps1p5",
            "feature_set": "price_volume_time",
            "window_size": 20,
        },
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "train_inner_fold_policy": "embargoed_train_inner_folds",
        "purge_policy": "horizon_bar_purge",
        "embargo_policy": "one_bar_embargo",
        "search_budget_tier": "quick",
        "trial_count_requested": 10,
        "trial_count_completed": 8,
        "trial_count_failed": 2,
        "trial_count_skipped": 0,
    }


def test_08x_run_manifest_valid_passes():
    c.validate_08x_run_manifest(_valid_08x_run_manifest())


def test_08x_run_manifest_rejects_wrong_stage():
    payload = _valid_08x_run_manifest()
    payload["stage"] = "08F"
    with pytest.raises(AssertionError, match="stage must be '08X'"):
        c.validate_08x_run_manifest(payload)


def test_08x_run_manifest_rejects_wrong_scope():
    payload = _valid_08x_run_manifest()
    payload["scope"] = "validation_only"
    with pytest.raises(AssertionError, match="scope must be 'exploratory'"):
        c.validate_08x_run_manifest(payload)


def test_08x_run_manifest_rejects_official_validation_used_True():
    payload = _valid_08x_run_manifest()
    payload["official_validation_used"] = True
    with pytest.raises(AssertionError, match="official_validation_used"):
        c.validate_08x_run_manifest(payload)


def test_08x_run_manifest_rejects_holdout_test_authorized_True():
    payload = _valid_08x_run_manifest()
    payload["holdout_test_authorized"] = True
    with pytest.raises(AssertionError, match="holdout_test_authorized"):
        c.validate_08x_run_manifest(payload)


@pytest.mark.parametrize(
    "dropped", sorted(c.REQUIRED_08X_RUN_MANIFEST_FIELDS)
)
def test_08x_run_manifest_rejects_missing_field(dropped: str):
    payload = _valid_08x_run_manifest()
    payload.pop(dropped)
    with pytest.raises(AssertionError, match="missing fields"):
        c.validate_08x_run_manifest(payload)


# ---------- §13.3 08o_run_manifest validator --------------------------------


def _valid_08o_run_manifest_real() -> dict:
    return {
        "stage": "08O",
        "scope": "validation_only",
        "primary_candidate_id": "ms_dlinear_tcn_deadbeef",
        "freeze_record_sha256": "f" * 64,
        "official_validation_readout_started_at": "2026-06-08T10:00:00Z",
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
        "same_row_dummy_present": True,
        "per_ticker_present": True,
        "seed_summary_present": True,
        "allowed_wording_bucket": "weak_mixed",
    }


def _valid_08o_run_manifest_stub() -> dict:
    """Round 7 #1 -- the stub-mode shape: stub=True, *_present=False, wording
    forced to no_candidate_freezable."""
    payload = _valid_08o_run_manifest_real()
    payload["schema_only_stub"] = True
    payload["same_row_dummy_present"] = False
    payload["per_ticker_present"] = False
    payload["seed_summary_present"] = False
    payload["allowed_wording_bucket"] = "no_candidate_freezable"
    return payload


def test_08o_run_manifest_real_mode_valid_passes():
    c.validate_08o_run_manifest(_valid_08o_run_manifest_real())


def test_08o_run_manifest_stub_mode_valid_passes():
    c.validate_08o_run_manifest(_valid_08o_run_manifest_stub())


def test_08o_run_manifest_real_mode_requires_same_row_dummy_present_True():
    payload = _valid_08o_run_manifest_real()
    payload["same_row_dummy_present"] = False
    with pytest.raises(AssertionError, match="same_row_dummy_present"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_real_mode_requires_per_ticker_present_True():
    payload = _valid_08o_run_manifest_real()
    payload["per_ticker_present"] = False
    with pytest.raises(AssertionError, match="per_ticker_present"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_real_mode_requires_seed_summary_present_True():
    payload = _valid_08o_run_manifest_real()
    payload["seed_summary_present"] = False
    with pytest.raises(AssertionError, match="seed_summary_present"):
        c.validate_08o_run_manifest(payload)


@pytest.mark.parametrize(
    "evidence_bucket",
    ["improvement", "weak_mixed", "low_compute_baseline", "unstable"],
)
def test_08o_run_manifest_stub_mode_rejects_evidence_wording(evidence_bucket: str):
    """Round 7 #1 -- the central safety. Stub manifests MUST carry the
    no_candidate_freezable wording bucket so a paper consumer cannot mistake
    empty artifacts for any kind of evidence."""
    payload = _valid_08o_run_manifest_stub()
    payload["allowed_wording_bucket"] = evidence_bucket
    with pytest.raises(AssertionError, match="no_candidate_freezable"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_stub_mode_allows_present_flags_False():
    """In stub mode the *_present flags can legitimately be False."""
    payload = _valid_08o_run_manifest_stub()
    # already False in the stub fixture -- this is the positive control.
    c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_rejects_wrong_stage():
    payload = _valid_08o_run_manifest_real()
    payload["stage"] = "08X"
    with pytest.raises(AssertionError, match="stage must be '08O'"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_rejects_wrong_scope():
    payload = _valid_08o_run_manifest_real()
    payload["scope"] = "exploratory"
    with pytest.raises(AssertionError, match="scope must be 'validation_only'"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_rejects_official_val_selection_True():
    payload = _valid_08o_run_manifest_real()
    payload["official_validation_used_for_selection"] = True
    with pytest.raises(AssertionError, match="official_validation_used_for_selection"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_rejects_holdout_test_authorized_True():
    payload = _valid_08o_run_manifest_real()
    payload["holdout_test_authorized"] = True
    with pytest.raises(AssertionError, match="holdout_test_authorized"):
        c.validate_08o_run_manifest(payload)


def test_08o_run_manifest_rejects_invalid_wording_bucket():
    payload = _valid_08o_run_manifest_real()
    payload["allowed_wording_bucket"] = "tradable"  # not in ALLOWED_WORDING_BUCKETS
    with pytest.raises(AssertionError, match="allowed_wording_bucket"):
        c.validate_08o_run_manifest(payload)


# ---------- §10.1 OPERATOR_READOUT_AUTHORIZATION_SHA canonical recipe ------
# These tests pin down the byte-for-byte recipe so a future code change that
# silently breaks the canonical form (e.g. removes the path-length prefix or
# switches to system line endings) is caught.


def test_operator_readout_authorization_sha_canonical_recipe(tmp_path: Path):
    """Same inputs -> same sha (basic determinism check)."""
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"a": 1, "b": [1, 2, 3]}', encoding="utf-8")
    text_path = tmp_path / "sample.md"
    text_path.write_text("# Title\nbody line\n", encoding="utf-8")
    inputs = [(json_path, "json_canonical"), (text_path, "text_lf")]
    sha1 = c.operator_readout_authorization_sha(inputs)
    sha2 = c.operator_readout_authorization_sha(inputs)
    assert sha1 == sha2
    assert len(sha1) == 64
    assert all(ch in "0123456789abcdef" for ch in sha1)


def test_operator_readout_authorization_sha_text_lf_normalizes_crlf(tmp_path: Path):
    """text_lf mode must collapse \\r\\n -> \\n so a Windows-saved doc and a
    Unix-saved doc produce the SAME sha."""
    lf_path = tmp_path / "lf.md"
    crlf_path = tmp_path / "crlf.md"
    lf_path.write_bytes(b"# Title\nbody line\n")
    crlf_path.write_bytes(b"# Title\r\nbody line\r\n")
    sha_lf = c.operator_readout_authorization_sha([(lf_path, "text_lf")])
    sha_crlf = c.operator_readout_authorization_sha([(crlf_path, "text_lf")])
    # Different file paths, so the prefix path-bytes differ -- but the
    # path-name byte differences trip the sha. To isolate the CRLF
    # normalization itself, use the same path for both reads.
    same_path = tmp_path / "same.md"
    same_path.write_bytes(b"# Title\nbody line\n")
    sha_a = c.operator_readout_authorization_sha([(same_path, "text_lf")])
    same_path.write_bytes(b"# Title\r\nbody line\r\n")
    sha_b = c.operator_readout_authorization_sha([(same_path, "text_lf")])
    assert sha_a == sha_b, (
        "text_lf canonicalization failed to collapse CRLF -> LF: "
        f"LF={sha_a} CRLF={sha_b}"
    )


def test_operator_readout_authorization_sha_json_canonical_sorts_keys(tmp_path: Path):
    """json_canonical mode must sort keys + use compact separators, so two
    semantically identical JSON files with different key orders produce the
    SAME sha."""
    a_path = tmp_path / "order_a.json"
    b_path = tmp_path / "order_a.json"  # same path on purpose -- see below
    a_path.write_bytes(b'{"alpha": 1, "beta": 2}')
    sha_a = c.operator_readout_authorization_sha([(a_path, "json_canonical")])
    b_path.write_bytes(b'{"beta": 2, "alpha": 1}')  # reorder, same content
    sha_b = c.operator_readout_authorization_sha([(b_path, "json_canonical")])
    assert sha_a == sha_b, (
        "json_canonical failed to sort keys: "
        f"alpha-first={sha_a} beta-first={sha_b}"
    )


def test_operator_readout_authorization_sha_json_canonical_rejects_nan(tmp_path: Path):
    """allow_nan=False must reject NaN tokens in the JSON input -- otherwise
    a freeze record with silently-NaN paper_safe_score could pass through."""
    nan_path = tmp_path / "nan.json"
    # raw JSON content with NaN literal (which json.loads tolerates by default
    # but json.dumps(allow_nan=False) refuses to emit).
    nan_path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises((ValueError, Exception)):
        # json.loads accepts the JS-style NaN, but allow_nan=False on dumps
        # raises ValueError when it sees float('nan').
        c.operator_readout_authorization_sha([(nan_path, "json_canonical")])


def test_operator_readout_authorization_sha_order_matters(tmp_path: Path):
    """The recipe says 'DO NOT sort, DO NOT change order' over inputs --
    a swapped input order MUST produce a different sha."""
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text('{"k": "a"}', encoding="utf-8")
    b_path.write_text('{"k": "b"}', encoding="utf-8")
    sha_ab = c.operator_readout_authorization_sha([
        (a_path, "json_canonical"),
        (b_path, "json_canonical"),
    ])
    sha_ba = c.operator_readout_authorization_sha([
        (b_path, "json_canonical"),
        (a_path, "json_canonical"),
    ])
    assert sha_ab != sha_ba, "order-sensitive recipe accepted swapped order"


def test_operator_readout_authorization_sha_rejects_unknown_mode(tmp_path: Path):
    """The recipe enumerates exactly 'json_canonical' and 'text_lf'; an
    unrecognized mode is a contract violation, not silently accepted."""
    p = tmp_path / "x.txt"
    p.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError, match="canonicalization mode"):
        c.operator_readout_authorization_sha([(p, "raw_bytes")])


def test_operator_readout_authorization_sha_path_in_hash(tmp_path: Path):
    """The recipe prefixes each input with its UTF-8 path bytes -- so a
    rename of the same file content MUST change the sha. This proves the
    sha binds path AND content together (not just content)."""
    p1 = tmp_path / "freeze_record.json"
    p2 = tmp_path / "freeze_record_v2.json"
    p1.write_text('{"k": "v"}', encoding="utf-8")
    p2.write_text('{"k": "v"}', encoding="utf-8")  # same content
    sha1 = c.operator_readout_authorization_sha([(p1, "json_canonical")])
    sha2 = c.operator_readout_authorization_sha([(p2, "json_canonical")])
    assert sha1 != sha2, (
        "rename of identical content produced same sha -- recipe must "
        "bind path AND content"
    )


# ===========================================================================
# Round 8 follow-up: 08O real-readout completeness gate.
#
# Round 7 hardened the MANIFEST. Round 8 closes the upstream gap: the
# generator's "any file non-empty" check was too permissive (one row in one
# artifact flipped the manifest into real-readout mode). The new gate
# requires EVERY required artifact present + non_empty + schema-complete.
# ===========================================================================


def _seed_required_08o_artifacts(
    output_dir: Path, *, populate: set[str] | None = None,
    extra_columns: dict[str, list[str]] | None = None,
    drop_columns: dict[str, list[str]] | None = None,
) -> None:
    """Write each required 08O artifact into ``output_dir``. By default every
    artifact gets one synthetic data row, so the gate sees a "real" readout.

    - ``populate``: optional subset of filenames to receive rows; others get
      header-only stubs (simulates partial population).
    - ``extra_columns``: optional mapping of filename to extra column names
      that get appended -- additive columns must NOT trip the gate.
    - ``drop_columns``: optional mapping of filename to required columns to
      drop -- schema drift MUST trip the gate.
    """
    extra_columns = extra_columns or {}
    drop_columns = drop_columns or {}
    # Define one canonical sample row per artifact. Values are placeholders;
    # only schema shape matters for the gate.
    sample_rows = {
        "08o_validation_readout.csv": [{
            "seed": 260501,
            "macro_f1": 0.55,
            "balanced_accuracy": 0.54,
            "accuracy": 0.55,
            "delta_macro_f1_vs_stratified_dummy_same_rows": 0.05,
            "delta_balanced_accuracy_vs_stratified_dummy_same_rows": 0.04,
            "validation_n": 200,
            "class0_pred_rate": 0.48,
            "class1_pred_rate": 0.52,
        }],
        "08o_validation_per_ticker.csv": [{
            "ticker": "CSCO",
            "macro_f1": 0.56,
            "delta_macro_f1_vs_dummy": 0.06,
            "n_rows": 40,
        }],
        "08o_seed_summary.csv": [{
            "metric": "macro_f1",
            "seed_mean": 0.55,
            "seed_std": 0.01,
            "seed_lcb_95": 0.53,
        }],
        "08o_same_row_baselines.csv": [{
            "baseline": "stratified_dummy",
            "macro_f1_mean": 0.50,
            "macro_f1_std": 0.005,
        }],
    }
    for filename, required_cols in c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.items():
        path = output_dir / filename
        cols = list(required_cols)
        for extra in extra_columns.get(filename, []):
            if extra not in cols:
                cols.append(extra)
        for to_drop in drop_columns.get(filename, []):
            if to_drop in cols:
                cols.remove(to_drop)
        should_populate = populate is None or filename in populate
        if should_populate:
            rows = []
            for sample in sample_rows[filename]:
                row = {col: sample.get(col, "x") for col in cols}
                rows.append(row)
            pd.DataFrame(rows, columns=cols).to_csv(
                path, index=False, lineterminator="\n"
            )
        else:
            pd.DataFrame(columns=cols).to_csv(
                path, index=False, lineterminator="\n"
            )


def test_completeness_gate_all_artifacts_complete_is_real(tmp_path: Path):
    """Positive control: every required artifact present, non-empty, schema-
    complete -> is_real_readout=True."""
    _seed_required_08o_artifacts(tmp_path)
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is True
    assert verdict["missing_artifacts"] == []
    assert verdict["empty_artifacts"] == []
    assert verdict["schema_drift"] == []
    for filename in c.REQUIRED_08O_REAL_READOUT_ARTIFACTS:
        per = verdict["per_artifact"][filename]
        assert per["present"] is True
        assert per["non_empty"] is True
        assert per["schema_complete"] is True
        assert per["row_count"] == 1


def test_completeness_gate_no_artifacts_stays_stub(tmp_path: Path):
    """Negative control: empty output dir -> is_real_readout=False."""
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is False
    assert set(verdict["missing_artifacts"]) == set(
        c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys()
    )


@pytest.mark.parametrize(
    "skipped", sorted(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys())
)
def test_completeness_gate_one_missing_artifact_stays_stub(
    tmp_path: Path, skipped: str
):
    """If any single required artifact is absent, the gate must refuse real
    mode -- evidence quality is no stronger than its weakest link."""
    keep = set(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys()) - {skipped}
    _seed_required_08o_artifacts(tmp_path, populate=keep)
    # Also remove the file entirely (the helper still creates a header-only
    # stub if we leave the path alone -- we want to test "absent" here).
    (tmp_path / skipped).unlink()
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is False
    assert skipped in verdict["missing_artifacts"]
    assert verdict["per_artifact"][skipped]["present"] is False


@pytest.mark.parametrize(
    "empty_one", sorted(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys())
)
def test_completeness_gate_one_empty_artifact_stays_stub(
    tmp_path: Path, empty_one: str
):
    """Round 8 #1 core regression: if even one required artifact is
    header-only (no data rows), the gate must refuse real mode. The previous
    `any(file non-empty)` check let this slip through."""
    populate = set(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys()) - {empty_one}
    _seed_required_08o_artifacts(tmp_path, populate=populate)
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is False
    assert empty_one in verdict["empty_artifacts"]
    assert verdict["per_artifact"][empty_one]["non_empty"] is False


def test_completeness_gate_one_only_populated_stays_stub(tmp_path: Path):
    """Direct regression for the previous gate's failure mode: writing rows
    to a single artifact must NOT promote the manifest to real mode."""
    only_one = "08o_validation_readout.csv"
    _seed_required_08o_artifacts(tmp_path, populate={only_one})
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is False
    assert verdict["per_artifact"][only_one]["non_empty"] is True
    assert len(verdict["empty_artifacts"]) == 3  # the other three are header-only


@pytest.mark.parametrize(
    "drifted", sorted(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS.keys())
)
def test_completeness_gate_one_schema_drift_stays_stub(
    tmp_path: Path, drifted: str
):
    """If a required artifact is non-empty but missing a required column, the
    gate must refuse real mode -- you cannot run a real readout against a
    schema-broken table."""
    drop_one = list(c.REQUIRED_08O_REAL_READOUT_ARTIFACTS[drifted])[:1]
    _seed_required_08o_artifacts(tmp_path, drop_columns={drifted: drop_one})
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is False
    assert drifted in verdict["schema_drift"]
    assert drop_one[0] in verdict["per_artifact"][drifted]["missing_columns"]


def test_completeness_gate_extra_columns_are_additive(tmp_path: Path):
    """Schema-complete means "REQUIRED columns are present"; extra columns
    are additive and MUST NOT trip the gate. This keeps the gate forward-
    compatible with future schema growth."""
    _seed_required_08o_artifacts(
        tmp_path,
        extra_columns={
            "08o_validation_readout.csv": ["future_lcb_95", "future_p_value"],
            "08o_validation_per_ticker.csv": ["future_ticker_volume"],
        },
    )
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is True
    assert verdict["schema_drift"] == []


def test_completeness_gate_concentration_and_failure_files_dont_gate(
    tmp_path: Path,
):
    """A real readout with zero concentration warnings AND zero failed seeds
    is legitimate. The gate must NOT require non-empty rows in
    08o_concentration_guardrails.csv or 08o_failure_rows.csv -- otherwise
    every clean readout would be flagged as a stub (false positive)."""
    _seed_required_08o_artifacts(tmp_path)
    # Drop the two files that should NOT gate. The gate must still pass.
    optional_files = [
        "08o_concentration_guardrails.csv",
        "08o_failure_rows.csv",
    ]
    for filename in optional_files:
        path = tmp_path / filename
        if path.exists():
            path.unlink()
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["is_real_readout"] is True, (
        "concentration/failure CSVs should NOT gate real-mode promotion"
    )


def test_completeness_gate_row_count_reported_correctly(tmp_path: Path):
    """row_count in the verdict should match the actual file row count so
    the manifest's audit trail is honest."""
    _seed_required_08o_artifacts(tmp_path)
    # Append two more rows to validation_readout, leaving the other three
    # at one row each.
    readout_path = tmp_path / "08o_validation_readout.csv"
    existing = pd.read_csv(readout_path)
    additional = pd.concat([existing, existing.copy()], ignore_index=True)
    additional.to_csv(readout_path, index=False, lineterminator="\n")
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert verdict["per_artifact"]["08o_validation_readout.csv"]["row_count"] == 2
    assert verdict["per_artifact"]["08o_validation_per_ticker.csv"]["row_count"] == 1
    assert verdict["is_real_readout"] is True


def test_completeness_gate_returns_dict_shape(tmp_path: Path):
    """Lock the public verdict shape so downstream code (manifest writer,
    /how-it-works docs, audit logs) can rely on it."""
    verdict = c.check_08o_real_readout_completeness(tmp_path)
    assert set(verdict.keys()) == {
        "is_real_readout",
        "per_artifact",
        "missing_artifacts",
        "empty_artifacts",
        "schema_drift",
    }
    assert isinstance(verdict["is_real_readout"], bool)
    assert isinstance(verdict["per_artifact"], dict)
    assert isinstance(verdict["missing_artifacts"], list)
    for filename, per in verdict["per_artifact"].items():
        assert set(per.keys()) == {
            "present", "non_empty", "schema_complete",
            "missing_columns", "row_count",
        }
