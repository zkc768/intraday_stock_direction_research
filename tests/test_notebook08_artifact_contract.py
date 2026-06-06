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

scripts/notebook08_contract.py does not exist yet; validators are inlined
here for now and should later move into that helper module.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest


# ---------- Schemas (inline) ------------------------------------------------


REQUIRED_DMC_FIELDS = {
    "dmc_role",
    "reviewer_identifier",
    "reviewed_08x_run_manifest_sha256",
    "reviewed_at_utc",
    "attestation_statement",
}

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

# Belt-and-suspenders: exact-substring layer catches snake_case identifiers
# that survive normalization without separator collapse.
FORBIDDEN_FALLBACK_RULE_SUBSTRINGS = (
    "official_validation_macro_f1",
    "official_validation_delta",
    "official_val_score",
    "official_validation_balanced_accuracy",
    "official_val_metric",
)


def _normalize_fallback_rule(text: str) -> str:
    """Lowercase + collapse [-_\\s] runs to single space so prose variants
    ('official-validation', 'official_val', 'official validation') normalize
    to the same form before regex matching."""
    return re.sub(r"[\s\-_]+", " ", text.lower())


# Regex layer catches English-prose abuses across separators. Patterns are
# applied to the normalized rule string.
FORBIDDEN_FALLBACK_PATTERNS_NORMALIZED = (
    # "official val ... <metric>"
    r"\bofficial val(?:idation)?\b[^.]*?\b(?:macro f1|balanced accuracy|delta|f1 score|f1|accuracy|metric|result|performance|auc|loss)\b",
    # "<metric> ... official val"   (note: 'score' alone excluded so "scoring official val" is OK)
    r"\b(?:macro f1|balanced accuracy|delta|f1 score|metric|result|performance|auc|loss)\b[^.]*?\bofficial val(?:idation)?\b",
    # "primary scor(es/ing) ... wors/lower/fail/poorly" — primary metric comparison
    r"\b(?:primary|fallback|model|deep)\b[^.]*?\b(?:scor|perform)[a-z]*\b[^.]*?\b(?:wors|worse|lower|low|fail|poorly|badly)\b",
    # explicit comparison verbs
    r"\b(?:fails? to beat|cannot beat|does not beat|outperformed by|beaten by|loses to)\b",
)


def validate_dmc_attestation(payload: dict) -> None:
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
) -> None:
    """08F MUST be operated in a separate Colab session by a non-08X-author OR
    with dmc_attestation.json present in the 08F input directory."""
    if dmc_attestation is None and same_session_as_08x:
        raise AssertionError(
            "08F entry violation: same session as 08X AND no dmc_attestation.json"
        )
    if dmc_attestation is not None:
        validate_dmc_attestation(dmc_attestation)


def validate_freeze_record(payload: dict) -> None:
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
    """08O must append exactly ONE row recording its read intent BEFORE
    actually reading official validation rows. The new row must:
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


# ---------- Fixtures --------------------------------------------------------


def _valid_dmc() -> dict:
    return {
        "dmc_role": "data_monitoring_committee_proxy",
        "reviewer_identifier": "reviewer-A",
        "reviewed_08x_run_manifest_sha256": "a" * 64,
        "reviewed_at_utc": "2026-06-07T12:34:56Z",
        "attestation_statement": "Reviewed 08X output; no official-validation contact detected.",
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


def test_08f_entry_accepts_separate_session_without_dmc():
    validate_08f_entry(dmc_attestation=None, same_session_as_08x=False)


def test_08f_entry_accepts_dmc_in_same_session():
    validate_08f_entry(dmc_attestation=_valid_dmc(), same_session_as_08x=True)


def test_08f_entry_rejects_same_session_without_dmc():
    with pytest.raises(AssertionError, match="dmc_attestation"):
        validate_08f_entry(dmc_attestation=None, same_session_as_08x=True)


def test_08f_entry_with_invalid_dmc_fails():
    bad_dmc = _valid_dmc()
    bad_dmc.pop("attestation_statement")
    with pytest.raises(AssertionError, match="missing fields"):
        validate_08f_entry(dmc_attestation=bad_dmc, same_session_as_08x=True)


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
