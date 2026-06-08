import pandas as pd
import pytest

from intraday_research.contracts.validation_synthesis_gap_audit import (
    validate_ledger_frame,
)
from intraday_research.stages.validation_budget_ledger import (
    LEDGER_COLUMNS,
    ValidationBudgetLedger,
    append_validation_budget_ledger_row,
    read_ledger_frame,
)


def test_append_row_flushes_valid_ledger(tmp_path):
    path = tmp_path / "notebook07_validation_budget_ledger.csv"
    ledger = ValidationBudgetLedger(path, now=lambda: "2026-06-07T00:00:00+00:00")

    row = ledger.append_row(
        "notebook05_run_manifest.json",
        "07A",
        "lockfile_scope_gate",
        "before_official_validation_read",
        "run_manifest",
        official_validation_rows_inspected=1,
        allowed_wording="validation_only",
    )

    assert row["cumulative_official_validation_inspections_across_notebooks"] == 1
    assert path.exists()
    on_disk = read_ledger_frame(path)
    assert list(on_disk.columns) == list(LEDGER_COLUMNS)
    assert len(on_disk) == 1
    assert on_disk.loc[0, "thresholds_or_coverages_considered"] == "n/a"
    assert on_disk.loc[0, "appended_by_notebook"] == "07"
    validate_ledger_frame(on_disk)


def test_new_instance_hydrates_existing_rows_and_extends_cumulative(tmp_path):
    path = tmp_path / "notebook07_validation_budget_ledger.csv"
    ValidationBudgetLedger(path, now=lambda: "2026-06-07T00:00:00+00:00").append_row(
        "first.csv",
        "07A",
        "first_read_intent",
        "before_official_validation_read",
        "comparison",
        official_validation_rows_inspected=1,
    )

    restarted = ValidationBudgetLedger(
        path,
        appended_by_notebook="08O",
        now=lambda: "2026-06-08T00:00:00+00:00",
    )
    row = restarted.append_row(
        "08o_run_manifest.json",
        "08O",
        "official_validation_readout_intent",
        "before_official_validation_read",
        "official_validation_readout",
        official_validation_rows_inspected=1,
    )

    assert row["cumulative_official_validation_inspections_across_notebooks"] == 2
    on_disk = read_ledger_frame(path)
    assert len(on_disk) == 2
    assert on_disk.loc[0, "thresholds_or_coverages_considered"] == "n/a"
    assert on_disk.loc[0, "appended_by_notebook"] == "07"
    assert on_disk.loc[1, "appended_by_notebook"] == "08O"
    validate_ledger_frame(on_disk)


def test_existing_prefix_modification_is_rejected_before_overwrite(tmp_path):
    path = tmp_path / "notebook07_validation_budget_ledger.csv"
    ledger = ValidationBudgetLedger(path, now=lambda: "2026-06-07T00:00:00+00:00")
    ledger.append_row(
        "first.csv",
        "07A",
        "first_read_intent",
        "before_official_validation_read",
        "comparison",
        risk_note="original",
    )

    tampered = read_ledger_frame(path)
    tampered.loc[0, "risk_note"] = "tampered"
    tampered.to_csv(path, index=False, lineterminator="\n")

    with pytest.raises(AssertionError, match="prefix invariance violated"):
        ledger.append_row(
            "second.csv",
            "07B",
            "second_read_intent",
            "before_official_validation_read",
            "comparison",
        )

    on_disk = read_ledger_frame(path)
    assert len(on_disk) == 1
    assert on_disk.loc[0, "risk_note"] == "tampered"


def test_one_shot_append_helper_preserves_literal_strings(tmp_path):
    path = tmp_path / "notebook07_validation_budget_ledger.csv"

    append_validation_budget_ledger_row(
        path,
        "first.csv",
        "07A",
        "first_read_intent",
        "before_official_validation_read",
        "comparison",
        thresholds_or_coverages_considered="n/a",
        seeds_used="07",
    )

    on_disk = read_ledger_frame(path)
    assert on_disk.loc[0, "thresholds_or_coverages_considered"] == "n/a"
    assert on_disk.loc[0, "seeds_used"] == "07"
    assert on_disk.loc[0, "appended_by_notebook"] == "07"


def test_flush_rejects_schema_drift_in_existing_file(tmp_path):
    path = tmp_path / "notebook07_validation_budget_ledger.csv"
    pd.DataFrame([{"artifact": "missing most columns"}]).to_csv(
        path,
        index=False,
        lineterminator="\n",
    )

    ledger = ValidationBudgetLedger(path, now=lambda: "2026-06-07T00:00:00+00:00")
    with pytest.raises(AssertionError, match="missing columns"):
        ledger.append_row(
            "second.csv",
            "07B",
            "second_read_intent",
            "before_official_validation_read",
            "comparison",
        )
