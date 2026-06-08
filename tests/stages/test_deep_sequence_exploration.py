"""Tests for stage deep_sequence_exploration -- 08X schema-smoke slice."""
import json

import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    OUTPUT_FILES_08O,
    OUTPUT_FILES_08X,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    check_08o_real_readout_completeness,
    validate_08x_run_manifest,
    validate_08x_search_space,
    validate_trial_ledger_frame,
)
from intraday_research.stages.validation_budget_ledger import read_ledger_frame
from intraday_research.stages import deep_sequence_exploration as m


def test_stage_constants():
    assert m.STAGE_NAME == "deep_sequence_exploration"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert "notebook07_validation_budget_ledger.csv" in m.REQUIRED_ARTIFACTS


def test_default_run_no_op(tmp_path):
    """All switches absent -> no files written; run_stage returns None."""
    config = {"outputs": {"results_dir": str(tmp_path)}}
    result = m.run_stage(config)
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_schema_smoke_emits_all_8_artifacts(tmp_path):
    """RUN_08X_SCHEMA_SMOKE=True writes the 8 OUTPUT_FILES_08X artifacts; output_dir kwarg path."""
    config = {"run_switches": {"RUN_08X_SCHEMA_SMOKE": True}}
    m.run_stage(config, output_dir=tmp_path)
    written = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert written == set(OUTPUT_FILES_08X)


def test_schema_smoke_passes_contract_validators(tmp_path):
    """Reloaded artifacts pass their validators; env_manifest carries expected keys.

    Exercises config-derived output_dir resolution (kwarg None).
    """
    config = {
        "run_switches": {"RUN_08X_SCHEMA_SMOKE": True},
        "outputs": {"results_dir": str(tmp_path)},
    }
    m.run_stage(config)

    # 08x_search_space.json
    search_space = json.loads(
        (tmp_path / "08x_search_space.json").read_text("utf-8")
    )
    validate_08x_search_space(search_space)
    assert search_space["architecture_families"] == ["dlinear_only"]
    assert search_space["official_validation_used"] is False

    # 08x_trial_ledger.csv (empty-df branch of validator)
    ledger = pd.read_csv(tmp_path / "08x_trial_ledger.csv")
    assert ledger.empty
    assert set(REQUIRED_TRIAL_LEDGER_COLUMNS).issubset(set(ledger.columns))
    validate_trial_ledger_frame(ledger)

    # 08x_run_manifest.json
    manifest = json.loads(
        (tmp_path / "08x_run_manifest.json").read_text("utf-8")
    )
    validate_08x_run_manifest(manifest)
    assert manifest["trial_count_requested"] == 0
    assert manifest["trial_count_completed"] == 0
    assert manifest["trial_count_failed"] == 0
    assert manifest["trial_count_skipped"] == 0

    # 08x_environment_manifest.json -- no validator in this slice; assert key presence
    env = json.loads(
        (tmp_path / "08x_environment_manifest.json").read_text("utf-8")
    )
    for key in m.ENV_MANIFEST_KEYS:
        assert key in env, f"environment_manifest missing required key: {key}"
    assert env["manifest_mode"] == "schema_smoke"


def test_schema_smoke_requires_output_dir_resolution():
    """SCHEMA_SMOKE=True with neither kwarg nor config['outputs']['results_dir'] raises."""
    config = {"run_switches": {"RUN_08X_SCHEMA_SMOKE": True}}
    with pytest.raises(ValueError, match="output_dir"):
        m.run_stage(config)


@pytest.mark.parametrize("switch", m.OTHER_SWITCHES)
def test_other_run_switches_raise_not_implemented(tmp_path, switch):
    """Any non-SCHEMA_SMOKE switch True raises NotImplementedError; no partial output."""
    config = {"run_switches": {switch: True}}
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(config, output_dir=tmp_path)
    msg = str(exc.value)
    assert switch in msg
    assert list(tmp_path.iterdir()) == []


def test_unknown_future_run_switch_raises(tmp_path):
    """Prefix-based guard (Codex impl review P1-1): a future RUN_*/BACKUP_*
    switch not yet declared in OTHER_SWITCHES still raises NotImplementedError
    when truthy, so silent drift cannot accumulate."""
    config = {"run_switches": {"RUN_08X_FUTURE_THING": True}}
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(config, output_dir=tmp_path)
    assert "RUN_08X_FUTURE_THING" in str(exc.value)
    assert list(tmp_path.iterdir()) == []


def _official_prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"seed": 1, "ticker": "AAA", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "AAA", "y_true": 1, "y_pred": 1},
            {"seed": 1, "ticker": "BBB", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "BBB", "y_true": 1, "y_pred": 1},
        ]
    )


def test_08o_official_readout_writes_artifacts_after_ledger_append(tmp_path):
    predictions_path = tmp_path / "official_validation_predictions.csv"
    decision_record = tmp_path / "08o_decision_record.json"
    ledger_path = tmp_path / "notebook07_validation_budget_ledger.csv"
    _official_prediction_rows().to_csv(predictions_path, index=False)
    decision_record.write_text("{}", encoding="utf-8")
    config = {
        "run_switches": {"RUN_08O_OFFICIAL_VALIDATION_READOUT": True},
        "inputs": {
            "official_validation_predictions_csv": str(predictions_path),
            "08o_decision_record": str(decision_record),
            "validation_budget_ledger": str(ledger_path),
        },
        "frozen_candidate": {"architecture_family": "dlinear_only"},
    }

    m.run_stage(config, output_dir=tmp_path / "out")

    written = {p.name for p in (tmp_path / "out").iterdir() if p.is_file()}
    assert set(OUTPUT_FILES_08O) - {"08o_decision_record.json", "08o_run_manifest.json"} <= written
    verdict = check_08o_real_readout_completeness(tmp_path / "out")
    assert verdict["is_real_readout"] is True
    ledger = read_ledger_frame(ledger_path)
    assert len(ledger) == 1
    assert ledger.loc[0, "appended_by_notebook"] == "08O"
    assert ledger.loc[0, "decision_timing"] == "before_official_validation_read"
    assert int(ledger.loc[0, "official_validation_rows_inspected"]) == 0


def test_08o_readout_appends_ledger_before_prediction_file_read(tmp_path):
    predictions_path = tmp_path / "official_validation_predictions.csv"
    decision_record = tmp_path / "08o_decision_record.json"
    ledger_path = tmp_path / "notebook07_validation_budget_ledger.csv"
    decision_record.write_text("{}", encoding="utf-8")
    config = {
        "run_switches": {"RUN_08O_OFFICIAL_VALIDATION_READOUT": True},
        "inputs": {
            "official_validation_predictions_csv": str(predictions_path),
            "08o_decision_record": str(decision_record),
            "validation_budget_ledger": str(ledger_path),
        },
    }

    with pytest.raises(FileNotFoundError):
        m.run_stage(config, output_dir=tmp_path / "out")

    ledger = read_ledger_frame(ledger_path)
    assert len(ledger) == 1
    assert ledger.loc[0, "appended_by_notebook"] == "08O"


def test_08o_readout_requires_entry_gate_decision_record(tmp_path):
    predictions_path = tmp_path / "official_validation_predictions.csv"
    _official_prediction_rows().to_csv(predictions_path, index=False)
    config = {
        "run_switches": {"RUN_08O_OFFICIAL_VALIDATION_READOUT": True},
        "inputs": {
            "official_validation_predictions_csv": str(predictions_path),
            "08o_decision_record": str(tmp_path / "missing_decision_record.json"),
            "validation_budget_ledger": str(tmp_path / "ledger.csv"),
        },
    }

    with pytest.raises(FileNotFoundError, match="decision record missing"):
        m.run_stage(config, output_dir=tmp_path / "out")


def test_schema_smoke_and_08o_readout_must_be_separate_invocations(tmp_path):
    config = {
        "run_switches": {
            "RUN_08X_SCHEMA_SMOKE": True,
            "RUN_08O_OFFICIAL_VALIDATION_READOUT": True,
        }
    }

    with pytest.raises(ValueError, match="separate invocations"):
        m.run_stage(config, output_dir=tmp_path)


def test_schema_smoke_unvalidated_csv_headers(tmp_path):
    """The 4 CSVs without contract validators carry their spec §4.2 columns.

    Added per Codex impl review (P1 audit: these are not protected by a
    contract validator, so the stage tests are the only guard against drift
    between this slice and later slices that may write real rows.)
    """
    config = {"run_switches": {"RUN_08X_SCHEMA_SMOKE": True}}
    m.run_stage(config, output_dir=tmp_path)

    expectations = {
        "08x_fold_results.csv": list(m.FOLD_RESULTS_COLUMNS),
        "08x_seed_summary.csv": list(m.SEED_SUMMARY_COLUMNS),
        "08x_failure_ledger.csv": list(m.FAILURE_LEDGER_COLUMNS),
        "08x_candidate_compression_table.csv": list(m.CANDIDATE_COMPRESSION_COLUMNS),
    }
    for filename, expected_cols in expectations.items():
        df = pd.read_csv(tmp_path / filename)
        assert df.empty, f"{filename}: schema-smoke should write header-only"
        assert list(df.columns) == expected_cols, (
            f"{filename}: column drift; expected {expected_cols}, got {list(df.columns)}"
        )
