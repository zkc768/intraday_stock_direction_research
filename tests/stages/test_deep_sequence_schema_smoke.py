import json

import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    OUTPUT_FILES_08X,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    validate_08x_run_manifest,
    validate_08x_search_space,
    validate_trial_ledger_frame,
)
from intraday_research.stages import deep_sequence_schema_smoke as sm


def test_resolve_output_dir_prefers_kwarg(tmp_path):
    config_path = tmp_path / "config"
    kwarg_path = tmp_path / "kwarg"

    resolved = sm.resolve_output_dir(
        {"outputs": {"results_dir": str(config_path)}},
        kwarg_path,
    )

    assert resolved == kwarg_path


def test_resolve_output_dir_uses_config_results_dir(tmp_path):
    config_path = tmp_path / "config"

    resolved = sm.resolve_output_dir(
        {"outputs": {"results_dir": str(config_path)}},
        None,
    )

    assert resolved == config_path


def test_resolve_output_dir_requires_one_source():
    with pytest.raises(ValueError, match="output_dir"):
        sm.resolve_output_dir({}, None)


def test_write_schema_smoke_artifacts_emits_valid_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_pip_freeze_sha256", lambda: "pip-freeze-sha")
    monkeypatch.setattr(sm, "_collect_dependency_versions", lambda: {"pandas": "x"})
    monkeypatch.setattr(sm, "_git_head_sha", lambda: "git-sha")
    monkeypatch.setattr(sm, "_git_dirty", lambda: False)

    sm.write_schema_smoke_artifacts(tmp_path)

    written = {path.name for path in tmp_path.iterdir() if path.is_file()}
    assert written == set(OUTPUT_FILES_08X)

    search_space = json.loads((tmp_path / "08x_search_space.json").read_text("utf-8"))
    validate_08x_search_space(search_space)
    assert search_space["official_validation_used"] is False
    assert search_space["holdout_test_authorized"] is False

    ledger = pd.read_csv(tmp_path / "08x_trial_ledger.csv")
    assert ledger.empty
    assert set(REQUIRED_TRIAL_LEDGER_COLUMNS).issubset(set(ledger.columns))
    validate_trial_ledger_frame(ledger)

    manifest = json.loads((tmp_path / "08x_run_manifest.json").read_text("utf-8"))
    validate_08x_run_manifest(manifest)
    assert manifest["official_validation_used"] is False
    assert manifest["holdout_test_authorized"] is False
    assert manifest["trial_count_requested"] == 0

    env = json.loads((tmp_path / "08x_environment_manifest.json").read_text("utf-8"))
    assert set(sm.ENV_MANIFEST_KEYS).issubset(env)
    assert env["pip_freeze_sha256"] == "pip-freeze-sha"
    assert env["dependency_versions"] == {"pandas": "x"}
    assert env["git_commit"] == "git-sha"
    assert env["git_dirty"] is False


def test_write_schema_smoke_artifacts_keeps_unvalidated_csv_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "_pip_freeze_sha256", lambda: "pip-freeze-sha")
    monkeypatch.setattr(sm, "_collect_dependency_versions", lambda: {})
    monkeypatch.setattr(sm, "_git_head_sha", lambda: "git-sha")
    monkeypatch.setattr(sm, "_git_dirty", lambda: False)

    sm.write_schema_smoke_artifacts(tmp_path)

    expectations = {
        "08x_fold_results.csv": list(sm.FOLD_RESULTS_COLUMNS),
        "08x_seed_summary.csv": list(sm.SEED_SUMMARY_COLUMNS),
        "08x_failure_ledger.csv": list(sm.FAILURE_LEDGER_COLUMNS),
        "08x_candidate_compression_table.csv": list(sm.CANDIDATE_COMPRESSION_COLUMNS),
    }
    for filename, expected_cols in expectations.items():
        df = pd.read_csv(tmp_path / filename)
        assert df.empty, f"{filename}: schema-smoke should write header-only"
        assert list(df.columns) == expected_cols
