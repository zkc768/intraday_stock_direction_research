import json

import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    validate_08x_run_manifest,
)
from intraday_research.stages.run_manifest import (
    validate_run_manifest_payload,
    write_run_manifest,
)


def _valid_08x_manifest() -> dict:
    return {
        "notebook08_version": "unit",
        "stage": "08X",
        "scope": "exploratory",
        "source_stage0_candidate": "schema_smoke_no_candidate",
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "train_inner_fold_policy": "none_smoke",
        "purge_policy": "none_smoke",
        "embargo_policy": "none_smoke",
        "search_budget_tier": "schema_smoke",
        "trial_count_requested": 0,
        "trial_count_completed": 0,
        "trial_count_failed": 0,
        "trial_count_skipped": 0,
    }


def test_validate_run_manifest_payload_enforces_generic_guards():
    payload = {"stage": "08X", "scope": "exploratory", "holdout": False}

    validate_run_manifest_payload(
        payload,
        required_fields=("stage", "scope", "holdout"),
        stage="08X",
        scope="exploratory",
        false_fields=("holdout",),
    )


def test_validate_run_manifest_payload_rejects_missing_stage_scope_and_true_flags():
    with pytest.raises(AssertionError, match="missing fields"):
        validate_run_manifest_payload({"stage": "08X"}, required_fields=("scope",))
    with pytest.raises(AssertionError, match="stage must be '08X'"):
        validate_run_manifest_payload({"stage": "08O"}, stage="08X")
    with pytest.raises(AssertionError, match="scope must be 'exploratory'"):
        validate_run_manifest_payload({"scope": "validation_only"}, scope="exploratory")
    with pytest.raises(AssertionError, match="holdout=True is forbidden"):
        validate_run_manifest_payload({"holdout": True}, false_fields=("holdout",))


def test_write_run_manifest_validates_and_writes_json(tmp_path):
    path = tmp_path / "08x_run_manifest.json"
    payload = _valid_08x_manifest()

    returned = write_run_manifest(
        path,
        payload,
        validator=validate_08x_run_manifest,
        stage="08X",
        scope="exploratory",
        false_fields=("official_validation_used", "holdout_test_authorized"),
    )

    assert returned == payload
    loaded = json.loads(path.read_text("utf-8"))
    assert loaded == payload
    validate_08x_run_manifest(loaded)


def test_write_run_manifest_does_not_write_when_validator_fails(tmp_path):
    path = tmp_path / "bad_run_manifest.json"
    payload = _valid_08x_manifest()
    payload["official_validation_used"] = True

    with pytest.raises(AssertionError, match="official_validation_used=True"):
        write_run_manifest(
            path,
            payload,
            validator=validate_08x_run_manifest,
            stage="08X",
            scope="exploratory",
            false_fields=("official_validation_used", "holdout_test_authorized"),
        )

    assert not path.exists()


def test_write_run_manifest_rejects_nan_payload(tmp_path):
    with pytest.raises(ValueError, match="Out of range float"):
        write_run_manifest(
            tmp_path / "nan_run_manifest.json",
            {"stage": "08X", "scope": "exploratory", "bad": float("nan")},
            stage="08X",
            scope="exploratory",
        )
