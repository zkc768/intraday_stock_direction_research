import json

import pandas as pd
import pytest

from intraday_research.artifact_preflight import (
    ArtifactSpec,
    assert_artifact_bundle_complete,
    build_bundle_manifest,
    check_artifact_bundle,
    csv_artifact,
    json_artifact,
    write_bundle_manifest,
)


def test_csv_bundle_complete_with_required_columns_and_extra_columns(tmp_path):
    pd.DataFrame(
        [{"a": 1, "b": 2, "future_column": 3}],
        columns=["a", "b", "future_column"],
    ).to_csv(tmp_path / "table.csv", index=False, lineterminator="\n")

    verdict = check_artifact_bundle(
        tmp_path,
        [csv_artifact("table.csv", ("a", "b"))],
    )

    assert verdict["bundle_complete"] is True
    assert verdict["missing_artifacts"] == []
    assert verdict["empty_artifacts"] == []
    assert verdict["schema_drift"] == []
    assert verdict["per_artifact"]["table.csv"]["row_count"] == 1
    assert verdict["per_artifact"]["table.csv"]["schema_complete"] is True


def test_csv_missing_empty_and_schema_drift_fail_closed(tmp_path):
    pd.DataFrame(columns=["a"]).to_csv(
        tmp_path / "empty.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame([{"a": 1}]).to_csv(
        tmp_path / "drift.csv", index=False, lineterminator="\n"
    )

    verdict = check_artifact_bundle(
        tmp_path,
        [
            csv_artifact("missing.csv", ("a",)),
            csv_artifact("empty.csv", ("a",)),
            csv_artifact("drift.csv", ("a", "b")),
        ],
    )

    assert verdict["bundle_complete"] is False
    assert verdict["missing_artifacts"] == ["missing.csv"]
    assert verdict["empty_artifacts"] == ["empty.csv"]
    assert verdict["schema_drift"] == ["drift.csv"]
    assert verdict["per_artifact"]["drift.csv"]["missing_columns"] == ["b"]


def test_header_only_csv_can_pass_when_non_empty_is_not_required(tmp_path):
    pd.DataFrame(columns=["a", "b"]).to_csv(
        tmp_path / "header_only.csv", index=False, lineterminator="\n"
    )

    verdict = check_artifact_bundle(
        tmp_path,
        [csv_artifact("header_only.csv", ("a", "b"), require_non_empty=False)],
    )

    assert verdict["bundle_complete"] is True
    assert verdict["per_artifact"]["header_only.csv"]["non_empty"] is False
    assert verdict["empty_artifacts"] == []


def test_json_artifact_checks_required_keys(tmp_path):
    (tmp_path / "payload.json").write_text(
        json.dumps({"a": 1}) + "\n",
        encoding="utf-8",
    )

    verdict = check_artifact_bundle(
        tmp_path,
        [json_artifact("payload.json", ("a", "b"))],
    )

    assert verdict["bundle_complete"] is False
    assert verdict["schema_drift"] == ["payload.json"]
    assert verdict["per_artifact"]["payload.json"]["missing_json_keys"] == ["b"]


def test_mapping_input_defaults_to_csv_artifacts(tmp_path):
    pd.DataFrame([{"a": 1}]).to_csv(
        tmp_path / "table.csv", index=False, lineterminator="\n"
    )

    verdict = check_artifact_bundle(tmp_path, {"table.csv": {"a"}})

    assert verdict["bundle_complete"] is True
    assert verdict["per_artifact"]["table.csv"]["schema_complete"] is True


def test_assert_artifact_bundle_complete_raises_with_summary(tmp_path):
    with pytest.raises(AssertionError, match="missing=\\['missing.csv'\\]"):
        assert_artifact_bundle_complete(
            tmp_path,
            [csv_artifact("missing.csv", ("a",))],
        )


def test_bundle_manifest_builder_and_writer_include_hashes(tmp_path):
    pd.DataFrame([{"a": 1}]).to_csv(
        tmp_path / "table.csv", index=False, lineterminator="\n"
    )

    manifest = build_bundle_manifest(
        tmp_path,
        [csv_artifact("table.csv", ("a",))],
        bundle_name="unit_bundle",
    )
    written = write_bundle_manifest(
        tmp_path / "manifest.json",
        tmp_path,
        [csv_artifact("table.csv", ("a",))],
        bundle_name="unit_bundle",
    )

    assert manifest["bundle_complete"] is True
    assert manifest["artifacts"][0]["sha256"] == written["artifacts"][0]["sha256"]
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert manifest["artifacts"][0]["required_columns"] == ["a"]
    loaded = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert loaded["bundle_name"] == "unit_bundle"
    assert loaded["artifacts"][0]["filename"] == "table.csv"
    assert loaded["artifacts"][0]["required_columns"] == ["a"]


def test_unknown_artifact_kind_raises(tmp_path):
    (tmp_path / "payload.bin").write_bytes(b"x")

    with pytest.raises(ValueError, match="unsupported artifact kind"):
        check_artifact_bundle(tmp_path, [ArtifactSpec("payload.bin", kind="weird")])
