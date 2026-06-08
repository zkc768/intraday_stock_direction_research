"""Artifact preflight and bundle manifest helpers.

These helpers inspect files that already exist on disk. They do not load raw
market data, run models, read holdout/test splits, or make selection decisions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.stages.io_helpers import sha256_file_or_unavailable, write_json


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    kind: str = "file"
    required_columns: tuple[str, ...] = ()
    required_json_keys: tuple[str, ...] = ()
    require_non_empty: bool = True


def csv_artifact(
    filename: str,
    required_columns: Iterable[str],
    *,
    require_non_empty: bool = True,
) -> ArtifactSpec:
    return ArtifactSpec(
        filename=filename,
        kind="csv",
        required_columns=tuple(required_columns),
        require_non_empty=require_non_empty,
    )


def json_artifact(
    filename: str,
    required_keys: Iterable[str],
    *,
    require_non_empty: bool = True,
) -> ArtifactSpec:
    return ArtifactSpec(
        filename=filename,
        kind="json",
        required_json_keys=tuple(required_keys),
        require_non_empty=require_non_empty,
    )


def check_artifact_bundle(
    output_dir: Path | str,
    specs: Sequence[ArtifactSpec] | Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    base = Path(output_dir)
    artifact_specs = _coerce_specs(specs)
    per_artifact: dict[str, dict[str, Any]] = {}
    missing_artifacts: list[str] = []
    empty_artifacts: list[str] = []
    schema_drift: list[str] = []

    for spec in artifact_specs:
        status = _inspect_artifact(base, spec)
        per_artifact[spec.filename] = status
        if not status["present"]:
            missing_artifacts.append(spec.filename)
        if status["present"] and spec.require_non_empty and not status["non_empty"]:
            empty_artifacts.append(spec.filename)
        if status["present"] and not status["schema_complete"]:
            schema_drift.append(spec.filename)

    bundle_complete = (
        not missing_artifacts
        and not empty_artifacts
        and not schema_drift
    )
    return {
        "bundle_complete": bundle_complete,
        "per_artifact": per_artifact,
        "missing_artifacts": missing_artifacts,
        "empty_artifacts": empty_artifacts,
        "schema_drift": schema_drift,
    }


def assert_artifact_bundle_complete(
    output_dir: Path | str,
    specs: Sequence[ArtifactSpec] | Mapping[str, Iterable[str]],
) -> dict[str, Any]:
    verdict = check_artifact_bundle(output_dir, specs)
    if not verdict["bundle_complete"]:
        raise AssertionError(
            "artifact bundle incomplete: "
            f"missing={verdict['missing_artifacts']}, "
            f"empty={verdict['empty_artifacts']}, "
            f"schema_drift={verdict['schema_drift']}"
        )
    return verdict


def build_bundle_manifest(
    output_dir: Path | str,
    specs: Sequence[ArtifactSpec] | Mapping[str, Iterable[str]],
    *,
    bundle_name: str,
) -> dict[str, Any]:
    base = Path(output_dir)
    artifact_specs = _coerce_specs(specs)
    verdict = check_artifact_bundle(base, artifact_specs)
    artifacts = []
    for spec in artifact_specs:
        status = verdict["per_artifact"][spec.filename]
        artifacts.append(
            {
                "filename": spec.filename,
                "kind": spec.kind,
                "path": str(base / spec.filename),
                "require_non_empty": spec.require_non_empty,
                "required_columns": list(spec.required_columns),
                "required_json_keys": list(spec.required_json_keys),
                **status,
            }
        )
    return {
        "bundle_name": bundle_name,
        "output_dir": str(base),
        "bundle_complete": verdict["bundle_complete"],
        "missing_artifacts": verdict["missing_artifacts"],
        "empty_artifacts": verdict["empty_artifacts"],
        "schema_drift": verdict["schema_drift"],
        "artifacts": artifacts,
    }


def write_bundle_manifest(
    path: Path | str,
    output_dir: Path | str,
    specs: Sequence[ArtifactSpec] | Mapping[str, Iterable[str]],
    *,
    bundle_name: str,
) -> dict[str, Any]:
    manifest = build_bundle_manifest(output_dir, specs, bundle_name=bundle_name)
    write_json(path, manifest)
    return manifest


def _coerce_specs(
    specs: Sequence[ArtifactSpec] | Mapping[str, Iterable[str]],
) -> tuple[ArtifactSpec, ...]:
    if isinstance(specs, Mapping):
        return tuple(
            csv_artifact(filename, required_columns)
            for filename, required_columns in specs.items()
        )
    return tuple(specs)


def _inspect_artifact(base: Path, spec: ArtifactSpec) -> dict[str, Any]:
    path = base / spec.filename
    present = path.is_file()
    status: dict[str, Any] = {
        "present": present,
        "non_empty": False,
        "schema_complete": False,
        "missing_columns": [],
        "missing_json_keys": [],
        "row_count": 0,
        "size_bytes": 0,
        "sha256": "unavailable",
        "read_error": "",
    }
    if not present:
        return status

    status["size_bytes"] = path.stat().st_size
    status["sha256"] = sha256_file_or_unavailable(path)

    if spec.kind == "csv":
        _inspect_csv(path, spec, status)
    elif spec.kind == "json":
        _inspect_json(path, spec, status)
    elif spec.kind == "file":
        status["non_empty"] = status["size_bytes"] > 0
        status["schema_complete"] = True
    else:
        raise ValueError(f"unsupported artifact kind: {spec.kind}")
    return status


def _inspect_csv(path: Path, spec: ArtifactSpec, status: dict[str, Any]) -> None:
    try:
        frame = pd.read_csv(path)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as err:
        status["read_error"] = type(err).__name__
        return
    status["row_count"] = int(len(frame))
    status["non_empty"] = status["row_count"] > 0
    missing = sorted(set(spec.required_columns) - set(frame.columns))
    status["missing_columns"] = missing
    status["schema_complete"] = not missing


def _inspect_json(path: Path, spec: ArtifactSpec, status: dict[str, Any]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        status["read_error"] = type(err).__name__
        return
    status["non_empty"] = bool(payload)
    if isinstance(payload, Mapping):
        missing = sorted(set(spec.required_json_keys) - set(payload.keys()))
    else:
        missing = list(spec.required_json_keys)
    status["missing_json_keys"] = missing
    status["schema_complete"] = not missing
