"""Run-manifest helpers for package-backed research stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from intraday_research.stages.io_helpers import write_json

RunManifestValidator = Callable[[dict[str, Any]], None]


def validate_run_manifest_payload(
    payload: Mapping[str, Any],
    *,
    required_fields: Iterable[str] = (),
    stage: str | None = None,
    scope: str | None = None,
    false_fields: Iterable[str] = (),
) -> None:
    """Validate generic stage/scope/boolean guards before writing a manifest."""
    missing = set(required_fields) - set(payload.keys())
    if missing:
        raise AssertionError(f"run_manifest missing fields: {sorted(missing)}")
    if stage is not None and payload.get("stage") != stage:
        raise AssertionError(
            f"run_manifest stage must be {stage!r}; got {payload.get('stage')!r}"
        )
    if scope is not None and payload.get("scope") != scope:
        raise AssertionError(
            f"run_manifest scope must be {scope!r}; got {payload.get('scope')!r}"
        )
    for field in false_fields:
        if bool(payload.get(field)):
            raise AssertionError(f"run_manifest {field}=True is forbidden")


def write_run_manifest(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    validator: RunManifestValidator | None = None,
    required_fields: Iterable[str] = (),
    stage: str | None = None,
    scope: str | None = None,
    false_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate and write a deterministic JSON run manifest.

    The caller owns research-stage-specific schema details. This helper keeps
    the write path deterministic and fail-closed without reaching into data,
    notebooks, training, or holdout/test artifacts.
    """
    manifest = dict(payload)
    validate_run_manifest_payload(
        manifest,
        required_fields=required_fields,
        stage=stage,
        scope=scope,
        false_fields=false_fields,
    )
    if validator is not None:
        validator(manifest)
    write_json(path, manifest)
    return manifest
