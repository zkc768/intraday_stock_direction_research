"""Schema-smoke artifact writers for the Notebook 08 package stage."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.artifact_preflight import (
    ArtifactSpec,
    assert_artifact_bundle_complete,
    csv_artifact,
    json_artifact,
)
from intraday_research.contracts.deep_sequence_exploration import (
    OUTPUT_FILES_08X,
    REQUIRED_08X_RUN_MANIFEST_FIELDS,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    validate_08x_run_manifest,
    validate_08x_search_space,
    validate_trial_ledger_frame,
)
from intraday_research.stages.io_helpers import (
    sha256_bytes,
    sha256_file_or_unavailable,
    write_json,
)
from intraday_research.stages.run_manifest import (
    write_run_manifest as write_run_manifest_json,
)


SCHEMA_SMOKE_VERSION = "08x_schema_smoke_v1"

FOLD_RESULTS_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "fold_scheme",
    "split_index",
    "train_inner_fit_n",
    "train_inner_validation_n",
    "purge_gap_k",
    "embargo_gap_k",
)
SEED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "metric",
    "seed_mean",
    "seed_std",
    "seed_lcb_95",
)
FAILURE_LEDGER_COLUMNS: tuple[str, ...] = (
    "trial_id",
    "failure_type",
    "failure_message",
    "fold_id",
    "seed",
    "candidate_family",
    "candidate_id",
)
CANDIDATE_COMPRESSION_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "candidate_family",
    "paper_safe_score",
    "z_lcb_delta",
    "z_mean_delta",
    "z_seed_stability",
    "z_fold_consistency",
    "z_per_ticker",
    "complexity_penalty",
    "compute_penalty",
    "compute_tier",
)
ENV_MANIFEST_KEYS: tuple[str, ...] = (
    "manifest_mode",
    "python_version",
    "python_executable_sha256",
    "pip_freeze_sha256",
    "dependency_versions",
    "platform",
    "git_commit",
    "git_dirty",
)
SEARCH_SPACE_KEYS: tuple[str, ...] = (
    "search_space_version",
    "stage",
    "scope",
    "architecture_families",
    "hpo_method",
    "eligibility_thresholds",
    "scientific_budget_cap_total_trials",
    "per_family_trial_budget",
    "low_compute_mode",
    "low_compute_submode",
    "seed_list",
    "deferred_07g_gaps",
    "official_validation_used",
    "holdout_test_authorized",
)
SCHEMA_SMOKE_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    json_artifact("08x_search_space.json", SEARCH_SPACE_KEYS),
    csv_artifact(
        "08x_trial_ledger.csv",
        sorted(REQUIRED_TRIAL_LEDGER_COLUMNS),
        require_non_empty=False,
    ),
    csv_artifact("08x_fold_results.csv", FOLD_RESULTS_COLUMNS, require_non_empty=False),
    csv_artifact("08x_seed_summary.csv", SEED_SUMMARY_COLUMNS, require_non_empty=False),
    csv_artifact(
        "08x_failure_ledger.csv", FAILURE_LEDGER_COLUMNS, require_non_empty=False
    ),
    csv_artifact(
        "08x_candidate_compression_table.csv",
        CANDIDATE_COMPRESSION_COLUMNS,
        require_non_empty=False,
    ),
    json_artifact("08x_run_manifest.json", sorted(REQUIRED_08X_RUN_MANIFEST_FIELDS)),
    json_artifact("08x_environment_manifest.json", ENV_MANIFEST_KEYS),
)


def resolve_output_dir(config: Mapping[str, Any], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    outputs = config.get("outputs", {})
    results_dir = outputs.get("results_dir") if isinstance(outputs, Mapping) else None
    if not results_dir:
        raise ValueError(
            "run_stage requires output_dir kwarg or config['outputs']['results_dir']"
        )
    return Path(results_dir)


def write_schema_smoke_artifacts(out: Path) -> None:
    _write_search_space(out)
    _write_trial_ledger_header(out)
    _write_fold_results_header(out)
    _write_seed_summary_header(out)
    _write_failure_ledger_header(out)
    _write_candidate_compression_header(out)
    _write_run_manifest(out)
    _write_environment_manifest(out)
    assert_artifact_bundle_complete(out, SCHEMA_SMOKE_ARTIFACT_SPECS)


def _write_search_space(out: Path) -> None:
    payload = {
        "search_space_version": SCHEMA_SMOKE_VERSION,
        "stage": "08X",
        "scope": "exploratory",
        "architecture_families": ["dlinear_only"],
        "hpo_method": "random_search",
        "eligibility_thresholds": {
            "min_train_inner_lcb_delta_macro_f1": 0.005,
        },
        "scientific_budget_cap_total_trials": 1,
        "per_family_trial_budget": {"dlinear_only": 1},
        "low_compute_mode": False,
        "low_compute_submode": "",
        "seed_list": [],
        "deferred_07g_gaps": {},
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }
    validate_08x_search_space(payload)
    write_json(out / "08x_search_space.json", payload)


def _write_trial_ledger_header(out: Path) -> None:
    columns = sorted(REQUIRED_TRIAL_LEDGER_COLUMNS)
    df = pd.DataFrame(columns=columns)
    validate_trial_ledger_frame(df)
    df.to_csv(out / "08x_trial_ledger.csv", index=False)


def _write_fold_results_header(out: Path) -> None:
    pd.DataFrame(columns=list(FOLD_RESULTS_COLUMNS)).to_csv(
        out / "08x_fold_results.csv", index=False
    )


def _write_seed_summary_header(out: Path) -> None:
    pd.DataFrame(columns=list(SEED_SUMMARY_COLUMNS)).to_csv(
        out / "08x_seed_summary.csv", index=False
    )


def _write_failure_ledger_header(out: Path) -> None:
    pd.DataFrame(columns=list(FAILURE_LEDGER_COLUMNS)).to_csv(
        out / "08x_failure_ledger.csv", index=False
    )


def _write_candidate_compression_header(out: Path) -> None:
    pd.DataFrame(columns=list(CANDIDATE_COMPRESSION_COLUMNS)).to_csv(
        out / "08x_candidate_compression_table.csv", index=False
    )


def _write_run_manifest(out: Path) -> None:
    payload = {
        "notebook08_version": SCHEMA_SMOKE_VERSION,
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
    write_run_manifest_json(
        out / "08x_run_manifest.json",
        payload,
        validator=validate_08x_run_manifest,
        stage="08X",
        scope="exploratory",
        false_fields=("official_validation_used", "holdout_test_authorized"),
    )


def _write_environment_manifest(out: Path) -> None:
    payload = {
        "manifest_mode": "schema_smoke",
        "python_version": sys.version.split(" ", 1)[0],
        "python_executable_sha256": sha256_file_or_unavailable(Path(sys.executable)),
        "pip_freeze_sha256": _pip_freeze_sha256(),
        "dependency_versions": _collect_dependency_versions(),
        "platform": sys.platform,
        "git_commit": _git_head_sha(),
        "git_dirty": _git_dirty(),
    }
    write_json(out / "08x_environment_manifest.json", payload)


def _pip_freeze_sha256() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unavailable"
    lines = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    return sha256_bytes("\n".join(lines).encode("utf-8"))


def _collect_dependency_versions() -> dict[str, str]:
    packages = ("torch", "scikit-learn", "numpy", "pandas", "lightgbm")
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version as package_version
    except ImportError:
        return {pkg: "unavailable" for pkg in packages}
    for pkg in packages:
        try:
            versions[pkg] = package_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "absent"
    return versions


def _git_repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _git_head_sha() -> str:
    repo = _git_repo_root()
    if repo is None:
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def _git_dirty() -> bool | None:
    repo = _git_repo_root()
    if repo is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return bool(result.stdout.strip())
