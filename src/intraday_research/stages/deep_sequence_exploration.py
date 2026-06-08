"""Stage deep_sequence_exploration -- 08X schema-smoke first slice.

Spec: docs/superpowers/specs/2026-06-07-n08-08x-schema-smoke-harness-design.md
Codex review: .humanize/skill/2026-06-07_20-04-41-891-d296d492/

This slice migrates `run_stage` from a NotImplementedError skeleton to a
schema-smoke body gated on `RUN_08X_SCHEMA_SMOKE`. When the switch is True,
the stage emits all 8 section 13.1 08X artifacts in minimal-valid mode
(header-only CSVs, minimal JSON), each passing its contract validator. No
trial loop, no fold construction, no model fit, no official-validation read.

Other RUN_08X_* / RUN_08F_* / RUN_08O_* / BACKUP_* switches are not migrated
in this slice and raise NotImplementedError with the offending switch name.

Governance supersession (recorded in spec section 7): tech design section 6.1's
"no active import from intraday_research" is stale frozen notebook-posture
text superseded by AGENTS.md / CODE_ORGANIZATION.md / NOTEBOOK08_RESUME_GATES.md
/ configs/pipeline.yaml. Package-first is canonical; substantive 08X work
lives here.

For N07/N08: see AGENTS.md section 4.3 -- any read of official-validation
metrics MUST append a ledger row BEFORE reading; pre-existing rows MUST NOT
be modified, dropped, or reordered. Schema-smoke does not read official
validation, so section 4.3 does not apply to this slice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    OUTPUT_FILES_08X,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    validate_08x_run_manifest,
    validate_08x_search_space,
    validate_trial_ledger_frame,
)


STAGE_NAME = "deep_sequence_exploration"
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "notebook07_validation_budget_ledger.csv",
)

SCHEMA_SMOKE_VERSION = "08x_schema_smoke_v1"
SCHEMA_SMOKE_SWITCH = "RUN_08X_SCHEMA_SMOKE"
# Explicit enumeration of the 13 non-smoke switches declared in
# `configs/stages/deep_sequence_exploration.yaml`. Used by the parametrized
# regression test to confirm each is rejected by name. The impl rejects ANY
# truthy `RUN_*` / `BACKUP_*` switch that is not `SCHEMA_SMOKE_SWITCH`
# (Codex impl review P1-1: forward-compat against future YAML additions).
OTHER_SWITCHES: tuple[str, ...] = (
    "RUN_08X_BUILD_TRAIN_INNER_FOLDS",
    "RUN_08X_SEARCH_SPACE_DRY_RUN",
    "RUN_08X_QUICK_SEARCH",
    "RUN_08X_MEDIUM_SEARCH",
    "RUN_08X_AGGRESSIVE_SEARCH",
    "RUN_08X_AGGREGATE_FAILURE_MAP",
    "RUN_08F_CONTRACT_GATE",
    "RUN_08F_CANDIDATE_COMPRESSION",
    "RUN_08F_WRITE_FREEZE_RECORD",
    "RUN_08O_ENTRY_GATE",
    "RUN_08O_OFFICIAL_VALIDATION_READOUT",
    "RUN_08O_AGGREGATE_AND_WRITE_MANIFEST",
    "BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE",
)
_UNIMPLEMENTED_SWITCH_PREFIXES: tuple[str, ...] = ("RUN_", "BACKUP_")

# Header columns for the four CSV ledgers without a contract validator. Mirrors
# tech design sections 8.2 / 8.4 / 9.2 / 13.3 -- spec section 4.2.
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

logger = logging.getLogger(__name__)


def run_stage(
    config: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> None:
    """Run the 08X schema-smoke harness.

    Gated on config["run_switches"]["RUN_08X_SCHEMA_SMOKE"]. Default False:
    no-op, log "no work ran". When True, emits 8 section 13.1 08X artifacts
    in minimal-valid mode through their contract validators.

    Any other RUN_* switch set to True raises NotImplementedError.

    Output directory resolution:
        1. ``output_dir`` kwarg if provided
        2. else Path(config["outputs"]["results_dir"])
    """
    switches = dict(config.get("run_switches", {}))

    # Codex impl review P1-1: prefix-based detection catches future
    # RUN_*/BACKUP_* switches that the YAML may grow but this slice does
    # not yet migrate. SCHEMA_SMOKE_SWITCH is the sole positive case.
    enabled_others = sorted(
        name for name, value in switches.items()
        if bool(value)
        and name != SCHEMA_SMOKE_SWITCH
        and name.startswith(_UNIMPLEMENTED_SWITCH_PREFIXES)
    )
    if enabled_others:
        raise NotImplementedError(
            f"Stage '{STAGE_NAME}' slice #5F-1 only implements "
            f"{SCHEMA_SMOKE_SWITCH}; the following enabled switches are not "
            f"yet migrated: {enabled_others}"
        )

    smoke_enabled = bool(switches.get(SCHEMA_SMOKE_SWITCH, False))
    if not smoke_enabled:
        logger.info(
            "stage %s: no run-switch enabled, exiting no-op", STAGE_NAME
        )
        return

    out = _resolve_output_dir(config, output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("stage %s: schema-smoke writing artifacts to %s", STAGE_NAME, out)

    _write_search_space(out)
    _write_trial_ledger_header(out)
    _write_fold_results_header(out)
    _write_seed_summary_header(out)
    _write_failure_ledger_header(out)
    _write_candidate_compression_header(out)
    _write_run_manifest(out)
    _write_environment_manifest(out)

    written = sorted(p.name for p in out.iterdir() if p.is_file())
    expected = sorted(OUTPUT_FILES_08X)
    missing = set(expected) - set(written)
    if missing:
        raise RuntimeError(
            f"schema-smoke missed expected artifacts: {sorted(missing)} "
            f"(written: {written})"
        )


# ---------------------------------------------------------------------------
# Output directory resolution
# ---------------------------------------------------------------------------


def _resolve_output_dir(
    config: Mapping[str, Any], output_dir: Path | None
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    outputs = config.get("outputs", {})
    results_dir = outputs.get("results_dir") if isinstance(outputs, Mapping) else None
    if not results_dir:
        raise ValueError(
            "run_stage requires output_dir kwarg or config['outputs']['results_dir']"
        )
    return Path(results_dir)


# ---------------------------------------------------------------------------
# Artifact writers (each validates via contract module where a validator exists)
# ---------------------------------------------------------------------------


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
    (out / "08x_search_space.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    validate_08x_run_manifest(payload)
    (out / "08x_run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_environment_manifest(out: Path) -> None:
    payload = {
        "manifest_mode": "schema_smoke",
        "python_version": sys.version.split(" ", 1)[0],
        "python_executable_sha256": _sha256_file(Path(sys.executable)),
        "pip_freeze_sha256": _sha256_pip_freeze(),
        "dependency_versions": _collect_dependency_versions(),
        "platform": sys.platform,
        "git_commit": _git_head_sha(),
        "git_dirty": _git_dirty(),
    }
    (out / "08x_environment_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Environment-manifest helpers (no contract validator yet; spec Q5 / R1)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    try:
        data = path.read_bytes()
    except (OSError, FileNotFoundError):
        return "unavailable"
    return hashlib.sha256(data).hexdigest()


def _sha256_pip_freeze() -> str:
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
    canonical = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _collect_dependency_versions() -> dict[str, str]:
    """Best-effort version lookup for a fixed set of research-critical packages.

    Schema-smoke records what is importable in the current env; missing
    packages are recorded as "absent" rather than raising, so a smoke run
    succeeds in environments where (e.g.) lightgbm is not yet installed.
    """
    packages = ("torch", "scikit-learn", "numpy", "pandas", "lightgbm")
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version as _version
    except ImportError:
        return {pkg: "unavailable" for pkg in packages}
    for pkg in packages:
        try:
            versions[pkg] = _version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "absent"
    return versions


def _git_repo_root() -> Path | None:
    """Find the .git-bearing ancestor of this module's source file.

    Walking from __file__ avoids depending on the caller's cwd (Codex
    impl review P1-2). Returns None if no .git directory is found, which
    happens in installed-package scenarios where the source lives under
    site-packages rather than a checkout.
    """
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
    """Honest dirty check via `git status --porcelain --untracked-files=normal`.

    Codex impl review P1-2: `git diff --quiet HEAD` ignores untracked
    files, so an in-progress new file would appear clean. Porcelain
    status catches modifications, untracked, and unmerged paths.
    """
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
