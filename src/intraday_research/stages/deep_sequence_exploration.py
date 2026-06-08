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

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intraday_research.contracts.deep_sequence_exploration import (
    OUTPUT_FILES_08X,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
)
from intraday_research.stages.deep_sequence_schema_smoke import (
    CANDIDATE_COMPRESSION_COLUMNS,
    ENV_MANIFEST_KEYS,
    FAILURE_LEDGER_COLUMNS,
    FOLD_RESULTS_COLUMNS,
    SCHEMA_SMOKE_VERSION,
    SEED_SUMMARY_COLUMNS,
    resolve_output_dir,
    write_schema_smoke_artifacts,
)


STAGE_NAME = "deep_sequence_exploration"
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "notebook07_validation_budget_ledger.csv",
)

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

    out = resolve_output_dir(config, output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("stage %s: schema-smoke writing artifacts to %s", STAGE_NAME, out)

    write_schema_smoke_artifacts(out)

    written = sorted(p.name for p in out.iterdir() if p.is_file())
    expected = sorted(OUTPUT_FILES_08X)
    missing = set(expected) - set(written)
    if missing:
        raise RuntimeError(
            f"schema-smoke missed expected artifacts: {sorted(missing)} "
            f"(written: {written})"
        )
