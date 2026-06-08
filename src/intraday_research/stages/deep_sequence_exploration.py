"""Stage deep_sequence_exploration -- 08X schema-smoke first slice.

Spec: docs/superpowers/specs/2026-06-07-n08-08x-schema-smoke-harness-design.md
Codex review: .humanize/skill/2026-06-07_20-04-41-891-d296d492/

This slice migrates `run_stage` from a NotImplementedError skeleton to a
schema-smoke body gated on `RUN_08X_SCHEMA_SMOKE`. When the switch is True,
the stage emits all 8 section 13.1 08X artifacts in minimal-valid mode
(header-only CSVs, minimal JSON), each passing its contract validator. No
trial loop, no fold construction, no model fit, no official-validation read.

The package stage also supports RUN_08O_OFFICIAL_VALIDATION_READOUT from an
already-frozen official-validation prediction CSV. It appends the project
validation-budget ledger intent row before reading that CSV. Other RUN_08X_* /
RUN_08F_* / RUN_08O_* / BACKUP_* switches remain unmigrated and raise
NotImplementedError with the offending switch name.

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

import pandas as pd

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
from intraday_research.stages.deep_sequence_official_readout import (
    preflight_08o_prediction_rows,
    preflight_08o_static_inputs,
    resolve_08o_readout_inputs,
    write_08o_readout_artifacts,
    write_08o_run_manifest,
)
from intraday_research.stages.io_helpers import sha256_file
from intraday_research.stages.validation_budget_ledger import ValidationBudgetLedger


STAGE_NAME = "deep_sequence_exploration"
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "notebook07_validation_budget_ledger.csv",
)

SCHEMA_SMOKE_SWITCH = "RUN_08X_SCHEMA_SMOKE"
OFFICIAL_READOUT_SWITCH = "RUN_08O_OFFICIAL_VALIDATION_READOUT"
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
    # not yet migrate. Only explicitly handled switches are positive cases.
    handled_switches = {SCHEMA_SMOKE_SWITCH, OFFICIAL_READOUT_SWITCH}
    enabled_others = sorted(
        name for name, value in switches.items()
        if bool(value)
        and name not in handled_switches
        and name.startswith(_UNIMPLEMENTED_SWITCH_PREFIXES)
    )
    if enabled_others:
        raise NotImplementedError(
            f"Stage '{STAGE_NAME}' slice #5F-1 only implements "
            f"{sorted(handled_switches)}; the following enabled switches are "
            f"not yet migrated: {enabled_others}"
        )

    smoke_enabled = bool(switches.get(SCHEMA_SMOKE_SWITCH, False))
    readout_enabled = bool(switches.get(OFFICIAL_READOUT_SWITCH, False))
    if smoke_enabled and readout_enabled:
        raise ValueError(
            f"{SCHEMA_SMOKE_SWITCH} and {OFFICIAL_READOUT_SWITCH} must run in "
            "separate invocations"
        )
    if not smoke_enabled and not readout_enabled:
        logger.info(
            "stage %s: no run-switch enabled, exiting no-op", STAGE_NAME
        )
        return

    out = resolve_output_dir(config, output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if readout_enabled:
        _run_08o_official_readout(config, out)
        return

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


def _run_08o_official_readout(config: Mapping[str, Any], out: Path) -> None:
    """Write 08O artifacts from frozen official-validation prediction rows."""
    paths = resolve_08o_readout_inputs(config)
    preflight = preflight_08o_static_inputs(paths)
    freeze_record = preflight["freeze_record"]

    # AGENTS.md section 4.3: append intent BEFORE reading official-validation
    # prediction rows or metrics.
    ledger = ValidationBudgetLedger(paths["ledger"], appended_by_notebook="08O")
    ledger.append_row(
        "08o_run_manifest.json",
        "08O",
        "official_validation_readout_intent",
        "before_official_validation_read",
        "official_validation_prediction_rows",
        model_families_considered=str(freeze_record.get("architecture_family", "")),
        profiles_or_trials_considered="frozen_primary_candidate",
        seeds_used=str(len(freeze_record.get("frozen_seed_list", []))),
        thresholds_or_coverages_considered="n/a",
        official_validation_rows_inspected=0,
        train_inner_only_decision=False,
        official_validation_informed_decision=False,
        diagnostic_only_readout=False,
        holdout_test_contact=False,
        allowed_wording="pending_08o_manifest",
        forbidden_wording="no holdout / no deploy / no live",
        risk_note="08O package readout intent before reading official-validation prediction rows",
    )

    predictions = pd.read_csv(paths["predictions"])
    expected_tickers = _expected_08o_tickers(config)
    prediction_provenance = preflight_08o_prediction_rows(
        predictions,
        freeze_record=freeze_record,
        expected_tickers=expected_tickers,
    )
    write_08o_readout_artifacts(
        out,
        predictions,
        primary_candidate_id=str(freeze_record["primary_candidate_id"]),
        expected_seeds=tuple(prediction_provenance["seeds"]),
        expected_tickers=expected_tickers,
    )
    write_08o_run_manifest(
        out,
        freeze_record=freeze_record,
        static_input_provenance=preflight["static_input_provenance"],
        prediction_provenance={
            **prediction_provenance,
            "predictions_csv_path": str(paths["predictions"]),
            "predictions_csv_sha256": sha256_file(paths["predictions"]),
        },
        constants=_mapping_or_empty(config.get("constants", {})),
    )


def _expected_08o_tickers(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    inputs = _mapping_or_empty(config.get("inputs", {}))
    policy = _mapping_or_empty(config.get("policy", {}))
    official_validation = _mapping_or_empty(policy.get("official_validation", {}))
    tickers = (
        inputs.get("expected_tickers")
        or official_validation.get("expected_tickers")
        or config.get("expected_tickers")
    )
    if tickers is None:
        return None
    if not isinstance(tickers, (list, tuple)):
        raise ValueError("08O expected_tickers must be a list or tuple")
    return tuple(sorted(str(ticker) for ticker in tickers))


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected mapping configuration value")
    return value
