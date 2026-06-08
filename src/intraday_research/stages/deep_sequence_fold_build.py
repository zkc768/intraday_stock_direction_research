"""08X RUN_08X_BUILD_TRAIN_INNER_FOLDS -- build train-inner folds, write results.

Spec: docs/superpowers/specs/2026-06-08-n08-08x-fold-build-design.md (#5F-2).
Codex design review: .humanize/skill/2026-06-07_23-51-18-53-08e6b005/

Wires the existing ``models/deep_sequence/folds.py`` builders into the package
stage. Given a windowed sample index, it restricts to the OFFICIAL-TRAIN
partition (AGENTS.md 4.1 / tech design 8.1 -- 08X never folds or reads the
official-validation or closed-holdout partitions), runs the explicitly selected
fold scheme(s), and writes real ``08x_fold_results.csv`` rows plus the fold
policy fields of ``08x_run_manifest.json``.

NO model fit, NO official-validation read. ``validation_design.fold_modes`` in
the stage YAML is an ALLOW-LIST, not the plan: only the explicitly selected
scheme(s) run (default: ``rolling_origin_folds`` -- the forward-chaining mode).
Fold construction failure (e.g. too few samples) fails loud; per-trial failure
rows are a later (fit) slice concern.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    FOLD_RESULTS_REQUIRED_COLUMNS,
    FOLD_SCHEMES,
    REQUIRED_08X_RUN_MANIFEST_FIELDS,
    validate_08x_fold_results_frame,
    validate_08x_run_manifest,
)
from intraday_research.data.splits import PARTITION_TRAIN, PARTITION_VALIDATION
from intraday_research.models.deep_sequence.folds import (
    embargoed_train_inner_folds,
    purged_time_series_folds,
    rolling_origin_folds,
)
from intraday_research.stages.deep_sequence_schema_smoke import (
    FOLD_RESULTS_COLUMNS,
    SCHEMA_SMOKE_VERSION,
    write_schema_smoke_artifacts,
)
from intraday_research.stages.run_manifest import write_run_manifest

# Fail loud at import time if the ordered column tuple drifts from the contract
# set -- the writer relies on the tuple for column ORDER, the validator on the set.
if set(FOLD_RESULTS_COLUMNS) != FOLD_RESULTS_REQUIRED_COLUMNS:
    raise RuntimeError(
        "FOLD_RESULTS_COLUMNS (schema_smoke) does not match "
        "FOLD_RESULTS_REQUIRED_COLUMNS (contract)"
    )

DEFAULT_SELECTED_FOLD_MODES: tuple[str, ...] = ("rolling_origin_folds",)
_TIER_N_FOLDS: dict[str, int] = {
    "quick": 2,
    "medium": 3,
    "aggressive": 3,
    "schema_smoke": 2,
    "build_folds": 2,
}


@dataclass(frozen=True)
class FoldSpec:
    """One fold scheme + its materialized numeric parameters."""

    scheme: str
    n_folds: int
    label_horizon_k: int
    inner_validation_size: int
    embargo_size: int


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int (no bool/float); got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; got {value}")
    return value


def build_fold_plan(config: Mapping[str, Any]) -> list[FoldSpec]:
    """Materialize the explicit fold plan from ``config['fold_plan']``.

    ``fold_plan`` fields: ``selected_fold_modes`` (default
    ``["rolling_origin_folds"]``), ``n_folds`` (or derived from
    ``search_budget_tier``), ``label_horizon_k`` (required), ``inner_validation_size``
    (default 1), ``embargo_size`` (default 0).
    """
    fold_plan = config.get("fold_plan", {})
    if not isinstance(fold_plan, Mapping):
        raise ValueError("config['fold_plan'] must be a mapping")

    selected = fold_plan.get("selected_fold_modes", list(DEFAULT_SELECTED_FOLD_MODES))
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("fold_plan.selected_fold_modes must be a non-empty list")
    unknown = [scheme for scheme in selected if scheme not in FOLD_SCHEMES]
    if unknown:
        raise ValueError(
            f"fold_plan.selected_fold_modes has unknown schemes: {sorted(unknown)} "
            f"(allowed: {sorted(FOLD_SCHEMES)})"
        )

    n_folds = fold_plan.get("n_folds")
    if n_folds is None:
        tier = str(
            config.get("search_budget_tier") or fold_plan.get("budget_tier") or "quick"
        )
        n_folds = _TIER_N_FOLDS.get(tier, 2)
    n_folds = _require_int(n_folds, "fold_plan.n_folds", minimum=1)

    if "label_horizon_k" not in fold_plan:
        raise ValueError("fold_plan.label_horizon_k is required")
    label_horizon_k = _require_int(
        fold_plan["label_horizon_k"], "fold_plan.label_horizon_k", minimum=0
    )
    inner_validation_size = _require_int(
        fold_plan.get("inner_validation_size", 1),
        "fold_plan.inner_validation_size",
        minimum=1,
    )
    embargo_size = _require_int(
        fold_plan.get("embargo_size", 0), "fold_plan.embargo_size", minimum=0
    )

    return [
        FoldSpec(
            scheme=scheme,
            n_folds=n_folds,
            label_horizon_k=label_horizon_k,
            inner_validation_size=inner_validation_size,
            embargo_size=embargo_size,
        )
        for scheme in selected
    ]


def _build_for_scheme(
    timestamps: np.ndarray, ticker_ids: np.ndarray, spec: FoldSpec
):
    if spec.scheme == "rolling_origin_folds":
        return rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=spec.n_folds,
            inner_validation_size=spec.inner_validation_size,
            label_horizon_k=spec.label_horizon_k,
        )
    if spec.scheme == "purged_time_series_folds":
        return purged_time_series_folds(
            timestamps,
            ticker_ids,
            n_folds=spec.n_folds,
            label_horizon_k=spec.label_horizon_k,
        )
    if spec.scheme == "embargoed_train_inner_folds":
        return embargoed_train_inner_folds(
            timestamps,
            ticker_ids,
            n_folds=spec.n_folds,
            label_horizon_k=spec.label_horizon_k,
            embargo_size=spec.embargo_size,
        )
    raise ValueError(f"unknown fold scheme: {spec.scheme!r}")


def build_fold_results(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    fold_plan: Sequence[FoldSpec],
) -> pd.DataFrame:
    """Pure executor: run each scheme, emit one row per ``(scheme, fold)``.

    ``timestamps`` / ``ticker_ids`` are the TRAIN-PARTITION windowed index. The
    fold builders enforce the per-ticker chronological split + label-horizon
    purge/embargo; this function only tabulates their fold sizes.
    """
    timestamps = np.asarray(timestamps)
    ticker_ids = np.asarray(ticker_ids)
    rows: list[dict[str, Any]] = []
    for spec in fold_plan:
        for split_index, (train_idx, val_idx) in enumerate(
            _build_for_scheme(timestamps, ticker_ids, spec)
        ):
            embargo = spec.embargo_size if spec.scheme == "embargoed_train_inner_folds" else 0
            rows.append(
                {
                    "fold_id": f"{spec.scheme}__{split_index}",
                    "fold_scheme": spec.scheme,
                    "split_index": int(split_index),
                    "train_inner_fit_n": int(len(train_idx)),
                    "train_inner_validation_n": int(len(val_idx)),
                    "purge_gap_k": int(spec.label_horizon_k),
                    "embargo_gap_k": int(embargo),
                }
            )
    df = pd.DataFrame(rows, columns=list(FOLD_RESULTS_COLUMNS))
    validate_08x_fold_results_frame(df, require_non_empty=False)
    return df


def resolve_train_inner_index(
    config: Mapping[str, Any],
    injected_window_index: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(timestamps, ticker_ids)`` for the OFFICIAL-TRAIN partition only.

    The windowed index is taken from ``injected_window_index`` (tests / runner)
    or ``config['windowed_index']``. Auto-loading the windowed index from raw
    bars (``raw_bars -> features -> labels -> splits -> windows``) is deferred to
    the next slice; until then this fails loud rather than silently returning an
    empty index.
    """
    window_index = injected_window_index
    if window_index is None:
        window_index = config.get("windowed_index")
    if window_index is None:
        raise NotImplementedError(
            "real-data windowed-index loading from raw bars is a later slice "
            "(#5F-3); pass injected_window_index or config['windowed_index'] "
            "with target_partition / target_timestamps / target_ticker_ids"
        )
    if not isinstance(window_index, Mapping):
        raise ValueError("windowed_index must be a mapping of numpy arrays")

    partition = np.asarray(window_index["target_partition"])
    timestamps = np.asarray(window_index["target_timestamps"])
    ticker_ids = np.asarray(window_index["target_ticker_ids"])
    if not (len(partition) == len(timestamps) == len(ticker_ids)):
        raise ValueError(
            "windowed_index arrays must be equal length; got "
            f"partition={len(partition)}, timestamps={len(timestamps)}, "
            f"ticker_ids={len(ticker_ids)}"
        )
    # Codex impl review P1-2: a `== PARTITION_TRAIN` filter silently drops any
    # unexpected partition code. Assert the domain is exactly {train, validation}
    # so a mislabeled/foreign code fails loud rather than being dropped. (This
    # trusts the upstream split's code assignment; it cannot catch a row wrongly
    # labeled train.)
    allowed_codes = {int(PARTITION_TRAIN), int(PARTITION_VALIDATION)}
    seen_codes = {int(code) for code in np.unique(partition)}
    unexpected = seen_codes - allowed_codes
    if unexpected:
        raise ValueError(
            f"windowed_index has unexpected partition codes {sorted(unexpected)}; "
            f"08X expects only train={int(PARTITION_TRAIN)} / "
            f"validation={int(PARTITION_VALIDATION)}"
        )
    train_mask = partition == PARTITION_TRAIN
    if not bool(train_mask.any()):
        raise ValueError(
            "windowed_index has no PARTITION_TRAIN rows; 08X requires "
            "train-inner rows to fold"
        )
    return timestamps[train_mask], ticker_ids[train_mask]


def write_fold_results(out: Path, df: pd.DataFrame) -> None:
    """Validate (require_non_empty) and overwrite ``08x_fold_results.csv``."""
    validate_08x_fold_results_frame(df, require_non_empty=True)
    df.to_csv(out / "08x_fold_results.csv", index=False)


def _default_fold_run_manifest() -> dict[str, Any]:
    return {
        "notebook08_version": SCHEMA_SMOKE_VERSION,
        "stage": "08X",
        "scope": "exploratory",
        "source_stage0_candidate": "build_folds_no_candidate",
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "train_inner_fold_policy": "none",
        "purge_policy": "none",
        "embargo_policy": "none",
        "search_budget_tier": "build_folds",
        "trial_count_requested": 0,
        "trial_count_completed": 0,
        "trial_count_failed": 0,
        "trial_count_skipped": 0,
    }


def write_fold_run_manifest(
    out: Path, fold_plan: Sequence[FoldSpec], *, candidate_id: str
) -> None:
    """Update the fold-policy + candidate-provenance fields of the run manifest.

    Reads an existing manifest if present (preserving its other fields) or builds
    a minimal valid skeleton, then overwrites the three policy fields and
    ``source_stage0_candidate`` with the real frozen-candidate id (Codex impl
    review P1-1: a clean-dir run lays a schema-smoke skeleton whose
    ``source_stage0_candidate`` is the placeholder string, which must not survive).
    """
    manifest_path = out / "08x_run_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = _default_fold_run_manifest()

    payload["source_stage0_candidate"] = candidate_id
    modes = [spec.scheme for spec in fold_plan]
    label_horizon_k = fold_plan[0].label_horizon_k if fold_plan else 0
    embargo = max(
        (spec.embargo_size for spec in fold_plan
         if spec.scheme == "embargoed_train_inner_folds"),
        default=0,
    )
    payload["train_inner_fold_policy"] = "+".join(modes)
    payload["purge_policy"] = f"horizon_bar_purge_k={label_horizon_k}"
    payload["embargo_policy"] = (
        "none" if embargo == 0 else f"symmetric_embargo_k={embargo}"
    )

    write_run_manifest(
        manifest_path,
        payload,
        validator=validate_08x_run_manifest,
        required_fields=sorted(REQUIRED_08X_RUN_MANIFEST_FIELDS),
        stage="08X",
        scope="exploratory",
        false_fields=("official_validation_used", "holdout_test_authorized"),
    )


def run_build_train_inner_folds(
    config: Mapping[str, Any], out: Path
) -> pd.DataFrame:
    """Dispatch body for RUN_08X_BUILD_TRAIN_INNER_FOLDS.

    1. resolve the train-partition windowed index;
    2. materialize the fold plan;
    3. provenance gate: fold label_horizon_k == frozen candidate horizon_k;
    4. build + write real fold_results;
    5. update run-manifest fold policies.
    """
    # Codex impl review P1-1: folds must not change under existing trial rows.
    _refuse_rebuild_over_existing_trials(out)

    timestamps, ticker_ids = resolve_train_inner_index(config)
    fold_plan = build_fold_plan(config)
    candidate = _frozen_candidate(config)
    _assert_label_horizon_provenance(candidate, fold_plan)

    # Lay down the rest of the 08X bundle as skeletons only when the output dir
    # has no prior 08X artifacts; never clobber a prior smoke/search run.
    if not (out / "08x_search_space.json").exists():
        write_schema_smoke_artifacts(out)

    df = build_fold_results(timestamps, ticker_ids, fold_plan)
    write_fold_results(out, df)
    write_fold_run_manifest(
        out, fold_plan, candidate_id=_candidate_provenance_id(candidate)
    )
    return df


def _refuse_rebuild_over_existing_trials(out: Path) -> None:
    """Fail loud if ``08x_trial_ledger.csv`` already carries trial rows.

    Rebuilding folds under existing trial evidence would silently change the
    train/validation split the trials were scored on (Codex impl review P1-1).
    """
    ledger_path = out / "08x_trial_ledger.csv"
    if not ledger_path.exists():
        return
    try:
        existing = pd.read_csv(ledger_path)
    except pd.errors.EmptyDataError:
        return
    if not existing.empty:
        raise ValueError(
            f"refusing to rebuild folds: {ledger_path.name} has "
            f"{len(existing)} trial rows; folds must not change under existing "
            "trial evidence"
        )


def _frozen_candidate(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen Stage-0 candidate mapping (config['frozen_candidate']).

    Renamed from ``source_stage0_candidate`` (Codex impl review P5) because the
    08X run manifest stores ``source_stage0_candidate`` as a STRING id, while the
    fold build needs a MAPPING with ``horizon_k``; sharing the name was confusing.
    """
    candidate = config.get("frozen_candidate")
    if not isinstance(candidate, Mapping) or "horizon_k" not in candidate:
        raise ValueError(
            "RUN_08X_BUILD_TRAIN_INNER_FOLDS requires config['frozen_candidate'] "
            "to be a mapping with 'horizon_k' to provenance the fold purge"
        )
    return candidate


def _candidate_provenance_id(candidate: Mapping[str, Any]) -> str:
    """Stable string id written into the manifest's ``source_stage0_candidate``."""
    candidate_id = candidate.get("candidate_id")
    if candidate_id:
        return str(candidate_id)
    return f"frozen_candidate_horizon_k={int(candidate['horizon_k'])}"


def _assert_label_horizon_provenance(
    candidate: Mapping[str, Any], fold_plan: Sequence[FoldSpec]
) -> None:
    """Fold purge width must equal the frozen candidate ``horizon_k`` (Codex Q5).

    Keeps the purge pre-registered: an unprovenanced purge is not auditable.
    """
    horizon_k = _require_int(
        candidate["horizon_k"], "frozen_candidate.horizon_k", minimum=0
    )
    mismatched = sorted(
        {spec.label_horizon_k for spec in fold_plan if spec.label_horizon_k != horizon_k}
    )
    if mismatched:
        raise ValueError(
            "fold_plan.label_horizon_k "
            f"{mismatched} != frozen candidate horizon_k {horizon_k}; "
            "the fold purge must match the pre-registered label horizon"
        )
