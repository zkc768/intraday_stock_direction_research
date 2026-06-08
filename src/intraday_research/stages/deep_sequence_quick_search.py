"""08X RUN_08X_QUICK_SEARCH (#5F-6) -- the quick-tier trial loop + evidence.

Spec: docs/superpowers/specs/2026-06-08-n08-08x-quick-search-design.md
Codex design review: .humanize/skill/2026-06-08_02-17-21-248-0497f9f5/

Runs a finite, pre-registered ``(candidate x fold x seed)`` loop for the QUICK
budget tier (section 11: 4-8 configs, 1-2 folds, 1-2 seeds), calling the #5F-5
``run_single_trial`` atom for each, then assembles the section 8.3 trial ledger,
the section 8.4 failure ledger, and the section 14.1 per-(candidate, metric) seed
summary. Train-inner only; no official-validation / holdout contact (AGENTS 4.1).

Preregistration order (Codex #5F-6 P1-3): the search space (families, candidates,
config hashes, HPO method, seeds, budgets, ``search_space_sha256``) is materialized,
validated, and persisted to ``08x_search_space.json`` BEFORE trial 0. Folds are
built once from ``build_fold_plan`` over the masked train arrays and hashed into the
run manifest so the evidence is provably scored on the recorded folds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA,
    CLASS_COLLAPSE_PRED_RATE_MIN,
    HPO_METHODS,
    REQUIRED_08X_RUN_MANIFEST_FIELDS,
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES,
    TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES,
    validate_08x_fold_results_frame,
    validate_08x_per_ticker_frame,
    validate_08x_run_manifest,
    validate_08x_search_space,
    validate_trial_ledger_frame,
)
from intraday_research.models.deep_sequence.registry import (
    SEQUENCE_CLASSIFIER_REGISTRY,
)
from intraday_research.stages.deep_sequence_fold_build import (
    _assert_label_horizon_provenance,
    _build_for_scheme,
    _candidate_provenance_id,
    _frozen_candidate,
    FoldSpec,
    build_fold_plan,
    fold_assignment_sha256,
    resolve_train_inner_arrays,
    train_inner_index_sha256,
)
from intraday_research.stages.deep_sequence_schema_smoke import (
    FAILURE_LEDGER_COLUMNS,
    FOLD_RESULTS_COLUMNS,
    PER_TICKER_COLUMNS,
    SEED_SUMMARY_COLUMNS,
    write_schema_smoke_artifacts,
)
from intraday_research.stages.deep_sequence_trial import run_single_trial
from intraday_research.stages.io_helpers import (
    sha256_bytes,
    sha256_file_or_unavailable,
    write_json,
)
from intraday_research.stages.run_manifest import write_run_manifest

QUICK_SEARCH_VERSION = "08x_quick_search_v1"

# Section 11 budget tier limits (§11).
TIER_LIMITS: dict[str, dict[str, int]] = {
    "quick": {"min_candidates": 4, "max_candidates": 8, "max_folds": 2, "max_seeds": 2},
    "medium": {"min_candidates": 20, "max_candidates": 40, "max_folds": 3, "max_seeds": 3},
    "aggressive": {"min_candidates": 80, "max_candidates": 200, "max_folds": 5, "max_seeds": 5},
}

# Backward-compat aliases used by existing tests.
QUICK_MIN_CANDIDATES = TIER_LIMITS["quick"]["min_candidates"]
QUICK_MAX_CANDIDATES = TIER_LIMITS["quick"]["max_candidates"]
QUICK_MAX_FOLDS = TIER_LIMITS["quick"]["max_folds"]
QUICK_MAX_SEEDS = TIER_LIMITS["quick"]["max_seeds"]
_QUICK_TIER_TRIAL_LIMIT = QUICK_MAX_CANDIDATES * QUICK_MAX_FOLDS * QUICK_MAX_SEEDS

_NAN = float("nan")
_METRIC_COLUMNS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "stratified_dummy_macro_f1_same_rows",
    "delta_macro_f1_vs_dummy",
    "class0_pred_rate",
    "class1_pred_rate",
    "ticker_max_share",
)
# Section 14.1 metrics summarized across seeds (Codex #5F-6 Q4).
_SEED_SUMMARY_METRICS: tuple[str, ...] = (
    "macro_f1",
    "delta_macro_f1_vs_dummy",
    "balanced_accuracy",
)


@dataclass(frozen=True)
class _Candidate:
    """One resolved search candidate: a family + model config + its config hash."""

    family: str
    candidate_id: str
    model_config: Mapping[str, Any]
    config_hash: str


def run_quick_search(
    config: Mapping[str, Any], out: Path, *, budget_tier: str = "quick",
) -> pd.DataFrame:
    """Run a (candidate x fold x seed) search loop for any budget tier.

    Returns the assembled trial ledger. Also writes per-ticker summary CSV.
    """
    if budget_tier not in TIER_LIMITS:
        raise ValueError(f"unknown budget_tier {budget_tier!r}; use {sorted(TIER_LIMITS)}")

    candidate = _frozen_candidate(config)
    X, y, ticker_ids, timestamps = resolve_train_inner_arrays(config)
    index_sha = train_inner_index_sha256(timestamps, ticker_ids, y)
    fold_plan = build_fold_plan(config)
    _assert_label_horizon_provenance(candidate, fold_plan)
    folds, fold_results_df, assignment_sha = _build_folds(timestamps, ticker_ids, fold_plan)

    candidates, seeds, hpo_method, scientific_cap = _resolve_candidates_and_seeds(config)
    families = sorted({cand.family for cand in candidates})
    n_candidates, n_folds, n_seeds = len(candidates), len(folds), len(seeds)
    tier_complete = _check_budget(n_candidates, n_folds, n_seeds, scientific_cap, budget_tier)

    if not (out / "08x_search_space.json").exists():
        write_schema_smoke_artifacts(out)

    _reconcile_fold_results(out, fold_results_df)
    search_space_sha = _write_search_space(
        out,
        families=families,
        candidates=candidates,
        seeds=seeds,
        hpo_method=hpo_method,
        scientific_cap=scientific_cap,
        n_folds=n_folds,
        n_seeds=n_seeds,
        quick_complete=tier_complete,
        budget_tier=budget_tier,
    )

    _dependency_preflight(families)
    fold_usable = {fold_id: _fold_usable(y, tr, va) for fold_id, tr, va in folds}
    expected_tickers = sorted(set(tk.item() if hasattr(tk, "item") else tk for tk in np.unique(ticker_ids)))
    n_trials_per_candidate = n_folds * n_seeds

    rows: list[dict[str, Any]] = []
    raw_per_ticker: list[dict[str, Any]] = []
    for cand in candidates:
        for fold_id, train_idx, val_idx in folds:
            for seed in seeds:
                trial_id = f"{cand.candidate_id}__{fold_id}__seed{int(seed)}"
                if not fold_usable[fold_id]:
                    rows.append(
                        _skipped_trial_row(
                            trial_id=trial_id,
                            candidate_family=cand.family,
                            candidate_id=cand.candidate_id,
                            config_hash=cand.config_hash,
                            fold_id=fold_id,
                            seed=int(seed),
                            train_n=int(train_idx.size),
                            val_n=int(val_idx.size),
                            reason="single-class train/val fold; skipped before fit",
                            budget_tier=budget_tier,
                        )
                    )
                    continue
                row, pt_rows = run_single_trial(
                    X,
                    y,
                    ticker_ids,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    trial_id=trial_id,
                    candidate_family=cand.family,
                    candidate_id=cand.candidate_id,
                    config_hash=cand.config_hash,
                    fold_id=fold_id,
                    seed=int(seed),
                    budget_tier=budget_tier,
                    model_config=cand.model_config,
                    collect_per_ticker=True,
                )
                _apply_class_collapse_guard(row)
                if row["fit_status"] == "completed" and pt_rows:
                    for pt in pt_rows:
                        pt["candidate_id"] = cand.candidate_id
                        pt["candidate_family"] = cand.family
                    raw_per_ticker.extend(pt_rows)
                rows.append(row)

    ledger_df = pd.DataFrame(rows, columns=sorted(REQUIRED_TRIAL_LEDGER_COLUMNS))
    validate_trial_ledger_frame(ledger_df)
    ledger_df.to_csv(out / "08x_trial_ledger.csv", index=False)

    _build_seed_summary(ledger_df).to_csv(out / "08x_seed_summary.csv", index=False)
    _build_failure_ledger(ledger_df).to_csv(out / "08x_failure_ledger.csv", index=False)

    per_ticker_df = _build_per_ticker_summary(
        raw_per_ticker, candidates, expected_tickers, n_trials_per_candidate,
    )
    per_ticker_df.to_csv(out / "08x_per_ticker.csv", index=False)

    _write_quick_run_manifest(
        out,
        fold_plan=fold_plan,
        candidate=candidate,
        ledger_df=ledger_df,
        requested=n_candidates * n_folds * n_seeds,
        search_space_sha=search_space_sha,
        fold_results_df=fold_results_df,
        assignment_sha=assignment_sha,
        index_sha=index_sha,
        config=config,
        quick_complete=tier_complete,
        budget_tier=budget_tier,
    )
    return ledger_df


def _build_folds(
    timestamps: np.ndarray, ticker_ids: np.ndarray, fold_plan: Sequence[FoldSpec]
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], pd.DataFrame, str]:
    """Build folds ONCE; return the index list, the fold-results frame, and the
    fold-assignment sha (so the trials are provably scored on the recorded folds)."""
    folds: list[tuple[str, np.ndarray, np.ndarray]] = []
    result_rows: list[dict[str, Any]] = []
    for spec in fold_plan:
        for split_index, (train_idx, val_idx) in enumerate(
            _build_for_scheme(timestamps, ticker_ids, spec)
        ):
            fold_id = f"{spec.scheme}__{split_index}"
            train_idx = np.asarray(train_idx)
            val_idx = np.asarray(val_idx)
            folds.append((fold_id, train_idx, val_idx))
            embargo = (
                spec.embargo_size if spec.scheme == "embargoed_train_inner_folds" else 0
            )
            result_rows.append(
                {
                    "fold_id": fold_id,
                    "fold_scheme": spec.scheme,
                    "split_index": int(split_index),
                    "train_inner_fit_n": int(train_idx.size),
                    "train_inner_validation_n": int(val_idx.size),
                    "purge_gap_k": int(spec.label_horizon_k),
                    "embargo_gap_k": int(embargo),
                }
            )
    fold_results_df = pd.DataFrame(result_rows, columns=list(FOLD_RESULTS_COLUMNS))
    validate_08x_fold_results_frame(fold_results_df, require_non_empty=True)
    return folds, fold_results_df, fold_assignment_sha256(folds)


def _reconcile_fold_results(out: Path, recomputed: pd.DataFrame) -> None:
    """Overwrite ``08x_fold_results.csv``; if a NON-empty one exists, full-frame
    compare first and fail loud on any drift (Codex #5F-6 P1-1)."""
    path = out / "08x_fold_results.csv"
    if path.exists():
        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        if not existing.empty:
            _assert_fold_results_match(existing, recomputed)
    recomputed.to_csv(path, index=False)


def _assert_fold_results_match(existing: pd.DataFrame, recomputed: pd.DataFrame) -> None:
    cols = list(FOLD_RESULTS_COLUMNS)
    left = existing.reindex(columns=cols).sort_values("fold_id").reset_index(drop=True)
    right = recomputed.reindex(columns=cols).sort_values("fold_id").reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError(
            "08x_fold_results.csv on disk has a different fold count than the "
            f"recomputed quick-search folds ({len(left)} != {len(right)}); folds "
            "changed under the search -- rebuild folds first."
        )
    for col in cols:
        if col in ("fold_id", "fold_scheme"):
            same = (left[col].astype(str).to_numpy() == right[col].astype(str).to_numpy()).all()
        else:
            same = (
                left[col].astype("int64").to_numpy() == right[col].astype("int64").to_numpy()
            ).all()
        if not bool(same):
            raise ValueError(
                f"08x_fold_results.csv column {col!r} differs from the recomputed "
                "quick-search folds; folds changed under the search -- rebuild first."
            )


def _resolve_candidates_and_seeds(
    config: Mapping[str, Any],
) -> tuple[list[_Candidate], list[int], str, int]:
    """Resolve the candidate list, seeds, HPO method, and scientific cap.

    Candidates come from ``config['search_space']['candidates']`` (pre-registered)
    or default to ONE per declared family (model_config = class defaults). Each
    candidate gets a deterministic ``config_hash`` (Codex #5F-6 Q3).
    """
    ss = config.get("search_space", {})
    if not isinstance(ss, Mapping):
        raise ValueError("config['search_space'] must be a mapping")

    hpo_method = ss.get("hpo_method", "random_search")
    if hpo_method not in HPO_METHODS:
        raise ValueError(
            f"search_space.hpo_method must be one of {sorted(HPO_METHODS)}; got {hpo_method!r}"
        )

    scientific_cap = ss.get(
        "scientific_budget_cap_total_trials", TOTAL_TRIAL_BUDGET_CAP_ACROSS_ALL_FAMILIES
    )
    if isinstance(scientific_cap, bool) or not isinstance(scientific_cap, int) or scientific_cap <= 0:
        raise ValueError(
            "search_space.scientific_budget_cap_total_trials must be a positive int"
        )

    seeds_raw = ss.get("seed_list") or [0]
    if not isinstance(seeds_raw, (list, tuple)) or not seeds_raw:
        raise ValueError("search_space.seed_list must be a non-empty list of ints")
    seeds: list[int] = []
    for seed in seeds_raw:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"search_space.seed_list entries must be ints; got {seed!r}")
        seeds.append(int(seed))
    if len(set(seeds)) != len(seeds):
        raise ValueError("search_space.seed_list must not contain duplicate seeds")

    explicit = ss.get("candidates")
    if explicit:
        if not isinstance(explicit, (list, tuple)):
            raise ValueError("search_space.candidates must be a list of mappings")
        candidates = [
            _make_candidate(
                str(item["family"]),
                str(item["candidate_id"]),
                dict(item.get("model_config", {})),
            )
            for item in explicit
        ]
    else:
        families = ss.get("architecture_families") or list(
            SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES
        )
        if not isinstance(families, (list, tuple)) or not families:
            raise ValueError("search_space.architecture_families must be a non-empty list")
        candidates = [_make_candidate(str(f), f"{f}__default", {}) for f in families]

    for cand in candidates:
        if cand.family not in SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES:
            raise ValueError(
                f"candidate family {cand.family!r} is not 08X search-eligible "
                f"(allowed: {sorted(SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES)})"
            )
        if cand.family not in SEQUENCE_CLASSIFIER_REGISTRY:
            raise ValueError(f"candidate family {cand.family!r} is not registered")
    ids = [cand.candidate_id for cand in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError(f"candidate_id values must be unique (trial-id collisions); got {ids}")
    return candidates, seeds, hpo_method, scientific_cap


def _make_candidate(family: str, candidate_id: str, model_config: dict[str, Any]) -> _Candidate:
    canonical = json.dumps(
        {"family": family, "model_config": model_config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _Candidate(
        family=family,
        candidate_id=candidate_id,
        model_config=model_config,
        config_hash=sha256_bytes(canonical.encode("utf-8")),
    )


def _check_budget(
    n_candidates: int, n_folds: int, n_seeds: int, scientific_cap: int,
    tier: str = "quick",
) -> bool:
    """Enforce tier envelope upper bounds + scientific cap; return whether the
    run meets the full tier lower-bound envelope."""
    limits = TIER_LIMITS[tier]
    max_c, max_f, max_s = limits["max_candidates"], limits["max_folds"], limits["max_seeds"]
    min_c = limits["min_candidates"]
    if n_folds > max_f:
        raise ValueError(f"{tier} tier allows <= {max_f} folds; got {n_folds}")
    if n_seeds > max_s:
        raise ValueError(f"{tier} tier allows <= {max_s} seeds; got {n_seeds}")
    if n_candidates > max_c:
        raise ValueError(f"{tier} tier allows <= {max_c} candidates; got {n_candidates}")
    grid = n_candidates * n_folds * n_seeds
    tier_limit = max_c * max_f * max_s
    cap = min(tier_limit, scientific_cap)
    if grid > cap:
        raise ValueError(
            f"{tier} grid {grid} exceeds cap {cap} "
            f"(min of tier limit {tier_limit} and scientific cap {scientific_cap})"
        )
    return min_c <= n_candidates <= max_c and 1 <= n_folds <= max_f and 1 <= n_seeds <= max_s


def _dependency_preflight(families: list[str]) -> None:
    """Fail loud BEFORE trial 0 if a declared family's dependency is missing.

    A missing optional dependency is an environment failure, NOT a per-trial
    ``training_divergence`` (Codex #5F-6 P1-4)."""
    try:
        import sklearn.metrics  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "RUN_08X_QUICK_SEARCH requires scikit-learn for trial metrics; "
            "install scikit-learn>=1.4."
        ) from exc
    if "last_step_lightgbm_control" in families:
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "RUN_08X_QUICK_SEARCH declares last_step_lightgbm_control but "
                "lightgbm is not importable; install lightgbm>=4.0 (env pins 4.6.0). "
                "A missing dependency is an environment failure, not a per-trial "
                "training_divergence."
            ) from exc


def _fold_usable(y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> bool:
    """A fold is usable iff BOTH its train and val slices carry both classes.

    ``compute_trial_metrics`` requires both classes in validation, and a 1-class
    train slice cannot fit a 2-class model -- so an unusable fold is skipped before
    any fit (Codex #5F-6 P1-5)."""
    train_classes = {int(v) for v in np.unique(y[train_idx])}
    val_classes = {int(v) for v in np.unique(y[val_idx])}
    return train_classes == {0, 1} and val_classes == {0, 1}


def _apply_class_collapse_guard(row: dict[str, Any]) -> None:
    """Section 14.4: a completed trial predicting <5% of one class is a collapse;
    rewrite it to a failed/class_collapse row but KEEP its metrics (Codex Q5)."""
    if row["fit_status"] != "completed":
        return
    class0 = row["class0_pred_rate"]
    class1 = row["class1_pred_rate"]
    if not (np.isfinite(class0) and np.isfinite(class1)):
        return
    worst = min(float(class0), float(class1))
    if worst < CLASS_COLLAPSE_PRED_RATE_MIN:
        row["fit_status"] = "failed"
        row["failure_type"] = "class_collapse"
        row["failure_message"] = (
            f"class collapse: min pred rate {worst:.4f} < {CLASS_COLLAPSE_PRED_RATE_MIN}"
        )


def _build_seed_summary(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Per (candidate_id, metric) over COMPLETED rows: per-seed mean across that
    seed's completed folds, then mean / std / 95% LCB across seeds (Codex Q4)."""
    completed = ledger_df[ledger_df["fit_status"].astype(str) == "completed"]
    rows: list[dict[str, Any]] = []
    if not completed.empty:
        for candidate_id, group in completed.groupby("candidate_id"):
            for metric in _SEED_SUMMARY_METRICS:
                per_seed = group.groupby("seed")[metric].mean().to_numpy(dtype=float)
                n_seeds = per_seed.size
                seed_mean = float(np.mean(per_seed))
                if n_seeds >= 2:
                    seed_std = float(np.std(per_seed, ddof=1))
                    seed_lcb_95 = seed_mean - 1.96 * seed_std / math.sqrt(n_seeds)
                else:
                    # n=1: std undefined; LCB collapses to the mean (low evidence).
                    seed_std = _NAN
                    seed_lcb_95 = seed_mean
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "metric": metric,
                        "seed_mean": seed_mean,
                        "seed_std": seed_std,
                        "seed_lcb_95": seed_lcb_95,
                    }
                )
    return pd.DataFrame(rows, columns=list(SEED_SUMMARY_COLUMNS))


def _build_failure_ledger(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Project the failed trial rows to the section 8.4 failure-ledger columns."""
    failed = ledger_df[ledger_df["fit_status"].astype(str) == "failed"]
    return failed.reindex(columns=list(FAILURE_LEDGER_COLUMNS)).reset_index(drop=True)


def _write_search_space(
    out: Path,
    *,
    families: list[str],
    candidates: list[_Candidate],
    seeds: list[int],
    hpo_method: str,
    scientific_cap: int,
    n_folds: int,
    n_seeds: int,
    quick_complete: bool,
    budget_tier: str = "quick",
) -> str:
    """Materialize + validate + persist ``08x_search_space.json`` BEFORE any trial;
    return its sha256."""
    per_family: dict[str, int] = {}
    for cand in candidates:
        per_family[cand.family] = per_family.get(cand.family, 0) + n_folds * n_seeds
    payload: dict[str, Any] = {
        "search_space_version": QUICK_SEARCH_VERSION,
        "stage": "08X",
        "scope": "exploratory",
        "architecture_families": families,
        "hpo_method": hpo_method,
        "eligibility_thresholds": {
            "min_train_inner_lcb_delta_macro_f1": (
                CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA
            ),
        },
        "scientific_budget_cap_total_trials": int(scientific_cap),
        "per_family_trial_budget": per_family,
        "low_compute_mode": False,
        "low_compute_submode": "",
        "seed_list": list(seeds),
        "deferred_07g_gaps": {},
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "budget_tier": budget_tier,
        "quick_evidence_complete": bool(quick_complete),
        "candidates": [
            {
                "family": cand.family,
                "candidate_id": cand.candidate_id,
                "model_config": dict(cand.model_config),
                "config_hash": cand.config_hash,
            }
            for cand in candidates
        ],
    }
    canonical = json.dumps(
        {k: v for k, v in payload.items() if k != "search_space_sha256"},
        sort_keys=True,
        separators=(",", ":"),
    )
    sha = sha256_bytes(canonical.encode("utf-8"))
    payload["search_space_sha256"] = sha
    validate_08x_search_space(payload)
    write_json(out / "08x_search_space.json", payload)
    return sha


def _data_source_sha256(config: Mapping[str, Any]) -> str | None:
    """sha256 over the sorted raw ``txt_manifest`` (ticker, path, file-sha) entries,
    or None when the windowed index is injected (no raw data path). Closes the
    'raw source provenance' half of the impl-review P1 for real-data runs."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    manifest = data.get("txt_manifest")
    if not isinstance(manifest, Mapping) or not manifest:
        return None
    entries = [
        f"{ticker}|{manifest[ticker]}|{sha256_file_or_unavailable(Path(manifest[ticker]))}"
        for ticker in sorted(manifest)
    ]
    return sha256_bytes("\n".join(entries).encode("utf-8"))


def _write_quick_run_manifest(
    out: Path,
    *,
    fold_plan: Sequence[FoldSpec],
    candidate: Mapping[str, Any],
    ledger_df: pd.DataFrame,
    requested: int,
    search_space_sha: str,
    fold_results_df: pd.DataFrame,
    assignment_sha: str,
    index_sha: str,
    config: Mapping[str, Any],
    quick_complete: bool,
    budget_tier: str = "quick",
) -> None:
    """Write the run manifest with provenance."""
    status = ledger_df["fit_status"].astype(str)
    modes = [spec.scheme for spec in fold_plan]
    label_horizon_k = fold_plan[0].label_horizon_k if fold_plan else 0
    embargo = max(
        (
            spec.embargo_size
            for spec in fold_plan
            if spec.scheme == "embargoed_train_inner_folds"
        ),
        default=0,
    )
    data_cfg = config.get("data")
    txt_present = bool(isinstance(data_cfg, Mapping) and data_cfg.get("txt_manifest"))
    fold_results_sha = sha256_bytes(fold_results_df.to_csv(index=False).encode("utf-8"))
    provenance_id = _candidate_provenance_id(candidate)
    payload = {
        "notebook08_version": QUICK_SEARCH_VERSION,
        "stage": "08X",
        "scope": "exploratory",
        "source_stage0_candidate": provenance_id,
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "train_inner_fold_policy": "+".join(modes),
        "purge_policy": f"horizon_bar_purge_k={label_horizon_k}",
        "embargo_policy": "none" if embargo == 0 else f"symmetric_embargo_k={embargo}",
        "search_budget_tier": budget_tier,
        "trial_count_requested": int(requested),
        "trial_count_completed": int((status == "completed").sum()),
        "trial_count_failed": int((status == "failed").sum()),
        "trial_count_skipped": int((status == "skipped").sum()),
        "provenance": {
            "search_space_sha256": search_space_sha,
            "fold_results_sha256": fold_results_sha,
            "fold_assignment_sha256": assignment_sha,
            # Binds the evidence to the masked train-inner row identity + raw source
            # so an unchanged positional fold layout over changed data is detected.
            "train_inner_index_sha256": index_sha,
            "data_source_sha256": _data_source_sha256(config),
            "frozen_candidate_provenance_id": provenance_id,
            "data_txt_manifest_present": txt_present,
            "quick_evidence_complete": bool(quick_complete),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_run_manifest(
        out / "08x_run_manifest.json",
        payload,
        validator=validate_08x_run_manifest,
        required_fields=sorted(REQUIRED_08X_RUN_MANIFEST_FIELDS),
        stage="08X",
        scope="exploratory",
        false_fields=("official_validation_used", "holdout_test_authorized"),
    )


def _assert_row_columns(row: dict[str, Any]) -> None:
    if set(row) != REQUIRED_TRIAL_LEDGER_COLUMNS:
        extra = sorted(set(row) - REQUIRED_TRIAL_LEDGER_COLUMNS)
        missing = sorted(REQUIRED_TRIAL_LEDGER_COLUMNS - set(row))
        raise RuntimeError(f"quick-search row column drift; extra={extra} missing={missing}")


def _skipped_trial_row(
    *,
    trial_id: str,
    candidate_family: str,
    candidate_id: str,
    config_hash: str,
    fold_id: str,
    seed: int,
    train_n: int,
    val_n: int,
    reason: str,
    budget_tier: str = "quick",
) -> dict[str, Any]:
    """A schema-valid ``fit_status='skipped'`` row for an unusable fold (no fit)."""
    row: dict[str, Any] = {
        "trial_id": str(trial_id),
        "candidate_family": str(candidate_family),
        "candidate_id": str(candidate_id),
        "config_hash": str(config_hash),
        "fold_id": str(fold_id),
        "seed": int(seed),
        "budget_tier": str(budget_tier),
        "max_epochs": _NAN,
        "actual_epochs": _NAN,
        "early_stop_reason": "",
        "fit_status": "skipped",
        "failure_type": "",
        "failure_message": str(reason),
        "train_inner_fit_n": int(train_n),
        "train_inner_validation_n": int(val_n),
        "actual_wall_clock_seconds": _NAN,
        "peak_memory_mb": _NAN,
        "gpu_seconds_or_null": None,
        "compute_tier": "full_compute",
        "scope": "exploratory",
        "official_validation_used": False,
        "holdout_test_authorized": False,
    }
    for col in _METRIC_COLUMNS:
        row[col] = _NAN
    _assert_row_columns(row)
    return row


def _build_per_ticker_summary(
    raw_rows: list[dict[str, Any]],
    candidates: list[_Candidate],
    expected_tickers: list[Any],
    n_trials_per_candidate: int,
) -> pd.DataFrame:
    """Aggregate raw per-ticker trial rows into the 08x_per_ticker.csv schema.

    Produces one row per (candidate, ticker) in the complete grid. Tickers with
    no contributing trials get NaN metrics and coverage_status='insufficient'.
    """
    result: list[dict[str, Any]] = []
    for cand in candidates:
        cand_rows = [r for r in raw_rows if r.get("candidate_id") == cand.candidate_id]
        for ticker in expected_tickers:
            ticker_rows = [r for r in cand_rows if r["ticker"] == ticker]
            contributing = [r for r in ticker_rows if r.get("both_classes_present", False)]
            n_contrib = len(contributing)
            coverage = n_contrib / n_trials_per_candidate if n_trials_per_candidate > 0 else 0.0
            n_rows_total = sum(r["n_rows"] for r in ticker_rows) if ticker_rows else 0
            if contributing:
                macro_f1_mean = float(np.mean([r["macro_f1"] for r in contributing]))
                delta_mean = float(np.mean([r["delta_macro_f1_vs_dummy"] for r in contributing]))
                positive = bool(delta_mean > 0)
            else:
                macro_f1_mean = _NAN
                delta_mean = _NAN
                positive = False
            result.append({
                "candidate_id": cand.candidate_id,
                "candidate_family": cand.family,
                "ticker": ticker,
                "n_rows_total": int(n_rows_total),
                "n_trials_expected": int(n_trials_per_candidate),
                "n_trials_contributing": int(n_contrib),
                "coverage_rate": float(coverage),
                "macro_f1_mean": macro_f1_mean,
                "delta_macro_f1_vs_dummy_mean": delta_mean,
                "positive_delta": positive,
                "coverage_status": "ok" if n_contrib > 0 else "insufficient",
            })
    return pd.DataFrame(result, columns=list(PER_TICKER_COLUMNS))


def check_escalation_gate(
    out: Path,
    current_tier: str,
) -> dict[str, Any]:
    """Check §11.1 tier escalation gate. Returns a dict with 'passed' bool + details.

    quick->medium: best deep candidate lcb_delta_macro_f1 >= 0.003 AND positive
    on >= 4 tickers (from per_ticker.csv).
    medium->aggressive: additionally, seed_std_macro_f1 <= 0.01 on the leader.
    """
    from intraday_research.contracts.deep_sequence_exploration import (
        TIER_ESCALATION_MEDIUM_TO_AGGRESSIVE_SEED_STD_MAX,
        TIER_ESCALATION_QUICK_TO_MEDIUM_LCB_DELTA_MIN,
        TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN,
    )

    seed_summary = pd.read_csv(out / "08x_seed_summary.csv")
    per_ticker = pd.read_csv(out / "08x_per_ticker.csv")

    # Find best deep candidate by lcb_delta_macro_f1 (exclude lightgbm control).
    delta_rows = seed_summary[
        (seed_summary["metric"] == "delta_macro_f1_vs_dummy")
    ].copy()
    if delta_rows.empty:
        return {"passed": False, "reason": "no delta_macro_f1_vs_dummy in seed_summary"}

    # Merge candidate_family from per_ticker to identify control.
    fam_map = per_ticker[["candidate_id", "candidate_family"]].drop_duplicates()
    delta_rows = delta_rows.merge(fam_map, on="candidate_id", how="left")
    deep_rows = delta_rows[delta_rows["candidate_family"] != "last_step_lightgbm_control"]
    if deep_rows.empty:
        return {"passed": False, "reason": "no deep candidates in seed_summary"}

    best_idx = deep_rows["seed_lcb_95"].idxmax()
    best_id = deep_rows.loc[best_idx, "candidate_id"]
    best_lcb = float(deep_rows.loc[best_idx, "seed_lcb_95"])

    # Count positive tickers for best candidate.
    best_pt = per_ticker[per_ticker["candidate_id"] == best_id]
    n_positive = int(best_pt["positive_delta"].sum())

    lcb_ok = best_lcb >= TIER_ESCALATION_QUICK_TO_MEDIUM_LCB_DELTA_MIN
    ticker_ok = n_positive >= TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN

    result: dict[str, Any] = {
        "best_candidate_id": best_id,
        "best_lcb_delta": best_lcb,
        "n_positive_tickers": n_positive,
        "lcb_gate": lcb_ok,
        "ticker_gate": ticker_ok,
    }

    if current_tier == "quick":
        result["passed"] = lcb_ok and ticker_ok
        if not result["passed"]:
            result["reason"] = (
                f"lcb_delta={best_lcb:.4f} (need>={TIER_ESCALATION_QUICK_TO_MEDIUM_LCB_DELTA_MIN}), "
                f"positive_tickers={n_positive} (need>={TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN})"
            )
        return result

    if current_tier == "medium":
        # Additional check: seed_std_macro_f1 <= 0.01.
        std_row = seed_summary[
            (seed_summary["candidate_id"] == best_id)
            & (seed_summary["metric"] == "macro_f1")
        ]
        seed_std = float(std_row["seed_std"].iloc[0]) if not std_row.empty else _NAN
        std_ok = np.isfinite(seed_std) and seed_std <= TIER_ESCALATION_MEDIUM_TO_AGGRESSIVE_SEED_STD_MAX
        result["seed_std_macro_f1"] = seed_std
        result["std_gate"] = std_ok
        result["passed"] = lcb_ok and ticker_ok and std_ok
        if not result["passed"]:
            result["reason"] = (
                f"lcb_delta={best_lcb:.4f}, positive_tickers={n_positive}, "
                f"seed_std={seed_std:.4f} (need<={TIER_ESCALATION_MEDIUM_TO_AGGRESSIVE_SEED_STD_MAX})"
            )
        return result

    result["passed"] = False
    result["reason"] = f"no escalation from tier {current_tier!r}"
    return result
