"""08X single-trial runner (#5F-5): fit one model on one fold/seed -> one row.

Spec: docs/superpowers/specs/2026-06-08-n08-08x-single-trial-runner-design.md
Codex review: .humanize/skill/2026-06-08_01-45-54-1678-66f57081/

`run_single_trial` fits a deep-sequence classifier on the train-inner-fit rows of
a fold and scores the train-inner-validation rows, returning ONE 29-column
§8.3 trial-ledger row. It fits ONLY on `X[train_idx]` and scores ONLY
`X[val_idx]` -- no official-validation or holdout data is touched. Model-fit
failures are recorded as a `failed` row (never raised); invalid fold indices
fail loud (caller contract). Determinism is isolated: the global Python / numpy /
torch RNG state is snapshotted and restored around each trial.
"""

from __future__ import annotations

import random
import time
import tracemalloc
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from intraday_research.contracts.deep_sequence_exploration import (
    REQUIRED_TRIAL_LEDGER_COLUMNS,
)
from intraday_research.models.deep_sequence.metrics import compute_trial_metrics
from intraday_research.models.deep_sequence.registry import build_classifier


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


def _snapshot_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _classify_failure(exc: BaseException) -> str:
    """Map a fit/predict exception to a §8.4 failure_type (Codex #5F-5 Q4)."""
    if isinstance(exc, MemoryError):
        return "memory_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, NotImplementedError):
        return "not_implemented"
    if isinstance(exc, (ValueError, TypeError)):
        return "artifact_schema_failure"
    return "training_divergence"


def _validate_indices(n: int, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    for name, idx in (("train_idx", train_idx), ("val_idx", val_idx)):
        if not isinstance(idx, np.ndarray) or idx.ndim != 1:
            raise ValueError(f"{name} must be a 1-D ndarray")
        if idx.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} must be integer dtype; got {idx.dtype}")
        if int(idx.min()) < 0 or int(idx.max()) >= n:
            raise ValueError(f"{name} out of bounds for n={n}")
        # Codex impl review P2: reject duplicate indices within a split. (Train/val
        # ORDER is intentionally NOT enforced -- purged/embargoed interior folds
        # legitimately train on rows after their validation block.)
        if np.unique(idx).size != idx.size:
            raise ValueError(f"{name} must not contain duplicate indices")
    if np.intersect1d(train_idx, val_idx).size > 0:
        raise ValueError("train_idx and val_idx must be disjoint (no leakage)")


def run_single_trial(
    X: np.ndarray,
    y: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    trial_id: str,
    candidate_family: str,
    candidate_id: str,
    config_hash: str,
    fold_id: str,
    seed: int,
    budget_tier: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one model on one fold/seed; return one 29-column trial-ledger row.

    Fits on ``X[train_idx]`` only and scores ``X[val_idx]`` only. On model-fit
    failure the row carries ``fit_status='failed'`` + a mapped ``failure_type`` +
    NaN metrics (no exception is raised to the caller). Invalid indices
    (out of bounds / empty / overlapping) raise ``ValueError`` (caller contract).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    ticker_ids = np.asarray(ticker_ids)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (n, window, features); got shape {X.shape}")
    n = int(X.shape[0])
    if y.shape[0] != n or ticker_ids.shape[0] != n:
        raise ValueError("y and ticker_ids must align with X rows")
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    _validate_indices(n, train_idx, val_idx)

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
        "fit_status": "failed",
        "failure_type": "",
        "failure_message": "",
        "train_inner_fit_n": int(train_idx.size),
        "train_inner_validation_n": int(val_idx.size),
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

    # model_config must not carry random_state (the runner owns the seed).
    config = {k: v for k, v in dict(model_config).items() if k != "random_state"}

    rng_state = _snapshot_rng()
    tm_was_tracing = tracemalloc.is_tracing()
    if not tm_was_tracing:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        model = build_classifier(candidate_family, random_state=int(seed), **config)
        # Codex #5F-6 Q1: non-epoch models (LastStepLightGBMControl, the late-fusion
        # wrappers) carry no `max_epochs`; getattr-guard so they record an honest
        # completed row instead of raising AttributeError here, which the
        # except-branch would mis-map to a spurious `training_divergence` failure.
        model_max_epochs = getattr(model, "max_epochs", None)
        row["max_epochs"] = (
            int(model_max_epochs) if type(model_max_epochs) is int else _NAN
        )
        model.fit(X[train_idx], y[train_idx])
        proba = np.asarray(model.predict_proba(X[val_idx]))
        # Codex impl review P1: a diverged model can emit NaN probabilities;
        # argmax(NaN) still yields ints and compute_trial_metrics would then
        # return finite metrics for a broken fit. Reject non-finite / wrong-shape
        # output as a training divergence (RuntimeError -> training_divergence).
        if proba.shape != (int(val_idx.size), 2) or not np.isfinite(proba).all():
            raise RuntimeError(
                "predict_proba returned non-finite or wrong-shape output "
                f"(shape={proba.shape}); treating as training divergence"
            )
        y_pred = proba.argmax(axis=1)
        metrics = compute_trial_metrics(y[val_idx], y_pred, ticker_ids[val_idx])
        for col in _METRIC_COLUMNS:
            row[col] = float(metrics[col])
        row["fit_status"] = "completed"
        row["failure_type"] = ""
        # Codex #5F-6 Q1: actual_epochs_ / early_stop_reason_ are torch-base attrs;
        # non-epoch models lack them, so getattr-guard rather than attribute-access.
        actual_epochs = getattr(model, "actual_epochs_", None)
        row["actual_epochs"] = (
            int(actual_epochs) if type(actual_epochs) is int else _NAN
        )
        row["early_stop_reason"] = getattr(model, "early_stop_reason_", None) or ""
    except Exception as exc:  # noqa: BLE001 -- recorded as a failed trial row
        row["fit_status"] = "failed"
        row["failure_type"] = _classify_failure(exc)
        row["failure_message"] = str(exc)[:500]
        if not isinstance(row["max_epochs"], int):
            fallback = config.get("max_epochs")
            # type() not isinstance(): bool is an int subclass and must not alias.
            row["max_epochs"] = int(fallback) if type(fallback) is int else _NAN
    finally:
        row["actual_wall_clock_seconds"] = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        if not tm_was_tracing:
            tracemalloc.stop()
        row["peak_memory_mb"] = peak / 1e6
        _restore_rng(rng_state)

    if set(row) != REQUIRED_TRIAL_LEDGER_COLUMNS:
        extra = sorted(set(row) - REQUIRED_TRIAL_LEDGER_COLUMNS)
        missing = sorted(REQUIRED_TRIAL_LEDGER_COLUMNS - set(row))
        raise RuntimeError(
            f"trial row column drift; extra={extra} missing={missing}"
        )
    return row
