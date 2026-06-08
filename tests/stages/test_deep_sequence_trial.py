"""Tests for the #5F-5 single-trial runner (synthetic data, tiny real DLinear fit)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    REQUIRED_TRIAL_LEDGER_COLUMNS,
    validate_trial_ledger_frame,
)
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.registry import build_classifier
from intraday_research.stages.deep_sequence_trial import run_single_trial


def _synth(n=40, t=8, f=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, t, f)).astype(np.float64)
    y = np.tile([0, 1], n // 2).astype(np.int64)[:n]
    ticker_ids = np.array([("AAA", "BBB")[i % 2] for i in range(n)], dtype=object)
    return X, y, ticker_ids


_TRAIN = np.arange(0, 30, dtype=np.int64)
_VAL = np.arange(30, 40, dtype=np.int64)


def _trial(X, y, tk, *, train_idx=_TRAIN, val_idx=_VAL, seed=101, model_config=None):
    return run_single_trial(
        X, y, tk,
        train_idx=train_idx, val_idx=val_idx,
        trial_id="t0", candidate_family="dlinear_only", candidate_id="dlinear_c0",
        config_hash="hash0", fold_id="rolling_origin_folds__0", seed=seed,
        budget_tier="quick",
        model_config=model_config if model_config is not None else {"max_epochs": 1, "batch_size": 8},
    )


def test_run_single_trial_completed_row():
    X, y, tk = _synth()
    row = _trial(X, y, tk)
    assert set(row) == set(REQUIRED_TRIAL_LEDGER_COLUMNS)
    assert row["fit_status"] == "completed"
    assert row["failure_type"] == ""
    assert row["scope"] == "exploratory"
    assert row["official_validation_used"] is False
    assert row["holdout_test_authorized"] is False
    assert row["gpu_seconds_or_null"] is None
    assert row["compute_tier"] == "full_compute"
    assert row["train_inner_fit_n"] == 30
    assert row["train_inner_validation_n"] == 10
    assert row["max_epochs"] == 1
    for col in ("macro_f1", "balanced_accuracy", "accuracy", "delta_macro_f1_vs_dummy"):
        assert math.isfinite(row[col])
    assert row["actual_wall_clock_seconds"] >= 0.0


def test_run_single_trial_row_passes_contract():
    X, y, tk = _synth()
    df = pd.DataFrame([_trial(X, y, tk)])
    validate_trial_ledger_frame(df)


def test_run_single_trial_deterministic():
    X, y, tk = _synth()
    r1 = _trial(X, y, tk, seed=7)
    r2 = _trial(X, y, tk, seed=7)
    assert r1["macro_f1"] == r2["macro_f1"]
    assert r1["accuracy"] == r2["accuracy"]


def test_run_single_trial_rng_isolated():
    X, y, tk = _synth()
    np.random.seed(12345)
    before = float(np.random.random())
    np.random.seed(12345)
    _trial(X, y, tk)
    after = float(np.random.random())
    assert before == after  # the trial restored the global np.random stream


def test_run_single_trial_failure_row_does_not_raise():
    X, y, tk = _synth()
    # moving_avg_kernel 999 is outside DLinear's frozen axis set -> __init__ ValueError
    row = _trial(X, y, tk, model_config={"max_epochs": 1, "batch_size": 8, "moving_avg_kernel": 999})
    assert row["fit_status"] == "failed"
    assert row["failure_type"] == "artifact_schema_failure"
    assert row["failure_message"]
    assert math.isnan(row["macro_f1"])
    # a failed row is still schema-valid
    validate_trial_ledger_frame(pd.DataFrame([row]))


def test_run_single_trial_invalid_indices_raise():
    X, y, tk = _synth()
    with pytest.raises(ValueError, match="disjoint"):
        _trial(X, y, tk, train_idx=np.arange(0, 30, dtype=np.int64), val_idx=np.arange(25, 40, dtype=np.int64))
    with pytest.raises(ValueError, match="non-empty"):
        _trial(X, y, tk, val_idx=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="out of bounds"):
        _trial(X, y, tk, val_idx=np.array([100], dtype=np.int64))


def test_registry_build_and_unknown():
    model = build_classifier("dlinear_only", random_state=0)
    assert isinstance(model, DLinearClassifier)
    with pytest.raises(ValueError, match="unknown candidate_family"):
        build_classifier("nope", random_state=0)


def test_run_single_trial_nonfinite_proba_is_training_divergence(monkeypatch):
    # Codex impl review P1: a diverged model (NaN proba) must NOT be recorded as a
    # completed row with finite metrics.
    from intraday_research.stages import deep_sequence_trial as trial_mod

    class _NanModel:
        max_epochs = 1
        actual_epochs_ = 1
        early_stop_reason_ = ""

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.full((X.shape[0], 2), np.nan)

    monkeypatch.setattr(trial_mod, "build_classifier", lambda *a, **k: _NanModel())
    X, y, tk = _synth()
    row = _trial(X, y, tk)
    assert row["fit_status"] == "failed"
    assert row["failure_type"] == "training_divergence"
    assert math.isnan(row["macro_f1"])


def test_run_single_trial_duplicate_indices_raise():
    # Codex impl review P2: duplicate indices within a split fail loud.
    X, y, tk = _synth()
    with pytest.raises(ValueError, match="duplicate"):
        _trial(X, y, tk, train_idx=np.array([0, 0, 1, 2], dtype=np.int64))


def test_run_single_trial_bool_max_epochs_not_aliased():
    # Codex impl review P2: a bool max_epochs must not alias to 1 on the fallback.
    X, y, tk = _synth()
    row = _trial(X, y, tk, model_config={"max_epochs": True, "batch_size": 8})
    assert row["fit_status"] == "failed"  # DLinear rejects bool max_epochs
    assert math.isnan(row["max_epochs"])
