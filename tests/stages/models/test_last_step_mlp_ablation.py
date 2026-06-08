"""Tests for the N08 #5D-7 last-step MLP sequence ablation (CPU PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.controls import LastStepMLPSequenceAblation


def _make_xy(n: int = 24, length: int = 20, c: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _fit(**kw):
    kw.setdefault("random_state", 0)
    return LastStepMLPSequenceAblation(**kw)


# ---- 1. Protocol + proba contract -------------------------------------

def test_is_sequence_classifier():
    assert isinstance(LastStepMLPSequenceAblation(), SequenceClassifier)


def test_fit_returns_self_and_proba_contract():
    X, y = _make_xy(n=24, seed=2)
    clf = _fit()
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (24, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


# ---- 2. Determinism + global state ------------------------------------

def test_determinism_same_seed_bit_exact():
    X, y = _make_xy(n=24, seed=3)
    p1 = _fit(random_state=7).fit(X, y).predict_proba(X)
    p2 = _fit(random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_fit_restores_global_deterministic_state():
    before_det = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    before_rng = torch.random.get_rng_state()
    X, y = _make_xy(n=20, seed=5)
    _fit(random_state=1).fit(X, y)
    assert torch.are_deterministic_algorithms_enabled() == before_det
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn
    assert torch.equal(torch.random.get_rng_state(), before_rng)


# ---- 3. The ablation: uses ONLY the last bar --------------------------

def test_uses_only_the_last_bar():
    X, y = _make_xy(n=24, c=2, seed=3)
    clf = _fit(random_state=0).fit(X, y)
    base = clf.predict_proba(X)
    # Perturbing ALL bars except the last must leave predictions BIT-IDENTICAL —
    # the hard invariant proving the model reads only X[:, -1, :].
    rng = np.random.default_rng(99)
    x_non_last = X.copy()
    x_non_last[:, :-1, :] = rng.standard_normal(X[:, :-1, :].shape)
    np.testing.assert_array_equal(clf.predict_proba(x_non_last), base)
    # Perturbing the last bar (large) DOES change predictions (robust check).
    x_last = X.copy()
    x_last[:, -1, :] = x_last[:, -1, :] + 100.0
    assert not np.array_equal(clf.predict_proba(x_last), base)


# ---- 4. Axis coverage --------------------------------------------------

@pytest.mark.parametrize("hidden_size", [8, 16, 32])
def test_axis_hidden_size(hidden_size):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(hidden_size=hidden_size).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.10])
def test_axis_dropout(dropout):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(dropout=dropout).fit(X, y).predict_proba(X).shape == (20, 2)


# ---- 5. Guards ---------------------------------------------------------

def test_reject_hidden_size_off_grid():
    with pytest.raises(ValueError, match="hidden_size"):
        LastStepMLPSequenceAblation(hidden_size=64)


def test_reject_hidden_size_bool_aliasing():
    with pytest.raises(ValueError, match="hidden_size"):
        LastStepMLPSequenceAblation(hidden_size=True)


def test_reject_dropout_off_grid():
    with pytest.raises(ValueError, match="dropout"):
        LastStepMLPSequenceAblation(dropout=0.5)


def test_reject_dropout_bool_aliasing_zero():
    with pytest.raises(ValueError, match="dropout"):
        LastStepMLPSequenceAblation(dropout=False)


def test_reject_random_state_none_at_fit():
    X, y = _make_xy(n=20, seed=2)
    assert LastStepMLPSequenceAblation().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        LastStepMLPSequenceAblation().fit(X, y)


def test_reject_random_state_bool_at_fit():
    X, y = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="random_state"):
        LastStepMLPSequenceAblation(random_state=True).fit(X, y)


def test_reject_x_not_3d():
    with pytest.raises(ValueError, match="3-D"):
        _fit().fit(np.zeros((10, 20), dtype=np.float64), np.zeros(10, dtype=np.int8))


def test_reject_y_single_class():
    X, _ = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="both classes"):
        _fit().fit(X, np.zeros(20, dtype=np.int8))


def test_reject_predict_before_fit():
    X, _ = _make_xy(n=10, seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        _fit().predict_proba(X)


def test_reject_predict_shape_drift():
    X, y = _make_xy(n=20, c=2, seed=2)
    clf = _fit().fit(X, y)
    X_bad, _ = _make_xy(n=5, c=3, seed=2)
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


# ---- 6. Early-stop bookkeeping ----------------------------------------

def test_early_stop_records_fields():
    X, y = _make_xy(n=200, seed=4)
    clf = _fit(max_epochs=10, early_stopping_patience=2).fit(X, y)
    assert 1 <= clf.actual_epochs_ <= 10
    assert clf.early_stop_reason_ in {"patience", "max_epochs", "no_internal_val"}
    assert clf.internal_val_n_ >= 1
