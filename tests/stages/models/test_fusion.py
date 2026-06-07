"""Tests for N08 #5D-5 fusion variants — slice 1 (late-average + logit-sum)
plus the shared ``_predict_logits`` base helper they rely on."""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.fusion import (
    DLinearLogitsPlusTCNLogitsFusion,
    LateAverageProbabilitiesFusion,
)
from intraday_research.models.deep_sequence.tcn import TCNClassifier

_IMPLEMENTED = [LateAverageProbabilitiesFusion, DLinearLogitsPlusTCNLogitsFusion]

# Small/fast sub-model configs so composing two torch models stays snappy.
_DL_FAST = {"max_epochs": 3, "batch_size": 16}
_TC_FAST = {"max_epochs": 3, "batch_size": 16, "num_blocks": 2, "channels": (16, 16)}


def _make_xy(n: int = 32, length: int = 20, c: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _mk(cls, *, random_state: int = 0, **overrides):
    kw = dict(
        dlinear_config=dict(_DL_FAST),
        tcn_config=dict(_TC_FAST),
        random_state=random_state,
    )
    kw.update(overrides)
    return cls(**kw)


# ---- 1. Protocol + proba contract -------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_is_sequence_classifier(cls):
    assert isinstance(cls(), SequenceClassifier)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_fit_returns_self_and_proba_contract(cls):
    X, y = _make_xy(seed=2)
    clf = _mk(cls)
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


# ---- 2. Determinism ----------------------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_predict_proba_empty_batch(cls):
    # n==0 must return (0, 2) like the single models (Codex P2: stable softmax
    # must not raise on a zero-size reduction).
    X, y = _make_xy(seed=2)
    clf = _mk(cls, random_state=0).fit(X, y)
    empty = np.zeros((0, 20, 2), dtype=np.float64)
    proba = clf.predict_proba(empty)
    assert proba.shape == (0, 2)
    assert proba.dtype == np.float64


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_determinism_same_seed_bit_exact(cls):
    X, y = _make_xy(seed=3)
    p1 = _mk(cls, random_state=7).fit(X, y).predict_proba(X)
    p2 = _mk(cls, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


# ---- 3. Fusion actually combines --------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_fusion_differs_from_each_component(cls):
    X, y = _make_xy(seed=4)
    fused = _mk(cls, random_state=0).fit(X, y).predict_proba(X)
    dlin = DLinearClassifier(random_state=0, **_DL_FAST).fit(X, y).predict_proba(X)
    tcn = TCNClassifier(random_state=0, **_TC_FAST).fit(X, y).predict_proba(X)
    assert not np.allclose(fused, dlin)
    assert not np.allclose(fused, tcn)


def test_late_average_differs_from_logit_sum():
    X, y = _make_xy(seed=5)
    avg = _mk(LateAverageProbabilitiesFusion, random_state=0).fit(X, y).predict_proba(X)
    logit = _mk(DLinearLogitsPlusTCNLogitsFusion, random_state=0).fit(X, y).predict_proba(X)
    assert not np.allclose(avg, logit)


# ---- 4. _predict_logits base helper -----------------------------------

def test_predict_logits_softmaxes_to_predict_proba():
    X, y = _make_xy(seed=6)
    clf = DLinearClassifier(random_state=0, **_DL_FAST).fit(X, y)
    logits = clf._predict_logits(X)
    assert logits.shape == (X.shape[0], 2)
    assert logits.dtype == np.float64
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm = e / e.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(sm, clf.predict_proba(X), atol=1e-10)


# ---- 5. Guards ---------------------------------------------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_random_state_none_at_fit(cls):
    X, y = _make_xy(seed=2)
    assert cls().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        cls().fit(X, y)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_random_state_bool_at_fit(cls):
    X, y = _make_xy(seed=2)
    with pytest.raises(ValueError, match="random_state"):
        _mk(cls, random_state=True).fit(X, y)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_nested_random_state_in_dlinear_config(cls):
    with pytest.raises(ValueError, match="random_state"):
        cls(dlinear_config={"random_state": 5})


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_nested_random_state_in_tcn_config(cls):
    with pytest.raises(ValueError, match="random_state"):
        cls(tcn_config={"random_state": 5})


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_bad_dlinear_config_type(cls):
    with pytest.raises(ValueError, match="dlinear_config"):
        cls(dlinear_config=[1, 2])


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_bad_tcn_config_type(cls):
    with pytest.raises(ValueError, match="tcn_config"):
        cls(tcn_config="nope")


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_predict_before_fit(cls):
    X, _ = _make_xy(seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        cls(random_state=0).predict_proba(X)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_reject_predict_shape_drift(cls):
    X, y = _make_xy(n=24, c=2, seed=2)
    clf = _mk(cls, random_state=0).fit(X, y)
    X_bad, _ = _make_xy(n=5, c=3, seed=2)
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


# ---- 6. Config forwarding (delegation to sub-models) -------------------

@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_config_forwarding_valid_axis(cls):
    X, y = _make_xy(seed=2)
    clf = cls(
        dlinear_config={**_DL_FAST, "moving_avg_kernel": 7},
        tcn_config={**_TC_FAST, "kernel_size": 5},
        random_state=0,
    ).fit(X, y)
    assert clf.predict_proba(X).shape == (X.shape[0], 2)


@pytest.mark.parametrize("cls", _IMPLEMENTED)
def test_config_forwarding_bad_axis_fails_loud(cls):
    X, y = _make_xy(seed=2)
    with pytest.raises(ValueError, match="moving_avg_kernel"):
        cls(dlinear_config={**_DL_FAST, "moving_avg_kernel": 4}, random_state=0).fit(X, y)
