"""Tests for the N08 #5D-2 causal TCN classifier body (CPU PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.tcn import TCNClassifier


def _make_xy(n: int = 24, length: int = 20, c: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _make_sequence_signal(n: int = 200, length: int = 20, seed: int = 1):
    """Label = early-vs-late slope sign; identical last bar (0) for every window."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, length, 1), dtype=np.float64)
    y = np.zeros(n, dtype=np.int8)
    t = np.linspace(-1.0, 1.0, length)
    for i in range(n):
        label = i % 2
        series = (1.0 if label else -1.0) * t + rng.standard_normal(length) * 0.05
        X[i, :, 0] = series - series[-1]
        y[i] = label
    return X, y


def _fit(**kw):
    kw.setdefault("num_blocks", 2)
    kw.setdefault("channels", (16, 16))
    kw.setdefault("random_state", 0)
    return TCNClassifier(**kw)


# ---- 1. Protocol -------------------------------------------------------

def test_is_sequence_classifier():
    assert isinstance(TCNClassifier(), SequenceClassifier)


def test_fit_returns_self_and_proba_contract():
    X, y = _make_xy(n=24, seed=2)
    clf = _fit()
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (24, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


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


# ---- 3. CAUSALITY (the §4.1 red line) ---------------------------------

@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"gating": True},
        {"normalization": "layer_norm"},
        {"normalization": "weight_norm"},
        {"kernel_size": 5},
        {"num_blocks": 3, "channels": (32, 32, 32)},
    ],
)
def test_causality_no_future_leak(kw):
    # Cover every axis that touches the conv path (Codex P3): a future-aware
    # pad / wrong dilation in any config must fail this.
    X, y = _make_xy(n=12, seed=3)
    clf = _fit(**kw).fit(X, y)
    rng = np.random.default_rng(99)
    t = 10
    xa = rng.standard_normal((1, 20, 2)).astype(np.float64)
    xb = xa.copy()
    xb[:, t + 1:, :] = rng.standard_normal((1, 20 - t - 1, 2))  # perturb the FUTURE
    fa = clf._forward_features(xa)  # (1, c_last, L)
    fb = clf._forward_features(xb)
    # Causal: conv-stack feature columns at time <= t depend only on inputs
    # <= t (which are identical) -> bit-identical. A future-aware pad fails this.
    np.testing.assert_array_equal(fa[:, :, : t + 1], fb[:, :, : t + 1])
    assert not np.array_equal(fa[:, :, t + 1:], fb[:, :, t + 1:])  # future does differ


# ---- 4. Sequence-only signal ------------------------------------------

def test_learns_sequence_only_signal():
    X, y = _make_sequence_signal(n=200, seed=1)
    clf = _fit(kernel_size=3, random_state=0, max_epochs=60).fit(X, y)
    pred = clf.predict_proba(X).argmax(axis=1)
    assert (pred == y).mean() > 0.6


# ---- 5. Axis coverage --------------------------------------------------

@pytest.mark.parametrize("kernel", [2, 3, 5])
def test_axis_kernel(kernel):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(kernel_size=kernel).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.10, 0.20])
def test_axis_dropout(dropout):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(dropout=dropout).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("gating", [False, True])
def test_axis_gating(gating):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(gating=gating).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("norm", ["none", "weight_norm", "layer_norm"])
def test_axis_normalization(norm):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(normalization=norm).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("head", ["last_step", "attention_pooling_pre_frozen"])
def test_axis_head(head):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(head=head).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("nb,ch", [(2, (16, 16)), (3, (32, 32, 32)), (3, (64, 32, 16))])
def test_axis_blocks_channels(nb, ch):
    X, y = _make_xy(n=20, seed=2)
    clf = TCNClassifier(num_blocks=nb, channels=ch, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (20, 2)


# ---- 6. Guards ---------------------------------------------------------

def test_reject_causal_false():
    with pytest.raises(ValueError, match="causal"):
        TCNClassifier(causal=False)


def test_reject_causal_int_one_truthiness():
    # The scaffold's `if not causal` would ACCEPT 1; exact-type must reject it.
    with pytest.raises(ValueError, match="causal"):
        TCNClassifier(causal=1)


def test_reject_num_blocks_channels_mismatch():
    with pytest.raises(ValueError, match="num_blocks"):
        TCNClassifier(num_blocks=3, channels=(16, 16))


def test_reject_channels_not_frozen():
    with pytest.raises(ValueError, match="channels"):
        TCNClassifier(num_blocks=2, channels=(8, 8))


def test_reject_dilation_base_not_2():
    with pytest.raises(ValueError, match="dilation_base"):
        TCNClassifier(dilation_base=3)


def test_reject_residual_false():
    with pytest.raises(ValueError, match="residual"):
        TCNClassifier(residual=False)


def test_reject_bad_kernel():
    with pytest.raises(ValueError, match="kernel_size"):
        TCNClassifier(kernel_size=4)


def test_reject_gating_int_aliasing():
    with pytest.raises(ValueError, match="gating"):
        TCNClassifier(gating=1)


def test_reject_random_state_none_at_fit():
    X, y = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="random_state"):
        TCNClassifier(num_blocks=2, channels=(16, 16)).fit(X, y)


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


# ---- 7. Early-stop bookkeeping ----------------------------------------

def test_early_stop_records_fields():
    X, y = _make_xy(n=200, seed=4)
    clf = _fit(max_epochs=10, early_stopping_patience=2).fit(X, y)
    assert 1 <= clf.actual_epochs_ <= 10
    assert clf.early_stop_reason_ in {"patience", "max_epochs", "no_internal_val"}
    assert clf.internal_val_n_ >= 1


def test_no_internal_val_path_small_n():
    X = np.random.default_rng(4).standard_normal((3, 20, 2)).astype(np.float64)
    y = np.array([0, 1, 0], dtype=np.int8)
    clf = _fit(max_epochs=3).fit(X, y)
    assert clf.early_stop_reason_ == "no_internal_val"
    assert clf.internal_val_n_ == 0
