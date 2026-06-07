"""Tests for the N08 #5D-3 shallow GRU classifier body (CPU PyTorch)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.gru import ShallowGRUClassifier


def _make_xy(n: int = 24, length: int = 20, c: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _make_sequence_signal(n: int = 200, length: int = 20, seed: int = 1):
    """Label = early-vs-late slope sign; the last INPUT bar is identical (0) for
    every window, so only a model that reads the temporal shape (the GRU output
    after integrating the sequence) can separate them."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, length, 1), dtype=np.float64)
    y = np.zeros(n, dtype=np.int8)
    t = np.linspace(-1.0, 1.0, length)
    for i in range(n):
        label = i % 2
        series = (1.0 if label else -1.0) * t + rng.standard_normal(length) * 0.05
        X[i, :, 0] = series - series[-1]  # force identical last value (0)
        y[i] = label
    return X, y


def _fit(**kw):
    kw.setdefault("random_state", 0)
    return ShallowGRUClassifier(**kw)


# ---- 1. Protocol -------------------------------------------------------

def test_is_sequence_classifier():
    assert isinstance(ShallowGRUClassifier(), SequenceClassifier)


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


# ---- 3. CAUSALITY (light §4.1 gate — guards a bidirectional regression) ----

@pytest.mark.parametrize("head", ["last_step", "attention_pooling_pre_frozen"])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_causality_no_future_leak(head, num_layers):
    X, y = _make_xy(n=12, seed=3)
    clf = _fit(head=head, num_layers=num_layers).fit(X, y)
    rng = np.random.default_rng(99)
    t = 10
    xa = rng.standard_normal((1, 20, 2)).astype(np.float64)
    xb = xa.copy()
    xb[:, t + 1:, :] = rng.standard_normal((1, 20 - t - 1, 2))  # perturb the FUTURE
    fa = clf._forward_features(xa)  # (1, L, H)
    fb = clf._forward_features(xb)
    # Causal: GRU output rows at time <= t depend only on inputs <= t (identical)
    # -> bit-identical. (Time is axis 1 for GRU vs axis 2 for TCN.) A bidirectional
    # GRU would mix future bars into earlier rows and fail this.
    np.testing.assert_array_equal(fa[:, : t + 1, :], fb[:, : t + 1, :])
    assert not np.array_equal(fa[:, t + 1:, :], fb[:, t + 1:, :])  # future does differ


# ---- 4. Sequence-only signal ------------------------------------------

def test_learns_sequence_only_signal():
    X, y = _make_sequence_signal(n=200, seed=1)
    clf = _fit(random_state=0, max_epochs=80, batch_size=32).fit(X, y)
    pred = clf.predict_proba(X).argmax(axis=1)
    assert (pred == y).mean() > 0.6


# ---- 5. Axis coverage --------------------------------------------------

@pytest.mark.parametrize("hidden_size", [16, 32, 64])
def test_axis_hidden_size(hidden_size):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(hidden_size=hidden_size).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_axis_num_layers(num_layers):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(num_layers=num_layers).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.10, 0.20])
def test_axis_dropout(dropout):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(dropout=dropout).fit(X, y).predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize("head", ["last_step", "attention_pooling_pre_frozen"])
def test_axis_head(head):
    X, y = _make_xy(n=20, seed=2)
    assert _fit(head=head).fit(X, y).predict_proba(X).shape == (20, 2)


# ---- 6. Guards ---------------------------------------------------------

@pytest.mark.parametrize("bad", [True, 0, 1])
def test_reject_bidirectional_not_false(bad):
    # `is False` singleton check must reject True, 0, AND 1 (Codex design review).
    with pytest.raises(ValueError, match="bidirectional"):
        ShallowGRUClassifier(bidirectional=bad)


def test_reject_hidden_size_off_grid():
    with pytest.raises(ValueError, match="hidden_size"):
        ShallowGRUClassifier(hidden_size=48)


def test_reject_hidden_size_bool_aliasing():
    with pytest.raises(ValueError, match="hidden_size"):
        ShallowGRUClassifier(hidden_size=True)


def test_reject_num_layers_off_grid():
    with pytest.raises(ValueError, match="num_layers"):
        ShallowGRUClassifier(num_layers=3)


def test_reject_dropout_off_grid():
    with pytest.raises(ValueError, match="dropout"):
        ShallowGRUClassifier(dropout=0.5)


def test_reject_dropout_bool_aliasing_zero():
    with pytest.raises(ValueError, match="dropout"):
        ShallowGRUClassifier(dropout=False)


def test_reject_bad_head():
    with pytest.raises(ValueError, match="head"):
        ShallowGRUClassifier(head="bogus")


def test_reject_random_state_none_at_fit():
    # Constructible with defaults (protocol/orchestrator), but fit requires a seed.
    X, y = _make_xy(n=20, seed=2)
    assert ShallowGRUClassifier().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        ShallowGRUClassifier().fit(X, y)


def test_reject_random_state_bool_at_fit():
    X, y = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="random_state"):
        ShallowGRUClassifier(random_state=True).fit(X, y)


def test_reject_max_epochs_bool():
    with pytest.raises(ValueError, match="max_epochs"):
        ShallowGRUClassifier(random_state=0, max_epochs=True)


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


# ---- 8. Dropout decision (no num_layers==1 warning; pooled-feature placement) ----

def test_dropout_num_layers_one_emits_no_dropout_warning():
    # The §3 dropout decision (dropout=0.0 to nn.GRU + a pooled nn.Dropout) must
    # suppress torch's "dropout expects num_layers>1" UserWarning.
    X, y = _make_xy(n=20, seed=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(dropout=0.10, num_layers=1).fit(X, y)
    dropout_warnings = [w for w in caught if "dropout" in str(w.message).lower()]
    assert not dropout_warnings, (
        f"unexpected dropout warning(s): {[str(w.message) for w in dropout_warnings]}"
    )


@pytest.mark.parametrize("num_layers", [1, 2])
def test_dropout_axis_is_pooled_not_gru_internal(num_layers):
    X, y = _make_xy(n=20, seed=2)
    clf = _fit(dropout=0.20, num_layers=num_layers).fit(X, y)
    # The dropout axis is realized as a pooled-feature nn.Dropout, never nn.GRU's
    # between-layer dropout (Codex P3).
    assert clf._model.gru.dropout == 0.0
    assert abs(clf._model.dropout.p - 0.20) < 1e-12
