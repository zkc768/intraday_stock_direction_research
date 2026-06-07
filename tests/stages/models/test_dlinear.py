"""Tests for the N08 #5D-1 DLinear classifier body (CPU PyTorch)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _make_xy(n: int = 48, length: int = 20, c: int = 3, seed: int = 0):
    """Random windows with both classes guaranteed present."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (rng.random(n) < 0.5).astype(np.int8)
    y[0] = 0
    y[1] = 1
    return X, y


def _make_sequence_signal(n: int = 240, length: int = 20, seed: int = 1):
    """Label depends on early-vs-late SLOPE; the last bar is identical (0) for
    every window, so a last-step-only model is at chance — only a model that
    reads the temporal shape can separate these."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, length, 1), dtype=np.float64)
    y = np.zeros(n, dtype=np.int8)
    t = np.linspace(-1.0, 1.0, length)
    for i in range(n):
        label = i % 2
        slope = 1.0 if label == 1 else -1.0
        series = slope * t + rng.standard_normal(length) * 0.05
        series = series - series[-1]  # force identical last value (0) everywhere
        X[i, :, 0] = series
        y[i] = label
    return X, y


# --------------------------------------------------------------------------
# 1. Protocol conformance
# --------------------------------------------------------------------------

def test_is_sequence_classifier():
    assert isinstance(DLinearClassifier(random_state=0), SequenceClassifier)


def test_fit_returns_self_and_proba_contract():
    X, y = _make_xy(n=40, seed=2)
    clf = DLinearClassifier(random_state=0)
    assert clf.fit(X, y) is clf
    proba = clf.predict_proba(X)
    assert proba.shape == (40, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


# --------------------------------------------------------------------------
# 2. Determinism + no global-state pollution
# --------------------------------------------------------------------------

def test_determinism_same_seed_bit_exact():
    X, y = _make_xy(n=48, seed=3)
    p1 = DLinearClassifier(random_state=7).fit(X, y).predict_proba(X)
    p2 = DLinearClassifier(random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_fit_restores_global_deterministic_state():
    before_det = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    before_rng = torch.random.get_rng_state()
    X, y = _make_xy(n=32, seed=5)
    DLinearClassifier(random_state=1).fit(X, y)
    assert torch.are_deterministic_algorithms_enabled() == before_det
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn
    # manual_seed() inside fit must not pollute the global torch RNG stream.
    assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_internal_early_stop_split_is_chronological_tail_not_random():
    # AGENTS.md §4.1 forbids random splits / shuffled validation. The model body
    # may only reserve a tail slice from the already chronology-safe fit order.
    y = np.array([0, 1] * 10, dtype=np.int8)
    fit_idx, val_idx = DLinearClassifier(
        random_state=0, early_stopping_fraction=0.20
    )._early_stop_split(y)
    np.testing.assert_array_equal(fit_idx, np.arange(16, dtype=np.int64))
    np.testing.assert_array_equal(val_idx, np.arange(16, 20, dtype=np.int64))


# --------------------------------------------------------------------------
# 3. Sequence-only signal (proves the temporal path beats a last-step shortcut)
# --------------------------------------------------------------------------

def test_learns_sequence_only_signal():
    X, y = _make_sequence_signal(n=240, seed=1)
    clf = DLinearClassifier(random_state=0, max_epochs=80).fit(X, y)
    pred = clf.predict_proba(X).argmax(axis=1)
    assert (pred == y).mean() > 0.7


# --------------------------------------------------------------------------
# 4. Search-axis coverage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", [3, 5, 7, 11])
def test_axis_moving_avg_kernel(kernel):
    X, y = _make_xy(n=36, seed=2)
    clf = DLinearClassifier(moving_avg_kernel=kernel, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (36, 2)


@pytest.mark.parametrize("individual", [False, True])
def test_axis_individual_channels(individual):
    X, y = _make_xy(n=36, seed=2)
    clf = DLinearClassifier(individual_channels=individual, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (36, 2)


@pytest.mark.parametrize("head", ["shared", "per_channel"])
def test_axis_linear_head(head):
    X, y = _make_xy(n=36, seed=2)
    clf = DLinearClassifier(linear_head=head, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (36, 2)


@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.10])
def test_axis_dropout(dropout):
    X, y = _make_xy(n=36, seed=2)
    clf = DLinearClassifier(seasonal_trend_dropout=dropout, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (36, 2)


@pytest.mark.parametrize("proj", ["none", "linear_bottleneck"])
def test_axis_input_projection(proj):
    X, y = _make_xy(n=36, seed=2)
    clf = DLinearClassifier(input_projection=proj, random_state=0).fit(X, y)
    assert clf.predict_proba(X).shape == (36, 2)


# --------------------------------------------------------------------------
# 5. Guards (exact-type axes + input validation)
# --------------------------------------------------------------------------

def test_reject_bad_axis_value():
    with pytest.raises(ValueError, match="moving_avg_kernel"):
        DLinearClassifier(moving_avg_kernel=4, random_state=0)


def test_reject_individual_channels_int_aliasing_true():
    with pytest.raises(ValueError, match="individual_channels"):
        DLinearClassifier(individual_channels=1, random_state=0)


def test_reject_dropout_bool_aliasing_zero():
    with pytest.raises(ValueError, match="seasonal_trend_dropout"):
        DLinearClassifier(seasonal_trend_dropout=False, random_state=0)


def test_reject_bad_string_axis():
    with pytest.raises(ValueError, match="linear_head"):
        DLinearClassifier(linear_head="bogus", random_state=0)


def test_reject_random_state_none():
    # Constructible with defaults (protocol/orchestrator), but fit requires a seed.
    X, y = _make_xy(n=20, seed=2)
    assert DLinearClassifier().random_state is None
    with pytest.raises(ValueError, match="random_state"):
        DLinearClassifier().fit(X, y)


def test_reject_random_state_bool():
    X, y = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="random_state"):
        DLinearClassifier(random_state=True).fit(X, y)


def test_reject_max_epochs_bool():
    with pytest.raises(ValueError, match="max_epochs"):
        DLinearClassifier(random_state=0, max_epochs=True)


def test_reject_nonpositive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        DLinearClassifier(random_state=0, batch_size=0)


def test_reject_early_stopping_fraction_out_of_range():
    with pytest.raises(ValueError, match="early_stopping_fraction"):
        DLinearClassifier(random_state=0, early_stopping_fraction=1.0)


def test_reject_x_not_3d():
    clf = DLinearClassifier(random_state=0)
    with pytest.raises(ValueError, match="3-D"):
        clf.fit(np.zeros((10, 20), dtype=np.float64), np.zeros(10, dtype=np.int8))


@pytest.mark.parametrize("shape", [(10, 0, 3), (10, 20, 0)])
def test_reject_x_empty_temporal_or_feature_axis(shape):
    X = np.zeros(shape, dtype=np.float64)
    y = np.array([0, 1] * 5, dtype=np.int8)
    with pytest.raises(ValueError, match="window_size and n_features"):
        DLinearClassifier(random_state=0).fit(X, y)


def test_reject_x_non_finite():
    X, y = _make_xy(n=20, seed=2)
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/inf"):
        DLinearClassifier(random_state=0).fit(X, y)


def test_reject_y_not_binary():
    X, y = _make_xy(n=20, seed=2)
    y = y.copy()
    y[0] = 2
    with pytest.raises(ValueError, match="must be in"):
        DLinearClassifier(random_state=0).fit(X, y)


def test_reject_y_single_class():
    X, _ = _make_xy(n=20, seed=2)
    y = np.zeros(20, dtype=np.int8)
    with pytest.raises(ValueError, match="both classes"):
        DLinearClassifier(random_state=0).fit(X, y)


def test_reject_y_length_mismatch():
    X, y = _make_xy(n=20, seed=2)
    with pytest.raises(ValueError, match="same length"):
        DLinearClassifier(random_state=0).fit(X, y[:-1])


def test_reject_predict_before_fit():
    X, _ = _make_xy(n=10, seed=2)
    with pytest.raises(RuntimeError, match="before fit"):
        DLinearClassifier(random_state=0).predict_proba(X)


def test_reject_predict_shape_drift():
    X, y = _make_xy(n=20, length=20, c=3, seed=2)
    clf = DLinearClassifier(random_state=0).fit(X, y)
    X_bad, _ = _make_xy(n=5, length=20, c=4, seed=2)  # different n_features
    with pytest.raises(ValueError, match="differs from the fitted"):
        clf.predict_proba(X_bad)


# --------------------------------------------------------------------------
# 6. Early-stop bookkeeping
# --------------------------------------------------------------------------

def test_early_stop_records_fields():
    X, y = _make_xy(n=200, seed=4)
    clf = DLinearClassifier(
        random_state=0, max_epochs=10, early_stopping_patience=2
    ).fit(X, y)
    assert 1 <= clf.actual_epochs_ <= 10
    assert clf.early_stop_reason_ in {"patience", "max_epochs", "no_internal_val"}
    assert clf.internal_val_n_ >= 1


def test_no_internal_val_path_small_n():
    X = np.random.default_rng(4).standard_normal((3, 20, 2)).astype(np.float64)
    y = np.array([0, 1, 0], dtype=np.int8)
    clf = DLinearClassifier(random_state=0, max_epochs=3).fit(X, y)
    assert clf.early_stop_reason_ == "no_internal_val"
    assert clf.internal_val_n_ == 0
    assert clf.actual_epochs_ == 3
