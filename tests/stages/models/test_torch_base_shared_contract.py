"""Shared contract tests for implemented CPU-PyTorch sequence classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.gru import ShallowGRUClassifier
from intraday_research.models.deep_sequence.lstm import ShallowLSTMClassifier
from intraday_research.models.deep_sequence.tcn import TCNClassifier


def _make_xy(n: int = 16, length: int = 20, c: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(260507)
    X = rng.standard_normal((n, length, c)).astype(np.float64)
    y = (np.arange(n) % 2).astype(np.int8)
    return X, y


def _implemented_classifiers():
    return (
        pytest.param(
            lambda: DLinearClassifier(random_state=0, max_epochs=2, batch_size=8),
            id="dlinear",
        ),
        pytest.param(
            lambda: TCNClassifier(random_state=0, max_epochs=2, batch_size=8),
            id="tcn",
        ),
        pytest.param(
            lambda: ShallowGRUClassifier(random_state=0, max_epochs=2, batch_size=8),
            id="gru",
        ),
        pytest.param(
            lambda: ShallowLSTMClassifier(random_state=0, max_epochs=2, batch_size=8),
            id="lstm",
        ),
    )


@pytest.mark.parametrize("make_clf", _implemented_classifiers())
def test_shared_predict_proba_contract(make_clf):
    X, y = _make_xy()
    clf = make_clf().fit(X, y)

    proba = clf.predict_proba(X[:5])

    assert proba.shape == (5, 2)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(proba).all()


@pytest.mark.parametrize("make_clf", _implemented_classifiers())
def test_shared_fit_rejects_non_finite_x(make_clf):
    X, y = _make_xy()
    X[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN/inf"):
        make_clf().fit(X, y)


@pytest.mark.parametrize("make_clf", _implemented_classifiers())
def test_shared_fit_rejects_y_length_mismatch(make_clf):
    X, y = _make_xy()

    with pytest.raises(ValueError, match="same length"):
        make_clf().fit(X, y[:-1])


@pytest.mark.parametrize("make_clf", _implemented_classifiers())
def test_shared_fit_rejects_y_out_of_domain(make_clf):
    X, y = _make_xy()
    y[0] = 2

    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        make_clf().fit(X, y)


@pytest.mark.parametrize("make_clf", _implemented_classifiers())
def test_shared_predict_rejects_before_fit(make_clf):
    X, _ = _make_xy()

    with pytest.raises(RuntimeError, match="before fit"):
        make_clf().predict_proba(X)
