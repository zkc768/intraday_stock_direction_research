"""Behavioral tests for ``LastStepLightGBMControl`` (#5A).

Synthetic-data tests only. No official validation, no holdout, no artifact
writes, no run_stage. Verifies the sklearn-style contract documented in
``src/intraday_research/models/deep_sequence/controls.py``:

  - fit returns self;
  - predict_proba returns shape ``(n, 2)`` with rows summing to 1;
  - only the last bar contributes (sequence ablation invariance);
  - calling predict_proba before fit fails fast (RuntimeError);
  - non-3D X / mismatched y rejected with ValueError;
  - missing ``lightgbm`` triggers explicit ImportError (not a silent fallback);
  - random_state produces reproducible predictions;
  - protocol satisfaction unchanged.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.controls import LastStepLightGBMControl


def _synthetic(n_samples: int = 64, window_size: int = 20, n_features: int = 4, seed: int = 0):
    """Synthetic 3-D windows where the signal lives in the last bar only."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, window_size, n_features)).astype(np.float32)
    # Label is a deterministic function of the last bar, first feature.
    y = (X[:, -1, 0] > 0).astype(np.int64)
    return X, y


def test_fit_returns_self():
    X, y = _synthetic()
    clf = LastStepLightGBMControl(n_estimators=10, random_state=0)
    out = clf.fit(X, y)
    assert out is clf


def test_predict_proba_returns_n_by_2_summing_to_one():
    X, y = _synthetic()
    clf = LastStepLightGBMControl(n_estimators=10, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    # Probabilities must be in [0, 1].
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_uses_only_last_bar():
    """Vary every bar except the last; predictions must be identical."""
    X, y = _synthetic()
    clf = LastStepLightGBMControl(n_estimators=20, random_state=0).fit(X, y)
    proba_original = clf.predict_proba(X)
    X_altered = X.copy()
    rng = np.random.default_rng(99)
    X_altered[:, :-1, :] = rng.standard_normal(
        (len(X), X.shape[1] - 1, X.shape[2])
    ).astype(X.dtype)
    proba_altered = clf.predict_proba(X_altered)
    np.testing.assert_array_equal(proba_original, proba_altered)


def test_unfit_predict_proba_raises_runtime_error():
    clf = LastStepLightGBMControl()
    X, _ = _synthetic(n_samples=4)
    with pytest.raises(RuntimeError, match="before fit"):
        clf.predict_proba(X)


def test_fit_rejects_non_3d_X():
    clf = LastStepLightGBMControl()
    with pytest.raises(ValueError, match=r"\(n_samples, window_size, n_features\)"):
        clf.fit(np.zeros((4, 3), dtype=np.float32), np.zeros(4, dtype=np.int64))


def test_fit_rejects_mismatched_y():
    clf = LastStepLightGBMControl()
    X = np.zeros((4, 20, 3), dtype=np.float32)
    bad_y = np.zeros((4, 2), dtype=np.int64)  # 2-D y
    with pytest.raises(ValueError, match="y must be a 1-D array"):
        clf.fit(X, bad_y)
    bad_y = np.zeros(5, dtype=np.int64)  # wrong length
    with pytest.raises(ValueError, match="y must be a 1-D array"):
        clf.fit(X, bad_y)


def test_random_state_reproducibility():
    X, y = _synthetic()
    clf_a = LastStepLightGBMControl(n_estimators=10, random_state=42).fit(X, y)
    clf_b = LastStepLightGBMControl(n_estimators=10, random_state=42).fit(X, y)
    np.testing.assert_array_equal(clf_a.predict_proba(X), clf_b.predict_proba(X))


def test_missing_lightgbm_raises_explicit_import_error(monkeypatch):
    """When lightgbm import resolves to None, fit must emit an actionable
    dependency error pointing to the project's pinned version, not a silent
    fallback or a bare ModuleNotFoundError."""
    # Block fresh `import lightgbm` by mapping it (and its submodule) to None
    # in sys.modules. The deferred import inside fit() will see ImportError.
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    clf = LastStepLightGBMControl()
    X, y = _synthetic(n_samples=4)
    with pytest.raises(ImportError, match="lightgbm"):
        clf.fit(X, y)


def test_protocol_still_satisfied_post_implementation():
    """LastStepLightGBMControl must continue to implement SequenceClassifier."""
    assert isinstance(LastStepLightGBMControl(), SequenceClassifier)


def test_fit_does_not_mutate_input_arrays():
    X, y = _synthetic()
    X_before = X.copy()
    y_before = y.copy()
    LastStepLightGBMControl(n_estimators=5, random_state=0).fit(X, y)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


# ---------------------------------------------------------------------------
# Single-class fit semantics (Codex #5A review P1).
#
# LightGBM may degenerate to a single-class predictor when a fold's y is
# entirely one class. Earlier the wrapper only triggered remapping when
# raw.shape[1] != 2; that left a silent inversion when LightGBM returned
# shape (n, 2) with classes_=[<single_class>]. The fixed implementation
# always consults classes_ to assign column semantics.
# ---------------------------------------------------------------------------


def test_single_class_all_zeros_returns_class_0_prob_near_1():
    X, _ = _synthetic(n_samples=32)
    y = np.zeros(32, dtype=np.int64)
    clf = LastStepLightGBMControl(n_estimators=5, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (32, 2)
    # Column 0 (class 0) should carry the probability mass.
    np.testing.assert_allclose(proba[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(proba[:, 1], 0.0, atol=1e-6)


def test_single_class_all_ones_returns_class_1_prob_near_1():
    X, _ = _synthetic(n_samples=32)
    y = np.ones(32, dtype=np.int64)
    clf = LastStepLightGBMControl(n_estimators=5, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (32, 2)
    # Column 1 (class 1) should carry the probability mass -- the bug fixed
    # in #5A would have placed mass in column 0 instead.
    np.testing.assert_allclose(proba[:, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(proba[:, 0], 0.0, atol=1e-6)


def test_fit_rejects_non_binary_labels():
    """LastStepLightGBMControl only handles the §7.1 binary direction labels."""
    X, _ = _synthetic(n_samples=8)
    bad_y = np.array([0, 2, 0, 2, 0, 2, 0, 2], dtype=np.int64)
    clf = LastStepLightGBMControl()
    with pytest.raises(ValueError, match=r"binary labels \{0, 1\}"):
        clf.fit(X, bad_y)
    bad_y = np.array([-1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match=r"binary labels \{0, 1\}"):
        clf.fit(X, bad_y)
