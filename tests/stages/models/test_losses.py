"""Tests for the N08 #5D-6 section-7.5 diagnostic loss functions."""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.losses import (
    balanced_softmax_loss,
    class_balanced_loss_effective_number,
    cross_entropy_loss,
    focal_loss,
    weighted_cross_entropy_train_prior_loss,
)


def _ref_ce(logits, targets, weight=None):
    """Independent reference cross-entropy (for cross-checking)."""
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    ce = -logp[np.arange(len(targets)), targets]
    if weight is None:
        return float(ce.mean())
    w = weight[targets]
    return float((w * ce).sum() / w.sum())


_LOGITS = np.array([[2.0, 0.0], [0.0, 1.0], [1.5, -0.5], [-1.0, 2.0]])
_TARGETS = np.array([0, 1, 0, 1], dtype=np.int64)


# ---- 1. Known value + reference cross-check ----------------------------

def test_cross_entropy_known_value():
    # Hand-computed: CE([[2,0],[0,1]],[0,1]) = (0.126928 + 0.313262)/2.
    logits = np.array([[2.0, 0.0], [0.0, 1.0]])
    targets = np.array([0, 1], dtype=np.int64)
    assert cross_entropy_loss(logits, targets) == pytest.approx(0.220095, abs=1e-5)


def test_cross_entropy_matches_reference():
    assert cross_entropy_loss(_LOGITS, _TARGETS) == pytest.approx(
        _ref_ce(_LOGITS, _TARGETS)
    )


def test_cross_entropy_uniform_weight_equals_unweighted():
    w = np.array([1.0, 1.0])
    assert cross_entropy_loss(_LOGITS, _TARGETS, weight=w) == pytest.approx(
        cross_entropy_loss(_LOGITS, _TARGETS)
    )


def test_cross_entropy_equal_huge_logits_is_log2():
    # P1 regression: equal huge logits -> p = [0.5, 0.5] -> CE = log(2), NOT 0
    # (a naive max + logsumexp loses the log term at magnitude 1e20).
    val = cross_entropy_loss(np.array([[1e20, 1e20]]), np.array([0], dtype=np.int64))
    assert val == pytest.approx(np.log(2.0), abs=1e-9)


def test_balanced_softmax_known_value_locks_log_prior_sign():
    # P3: equal logits + prior (0.9, 0.1), target 0 -> -log(0.9); locks that the
    # log-prior is ADDED (not subtracted) and survives (no huge-offset rounding).
    val = balanced_softmax_loss(
        np.array([[0.0, 0.0]]), np.array([0], dtype=np.int64), train_class_prior=(0.9, 0.1)
    )
    assert val == pytest.approx(-np.log(0.9), abs=1e-9)


def test_balanced_softmax_huge_offset_preserves_prior():
    # P1 regression: a huge common logit offset must not round the prior away.
    val = balanced_softmax_loss(
        np.array([[1e20, 1e20]]), np.array([0], dtype=np.int64), train_class_prior=(0.9, 0.1)
    )
    assert val == pytest.approx(-np.log(0.9), abs=1e-9)


# ---- 2. Reduce-to-CE equivalences -------------------------------------

def test_focal_gamma_zero_equals_ce():
    assert focal_loss(_LOGITS, _TARGETS, gamma=0.0) == pytest.approx(
        cross_entropy_loss(_LOGITS, _TARGETS)
    )


def test_balanced_softmax_uniform_prior_equals_ce():
    assert balanced_softmax_loss(
        _LOGITS, _TARGETS, train_class_prior=(0.5, 0.5)
    ) == pytest.approx(cross_entropy_loss(_LOGITS, _TARGETS))


def test_weighted_ce_balanced_prior_equals_ce():
    assert weighted_cross_entropy_train_prior_loss(
        _LOGITS, _TARGETS, train_class_prior=(0.5, 0.5)
    ) == pytest.approx(cross_entropy_loss(_LOGITS, _TARGETS))


def test_class_balanced_equal_samples_equals_ce():
    assert class_balanced_loss_effective_number(
        _LOGITS, _TARGETS, samples_per_class=(50, 50)
    ) == pytest.approx(cross_entropy_loss(_LOGITS, _TARGETS))


# ---- 3. Direction checks ----------------------------------------------

def test_focal_downweights_easy_examples():
    # Confident-correct samples: focal (gamma>0) must be below plain CE.
    logits = np.array([[5.0, 0.0], [0.0, 5.0]])
    targets = np.array([0, 1], dtype=np.int64)
    assert focal_loss(logits, targets, gamma=2.0) < cross_entropy_loss(logits, targets)


def test_weighted_and_cb_upweight_mispredicted_minority():
    # Majority class 0 (3 easy-correct) + minority class 1 (1 mispredicted, high CE).
    logits = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [2.0, 0.0]])
    targets = np.array([0, 0, 0, 1], dtype=np.int64)
    ce = cross_entropy_loss(logits, targets)
    # Up-weighting the high-CE minority must raise the loss above plain CE.
    assert weighted_cross_entropy_train_prior_loss(
        logits, targets, train_class_prior=(0.75, 0.25)
    ) > ce
    assert class_balanced_loss_effective_number(
        logits, targets, samples_per_class=(3, 1)
    ) > ce


def test_balanced_softmax_skewed_prior_differs_from_ce():
    bs = balanced_softmax_loss(_LOGITS, _TARGETS, train_class_prior=(0.9, 0.1))
    assert bs != pytest.approx(cross_entropy_loss(_LOGITS, _TARGETS))


# ---- 4. Numerical stability -------------------------------------------

def test_large_logits_no_overflow():
    logits = np.array([[50.0, -50.0], [-50.0, 50.0]])
    targets = np.array([0, 1], dtype=np.int64)
    for value in (
        cross_entropy_loss(logits, targets),
        focal_loss(logits, targets, gamma=2.0),
        balanced_softmax_loss(logits, targets, train_class_prior=(0.5, 0.5)),
        class_balanced_loss_effective_number(logits, targets, samples_per_class=(10, 10)),
    ):
        assert np.isfinite(value)


def test_class_balanced_stable_near_beta_one():
    # beta=0.9999 with very different, large class counts must stay finite.
    val = class_balanced_loss_effective_number(
        _LOGITS, _TARGETS, samples_per_class=(1_000_000, 10), beta=0.9999
    )
    assert np.isfinite(val)


# ---- 5. Guards ---------------------------------------------------------

def test_reject_empty_batch():
    with pytest.raises(ValueError, match="n_samples >= 1"):
        cross_entropy_loss(np.zeros((0, 2)), np.zeros((0,), dtype=np.int64))


def test_reject_logits_not_n_by_2():
    with pytest.raises(ValueError, match=r"\(n_samples, 2\)"):
        cross_entropy_loss(np.zeros((3, 3)), np.array([0, 1, 0], dtype=np.int64))


def test_reject_non_finite_logits():
    bad = _LOGITS.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/inf"):
        cross_entropy_loss(bad, _TARGETS)


def test_reject_bool_targets():
    with pytest.raises(ValueError, match="non-bool"):
        cross_entropy_loss(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([True, False]))


def test_reject_targets_out_of_domain():
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        cross_entropy_loss(_LOGITS, np.array([0, 1, 0, 2], dtype=np.int64))


def test_reject_prior_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        weighted_cross_entropy_train_prior_loss(_LOGITS, _TARGETS, train_class_prior=(0.5, 0.6))


def test_reject_prior_with_zero():
    with pytest.raises(ValueError, match="in .0, 1."):
        balanced_softmax_loss(_LOGITS, _TARGETS, train_class_prior=(0.0, 1.0))


def test_reject_negative_gamma():
    with pytest.raises(ValueError, match="gamma"):
        focal_loss(_LOGITS, _TARGETS, gamma=-1.0)


def test_reject_bool_gamma():
    with pytest.raises(ValueError, match="gamma"):
        focal_loss(_LOGITS, _TARGETS, gamma=True)


def test_reject_alpha_out_of_range():
    with pytest.raises(ValueError, match="alpha"):
        focal_loss(_LOGITS, _TARGETS, gamma=2.0, alpha=1.5)


def test_reject_beta_out_of_range():
    with pytest.raises(ValueError, match="beta"):
        class_balanced_loss_effective_number(_LOGITS, _TARGETS, samples_per_class=(10, 10), beta=1.5)


def test_reject_samples_non_positive():
    with pytest.raises(ValueError, match="samples_per_class"):
        class_balanced_loss_effective_number(_LOGITS, _TARGETS, samples_per_class=(0, 10))


def test_reject_samples_bool():
    with pytest.raises(ValueError, match="samples_per_class"):
        class_balanced_loss_effective_number(_LOGITS, _TARGETS, samples_per_class=(True, 10))


def test_reject_weight_wrong_shape():
    with pytest.raises(ValueError, match="weight"):
        cross_entropy_loss(_LOGITS, _TARGETS, weight=np.array([1.0]))


def test_reject_weight_non_positive():
    with pytest.raises(ValueError, match="weight"):
        cross_entropy_loss(_LOGITS, _TARGETS, weight=np.array([1.0, -1.0]))
