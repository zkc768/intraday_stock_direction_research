"""Loss functions for N08 section 7.5 — diagnostic numpy value-losses.

See docs/superpowers/specs/2026-06-07-n08-deep-losses-design.md.

Five allowed losses:
  - ``cross_entropy_loss``                       baseline
  - ``weighted_cross_entropy_train_prior_loss``  weighted by train-fold prior
  - ``focal_loss``                               Lin et al. 2017
  - ``class_balanced_loss_effective_number``     Cui et al. 2019
  - ``balanced_softmax_loss``                    Ren et al. 2020

Each takes pre-softmax ``logits`` ``(n_samples, 2)`` + binary ``targets``
``(n_samples,)`` and returns the scalar mean loss as a Python ``float``. These
are DIAGNOSTIC value-losses (§7.5: "loss variants are diagnostic unless
train-inner selection was predeclared; official validation may not pick a
loss") — not differentiable torch training objectives; training-loss selection
is a separate future concern (08X harness).

``train_class_prior`` / ``samples_per_class`` MUST be computed by the caller on
train-inner-fit rows only (AGENTS.md section 4.1); these functions only consume
the passed-in values.
"""

from __future__ import annotations

import numpy as np


# ---- shared helpers ---------------------------------------------------------


def _validate_logits_targets(
    logits: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate + coerce to ``(float64 (n,2), int64 (n,))``; fail loud."""
    if not isinstance(logits, np.ndarray) or logits.ndim != 2 or logits.shape[1] != 2:
        shape = logits.shape if isinstance(logits, np.ndarray) else None
        raise ValueError(f"logits must be a 2-D ndarray (n_samples, 2); got shape {shape}")
    if logits.shape[0] < 1:
        raise ValueError("logits must have n_samples >= 1; got an empty batch")
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError(f"logits must be a float ndarray; got {logits.dtype}")
    if not np.isfinite(logits).all():
        raise ValueError("logits contains NaN/inf")
    if not isinstance(targets, np.ndarray) or targets.ndim != 1:
        shape = targets.shape if isinstance(targets, np.ndarray) else None
        raise ValueError(f"targets must be a 1-D ndarray (n_samples,); got shape {shape}")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError(
            f"targets length {targets.shape[0]} != logits n_samples {logits.shape[0]}"
        )
    if targets.dtype == np.bool_ or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError(
            f"targets must be an integer (non-bool) ndarray in {{0, 1}}; got {targets.dtype}"
        )
    classes = set(int(v) for v in np.unique(targets))
    if not classes.issubset({0, 1}):
        raise ValueError(f"targets must be in {{0, 1}}; got classes {sorted(classes)}")
    return logits.astype(np.float64, copy=False), targets.astype(np.int64, copy=False)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax over the 2 classes.

    Returns from the MAX-CENTERED logits (``z = logits - max``) rather than
    ``logits - (max + logsumexp)``: the latter loses the ``logsumexp`` term to
    rounding when logits share a huge offset (e.g. ``[1e20, 1e20]`` -> CE 0 instead
    of log 2). Centering first keeps every quantity small (Codex P1).
    """
    z = logits - logits.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _ce_per_sample(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Per-sample cross-entropy ``-log p[i, targets[i]]`` (stable)."""
    log_p = _log_softmax(logits)
    return -log_p[np.arange(targets.shape[0]), targets]


def _weighted_mean(values: np.ndarray, sample_weights: np.ndarray) -> float:
    """``Σ w·v / Σ w`` — PyTorch ``reduction="mean"`` for weighted losses.

    Weights are scaled by their max before reduction (value-invariant) so extreme
    but valid weights cannot overflow ``Σ w·v`` or underflow ``Σ w`` (Codex P2);
    ``w.max() > 0`` since all weights are positive and ``n >= 1``.
    """
    w = sample_weights.astype(np.float64, copy=False)
    w = w / w.max()
    return float((w * values).sum() / w.sum())


def _validate_class_weight(ce: np.ndarray, targets: np.ndarray, weight: np.ndarray) -> float:
    if not isinstance(weight, np.ndarray) or weight.shape != (2,):
        shape = weight.shape if isinstance(weight, np.ndarray) else None
        raise ValueError(f"weight must be a (2,) ndarray or None; got shape {shape}")
    if (
        not np.issubdtype(weight.dtype, np.floating)
        or not np.isfinite(weight).all()
        or (weight <= 0.0).any()
    ):
        raise ValueError(f"weight must be a finite, all-positive float (2,) ndarray; got {weight!r}")
    return _weighted_mean(ce, weight.astype(np.float64)[targets])


def _validate_prior(train_class_prior: tuple[float, float], *, name: str = "train_class_prior") -> None:
    if type(train_class_prior) is not tuple or len(train_class_prior) != 2:
        raise ValueError(f"{name} must be a 2-tuple (prior_0, prior_1); got {train_class_prior!r}")
    for p in train_class_prior:
        if type(p) is not float or not (0.0 < p < 1.0):
            raise ValueError(f"{name} entries must be floats in (0, 1); got {train_class_prior!r}")
    if abs(sum(train_class_prior) - 1.0) > 1e-6:
        raise ValueError(
            f"{name} must sum to 1.0; got {train_class_prior!r} (sum={sum(train_class_prior)})"
        )


# ---- the five losses --------------------------------------------------------


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    weight: np.ndarray | None = None,
) -> float:
    """Section 7.5 ``cross_entropy``. Optional per-class ``weight`` (2,) uses the
    PyTorch ``reduction="mean"`` denominator ``Σ weight[targets]``."""
    log_arr, t = _validate_logits_targets(logits, targets)
    ce = _ce_per_sample(log_arr, t)
    if weight is None:
        return float(ce.mean())
    return _validate_class_weight(ce, t, weight)


def weighted_cross_entropy_train_prior_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    train_class_prior: tuple[float, float],
) -> float:
    """Section 7.5 ``weighted_cross_entropy_train_prior``.

    Inverse-prior class weighting (``weight[c] = 1 / prior[c]``) so the loss does
    not inherit the train imbalance. ``train_class_prior`` MUST be computed on
    train-inner-fit rows only (AGENTS.md section 4.1). Reduces to plain CE under a
    balanced prior (equal weights cancel in the weighted mean).
    """
    _validate_prior(train_class_prior)
    weight = np.array(
        [1.0 / train_class_prior[0], 1.0 / train_class_prior[1]], dtype=np.float64
    )
    return cross_entropy_loss(logits, targets, weight=weight)


def focal_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> float:
    """Section 7.5 ``focal_loss`` (Lin et al. 2017):
    ``FL = mean( α_t · (1 - p_t)^gamma · (-log p_t) )`` where ``p_t`` is the
    softmax probability of the true class and ``α_t = alpha if y==1 else 1-alpha``
    (no ``alpha`` -> ``α_t = 1``). Reduces to CE at ``gamma=0, alpha=None``.
    """
    log_arr, t = _validate_logits_targets(logits, targets)
    if type(gamma) is not float or gamma < 0.0:
        raise ValueError(f"gamma must be a non-negative float; got {gamma!r}")
    if alpha is not None and (type(alpha) is not float or not (0.0 <= alpha <= 1.0)):
        raise ValueError(f"alpha must be a float in [0, 1] or None; got {alpha!r}")
    log_p_t = _log_softmax(log_arr)[np.arange(t.shape[0]), t]
    p_t = np.exp(log_p_t)
    fl = (1.0 - p_t) ** gamma * (-log_p_t)
    if alpha is not None:
        fl = np.where(t == 1, alpha, 1.0 - alpha) * fl
    return float(fl.mean())


def class_balanced_loss_effective_number(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    samples_per_class: tuple[int, int],
    beta: float = 0.9999,
) -> float:
    """Section 7.5 ``class_balanced_loss_effective_number`` (Cui et al. 2019):
    weight each class by ``1 / E_c`` with effective number
    ``E_c = (1 - beta^{n_c}) / (1 - beta)``, then weighted CE.
    ``samples_per_class`` MUST come from train-inner-fit rows (AGENTS.md 4.1).
    """
    log_arr, t = _validate_logits_targets(logits, targets)
    if type(samples_per_class) is not tuple or len(samples_per_class) != 2:
        raise ValueError(f"samples_per_class must be a 2-tuple (n_0, n_1); got {samples_per_class!r}")
    for n_c in samples_per_class:
        if type(n_c) is not int or n_c < 1:
            raise ValueError(f"samples_per_class entries must be ints >= 1; got {samples_per_class!r}")
    if type(beta) is not float or not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be a float in (0, 1); got {beta!r}")
    counts = np.array(samples_per_class, dtype=np.float64)
    # E_c = (1 - beta^{n_c}) / (1 - beta), computed via expm1/log1p so it stays
    # stable as beta -> 1 (default 0.9999) instead of cancelling in 1 - beta**n_c.
    effective = -np.expm1(counts * np.log1p(beta - 1.0)) / (1.0 - beta)
    weight = 1.0 / effective
    ce = _ce_per_sample(log_arr, t)
    return _weighted_mean(ce, weight[t])


def balanced_softmax_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    train_class_prior: tuple[float, float],
) -> float:
    """Section 7.5 ``balanced_softmax`` (Ren et al. 2020): add the log train-prior
    to the logits before softmax CE — ``CE(logits + log(prior), targets)``.
    ``train_class_prior`` MUST come from train-inner-fit rows (AGENTS.md 4.1).
    Reduces to CE under a uniform prior (a constant logit shift cancels).
    """
    log_arr, t = _validate_logits_targets(logits, targets)
    _validate_prior(train_class_prior)
    log_prior = np.log(np.array(train_class_prior, dtype=np.float64))
    # Center the logits BEFORE adding the log-prior (softmax-CE is shift-invariant
    # per row), so a huge common logit offset cannot round the prior away before
    # the softmax sees it (Codex P1).
    centered = log_arr - log_arr.max(axis=1, keepdims=True)
    return float(_ce_per_sample(centered + log_prior[None, :], t).mean())
