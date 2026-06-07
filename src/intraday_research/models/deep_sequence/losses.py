"""Loss function scaffolds for N08 section 7.5.

Five allowed losses:
  - ``cross_entropy_loss``                       baseline
  - ``weighted_cross_entropy_train_prior_loss``  weighted by train-fold prior
  - ``focal_loss``                               Lin et al. 2017
  - ``class_balanced_loss_effective_number``     Cui et al. 2019
  - ``balanced_softmax_loss``                    Ren et al. 2020

Section 7.5 marks loss variants as diagnostic unless train-inner selection
was predeclared; official validation may NOT be used to pick a loss.

Each function takes pre-softmax ``logits`` of shape ``(n_samples, 2)`` and
binary ``targets`` of shape ``(n_samples,)`` and returns a scalar loss.

Substantive bodies are the second half of N08 task #4.
"""

from __future__ import annotations

import numpy as np


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    weight: np.ndarray | None = None,
) -> float:
    """Section 7.5 ``cross_entropy``."""
    raise NotImplementedError(
        "cross_entropy_loss is a scaffold; N08 task #4 half 2."
    )


def weighted_cross_entropy_train_prior_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    train_class_prior: tuple[float, float],
) -> float:
    """Section 7.5 ``weighted_cross_entropy_train_prior``.

    Weights inversely with the train-fold class prior so the loss does not
    inherit the train imbalance. ``train_class_prior`` MUST be computed on
    train-inner-fit rows only (AGENTS.md section 4.1).
    """
    raise NotImplementedError(
        "weighted_cross_entropy_train_prior_loss is a scaffold; N08 task #4 half 2."
    )


def focal_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> float:
    """Section 7.5 ``focal_loss`` (Lin et al. 2017)."""
    raise NotImplementedError(
        "focal_loss is a scaffold; N08 task #4 half 2."
    )


def class_balanced_loss_effective_number(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    samples_per_class: tuple[int, int],
    beta: float = 0.9999,
) -> float:
    """Section 7.5 ``class_balanced_loss_effective_number`` (Cui et al. 2019)."""
    raise NotImplementedError(
        "class_balanced_loss_effective_number is a scaffold; N08 task #4 half 2."
    )


def balanced_softmax_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    train_class_prior: tuple[float, float],
) -> float:
    """Section 7.5 ``balanced_softmax`` (Ren et al. 2020)."""
    raise NotImplementedError(
        "balanced_softmax_loss is a scaffold; N08 task #4 half 2."
    )
