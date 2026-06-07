"""Shared sklearn-style protocol for N08 deep-sequence model families.

Every classifier family in this subpackage implements ``SequenceClassifier`` so
the N08 stage orchestrator can iterate over candidate families without
family-specific code paths.

Input convention (matches active Stage 0 freeze):
  X: ``ndarray`` of shape ``(n_samples, window_size, n_features)`` -- one
     chronological inner window per sample. ``window_size`` is locked to 20
     per ``docs/CONFIG_SCREENING_FREEZE_2026-06-04.md``.
  y: ``ndarray`` of shape ``(n_samples,)`` with binary values in ``{0, 1}``.
     Label semantics follow the frozen ``h03_bps1p5`` label config.

Output convention:
  ``predict_proba`` returns shape ``(n_samples, 2)`` with rows summing to 1.

This module declares the protocol only; nothing here trains a model, reads
data, or writes artifacts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SequenceClassifier(Protocol):
    """Sklearn-style binary classifier over fixed-length sequence windows.

    Implementations may add family-specific keyword arguments to ``__init__``
    (see ``dlinear.py``, ``tcn.py``, etc.) but ``fit`` and ``predict_proba``
    signatures are fixed so the N08 orchestrator can call them uniformly.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SequenceClassifier":
        """Fit on train-inner rows only; never see official validation or holdout."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class probabilities of shape ``(n_samples, 2)``."""
        ...
