"""Control / ablation classifier scaffolds for N08 section 7.1.

  - ``LastStepLightGBMControl``       last-step features fit by LightGBM; the
                                      simple-control that deep families must
                                      beat (section 11.1 tier escalation +
                                      section 9.4 hard stop).
  - ``LastStepMLPSequenceAblation``   last-step features fit by a tiny MLP;
                                      isolates the "sequence vs. last-step"
                                      effect.

Both consume the same ``X`` shape as deep families
(``(n_samples, window_size, n_features)``) and internally slice to the last
bar (``X[:, -1, :]``). The N08 orchestrator does not need to know whether a
candidate is a deep model or a control.

Substantive bodies are the second half of N08 task #4.
"""

from __future__ import annotations

import numpy as np


class LastStepLightGBMControl:
    """LightGBM control on last-bar features (sklearn-style scaffold).

    Hyperparameters mirror the N05 / N03 LightGBM family; the orchestrator
    picks the same per-fold split policy and dummy-baseline contract.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastStepLightGBMControl":
        raise NotImplementedError(
            "LastStepLightGBMControl.fit is a scaffold; N08 task #4 half 2 "
            "wires this against the locked Stage 0 fold rows."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LastStepLightGBMControl.predict_proba is a scaffold; see fit."
        )


class LastStepMLPSequenceAblation:
    """Last-step MLP ablation (sklearn-style scaffold).

    Isolates the contribution of sequence information by training a tiny MLP
    on ``X[:, -1, :]`` only.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastStepMLPSequenceAblation":
        raise NotImplementedError(
            "LastStepMLPSequenceAblation.fit is a scaffold; N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LastStepMLPSequenceAblation.predict_proba is a scaffold; see fit."
        )
