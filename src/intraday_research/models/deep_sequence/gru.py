"""Shallow GRU classifier scaffold for N08 section 7.1 ``shallow_gru`` family.

Substantive training body is the second half of N08 task #4 and is gated on
Resume Gate Phase 1+2+4+7 passing.
"""

from __future__ import annotations

import numpy as np


class ShallowGRUClassifier:
    """Shallow GRU sklearn-style classifier scaffold.

    Implements ``SequenceClassifier`` protocol from ``base.py``. ``fit`` and
    ``predict_proba`` raise ``NotImplementedError`` until substantive work
    lands.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        head: str = "last_step",
        random_state: int | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head = head
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShallowGRUClassifier":
        raise NotImplementedError(
            "ShallowGRUClassifier.fit is a scaffold; substantive deep model "
            "code is N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ShallowGRUClassifier.predict_proba is a scaffold; see fit."
        )
