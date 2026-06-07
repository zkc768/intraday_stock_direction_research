"""DLinear classifier scaffold for N08 section 7.2.

Search axes (section 7.2):
  - ``moving_avg_kernel`` in {3, 5, 7, 11}
  - ``individual_channels`` in {False, True}
  - ``linear_head`` in {'shared', 'per_channel'}
  - ``seasonal_trend_dropout`` in {0.0, 0.05, 0.10}
  - ``input_projection`` in {'none', 'linear_bottleneck'}

Substantive training body is the second half of N08 task #4 and is gated on
Resume Gate Phase 1+2+4+7 passing.
"""

from __future__ import annotations

import numpy as np


class DLinearClassifier:
    """DLinear sklearn-style classifier scaffold.

    Implements ``SequenceClassifier`` protocol from ``base.py``. ``fit`` and
    ``predict_proba`` raise ``NotImplementedError`` until substantive work
    lands.
    """

    def __init__(
        self,
        *,
        moving_avg_kernel: int = 5,
        individual_channels: bool = False,
        linear_head: str = "shared",
        seasonal_trend_dropout: float = 0.0,
        input_projection: str = "none",
        random_state: int | None = None,
    ) -> None:
        self.moving_avg_kernel = moving_avg_kernel
        self.individual_channels = individual_channels
        self.linear_head = linear_head
        self.seasonal_trend_dropout = seasonal_trend_dropout
        self.input_projection = input_projection
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DLinearClassifier":
        raise NotImplementedError(
            "DLinearClassifier.fit is a scaffold; substantive deep model "
            "code is N08 task #4 half 2, gated on Resume Gate section 3 "
            "post-migration tests being green."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DLinearClassifier.predict_proba is a scaffold; see fit."
        )
