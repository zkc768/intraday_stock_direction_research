"""TCN classifier scaffold for N08 section 7.3.

Search axes (section 7.3):
  - ``num_blocks`` in {2, 3, 4}
  - ``channels`` in {[16,16], [32,32], [32,32,32], [64,32,16]}
  - ``kernel_size`` in {2, 3, 5}
  - ``dilation_base`` = 2 (fixed)
  - ``dropout`` in {0.0, 0.05, 0.10, 0.20}
  - ``residual`` = True (fixed)
  - ``gating`` in {False, True}
  - ``normalization`` in {'none', 'weight_norm', 'layer_norm'}
  - ``causal`` = True (fixed; non-causal forbidden by AGENTS.md section 4.1)
  - ``head`` in {'last_step', 'attention_pooling_pre_frozen'}

Substantive training body is the second half of N08 task #4 and is gated on
Resume Gate Phase 1+2+4+7 passing.
"""

from __future__ import annotations

import numpy as np


class TCNClassifier:
    """TCN sklearn-style classifier scaffold.

    Implements ``SequenceClassifier`` protocol from ``base.py``. ``fit`` and
    ``predict_proba`` raise ``NotImplementedError`` until substantive work
    lands.
    """

    def __init__(
        self,
        *,
        num_blocks: int = 3,
        channels: tuple[int, ...] = (32, 32, 32),
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        gating: bool = False,
        normalization: str = "none",
        causal: bool = True,
        head: str = "last_step",
        random_state: int | None = None,
    ) -> None:
        if not causal:
            raise ValueError(
                "TCNClassifier requires causal=True; non-causal convolution is "
                "forbidden per AGENTS.md section 4.1 (no forward-looking features)."
            )
        self.num_blocks = num_blocks
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.residual = residual
        self.gating = gating
        self.normalization = normalization
        self.causal = causal
        self.head = head
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNClassifier":
        raise NotImplementedError(
            "TCNClassifier.fit is a scaffold; substantive deep model code is "
            "N08 task #4 half 2, gated on Resume Gate section 3 post-migration "
            "tests being green."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "TCNClassifier.predict_proba is a scaffold; see fit."
        )
