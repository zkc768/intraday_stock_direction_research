"""Fusion classifier scaffolds for N08 section 7.4.

Four fusion variants:
  - ``DLinearTrendPlusTCNResidualFusion``       -- DLinear trend + TCN residual
  - ``DLinearLogitsPlusTCNLogitsFusion``         -- combine pre-softmax logits
  - ``LateAverageProbabilitiesFusion``           -- average post-softmax probs
  - ``SmallFusionMLP``                           -- small MLP head over both

Section 7.4 requires that fusion train-inner ``lcb_delta_macro_f1`` vs the
better of {``dlinear_only``, ``tcn_only``} exceeds
``FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS`` (= 0.003 in
``contracts.deep_sequence_exploration``); 08F refuses to freeze fusion
candidates that fail this gate. The check happens in the orchestrator, not
in this scaffold.

Substantive training bodies are the second half of N08 task #4.
"""

from __future__ import annotations

import numpy as np


class _BaseFusionScaffold:
    """Shared scaffold for fusion classifiers; not part of the public API."""

    def __init__(
        self,
        *,
        dlinear_config: dict | None = None,
        tcn_config: dict | None = None,
        random_state: int | None = None,
    ) -> None:
        self.dlinear_config = dlinear_config or {}
        self.tcn_config = tcn_config or {}
        self.random_state = random_state


class DLinearTrendPlusTCNResidualFusion(_BaseFusionScaffold):
    """Section 7.4 ``dlinear_trend_plus_tcn_residual`` variant scaffold."""

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "DLinearTrendPlusTCNResidualFusion":
        raise NotImplementedError(
            "DLinearTrendPlusTCNResidualFusion.fit is a scaffold; N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DLinearTrendPlusTCNResidualFusion.predict_proba is a scaffold; see fit."
        )


class DLinearLogitsPlusTCNLogitsFusion(_BaseFusionScaffold):
    """Section 7.4 ``dlinear_logits_plus_tcn_logits`` variant scaffold."""

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "DLinearLogitsPlusTCNLogitsFusion":
        raise NotImplementedError(
            "DLinearLogitsPlusTCNLogitsFusion.fit is a scaffold; N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DLinearLogitsPlusTCNLogitsFusion.predict_proba is a scaffold; see fit."
        )


class LateAverageProbabilitiesFusion(_BaseFusionScaffold):
    """Section 7.4 ``late_average_probabilities`` variant scaffold."""

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "LateAverageProbabilitiesFusion":
        raise NotImplementedError(
            "LateAverageProbabilitiesFusion.fit is a scaffold; N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LateAverageProbabilitiesFusion.predict_proba is a scaffold; see fit."
        )


class SmallFusionMLP(_BaseFusionScaffold):
    """Section 7.4 ``small_fusion_mlp`` variant scaffold."""

    def __init__(
        self,
        *,
        dlinear_config: dict | None = None,
        tcn_config: dict | None = None,
        mlp_hidden_size: int = 16,
        mlp_dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            dlinear_config=dlinear_config,
            tcn_config=tcn_config,
            random_state=random_state,
        )
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmallFusionMLP":
        raise NotImplementedError(
            "SmallFusionMLP.fit is a scaffold; N08 task #4 half 2."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "SmallFusionMLP.predict_proba is a scaffold; see fit."
        )
