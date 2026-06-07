"""Fusion classifiers for N08 section 7.4 — compose DLinear + TCN bodies.

See docs/superpowers/specs/2026-06-07-n08-deep-fusion-variants-design.md.

Four variants:
  - ``LateAverageProbabilitiesFusion``      -- mean of the two models' probs (impl)
  - ``DLinearLogitsPlusTCNLogitsFusion``    -- sum the two models' logits   (impl)
  - ``SmallFusionMLP``                      -- small MLP head over both      (scaffold, slice 2)
  - ``DLinearTrendPlusTCNResidualFusion``   -- joint trend/residual module   (scaffold, slice 3)

These are COMPOSITION wrappers: each holds a ``DLinearClassifier`` +
``TCNClassifier`` and combines their outputs (they are not single-module
``_SequenceTorchClassifier`` subclasses). Shared mechanics live in the non-public
``_FusionBase``. Section 7.4's "fusion must beat the better component by
``FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS`` (0.003) on train-inner LCB" gate is
the future 08F orchestrator's job, not these bodies'.

Scope boundary: data-agnostic model bodies. No data loading, folds, train/val
splitting, official-validation, or holdout contact.
"""

from __future__ import annotations

import numpy as np


class _FusionBase:
    """Shared mechanics for the section 7.4 fusion wrappers.

    Centralizes ``dlinear_config`` / ``tcn_config`` validation + defensive copy,
    nested-``random_state`` rejection, single-seed injection into both
    sub-models, the fitted-check, and a numerically stable softmax. Not a
    ``SequenceClassifier`` itself and not part of the public API.
    """

    def __init__(
        self,
        *,
        dlinear_config: dict | None = None,
        tcn_config: dict | None = None,
        random_state: int | None = None,
    ) -> None:
        if dlinear_config is not None and type(dlinear_config) is not dict:
            raise ValueError(
                f"dlinear_config must be an exact dict or None; got "
                f"{type(dlinear_config).__name__}"
            )
        if tcn_config is not None and type(tcn_config) is not dict:
            raise ValueError(
                f"tcn_config must be an exact dict or None; got "
                f"{type(tcn_config).__name__}"
            )
        # Defensive copy so later mutation of the caller's dict cannot change a
        # constructed fusion's component axes.
        self.dlinear_config = dict(dlinear_config or {})
        self.tcn_config = dict(tcn_config or {})
        self.random_state = random_state
        # Post-fit state.
        self._dlinear = None
        self._tcn = None
        if "random_state" in self.dlinear_config or "random_state" in self.tcn_config:
            raise ValueError(
                "random_state must be set on the fusion itself, not inside "
                "dlinear_config / tcn_config (ambiguous double-seed)"
            )

    # ---- shared helpers -------------------------------------------------

    def _check_random_state(self) -> None:
        if type(self.random_state) is not int:
            raise ValueError(
                "fit: random_state must be an int (None/bool rejected) — a fusion "
                "must be seeded for the reproducibility freeze contract"
            )

    def _fit_components(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a seeded DLinear + TCN on ``(X, y)``. Lazy import keeps merely
        constructing a fusion (and the interface/import tests) torch-free."""
        from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
        from intraday_research.models.deep_sequence.tcn import TCNClassifier

        # Assign only after BOTH sub-models fit successfully, so a failed refit
        # cannot leave a fresh DLinear paired with a stale TCN (Codex P2: atomic).
        dlinear = DLinearClassifier(
            random_state=self.random_state, **self.dlinear_config
        ).fit(X, y)
        tcn = TCNClassifier(
            random_state=self.random_state, **self.tcn_config
        ).fit(X, y)
        self._dlinear = dlinear
        self._tcn = tcn

    def _check_fitted(self) -> None:
        if self._dlinear is None or self._tcn is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict_proba called before fit; "
                "call .fit(X, y) first."
            )

    @staticmethod
    def _stable_softmax(z: np.ndarray) -> np.ndarray:
        """Row-wise softmax over the last axis, max-subtracted for stability.

        Handles the empty-batch case (n == 0) so logit fusion matches the single
        models, whose torch softmax already returns an empty ``(0, 2)`` rather
        than raising on a zero-size reduction (Codex P2).
        """
        z = np.asarray(z, dtype=np.float64)
        if z.shape[0] == 0:
            return z
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


class LateAverageProbabilitiesFusion(_FusionBase):
    """Section 7.4 ``late_average_probabilities`` — mean of the two models'
    post-softmax probabilities. The mean of two distributions is a distribution,
    so rows still sum to 1."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LateAverageProbabilitiesFusion":
        self._check_random_state()
        self._fit_components(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        proba = (self._dlinear.predict_proba(X) + self._tcn.predict_proba(X)) / 2.0
        return proba.astype(np.float64)


class DLinearLogitsPlusTCNLogitsFusion(_FusionBase):
    """Section 7.4 ``dlinear_logits_plus_tcn_logits`` — sum the two models'
    pre-softmax logits, then a single softmax."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DLinearLogitsPlusTCNLogitsFusion":
        self._check_random_state()
        self._fit_components(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        logits = self._dlinear._predict_logits(X) + self._tcn._predict_logits(X)
        return self._stable_softmax(logits)


class SmallFusionMLP(_FusionBase):
    """Section 7.4 ``small_fusion_mlp`` variant scaffold (slice 2).

    A small MLP head over the two models' logits, trained on chronological-OOF
    logits (design spec section 4.3). Substantive body is fusion slice 2.
    """

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
            "SmallFusionMLP.fit is a scaffold; fusion slice 2 (N08 #5D-5)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "SmallFusionMLP.predict_proba is a scaffold; see fit."
        )


class DLinearTrendPlusTCNResidualFusion(_FusionBase):
    """Section 7.4 ``dlinear_trend_plus_tcn_residual`` variant scaffold (slice 3).

    Joint module: DLinear trend branch (on a CAUSAL moving average) + TCN causal
    residual branch, logits summed and co-trained (design spec section 4.4).
    Substantive body is fusion slice 3.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DLinearTrendPlusTCNResidualFusion":
        raise NotImplementedError(
            "DLinearTrendPlusTCNResidualFusion.fit is a scaffold; fusion slice 3 "
            "(N08 #5D-5)."
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "DLinearTrendPlusTCNResidualFusion.predict_proba is a scaffold; see fit."
        )
