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


# ---- SmallFusionMLP (slice 2) -----------------------------------------

# Spec-introduced "shallow MLP head" axes. 08X-eligibility: the ``ms_dlinear_tcn``
# family is search-eligible, but these variant sub-axes need a config /
# search-space mirror before an 08X run varies them (parallels the GRU/LSTM gate).
_MLP_HIDDEN_SIZES: tuple[int, ...] = (8, 16, 32)
_MLP_DROPOUTS: tuple[float, ...] = (0.0, 0.05, 0.10)
# Fixed internal MLP-stacking hyperparameters (NOT search axes).
_OOF_TAIL_FRACTION = 0.3
_MLP_LEARNING_RATE = 1e-3
_MLP_MAX_EPOCHS = 200


def _build_fusion_mlp(in_features: int, hidden_size: int, dropout: float):
    """Small stacking head: ``Linear -> ReLU -> Dropout -> Linear(2)``."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 2),
    )


class SmallFusionMLP(_FusionBase):
    """Section 7.4 ``small_fusion_mlp`` — a small MLP head over the two models'
    pre-softmax logits, trained on chronological out-of-fold (OOF) logits.

    To avoid an optimistic stacker (Codex design review), the base models are fit
    on a chronological PREFIX and the MLP trains on their logits over the trailing
    OOF TAIL (no random split / no shuffle — AGENTS §4.1). Prediction uses the
    prefix-fit base models + the trained MLP, so the MLP never sees in-sample
    sub-model logits.

    08X-eligibility: the ``ms_dlinear_tcn`` family is search-eligible, but
    ``mlp_hidden_size`` / ``mlp_dropout`` are spec-introduced sub-axes that need a
    config / search-space mirror before an 08X run varies them.
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
        self._mlp = None
        self._validate_mlp_axes()

    def _validate_mlp_axes(self) -> None:
        if type(self.mlp_hidden_size) is not int or (
            self.mlp_hidden_size not in _MLP_HIDDEN_SIZES
        ):
            raise ValueError(
                f"mlp_hidden_size must be one of {_MLP_HIDDEN_SIZES} (exact int); "
                f"got {self.mlp_hidden_size!r}"
            )
        if type(self.mlp_dropout) is not float or self.mlp_dropout not in _MLP_DROPOUTS:
            raise ValueError(
                f"mlp_dropout must be one of {_MLP_DROPOUTS} (exact float); "
                f"got {self.mlp_dropout!r}"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmallFusionMLP":
        self._check_random_state()
        from intraday_research.models.deep_sequence._torch_base import (
            _SequenceTorchClassifier,
        )
        from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
        from intraday_research.models.deep_sequence.tcn import TCNClassifier

        x_arr = _SequenceTorchClassifier._validate_x(X, where="fit")
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                "fit: y must be 1-D with the same length as X; got "
                f"y.shape={y_arr.shape}, X.shape={x_arr.shape}"
            )
        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError(f"fit: y must be integer in {{0, 1}}; got {y_arr.dtype}")
        if set(int(v) for v in np.unique(y_arr)) != {0, 1}:
            raise ValueError("fit: y must contain both classes 0 and 1")
        y_arr = y_arr.astype(np.int64, copy=False)

        # Chronological OOF split: fit base on the prefix, train the MLP on the
        # trailing OOF tail (AGENTS §4.1 — no random split, no reshuffle).
        n = x_arr.shape[0]
        n_tail = int(round(n * _OOF_TAIL_FRACTION))
        if n_tail < 1 or (n - n_tail) < 2:
            raise ValueError(
                f"SmallFusionMLP: too few rows ({n}) for a chronological OOF "
                "split (need a both-class prefix plus a non-empty tail)"
            )
        split_at = n - n_tail
        if set(int(v) for v in np.unique(y_arr[:split_at])) != {0, 1}:
            raise ValueError(
                "SmallFusionMLP: the chronological prefix lacks both classes; OOF "
                "stacking needs a both-class prefix + tail (AGENTS §4.1 forbids "
                "reshuffling rows to fix this)"
            )
        if set(int(v) for v in np.unique(y_arr[split_at:])) != {0, 1}:
            raise ValueError(
                "SmallFusionMLP: the chronological OOF tail lacks both classes; "
                "OOF stacking needs a both-class prefix + tail"
            )

        dlinear = DLinearClassifier(
            random_state=self.random_state, **self.dlinear_config
        ).fit(x_arr[:split_at], y_arr[:split_at])
        tcn = TCNClassifier(
            random_state=self.random_state, **self.tcn_config
        ).fit(x_arr[:split_at], y_arr[:split_at])

        feat = np.concatenate(
            [
                dlinear._predict_logits(x_arr[split_at:]),
                tcn._predict_logits(x_arr[split_at:]),
            ],
            axis=1,
        ).astype(np.float32)
        mlp = self._train_mlp(feat, y_arr[split_at:])

        # Atomic assign after all three succeed (Codex P2).
        self._dlinear = dlinear
        self._tcn = tcn
        self._mlp = mlp
        return self

    def _train_mlp(self, feat: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn

        # Same determinism global-state save/restore as the single-model base.
        prev_det = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        prev_rng = torch.random.get_rng_state()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(self.random_state)
            mlp = _build_fusion_mlp(
                feat.shape[1], self.mlp_hidden_size, self.mlp_dropout
            )
            x_t = torch.from_numpy(feat)
            y_t = torch.from_numpy(y)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=_MLP_LEARNING_RATE)
            loss_fn = nn.CrossEntropyLoss()
            mlp.train()
            for _ in range(_MLP_MAX_EPOCHS):  # full-batch; no shuffle (AGENTS §4.1)
                optimizer.zero_grad()
                loss_fn(mlp(x_t), y_t).backward()
                optimizer.step()
            mlp.eval()
        finally:
            torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
            torch.random.set_rng_state(prev_rng)
        return mlp

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mlp is None or self._dlinear is None or self._tcn is None:
            raise RuntimeError(
                f"{type(self).__name__}.predict_proba called before fit; "
                "call .fit(X, y) first."
            )
        import torch

        feat = np.concatenate(
            [self._dlinear._predict_logits(X), self._tcn._predict_logits(X)], axis=1
        ).astype(np.float32)
        self._mlp.eval()  # force eval at predict (defensive; mirrors the base)
        with torch.no_grad():
            logits = self._mlp(torch.from_numpy(feat)).numpy().astype(np.float64)
        return self._stable_softmax(logits)


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
