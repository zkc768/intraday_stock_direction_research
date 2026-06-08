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
from torch import nn

from intraday_research.models.deep_sequence._torch_base import _SequenceTorchClassifier


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


# ---- DLinearTrendPlusTCNResidualFusion (slice 3) ----------------------

# Joint-module training kwargs come from the class, not the branch configs;
# reject them inside tcn_config rather than silently ignoring them (Codex P2).
_FUSION_TRAINING_KWARGS = frozenset(
    {
        "max_epochs",
        "learning_rate",
        "batch_size",
        "early_stopping_patience",
        "early_stopping_fraction",
        "weight_decay",
    }
)


class _TrendResidualModule(nn.Module):
    """Joint module: a DLinear-style trend branch (on a CAUSAL trailing moving
    average) + a full causal TCN residual branch, logits summed and co-trained.

    The CAUSAL (left-pad) moving average is what keeps ``residual[t]`` dependent
    on inputs ``<= t`` only, so the residual TCN branch is genuinely causal
    end-to-end (Codex design review P1). Standalone DLinear uses a CENTERED MA,
    which would make the residual non-causal — hence a distinct causal MA here.
    """

    def __init__(
        self, *, window_size: int, n_features: int, ma_kernel: int, tcn_module: nn.Module
    ) -> None:
        super().__init__()
        self.ma_kernel = ma_kernel
        # Trend branch: per-channel temporal linear (L -> 1) then a class head.
        self.trend_temporal = nn.Linear(window_size, 1)
        self.trend_head = nn.Linear(n_features, 2)
        # Residual branch: a full causal _TCNModule (built by a TCNClassifier proxy).
        self.tcn = tcn_module

    def _causal_moving_average(self, x: "object") -> "object":  # x: (b, L, C)
        pad = self.ma_kernel - 1
        xt = x.transpose(1, 2)  # (b, C, L)
        xt = nn.functional.pad(xt, (pad, 0), mode="replicate")  # PAST-only (left) pad
        trend = nn.functional.avg_pool1d(xt, kernel_size=self.ma_kernel, stride=1)
        return trend.transpose(1, 2)  # (b, L, C); trend[t] = mean(x[t-pad .. t])

    def _trend_logits(self, trend: "object") -> "object":  # (b, L, C) -> (b, 2)
        collapsed = self.trend_temporal(trend.transpose(1, 2)).squeeze(-1)  # (b, C)
        return self.trend_head(collapsed)

    def forward_features(self, x: "object") -> "object":
        # Residual-branch causal conv features (for the §4.1 leak test). Causal MA
        # -> residual[t] depends only on x[<=t] -> conv features[:, :, <=t] too.
        residual = x - self._causal_moving_average(x)
        return self.tcn.forward_features(residual)

    def forward(self, x: "object") -> "object":
        trend = self._causal_moving_average(x)
        residual = x - trend
        return self._trend_logits(trend) + self.tcn(residual)


class DLinearTrendPlusTCNResidualFusion(_SequenceTorchClassifier):
    """Section 7.4 ``dlinear_trend_plus_tcn_residual`` — a single JOINT module
    (not a late-fusion wrapper): a DLinear-style trend branch on a causal
    moving-average trend + a causal TCN branch on the residual, logits summed and
    co-trained. Subclasses the shared training base (one module trained on ``X``).

    ``dlinear_config`` supplies the trend ``moving_avg_kernel`` (the only DLinear
    axis the simplified linear trend branch uses); ``tcn_config`` supplies the
    residual branch's TCN architecture axes. Training kwargs (``max_epochs`` etc.)
    come from this class, not the configs.
    """

    def __init__(
        self,
        *,
        dlinear_config: dict | None = None,
        tcn_config: dict | None = None,
        random_state: int | None = None,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        early_stopping_fraction: float = 0.15,
        weight_decay: float = 0.0,
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
        self.dlinear_config = dict(dlinear_config or {})
        self.tcn_config = dict(tcn_config or {})
        if "random_state" in self.dlinear_config or "random_state" in self.tcn_config:
            raise ValueError(
                "random_state must be set on the fusion itself, not inside "
                "dlinear_config / tcn_config (ambiguous double-seed)"
            )
        super().__init__(
            random_state=random_state,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_fraction=early_stopping_fraction,
            weight_decay=weight_decay,
        )

    def _validate_axes(self) -> None:
        from intraday_research.models.deep_sequence.dlinear import _MOVING_AVG_KERNELS
        from intraday_research.models.deep_sequence.tcn import TCNClassifier

        # The simplified trend branch supports ONLY moving_avg_kernel.
        extra = set(self.dlinear_config) - {"moving_avg_kernel"}
        if extra:
            raise ValueError(
                "dlinear_config for the trend branch supports only "
                f"'moving_avg_kernel'; got unsupported keys {sorted(extra)}"
            )
        ma_kernel = self.dlinear_config.get("moving_avg_kernel", 5)
        if type(ma_kernel) is not int or ma_kernel not in _MOVING_AVG_KERNELS:
            raise ValueError(
                f"moving_avg_kernel must be one of {_MOVING_AVG_KERNELS} (exact "
                f"int); got {ma_kernel!r}"
            )
        self._ma_kernel = ma_kernel
        # tcn_config is residual-branch ARCHITECTURE only; training kwargs come
        # from this class (Codex P2: reject rather than silently ignore).
        training_in_tcn = set(self.tcn_config) & _FUSION_TRAINING_KWARGS
        if training_in_tcn:
            raise ValueError(
                "tcn_config must contain residual-branch TCN architecture axes "
                f"only; training kwargs {sorted(training_in_tcn)} belong on "
                "DLinearTrendPlusTCNResidualFusion itself"
            )
        # Validate the residual-branch TCN axes by constructing a proxy
        # (TCNClassifier.__init__ validates; no fit / RNG draw). Reused by
        # _build_module so all TCN architecture logic stays in one place.
        self._tcn_proxy = TCNClassifier(**self.tcn_config)

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        tcn_module = self._tcn_proxy._build_module(
            window_size=window_size, n_features=n_features
        )
        return _TrendResidualModule(
            window_size=window_size,
            n_features=n_features,
            ma_kernel=self._ma_kernel,
            tcn_module=tcn_module,
        )
