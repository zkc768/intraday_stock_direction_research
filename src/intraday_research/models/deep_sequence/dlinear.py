"""DLinear classifier for N08 section 7.2 — CPU PyTorch SequenceClassifier body.

See docs/superpowers/specs/2026-06-07-n08-deep-dlinear-classifier-design.md.

The model decomposes each input window into a moving-average trend and a
seasonal residual (the DLinear core), applies a per-component temporal linear
map that collapses the time axis to one scalar per channel, sums the two
components, and classifies with a small head.

Scope boundary: this is a data-agnostic model body. It never loads data, builds
folds, splits train/validation, or touches official-validation / holdout rows;
those are upstream (#5C window builder) and future-orchestrator concerns. The
class only fits and predicts on the ``(X, y)`` arrays it is handed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Frozen N08 section 7.2 search-axis value sets.
_MOVING_AVG_KERNELS: tuple[int, ...] = (3, 5, 7, 11)
_LINEAR_HEADS: tuple[str, ...] = ("shared", "per_channel")
_INPUT_PROJECTIONS: tuple[str, ...] = ("none", "linear_bottleneck")
_DROPOUTS: tuple[float, ...] = (0.0, 0.05, 0.10)


class _DLinearModule(nn.Module):
    """The torch network. Constructed under a seeded RNG by ``fit``."""

    def __init__(
        self,
        *,
        window_size: int,
        n_features: int,
        moving_avg_kernel: int,
        individual_channels: bool,
        linear_head: str,
        seasonal_trend_dropout: float,
        input_projection: str,
    ) -> None:
        super().__init__()
        self.kernel = moving_avg_kernel
        self.individual_channels = individual_channels
        self.linear_head = linear_head

        if input_projection == "linear_bottleneck":
            c_eff = max(1, n_features // 2)
            self.proj: nn.Linear | None = nn.Linear(n_features, c_eff, bias=False)
        else:
            c_eff = n_features
            self.proj = None
        self.c_eff = c_eff

        self.dropout = nn.Dropout(seasonal_trend_dropout)

        # Per-component temporal linear banks: time axis L -> 1 scalar/channel.
        if individual_channels:
            self.trend_w = nn.Parameter(torch.empty(c_eff, window_size))
            self.trend_b = nn.Parameter(torch.zeros(c_eff))
            self.seasonal_w = nn.Parameter(torch.empty(c_eff, window_size))
            self.seasonal_b = nn.Parameter(torch.zeros(c_eff))
            nn.init.xavier_uniform_(self.trend_w)
            nn.init.xavier_uniform_(self.seasonal_w)
        else:
            self.trend_lin = nn.Linear(window_size, 1)
            self.seasonal_lin = nn.Linear(window_size, 1)

        # Classification head -> 2 logits.
        if linear_head == "shared":
            self.head = nn.Linear(c_eff, 2)
        else:  # per_channel: shared Linear(1, 2) applied per channel, mean over channels.
            self.head = nn.Linear(1, 2)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, L, C). Edge-replicate pad over time then average-pool -> trend.
        pad = (self.kernel - 1) // 2
        xt = x.transpose(1, 2)  # (b, C, L)
        xt = nn.functional.pad(xt, (pad, pad), mode="replicate")
        trend = nn.functional.avg_pool1d(xt, kernel_size=self.kernel, stride=1)
        return trend.transpose(1, 2)  # (b, L, C)

    def _temporal(self, comp: torch.Tensor, *, is_trend: bool) -> torch.Tensor:
        # comp: (b, L, C_eff) -> (b, C_eff).
        if self.individual_channels:
            w = self.trend_w if is_trend else self.seasonal_w  # (C_eff, L)
            b = self.trend_b if is_trend else self.seasonal_b  # (C_eff,)
            return torch.einsum("blc,cl->bc", comp, w) + b
        lin = self.trend_lin if is_trend else self.seasonal_lin
        return lin(comp.transpose(1, 2)).squeeze(-1)  # (b, C_eff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            x = self.proj(x)  # (b, L, C_eff)
        trend = self._moving_average(x)
        seasonal = x - trend
        trend_feat = self.dropout(self._temporal(trend, is_trend=True))
        seasonal_feat = self.dropout(self._temporal(seasonal, is_trend=False))
        feat = trend_feat + seasonal_feat  # (b, C_eff)
        if self.linear_head == "shared":
            return self.head(feat)  # (b, 2)
        per = self.head(feat.unsqueeze(-1))  # (b, C_eff, 2)
        return per.mean(dim=1)  # (b, 2)


class DLinearClassifier:
    """DLinear sklearn-style binary classifier over fixed-length windows.

    Implements the ``SequenceClassifier`` protocol (``base.py``). Consumes
    ``X`` shaped ``(n_samples, window_size, n_features)`` (float) and ``y`` in
    ``{0, 1}``; ``predict_proba`` returns ``(n_samples, 2)`` float64 rows that
    sum to 1. Training is deterministic given ``random_state`` (required).

    No official-validation or holdout data is touched; the class only fits and
    predicts on whatever ``(X, y)`` it receives.
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
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        early_stopping_fraction: float = 0.15,
        weight_decay: float = 0.0,
    ) -> None:
        self.moving_avg_kernel = moving_avg_kernel
        self.individual_channels = individual_channels
        self.linear_head = linear_head
        self.seasonal_trend_dropout = seasonal_trend_dropout
        self.input_projection = input_projection
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_fraction = early_stopping_fraction
        self.weight_decay = weight_decay
        # Post-fit state (None / unset until ``fit`` succeeds).
        self._model: _DLinearModule | None = None
        self._window_size: int | None = None
        self._n_features: int | None = None
        self.actual_epochs_: int | None = None
        self.early_stop_reason_: str | None = None
        self.internal_val_n_: int | None = None
        self._validate_init()

    # ---- validation -----------------------------------------------------

    def _validate_init(self) -> None:
        """Exact-type + frozen-set validation (§4/§6; Codex P1: membership is
        not enough — bool is an int subclass and must not alias axes)."""
        if type(self.moving_avg_kernel) is not int or (
            self.moving_avg_kernel not in _MOVING_AVG_KERNELS
        ):
            raise ValueError(
                f"moving_avg_kernel must be one of {_MOVING_AVG_KERNELS} "
                f"(exact int); got {self.moving_avg_kernel!r}"
            )
        if type(self.individual_channels) is not bool:
            raise ValueError(
                "individual_channels must be exactly bool; got "
                f"{type(self.individual_channels).__name__}"
            )
        if type(self.linear_head) is not str or self.linear_head not in _LINEAR_HEADS:
            raise ValueError(
                f"linear_head must be one of {_LINEAR_HEADS}; got {self.linear_head!r}"
            )
        if type(self.seasonal_trend_dropout) is not float or (
            self.seasonal_trend_dropout not in _DROPOUTS
        ):
            raise ValueError(
                f"seasonal_trend_dropout must be one of {_DROPOUTS} (exact "
                f"float); got {self.seasonal_trend_dropout!r}"
            )
        if type(self.input_projection) is not str or (
            self.input_projection not in _INPUT_PROJECTIONS
        ):
            raise ValueError(
                f"input_projection must be one of {_INPUT_PROJECTIONS}; "
                f"got {self.input_projection!r}"
            )
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError(f"max_epochs must be a positive int; got {self.max_epochs!r}")
        if type(self.learning_rate) is not float or self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be a positive float; got {self.learning_rate!r}"
            )
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive int; got {self.batch_size!r}")
        if type(self.early_stopping_patience) is not int or (
            self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be a positive int; got "
                f"{self.early_stopping_patience!r}"
            )
        if type(self.early_stopping_fraction) is not float or not (
            0.0 < self.early_stopping_fraction < 1.0
        ):
            raise ValueError(
                "early_stopping_fraction must be a float in (0, 1); got "
                f"{self.early_stopping_fraction!r}"
            )
        if type(self.weight_decay) is not float or self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be a non-negative float; got {self.weight_decay!r}"
            )

    @staticmethod
    def _validate_x(X: np.ndarray, *, where: str) -> np.ndarray:
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            shape = X.shape if isinstance(X, np.ndarray) else None
            raise ValueError(
                f"{where}: X must be a 3-D ndarray (n, window_size, "
                f"n_features); got shape {shape}"
            )
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError(f"{where}: X must be a float ndarray; got {X.dtype}")
        if not np.isfinite(X).all():
            raise ValueError(f"{where}: X contains NaN/inf")
        return np.ascontiguousarray(X, dtype=np.float32)

    # ---- fit / predict --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DLinearClassifier":
        """Fit deterministically on ``(X, y)`` (train-inner-fit rows only)."""
        # random_state is validated here (not __init__) so the model is
        # constructible with defaults for the SequenceClassifier protocol /
        # orchestrator iteration, but a deep fit always requires a seed.
        if type(self.random_state) is not int:
            raise ValueError(
                "fit: random_state must be an int (None/bool rejected) — a "
                "deep model must be seeded for the reproducibility freeze "
                "contract"
            )
        x_arr = self._validate_x(X, where="fit")
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                "fit: y must be 1-D with the same length as X; got "
                f"y.shape={y_arr.shape}, X.shape={x_arr.shape}"
            )
        if not np.issubdtype(y_arr.dtype, np.integer):
            raise ValueError(f"fit: y must be integer in {{0, 1}}; got {y_arr.dtype}")
        classes = set(int(v) for v in np.unique(y_arr))
        if not classes.issubset({0, 1}):
            raise ValueError(f"fit: y must be in {{0, 1}}; got classes {sorted(classes)}")
        if classes != {0, 1}:
            raise ValueError(
                "fit: y must contain both classes 0 and 1; got "
                f"{sorted(classes)} (single-class fit rejected)"
            )

        window_size = x_arr.shape[1]
        n_features = x_arr.shape[2]
        y_arr = y_arr.astype(np.int64, copy=False)

        # Save ALL global torch state fit() mutates and restore it in finally
        # (Codex P2): the deterministic-algorithms flag + its warn_only
        # sub-state, and the global RNG stream that manual_seed() perturbs, so
        # fit() never pollutes later torch consumers.
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        prev_rng_state = torch.random.get_rng_state()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(self.random_state)
            rng = np.random.default_rng(self.random_state)
            fit_idx, val_idx = self._early_stop_split(y_arr, rng)
            self.internal_val_n_ = 0 if val_idx is None else int(val_idx.size)

            model = _DLinearModule(
                window_size=window_size,
                n_features=n_features,
                moving_avg_kernel=self.moving_avg_kernel,
                individual_channels=self.individual_channels,
                linear_head=self.linear_head,
                seasonal_trend_dropout=self.seasonal_trend_dropout,
                input_projection=self.input_projection,
            )
            self._train(model, x_arr, y_arr, fit_idx, val_idx, rng)
            self._model = model
            self._window_size = window_size
            self._n_features = n_features
        finally:
            torch.use_deterministic_algorithms(
                prev_deterministic, warn_only=prev_warn_only
            )
            torch.random.set_rng_state(prev_rng_state)
        return self

    def _early_stop_split(
        self, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Seeded internal split used ONLY for early stopping. Falls back to
        no-split when the data cannot spare a both-class fit set + non-empty
        val set."""
        n = y.shape[0]
        n_val = int(round(n * self.early_stopping_fraction))
        if n_val >= 1 and (n - n_val) >= 2:
            perm = rng.permutation(n)
            cand_val, cand_fit = perm[:n_val], perm[n_val:]
            if set(int(v) for v in np.unique(y[cand_fit])) == {0, 1} and cand_val.size:
                return cand_fit, cand_val
        return np.arange(n), None

    def _train(
        self,
        model: _DLinearModule,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        fit_idx: np.ndarray,
        val_idx: np.ndarray | None,
        rng: np.random.Generator,
    ) -> None:
        x_t = torch.from_numpy(x_arr)
        y_t = torch.from_numpy(y_arr)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        use_es = val_idx is not None
        best_val = float("inf")
        best_state: dict | None = None
        patience_ctr = 0
        reason = "no_internal_val" if not use_es else "max_epochs"

        for epoch in range(self.max_epochs):
            model.train()
            order = rng.permutation(fit_idx)
            for start in range(0, order.size, self.batch_size):
                batch = order[start: start + self.batch_size]
                idx = torch.from_numpy(np.ascontiguousarray(batch))
                optimizer.zero_grad()
                loss = loss_fn(model(x_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
            self.actual_epochs_ = epoch + 1
            if not use_es:
                continue
            model.eval()
            with torch.no_grad():
                vidx = torch.from_numpy(np.ascontiguousarray(val_idx))
                v_loss = loss_fn(model(x_t[vidx]), y_t[vidx]).item()
            if v_loss < best_val - 1e-6:
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.early_stopping_patience:
                    reason = "patience"
                    break
        if use_es and best_state is not None:
            model.load_state_dict(best_state)
        self.early_stop_reason_ = reason

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(
                "DLinearClassifier.predict_proba called before fit; "
                "call .fit(X, y) first."
            )
        x_arr = self._validate_x(X, where="predict_proba")
        if x_arr.shape[1:] != (self._window_size, self._n_features):
            raise ValueError(
                "predict_proba: X window/feature shape "
                f"{x_arr.shape[1:]} differs from the fitted "
                f"{(self._window_size, self._n_features)}"
            )
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(x_arr))
            proba = torch.softmax(logits, dim=1)
        return proba.numpy().astype(np.float64)
