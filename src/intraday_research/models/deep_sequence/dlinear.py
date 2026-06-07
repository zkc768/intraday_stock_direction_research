"""DLinear classifier for N08 section 7.2 — CPU PyTorch SequenceClassifier body.

See docs/superpowers/specs/2026-06-07-n08-deep-dlinear-classifier-design.md.

The model decomposes each input window into a moving-average trend and a
seasonal residual (the DLinear core), applies a per-component temporal linear
map that collapses the time axis to one scalar per channel, sums the two
components, and classifies with a small head.

Training (fit / predict / chronological-tail early stop / determinism / X
validation) is inherited from ``_SequenceTorchClassifier`` (``_torch_base.py``);
this module supplies only the DLinear axes (``_validate_axes``) and network
(``_build_module``).

Scope boundary: this is a data-agnostic model body. It never loads data, builds
folds, splits train/validation, or touches official-validation / holdout rows;
those are upstream (#5C window builder) and future-orchestrator concerns. The
class only fits and predicts on the ``(X, y)`` arrays it is handed.
"""

from __future__ import annotations

import torch
from torch import nn

from intraday_research.models.deep_sequence._torch_base import _SequenceTorchClassifier

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


class DLinearClassifier(_SequenceTorchClassifier):
    """DLinear sklearn-style binary classifier over fixed-length windows.

    Implements the ``SequenceClassifier`` protocol (``base.py``) via the shared
    ``_SequenceTorchClassifier`` base. Consumes ``X`` shaped
    ``(n_samples, window_size, n_features)`` (float) and ``y`` in ``{0, 1}``;
    ``predict_proba`` returns ``(n_samples, 2)`` float64 rows that sum to 1.
    Training is deterministic given ``random_state`` (required at fit).

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

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        return _DLinearModule(
            window_size=window_size,
            n_features=n_features,
            moving_avg_kernel=self.moving_avg_kernel,
            individual_channels=self.individual_channels,
            linear_head=self.linear_head,
            seasonal_trend_dropout=self.seasonal_trend_dropout,
            input_projection=self.input_projection,
        )
