"""TCN classifier for N08 section 7.3 — causal dilated-conv SequenceClassifier.

See docs/superpowers/specs/2026-06-07-n08-deep-tcn-classifier-design.md.

Causal: every conv left-pads the time axis by ``(kernel-1)*dilation`` (PAST
only) and uses ``padding=0``, so output ``t`` depends solely on inputs ``<= t``
(AGENTS.md section 4.1 no-future-leak). Training (fit / predict / chronological-
tail early stop / determinism / X validation / the ``_forward_features`` causal
hook) is inherited from ``_SequenceTorchClassifier`` (``_torch_base.py``); this
module supplies only the TCN axes (``_validate_axes``) and network
(``_build_module``). ``_TCNModule.forward_features`` is the causal feature path
the inherited ``_forward_features`` exposes for the leakage test.

Scope boundary: data-agnostic model body. No data loading, folds, train/val
splitting, official-validation, or holdout contact.
"""

from __future__ import annotations

import torch
from torch import nn

from intraday_research.models.deep_sequence._torch_base import _SequenceTorchClassifier

# Frozen N08 section 7.3 search-axis value sets.
_CHANNELS_FROZEN: tuple[tuple[int, ...], ...] = (
    (16, 16),
    (32, 32),
    (32, 32, 32),
    (64, 32, 16),
)
_KERNELS: tuple[int, ...] = (2, 3, 5)
_DROPOUTS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
_NORMALIZATIONS: tuple[str, ...] = ("none", "weight_norm", "layer_norm")
_HEADS: tuple[str, ...] = ("last_step", "attention_pooling_pre_frozen")


class _CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding; output length == input length."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, dilation: int, *, weight_norm: bool
    ) -> None:
        super().__init__()
        self.left_pad = (kernel - 1) * dilation
        conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)
        # Non-deprecated parametrizations API (legacy torch.nn.utils.weight_norm
        # emits a DeprecationWarning that pytest.ini promotes to a test error).
        self.conv = (
            torch.nn.utils.parametrizations.weight_norm(conv) if weight_norm else conv
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, C, L)
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class _TCNBlock(nn.Module):
    """Two causal dilated convs + (norm) + activation + dropout + residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel: int,
        dilation: int,
        dropout: float,
        gating: bool,
        normalization: str,
    ) -> None:
        super().__init__()
        self.gating = gating
        out_eff = 2 * out_ch if gating else out_ch
        wn = normalization == "weight_norm"
        self.conv1 = _CausalConv1d(in_ch, out_eff, kernel, dilation, weight_norm=wn)
        self.conv2 = _CausalConv1d(out_ch, out_eff, kernel, dilation, weight_norm=wn)
        self.dropout = nn.Dropout(dropout)
        if normalization == "layer_norm":
            self.norm1: nn.LayerNorm | None = nn.LayerNorm(out_eff)
            self.norm2: nn.LayerNorm | None = nn.LayerNorm(out_eff)
        else:
            self.norm1 = None
            self.norm2 = None
        self.res = None if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    @staticmethod
    def _apply_norm(h: torch.Tensor, norm: nn.LayerNorm | None) -> torch.Tensor:
        if norm is None:
            return h
        # LayerNorm over channels per timestep (no time mixing -> causal-safe).
        return norm(h.transpose(1, 2)).transpose(1, 2)

    def _activate(self, z: torch.Tensor) -> torch.Tensor:
        if self.gating:
            a, b = z.chunk(2, dim=1)
            return torch.tanh(a) * torch.sigmoid(b)
        return torch.relu(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (b, in_ch, L)
        h = self.dropout(self._activate(self._apply_norm(self.conv1(x), self.norm1)))
        h = self.dropout(self._activate(self._apply_norm(self.conv2(h), self.norm2)))
        res = x if self.res is None else self.res(x)
        return h + res


class _TCNModule(nn.Module):
    """Stacked causal TCN blocks + classification head -> 2 logits."""

    def __init__(
        self,
        *,
        n_features: int,
        num_blocks: int,
        channels: tuple[int, ...],
        kernel_size: int,
        dilation_base: int,
        dropout: float,
        gating: bool,
        normalization: str,
        head: str,
    ) -> None:
        super().__init__()
        self.head_kind = head
        blocks: list[nn.Module] = []
        in_ch = n_features
        for i in range(num_blocks):
            blocks.append(
                _TCNBlock(
                    in_ch,
                    channels[i],
                    kernel=kernel_size,
                    dilation=dilation_base ** i,
                    dropout=dropout,
                    gating=gating,
                    normalization=normalization,
                )
            )
            in_ch = channels[i]
        self.blocks = nn.ModuleList(blocks)
        c_last = channels[-1]
        self.attn = nn.Linear(c_last, 1) if head == "attention_pooling_pre_frozen" else None
        self.classifier = nn.Linear(c_last, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, L, C_in) -> conv-stack features (b, c_last, L). Causal: column t
        # depends only on inputs <= t.
        h = x.transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)  # (b, c_last, L)
        if self.head_kind == "last_step":
            pooled = feats[:, :, -1]  # (b, c_last)
        else:
            ft = feats.transpose(1, 2)  # (b, L, c_last)
            weights = torch.softmax(self.attn(ft), dim=1)  # (b, L, 1) over time
            pooled = (ft * weights).sum(dim=1)  # (b, c_last)
        return self.classifier(pooled)  # (b, 2)


class TCNClassifier(_SequenceTorchClassifier):
    """Causal TCN sklearn-style binary classifier over fixed-length windows.

    Implements the ``SequenceClassifier`` protocol (``base.py``) via the shared
    ``_SequenceTorchClassifier`` base. Same I/O, determinism, and chronological-
    tail early-stop contract as ``DLinearClassifier``. No official-validation or
    holdout data is touched.
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
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        early_stopping_fraction: float = 0.15,
        weight_decay: float = 0.0,
    ) -> None:
        self.num_blocks = num_blocks
        self.channels = tuple(channels) if isinstance(channels, (list, tuple)) else channels
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.dropout = dropout
        self.residual = residual
        self.gating = gating
        self.normalization = normalization
        self.causal = causal
        self.head = head
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
        """Exact-type + frozen-set validation (§4/§6; reject bool aliases)."""
        # causal: exact True (replaces the scaffold's truthiness `if not causal`).
        if self.causal is not True:
            raise ValueError(
                "TCNClassifier requires causal is True exactly (reject False / 1 / "
                "'true'); non-causal convolution is forbidden per AGENTS.md "
                f"section 4.1. Got {self.causal!r}"
            )
        if self.residual is not True:
            raise ValueError(f"residual must be True (fixed axis); got {self.residual!r}")
        if type(self.num_blocks) is not int or self.num_blocks < 1:
            raise ValueError(f"num_blocks must be a positive int; got {self.num_blocks!r}")
        if type(self.channels) is not tuple or self.channels not in _CHANNELS_FROZEN:
            raise ValueError(
                f"channels must be one of the frozen tuples {_CHANNELS_FROZEN}; "
                f"got {self.channels!r}"
            )
        for c in self.channels:
            if type(c) is not int or c < 1:
                raise ValueError(f"channels must be positive ints; got {self.channels!r}")
        if self.num_blocks != len(self.channels):
            raise ValueError(
                f"num_blocks ({self.num_blocks}) must equal len(channels) "
                f"({len(self.channels)})"
            )
        if type(self.kernel_size) is not int or self.kernel_size not in _KERNELS:
            raise ValueError(f"kernel_size must be one of {_KERNELS}; got {self.kernel_size!r}")
        if type(self.dilation_base) is not int or self.dilation_base != 2:
            raise ValueError(f"dilation_base must be 2 (fixed); got {self.dilation_base!r}")
        if type(self.dropout) is not float or self.dropout not in _DROPOUTS:
            raise ValueError(f"dropout must be one of {_DROPOUTS}; got {self.dropout!r}")
        if type(self.gating) is not bool:
            raise ValueError(f"gating must be exactly bool; got {type(self.gating).__name__}")
        if type(self.normalization) is not str or self.normalization not in _NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_NORMALIZATIONS}; got {self.normalization!r}"
            )
        if type(self.head) is not str or self.head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}; got {self.head!r}")

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        return _TCNModule(
            n_features=n_features,
            num_blocks=self.num_blocks,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dilation_base=self.dilation_base,
            dropout=self.dropout,
            gating=self.gating,
            normalization=self.normalization,
            head=self.head,
        )
