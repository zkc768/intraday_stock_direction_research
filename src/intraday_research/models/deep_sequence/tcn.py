"""TCN classifier for N08 section 7.3 — causal dilated-conv SequenceClassifier.

See docs/superpowers/specs/2026-06-07-n08-deep-tcn-classifier-design.md.

Causal: every conv left-pads the time axis by ``(kernel-1)*dilation`` (PAST
only) and uses ``padding=0``, so output ``t`` depends solely on inputs ``<= t``
(AGENTS.md section 4.1 no-future-leak). Reuses the #5D-1 DLinear training
contract (chronological-tail early-stop split, no shuffle, determinism with full
global-state restore, exact-type axis validation); the shared training loop is
duplicated for now and will be hoisted to a base class at the 3rd model (GRU).

Scope boundary: data-agnostic model body. No data loading, folds, train/val
splitting, official-validation, or holdout contact.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

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


class TCNClassifier:
    """Causal TCN sklearn-style binary classifier over fixed-length windows.

    Implements the ``SequenceClassifier`` protocol (``base.py``). Same I/O,
    determinism, and chronological-tail early-stop contract as
    ``DLinearClassifier``. No official-validation or holdout data is touched.
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
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_fraction = early_stopping_fraction
        self.weight_decay = weight_decay
        # Post-fit state (None until ``fit`` succeeds).
        self._model: _TCNModule | None = None
        self._window_size: int | None = None
        self._n_features: int | None = None
        self.actual_epochs_: int | None = None
        self.early_stop_reason_: str | None = None
        self.internal_val_n_: int | None = None
        self._validate_init()

    # ---- validation -----------------------------------------------------

    def _validate_init(self) -> None:
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
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError(f"max_epochs must be a positive int; got {self.max_epochs!r}")
        if type(self.learning_rate) is not float or self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be a positive float; got {self.learning_rate!r}")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive int; got {self.batch_size!r}")
        if type(self.early_stopping_patience) is not int or self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be a positive int; got "
                f"{self.early_stopping_patience!r}"
            )
        if type(self.early_stopping_fraction) is not float or not (
            0.0 < self.early_stopping_fraction < 1.0
        ):
            raise ValueError(
                f"early_stopping_fraction must be a float in (0, 1); got "
                f"{self.early_stopping_fraction!r}"
            )
        if type(self.weight_decay) is not float or self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be a non-negative float; got {self.weight_decay!r}")

    @staticmethod
    def _validate_x(X: np.ndarray, *, where: str) -> np.ndarray:
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            shape = X.shape if isinstance(X, np.ndarray) else None
            raise ValueError(
                f"{where}: X must be a 3-D ndarray (n, window_size, n_features); "
                f"got shape {shape}"
            )
        if X.shape[1] < 1 or X.shape[2] < 1:
            raise ValueError(
                f"{where}: X must have window_size >= 1 and n_features >= 1; "
                f"got shape {X.shape}"
            )
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError(f"{where}: X must be a float ndarray; got {X.dtype}")
        if not np.isfinite(X).all():
            raise ValueError(f"{where}: X contains NaN/inf")
        return np.ascontiguousarray(X, dtype=np.float32)

    # ---- fit / predict --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TCNClassifier":
        if type(self.random_state) is not int:
            raise ValueError(
                "fit: random_state must be an int (None/bool rejected) — a deep "
                "model must be seeded for the reproducibility freeze contract"
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

        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        prev_rng_state = torch.random.get_rng_state()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(self.random_state)
            fit_idx, val_idx = self._early_stop_split(y_arr)
            self.internal_val_n_ = 0 if val_idx is None else int(val_idx.size)
            model = _TCNModule(
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
            self._train(model, x_arr, y_arr, fit_idx, val_idx)
            self._model = model
            self._window_size = window_size
            self._n_features = n_features
        finally:
            torch.use_deterministic_algorithms(prev_deterministic, warn_only=prev_warn_only)
            torch.random.set_rng_state(prev_rng_state)
        return self

    def _early_stop_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Chronological-tail split used ONLY for early stopping (AGENTS.md
        section 4.1 forbids random internal splits). Fit = leading rows, val =
        trailing slice; falls back to no-split when too small."""
        n = y.shape[0]
        n_val = int(round(n * self.early_stopping_fraction))
        if n_val >= 1 and (n - n_val) >= 2:
            split_at = n - n_val
            cand_fit = np.arange(split_at, dtype=np.int64)
            cand_val = np.arange(split_at, n, dtype=np.int64)
            if set(int(v) for v in np.unique(y[cand_fit])) == {0, 1} and cand_val.size:
                return cand_fit, cand_val
        return np.arange(n, dtype=np.int64), None

    def _train(
        self,
        model: _TCNModule,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        fit_idx: np.ndarray,
        val_idx: np.ndarray | None,
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
            for start in range(0, fit_idx.size, self.batch_size):  # caller order; no shuffle
                batch = fit_idx[start: start + self.batch_size]
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
                "TCNClassifier.predict_proba called before fit; call .fit(X, y) first."
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
            proba = torch.softmax(self._model(torch.from_numpy(x_arr)), dim=1)
        return proba.numpy().astype(np.float64)

    def _forward_features(self, X: np.ndarray) -> np.ndarray:
        """Conv-stack features ``(n, c_last, L)`` — exposed for the causal-leak
        test (column ``t`` must depend only on inputs ``<= t``)."""
        if self._model is None:
            raise RuntimeError("TCNClassifier._forward_features called before fit.")
        x_arr = self._validate_x(X, where="_forward_features")
        if x_arr.shape[1:] != (self._window_size, self._n_features):
            raise ValueError(
                "_forward_features: X window/feature shape "
                f"{x_arr.shape[1:]} differs from the fitted "
                f"{(self._window_size, self._n_features)}"
            )
        self._model.eval()
        with torch.no_grad():
            feats = self._model.forward_features(torch.from_numpy(x_arr))
        return feats.numpy().astype(np.float64)
