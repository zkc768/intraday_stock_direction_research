"""Shallow GRU classifier for N08 section 7.1 ``shallow_gru`` family.

See docs/superpowers/specs/2026-06-07-n08-deep-gru-classifier-design.md.

A unidirectional ``nn.GRU`` over the input window, then a within-window head
(``last_step`` or learned temporal ``attention_pooling_pre_frozen``), dropout on
the pooled features, and a linear classifier to 2 logits. Training (fit / predict
/ chronological-tail early stop / determinism / X validation / the
``_forward_features`` causal hook) is inherited from ``_SequenceTorchClassifier``
(``_torch_base.py``); this module supplies only the GRU axes (``_validate_axes``)
and network (``_build_module``).

Causal by construction (AGENTS.md section 4.1): a unidirectional GRU is a strict
left-to-right recurrence, so output ``t`` depends only on inputs ``<= t``.
``bidirectional`` is a fixed-``False`` axis; a bidirectional GRU would leak future
bars into earlier timesteps and is rejected.

Dropout semantics: PyTorch's ``nn.GRU(dropout=p)`` applies dropout only between
stacked layers and warns when ``num_layers == 1``. To give the ``dropout`` axis
uniform, layer-count-independent meaning AND avoid that warning, the module passes
``dropout=0.0`` to ``nn.GRU`` and realizes the axis as a single ``nn.Dropout`` on
the pooled features before the classifier.

Search-space note (08X-eligibility gate): N08 section 7.1 lists ``shallow_gru`` as
a candidate family but freezes no hyperparameter axis table, and the config YAML
has no ``gru:`` block. The frozen axes below exist for fail-loud, hash-stable
*construction* only; this body is NOT eligible for inclusion in an 08X search run
until its axes are mirrored into ``configs/stages/deep_sequence_exploration.yaml``
and ``08x_search_space.json`` and sha-stamped before trial 0 (a future, in-scope
08X-harness change). The axis VALUES below are a reviewable default.

Scope boundary: data-agnostic model body. No data loading, folds, train/val
splitting, official-validation, or holdout contact.
"""

from __future__ import annotations

import torch
from torch import nn

from intraday_research.models.deep_sequence._torch_base import _SequenceTorchClassifier

# Proposed "shallow" search-axis value sets (spec-introduced; see the module
# docstring's 08X-eligibility gate — not an upstream freeze).
_HIDDEN_SIZES: tuple[int, ...] = (16, 32, 64)
_NUM_LAYERS: tuple[int, ...] = (1, 2)
_DROPOUTS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
_HEADS: tuple[str, ...] = ("last_step", "attention_pooling_pre_frozen")


class _GRUModule(nn.Module):
    """Unidirectional GRU + within-window head -> 2 logits."""

    def __init__(
        self,
        *,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        head: str,
    ) -> None:
        super().__init__()
        self.head_kind = head
        # dropout=0.0 to nn.GRU on purpose: the `dropout` axis is realized as a
        # single pooled-feature nn.Dropout below, so it is layer-count-independent
        # and never triggers the num_layers==1 UserWarning.
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.attn = (
            nn.Linear(hidden_size, 1) if head == "attention_pooling_pre_frozen" else None
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, L, C_in) -> GRU output sequence (b, L, H). Causal: row t depends
        # only on inputs <= t (strict left-to-right recurrence).
        out, _ = self.gru(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)  # (b, L, H)
        if self.head_kind == "last_step":
            pooled = feats[:, -1, :]  # (b, H)
        else:
            weights = torch.softmax(self.attn(feats), dim=1)  # (b, L, 1) over time
            pooled = (feats * weights).sum(dim=1)  # (b, H)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)  # (b, 2)


class ShallowGRUClassifier(_SequenceTorchClassifier):
    """Shallow unidirectional-GRU sklearn-style binary classifier over windows.

    Implements the ``SequenceClassifier`` protocol (``base.py``) via the shared
    ``_SequenceTorchClassifier`` base. Same I/O, determinism, and chronological-
    tail early-stop contract as ``DLinearClassifier`` / ``TCNClassifier``. No
    official-validation or holdout data is touched.

    08X-eligibility: see the module docstring — the frozen axes are for
    construction validation only; this family is not 08X-search-eligible until
    its axes are mirrored into the frozen config + search-space and sha-stamped.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        head: str = "last_step",
        bidirectional: bool = False,
        random_state: int | None = None,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        early_stopping_fraction: float = 0.15,
        weight_decay: float = 0.0,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.head = head
        self.bidirectional = bidirectional
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
        """Exact-type + frozen-set validation (reject bool/int aliases)."""
        # bidirectional: exact False singleton (rejects True / 0 / 1) — the §4.1
        # causal red line, parallel to TCN's `causal is True`.
        if self.bidirectional is not False:
            raise ValueError(
                "ShallowGRUClassifier requires bidirectional is False exactly "
                "(reject True / 0 / 1); a bidirectional GRU leaks future bars into "
                f"earlier timesteps, forbidden per AGENTS.md section 4.1. Got "
                f"{self.bidirectional!r}"
            )
        if type(self.hidden_size) is not int or self.hidden_size not in _HIDDEN_SIZES:
            raise ValueError(
                f"hidden_size must be one of {_HIDDEN_SIZES} (exact int); "
                f"got {self.hidden_size!r}"
            )
        if type(self.num_layers) is not int or self.num_layers not in _NUM_LAYERS:
            raise ValueError(
                f"num_layers must be one of {_NUM_LAYERS} (exact int); "
                f"got {self.num_layers!r}"
            )
        if type(self.dropout) is not float or self.dropout not in _DROPOUTS:
            raise ValueError(
                f"dropout must be one of {_DROPOUTS} (exact float); got {self.dropout!r}"
            )
        if type(self.head) is not str or self.head not in _HEADS:
            raise ValueError(f"head must be one of {_HEADS}; got {self.head!r}")

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        return _GRUModule(
            n_features=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            head=self.head,
        )
