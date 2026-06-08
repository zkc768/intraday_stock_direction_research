"""Control / ablation classifier implementations for N08 section 7.1.

  - ``LastStepLightGBMControl``       last-step features fit by LightGBM; the
                                      simple-control that deep families must
                                      beat (section 11.1 tier escalation +
                                      section 9.4 hard stop). Implemented in
                                      task #5A.
  - ``LastStepMLPSequenceAblation``   last-step features fit by a tiny CPU-PyTorch
                                      MLP; isolates the "sequence vs. last-step"
                                      effect. Implemented in task #5D-7 as a thin
                                      ``_SequenceTorchClassifier`` subclass.

Both consume the same ``X`` shape as deep families
(``(n_samples, window_size, n_features)``) and internally slice to the last
bar (``X[:, -1, :]``). The N08 orchestrator does not need to know whether a
candidate is a deep model or a control.

Dependency note: since #5D-7 this module imports ``torch`` (the MLP ablation is a
``_SequenceTorchClassifier`` subclass), so importing ``controls`` pulls torch —
``LastStepLightGBMControl`` no longer has a torch-free import path. This is fine
for the torch-centric N08 env; the lazy ``__init__`` keeps ``base``/``folds``
imports torch-free regardless. ``lightgbm`` itself stays a deferred import.
"""

from __future__ import annotations

import numpy as np
from torch import nn

from intraday_research.models.deep_sequence._torch_base import _SequenceTorchClassifier


class LastStepLightGBMControl:
    """LightGBM control on last-bar features (sklearn-style classifier).

    Hyperparameters mirror the N05 / N03 LightGBM family. The classifier
    consumes ``X`` shaped ``(n_samples, window_size, n_features)`` and
    internally slices to the last bar ``X[:, -1, :]``; the rest of the
    window is intentionally ignored so this baseline isolates the
    sequence-vs-last-step contribution that the deep families must beat
    (sections 7.1 + 11.1 + 9.4).

    No official-validation or holdout data is touched by this class; it
    only fits and predicts on whatever ``(X, y)`` it receives. Train-only
    preprocessing and split policy live in the orchestrator (AGENTS.md
    section 4.1).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        # Post-fit state. None until ``fit`` succeeds.
        self._lgbm: object | None = None
        self._classes: np.ndarray | None = None

    @staticmethod
    def _take_last_bar(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                "LastStepLightGBMControl expects X shaped "
                "(n_samples, window_size, n_features); got shape "
                f"{X.shape}."
            )
        return X[:, -1, :]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastStepLightGBMControl":
        # Deferred import so a missing lightgbm produces an explicit, actionable
        # dependency error -- not a cryptic ModuleNotFoundError, not a silent
        # fallback (per Codex review of task #5A).
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - covered by monkeypatched test
            raise ImportError(
                "LastStepLightGBMControl requires the `lightgbm` package. "
                "Install it via `pip install lightgbm>=4.0` "
                "(this project pins lightgbm==4.6.0 in requirements.txt)."
            ) from exc
        x_last = self._take_last_bar(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != x_last.shape[0]:
            raise ValueError(
                "y must be a 1-D array with the same length as X; got "
                f"y.shape={y.shape}, X.shape={X.shape}."
            )
        # N08 §7.1 last_step_lightgbm_control only handles the binary
        # direction-classification labels of the active Stage 0 freeze;
        # anything else is a contract violation upstream and must fail
        # at fit time, not silently mis-map probability columns later.
        unique_labels = set(int(v) for v in np.unique(y))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                "LastStepLightGBMControl supports binary labels {0, 1} only; "
                f"got np.unique(y)={sorted(unique_labels)}."
            )
        self._lgbm = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        self._lgbm.fit(x_last, y)
        self._classes = np.asarray(self._lgbm.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._lgbm is None:
            raise RuntimeError(
                "LastStepLightGBMControl.predict_proba called before fit; "
                "call .fit(X, y) first."
            )
        assert self._classes is not None
        x_last = self._take_last_bar(X)
        raw = np.asarray(self._lgbm.predict_proba(x_last))
        classes = np.asarray(self._classes)
        # Fast path: standard binary fit (both classes seen, columns ordered
        # [0, 1] by sklearn convention).
        if raw.shape[1] == 2 and np.array_equal(classes, np.array([0, 1])):
            return raw
        # General path: map raw columns to the canonical (col 0 = class 0,
        # col 1 = class 1) layout using ``classes_``. This handles the
        # single-class-fit case where LightGBM may return shape (n, 1) or
        # even (n, 2) with classes_ = [<single_class>]; in that case the
        # observed class column gets its probability mass, the unobserved
        # class column stays at 0 (per §14.4 the orchestrator's
        # class-collapse guardrail then flags the trial).
        full = np.zeros((raw.shape[0], 2), dtype=raw.dtype)
        for col, cls in enumerate(classes):
            cls_int = int(cls)
            if cls_int not in (0, 1):
                raise ValueError(
                    "LastStepLightGBMControl supports binary labels {0, 1}; "
                    f"got classes_={classes.tolist()}."
                )
            source_col = col if col < raw.shape[1] else 0
            full[:, cls_int] = raw[:, source_col]
        return full


# Spec-introduced "tiny MLP head" axes (mirror SmallFusionMLP). 08X-eligibility:
# last_step_mlp_sequence_ablation is already a SEARCH_ELIGIBLE family, but these
# sub-axes need a config / search-space mirror before an 08X run varies them.
_LAST_STEP_HIDDEN_SIZES: tuple[int, ...] = (8, 16, 32)
_LAST_STEP_DROPOUTS: tuple[float, ...] = (0.0, 0.05, 0.10)


class _LastStepMLPModule(nn.Module):
    """Tiny MLP over the LAST bar only -> 2 logits. Discarding the rest of the
    window is the ablation (isolates the last-step contribution)."""

    def __init__(self, *, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: "object") -> "object":  # x: (b, L, C)
        return self.net(x[:, -1, :])  # last completed bar -> (b, 2)


class LastStepMLPSequenceAblation(_SequenceTorchClassifier):
    """Last-step MLP ablation — a tiny MLP on ``X[:, -1, :]`` only, via the shared
    ``_SequenceTorchClassifier`` base (fit / determinism / chronological-tail
    early stop / predict_proba all inherited). The rest of the window is
    intentionally discarded so this baseline isolates the sequence-vs-last-step
    contribution the deep families must beat (design §14.5 / §11.1 / §9.4). No
    causal gate is needed — it reads only the last completed bar, no future path.

    08X-eligibility: the ``last_step_mlp_sequence_ablation`` family is already
    search-eligible, but ``hidden_size`` / ``dropout`` are spec-introduced
    sub-axes needing a config / search-space mirror before an 08X run varies them.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 16,
        dropout: float = 0.0,
        random_state: int | None = None,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        early_stopping_patience: int = 5,
        early_stopping_fraction: float = 0.15,
        weight_decay: float = 0.0,
    ) -> None:
        self.hidden_size = hidden_size
        self.dropout = dropout
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
        if type(self.hidden_size) is not int or self.hidden_size not in _LAST_STEP_HIDDEN_SIZES:
            raise ValueError(
                f"hidden_size must be one of {_LAST_STEP_HIDDEN_SIZES} (exact int); "
                f"got {self.hidden_size!r}"
            )
        if type(self.dropout) is not float or self.dropout not in _LAST_STEP_DROPOUTS:
            raise ValueError(
                f"dropout must be one of {_LAST_STEP_DROPOUTS} (exact float); "
                f"got {self.dropout!r}"
            )

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        return _LastStepMLPModule(
            n_features=n_features, hidden_size=self.hidden_size, dropout=self.dropout
        )
