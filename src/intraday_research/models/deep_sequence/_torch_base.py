"""Shared training/predict machinery for N08 deep-sequence model bodies.

``_SequenceTorchClassifier`` hoists the CPU-PyTorch fit/predict loop that DLinear,
TCN, and GRU (#5D-1/2/3) would otherwise duplicate verbatim: the chronological-
tail internal early-stop split (AGENTS.md section 4.1 — no random internal split,
no mini-batch shuffle), the determinism-with-full-global-state-restore fit loop,
exact-type training-kwarg validation, X validation, ``predict_proba``, and the
optional ``forward_features`` hook used by causal-leak tests. Each concrete model
body stays a thin subclass that supplies only ``_validate_axes`` (model-specific
frozen-set axis checks) and ``_build_module`` (the torch network).

Internal module: the public ``SequenceClassifier`` protocol lives in ``base.py``
(intentionally torch-free); ``_SequenceTorchClassifier`` is never re-exported in
``__init__.__all__`` and never instantiated directly.

Scope boundary: data-agnostic. No data loading, folds, train/val splitting,
official-validation, or holdout contact — the base only fits and predicts on the
``(X, y)`` arrays a subclass is handed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _SequenceTorchClassifier:
    """Shared sklearn-style training base for deep-sequence classifiers.

    Subclasses MUST (1) set every model-specific axis attribute BEFORE calling
    ``super().__init__(...)`` (which validates them via ``_validate_axes``), and
    (2) implement ``_validate_axes`` + ``_build_module``. Everything else —
    ``fit`` / ``predict_proba`` / early stopping / determinism / X validation —
    is inherited unchanged so each model body stays thin.
    """

    def __init__(
        self,
        *,
        random_state: int | None,
        max_epochs: int,
        learning_rate: float,
        batch_size: int,
        early_stopping_patience: int,
        early_stopping_fraction: float,
        weight_decay: float,
    ) -> None:
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_fraction = early_stopping_fraction
        self.weight_decay = weight_decay
        # Post-fit state (None / unset until ``fit`` succeeds).
        self._model: nn.Module | None = None
        self._window_size: int | None = None
        self._n_features: int | None = None
        self.actual_epochs_: int | None = None
        self.early_stop_reason_: str | None = None
        self.internal_val_n_: int | None = None
        # Axes first (preserves the original DLinear/TCN validation order: both
        # validated axes before training kwargs), then the shared training
        # kwargs. The subclass has already set its axis attributes above.
        self._validate_axes()
        self._validate_training_kwargs()

    # ---- subclass hooks -------------------------------------------------

    def _validate_axes(self) -> None:
        """Validate model-specific search axes (exact-type, frozen sets)."""
        raise NotImplementedError("subclasses must implement _validate_axes")

    def _build_module(self, *, window_size: int, n_features: int) -> nn.Module:
        """Build the torch module (called under the seeded RNG inside ``fit``)."""
        raise NotImplementedError("subclasses must implement _build_module")

    # ---- shared validation ----------------------------------------------

    def _validate_training_kwargs(self) -> None:
        """Exact-type + range validation for the six shared training kwargs.

        ``random_state`` is intentionally NOT validated here: the model must be
        constructible with defaults for the ``SequenceClassifier`` protocol /
        orchestrator iteration, and a seed is required only at ``fit``.
        """
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
        """3-D float, finite, positive-dims check; returns contiguous float32.

        Messages are the DLinear-compatible wording (the shared substrings
        ``"3-D"`` / ``"window_size and n_features"`` / ``"NaN/inf"`` are matched
        by the existing model tests).
        """
        if not isinstance(X, np.ndarray) or X.ndim != 3:
            shape = X.shape if isinstance(X, np.ndarray) else None
            raise ValueError(
                f"{where}: X must be a 3-D ndarray (n, window_size, "
                f"n_features); got shape {shape}"
            )
        if X.shape[1] < 1 or X.shape[2] < 1:
            raise ValueError(
                f"{where}: X window_size and n_features must be positive; "
                f"got shape {X.shape}"
            )
        if not np.issubdtype(X.dtype, np.floating):
            raise ValueError(f"{where}: X must be a float ndarray; got {X.dtype}")
        if not np.isfinite(X).all():
            raise ValueError(f"{where}: X contains NaN/inf")
        return np.ascontiguousarray(X, dtype=np.float32)

    # ---- fit / predict --------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SequenceTorchClassifier":
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

        # Save ALL global torch state fit() mutates and restore it in finally:
        # the deterministic-algorithms flag + its warn_only sub-state, and the
        # global RNG stream that manual_seed() perturbs, so fit() never pollutes
        # later torch consumers.
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        prev_rng_state = torch.random.get_rng_state()
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(self.random_state)
            fit_idx, val_idx = self._early_stop_split(y_arr)
            self.internal_val_n_ = 0 if val_idx is None else int(val_idx.size)
            # _build_module draws from the seeded stream (weight init).
            model = self._build_module(window_size=window_size, n_features=n_features)
            self._train(model, x_arr, y_arr, fit_idx, val_idx)
            self._model = model
            self._window_size = window_size
            self._n_features = n_features
        finally:
            torch.use_deterministic_algorithms(
                prev_deterministic, warn_only=prev_warn_only
            )
            torch.random.set_rng_state(prev_rng_state)
        return self

    def _early_stop_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Chronological-tail split used ONLY for early stopping.

        Random internal splits are forbidden by AGENTS.md section 4.1. The
        caller is responsible for passing rows in a chronology-safe order; this
        method preserves that order and takes the tail as the monitoring slice.
        It falls back to no-split when the data cannot spare a both-class fit
        prefix plus a non-empty tail.
        """
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
        model: nn.Module,
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
            # Mini-batches in caller order, NO shuffle (AGENTS.md section 4.1).
            for start in range(0, fit_idx.size, self.batch_size):
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
                f"{type(self).__name__}.predict_proba called before fit; "
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

    def _forward_features(self, X: np.ndarray) -> np.ndarray:
        """Raw pre-head features from the fitted module — exposed for causal-leak
        tests. Requires the module to implement ``forward_features``; the base
        guards it so a subclass whose module lacks one (e.g. DLinear) fails loud
        and clear instead of raising a confusing bare ``AttributeError``.
        """
        if self._model is None:
            raise RuntimeError(
                f"{type(self).__name__}._forward_features called before fit."
            )
        if not hasattr(self._model, "forward_features"):
            raise NotImplementedError(
                f"{type(self).__name__} does not expose a forward_features path "
                "(its module has no causal feature hook)."
            )
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
