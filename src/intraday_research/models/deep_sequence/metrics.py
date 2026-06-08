"""Trial-evaluation metrics for N08 section 8.3 — the 08X trial-ledger metric columns.

See docs/superpowers/specs/2026-06-07-n08-trial-eval-metrics-design.md.

``compute_trial_metrics(y_true, y_pred, ticker_ids)`` returns the eight per-trial
METRIC columns the contract's ``REQUIRED_TRIAL_LEDGER_COLUMNS`` requires
(``macro_f1``, ``balanced_accuracy``, ``accuracy``,
``stratified_dummy_macro_f1_same_rows``, ``delta_macro_f1_vs_dummy``,
``class0_pred_rate``, ``class1_pred_rate``, ``ticker_max_share``) as a
``dict[str, float]``. Pure, DETERMINISTIC, self-contained (no seed). The contract
validates the ledger SCHEMA; this module produces the VALUES — the future 08X
harness composes a ledger row from its bookkeeping columns + this dict.

08X red line (design §4.1): these are TRAIN-INNER discovery metrics only —
nothing here reads or scores official validation / holdout.
"""

from __future__ import annotations

import numpy as np

# The exact set of metric columns this module produces (subset of the contract's
# REQUIRED_TRIAL_LEDGER_COLUMNS; the harness supplies the bookkeeping columns).
METRIC_COLUMNS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "stratified_dummy_macro_f1_same_rows",
    "delta_macro_f1_vs_dummy",
    "class0_pred_rate",
    "class1_pred_rate",
    "ticker_max_share",
)


def _ticker_ids_have_missing(ticker_ids: np.ndarray) -> bool:
    """True if ``ticker_ids`` contains any missing marker, so ``ticker_max_share``
    stays auditable. Uses ``pandas.isna`` (deferred) to catch None, ``np.nan``
    (Python float AND object-array ``np.float32/64`` NaN), ``pd.NA`` and
    ``pd.NaT`` uniformly across dtypes."""
    import pandas as pd

    return bool(np.asarray(pd.isna(ticker_ids)).any())


def _validate(
    y_true: np.ndarray, y_pred: np.ndarray, ticker_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ticker_ids = np.asarray(ticker_ids)
    if y_true.ndim != 1 or y_pred.ndim != 1 or ticker_ids.ndim != 1:
        raise ValueError(
            "y_true, y_pred, ticker_ids must be 1-D arrays; got shapes "
            f"{y_true.shape}, {y_pred.shape}, {ticker_ids.shape}."
        )
    n = y_true.shape[0]
    if n < 1:
        raise ValueError("y_true must be non-empty (>= 1 sample); got 0.")
    if y_pred.shape[0] != n or ticker_ids.shape[0] != n:
        raise ValueError(
            "y_true, y_pred, ticker_ids must have the same length; got "
            f"{n}, {y_pred.shape[0]}, {ticker_ids.shape[0]}."
        )
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"{name} must be an integer (non-bool) ndarray in {{0, 1}}; "
                f"got {arr.dtype}."
            )
        classes = set(int(v) for v in np.unique(arr))
        if not classes.issubset({0, 1}):
            raise ValueError(f"{name} must be in {{0, 1}}; got classes {sorted(classes)}.")
    if set(int(v) for v in np.unique(y_true)) != {0, 1}:
        raise ValueError(
            "y_true must contain both classes 0 and 1; a single-class slice makes "
            "macro_f1 and the stratified-null delta ill-defined."
        )
    if _ticker_ids_have_missing(ticker_ids):
        raise ValueError("ticker_ids must not contain missing/NaN/None entries.")
    return y_true.astype(np.int64, copy=False), y_pred.astype(np.int64, copy=False), ticker_ids


def _stratified_null_macro_f1(y_true: np.ndarray) -> float:
    """Deterministic class-balance (PLUG-IN) stratified null: the macro-F1 of the
    EXPECTED confusion matrix of a stratified guesser (predicts class ``c`` at
    rate ``q_c = n_c / n``), i.e. ``F1(E[confusion])`` not ``E[macro-F1]``.

    Per class ``F1_null_c = 2*n_c*q_c / (n_c + n*q_c)`` (= ``q_c``), averaged
    over classes = **0.5 whenever both classes are present** (the caller requires
    that). It is used DELIBERATELY (Codex review): a deterministic, fold-size-
    independent, noise-free baseline so ``delta_macro_f1_vs_dummy`` is a stable
    selection signal under the §9.1 LCB margin. This is NOT the finite-sample
    expected macro-F1 of a seeded ``DummyClassifier(strategy="stratified")`` draw,
    which is lower for small folds (e.g. ~0.4167 at ``n=2``)."""
    n = y_true.shape[0]
    f1_null = []
    for c in (0, 1):
        n_c = int(np.sum(y_true == c))
        q_c = n_c / n
        f1_null.append(2.0 * n_c * q_c / (n_c + n * q_c))
    return float(np.mean(f1_null))


def compute_trial_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, ticker_ids: np.ndarray
) -> dict[str, float]:
    """Compute the eight §8.3 trial-ledger metric columns from a fold's true
    labels, candidate predictions, and per-row ticker ids. Deterministic; the
    stratified-null baseline is analytical (§3 of the design spec).
    """
    yt, yp, tk = _validate(y_true, y_pred, ticker_ids)
    # Deferred import so a missing scikit-learn yields an explicit, actionable
    # dependency error (mirrors LastStepLightGBMControl's lightgbm import).
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            f1_score,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "compute_trial_metrics requires scikit-learn. Install it via "
            "`pip install scikit-learn>=1.4` (this project pins it for N08 metrics)."
        ) from exc

    n = yt.shape[0]
    macro_f1 = float(f1_score(yt, yp, labels=[0, 1], average="macro", zero_division=0))
    dummy = _stratified_null_macro_f1(yt)
    _, counts = np.unique(tk, return_counts=True)
    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "stratified_dummy_macro_f1_same_rows": dummy,
        "delta_macro_f1_vs_dummy": macro_f1 - dummy,
        "class0_pred_rate": float(np.mean(yp == 0)),
        "class1_pred_rate": float(np.mean(yp == 1)),
        "ticker_max_share": float(counts.max() / n),
    }
