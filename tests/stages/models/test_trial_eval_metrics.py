"""Tests for the N08 #5E-2 08X trial-eval metrics (`deep_sequence/metrics.py`)."""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.metrics import (
    METRIC_COLUMNS,
    compute_trial_metrics,
)


def _tk(n, name="A"):
    return np.full(n, name, dtype=object)


# ---- 1. Known values ---------------------------------------------------

def test_perfect_predictions():
    m = compute_trial_metrics(
        np.array([0, 1]), np.array([0, 1]), _tk(2)
    )
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["stratified_dummy_macro_f1_same_rows"] == pytest.approx(0.5)
    assert m["delta_macro_f1_vs_dummy"] == pytest.approx(0.5)
    assert m["class0_pred_rate"] == pytest.approx(0.5)
    assert m["class1_pred_rate"] == pytest.approx(0.5)
    assert m["ticker_max_share"] == pytest.approx(1.0)


def test_known_imperfect_case():
    # y_true=[0,0,1,1], y_pred=[0,1,1,1]: F1_0=2/3, F1_1=0.8 -> macro 0.73333.
    m = compute_trial_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), _tk(4)
    )
    assert m["macro_f1"] == pytest.approx(0.733333, abs=1e-5)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["class0_pred_rate"] == pytest.approx(0.25)
    assert m["class1_pred_rate"] == pytest.approx(0.75)
    assert m["delta_macro_f1_vs_dummy"] == pytest.approx(0.733333 - 0.5, abs=1e-5)


# ---- 2. Schema match ---------------------------------------------------

def test_keys_match_metric_columns_exactly():
    m = compute_trial_metrics(np.array([0, 1]), np.array([0, 1]), _tk(2))
    assert tuple(m.keys()) == METRIC_COLUMNS


def test_metric_columns_subset_of_contract_ledger_schema():
    from intraday_research.contracts.deep_sequence_exploration import (
        REQUIRED_TRIAL_LEDGER_COLUMNS,
    )
    assert set(METRIC_COLUMNS).issubset(REQUIRED_TRIAL_LEDGER_COLUMNS)


# ---- 3. Deterministic stratified null = 0.5 ----------------------------

@pytest.mark.parametrize(
    "y_true",
    [
        np.array([0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),  # imbalanced 1:3
        np.array([0] * 7 + [1] * 3),  # imbalanced 7:3
    ],
)
def test_stratified_null_is_half_for_any_both_class_truth(y_true):
    y_pred = y_true.copy()
    m = compute_trial_metrics(y_true, y_pred, _tk(len(y_true)))
    assert m["stratified_dummy_macro_f1_same_rows"] == pytest.approx(0.5)
    assert m["delta_macro_f1_vs_dummy"] == pytest.approx(m["macro_f1"] - 0.5)


def test_deterministic_repeated_calls_identical():
    args = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), _tk(4))
    assert compute_trial_metrics(*args) == compute_trial_metrics(*args)


# ---- 4. Class collapse -------------------------------------------------

def test_class_collapse_all_zero_predictions():
    m = compute_trial_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), _tk(4)
    )
    assert m["class0_pred_rate"] == pytest.approx(1.0)
    assert m["class1_pred_rate"] == pytest.approx(0.0)  # < 0.05 -> class-collapse guard
    # F1_0=2/3, F1_1=0 (zero_division=0) -> macro 1/3.
    assert m["macro_f1"] == pytest.approx(1.0 / 3.0, abs=1e-6)


# ---- 5. ticker_max_share ----------------------------------------------

def test_ticker_max_share_single_ticker():
    m = compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), _tk(4))
    assert m["ticker_max_share"] == pytest.approx(1.0)


def test_ticker_max_share_balanced_two_tickers():
    tk = np.array(["A", "A", "B", "B"], dtype=object)
    m = compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), tk)
    assert m["ticker_max_share"] == pytest.approx(0.5)


def test_ticker_max_share_skewed():
    tk = np.array(["A", "A", "A", "B"], dtype=object)
    m = compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), tk)
    assert m["ticker_max_share"] == pytest.approx(0.75)


# ---- 6. Guards ---------------------------------------------------------

def test_reject_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        compute_trial_metrics(np.zeros((2, 2), dtype=np.int64), np.array([0, 1]), _tk(2))


def test_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_trial_metrics(np.array([0, 1, 0]), np.array([0, 1]), _tk(2))


def test_reject_non_binary():
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        compute_trial_metrics(np.array([0, 2]), np.array([0, 1]), _tk(2))


def test_reject_bool_pred():
    with pytest.raises(ValueError, match="non-bool"):
        compute_trial_metrics(np.array([0, 1]), np.array([True, False]), _tk(2))


def test_reject_single_class_y_true():
    with pytest.raises(ValueError, match="both classes"):
        compute_trial_metrics(np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1]), _tk(4))


def test_reject_missing_ticker_id():
    tk = np.array(["A", None, "A", "A"], dtype=object)
    with pytest.raises(ValueError, match="missing"):
        compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), tk)


def test_reject_object_array_numpy_float_nan_ticker():
    # np.float32 NaN inside an object array is NOT a Python float (Codex P2).
    tk = np.array(["A", np.float32("nan"), "A", "A"], dtype=object)
    with pytest.raises(ValueError, match="missing"):
        compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), tk)


def test_reject_pandas_na_ticker():
    # pd.NA must be caught by the pandas.isna path (Codex follow-up P3), not blow
    # up later in np.unique with a non-builder TypeError.
    import pandas as pd

    tk = np.array(["A", pd.NA, "A", "A"], dtype=object)
    with pytest.raises(ValueError, match="missing"):
        compute_trial_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), tk)
