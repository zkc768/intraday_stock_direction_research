"""Behavioral tests for ``purged_time_series_folds`` + ``embargoed_train_inner_folds`` (#5E-1).

Synthetic-data only. Verifies the §8.2 interior-block K-fold contract: per-ticker
chronological tiling, the SYMMETRIC label-horizon purge ``[a-k, b+k)`` (+ embargo),
disjoint/sorted/int64 pooled indices, n_folds pairs, and fail-loud guards. The
purge/embargo bands are the AGENTS §4.1 leakage red line.
"""

from __future__ import annotations

import numpy as np
import pytest

from intraday_research.models.deep_sequence.folds import (
    embargoed_train_inner_folds,
    purged_time_series_folds,
)


def _single_ticker(n: int = 30, start: int = 0):
    """Single ticker; monotonic int timestamps so global index == chrono rank."""
    timestamps = np.arange(start, start + n, dtype=np.int64)
    ticker_ids = np.full(n, "A", dtype=object)
    return timestamps, ticker_ids


def _two_tickers(n_per: int = 30):
    """Two interleaved tickers; positional order != per-ticker chrono order."""
    timestamps = np.empty(2 * n_per, dtype=np.int64)
    ticker_ids = np.empty(2 * n_per, dtype=object)
    timestamps[0::2] = np.arange(n_per, dtype=np.int64)
    timestamps[1::2] = np.arange(n_per, dtype=np.int64)
    ticker_ids[0::2] = "A"
    ticker_ids[1::2] = "B"
    return timestamps, ticker_ids


def _blocks(m: int, n_folds: int):
    return [(int(b[0]), int(b[-1]) + 1) for b in np.array_split(np.arange(m), n_folds)]


# ---- Block tiling -----------------------------------------------------

def test_blocks_tile_and_each_is_val_once():
    m, n_folds = 30, 3
    ts, tk = _single_ticker(m)
    folds = list(purged_time_series_folds(ts, tk, n_folds=n_folds, label_horizon_k=0))
    assert len(folds) == n_folds
    expected = _blocks(m, n_folds)
    covered = []
    for (_, val), (a, b) in zip(folds, expected):
        np.testing.assert_array_equal(val, np.arange(a, b))
        covered.extend(val.tolist())
    # Val blocks contiguous, disjoint, cover the whole single-ticker series.
    np.testing.assert_array_equal(np.sort(covered), np.arange(m))


# ---- Interior train on both sides -------------------------------------

def test_interior_fold_trains_both_sides_endpoints_one_side():
    m, n_folds = 30, 3
    ts, tk = _single_ticker(m)
    folds = list(purged_time_series_folds(ts, tk, n_folds=n_folds, label_horizon_k=0))
    blocks = _blocks(m, n_folds)
    # Fold 0: val is first block -> train only AFTER.
    train0, _ = folds[0]
    a0, b0 = blocks[0]
    assert not (train0 < a0).any()
    assert (train0 >= b0).any()
    # Interior fold 1: train BEFORE and AFTER.
    train1, _ = folds[1]
    a1, b1 = blocks[1]
    assert (train1 < a1).any() and (train1 >= b1).any()
    # Last fold: train only BEFORE.
    train2, _ = folds[2]
    a2, _ = blocks[2]
    assert (train2 < a2).any()
    assert not (train2 >= a2).any()


# ---- Symmetric purge / embargo bands (the §4.1 red line) --------------

@pytest.mark.parametrize("k", [0, 1, 3])
def test_purged_symmetric_band_absent(k):
    m, n_folds = 40, 4
    ts, tk = _single_ticker(m)
    folds = list(purged_time_series_folds(ts, tk, n_folds=n_folds, label_horizon_k=k))
    for (train, val), (a, b) in zip(folds, _blocks(m, n_folds)):
        np.testing.assert_array_equal(val, np.arange(a, b))
        assert train.dtype == np.int64
        # No train index lands in the symmetric purge band [a-k, b+k).
        assert np.all((train < a - k) | (train >= b + k))
        # Rows just outside the band are present (when in range).
        if a - k - 1 >= 0:
            assert (a - k - 1) in set(train.tolist())
        if b + k < m:
            assert (b + k) in set(train.tolist())


@pytest.mark.parametrize("k,e", [(0, 0), (1, 2), (2, 3)])
def test_embargoed_symmetric_band_absent(k, e):
    m, n_folds = 50, 4
    ts, tk = _single_ticker(m)
    folds = list(
        embargoed_train_inner_folds(
            ts, tk, n_folds=n_folds, label_horizon_k=k, embargo_size=e
        )
    )
    gap = k + e
    for (train, val), (a, b) in zip(folds, _blocks(m, n_folds)):
        np.testing.assert_array_equal(val, np.arange(a, b))
        assert np.all((train < a - gap) | (train >= b + gap))
        if a - gap - 1 >= 0:
            assert (a - gap - 1) in set(train.tolist())
        if b + gap < m:
            assert (b + gap) in set(train.tolist())


def test_embargo_widens_exclusion_vs_purge():
    # embargo_size>0 must remove strictly MORE train rows than pure purge.
    m, n_folds, k = 50, 4, 1
    ts, tk = _single_ticker(m)
    purged = list(purged_time_series_folds(ts, tk, n_folds=n_folds, label_horizon_k=k))
    embargoed = list(
        embargoed_train_inner_folds(ts, tk, n_folds=n_folds, label_horizon_k=k, embargo_size=3)
    )
    for (tr_p, _), (tr_e, _) in zip(purged, embargoed):
        assert len(tr_e) < len(tr_p)
        assert set(tr_e.tolist()).issubset(set(tr_p.tolist()))


# ---- Disjoint / sorted / pooled across tickers ------------------------

def test_train_val_disjoint_sorted_and_count():
    ts, tk = _single_ticker(40)
    for train, val in purged_time_series_folds(ts, tk, n_folds=4, label_horizon_k=2):
        assert not set(train.tolist()) & set(val.tolist())
        assert np.array_equal(train, np.sort(train))
        assert np.array_equal(val, np.sort(val))


def test_two_tickers_pooled_exact_global_mapping():
    # Interleaved fixture: ticker A rank r -> global index 2r, B rank r -> 2r+1.
    # Assert the EXACT pooled global indices per fold (Codex P3: %2 was too weak
    # to catch a global-blocking bug).
    n_per, n_folds, k = 30, 3, 1
    ts, tk = _two_tickers(n_per=n_per)
    folds = list(purged_time_series_folds(ts, tk, n_folds=n_folds, label_horizon_k=k))
    assert len(folds) == n_folds
    for (train, val), (a, b) in zip(folds, _blocks(n_per, n_folds)):
        val_ranks = range(a, b)
        expected_val = sorted([2 * r for r in val_ranks] + [2 * r + 1 for r in val_ranks])
        np.testing.assert_array_equal(val, np.array(expected_val, dtype=np.int64))
        lo, hi = max(0, a - k), min(n_per, b + k)
        train_ranks = list(range(0, lo)) + list(range(hi, n_per))
        expected_train = sorted([2 * r for r in train_ranks] + [2 * r + 1 for r in train_ranks])
        np.testing.assert_array_equal(train, np.array(expected_train, dtype=np.int64))


# ---- Guards -----------------------------------------------------------

def test_reject_shape_mismatch():
    ts, tk = _single_ticker(30)
    with pytest.raises(ValueError, match="same length"):
        list(purged_time_series_folds(ts, tk[:-1], n_folds=3, label_horizon_k=0))


def test_reject_n_folds_below_two():
    ts, tk = _single_ticker(30)
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        list(purged_time_series_folds(ts, tk, n_folds=1, label_horizon_k=0))


def test_reject_negative_label_horizon():
    ts, tk = _single_ticker(30)
    with pytest.raises(ValueError, match="label_horizon_k"):
        list(purged_time_series_folds(ts, tk, n_folds=3, label_horizon_k=-1))


def test_reject_negative_embargo():
    ts, tk = _single_ticker(30)
    with pytest.raises(ValueError, match="embargo_size"):
        list(embargoed_train_inner_folds(ts, tk, n_folds=3, label_horizon_k=0, embargo_size=-1))


def test_reject_too_few_samples_for_n_folds():
    ts, tk = _single_ticker(2)
    with pytest.raises(ValueError, match="needs at least n_folds"):
        list(purged_time_series_folds(ts, tk, n_folds=3, label_horizon_k=0))


def test_reject_gap_empties_a_fold():
    # m=12, n_folds=3 -> blocks of 4; k=5 empties train of the endpoint folds.
    ts, tk = _single_ticker(12)
    with pytest.raises(ValueError, match="empty"):
        list(purged_time_series_folds(ts, tk, n_folds=3, label_horizon_k=5))


def test_reject_empty_input():
    # Builder-owned fail-loud message, not a bare np.concatenate error (Codex P3).
    empty_ts = np.array([], dtype=np.int64)
    empty_tk = np.array([], dtype=object)
    with pytest.raises(ValueError, match="non-empty"):
        list(purged_time_series_folds(empty_ts, empty_tk, n_folds=2, label_horizon_k=0))
