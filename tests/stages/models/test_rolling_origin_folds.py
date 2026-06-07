"""Behavioral tests for ``rolling_origin_folds`` (#5B).

Synthetic-data tests only. No raw bar data, no official validation, no
holdout, no artifact writes, no run_stage. Verifies the §8.2 contract
documented in ``src/intraday_research/models/deep_sequence/folds.py``:

  - per-ticker chronological split before pooling;
  - n_folds expanding-window validation slices;
  - label-horizon embargo on the last k train rows before each val slice;
  - per-fold train and val index arrays are disjoint, sorted, int64;
  - generator yields exactly n_folds pairs;
  - shape / parameter / sufficient-samples guards reject invalid inputs.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from intraday_research.models.deep_sequence.folds import rolling_origin_folds


def _single_ticker(n: int = 100, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Single ticker, monotonically increasing integer timestamps."""
    timestamps = np.arange(start, start + n, dtype=np.int64)
    ticker_ids = np.full(n, "A", dtype=object)
    return timestamps, ticker_ids


def _two_tickers(n_per: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Two tickers A and B, interleaved so positional order does NOT match
    per-ticker chronological order."""
    timestamps = np.empty(2 * n_per, dtype=np.int64)
    ticker_ids = np.empty(2 * n_per, dtype=object)
    timestamps[0::2] = np.arange(n_per, dtype=np.int64)
    timestamps[1::2] = np.arange(n_per, dtype=np.int64)
    ticker_ids[0::2] = "A"
    ticker_ids[1::2] = "B"
    return timestamps, ticker_ids


# ---------------------------------------------------------------------------
# Happy-path semantics
# ---------------------------------------------------------------------------


def test_single_ticker_expanding_train_fixed_val():
    timestamps, ticker_ids = _single_ticker(n=100)
    folds = list(
        rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=5,
            inner_validation_size=10,
            label_horizon_k=0,
        )
    )
    assert len(folds) == 5
    # With n=100, n_folds=5, ivs=10, k=0:
    #   fold 0: val [50, 60), train [0, 50)
    #   fold 1: val [60, 70), train [0, 60)
    #   fold 2: val [70, 80), train [0, 70)
    #   fold 3: val [80, 90), train [0, 80)
    #   fold 4: val [90, 100), train [0, 90)
    expected_train_sizes = [50, 60, 70, 80, 90]
    for (train_idx, val_idx), exp_train in zip(folds, expected_train_sizes):
        assert len(val_idx) == 10
        assert len(train_idx) == exp_train


def test_label_horizon_k_embargoes_last_k_train_rows():
    timestamps, ticker_ids = _single_ticker(n=100)
    folds = list(
        rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=5,
            inner_validation_size=10,
            label_horizon_k=3,
        )
    )
    # k=3 should shrink each train by 3 rows.
    expected_train_sizes = [50 - 3, 60 - 3, 70 - 3, 80 - 3, 90 - 3]
    for (train_idx, _), exp_train in zip(folds, expected_train_sizes):
        assert len(train_idx) == exp_train


def test_each_fold_is_chronological_per_ticker():
    timestamps, ticker_ids = _single_ticker(n=80)
    folds = list(
        rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=4,
            inner_validation_size=10,
            label_horizon_k=2,
        )
    )
    for train_idx, val_idx in folds:
        train_ts = timestamps[train_idx]
        val_ts = timestamps[val_idx]
        # All train timestamps strictly less than all val timestamps; the
        # gap is the label-horizon embargo, which leaves train_max +
        # label_horizon_k <= val_min.
        assert train_ts.max() < val_ts.min()


def test_train_val_indices_are_disjoint():
    timestamps, ticker_ids = _single_ticker(n=100)
    for train_idx, val_idx in rolling_origin_folds(
        timestamps,
        ticker_ids,
        n_folds=5,
        inner_validation_size=10,
        label_horizon_k=2,
    ):
        assert not set(train_idx.tolist()) & set(val_idx.tolist())


def test_multi_ticker_pooled_indices_pool_correctly():
    """Each ticker contributes its own per-fold slice; pooled indices are
    the union, sorted, with no cross-ticker positions leaking into the
    wrong ticker's interval."""
    timestamps, ticker_ids = _two_tickers(n_per=80)
    folds = list(
        rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=4,
            inner_validation_size=10,
            label_horizon_k=0,
        )
    )
    assert len(folds) == 4
    for train_idx, val_idx in folds:
        # Each fold's val is 10 per ticker * 2 tickers = 20 positions.
        assert len(val_idx) == 20
        # Each ticker's val slice covers the same per-ticker chronological
        # interval; check that each ticker contributes 10 val positions.
        val_tickers = ticker_ids[val_idx]
        assert (val_tickers == "A").sum() == 10
        assert (val_tickers == "B").sum() == 10
        # Train is similarly split.
        train_tickers = ticker_ids[train_idx]
        assert (train_tickers == "A").sum() == (train_tickers == "B").sum()


def test_multi_ticker_chronology_holds_within_each_ticker():
    """Per ticker, all train timestamps must precede all val timestamps."""
    timestamps, ticker_ids = _two_tickers(n_per=80)
    for train_idx, val_idx in rolling_origin_folds(
        timestamps,
        ticker_ids,
        n_folds=4,
        inner_validation_size=10,
        label_horizon_k=2,
    ):
        for tk in ("A", "B"):
            ticker_train = timestamps[train_idx][ticker_ids[train_idx] == tk]
            ticker_val = timestamps[val_idx][ticker_ids[val_idx] == tk]
            assert ticker_train.max() < ticker_val.min()


def test_returns_iterator_not_list():
    timestamps, ticker_ids = _single_ticker(n=60)
    result = rolling_origin_folds(
        timestamps,
        ticker_ids,
        n_folds=3,
        inner_validation_size=10,
        label_horizon_k=0,
    )
    assert isinstance(result, Iterator)


def test_yields_exactly_n_folds():
    timestamps, ticker_ids = _single_ticker(n=100)
    count = sum(
        1
        for _ in rolling_origin_folds(
            timestamps,
            ticker_ids,
            n_folds=7,
            inner_validation_size=5,
            label_horizon_k=0,
        )
    )
    assert count == 7


def test_unsorted_input_timestamps_still_produce_chronological_folds():
    """Input order should not matter; the builder sorts per-ticker first."""
    timestamps, ticker_ids = _single_ticker(n=80)
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(timestamps))
    shuffled_ts = timestamps[permutation]
    shuffled_tk = ticker_ids[permutation]
    folds = list(
        rolling_origin_folds(
            shuffled_ts,
            shuffled_tk,
            n_folds=4,
            inner_validation_size=10,
            label_horizon_k=0,
        )
    )
    for train_idx, val_idx in folds:
        train_ts = shuffled_ts[train_idx]
        val_ts = shuffled_ts[val_idx]
        assert train_ts.max() < val_ts.min()


def test_returned_indices_are_int64_and_sorted():
    timestamps, ticker_ids = _single_ticker(n=100)
    for train_idx, val_idx in rolling_origin_folds(
        timestamps,
        ticker_ids,
        n_folds=5,
        inner_validation_size=10,
        label_horizon_k=0,
    ):
        assert train_idx.dtype == np.int64
        assert val_idx.dtype == np.int64
        assert np.array_equal(train_idx, np.sort(train_idx))
        assert np.array_equal(val_idx, np.sort(val_idx))


# ---------------------------------------------------------------------------
# Input / parameter guards
# ---------------------------------------------------------------------------


def test_rejects_misaligned_input_lengths():
    timestamps = np.arange(50, dtype=np.int64)
    ticker_ids = np.full(40, "A", dtype=object)
    with pytest.raises(ValueError, match="same length"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=3,
                inner_validation_size=5,
                label_horizon_k=0,
            )
        )


def test_rejects_non_1d_inputs():
    timestamps = np.zeros((10, 2), dtype=np.int64)
    ticker_ids = np.full(10, "A", dtype=object)
    with pytest.raises(ValueError, match="1-D arrays"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=3,
                inner_validation_size=2,
                label_horizon_k=0,
            )
        )


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_rejects_invalid_n_folds(bad):
    timestamps, ticker_ids = _single_ticker(n=100)
    with pytest.raises(ValueError, match="n_folds"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=bad,
                inner_validation_size=10,
                label_horizon_k=0,
            )
        )


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_rejects_invalid_inner_validation_size(bad):
    timestamps, ticker_ids = _single_ticker(n=100)
    with pytest.raises(ValueError, match="inner_validation_size"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=5,
                inner_validation_size=bad,
                label_horizon_k=0,
            )
        )


@pytest.mark.parametrize("bad", [-1, -10])
def test_rejects_invalid_label_horizon_k(bad):
    timestamps, ticker_ids = _single_ticker(n=100)
    with pytest.raises(ValueError, match="label_horizon_k"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=5,
                inner_validation_size=10,
                label_horizon_k=bad,
            )
        )


def test_rejects_insufficient_samples_for_a_ticker():
    """A ticker with too few samples for n_folds * ivs + k + 1 must raise."""
    timestamps = np.concatenate(
        [np.arange(100, dtype=np.int64), np.arange(10, dtype=np.int64)]
    )
    ticker_ids = np.concatenate(
        [np.full(100, "A", dtype=object), np.full(10, "B", dtype=object)]
    )
    with pytest.raises(ValueError, match="ticker 'B'"):
        list(
            rolling_origin_folds(
                timestamps,
                ticker_ids,
                n_folds=5,
                inner_validation_size=10,
                label_horizon_k=2,
            )
        )
