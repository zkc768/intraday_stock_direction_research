"""Tests for the N08 #5C-5 sliding-window builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intraday_research.data import windows


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------

def _bars(day: str, n: int) -> np.ndarray:
    """n consecutive 5-min datetime64[ns] bars starting 09:30 on `day`."""
    start = np.datetime64(f"{day}T09:30:00", "ns")
    step = np.timedelta64(5, "m").astype("timedelta64[ns]")
    return start + step * np.arange(n, dtype="int64")


def _clean_single_day(n: int = 6, f: int = 3, partition_value: int = 0):
    """All-valid single-day inputs for the core builder."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, f)).astype(np.float64)
    labels = np.tile(np.array([0, 1], dtype=np.int8), n)[:n]
    timestamps = _bars("2014-03-03", n)
    partition = np.full(n, partition_value, dtype=np.int8)
    feature_valid_mask = np.ones(n, dtype=np.bool_)
    target_valid_mask = np.ones(n, dtype=np.bool_)
    return dict(
        features=features,
        labels=labels,
        timestamps=timestamps,
        partition=partition,
        feature_valid_mask=feature_valid_mask,
        target_valid_mask=target_valid_mask,
    )


def test_build_windows_single_ticker_basic_single_day():
    kw = _clean_single_day(n=6, f=3, partition_value=1)
    ws = 4
    result = windows.build_windows_single_ticker(
        kw["features"], kw["labels"], kw["timestamps"],
        partition=kw["partition"],
        feature_valid_mask=kw["feature_valid_mask"],
        target_valid_mask=kw["target_valid_mask"],
        window_size=ws,
    )
    # 6 bars, ws=4, all same day/valid -> targets at end_pos 3,4,5 -> 3 windows.
    assert result["X"].shape == (3, ws, 3)
    assert result["X"].dtype == np.float64
    # First window covers rows 0..3, its features verbatim.
    np.testing.assert_array_equal(result["X"][0], kw["features"][0:4])
    np.testing.assert_array_equal(result["y"], kw["labels"][3:6])
    np.testing.assert_array_equal(
        result["target_timestamps"], kw["timestamps"][3:6]
    )
    np.testing.assert_array_equal(
        result["target_row_positions"], np.array([3, 4, 5], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        result["target_partition"], np.array([1, 1, 1], dtype=np.int8)
    )


from intraday_research import baseline_v1


def _multi_day_fixture():
    """3 trading days, F=4, date-aligned partition (day 1 train, days 2-3 val),
    a couple of feature/target invalid rows."""
    f = 4
    feature_names = [f"feat{i}" for i in range(f)]
    day_lens = {"2014-03-03": 7, "2014-03-04": 8, "2014-03-05": 6}
    ts_parts = [_bars(day, k) for day, k in day_lens.items()]
    timestamps = np.concatenate(ts_parts)
    n = timestamps.shape[0]

    rng = np.random.default_rng(7)
    features = rng.standard_normal((n, f)).astype(np.float64)

    # date-aligned partition: first day train(0), rest validation(1).
    first_day_len = day_lens["2014-03-03"]
    partition = np.where(
        np.arange(n) < first_day_len, 0, 1
    ).astype(np.int8)

    feature_valid_mask = np.ones(n, dtype=np.bool_)
    feature_valid_mask[2] = False          # kills any window containing row 2
    target_valid_mask = np.ones(n, dtype=np.bool_)
    target_valid_mask[first_day_len + 1] = False  # a dropped target

    labels = np.where(np.arange(n) % 2 == 0, 0, 1).astype(np.int8)
    labels[~target_valid_mask] = np.int8(-1)  # invalid targets carry -1
    return dict(
        features=features, labels=labels, timestamps=timestamps,
        partition=partition, feature_valid_mask=feature_valid_mask,
        target_valid_mask=target_valid_mask, feature_names=feature_names,
    )


def _baseline_reference(fx, window_size):
    """Build baseline_v1 windows for both splits and return sorted X/y/ts."""
    names = fx["feature_names"]
    df = pd.DataFrame({
        "ticker": "TEST",
        "timestamp": fx["timestamps"],
        "split": np.where(fx["partition"] == 0, "train", "validation"),
        "label": np.where(
            fx["target_valid_mask"], fx["labels"].astype(float), np.nan
        ),
    })
    for i, name in enumerate(names):
        df[f"{name}_scaled"] = np.where(
            fx["feature_valid_mask"], fx["features"][:, i], np.nan
        )
    parts = []
    for split_name in ("train", "validation"):
        seg = baseline_v1.build_windows_for_segment(
            df, split_name, names, window_size
        )
        meta = seg["metadata"]
        for j in range(seg["y"].shape[0]):
            parts.append((
                seg["X"][j],
                int(seg["y"][j]),
                np.datetime64(meta["target_timestamp"].iloc[j], "ns"),
            ))
    parts.sort(key=lambda t: t[2])  # by target timestamp
    return parts


def test_build_windows_single_ticker_matches_baseline_v1():
    fx = _multi_day_fixture()
    ws = 3
    ours = windows.build_windows_single_ticker(
        fx["features"], fx["labels"], fx["timestamps"],
        partition=fx["partition"],
        feature_valid_mask=fx["feature_valid_mask"],
        target_valid_mask=fx["target_valid_mask"],
        window_size=ws,
    )
    ref = _baseline_reference(fx, ws)

    assert ours["y"].shape[0] == len(ref)
    # ours is already in global time order; ref sorted by target timestamp.
    for j, (x_ref, y_ref, ts_ref) in enumerate(ref):
        np.testing.assert_array_equal(ours["X"][j], x_ref)        # exact copy
        assert int(ours["y"][j]) == y_ref                          # value equal
        assert ours["target_timestamps"][j] == ts_ref


def _kw(n=6, f=3):
    fx = _clean_single_day(n=n, f=f)
    return dict(
        features=fx["features"], labels=fx["labels"],
        timestamps=fx["timestamps"], partition=fx["partition"],
        feature_valid_mask=fx["feature_valid_mask"],
        target_valid_mask=fx["target_valid_mask"], window_size=4,
    )


def _call(kw):
    return windows.build_windows_single_ticker(
        kw["features"], kw["labels"], kw["timestamps"],
        partition=kw["partition"],
        feature_valid_mask=kw["feature_valid_mask"],
        target_valid_mask=kw["target_valid_mask"],
        window_size=kw["window_size"],
    )


@pytest.mark.parametrize(
    "arg",
    ["features", "labels", "timestamps", "partition",
     "feature_valid_mask", "target_valid_mask"],
)
def test_rejects_non_ndarray_arg(arg):
    kw = _kw()
    kw[arg] = kw[arg].tolist()
    with pytest.raises(TypeError):
        _call(kw)


def test_rejects_features_not_2d():
    kw = _kw()
    kw["features"] = kw["features"][:, 0]  # now 1-D
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_zero_feature_columns():
    kw = _kw()
    kw["features"] = np.empty((6, 0), dtype=np.float64)
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_aux_not_1d():
    kw = _kw()
    kw["partition"] = kw["partition"].reshape(-1, 1)
    with pytest.raises(ValueError):
        _call(kw)


@pytest.mark.parametrize("arg", ["labels", "timestamps", "partition"])
def test_rejects_length_mismatch(arg):
    kw = _kw()
    kw[arg] = kw[arg][:-1]
    with pytest.raises(ValueError):
        _call(kw)


@pytest.mark.parametrize(
    "arg,bad",
    [
        ("features", np.zeros((6, 3), dtype=np.float32)),
        ("labels", np.zeros(6, dtype=np.int16)),
        ("timestamps", np.arange(6).astype("datetime64[s]")),
        ("partition", np.zeros(6, dtype=np.int16)),
        ("feature_valid_mask", np.zeros(6, dtype=np.int8)),
        ("target_valid_mask", np.ones(6, dtype=np.int8)),
    ],
)
def test_rejects_bad_dtype(arg, bad):
    kw = _kw()
    kw[arg] = bad
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_partition_out_of_range():
    kw = _kw()
    kw["partition"] = kw["partition"].copy()
    kw["partition"][0] = np.int8(2)
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_window_size_bool():
    kw = _kw()
    kw["window_size"] = True  # bool subclass of int -> must be rejected
    with pytest.raises(TypeError):
        _call(kw)


def test_rejects_window_size_non_int():
    kw = _kw()
    kw["window_size"] = 4.0
    with pytest.raises(TypeError):
        _call(kw)


@pytest.mark.parametrize("ws", [0, -1])
def test_rejects_window_size_nonpositive(ws):
    kw = _kw()
    kw["window_size"] = ws
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_unsorted_timestamps():
    kw = _kw()
    ts = kw["timestamps"].copy()
    ts[2], ts[3] = ts[3], ts[2]  # break monotonicity
    kw["timestamps"] = ts
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_label_contract_violation():
    kw = _kw()
    kw["labels"] = kw["labels"].copy()
    kw["labels"][2] = np.int8(-1)  # row 2 is target_valid=True
    with pytest.raises(ValueError):
        _call(kw)


def test_rejects_label_out_of_domain():
    # 99 is not in the {0, 1, -1} label domain -> fail loud (an invalid-target
    # row must not be able to smuggle an arbitrary int8 through).
    kw = _kw()
    kw["labels"] = kw["labels"].copy()
    kw["labels"][0] = np.int8(99)
    with pytest.raises(ValueError, match="labels must be in"):
        _call(kw)


def test_feature_invalid_in_window_drops_it():
    kw = _kw(n=6)
    kw["feature_valid_mask"] = kw["feature_valid_mask"].copy()
    kw["feature_valid_mask"][1] = False  # kills windows covering row 1
    res = _call(kw)
    # ws=4: windows end at 3,4,5; row 1 is in windows ending 3 and 4 only.
    assert res["target_row_positions"].tolist() == [5]


def test_target_invalid_drops_it():
    kw = _kw(n=6)
    kw["target_valid_mask"] = kw["target_valid_mask"].copy()
    kw["target_valid_mask"][4] = False
    res = _call(kw)
    assert res["target_row_positions"].tolist() == [3, 5]


def test_target_valid_but_target_row_feature_invalid_drops_it():
    kw = _kw(n=6)
    kw["feature_valid_mask"] = kw["feature_valid_mask"].copy()
    kw["feature_valid_mask"][5] = False   # target row of the last window
    # row 5 only participates as the target of the window ending at 5.
    res = _call(kw)
    assert 5 not in res["target_row_positions"].tolist()


def test_cross_day_window_dropped():
    # 4 bars on day A, 4 on day B; ws=4. Only the all-A and all-B windows
    # survive; any straddling window is dropped.
    ts = np.concatenate([_bars("2014-03-03", 4), _bars("2014-03-04", 4)])
    n = 8
    kw = dict(
        features=np.random.default_rng(1).standard_normal((n, 2)),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=ts,
        partition=np.zeros(n, dtype=np.int8),
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=4,
    )
    res = _call(kw)
    # end_pos 3 (all day A) and 7 (all day B) only.
    assert res["target_row_positions"].tolist() == [3, 7]


def test_window_size_one_degenerate():
    kw = _kw(n=4)
    kw["window_size"] = 1
    res = _call(kw)
    assert res["X"].shape == (4, 1, 3)
    assert res["target_row_positions"].tolist() == [0, 1, 2, 3]


def test_short_day_contributes_nothing_others_emit():
    # day A has 2 bars (< ws=3 -> 0 windows), day B has 4 bars -> 2 windows.
    ts = np.concatenate([_bars("2014-03-03", 2), _bars("2014-03-04", 4)])
    n = 6
    kw = dict(
        features=np.random.default_rng(2).standard_normal((n, 2)),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=ts,
        partition=np.zeros(n, dtype=np.int8),
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=3,
    )
    res = _call(kw)
    assert res["target_row_positions"].tolist() == [4, 5]


def test_target_partition_value_and_dtype():
    kw = _kw(n=6)
    kw["partition"] = np.ones(6, dtype=np.int8)  # all validation
    res = _call(kw)
    assert res["target_partition"].dtype == np.int8
    assert set(res["target_partition"].tolist()) == {1}


def test_boundary_window_ends_first_validation_bar():
    # 8 bars, all same day; first 4 train, last 4 validation (date-aligned in
    # real data; here we keep them same-day to exercise the value, not the
    # uniformity guard, so use a SECOND day for the validation bars).
    ts = np.concatenate([_bars("2014-03-03", 4), _bars("2014-03-04", 4)])
    n = 8
    partition = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    kw = dict(
        features=np.random.default_rng(3).standard_normal((n, 2)),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=ts, partition=partition,
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=4,
    )
    res = _call(kw)
    # window ending at 7 is the first all-validation window.
    by_pos = dict(zip(res["target_row_positions"].tolist(),
                      res["target_partition"].tolist()))
    assert by_pos[7] == 1
    assert by_pos[3] == 0


def test_partition_uniformity_violation_fails_loud():
    # Same-day window with a partition flip INSIDE the day -> contract breach.
    n = 6
    kw = dict(
        features=np.random.default_rng(4).standard_normal((n, 2)),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=_bars("2014-03-03", n),       # all one day
        partition=np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),  # flips mid-day
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=4,
    )
    with pytest.raises(ValueError, match="non-uniform partition"):
        _call(kw)


_CORE_KEYS = {
    "X", "y", "target_partition", "target_timestamps", "target_row_positions",
}


def _empty_inputs(n, f=3):
    return dict(
        features=np.empty((n, f), dtype=np.float64),
        labels=np.empty(n, dtype=np.int8),
        timestamps=np.empty(n, dtype="datetime64[ns]"),
        partition=np.empty(n, dtype=np.int8),
        feature_valid_mask=np.empty(n, dtype=np.bool_),
        target_valid_mask=np.empty(n, dtype=np.bool_),
        window_size=4,
    )


def test_empty_n_zero():
    res = _call(_empty_inputs(0))
    assert res["X"].shape == (0, 4, 3)
    assert res["y"].shape == (0,)


def test_n_less_than_window_size_empty():
    kw = _kw(n=3)
    kw["window_size"] = 5
    res = _call(kw)
    assert res["X"].shape == (0, 5, 3)
    assert res["target_row_positions"].shape == (0,)


def test_all_windows_filtered_empty():
    kw = _kw(n=6)
    kw["target_valid_mask"] = np.zeros(6, dtype=np.bool_)  # no valid targets
    res = _call(kw)
    assert res["X"].shape == (0, 4, 3)


def test_empty_schema_exact_dtypes():
    res = _call(_empty_inputs(0))
    assert res["X"].dtype == np.float64
    assert res["y"].dtype == np.int8
    assert res["target_partition"].dtype == np.int8
    assert res["target_timestamps"].dtype == np.dtype("datetime64[ns]")
    assert res["target_row_positions"].dtype == np.int64


def test_output_key_set():
    res = _call(_kw())
    assert set(res.keys()) == _CORE_KEYS


def test_target_row_positions_dtype_int64():
    res = _call(_kw())
    assert res["target_row_positions"].dtype == np.int64


def test_y_dtype_int8():
    res = _call(_kw())
    assert res["y"].dtype == np.int8


def test_target_row_positions_provenance_single():
    kw = _kw(n=6)
    res = _call(kw)
    pos = res["target_row_positions"]
    np.testing.assert_array_equal(kw["timestamps"][pos], res["target_timestamps"])
    np.testing.assert_array_equal(kw["labels"][pos], res["y"])


def _pooled_two_tickers():
    # ticker 1 on day A (6 bars), ticker 0 on day B (5 bars); pooled sorted by
    # (ticker, timestamp) is NOT required, but each ticker's rows are in time
    # order within the array.
    ts0 = _bars("2014-03-04", 5)
    ts1 = _bars("2014-03-03", 6)
    timestamps = np.concatenate([ts1, ts0])
    ticker_ids = np.array([1] * 6 + [0] * 5)
    n = 11
    rng = np.random.default_rng(11)
    return dict(
        features=rng.standard_normal((n, 2)),
        labels=np.tile(np.array([0, 1], np.int8), n)[:n],
        timestamps=timestamps,
        ticker_ids=ticker_ids,
        partition=np.zeros(n, dtype=np.int8),
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=4,
    )


def _call_pooled(kw):
    return windows.build_windows(
        kw["features"], kw["labels"], kw["timestamps"], kw["ticker_ids"],
        partition=kw["partition"],
        feature_valid_mask=kw["feature_valid_mask"],
        target_valid_mask=kw["target_valid_mask"],
        window_size=kw["window_size"],
    )


def test_build_windows_pooled_basic_two_tickers():
    kw = _pooled_two_tickers()
    res = _call_pooled(kw)
    # ticker 1: 6 bars ws=4 -> 3 windows; ticker 0: 5 bars -> 2 windows.
    assert res["y"].shape[0] == 5
    assert res["X"].shape == (5, 4, 2)
    assert set(res.keys()) == _CORE_KEYS | {"target_ticker_ids"}


def test_build_windows_pooled_matches_single_concatenation():
    kw = _pooled_two_tickers()
    pooled = _call_pooled(kw)
    # Re-derive ticker 0's block independently and compare its slice.
    mask0 = kw["ticker_ids"] == 0
    g0 = np.flatnonzero(mask0)
    single0 = windows.build_windows_single_ticker(
        kw["features"][g0], kw["labels"][g0], kw["timestamps"][g0],
        partition=kw["partition"][g0],
        feature_valid_mask=kw["feature_valid_mask"][g0],
        target_valid_mask=kw["target_valid_mask"][g0],
        window_size=4,
    )
    sel = pooled["target_ticker_ids"] == 0
    np.testing.assert_array_equal(pooled["X"][sel], single0["X"])
    np.testing.assert_array_equal(pooled["y"][sel], single0["y"])


def test_pooled_no_cross_ticker_window():
    # Interleave two tickers on the SAME day; a naive global slide would build
    # cross-ticker windows. The pooled builder must not.
    ts = _bars("2014-03-03", 8)
    ticker_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    n = 8
    kw = dict(
        features=np.random.default_rng(20).standard_normal((n, 2)),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=ts, ticker_ids=ticker_ids,
        partition=np.zeros(n, dtype=np.int8),
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=3,
    )
    # within a ticker the 4 bars are NOT 5-min contiguous (they are 10-min
    # apart), but they ARE the same calendar day, so windows form per ticker.
    res = _call_pooled(kw)
    # 4 bars per ticker, ws=3 -> 2 windows each -> 4 total.
    assert res["y"].shape[0] == 4
    # every window's rows come from one ticker: provenance positions per window
    # must all share a ticker_id.
    for w_i in range(res["y"].shape[0]):
        end = res["target_row_positions"][w_i]
        tid = ticker_ids[end]
        assert res["target_ticker_ids"][w_i] == tid


def test_pooled_block_order_and_within_block_ascending():
    kw = _pooled_two_tickers()
    res = _call_pooled(kw)
    # np.unique order -> ticker 0 block first, then ticker 1.
    tids = res["target_ticker_ids"].tolist()
    assert tids == sorted(tids)
    # within each block, target timestamps ascend.
    for t in (0, 1):
        sel = res["target_ticker_ids"] == t
        ts_block = res["target_timestamps"][sel]
        assert np.all(ts_block[:-1] <= ts_block[1:])


def test_pooled_target_row_positions_provenance():
    kw = _pooled_two_tickers()
    res = _call_pooled(kw)
    pos = res["target_row_positions"]
    np.testing.assert_array_equal(kw["timestamps"][pos], res["target_timestamps"])
    np.testing.assert_array_equal(kw["ticker_ids"][pos], res["target_ticker_ids"])


@pytest.mark.parametrize(
    "ticker_ids",
    [
        np.array([1] * 6 + [0] * 5),                 # numeric
        np.array(["B"] * 6 + ["A"] * 5),             # string
    ],
)
def test_pooled_target_ticker_ids_dtype_passthrough(ticker_ids):
    kw = _pooled_two_tickers()
    kw["ticker_ids"] = ticker_ids
    res = _call_pooled(kw)
    assert res["target_ticker_ids"].dtype == ticker_ids.dtype


def test_pooled_ticker_context_on_core_error():
    # Make ONE ticker's rows non-nondecreasing -> core raises -> wrapper wraps.
    kw = _pooled_two_tickers()
    ts = kw["timestamps"].copy()
    ts[0], ts[1] = ts[1], ts[0]  # break order within ticker 1's block
    kw["timestamps"] = ts
    with pytest.raises(ValueError, match="ticker"):
        _call_pooled(kw)


def test_pooled_empty_n_zero_includes_ticker_ids_dtype():
    kw = _empty_inputs(0)
    kw["ticker_ids"] = np.empty(0, dtype="<U4")
    res = windows.build_windows(
        kw["features"], kw["labels"], kw["timestamps"], kw["ticker_ids"],
        partition=kw["partition"],
        feature_valid_mask=kw["feature_valid_mask"],
        target_valid_mask=kw["target_valid_mask"],
        window_size=kw["window_size"],
    )
    assert res["target_ticker_ids"].shape == (0,)
    assert res["target_ticker_ids"].dtype == np.dtype("<U4")


def test_pooled_all_blocks_empty():
    # Two tickers, each with < window_size rows -> all blocks empty.
    ts = np.concatenate([_bars("2014-03-03", 2), _bars("2014-03-04", 2)])
    n = 4
    kw = dict(
        features=np.zeros((n, 2), dtype=np.float64),
        labels=np.zeros(n, dtype=np.int8),
        timestamps=ts,
        ticker_ids=np.array([0, 0, 1, 1]),
        partition=np.zeros(n, dtype=np.int8),
        feature_valid_mask=np.ones(n, dtype=np.bool_),
        target_valid_mask=np.ones(n, dtype=np.bool_),
        window_size=3,
    )
    res = _call_pooled(kw)
    assert res["X"].shape == (0, 3, 2)
    assert res["target_ticker_ids"].shape == (0,)


def test_pooled_rejects_mixed_object_ticker_ids():
    kw = _pooled_two_tickers()
    kw["ticker_ids"] = np.array([1, "A", 1, "A", 1, "A", 1, 0, 0, 0, 0],
                                dtype=object)
    with pytest.raises(ValueError):
        _call_pooled(kw)


def test_pooled_rejects_ticker_ids_length_mismatch():
    kw = _pooled_two_tickers()
    kw["ticker_ids"] = kw["ticker_ids"][:-1]
    with pytest.raises(ValueError):
        _call_pooled(kw)


def test_pooled_rejects_nan_ticker_ids():
    # A float NaN ticker id is not self-comparable; grouping by equality would
    # silently drop the NaN-tagged rows, so the builder must fail loud instead.
    kw = _pooled_two_tickers()
    kw["ticker_ids"] = np.array(
        [1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError):
        _call_pooled(kw)
