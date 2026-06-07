"""Behavioral tests for ``intraday_research.data.splits`` (N08 #5C-4).

Synthetic-data tests only. No raw bar I/O, no fixture files committed
to the repo, no official validation, no holdout. Verifies the section 4
contract documented in
``docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import assign_calendar_split
from intraday_research.data.raw_bars import VAL_END as RAW_BARS_VAL_END
from intraday_research.data.splits import (
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    VALIDATION_START,
    apply_chronological_split,
    apply_stage0_chronological_split,
)


_PARTITION_NAME_TO_CODE = {
    "train": int(PARTITION_TRAIN),
    "validation": int(PARTITION_VALIDATION),
}


def _baseline_v1_splits_dict(
    validation_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> dict:
    """baseline_v1.assign_calendar_split expects a 3-key dict. The
    `closed_holdout_boundary_only` slot is a placeholder pointing far
    into the future — our val_end fail-loud check rejects any row that
    would land in it, so the slot's range never actually matters."""
    return {
        "train": (pd.Timestamp("1998-01-02"), validation_start),
        "validation": (validation_start, val_end),
        "closed_holdout_boundary_only": (val_end, pd.Timestamp("2099-01-01")),
    }


def _synthetic_timestamps(
    start: str = "2013-09-15 09:30:00",
    periods: int = 120,
    freq: str = "5min",
) -> np.ndarray:
    return pd.date_range(start, periods=periods, freq=freq).to_numpy()


def test_partition_matches_baseline_v1_assign_calendar_split_per_row():
    """Anti-drift gate: every row's int8 partition equals the
    name→code mapping of baseline_v1.assign_calendar_split."""
    # Span the train->validation boundary so the anti-drift gate actually
    # exercises the split transition (1h of bars before VALIDATION_START
    # crossing into validation), not just an all-train block.
    timestamps = _synthetic_timestamps(
        start=str(VALIDATION_START - pd.Timedelta(hours=1)), periods=120,
    )
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    # The fixture MUST straddle the boundary for this gate to mean anything.
    assert (partition == int(PARTITION_TRAIN)).any()
    assert (partition == int(PARTITION_VALIDATION)).any()
    splits_dict = _baseline_v1_splits_dict(
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
    )
    for i, ts in enumerate(timestamps):
        expected_name = assign_calendar_split(ts, splits_dict)
        assert expected_name in _PARTITION_NAME_TO_CODE, (
            f"baseline_v1 returned partition {expected_name!r} at row {i}, "
            "but #5C-4's two-partition encoding only supports train and "
            "validation (val_end fail-loud should have rejected anything else)"
        )
        expected_code = _PARTITION_NAME_TO_CODE[expected_name]
        assert int(partition[i]) == expected_code, (
            f"partition mismatch at row {i} (ts={ts}): "
            f"baseline_v1={expected_name!r} (code {expected_code}), "
            f"#5C-4={int(partition[i])}"
        )


def test_timestamp_exactly_at_validation_start_is_validation():
    """The ``<`` in the partition computation means timestamp ==
    validation_start lands in VALIDATION, not TRAIN."""
    timestamps = pd.date_range(
        VALIDATION_START, periods=3, freq="5min"
    ).to_numpy()
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=1,
    )
    assert int(partition[0]) == int(PARTITION_VALIDATION)


def test_timestamp_one_bar_before_validation_start_is_train():
    one_bar_before = VALIDATION_START - pd.Timedelta(minutes=5)
    timestamps = pd.date_range(one_bar_before, periods=2, freq="5min").to_numpy()
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=1,
    )
    assert int(partition[0]) == int(PARTITION_TRAIN)
    assert int(partition[1]) == int(PARTITION_VALIDATION)


def test_timestamp_far_before_validation_start_is_train():
    timestamps = pd.date_range(
        "2000-01-03 09:30:00", periods=5, freq="5min",
    ).to_numpy()
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=1,
    )
    assert (partition == int(PARTITION_TRAIN)).all()


def test_timestamp_near_but_before_val_end_is_validation():
    """One 5-min bar before val_end: still VALIDATION (not raised)."""
    one_bar_before_val_end = RAW_BARS_VAL_END - pd.Timedelta(minutes=5)
    timestamps = np.array(
        [one_bar_before_val_end.to_datetime64()], dtype="datetime64[ns]",
    )
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=1,
    )
    assert int(partition[0]) == int(PARTITION_VALIDATION)


def test_all_train_array_last_horizon_rows_invalid():
    """All-TRAIN array, horizon_k=3: last 3 rows have no horizon successor."""
    timestamps = pd.date_range(
        "2010-01-04 09:30:00", periods=20, freq="5min",
    ).to_numpy()
    _, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    assert valid_mask[:17].all()
    assert not valid_mask[17:].any()


def test_all_validation_array_last_horizon_rows_invalid():
    timestamps = pd.date_range(
        "2014-06-02 09:30:00", periods=20, freq="5min",
    ).to_numpy()
    _, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    assert valid_mask[:17].all()
    assert not valid_mask[17:].any()


def test_train_end_crosses_into_validation_marks_last_train_rows_invalid():
    """Mixed array straddling validation_start: the last horizon_k TRAIN
    rows have partition[t]=TRAIN but partition[t+horizon_k]=VALIDATION,
    so valid_mask[t]=False even though they have a horizon successor."""
    # 10 bars ending at validation_start - 5min (all TRAIN), then 10
    # bars starting at validation_start (all VALIDATION).
    train_bars = pd.date_range(
        VALIDATION_START - pd.Timedelta(minutes=5 * 10),
        periods=10, freq="5min",
    )
    val_bars = pd.date_range(VALIDATION_START, periods=10, freq="5min")
    timestamps = np.concatenate(
        [train_bars.to_numpy(), val_bars.to_numpy()]
    )
    partition, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    # Indices 7, 8, 9 are the last 3 TRAIN rows; their horizon
    # successors (10, 11, 12) are VALIDATION → cross-split → False.
    assert int(partition[7]) == int(PARTITION_TRAIN)
    assert int(partition[8]) == int(PARTITION_TRAIN)
    assert int(partition[9]) == int(PARTITION_TRAIN)
    assert int(partition[10]) == int(PARTITION_VALIDATION)
    assert not valid_mask[7]
    assert not valid_mask[8]
    assert not valid_mask[9]
    # Earlier TRAIN rows (0-6) still have TRAIN successors → True.
    assert valid_mask[:7].all()


def test_validation_end_with_insufficient_horizon_rows_marks_them_invalid():
    """If the validation slice ends with fewer than horizon_k bars
    remaining, those rows lose validity (no horizon successor at all)."""
    timestamps = pd.date_range(
        "2016-12-01 09:30:00", periods=10, freq="5min",
    ).to_numpy()
    _, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=5,
    )
    assert valid_mask[:5].all()
    assert not valid_mask[5:].any()


def test_n_less_than_or_equal_to_horizon_k_all_invalid():
    """Every row's horizon successor lies past the array end."""
    timestamps = pd.date_range(
        "2014-06-02 09:30:00", periods=3, freq="5min",
    ).to_numpy()
    _, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    assert not valid_mask.any()


@pytest.mark.parametrize("horizon_k", [3, 9, 24])
def test_stage0_alias_matches_generic_at_each_horizon(horizon_k):
    """The Stage 0 alias is a parameter-pinned thin wrapper. Cover the
    three frozen Stage 0 label-config horizons (h03, h09, h24)."""
    timestamps = pd.date_range(
        "2013-09-15 09:30:00", periods=100, freq="5min",
    ).to_numpy()
    via_alias = apply_stage0_chronological_split(
        timestamps, horizon_k=horizon_k,
    )
    via_generic = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=horizon_k,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_timestamp_exactly_at_val_end_raises():
    timestamps = np.array(
        [RAW_BARS_VAL_END.to_datetime64()], dtype="datetime64[ns]",
    )
    with pytest.raises(ValueError) as excinfo:
        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=1,
        )
    msg = str(excinfo.value)
    assert "holdout closure violated" in msg
    assert "first contaminated timestamp=2017-01-25" in msg
    assert "rows=1/1" in msg


def test_multiple_post_val_end_rows_carry_full_count():
    timestamps = pd.date_range(
        RAW_BARS_VAL_END, periods=5, freq="5min",
    ).to_numpy()
    with pytest.raises(ValueError) as excinfo:
        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=1,
        )
    msg = str(excinfo.value)
    assert "rows=5/5" in msg


def test_all_pre_val_end_passes_without_raising():
    timestamps = pd.date_range(
        "2016-12-01 09:30:00", periods=10, freq="5min",
    ).to_numpy()
    partition, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=1,
    )
    assert partition.shape == (10,)
    assert valid_mask.shape == (10,)


def test_custom_val_end_classifies_2018_as_validation():
    """A 2018 timestamp is past the default val_end (would normally
    fail-loud), but with a custom val_end of 2020-01-02 it falls inside
    [validation_start, val_end) and is classified as VALIDATION."""
    timestamps = pd.date_range(
        "2018-06-01 09:30:00", periods=5, freq="5min",
    ).to_numpy()
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=pd.Timestamp("2020-01-02"),
        horizon_k=1,
    )
    assert (partition == int(PARTITION_VALIDATION)).all()


def _valid_timestamps_n10() -> np.ndarray:
    return pd.date_range(
        "2014-06-02 09:30:00", periods=10, freq="5min",
    ).to_numpy()


def test_validation_start_non_timestamp_raises_type_error():
    with pytest.raises(TypeError, match="validation_start must be pd.Timestamp"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start="2013-09-16",
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_validation_start_tz_aware_raises():
    with pytest.raises(ValueError, match="validation_start must be timezone-naive"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=pd.Timestamp("2013-09-16", tz="UTC"),
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_val_end_non_timestamp_raises_type_error():
    with pytest.raises(TypeError, match="val_end must be pd.Timestamp"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=VALIDATION_START,
            val_end="2017-01-25",
            horizon_k=3,
        )


def test_val_end_tz_aware_raises():
    with pytest.raises(ValueError, match="val_end must be timezone-naive"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=VALIDATION_START,
            val_end=pd.Timestamp("2017-01-25", tz="UTC"),
            horizon_k=3,
        )


def test_validation_start_not_less_than_val_end_raises():
    with pytest.raises(ValueError, match="validation_start"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=pd.Timestamp("2018-01-01"),
            val_end=pd.Timestamp("2017-01-25"),
            horizon_k=3,
        )


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_horizon_k_non_positive_raises(bad):
    with pytest.raises(ValueError, match="horizon_k must be a positive int"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=bad,
        )


def test_horizon_k_bool_true_raises_even_though_bool_is_int_subclass():
    with pytest.raises(ValueError, match="horizon_k must be a positive int"):
        apply_chronological_split(
            _valid_timestamps_n10(),
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=True,
        )


def test_non_1d_timestamps_raises():
    timestamps_2d = _valid_timestamps_n10().reshape(2, 5)
    with pytest.raises(ValueError, match="must be a 1-D ndarray"):
        apply_chronological_split(
            timestamps_2d,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_int_dtype_timestamps_raises():
    timestamps_int = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="must be datetime64"):
        apply_chronological_split(
            timestamps_int,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_non_ns_datetime64_raises():
    # datetime64[s] is datetime64 but not the required [ns] precision; the
    # spec pins datetime64[ns], so a coarser precision must be rejected.
    timestamps_sec = _valid_timestamps_n10().astype("datetime64[s]")
    with pytest.raises(ValueError, match="datetime64"):
        apply_chronological_split(
            timestamps_sec,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_tz_aware_timestamps_raises():
    # A tz-aware DatetimeIndex has no native numpy datetime64 form, so
    # .to_numpy() yields an OBJECT array of Timestamps. numpy cannot carry a
    # tz-aware datetime64, so the datetime64-dtype guard is the correct
    # fail-loud channel for tz-aware input reaching this numpy-faced API.
    timestamps = (
        pd.date_range("2014-06-02 09:30:00", periods=10, freq="5min")
        .tz_localize("UTC")
        .to_numpy()
    )
    assert timestamps.dtype == object
    with pytest.raises(ValueError, match="must be datetime64"):
        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_nat_in_timestamps_raises():
    timestamps = _valid_timestamps_n10().copy()
    timestamps[3] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="contains NaT"):
        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_unsorted_timestamps_raises():
    timestamps = _valid_timestamps_n10()[::-1].copy()
    with pytest.raises(ValueError, match="must be sorted ascending"):
        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=RAW_BARS_VAL_END,
            horizon_k=3,
        )


def test_empty_timestamps_returns_empty_arrays_does_not_raise():
    timestamps = np.array([], dtype="datetime64[ns]")
    partition, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    assert partition.shape == (0,)
    assert valid_mask.shape == (0,)
    assert partition.dtype == np.int8
    assert valid_mask.dtype == np.bool_


def test_n_equal_to_horizon_k_returns_all_false_valid_mask():
    timestamps = pd.date_range(
        "2014-06-02 09:30:00", periods=3, freq="5min",
    ).to_numpy()
    _, valid_mask = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
    assert valid_mask.shape == (3,)
    assert not valid_mask.any()


def test_partition_codes_are_locked_to_int8_zero_and_one():
    assert PARTITION_TRAIN == np.int8(0)
    assert PARTITION_VALIDATION == np.int8(1)
    assert PARTITION_TRAIN.dtype == np.int8
    assert PARTITION_VALIDATION.dtype == np.int8


def test_validation_start_constant_is_locked():
    assert VALIDATION_START == pd.Timestamp("2013-09-16")
    assert VALIDATION_START.tzinfo is None


def test_val_end_imported_from_raw_bars_single_source_of_truth():
    """splits.py must NOT redefine VAL_END locally; it must import the
    raw_bars module-level VAL_END so the project has exactly one canonical
    value for the holdout boundary."""
    from intraday_research.data import splits as splits_module
    # The module imports VAL_END from raw_bars. Re-import explicitly to
    # confirm the value matches the raw_bars source of truth.
    assert splits_module.VAL_END == RAW_BARS_VAL_END
    assert RAW_BARS_VAL_END == pd.Timestamp("2017-01-25")
