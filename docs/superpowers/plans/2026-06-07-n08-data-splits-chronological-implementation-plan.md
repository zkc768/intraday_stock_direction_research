# N08 #5C-4 — `data/splits.py` Chronological Split Markers Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.
> Execute tasks in order; each task is a self-contained RED→GREEN→VERIFY
> cycle. Do NOT commit until Task 8 — every intermediate verification is
> read-only.
>
> **Shell assumption:** All commands assume **Git Bash on Windows** (the
> project's standard shell). The Task 8 heredoc for `git commit -m`
> requires Git Bash; from PowerShell, write the message to a file and
> use `git commit -F <file>`, or invoke the PowerShell sibling
> `scripts/check_n08_resume_gate.ps1` for the gate. All other commands
> use the explicit project-Python path
> (`E:/codex_workspace/_envs/py311_shared/python.exe`) and avoid shell
> env-var shorthands (no `$PYTHON`, no `head -1`).

**Goal:** Implement `src/intraday_research/data/splits.py` as a
pure-numpy `apply_chronological_split` function returning per-row
partition codes (`PARTITION_TRAIN=0` / `PARTITION_VALIDATION=1`) plus a
cross-split `valid_mask` that encodes "label horizon does not cross a
split boundary". One generic + one Stage 0 alias. Single commit,
~33 cross-checked tests.

**Architecture:** Three module-level constants (`PARTITION_TRAIN`,
`PARTITION_VALIDATION`, `VALIDATION_START`; `VAL_END` is imported from
`raw_bars.py` to keep a single source of truth). The implementation is
pure numpy — no wrap of `baseline_v1.add_split_and_invalidate_boundaries`
because that function couples DataFrame plumbing and
`future_cumulative_return` for a much wider responsibility. Anti-drift
is tested per-row against `baseline_v1.assign_calendar_split`.

**Tech Stack:** Python 3.11 / numpy / pandas / pytest. Project Python
`E:/codex_workspace/_envs/py311_shared/python.exe`.

**Reference commits:**
- #5C-2 `ecfbc95` (features.py) — same wrap-baseline_v1 philosophy,
  same `isinstance(dtype, pd.DatetimeTZDtype)` idiom.
- #5C-3 `e540e68` (raw_bars.py) — same val_end fail-loud holdout
  closure discipline; defines `VAL_END = pd.Timestamp("2017-01-25")`.
- #5C-1 `8ce2829` (labels.py) — same pure-1D-numpy interface style.

**Spec:** `docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md`
(committed in `e9ffe36`).

---

## Files

| Path | Action | Notes |
|---|---|---|
| `src/intraday_research/data/__init__.py` | modify | add `splits` to submodule docstring list |
| `src/intraday_research/data/splits.py` | create | 3 constants + 1 generic + 1 alias, ~85 lines |
| `tests/data/test_splits.py` | create | ~33 tests across 7 spec §4 categories |

No `baseline_v1.py` modifications (anti-drift cross-check only reads).

---

## Task 1: Scaffold + first cross-check test (RED→GREEN)

**Files:**
- Create: `src/intraday_research/data/splits.py`
- Create: `tests/data/test_splits.py`

- [ ] **Step 1.1: Create the stub `splits.py`**

The stub pre-defines all module-level constants and both function
signatures so the test file in Step 1.2 imports cleanly. RED comes
from `NotImplementedError` at call time.

Write `src/intraday_research/data/splits.py`:

```python
"""Chronological train/validation split markers for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_research.data.raw_bars import VAL_END


# Project-frozen Stage 0 train→validation boundary.
# (CONFIG_SCREENING_FREEZE_2026-06-04.md: TRAIN ends 2013-09-16,
# VALIDATION starts 2013-09-16, VALIDATION ends 2017-01-25 = VAL_END.)
VALIDATION_START: pd.Timestamp = pd.Timestamp("2013-09-16")

# Two-partition int8 codes. PARTITION_TRAIN and PARTITION_VALIDATION
# are exported so downstream filters (e.g. #5C-5 window builder) can
# slice the pooled DataFrame by partition without re-deriving the
# numeric values.
PARTITION_TRAIN: np.int8 = np.int8(0)
PARTITION_VALIDATION: np.int8 = np.int8(1)


def apply_chronological_split(
    timestamps: np.ndarray,
    *,
    validation_start: pd.Timestamp,
    val_end: pd.Timestamp,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """See docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md.

    Implementation lands in Task 1 step 1.4.
    """
    raise NotImplementedError(
        "apply_chronological_split — Task 1 step 1.4"
    )


def apply_stage0_chronological_split(
    timestamps: np.ndarray,
    *,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project-frozen Stage 0 alias. Body lands in Task 1 step 1.4."""
    raise NotImplementedError(
        "apply_stage0_chronological_split — Task 1 step 1.4"
    )
```

- [ ] **Step 1.2: Write the cross-check test (anti-drift gate)**

Write `tests/data/test_splits.py`:

```python
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
    timestamps = _synthetic_timestamps(
        start="2013-09-15 09:30:00", periods=120,
    )
    partition, _ = apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=RAW_BARS_VAL_END,
        horizon_k=3,
    )
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
```

- [ ] **Step 1.3: Run the test and verify it FAILS**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py::test_partition_matches_baseline_v1_assign_calendar_split_per_row -v
```

Expected: `FAILED` with
`NotImplementedError("apply_chronological_split — Task 1 step 1.4")`.
Imports succeed because the stub pre-defines all symbols.

- [ ] **Step 1.4: Implement the full body**

Replace `src/intraday_research/data/splits.py` with the full
implementation:

```python
"""Chronological train/validation split markers for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_research.data.raw_bars import VAL_END


# Project-frozen Stage 0 train→validation boundary.
# (CONFIG_SCREENING_FREEZE_2026-06-04.md: TRAIN ends 2013-09-16,
# VALIDATION starts 2013-09-16, VALIDATION ends 2017-01-25 = VAL_END.)
VALIDATION_START: pd.Timestamp = pd.Timestamp("2013-09-16")

# Two-partition int8 codes. PARTITION_TRAIN and PARTITION_VALIDATION
# are exported so downstream filters (e.g. #5C-5 window builder) can
# slice the pooled DataFrame by partition without re-deriving the
# numeric values.
PARTITION_TRAIN: np.int8 = np.int8(0)
PARTITION_VALIDATION: np.int8 = np.int8(1)


def apply_chronological_split(
    timestamps: np.ndarray,
    *,
    validation_start: pd.Timestamp,
    val_end: pd.Timestamp,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row partition codes + cross-split valid_mask.

    Args:
        timestamps: 1-D ``datetime64[ns]`` array, timezone-naive,
            sorted ascending.
        validation_start: train→validation boundary (Timestamp,
            timezone-naive). Rows where ``timestamps < validation_start``
            are classified as ``PARTITION_TRAIN``; rows in
            ``[validation_start, val_end)`` are
            ``PARTITION_VALIDATION``.
        val_end: validation end / holdout boundary (Timestamp,
            timezone-naive). Any ``timestamp >= val_end`` raises a
            ValueError (holdout closure fail-loud).
        horizon_k: positive integer label horizon. ``valid_mask[t]`` is
            True iff row ``t + horizon_k`` exists AND
            ``partition[t] == partition[t + horizon_k]``.

    Returns:
        ``(partition, valid_mask)`` where ``partition`` is ``int8``
        shape ``(n,)`` with values in
        ``{PARTITION_TRAIN, PARTITION_VALIDATION}`` and ``valid_mask``
        is ``bool_`` shape ``(n,)``.
    """
    # ---- Step 1: validate validation_start ----
    if not isinstance(validation_start, pd.Timestamp):
        raise TypeError(
            "validation_start must be pd.Timestamp; "
            f"got {type(validation_start).__name__}"
        )
    if validation_start.tzinfo is not None:
        raise ValueError(
            "validation_start must be timezone-naive; "
            f"got tz={validation_start.tzinfo}"
        )

    # ---- Step 2: validate val_end ----
    if not isinstance(val_end, pd.Timestamp):
        raise TypeError(
            f"val_end must be pd.Timestamp; got {type(val_end).__name__}"
        )
    if val_end.tzinfo is not None:
        raise ValueError(
            f"val_end must be timezone-naive; got tz={val_end.tzinfo}"
        )

    # ---- Step 3: validate boundary ordering ----
    if not validation_start < val_end:
        raise ValueError(
            f"validation_start ({validation_start}) must be "
            f"< val_end ({val_end})"
        )

    # ---- Step 4: validate horizon_k ----
    if (
        isinstance(horizon_k, bool)
        or not isinstance(horizon_k, int)
        or horizon_k <= 0
    ):
        raise ValueError(
            f"horizon_k must be a positive int; got {horizon_k!r}"
        )

    # ---- Step 5: validate timestamps ----
    if not isinstance(timestamps, np.ndarray) or timestamps.ndim != 1:
        shape = (
            timestamps.shape if isinstance(timestamps, np.ndarray) else None
        )
        raise ValueError(
            f"timestamps must be a 1-D ndarray; got shape {shape}"
        )
    if not np.issubdtype(timestamps.dtype, np.datetime64):
        raise ValueError(
            f"timestamps dtype must be datetime64; got {timestamps.dtype}"
        )
    # NOTE: do NOT use pd.api.types.is_datetime64tz_dtype(...). It is
    # deprecated and emits a DeprecationWarning; pytest.ini turns
    # Warnings from intraday_research.* into errors. Use the supported
    # isinstance check on pd.DatetimeTZDtype instead (same approach as
    # #5C-2 features.py).
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "timestamps must be timezone-naive; "
            f"got tz={timestamps.dtype.tz}"
        )
    if np.isnat(timestamps).any():
        raise ValueError("timestamps contains NaT")
    if timestamps.size > 1 and (timestamps[1:] < timestamps[:-1]).any():
        raise ValueError("timestamps must be sorted ascending")

    # ---- Step 6: HOLDOUT CLOSURE CHECK (fail-loud) ----
    val_end_np = np.datetime64(val_end.to_datetime64(), "ns")
    contam_mask = timestamps >= val_end_np
    if contam_mask.any():
        n_contam = int(contam_mask.sum())
        n_total = int(timestamps.size)
        first_bad = pd.Timestamp(timestamps[contam_mask][0])
        raise ValueError(
            "holdout closure violated; "
            f"first contaminated timestamp={first_bad}; "
            f"rows={n_contam}/{n_total}"
        )

    # ---- Step 7: n=0 short-circuit ----
    n = int(timestamps.size)
    if n == 0:
        return (
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.bool_),
        )

    # ---- Step 8: compute partition (pure numpy) ----
    validation_start_np = np.datetime64(
        validation_start.to_datetime64(), "ns"
    )
    partition = np.where(
        timestamps < validation_start_np,
        PARTITION_TRAIN,
        PARTITION_VALIDATION,
    ).astype(np.int8)

    # ---- Step 9: compute valid_mask ----
    valid_mask = np.zeros(n, dtype=np.bool_)
    if n > horizon_k:
        same_partition = (
            partition[: n - horizon_k] == partition[horizon_k:]
        )
        valid_mask[: n - horizon_k] = same_partition

    return partition, valid_mask


def apply_stage0_chronological_split(
    timestamps: np.ndarray,
    *,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project-frozen Stage 0 alias.

    Equivalent to::

        apply_chronological_split(
            timestamps,
            validation_start=VALIDATION_START,
            val_end=VAL_END,
            horizon_k=horizon_k,
        )

    where ``VAL_END`` is imported from ``intraday_research.data.raw_bars``
    as the single source of truth.
    """
    return apply_chronological_split(
        timestamps,
        validation_start=VALIDATION_START,
        val_end=VAL_END,
        horizon_k=horizon_k,
    )
```

- [ ] **Step 1.5: Update `data/__init__.py` docstring**

Replace the submodule list in `src/intraday_research/data/__init__.py`:

```python
"""Raw-data ingestion + label/feature/window helpers for the N08 #5C pipeline.

Submodules:
  - ``labels``    no-trade-band binary labels (#5C-1)
  - ``raw_bars``  5-min pre-aggregated per-ticker CSV loader (#5C-3)
  - ``features``  price_volume_time feature builder (#5C-2)
  - ``splits``    chronological train/validation split markers (#5C-4)
  - ``windows``  arrives in sibling commit #5C-5.

Validation-only scope (AGENTS.md section 4.1); no holdout/test data is read
by anything in this subpackage.
"""
```

- [ ] **Step 1.6: Run the test and verify it PASSES**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py::test_partition_matches_baseline_v1_assign_calendar_split_per_row -v
```

Expected: `1 passed`.

---

## Task 2: Partition encoding + boundary precision tests

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 2.1: Append 4 boundary-precision tests**

Append to `tests/data/test_splits.py`:

```python
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
```

- [ ] **Step 2.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `5 passed` (1 + 4 new).

---

## Task 3: `valid_mask` cross-split semantics tests

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 3.1: Append 5 cross-split semantics tests**

Append to `tests/data/test_splits.py`:

```python
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
```

- [ ] **Step 3.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `10 passed` (5 + 5 new).

---

## Task 4: Stage 0 alias equivalence tests

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 4.1: Append 3 alias-equivalence tests**

Append to `tests/data/test_splits.py`:

```python
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
```

- [ ] **Step 4.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `13 passed` (10 + 3 parametrized).

---

## Task 5: Holdout closure fail-loud tests (val_end guard)

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 5.1: Append 4 fail-loud holdout tests**

Append to `tests/data/test_splits.py`:

```python
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
```

- [ ] **Step 5.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `17 passed` (13 + 4 new).

---

## Task 6: Wrapper-layer input guards

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 6.1: Append wrapper-layer guard tests**

Append to `tests/data/test_splits.py`:

```python
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


def test_tz_aware_timestamps_raises():
    timestamps = (
        pd.date_range("2014-06-02 09:30:00", periods=10, freq="5min")
        .tz_localize("UTC")
        .to_numpy()
    )
    with pytest.raises(ValueError, match="must be timezone-naive"):
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
```

- [ ] **Step 6.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `30 passed` (17 + 13 new, including the 3 parametrized
`horizon_k` cases).

---

## Task 7: Edge cases + frozen constant lock

**Files:**
- Modify: `tests/data/test_splits.py` — append tests

- [ ] **Step 7.1: Append edge-case + frozen-constant tests**

Append to `tests/data/test_splits.py`:

```python
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
```

- [ ] **Step 7.2: Run full test file (final count)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q
```

Expected: `35 passed` (30 + 5 new = final).

---

## Task 8: Three-command verification + STOP + commit

**Files:**
- No new files; stages all prior changes.

- [ ] **Step 8.1: Run the models-tests gate (no regression in #5A/#5B)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/stages/models -q
```

Expected: `80 passed`.

- [ ] **Step 8.2: Run the N08 face + data tests**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  tests/test_package_import.py \
  tests/stages \
  tests/data \
  -q
```

Expected: previous `381 passed` (#5C-2 baseline) + 35 new = `416 passed`.

- [ ] **Step 8.3: Run the Resume Gate**

Command:
```bash
bash scripts/check_n08_resume_gate.sh; echo "RESUME_GATE_EXIT=$?"
```

Expected:
```text
GATE PASSED. Substantive N08 work may proceed.
RESUME_GATE_EXIT=0
```

- [ ] **Step 8.4: Inventory the changes**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git status --short
git diff --stat HEAD
```

Expected:
```text
M  src/intraday_research/data/__init__.py
?? src/intraday_research/data/splits.py
?? tests/data/test_splits.py
```

- [ ] **Step 8.5: STOP and report to user for explicit commit authorization**

Before staging or committing, the agent reports:
- The three verification command outputs.
- Files to be staged.
- Proposed commit message (Step 8.7 below).

WAIT for the user's explicit `stage + commit` authorization. Do NOT
proceed without it (AGENTS.md §9).

- [ ] **Step 8.6: Stage files by name**

Command (only after user authorizes):
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git add \
  src/intraday_research/data/__init__.py \
  src/intraday_research/data/splits.py \
  tests/data/test_splits.py
git status --short
git diff --cached --stat
```

Expected: 1 `M` (modified) + 2 `A` (added), ~550 lines staged total.

- [ ] **Step 8.7: Commit**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git commit -m "$(cat <<'EOF'
feat(n08): implement apply_chronological_split in data/splits.py (#5C-4)

Fifth piece of the #5C raw-data pipeline. Adds
src/intraday_research/data/splits.py: a pure-numpy function returning
per-row partition codes (PARTITION_TRAIN=0 / PARTITION_VALIDATION=1)
plus a cross-split valid_mask that encodes "label horizon does not
cross a split boundary" per AGENTS.md section 4.1.

Behavior:
  - Pure-numpy implementation; does NOT wrap
    baseline_v1.add_split_and_invalidate_boundaries (its DataFrame +
    future_cumulative_return coupling is overkill for #5C-4's narrow
    split-boundary scope). Anti-drift is tested per-row against
    baseline_v1.assign_calendar_split.
  - 1-D ndarray of datetime64[ns] timestamps in -> (partition: int8,
    valid_mask: bool_) ndarray tuple out.
  - Two-partition int8 encoding (PARTITION_TRAIN=0,
    PARTITION_VALIDATION=1); no outside_defined_calendar code because
    val_end fail-loud rejects holdout rows and anything < validation_start
    is classified as TRAIN with no lower bound (raw_bars handles that).
  - valid_mask[t] is True iff t + horizon_k < n AND partition[t] ==
    partition[t + horizon_k]. The last horizon_k rows of train (whose
    horizon row lands in validation) are correctly invalidated.
  - VAL_END is imported from raw_bars.py as single source of truth;
    VALIDATION_START = pd.Timestamp("2013-09-16") is defined locally.
  - Wrapper-layer fail-fast checks (in order): validation_start /
    val_end / horizon_k type + boundary ordering, then timestamps
    1-D / datetime64 / tz-naive / no-NaT / sorted, then val_end fail-
    loud, then n=0 short-circuit.
  - Tz-aware detection uses isinstance(arr.dtype, pd.DatetimeTZDtype)
    rather than the deprecated pd.api.types.is_datetime64tz_dtype
    (same fix as #5C-2 features.py to avoid DeprecationWarning being
    promoted to a test error by pytest.ini's filterwarnings).
  - val_end fail-loud mirrors raw_bars.load_ticker_bars's holdout
    closure discipline as a safety net; in practice raw_bars has
    already rejected those rows.
  - cross-day handling is intentionally NOT in #5C-4; it lives in
    #5C-1 labels and #5C-5 windows.

Project-frozen alias apply_stage0_chronological_split(timestamps,
horizon_k=...) pins validation_start=VALIDATION_START and
val_end=VAL_END so callers don't have to repeat them.

Updates src/intraday_research/data/__init__.py docstring to record
that splits.py has now arrived (alongside labels, raw_bars, features).

Tests in tests/data/test_splits.py cover the section 4 contract on
synthetic timestamps (35 tests across 7 categories): per-row cross-
check against baseline_v1.assign_calendar_split (anti-drift gate;
constructs the 3-key splits dict baseline_v1 expects with a
placeholder closed_holdout_boundary_only that never matches due to
val_end fail-loud), partition encoding + boundary precision (exact-
at-validation_start, one-bar-before, far-before, near-val_end),
valid_mask cross-split semantics (all-train, all-validation, mixed
train->validation invalidates last horizon_k train rows, validation
near val_end with insufficient horizon, n <= horizon_k), three Stage
0 alias horizons (h03/h09/h24), four holdout closure fail-loud cases,
13 wrapper-layer input guards (incl. parametrized horizon_k and
bool-as-int-subclass), and edge cases + frozen constant lock
(empty short-circuit, n == horizon_k, PARTITION codes pinned to
int8(0)/int8(1), VALIDATION_START pinned to 2013-09-16, VAL_END
imported from raw_bars module as single source of truth).

No changes to:
  - baseline_v1.py (anti-drift cross-check only reads it)
  - contract module
  - stage Python module
  - models/deep_sequence/ (controls + folds still as implemented in
    #5A / #5B)
  - labels.py (#5C-1), raw_bars.py (#5C-3), features.py (#5C-2) all
    unchanged
  - notebook content / design doc / configs

Verified:
  - pytest tests/stages/models = 80 passed (no regression)
  - pytest N08 face + tests/data = 416 passed
  - check_n08_resume_gate.{sh,ps1} exits 0; GATE PASSED

Spec: docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md
Plan: docs/superpowers/plans/2026-06-07-n08-data-splits-chronological-implementation-plan.md
EOF
)"
```

- [ ] **Step 8.8: Post-commit verification**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git log -1 --stat
git status --branch --short
```

Expected: commit SHA shown, working tree clean (no `M`/`R`/`A`/`??`
entries below the branch line), `[ahead N]` indicator showing the new
commit is local (not yet pushed).

- [ ] **Step 8.9: Report completion**

Report to user:
- Final commit SHA.
- Test counts before/after (381 → 416).
- Resume Gate state.
- Updated task list (#13 / `#5C-4` → completed).
- Suggest next: push (user authorization required) and/or open #5C-5
  (window builder) — the LAST piece of #5C.

---

## Pre-Commit Checklist (Task 8 condensed)

Run before authorizing commit, all from Git Bash on Windows with
explicit project Python path:

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/stages/models -q

E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  tests/test_package_import.py \
  tests/stages \
  tests/data \
  -q

bash scripts/check_n08_resume_gate.sh

git status --short
```

Expected (in order):
- `80 passed`
- `416 passed`
- `GATE PASSED`, exit 0
- Three changes: `M` `src/intraday_research/data/__init__.py`, `??` for
  `src/intraday_research/data/splits.py` and `tests/data/test_splits.py`

If any one fails, STOP and report. Do NOT debug under
brainstorming/writing-plans gates without re-engaging the user.

---

## Out of Scope

Explicitly NOT in this plan:

- The other pieces of #5C (#5C-1 labels done, #5C-3 raw_bars done,
  #5C-2 features done; #5C-5 window builder is the sibling).
- Editing or refactoring `baseline_v1.assign_calendar_split` or
  `baseline_v1.add_split_and_invalidate_boundaries`.
- A `train_start` lower bound — anything `< validation_start` is
  TRAIN; `train_start = 1998-01-02` is enforced upstream by raw bar
  data availability.
- A `closed_holdout_boundary_only` partition code — rows past
  `val_end` are fail-loud rejected; they never reach the partition
  array.
- Cross-day boundary invalidation — that lives in #5C-1 labels and
  #5C-5 windows.
- Train-only scaler fitting — handled by the orchestrator using
  `baseline_v1.fit_train_only_scaler`.
- Pooled multi-ticker handling — caller iterates per-ticker after
  `#5C-3 load_ticker_bars` returns the pooled DataFrame.
- Pushing the commit (push is a separate user-authorized step per
  AGENTS.md §9).
- `tests/__init__.py` / `tests/data/__init__.py` (pytest auto-discovery
  handles this; matches existing `tests/data/test_*.py` conventions).

---

## Known Risks

1. **`baseline_v1.assign_calendar_split` requires a 3-key splits dict.**
   The cross-check test in Task 1 constructs a synthetic 3-key dict
   that includes a `closed_holdout_boundary_only` slot pointing to
   `(val_end, "2099-01-01")`. This placeholder slot never actually
   matches a kept row because the val_end fail-loud check (Step 6 of
   the data flow) rejects any `timestamp >= val_end` BEFORE the
   partition is computed. The test asserts that
   `assign_calendar_split` returns either `"train"` or `"validation"`
   for every kept row (it raises if it returns `"closed_holdout_..."`
   or `"outside_defined_calendar"`, because our two-partition
   encoding has no code for either).

2. **`np.datetime64` vs `pd.Timestamp` comparison.** The implementation
   converts the `validation_start` and `val_end` Timestamps to
   `np.datetime64[ns]` before comparing against the timestamp array
   (`timestamps < validation_start_np`). This avoids implicit
   conversion at every comparison element and keeps numpy semantics.
   Risk: if a future pandas / numpy version changes the conversion
   semantics, the cross-check test against `baseline_v1.assign_calendar_split`
   would catch the drift.

3. **Bool subclass of int.** `bool` is a subclass of `int` in Python,
   so `isinstance(True, int)` returns True. The implementation checks
   `isinstance(horizon_k, bool)` BEFORE `isinstance(horizon_k, int)`
   so that `horizon_k=True` is rejected (not silently treated as
   `horizon_k=1`). Test 6.7 locks this behavior.

4. **`isinstance(arr.dtype, pd.DatetimeTZDtype)` idiom.** Same fix as
   #5C-2: do NOT use `pd.api.types.is_datetime64tz_dtype(arr)`, which
   emits a DeprecationWarning that pytest.ini's
   `filterwarnings = error::Warning:intraday_research\..*` would
   promote to a test error.

5. **`val_end` fail-loud safety net.** In practice
   `raw_bars.load_ticker_bars` has already rejected any
   `timestamp >= val_end` before this function ever sees the data,
   so the fail-loud check inside `#5C-4` should never fire on real
   pipelines. It exists as defense-in-depth: a caller that bypasses
   raw_bars (e.g. constructs timestamps from a different source)
   still cannot smuggle holdout rows into the split logic.

---

## Self-Review (skill checklist, 4-way sync)

Compared against
`docs/superpowers/specs/2026-06-07-n08-data-splits-chronological-design.md`:

**1. Spec coverage**: every spec §4 test category and every spec §3
error mode maps to a task / step in this plan:

| Spec section | Plan task |
|---|---|
| §1 architecture (constants + generic + Stage 0 alias) | Task 1 step 1.4 (implementation) + Task 4 (alias) |
| §2 data flow step 1 (validation_start validate) | Task 6 (TypeError + ValueError tests) |
| §2 data flow step 2 (val_end validate) | Task 6 (TypeError + ValueError tests) |
| §2 data flow step 3 (boundary ordering) | Task 6 (validation_start >= val_end test) |
| §2 data flow step 4 (horizon_k validate) | Task 6 (parametrized + bool test) |
| §2 data flow step 5 (timestamps validate) | Task 6 (5 tests: non-1D, int dtype, tz-aware, NaT, unsorted) |
| §2 data flow step 6 (val_end fail-loud) | Task 5 (4 tests) |
| §2 data flow step 7 (n=0 short-circuit) | Task 7 (empty test) |
| §2 data flow step 8 (partition compute) | Task 1 (cross-check) + Task 2 (boundary precision) |
| §2 data flow step 9 (valid_mask compute) | Task 3 (5 tests) + Task 7 (n == horizon_k) |
| §3 errors (13 modes) | All addressed across Tasks 5, 6 |
| §4 categories 1–7 | Tasks 1, 2, 3, 4, 5, 6, 7 (1:1 mapping) |

**2. Placeholder scan**: no "TBD" / "TODO" / "appropriate handling" /
"similar to" / undefined methods. All code blocks contain runnable
code. ✓

**3. Type consistency**: `apply_chronological_split` signature,
return type, and parameter names are identical across Task 1 stub
(Step 1.1), Task 1 implementation (Step 1.4), and all test calls
(Tasks 2–7). `apply_stage0_chronological_split` signature matches
between stub and implementation. `PARTITION_TRAIN`, `PARTITION_VALIDATION`,
and `VALIDATION_START` constant types (`np.int8`, `pd.Timestamp`) are
consistent. `VAL_END` is consistently imported from
`intraday_research.data.raw_bars` (not redefined). ✓

---

## Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-06-07-n08-data-splits-chronological-implementation-plan.md`.

Per the user's standing instruction, the plan is NOT auto-executed.
Awaiting explicit user review of this plan and authorization to begin
Task 1. Do not invoke an execution skill before that authorization.
