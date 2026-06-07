# N08 #5C-5 Sliding-Window Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/intraday_research/data/windows.py` — a pure-numpy sliding-window builder turning per-row package arrays (#5C-1/#5C-2/#5C-4 outputs) into supervised `(X, y, ...)` training windows for the N08 deep-sequence models and the LightGBM control.

**Architecture:** Two public functions. `build_windows_single_ticker` is the core (all window geometry + fail-loud validation). `build_windows` is a thin pooled wrapper: group by `ticker_ids`, call the core per ticker, concatenate, remap provenance indices. No second copy of windowing logic. Pure numpy; baseline_v1 is touched only as an anti-drift test oracle, never imported by the module.

**Tech Stack:** Python 3.11, numpy. pytest for tests.

**Spec:** `docs/superpowers/specs/2026-06-07-n08-data-windows-design.md` (committed `bd646f1`).

---

## Execution Environment (read before any command)

- **Shell assumption: Git Bash on Windows.** All commands below are Git Bash.
- **Python is always the explicit interpreter path** — never `python`/`py`:
  `E:/codex_workspace/_envs/py311_shared/python.exe`
- Tests are pytest. Do **not** use `head -1` / interactive git flags.
- Line length ≤ 100, `snake_case`, type hints use `| None` not `Optional[X]`.
- `pytest.ini` promotes `Warning:intraday_research\..*` to **errors** — any
  DeprecationWarning from our module fails the suite. Use
  `isinstance(arr.dtype, pd.DatetimeTZDtype)` idioms, never the deprecated
  `pd.api.types.is_datetime64tz_dtype`. (This module needs no pandas at all —
  see Known Risks.)
- **Do NOT commit or push** until Task 8's STOP gate and explicit user
  authorization. **Never push.**
- Forbidden to touch: `baseline_v1.py`, `AGENTS.md`, `pytest.ini`, any
  contract/notebook/config/holdout file.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/intraday_research/data/windows.py` | **Create** | the two public builders + private helpers |
| `tests/data/test_windows.py` | **Create** | ~57 tests (no `__init__.py` in `tests/data/`) |
| `src/intraday_research/data/__init__.py` | **Modify** | line-7 docstring only |

The module decomposes into: `_validate_core_inputs` (shared fail-loud guard),
`_empty_core_result` (empty-schema factory), `build_windows_single_ticker`
(core), `build_windows` (pooled wrapper). One file — the units are small and
change together.

---

## Task 1: Scaffold `windows.py` + complete core, driven by a happy-path test

This task writes the **complete** core implementation (validation + slide +
assemble), driven by one happy-path test. Tasks 2–5 then add comprehensive
coverage against this finished core. (Mirrors the #5C-4 plan's Task-1 shape:
implement the cohesive validated unit once, harden coverage after.)

**Files:**
- Create: `src/intraday_research/data/windows.py`
- Test: `tests/data/test_windows.py`
- Modify: `src/intraday_research/data/__init__.py:7`

- [ ] **Step 1.1: Write the failing happy-path test**

Create `tests/data/test_windows.py`:

```python
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
```

- [ ] **Step 1.2: Run it to confirm RED**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py::test_build_windows_single_ticker_basic_single_day -v
```

Expected: FAIL — `module 'intraday_research.data.windows' has no attribute ...`
(module does not exist yet).

- [ ] **Step 1.3: Write the complete core implementation**

Create `src/intraday_research/data/windows.py`:

```python
"""Numpy-faced sliding-window builder for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-windows-design.md.
"""

from __future__ import annotations

import numpy as np

_BINARY = np.array([0, 1], dtype=np.int8)
_DATETIME_NS = np.dtype("datetime64[ns]")
_ZERO_NS = np.timedelta64(0, "ns")


def _validate_core_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> int:
    """Fail-loud validation shared by both public builders.

    Returns the row count ``n`` on success.
    """
    aux = {
        "labels": labels,
        "timestamps": timestamps,
        "partition": partition,
        "feature_valid_mask": feature_valid_mask,
        "target_valid_mask": target_valid_mask,
    }
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be np.ndarray; got {type(features).__name__}")
    for name, arr in aux.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be np.ndarray; got {type(arr).__name__}")

    # bool is a subclass of int -> check bool FIRST.
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise TypeError(
            f"window_size must be int (not bool); got {type(window_size).__name__}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0; got {window_size}")

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got ndim={features.ndim}")
    if features.shape[1] < 1:
        raise ValueError(f"features must have >= 1 column; got F={features.shape[1]}")
    for name, arr in aux.items():
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got ndim={arr.ndim}")

    n = features.shape[0]
    for name, arr in aux.items():
        if arr.shape[0] != n:
            raise ValueError(
                f"{name} length {arr.shape[0]} != features rows {n}"
            )

    if features.dtype != np.float64:
        raise ValueError(f"features must be float64; got {features.dtype}")
    if labels.dtype != np.int8:
        raise ValueError(f"labels must be int8; got {labels.dtype}")
    if timestamps.dtype != _DATETIME_NS:
        raise ValueError(f"timestamps must be datetime64[ns]; got {timestamps.dtype}")
    if partition.dtype != np.int8:
        raise ValueError(f"partition must be int8; got {partition.dtype}")
    if feature_valid_mask.dtype != np.bool_:
        raise ValueError(
            f"feature_valid_mask must be bool; got {feature_valid_mask.dtype}"
        )
    if target_valid_mask.dtype != np.bool_:
        raise ValueError(
            f"target_valid_mask must be bool; got {target_valid_mask.dtype}"
        )

    if not np.isin(partition, _BINARY).all():
        raise ValueError("partition values must be in {0, 1}")

    # Label-contract pre-pass: a target-valid row must carry a binary label.
    bad = target_valid_mask & ~np.isin(labels, _BINARY)
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            "target_valid_mask=True rows must have labels in {0, 1}; "
            f"row {first} has label {int(labels[first])}"
        )
    return n


def _empty_core_result(window_size: int, n_features: int) -> dict[str, np.ndarray]:
    """Empty output schema with exact dtypes (shared by both builders)."""
    return {
        "X": np.empty((0, window_size, n_features), dtype=np.float64),
        "y": np.empty((0,), dtype=np.int8),
        "target_partition": np.empty((0,), dtype=np.int8),
        "target_timestamps": np.empty((0,), dtype=_DATETIME_NS),
        "target_row_positions": np.empty((0,), dtype=np.int64),
    }


def build_windows_single_ticker(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Build stride-1 same-day sliding windows for a single ticker.

    Args:
        features: ``(n, F)`` float64 feature matrix, ``F >= 1``. Finiteness is
            NOT re-checked here; the caller asserts it via ``feature_valid_mask``.
        labels: ``(n,)`` int8 in ``{0, 1, -1}``; ``-1`` only where
            ``target_valid_mask`` is False.
        timestamps: ``(n,)`` datetime64[ns], tz-naive, nondecreasing.
        partition: ``(n,)`` int8 in ``{0, 1}`` (0=train, 1=validation).
        feature_valid_mask: ``(n,)`` bool; True iff that row's features are usable.
        target_valid_mask: ``(n,)`` bool; True iff that row may be a target.
        window_size: positive int (not bool); window length, stride is 1.

    Returns:
        dict with keys ``X`` (W, window_size, F) float64, ``y`` (W,) int8 in
        ``{0, 1}``, ``target_partition`` (W,) int8, ``target_timestamps`` (W,)
        datetime64[ns], ``target_row_positions`` (W,) int64 (positions into the
        input arrays). ``W`` may be 0.

    Raises:
        TypeError / ValueError: see the spec error-mode table.
    """
    n = _validate_core_inputs(
        features, labels, timestamps, partition,
        feature_valid_mask, target_valid_mask, window_size,
    )
    n_features = features.shape[1]

    if n > 1 and (np.diff(timestamps) < _ZERO_NS).any():
        raise ValueError("timestamps must be nondecreasing")

    if n < window_size:
        return _empty_core_result(window_size, n_features)

    dates = timestamps.astype("datetime64[D]")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.int8] = []
    part_rows: list[np.int8] = []
    ts_rows: list[np.datetime64] = []
    pos_rows: list[int] = []

    for end_pos in range(window_size - 1, n):
        sl = slice(end_pos - window_size + 1, end_pos + 1)
        if not (dates[sl] == dates[end_pos]).all():
            continue  # window crosses a trading-day boundary
        if not (partition[sl] == partition[end_pos]).all():
            raise ValueError(
                "same-day window has non-uniform partition at end_pos "
                f"{end_pos} (timestamp {timestamps[end_pos]}); the caller's "
                "partition is not date-aligned"
            )
        if not feature_valid_mask[sl].all():
            continue
        if not target_valid_mask[end_pos]:
            continue
        x_rows.append(features[sl])
        y_rows.append(labels[end_pos])
        part_rows.append(partition[end_pos])
        ts_rows.append(timestamps[end_pos])
        pos_rows.append(end_pos)

    if not x_rows:
        return _empty_core_result(window_size, n_features)

    return {
        "X": np.stack(x_rows).astype(np.float64, copy=False),
        "y": np.asarray(y_rows, dtype=np.int8),
        "target_partition": np.asarray(part_rows, dtype=np.int8),
        "target_timestamps": np.asarray(ts_rows, dtype=_DATETIME_NS),
        "target_row_positions": np.asarray(pos_rows, dtype=np.int64),
    }
```

- [ ] **Step 1.4: Update the `data/__init__.py` docstring (line 7)**

Replace the line:

```
  - ``splits``, ``windows``  arrive in sibling commits #5C-4 / #5C-5.
```

with:

```
  - ``splits``    chronological train/validation partition (#5C-4)
  - ``windows``   stride-1 same-day sliding-window builder (#5C-5)
```

No import re-export is added.

- [ ] **Step 1.5: Run the happy-path test to confirm GREEN**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py::test_build_windows_single_ticker_basic_single_day -v
```

Expected: PASS.

---

## Task 2: Anti-drift cross-check vs `baseline_v1.build_windows_for_segment`

**Files:**
- Test: `tests/data/test_windows.py`

- [ ] **Step 2.1: Add the cross-check test**

Append to `tests/data/test_windows.py`:

```python
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
```

- [ ] **Step 2.2: Run the cross-check**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py::test_build_windows_single_ticker_matches_baseline_v1 -v
```

Expected: PASS. (If it fails, the divergence is real — do **not** weaken the
test; reconcile the implementation against baseline_v1's filter-then-window
semantics.)

---

## Task 3: Input-guard tests (14 error modes)

**Files:**
- Test: `tests/data/test_windows.py`

- [ ] **Step 3.1: Add the guard tests**

Append:

```python
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
```

- [ ] **Step 3.2: Run the guard tests**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -k "rejects" -v
```

Expected: all PASS (25 items: 6 + 1 + 1 + 1 + 3 + 6 + 1 + 1 + 1 + 2 + 1 + 1).

---

## Task 4: Window-semantics tests (mask / cross-day / partition / boundary)

**Files:**
- Test: `tests/data/test_windows.py`

- [ ] **Step 4.1: Add the semantics tests**

Append:

```python
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
```

- [ ] **Step 4.2: Run the semantics tests**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -k "feature_invalid or target_invalid or target_valid_but or cross_day or window_size_one or short_day or target_partition_value or boundary_window or partition_uniformity" -v
```

Expected: all PASS (10 items).

---

## Task 5: Edge / empty / dtype-lock tests (core)

**Files:**
- Test: `tests/data/test_windows.py`

- [ ] **Step 5.1: Add the edge tests**

Append:

```python
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
```

- [ ] **Step 5.2: Run the edge tests**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -k "empty or n_less_than or all_windows_filtered or output_key or dtype_int64 or y_dtype or provenance_single" -v
```

Expected: all PASS (8 items).

---

## Task 6: Pooled wrapper `build_windows`, driven by a basic test

**Files:**
- Modify: `src/intraday_research/data/windows.py`
- Test: `tests/data/test_windows.py`

- [ ] **Step 6.1: Write the failing pooled test**

Append:

```python
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
```

- [ ] **Step 6.2: Run it to confirm RED**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py::test_build_windows_pooled_basic_two_tickers -v
```

Expected: FAIL — `module ... has no attribute 'build_windows'`.

- [ ] **Step 6.3: Implement `build_windows`**

Append to `src/intraday_research/data/windows.py`:

```python
def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    *,
    partition: np.ndarray,
    feature_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Pooled multi-ticker wrapper around ``build_windows_single_ticker``.

    Groups rows by ``ticker_ids`` (sorted-unique order), runs the single-ticker
    core per group, remaps ``target_row_positions`` back to positions into the
    pooled input arrays, and concatenates the per-ticker blocks (per-ticker
    block order, block-internal ascending by target timestamp).

    ``ticker_ids`` may be numeric, string, or homogeneous object dtype; a mixed
    incomparable object array is rejected. All other arguments and the output
    schema match the core, with one extra key ``target_ticker_ids`` (dtype
    equal to ``ticker_ids.dtype``).
    """
    n = _validate_core_inputs(
        features, labels, timestamps, partition,
        feature_valid_mask, target_valid_mask, window_size,
    )
    if not isinstance(ticker_ids, np.ndarray):
        raise TypeError(
            f"ticker_ids must be np.ndarray; got {type(ticker_ids).__name__}"
        )
    if ticker_ids.ndim != 1:
        raise ValueError(f"ticker_ids must be 1-D; got ndim={ticker_ids.ndim}")
    if ticker_ids.shape[0] != n:
        raise ValueError(
            f"ticker_ids length {ticker_ids.shape[0]} != features rows {n}"
        )
    n_features = features.shape[1]

    def _empty_pooled() -> dict[str, np.ndarray]:
        res = _empty_core_result(window_size, n_features)
        res["target_ticker_ids"] = np.empty((0,), dtype=ticker_ids.dtype)
        return res

    if n == 0:
        return _empty_pooled()

    try:
        unique_tickers = np.unique(ticker_ids)
    except TypeError as exc:
        raise ValueError(
            f"ticker_ids must be homogeneous comparable dtype; "
            f"got {ticker_ids.dtype} ({exc})"
        ) from exc

    blocks: list[dict[str, np.ndarray]] = []
    for ticker in unique_tickers:
        global_pos = np.flatnonzero(ticker_ids == ticker)
        try:
            block = build_windows_single_ticker(
                features[global_pos],
                labels[global_pos],
                timestamps[global_pos],
                partition=partition[global_pos],
                feature_valid_mask=feature_valid_mask[global_pos],
                target_valid_mask=target_valid_mask[global_pos],
                window_size=window_size,
            )
        except ValueError as exc:
            raise ValueError(f"ticker {ticker!r}: {exc}") from exc
        w = block["y"].shape[0]
        if w == 0:
            continue
        block["target_row_positions"] = global_pos[block["target_row_positions"]]
        block["target_ticker_ids"] = np.full(w, ticker, dtype=ticker_ids.dtype)
        blocks.append(block)

    if not blocks:
        return _empty_pooled()

    keys = (
        "X", "y", "target_partition", "target_timestamps",
        "target_row_positions", "target_ticker_ids",
    )
    return {key: np.concatenate([b[key] for b in blocks]) for key in keys}
```

- [ ] **Step 6.4: Run the pooled basic test + a single/pooled consistency test**

Append:

```python
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
```

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -k "pooled_basic or pooled_matches" -v
```

Expected: PASS (2 items).

---

## Task 7: Pooled-specific tests (cross-ticker / provenance / dtype / errors)

**Files:**
- Test: `tests/data/test_windows.py`

- [ ] **Step 7.1: Add the pooled tests**

Append:

```python
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
```

- [ ] **Step 7.2: Run the pooled tests**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -k "pooled" -v
```

Expected: all PASS (pooled items: basic 1 + matches 1 + no_cross_ticker 1 +
block_order 1 + provenance 1 + dtype_passthrough 2 + ticker_context 1 +
empty_n_zero 1 + all_blocks_empty 1 + mixed_object 1 + length_mismatch 1 = 12).

- [ ] **Step 7.3: Run the whole new test file**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_windows.py -q
```

Expected: **all green, 0 failed / 0 error / 0 skipped.** Count ≈ 57
(1 + 1 + 25 + 10 + 8 + 2 + 12). The HARD gate is "all green, no skip/xfail";
the exact integer is informational.

---

## Task 8: Three-command verification + STOP + commit

**Files:**
- No new files; stages all prior changes.

- [ ] **Step 8.1: Models-tests gate (no #5A/#5B regression)**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/stages/models -q
```

Expected: `80 passed`.

- [ ] **Step 8.2: N08 face + data tests**

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  tests/test_package_import.py \
  tests/stages \
  tests/data \
  -q
```

Expected: prior `416 passed` (post-#5C-4 baseline) + ~57 new = **~473 passed,
0 failed / 0 skipped**. If #5C-4 is not yet merged, the baseline is 381 and the
total is ~438; record the actual baseline before this task.

- [ ] **Step 8.3: Resume Gate**

```bash
bash scripts/check_n08_resume_gate.sh; echo "RESUME_GATE_EXIT=$?"
```

Expected:
```text
GATE PASSED. Substantive N08 work may proceed.
RESUME_GATE_EXIT=0
```

- [ ] **Step 8.4: Inventory the changes (diff scope must be exactly these 3)**

```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git status --short
git diff --stat HEAD
```

Expected exactly:
```text
 M src/intraday_research/data/__init__.py
?? src/intraday_research/data/windows.py
?? tests/data/test_windows.py
```
Any other touched path → STOP, do not commit, report.

- [ ] **Step 8.5: STOP and report for explicit commit authorization**

Report the three-command results + diff scope. **Do not stage or commit** until
the user authorizes. (Under the automated overnight loop, the commit may
proceed only if ALL hold: piece tests all green; N08 face all green;
Resume Gate exit 0; no skip/xfail/deselect; diff is exactly the 3 files above;
no forbidden file touched. Otherwise write the runlog and stop.)

- [ ] **Step 8.6: Stage + commit (after authorization)**

```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git add src/intraday_research/data/windows.py \
        tests/data/test_windows.py \
        src/intraday_research/data/__init__.py
git commit -F - <<'EOF'
feat(n08): implement build_windows in data/windows.py (#5C-5)

Pure-numpy stride-1 same-day sliding-window builder, final #5C piece.

- build_windows_single_ticker: core geometry + fail-loud validation
  (14 error modes); same-day windows via timestamps.astype(datetime64[D]);
  keyword-only partition / masks / window_size to block silent swaps;
  label-contract pre-pass; defensive same-day partition-uniformity check.
- build_windows: thin pooled wrapper; per-ticker slice -> core -> remap
  target_row_positions to pooled global indices -> concatenate; wraps core
  errors with ticker context; structural no-cross-ticker guarantee.
- Output dict: X f64 (W,ws,F), y int8 {0,1}, target_partition int8,
  target_timestamps datetime64[ns], target_row_positions int64; pooled adds
  target_ticker_ids (input dtype passthrough).
- ~57 tests incl. anti-drift cross-check vs
  baseline_v1.build_windows_for_segment.

Spec: docs/superpowers/specs/2026-06-07-n08-data-windows-design.md
Plan: docs/superpowers/plans/2026-06-07-n08-data-windows-implementation-plan.md
EOF
```

- [ ] **Step 8.7: Post-commit sanity (no push)**

```bash
git -C E:/codex_workspace/projects/intraday_stock_direction_research log --oneline -1
git -C E:/codex_workspace/projects/intraday_stock_direction_research status --short
```

Expected: the new commit on top; clean tree. **Do not push.**

---

## Pre-Commit Checklist (Task 8 gate, all must hold)

1. `pytest tests/stages/models -q` → `80 passed`.
2. N08 face (Step 8.2 command) → all green, 0 failed/skipped.
3. `bash scripts/check_n08_resume_gate.sh` → `RESUME_GATE_EXIT=0`.
4. `git status --short` shows exactly the 3 expected paths.
5. No skip / xfail / deselect anywhere in the run.
6. No forbidden file touched (`baseline_v1.py`, `AGENTS.md`, `pytest.ini`,
   contracts/notebooks/config/holdout).

## Known Risks / Notes

1. **No pandas import in the module.** `windows.py` is pure numpy; pandas
   appears only in the test file for the baseline_v1 cross-check fixture. This
   sidesteps the `pytest.ini` Warning→error trap entirely.
2. **bool is a subclass of int.** `window_size` validation checks
   `isinstance(window_size, bool)` **before** `isinstance(window_size, int)`
   (test 3: `test_rejects_window_size_bool`).
3. **Cross-check fixture must be date-aligned.** baseline_v1 filters by split
   *before* windowing; a non-date-aligned partition would diverge. The fixture
   puts train on day 1 and validation on days 2-3 (Task 2).
4. **`target_row_positions` is row space**, distinct from the window space
   (`0..W-1`) consumed by `models/deep_sequence/folds.py`. Provenance tests
   (5/7) assert it indexes the inputs.
5. **Pooled remap.** The core returns slice-relative positions; the wrapper
   remaps via `global_pos[...]` (Step 6.3). Test 7 asserts pooled provenance.
6. **Eager float64, multi-GB at full scale.** Matches baseline_v1; out of scope
   to chunk/memmap/float32 here (spec §6).
7. **Exact face count is informational.** The hard gate is all-green +
   no-skip; the loop must not weaken a test to hit a number.

## Out of Scope

No feature scaling; no `stride`/`target` parameters; no global chronological
re-sort in the pooled wrapper; no generator/memmap/chunking; no float32; no
re-check of `np.isfinite(features)` (caller's `feature_valid_mask` owns it).

## Self-Review (4-way sync with spec)

- **Interface (spec §2) ↔ Task 1/6 signatures:** keyword-only boundary after
  the core triple; pooled adds `ticker_ids` positional + `target_ticker_ids`
  output. ✔
- **Data flow (spec §3) ↔ Task 1 Step 1.3 / Task 6 Step 6.3:** validate →
  order-check → label pre-pass → empty fast-path → date key → slide
  (same-day → partition-uniform → feature → target → emit) → assemble; pooled
  unique→slice→core→remap→concat. ✔
- **Error table (spec §4, 14 modes) ↔ Task 3 + Task 7:** modes 1-12 in Task 3
  (25 items), mode 13 in Task 4 (`partition_uniformity`), mode 14 in Task 7
  (`mixed_object`). ✔
- **Testing strategy (spec §5) ↔ Tasks 2-7:** cross-check (Task 2) + 7
  categories (guards/partition/mask/cross-day/pooled/edge/locks) distributed
  across Tasks 3-7. ✔
- **Placeholder scan:** none — every step has concrete code and an exact
  command with expected output.
- **Type consistency:** `_validate_core_inputs`, `_empty_core_result`,
  `build_windows_single_ticker`, `build_windows` names + signatures identical
  across Task 1, 6, and the commit message. ✔
