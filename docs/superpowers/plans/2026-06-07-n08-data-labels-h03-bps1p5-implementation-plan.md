# N08 #5C-1 — `data/labels.py` Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.
> Execute tasks in order; each task is a self-contained RED→GREEN→VERIFY
> cycle. Do NOT commit until Task 6 — every intermediate verification is
> read-only.
>
> **Shell assumption:** All commands assume **Git Bash on Windows** (the
> project's standard shell; `bash scripts/check_n08_resume_gate.sh` and
> the `.sh` siblings already depend on it). The Task 6 heredoc for
> `git commit -m` requires Git Bash; from PowerShell, write the message
> to a file first and use `git commit -F <file>`, or invoke the
> PowerShell sibling `scripts/check_n08_resume_gate.ps1` for the gate.
> All other commands use the explicit project-Python path
> (`E:/codex_workspace/_envs/py311_shared/python.exe`) and avoid shell
> env-var shorthands (`$PYTHON`, `head -1`).

**Goal:** Implement `src/intraday_research/data/labels.py` as a numpy-faced
wrapper around `baseline_v1.make_no_trade_band_labels`, with one generic
function and three frozen-config aliases, covered by 34 cross-checked
tests and shipped in a single commit.

**Architecture:** Wrap the frozen `baseline_v1.make_no_trade_band_labels`
behind a pandas-glue boundary; expose `(int8 labels with -1 sentinel,
bool_ valid_mask)` numpy tuple. No reimplementation of label semantics.
Cross-split invalidation deferred to #5C-4.

**Tech Stack:** Python 3.11 / numpy / pandas / pytest. Project Python
`E:/codex_workspace/_envs/py311_shared/python.exe`.

**Reference commits:** #5A `0616701` (controls.py LightGBM), #5B `e85b55e`
(folds.py rolling-origin) — same scope discipline and verification gates.

**Spec:** `docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md`
(committed in `023270b`).

---

## Files

| Path | Action | Notes |
|---|---|---|
| `src/intraday_research/data/__init__.py` | create | subpackage docstring only |
| `src/intraday_research/data/labels.py` | create | 3 constants + 1 generic + 3 aliases |
| `tests/data/test_labels.py` | create | 34 tests across 6 spec §4 categories |

No existing files are modified. No `__init__.py` is added under `tests/`
(matches the `tests/stages/models/` convention).

---

## Task 1: Package scaffold + first cross-check test (RED→GREEN)

**Files:**
- Create: `src/intraday_research/data/__init__.py`
- Create: `src/intraday_research/data/labels.py`
- Create: `tests/data/test_labels.py`

- [ ] **Step 1.1: Create the `data/` subpackage scaffold**

Create `src/intraday_research/data/__init__.py` with:

```python
"""Raw-data ingestion + label/feature/window helpers for the N08 #5C pipeline.

Submodules:
  - ``labels``  no-trade-band binary labels (#5C-1, this commit)
  - ``raw_bars``, ``features``, ``splits``, ``windows``  arrive in sibling
    commits #5C-2 .. #5C-5.

Validation-only scope (AGENTS.md §4.1); no holdout/test data is read by
anything in this subpackage.
"""
```

Create `src/intraday_research/data/labels.py` with the stub. The stub
pre-defines the three frozen configuration constants and the three
alias stubs so the test file in Step 1.2 imports successfully — the
RED then comes from `NotImplementedError` at call time, not from a
collection-level `ImportError` (P3 review #4):

```python
"""No-trade-band binary labels for the frozen Stage 0 candidate space.

Wraps ``intraday_research.baseline_v1.make_no_trade_band_labels`` with a
numpy-faced API. ``baseline_v1`` is the single source of truth for the
label semantics; this module's only job is the numpy <-> pandas glue and
input validation. Cross-split invalidation is deferred to #5C-4.

Frozen configurations (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md):

  - ``h03_bps1p5``  horizon_k=3,  threshold_bps=1.5
  - ``h09_bps3p0``  horizon_k=9,  threshold_bps=3.0
  - ``h24_bps7p5``  horizon_k=24, threshold_bps=7.5

See docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md.
"""

from __future__ import annotations

import numpy as np


H03_BPS1P5: dict = {"horizon_k": 3, "threshold_bps": 1.5}
H09_BPS3P0: dict = {"horizon_k": 9, "threshold_bps": 3.0}
H24_BPS7P5: dict = {"horizon_k": 24, "threshold_bps": 7.5}


def build_no_trade_band_labels(
    close: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon_k: int,
    threshold_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-faced wrapper around baseline_v1.make_no_trade_band_labels.

    Implementation lands in Task 1 step 1.4.
    """
    raise NotImplementedError("build_no_trade_band_labels — Task 1 step 1.4")


def build_h03_bps1p5_labels(close, timestamps):
    """h03_bps1p5 frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError("build_h03_bps1p5_labels — Task 1 step 1.4")


def build_h09_bps3p0_labels(close, timestamps):
    """h09_bps3p0 frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError("build_h09_bps3p0_labels — Task 1 step 1.4")


def build_h24_bps7p5_labels(close, timestamps):
    """h24_bps7p5 frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError("build_h24_bps7p5_labels — Task 1 step 1.4")
```

- [ ] **Step 1.2: Write the cross-check test (anti-drift gate)**

Create `tests/data/test_labels.py` with:

```python
"""Behavioral tests for ``intraday_research.data.labels`` (N08 #5C-1).

Synthetic-data tests only. No raw bar I/O, no fixture files, no official
validation, no holdout. Verifies the §4 contract documented in
``docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import make_no_trade_band_labels
from intraday_research.data.labels import (
    H03_BPS1P5,
    H09_BPS3P0,
    H24_BPS7P5,
    build_h03_bps1p5_labels,
    build_h09_bps3p0_labels,
    build_h24_bps7p5_labels,
    build_no_trade_band_labels,
)


def _synthetic_intraday_session(
    n: int = 80,
    start: str = "2025-01-02 09:30",
    close_seed: int = 0,
    drift_bps_per_bar: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """5-minute bars within one trading day. Close = 100 * cumulative product
    of (1 + per-bar drift + noise) so labels can be reasoned about by hand."""
    rng = np.random.default_rng(close_seed)
    timestamps = pd.date_range(start, periods=n, freq="5min").to_numpy()
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5  # ~0.1 bps -- below default threshold
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return close.astype(np.float64), timestamps


def test_wrapper_matches_baseline_v1_on_identical_inputs():
    """Anti-drift gate: same numeric output as baseline_v1 directly called."""
    close, timestamps = _synthetic_intraday_session(n=60, drift_bps_per_bar=2.0)
    horizon_k, threshold_bps = 3, 1.5

    # Baseline_v1 path -- the source of truth.
    frame = pd.DataFrame({
        "ticker": "_synthetic",
        "timestamp": pd.to_datetime(timestamps),
        "close": close,
    })
    expected = make_no_trade_band_labels(
        frame, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )
    expected_labels_float = expected["label"].to_numpy()
    expected_valid_mask = ~np.isnan(expected_labels_float)

    # Wrapper path.
    labels, valid_mask = build_no_trade_band_labels(
        close, timestamps, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )

    np.testing.assert_array_equal(valid_mask, expected_valid_mask)
    # Compare valid-position labels (sentinel -1 vs NaN differ by design).
    np.testing.assert_array_equal(
        labels[valid_mask], expected_labels_float[expected_valid_mask].astype(np.int8),
    )
```

- [ ] **Step 1.3: Run the test and verify it FAILS**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py::test_wrapper_matches_baseline_v1_on_identical_inputs -v
```

Expected: `FAILED` with `NotImplementedError("build_no_trade_band_labels
— Task 1 step 1.4")` (the stub from Step 1.1 pre-defines all imported
symbols, so collection succeeds and the failure happens at call time).
Confirm the red is for the right reason (missing implementation body,
not a typo or missing import).

- [ ] **Step 1.4: Implement the minimum body to make the test pass**

Replace the body of `src/intraday_research/data/labels.py` with:

```python
"""No-trade-band binary labels for the frozen Stage 0 candidate space.

Wraps ``intraday_research.baseline_v1.make_no_trade_band_labels`` with a
numpy-faced API. ``baseline_v1`` is the single source of truth for the
label semantics; this module's only job is the numpy <-> pandas glue and
input validation. Cross-split invalidation is deferred to #5C-4.

Frozen configurations (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md):

  - ``h03_bps1p5``  horizon_k=3,  threshold_bps=1.5
  - ``h09_bps3p0``  horizon_k=9,  threshold_bps=3.0
  - ``h24_bps7p5``  horizon_k=24, threshold_bps=7.5

See docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from intraday_research.baseline_v1 import make_no_trade_band_labels


H03_BPS1P5: dict = {"horizon_k": 3, "threshold_bps": 1.5}
H09_BPS3P0: dict = {"horizon_k": 9, "threshold_bps": 3.0}
H24_BPS7P5: dict = {"horizon_k": 24, "threshold_bps": 7.5}


def build_no_trade_band_labels(
    close: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon_k: int,
    threshold_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap ``baseline_v1.make_no_trade_band_labels`` with a numpy-faced API.

    Args:
        close: 1-D ``float64`` close prices, per-ticker chronological.
        timestamps: 1-D ``datetime64[ns]`` aligned with ``close``, sorted
            ascending (caller responsibility).
        horizon_k: positive integer bars to look ahead.
        threshold_bps: non-negative finite no-trade-band half-width in bps.

    Returns:
        ``(labels, valid_mask)`` where labels is ``int8`` with values in
        ``{0, 1, -1}`` (``-1`` is the invalid placeholder, never to be
        interpreted as class 0), and valid_mask is ``bool_``. Both have
        ``shape == close.shape`` and position-aligned with the inputs.
    """
    close = np.asarray(close)
    timestamps = np.asarray(timestamps)
    if close.ndim != 1 or timestamps.ndim != 1:
        raise ValueError(
            "close and timestamps must be 1-D arrays; got shapes "
            f"{close.shape}, {timestamps.shape}."
        )
    if close.shape != timestamps.shape:
        raise ValueError(
            "close and timestamps must be same length; got "
            f"{close.shape} and {timestamps.shape}."
        )
    if isinstance(horizon_k, bool) or not isinstance(horizon_k, int) or horizon_k <= 0:
        raise ValueError(
            f"horizon_k must be a positive int; got {horizon_k!r}."
        )
    if isinstance(threshold_bps, bool) or not isinstance(threshold_bps, (int, float)):
        raise ValueError(
            f"threshold_bps must be numeric; got {threshold_bps!r}."
        )
    if not math.isfinite(float(threshold_bps)) or threshold_bps < 0:
        raise ValueError(
            "threshold_bps must be non-negative and finite; got "
            f"{threshold_bps!r}."
        )

    n = int(close.shape[0])
    if n == 0:
        return (
            np.array([], dtype=np.int8),
            np.array([], dtype=np.bool_),
        )

    ts = pd.to_datetime(timestamps)
    if ts.isna().any():
        raise ValueError("timestamps contains NaT.")
    if not pd.Series(ts).is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending.")

    frame = pd.DataFrame({
        "ticker": "_anon",
        "timestamp": ts,
        "close": close.astype(float),
    })
    result = make_no_trade_band_labels(
        frame, horizon_k=horizon_k, threshold_bps=threshold_bps,
    )
    label_float = result["label"].to_numpy()
    valid_mask = ~np.isnan(label_float)
    labels = np.where(valid_mask, label_float, -1).astype(np.int8)
    return labels, valid_mask.astype(np.bool_)


def build_h03_bps1p5_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h03_bps1p5 frozen alias (horizon_k=3, threshold_bps=1.5)."""
    return build_no_trade_band_labels(close, timestamps, **H03_BPS1P5)


def build_h09_bps3p0_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h09_bps3p0 frozen alias (horizon_k=9, threshold_bps=3.0)."""
    return build_no_trade_band_labels(close, timestamps, **H09_BPS3P0)


def build_h24_bps7p5_labels(
    close: np.ndarray, timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """h24_bps7p5 frozen alias (horizon_k=24, threshold_bps=7.5)."""
    return build_no_trade_band_labels(close, timestamps, **H24_BPS7P5)
```

- [ ] **Step 1.5: Run the test and verify it PASSES**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py::test_wrapper_matches_baseline_v1_on_identical_inputs -v
```

Expected: `1 passed`.

---

## Task 2: Semantic tests — three invalid reasons + threshold=0

**Files:**
- Modify: `tests/data/test_labels.py` — append tests below

- [ ] **Step 2.1: Append the up-only / down-only / within-band / cross-day / threshold-zero tests**

Append to `tests/data/test_labels.py`:

```python
def test_up_only_drift_produces_label_1_then_invalid_tail():
    """Drift > +1.5bps/bar -> every valid sample labels 1, last 3 invalid."""
    close, timestamps = _synthetic_intraday_session(
        n=20, drift_bps_per_bar=10.0,  # 10 bps drift dominates the noise
    )
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # First 17 samples are valid (h=3 lookahead exists); last 3 invalid.
    assert valid_mask[:17].all()
    assert not valid_mask[17:].any()
    assert (labels[:17] == 1).all()
    assert (labels[17:] == -1).all()


def test_down_only_drift_produces_label_0():
    close, timestamps = _synthetic_intraday_session(
        n=20, drift_bps_per_bar=-10.0,
    )
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert (labels[valid_mask] == 0).all()


def test_within_band_returns_yield_no_trade_band_invalid():
    """Constant close -> future_return == 0 -> within [-1.5bps, +1.5bps] band."""
    n = 20
    timestamps = pd.date_range("2025-01-02 09:30", periods=n, freq="5min").to_numpy()
    close = np.full(n, 100.0, dtype=np.float64)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Every position in [0, 17) should be no-trade-band invalid.
    assert not valid_mask[:17].any()
    assert (labels[:17] == -1).all()


def test_cross_day_horizon_invalidates_late_bars_of_day_one():
    """Last 3 bars of trading day 1 have horizon falling into day 2 -> invalid."""
    day1 = pd.date_range("2025-01-02 09:30", periods=8, freq="5min")
    day2 = pd.date_range("2025-01-03 09:30", periods=8, freq="5min")
    timestamps = np.concatenate([day1.to_numpy(), day2.to_numpy()])
    close = np.linspace(100.0, 110.0, 16).astype(np.float64)  # strong up drift
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # day1 indices 5, 6, 7 (last 3) -> horizon falls into day2 -> invalid.
    assert not valid_mask[5:8].any()
    assert (labels[5:8] == -1).all()
    # day1 indices 0..4 should be valid with up-drift labels.
    assert valid_mask[:5].all()
    assert (labels[:5] == 1).all()


def test_threshold_bps_zero_degenerate_strict_sign_labels():
    """threshold_bps=0.0 mirrors baseline_v1: any sign-deterministic return labels."""
    close, timestamps = _synthetic_intraday_session(
        n=15, drift_bps_per_bar=0.5,  # below 1.5 bps but above 0
    )
    labels, valid_mask = build_no_trade_band_labels(
        close, timestamps, horizon_k=3, threshold_bps=0.0,
    )
    # Every valid sample should be label 1 (up-drift) because threshold=0 leaves
    # no no-trade-band; only strictly-zero returns would be invalid.
    assert (labels[valid_mask] == 1).all()
    assert valid_mask[:12].all()  # first 12 have h=3 lookahead


def test_close_nan_propagates_to_invalid_mask():
    close, timestamps = _synthetic_intraday_session(n=20)
    close[5] = np.nan
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Index 5 (NaN close itself) AND indices 2, 3, 4 (whose t+3 = 5, 6, 7)
    # may all be invalid due to NaN propagation through future_cumulative_return.
    assert not valid_mask[5]
    # The wrapper does not raise; sentinel -1 at invalid positions.
    assert labels[5] == -1
```

- [ ] **Step 2.2: Run the new tests; expect ALL pass**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py -q
```

Expected: `7 passed` (cross-check + 6 semantic). All semantic tests
pass on the first try because the wrapper delegates to baseline_v1 —
their goal is to LOCK the semantics so future refactors don't drift.

---

## Task 3: Output format tests — dtype, shape, sentinel

**Files:**
- Modify: `tests/data/test_labels.py` — append tests below

- [ ] **Step 3.1: Append dtype + shape + sentinel tests**

Append to `tests/data/test_labels.py`:

```python
def test_output_dtypes_and_shapes():
    close, timestamps = _synthetic_intraday_session(n=30, drift_bps_per_bar=3.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert labels.dtype == np.int8
    assert valid_mask.dtype == np.bool_
    assert labels.shape == close.shape
    assert valid_mask.shape == close.shape


def test_sentinel_minus_one_only_at_invalid_positions():
    close, timestamps = _synthetic_intraday_session(n=30, drift_bps_per_bar=3.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    # Invalid positions carry sentinel -1.
    assert (labels[~valid_mask] == -1).all()
    # Valid positions carry only 0 or 1.
    assert set(labels[valid_mask].tolist()).issubset({0, 1})


def test_empty_input_returns_empty_outputs():
    close = np.array([], dtype=np.float64)
    timestamps = np.array([], dtype="datetime64[ns]")
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert labels.shape == (0,)
    assert valid_mask.shape == (0,)
    assert labels.dtype == np.int8
    assert valid_mask.dtype == np.bool_


def test_n_less_than_horizon_plus_one_all_invalid():
    """h=3 needs at least 4 samples for any to be valid; with 3 every row invalid."""
    close, timestamps = _synthetic_intraday_session(n=3, drift_bps_per_bar=5.0)
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    assert not valid_mask.any()
    assert (labels == -1).all()
```

- [ ] **Step 3.2: Run new tests; expect ALL pass**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py -q
```

Expected: `11 passed` total.

---

## Task 4: Input-validation guards (RED→GREEN)

**Files:**
- Modify: `tests/data/test_labels.py` — append guard tests

- [ ] **Step 4.1: Append the input-validation tests**

Append to `tests/data/test_labels.py`:

```python
def test_rejects_non_1d_close():
    close = np.zeros((4, 3), dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    with pytest.raises(ValueError, match="1-D arrays"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_non_1d_timestamps():
    close = np.zeros(12, dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    timestamps_2d = timestamps.reshape(4, 3)
    with pytest.raises(ValueError, match="1-D arrays"):
        build_h03_bps1p5_labels(close, timestamps_2d)


def test_rejects_misaligned_lengths():
    close = np.zeros(10, dtype=np.float64)
    timestamps = pd.date_range("2025-01-02 09:30", periods=12, freq="5min").to_numpy()
    with pytest.raises(ValueError, match="same length"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_unsorted_timestamps():
    close, timestamps = _synthetic_intraday_session(n=10)
    # Swap two adjacent timestamps to break monotonicity.
    timestamps = timestamps.copy()
    timestamps[3], timestamps[5] = timestamps[5], timestamps[3]
    with pytest.raises(ValueError, match="sorted ascending"):
        build_h03_bps1p5_labels(close, timestamps)


def test_rejects_nat_in_timestamps():
    close, timestamps = _synthetic_intraday_session(n=10)
    timestamps = timestamps.copy()
    timestamps[4] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="NaT"):
        build_h03_bps1p5_labels(close, timestamps)


@pytest.mark.parametrize("bad", [0, -1, -5, 1.5, True, False])
def test_rejects_invalid_horizon_k(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="horizon_k must be a positive int"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=bad, threshold_bps=1.5,
        )


@pytest.mark.parametrize("bad", [-0.5, -1.0, math.inf, -math.inf, math.nan])
def test_rejects_invalid_threshold_bps(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="threshold_bps must be non-negative and finite"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=3, threshold_bps=bad,
        )


@pytest.mark.parametrize("bad", ["abc", None, [1.0]])
def test_rejects_non_numeric_threshold_bps(bad):
    close, timestamps = _synthetic_intraday_session(n=20)
    with pytest.raises(ValueError, match="threshold_bps must be numeric"):
        build_no_trade_band_labels(
            close, timestamps, horizon_k=3, threshold_bps=bad,
        )
```

- [ ] **Step 4.2: Run the new validation tests**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py -q
```

Expected: `30 passed` total (= 11 from Task 1–3 + 19 new in Task 4:
5 single-case + 6 parametrized horizon_k + 5 parametrized
threshold_bps + 3 parametrized non-numeric threshold_bps). The
implementation from Task 1.4 already includes the guards, so these are
PASS-from-start; their purpose is to LOCK the error contracts against
future regressions.

---

## Task 5: Frozen alias equivalence tests

**Files:**
- Modify: `tests/data/test_labels.py` — append alias tests

- [ ] **Step 5.1: Append three alias-equivalence tests**

Append to `tests/data/test_labels.py`:

```python
def test_h03_bps1p5_alias_matches_generic_call():
    close, timestamps = _synthetic_intraday_session(n=40, drift_bps_per_bar=2.0)
    via_alias = build_h03_bps1p5_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H03_BPS1P5,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_h09_bps3p0_alias_matches_generic_call():
    # h=9 needs >= 9 same-day bars; use one trading day of 78 bars.
    close, timestamps = _synthetic_intraday_session(n=78, drift_bps_per_bar=4.0)
    via_alias = build_h09_bps3p0_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H09_BPS3P0,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_h24_bps7p5_alias_matches_generic_call():
    close, timestamps = _synthetic_intraday_session(n=78, drift_bps_per_bar=10.0)
    via_alias = build_h24_bps7p5_labels(close, timestamps)
    via_generic = build_no_trade_band_labels(
        close, timestamps, **H24_BPS7P5,
    )
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_frozen_config_constants_match_screening_freeze():
    """Lock the frozen config values from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md."""
    assert H03_BPS1P5 == {"horizon_k": 3, "threshold_bps": 1.5}
    assert H09_BPS3P0 == {"horizon_k": 9, "threshold_bps": 3.0}
    assert H24_BPS7P5 == {"horizon_k": 24, "threshold_bps": 7.5}
```

- [ ] **Step 5.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_labels.py -q
```

Expected: `34 passed` total
(= 1 cross-check + 6 semantic + 4 format + 5 single-case guards +
6 parametrized horizon_k + 5 parametrized threshold_bps +
3 parametrized non-numeric threshold_bps + 4 alias = 34).

---

## Task 6: Three-command verification + commit

**Files:**
- No new files; stages prior changes.

- [ ] **Step 6.1: Run the models-tests gate (no regression in #5A/#5B)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/stages/models -q
```

Expected: `80 passed` (unchanged from #5B baseline).

- [ ] **Step 6.2: Run the N08 face + new data tests**

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

Expected: previous `279 passed` + new 34 = `313 passed`. The gate is
"no failures"; if the actual number differs by a few due to future
parametrize-expansion drift, that is acceptable as long as nothing
fails.

- [ ] **Step 6.3: Run the Resume Gate**

Command:
```bash
bash scripts/check_n08_resume_gate.sh; echo "RESUME_GATE_EXIT=$?"
```

Expected:
```text
GATE PASSED. Substantive N08 work may proceed.
RESUME_GATE_EXIT=0
```

- [ ] **Step 6.4: Inventory the changes**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git status --short
git diff --stat HEAD
```

Expected (untracked):
```text
?? src/intraday_research/data/
?? tests/data/
```

- [ ] **Step 6.5: STOP and report to user for explicit commit authorization**

Before staging or committing, the agent reports:
- The three verification command outputs (verbatim or summarized).
- Files staged.
- Proposed commit message (below).

WAIT for the user's explicit `stage + commit` authorization. Do NOT
proceed without it (AGENTS.md §9).

- [ ] **Step 6.6: Stage files by name**

Command (only after user authorizes):
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git add \
  src/intraday_research/data/__init__.py \
  src/intraday_research/data/labels.py \
  tests/data/test_labels.py
git status --short
git diff --cached --stat
```

Expected: 3 `A` (added) lines, ~400 lines staged total.

- [ ] **Step 6.7: Commit**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git commit -m "$(cat <<'EOF'
feat(n08): implement build_h03_bps1p5_labels in data/labels.py (#5C-1)

First piece of the #5C raw-data pipeline. Adds the
src/intraday_research/data/ subpackage with labels.py: a numpy-faced
wrapper around the frozen baseline_v1.make_no_trade_band_labels that
returns (int8 labels with -1 sentinel, bool_ valid_mask) instead of the
pandas DataFrame with float NaN that baseline_v1 produces.

Three frozen aliases (h03_bps1p5 / h09_bps3p0 / h24_bps7p5) wrap a single
generic build_no_trade_band_labels(close, timestamps, *, horizon_k,
threshold_bps). Frozen alias values are locked to the Stage 0 freeze
(docs/CONFIG_SCREENING_FREEZE_2026-06-04.md) and tested.

Behavior:
  - Wraps baseline_v1 as single source of truth; no reimplementation of
    label semantics.
  - Returns (labels: np.int8 in {0, 1, -1}, valid_mask: np.bool_); -1 is
    an invalid placeholder, never to be interpreted as class 0.
  - Requires sorted timestamps and fails fast on unsorted input
    (no internal reorder/remap).
  - threshold_bps == 0.0 is accepted (mirrors baseline_v1's degenerate
    no-trade-band).
  - Cross-split invalidation is deferred to #5C-4; this module only
    handles the three invalid reasons documented in baseline_v1:
    missing future, cross-day, and within-no-trade-band.

Tests in tests/data/test_labels.py cover the §4 contract on synthetic
data: cross-check equivalence against baseline_v1 (anti-drift gate),
up-only / down-only / within-band / cross-day / threshold=0 semantics,
NaN-in-close propagation, output dtypes/shape, sentinel discipline,
empty and short-input edge cases, three frozen alias equivalence with
the generic, and parametrized input-validation rejections for non-1D /
misaligned / unsorted / NaT / invalid horizon_k / invalid threshold_bps.

No changes to:
  - baseline_v1.py (the source-of-truth label implementation)
  - contract module
  - stage Python module
  - models/deep_sequence/ (controls + folds still as implemented in
    #5A / #5B)
  - notebook content / design doc / configs

Verified:
  - pytest tests/stages/models = 80 passed (no regression)
  - pytest N08 face + tests/data = 313 passed
  - check_n08_resume_gate.{sh,ps1} exits 0; GATE PASSED

Spec: docs/superpowers/specs/2026-06-07-n08-data-labels-h03-bps1p5-design.md
Plan: docs/superpowers/plans/2026-06-07-n08-data-labels-h03-bps1p5-implementation-plan.md
EOF
)"
```

- [ ] **Step 6.8: Post-commit verification**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git log -1 --stat
git status --branch --short
```

Expected: commit SHA shown, working tree clean (no `M`/`R`/`A`/`??`
entries below the branch line), `[ahead N]` indicator on the branch
line showing the new commit is local (not yet pushed).

- [ ] **Step 6.9: Report completion**

Report to user:
- Final commit SHA.
- Test counts before/after.
- Resume Gate state.
- Updated task list (#9 / `#5C-1` → completed).
- Suggest next: push (user authorization required) and / or open #5C-3
  (CSV loader) per the planned 5-piece breakdown.

---

## Pre-Commit Checklist (Task 6 condensed)

Run before authorizing commit (Git Bash on Windows; project Python
path is given explicitly each time — no env-var shorthands):

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest tests/stages/models -q

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
- `313 passed`
- `GATE PASSED`, exit 0
- Two untracked entries: `src/intraday_research/data/` and `tests/data/`

If any one fails, STOP and report. Do NOT attempt to debug under
brainstorming/writing-plans gates without re-engaging the user.

---

## Out of Scope

Explicitly NOT in this plan:

- The other four pieces of #5C: CSV loader (#5C-3), features (#5C-2),
  split markers + cross-split invalidation (#5C-4), window builder
  (#5C-5). Each gets its own spec and plan.
- Editing or refactoring `baseline_v1.py`.
- Real raw bar CSV reads (no `pd.read_csv`).
- Any holdout/test data access.
- Pushing the commit. Push is a separate user-authorized step
  (AGENTS.md §9).
- The `tests/__init__.py` / `tests/data/__init__.py` files (pytest auto-
  discovery handles this; matches `tests/stages/models/` convention).

---

## Known Risks

1. **pandas version drift**: `pd.Series.is_monotonic_increasing` is
   stable but the deprecated `is_monotonic` exists in some 1.x lines.
   The plan uses the modern name; project env pins pandas via
   `requirements.txt`. Risk: low.

2. **datetime64 dtype coercion**: `pd.to_datetime(timestamps)` may
   silently convert object arrays to datetime64. Tests use
   `pd.date_range(...).to_numpy()` which produces `datetime64[ns]`
   directly, so this is exercised. Risk: low.

3. **baseline_v1 ordering**: `baseline_v1.make_no_trade_band_labels`
   internally calls `frame.sort_values("timestamp")`. Because the
   wrapper requires sorted input, the internal sort is a no-op and
   output preserves input order. If `baseline_v1` is later modified to
   change row identity during sort, the wrapper would silently break;
   the cross-check test (Task 1) is the canary.

4. **NaN-in-close propagation depth**: When `close[t] == NaN`, neighboring
   `future_cumulative_return` values can also become NaN. The test in
   Task 2 asserts only the obvious case (`valid_mask[5] == False`); a
   tighter test that pins exact propagation distance is intentionally
   omitted to avoid coupling to baseline_v1 internals. The cross-check
   test catches any drift in this behavior.

---

## Self-Review (skill checklist)

- **Spec coverage**: every spec §4 test category and every error in the
  §3 table maps to a Task 1–5 step. The §10.1-style verification
  commands in the spec map to Task 6.1–6.3. ✓
- **Placeholder scan**: no "TBD" / "TODO" / "appropriate handling" /
  "similar to" / undefined methods. All code blocks contain runnable
  code. ✓
- **Type consistency**: `build_no_trade_band_labels` signature, return
  type, and parameter names are identical in implementation (Task 1.4),
  guard tests (Task 4), alias tests (Task 5). `H03_BPS1P5` dict keys
  match between constant definition (Task 1.4) and frozen-alias test
  (Task 5). ✓

---

## Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-06-07-n08-data-labels-h03-bps1p5-implementation-plan.md`.

Per the user's standing instruction, the plan is NOT auto-executed.
Awaiting explicit user review of this plan and authorization to begin
Task 1. Do not invoke an execution skill before that authorization.
