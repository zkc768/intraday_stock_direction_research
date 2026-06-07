# N08 #5C-2 — `data/features.py` Feature Builder Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.
> Execute tasks in order; each task is a self-contained RED→GREEN→VERIFY
> cycle. Do NOT commit until Task 8 — every intermediate verification is
> read-only.
>
> **Shell assumption:** All commands assume **Git Bash on Windows** (the
> project's standard shell; the `.sh` Resume Gate sibling already depends
> on it). The Task 8 heredoc for `git commit -m` requires Git Bash; from
> PowerShell, write the message to a file and use `git commit -F <file>`,
> or invoke the PowerShell sibling
> `scripts/check_n08_resume_gate.ps1` for the gate. All other commands
> use the explicit project-Python path
> (`E:/codex_workspace/_envs/py311_shared/python.exe`) and avoid shell
> env-var shorthands (no `$PYTHON`, no `head -1`).

**Goal:** Implement `src/intraday_research/data/features.py` as a
numpy-faced wrapper around the frozen
`baseline_v1.add_baseline_v1_features` implementation, returning
`(features, valid_mask)` arrays for the three Stage 0 feature sets
(`price_action_core` 3 cols, `technical_price` 5 cols,
`price_volume_time` 10 cols). Single commit, ~33 cross-checked tests.

**Architecture:** One module-level constant `FEATURE_SETS` mirroring
the frozen feature sets from
`docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`, one generic function
`build_features(frame, *, feature_set)`, and three thin aliases.
Wrapper-layer fail-fast on schema / dtype / tz / NaT / sort; n=0 short-
circuits BEFORE delegating to baseline_v1 (because
`_require_single_ticker_frame` would otherwise raise on `nunique=0`).
Single-ticker uniqueness and OHLCV value sanity (including raw NaN/inf
fail-loud) are delegated to baseline_v1.

**Tech Stack:** Python 3.11 / numpy / pandas / pytest. Project Python
`E:/codex_workspace/_envs/py311_shared/python.exe`.

**Reference commits:**
- #5C-3 `e540e68` (raw_bars.py) — pre-loop guard pattern, fail-loud
  discipline, tmp_path test fixture convention.
- #5C-1 `8ce2829` (labels.py) — same wrap-baseline_v1 philosophy,
  numpy-faced return tuple.
- #5B `e85b55e` (folds.py) / #5A `0616701` (controls.py) — same scope
  discipline.

**Spec:** `docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md`
(committed in `e7426bc`).

---

## Files

| Path | Action | Notes |
|---|---|---|
| `src/intraday_research/data/__init__.py` | modify | add `features` to submodule docstring list |
| `src/intraday_research/data/features.py` | create | 1 constant + 1 generic + 3 aliases, ~100 lines |
| `tests/data/test_features.py` | create | ~33 tests across 8 spec §4 categories |

No `baseline_v1.py` modifications (single source of truth).
`src/intraday_research/data/__init__.py` already exists from #5C-1.

---

## Task 1: Scaffold + first cross-check test (RED→GREEN)

**Files:**
- Create: `src/intraday_research/data/features.py`
- Create: `tests/data/test_features.py`

- [ ] **Step 1.1: Create the stub `features.py`**

The stub pre-defines `FEATURE_SETS` and the function signatures so the
test file in Step 1.2 imports cleanly. RED comes from
`NotImplementedError` at call time.

Write `src/intraday_research/data/features.py`:

```python
"""Numpy-faced feature builder for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


# Frozen feature sets from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md.
# Any change here MUST be accompanied by an update to the freeze
# document; tests in Task 7 lock the values verbatim.
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "price_action_core": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
    ),
    "technical_price": (
        "log_return",
        "high_low_range",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
    ),
    "price_volume_time": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
        "rolling_volatility_20",
        "normalized_volume_20",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
        "time_of_day_sin",
        "time_of_day_cos",
    ),
}


def build_features(
    frame: pd.DataFrame,
    *,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """See docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md.

    Implementation lands in Task 1 step 1.4.
    """
    raise NotImplementedError("build_features — Task 1 step 1.4")


def build_price_action_core_features(frame):
    """price_action_core frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError(
        "build_price_action_core_features — Task 1 step 1.4"
    )


def build_technical_price_features(frame):
    """technical_price frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError(
        "build_technical_price_features — Task 1 step 1.4"
    )


def build_price_volume_time_features(frame):
    """price_volume_time frozen alias stub. Body lands in Task 1 step 1.4."""
    raise NotImplementedError(
        "build_price_volume_time_features — Task 1 step 1.4"
    )
```

- [ ] **Step 1.2: Write the cross-check test (anti-drift gate)**

Write `tests/data/test_features.py`:

```python
"""Behavioral tests for ``intraday_research.data.features`` (N08 #5C-2).

Synthetic-data tests only. No raw bar I/O, no fixture files committed
to the repo, no official validation, no holdout. Verifies the §4
contract documented in
``docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intraday_research.baseline_v1 import add_baseline_v1_features
from intraday_research.data.features import (
    FEATURE_SETS,
    build_features,
    build_price_action_core_features,
    build_price_volume_time_features,
    build_technical_price_features,
)


def _synthetic_intraday_session(
    n: int = 80,
    start: str = "2010-01-04 09:30:00",
    ticker: str = "CSCO",
    drift_bps_per_bar: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """5-min OHLCV bars that pass baseline_v1._validated_ohlcv.

    Default `n=80` is large enough that every feature in price_volume_time
    has warmed up by the end (longest warmup is normalized_macd_hist with
    cumulative EWM 12+26+9 ≈ 47 effective lag).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="5min")
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return pd.DataFrame({
        "ticker": ticker,
        "timestamp": timestamps,
        "open": close * 0.9995,
        "high": close * 1.0005,
        "low": close * 0.9990,
        "close": close,
        "volume": rng.integers(1000, 10_000, n),
    })


def test_price_volume_time_matches_baseline_v1_column_by_column():
    """Anti-drift gate: every feature in price_volume_time matches
    baseline_v1.add_baseline_v1_features value-for-value at every valid row."""
    frame = _synthetic_intraday_session(n=80)
    expected_df = add_baseline_v1_features(frame)
    features, valid_mask = build_features(
        frame, feature_set="price_volume_time"
    )

    cols = FEATURE_SETS["price_volume_time"]
    assert features.shape == (len(frame), len(cols))
    assert valid_mask.shape == (len(frame),)

    # Compare each column at every row; NaNs in expected must coincide
    # with NaNs in features (np.isnan equivalence).
    for col_idx, col_name in enumerate(cols):
        expected_col = expected_df[col_name].to_numpy(dtype=np.float64)
        actual_col = features[:, col_idx]
        # Both NaN at same positions, equal values where both finite.
        np.testing.assert_array_equal(
            np.isnan(expected_col), np.isnan(actual_col),
            err_msg=f"NaN positions differ for column {col_name!r}",
        )
        finite_mask = np.isfinite(expected_col) & np.isfinite(actual_col)
        np.testing.assert_allclose(
            expected_col[finite_mask], actual_col[finite_mask],
            rtol=0.0, atol=0.0,
            err_msg=f"Numeric mismatch for column {col_name!r}",
        )
```

- [ ] **Step 1.3: Run the test and verify it FAILS**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py::test_price_volume_time_matches_baseline_v1_column_by_column -v
```

Expected: `FAILED` with
`NotImplementedError("build_features — Task 1 step 1.4")`. Imports
succeed (stub pre-defines FEATURE_SETS and all four functions).

- [ ] **Step 1.4: Implement the full body**

Replace `src/intraday_research/data/features.py` with the full
implementation:

```python
"""Numpy-faced feature builder for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from intraday_research.baseline_v1 import add_baseline_v1_features


# Frozen feature sets from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md.
# Any change here MUST be accompanied by an update to the freeze
# document; tests in Task 7 lock the values verbatim.
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    "price_action_core": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
    ),
    "technical_price": (
        "log_return",
        "high_low_range",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
    ),
    "price_volume_time": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
        "rolling_volatility_20",
        "normalized_volume_20",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
        "time_of_day_sin",
        "time_of_day_cos",
    ),
}

_REQUIRED_BASE_COLUMNS: frozenset[str] = frozenset({
    "ticker", "timestamp", "open", "high", "low", "close", "volume",
})


def build_features(
    frame: pd.DataFrame,
    *,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap baseline_v1.add_baseline_v1_features with a numpy-faced API.

    Args:
        frame: single-ticker DataFrame with columns
            ``{ticker, timestamp, open, high, low, close, volume}``;
            ``timestamp`` is ``datetime64[ns]``, timezone-naive, and
            sorted ascending.
        feature_set: one of ``"price_action_core"``, ``"technical_price"``,
            or ``"price_volume_time"``.

    Returns:
        ``(features, valid_mask)`` where:
          - ``features`` is ``float64`` shape
            ``(len(frame), len(FEATURE_SETS[feature_set]))``; column
            order verbatim matches ``FEATURE_SETS[feature_set]`` tuple.
          - ``valid_mask`` is ``bool_`` shape ``(len(frame),)``; True
            iff EVERY column at row ``t`` is finite. Reserved for
            derived-feature NaN (warmup, denominator-zero); raw OHLCV
            NaN/inf is rejected fail-loud by ``baseline_v1`` before
            this mask is computed.
    """
    # ---- Step 1: validate feature_set ----
    if not isinstance(feature_set, str):
        raise TypeError(
            f"feature_set must be a str; got {type(feature_set).__name__}"
        )
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {sorted(FEATURE_SETS)}; "
            f"got {feature_set!r}"
        )

    # ---- Step 2: validate frame ----
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"frame must be pd.DataFrame; got {type(frame).__name__}"
        )
    missing = sorted(_REQUIRED_BASE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"frame missing required columns: {missing}"
        )
    ts = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError(
            "frame['timestamp'] must be datetime64; "
            f"got {ts.dtype}"
        )
    # NOTE: do NOT use pd.api.types.is_datetime64tz_dtype(ts). It is
    # deprecated in recent pandas versions and emits a DeprecationWarning;
    # the project's pytest.ini turns Warnings from intraday_research.*
    # into errors, which would break this code at test time. Use the
    # supported isinstance check on pd.DatetimeTZDtype instead.
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"frame['timestamp'] must be timezone-naive; got tz={ts.dt.tz}"
        )
    if ts.isna().any():
        raise ValueError("frame['timestamp'] contains NaT")
    if not ts.is_monotonic_increasing:
        raise ValueError("frame['timestamp'] must be sorted ascending")

    cols = FEATURE_SETS[feature_set]
    k = len(cols)

    # ---- Step 3: n=0 short-circuit (BEFORE delegating to baseline_v1) ----
    # baseline_v1._require_single_ticker_frame would otherwise raise
    # "Expected a single ticker frame." on an empty frame because
    # frame['ticker'].nunique(dropna=True) == 0 != 1. We handle the
    # empty case here so callers get a clean empty-array return.
    if len(frame) == 0:
        return (
            np.empty((0, k), dtype=np.float64),
            np.empty((0,), dtype=np.bool_),
        )

    # ---- Step 4: delegate to baseline_v1 ----
    # Validates single-ticker uniqueness, OHLCV value sanity (high>=low,
    # open/close in [low, high], positive prices, non-negative volume,
    # raw OHLCV NaN/inf fail-loud). Returns the input frame with all 10
    # feature columns appended.
    enriched = add_baseline_v1_features(frame)

    # ---- Step 5: select feature subset in canonical column order ----
    features_df = enriched[list(cols)]

    # ---- Step 6: convert to ndarray ----
    features = features_df.to_numpy(dtype=np.float64)

    # ---- Step 7: compute row-level valid_mask ----
    valid_mask = np.isfinite(features).all(axis=1).astype(np.bool_)

    return features, valid_mask


def build_price_action_core_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """price_action_core frozen alias (3 features)."""
    return build_features(frame, feature_set="price_action_core")


def build_technical_price_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """technical_price frozen alias (5 features)."""
    return build_features(frame, feature_set="technical_price")


def build_price_volume_time_features(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """price_volume_time frozen alias (10 features; Stage 0 default)."""
    return build_features(frame, feature_set="price_volume_time")
```

- [ ] **Step 1.5: Update `data/__init__.py` docstring**

Replace the submodule list in `src/intraday_research/data/__init__.py`:

```python
"""Raw-data ingestion + label/feature/window helpers for the N08 #5C pipeline.

Submodules:
  - ``labels``    no-trade-band binary labels (#5C-1)
  - ``raw_bars``  5-min pre-aggregated per-ticker CSV loader (#5C-3)
  - ``features``  price_volume_time feature builder (#5C-2)
  - ``splits``, ``windows``  arrive in sibling commits #5C-4 / #5C-5.

Validation-only scope (AGENTS.md section 4.1); no holdout/test data is read
by anything in this subpackage.
"""
```

- [ ] **Step 1.6: Run the test and verify it PASSES**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py::test_price_volume_time_matches_baseline_v1_column_by_column -v
```

Expected: `1 passed`.

---

## Task 2: feature_set subset selection + alias tests

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 2.1: Append subset and alias tests**

Append to `tests/data/test_features.py`:

```python
@pytest.mark.parametrize("feature_set,expected_n_cols", [
    ("price_action_core", 3),
    ("technical_price", 5),
    ("price_volume_time", 10),
])
def test_each_feature_set_returns_correct_shape(feature_set, expected_n_cols):
    frame = _synthetic_intraday_session(n=80)
    features, _ = build_features(frame, feature_set=feature_set)
    assert features.shape == (len(frame), expected_n_cols)


@pytest.mark.parametrize("feature_set", [
    "price_action_core", "technical_price", "price_volume_time",
])
def test_column_order_matches_FEATURE_SETS_tuple_verbatim(feature_set):
    """Sanity: cross-check baseline_v1 column-by-column for THIS feature_set."""
    frame = _synthetic_intraday_session(n=80)
    expected_df = add_baseline_v1_features(frame)
    features, _ = build_features(frame, feature_set=feature_set)
    cols = FEATURE_SETS[feature_set]
    for col_idx, col_name in enumerate(cols):
        expected_col = expected_df[col_name].to_numpy(dtype=np.float64)
        actual_col = features[:, col_idx]
        finite_mask = np.isfinite(expected_col) & np.isfinite(actual_col)
        np.testing.assert_allclose(
            expected_col[finite_mask], actual_col[finite_mask],
            rtol=0.0, atol=0.0,
        )


def test_price_action_core_alias_matches_generic():
    frame = _synthetic_intraday_session(n=40)
    via_alias = build_price_action_core_features(frame)
    via_generic = build_features(frame, feature_set="price_action_core")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_technical_price_alias_matches_generic():
    frame = _synthetic_intraday_session(n=80)
    via_alias = build_technical_price_features(frame)
    via_generic = build_features(frame, feature_set="technical_price")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])


def test_price_volume_time_alias_matches_generic():
    frame = _synthetic_intraday_session(n=80)
    via_alias = build_price_volume_time_features(frame)
    via_generic = build_features(frame, feature_set="price_volume_time")
    np.testing.assert_array_equal(via_alias[0], via_generic[0])
    np.testing.assert_array_equal(via_alias[1], via_generic[1])
```

- [ ] **Step 2.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `10 passed` (1 from Task 1 + 3 parametrized shape + 3
parametrized column-order + 3 alias = 10).

---

## Task 3: Warmup qualitative ordering tests

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 3.1: Append warmup-ordering tests**

Append to `tests/data/test_features.py`:

```python
def test_initial_rows_invalid_then_valid_rows_appear():
    """n=80 frame: at least some early rows are invalid (warmup not done)
    AND at least some late rows are valid (warmup completed). We do NOT
    lock specific bar-count thresholds — that depends on baseline_v1
    internals; the cross-check test (Task 1) is the numerical anchor."""
    frame = _synthetic_intraday_session(n=80)
    _, valid_mask = build_features(
        frame, feature_set="price_volume_time"
    )
    assert not valid_mask[0], "row 0 must be invalid (warmup)"
    assert valid_mask.any(), (
        "at least one row should reach valid_mask=True with n=80; "
        "test fixture may be too short"
    )


def test_price_action_core_reaches_valid_no_later_than_technical_price():
    """price_action_core has only same-day rolling/diff features (short
    warmup); technical_price additionally contains rsi_14, bollinger_pctb,
    and the cross-day normalized_macd_hist (longer warmup). So the first
    True position in price_action_core's valid_mask must be <= the first
    True position in technical_price's valid_mask."""
    frame = _synthetic_intraday_session(n=80)
    _, pac_mask = build_features(frame, feature_set="price_action_core")
    _, tp_mask = build_features(frame, feature_set="technical_price")
    pac_first_valid = int(np.argmax(pac_mask)) if pac_mask.any() else None
    tp_first_valid = int(np.argmax(tp_mask)) if tp_mask.any() else None
    assert pac_first_valid is not None
    assert tp_first_valid is not None
    assert pac_first_valid <= tp_first_valid


def test_price_action_core_reaches_valid_no_later_than_price_volume_time():
    """Same invariant vs price_volume_time (which has the most features
    and the longest cumulative warmup chain)."""
    frame = _synthetic_intraday_session(n=80)
    _, pac_mask = build_features(frame, feature_set="price_action_core")
    _, pvt_mask = build_features(frame, feature_set="price_volume_time")
    pac_first_valid = int(np.argmax(pac_mask)) if pac_mask.any() else None
    pvt_first_valid = int(np.argmax(pvt_mask)) if pvt_mask.any() else None
    assert pac_first_valid is not None
    assert pvt_first_valid is not None
    assert pac_first_valid <= pvt_first_valid


def test_valid_mask_eventually_becomes_true_for_each_feature_set():
    """Sanity: with n=80 drift-only bars the test fixture is long enough
    for every feature set to reach at least one valid row."""
    frame = _synthetic_intraday_session(n=80)
    for feature_set in ("price_action_core", "technical_price", "price_volume_time"):
        _, mask = build_features(frame, feature_set=feature_set)
        assert mask.any(), (
            f"feature_set={feature_set!r} produced no valid rows; "
            f"the n=80 fixture may be insufficient"
        )
```

- [ ] **Step 3.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `14 passed` (10 + 4 new).

---

## Task 4: Output dtype + shape tests

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 4.1: Append dtype/shape tests**

Append to `tests/data/test_features.py`:

```python
def test_features_dtype_is_float64():
    frame = _synthetic_intraday_session(n=30)
    features, _ = build_features(frame, feature_set="price_volume_time")
    assert features.dtype == np.float64


def test_valid_mask_dtype_is_bool():
    frame = _synthetic_intraday_session(n=30)
    _, valid_mask = build_features(frame, feature_set="price_volume_time")
    assert valid_mask.dtype == np.bool_


def test_features_and_valid_mask_shapes_align_with_input_length():
    frame = _synthetic_intraday_session(n=37)
    features, valid_mask = build_features(
        frame, feature_set="technical_price"
    )
    assert features.shape == (37, 5)
    assert valid_mask.shape == (37,)
```

- [ ] **Step 4.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `17 passed` (14 + 3 new).

---

## Task 5: Wrapper-layer input guards

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 5.1: Append wrapper-layer guard tests**

Append to `tests/data/test_features.py`:

```python
def test_non_dataframe_input_raises_type_error():
    with pytest.raises(TypeError, match="frame must be pd.DataFrame"):
        build_features({"ticker": "CSCO"}, feature_set="price_volume_time")


def test_missing_timestamp_column_raises_with_list():
    frame = _synthetic_intraday_session(n=10).drop(columns=["timestamp"])
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "missing required columns" in msg
    assert "'timestamp'" in msg


def test_missing_ticker_column_raises_clean_value_error_not_keyerror():
    """Locks the wrapper's required-column check running BEFORE
    baseline_v1._require_single_ticker_frame; without this, a missing
    ticker would surface as a downstream KeyError or the less-helpful
    'Expected a ticker column ...' message."""
    frame = _synthetic_intraday_session(n=10).drop(columns=["ticker"])
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "missing required columns" in msg
    assert "'ticker'" in msg


def test_missing_multiple_required_columns_lists_all_sorted():
    frame = (
        _synthetic_intraday_session(n=10)
        .drop(columns=["high", "volume"])
    )
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="price_volume_time")
    msg = str(excinfo.value)
    assert "'high'" in msg
    assert "'volume'" in msg


def test_timestamp_int_dtype_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame["timestamp"] = np.arange(len(frame), dtype=np.int64)
    with pytest.raises(ValueError, match="must be datetime64"):
        build_features(frame, feature_set="price_volume_time")


def test_tz_aware_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="must be timezone-naive"):
        build_features(frame, feature_set="price_volume_time")


def test_nat_in_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="contains NaT"):
        build_features(frame, feature_set="price_volume_time")


def test_unsorted_timestamp_raises():
    frame = _synthetic_intraday_session(n=10).copy()
    frame = frame.iloc[::-1].reset_index(drop=True)  # reverse order
    with pytest.raises(ValueError, match="must be sorted ascending"):
        build_features(frame, feature_set="price_volume_time")


def test_invalid_feature_set_name_raises_with_valid_choices():
    frame = _synthetic_intraday_session(n=10)
    with pytest.raises(ValueError) as excinfo:
        build_features(frame, feature_set="nonexistent_set")
    msg = str(excinfo.value)
    assert "feature_set must be one of" in msg
    assert "'price_volume_time'" in msg


def test_non_str_feature_set_raises_type_error():
    frame = _synthetic_intraday_session(n=10)
    with pytest.raises(TypeError, match="feature_set must be a str"):
        build_features(frame, feature_set=42)
```

- [ ] **Step 5.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `27 passed` (17 + 10 new).

---

## Task 6: Delegated guards from baseline_v1

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 6.1: Append baseline_v1-delegated guard tests**

Append to `tests/data/test_features.py`:

```python
def test_multi_ticker_frame_raises_via_baseline_v1():
    """baseline_v1._require_single_ticker_frame catches multi-ticker."""
    frame_a = _synthetic_intraday_session(n=10, ticker="CSCO")
    frame_b = _synthetic_intraday_session(n=10, ticker="JPM", start="2010-01-04 10:20:00")
    pooled = pd.concat([frame_a, frame_b], ignore_index=True)
    with pytest.raises(ValueError, match="single ticker frame"):
        build_features(pooled, feature_set="price_volume_time")


def test_ohlc_high_less_than_low_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "high"] = 50.0
    frame.loc[3, "low"] = 200.0
    with pytest.raises(ValueError, match="high must be >= low"):
        build_features(frame, feature_set="price_volume_time")


def test_non_positive_price_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "open"] = -1.0
    with pytest.raises(ValueError):
        build_features(frame, feature_set="price_volume_time")


def test_negative_volume_raises_via_baseline_v1():
    frame = _synthetic_intraday_session(n=10).copy()
    frame.loc[3, "volume"] = -100
    with pytest.raises(ValueError):
        build_features(frame, feature_set="price_volume_time")
```

- [ ] **Step 6.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `31 passed` (27 + 4 new).

---

## Task 7: Edge cases + frozen constant lock

**Files:**
- Modify: `tests/data/test_features.py` — append tests

- [ ] **Step 7.1: Append edge-case + frozen-constant tests**

Append to `tests/data/test_features.py`:

```python
def test_empty_frame_with_full_schema_returns_empty_arrays():
    """n=0 short-circuits BEFORE baseline_v1 delegation; without the
    short-circuit, baseline_v1._require_single_ticker_frame would raise
    'Expected a single ticker frame.' because nunique=0 != 1."""
    empty = _synthetic_intraday_session(n=0)
    features, valid_mask = build_features(
        empty, feature_set="price_volume_time"
    )
    assert features.shape == (0, 10)
    assert valid_mask.shape == (0,)
    assert features.dtype == np.float64
    assert valid_mask.dtype == np.bool_


def test_empty_frame_missing_required_column_still_raises():
    """Schema check runs BEFORE the n=0 short-circuit, so a missing
    column on an empty frame is still a ValueError."""
    empty = _synthetic_intraday_session(n=0).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(empty, feature_set="price_volume_time")


def test_raw_close_nan_is_fail_loud_via_baseline_v1_not_valid_mask():
    """Raw OHLCV NaN/inf does NOT propagate to valid_mask=False; it
    raises via baseline_v1._validated_ohlcv. Locks the
    'raw vs derived NaN' distinction in spec §"Key data-flow guarantees"."""
    frame = _synthetic_intraday_session(n=30).copy()
    frame.loc[5, "close"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_features(frame, feature_set="price_volume_time")


def test_constant_close_produces_bollinger_nan_so_valid_mask_false():
    """With constant close, rolling_std_20 = 0 → Bollinger band width = 0
    → bollinger_denom replaced with NaN → bollinger_pctb = NaN for every
    row. price_action_core (no bollinger) reaches valid_mask=True after
    warmup; technical_price and price_volume_time (both contain
    bollinger_pctb) never reach valid_mask=True. Wrapper does NOT raise;
    the affected rows just record valid_mask=False. This is the canonical
    'derived NaN → valid_mask' case (replaces 'raw close NaN' which raises
    via baseline_v1)."""
    n = 80
    timestamps = pd.date_range("2010-01-04 09:30:00", periods=n, freq="5min")
    frame = pd.DataFrame({
        "ticker": "CSCO",
        "timestamp": timestamps,
        # Constant close = 100.0; slight band on high/low so OHLC sanity
        # passes (high >= low, open/close in [low, high]).
        "open": 100.0,
        "high": 100.05,
        "low": 99.95,
        "close": 100.0,
        "volume": 5_000,
    })

    # price_action_core has no bollinger_pctb → eventually valid.
    _, pac_mask = build_features(frame, feature_set="price_action_core")
    assert pac_mask.any(), (
        "price_action_core (no bollinger_pctb) should reach valid_mask=True "
        "with constant close once warmup completes"
    )

    # technical_price contains bollinger_pctb → NEVER valid with constant close.
    _, tp_mask = build_features(frame, feature_set="technical_price")
    assert not tp_mask.any(), (
        "technical_price (contains bollinger_pctb) should never reach "
        "valid_mask=True under constant close because the Bollinger "
        "band width is 0 → bollinger_pctb = NaN for every row"
    )

    # price_volume_time contains bollinger_pctb → NEVER valid.
    _, pvt_mask = build_features(frame, feature_set="price_volume_time")
    assert not pvt_mask.any()


def test_FEATURE_SETS_locked_to_CONFIG_SCREENING_FREEZE_verbatim():
    """Locks the FEATURE_SETS constant against drift. If the freeze
    document changes, this test fails LOUD and forces a coordinated
    update to both this module and the freeze document."""
    assert FEATURE_SETS == {
        "price_action_core": (
            "log_return",
            "close_to_open_return",
            "high_low_range",
        ),
        "technical_price": (
            "log_return",
            "high_low_range",
            "rsi_14",
            "bollinger_pctb",
            "normalized_macd_hist",
        ),
        "price_volume_time": (
            "log_return",
            "close_to_open_return",
            "high_low_range",
            "rolling_volatility_20",
            "normalized_volume_20",
            "rsi_14",
            "bollinger_pctb",
            "normalized_macd_hist",
            "time_of_day_sin",
            "time_of_day_cos",
        ),
    }
```

- [ ] **Step 7.2: Run full test file (final count)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q
```

Expected: `36 passed` (31 + 5 new = final).

---

## Task 8: Three-command verification + STOP + commit

**Files:**
- No new files; stages prior changes.

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

Expected: previous `345 passed` (#5C-3 baseline) + 36 new = `381 passed`.

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
?? src/intraday_research/data/features.py
?? tests/data/test_features.py
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
  src/intraday_research/data/features.py \
  tests/data/test_features.py
git status --short
git diff --cached --stat
```

Expected: 1 `M` (modified) + 2 `A` (added), ~550 lines staged total.

- [ ] **Step 8.7: Commit**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git commit -m "$(cat <<'EOF'
feat(n08): implement build_features in data/features.py (#5C-2)

Fourth piece of the #5C raw-data pipeline. Adds
src/intraday_research/data/features.py: a numpy-faced wrapper around
the frozen baseline_v1.add_baseline_v1_features that returns
(float64 ndarray (n, k), bool_ valid_mask) for the three Stage 0
feature sets from docs/CONFIG_SCREENING_FREEZE_2026-06-04.md
(price_action_core 3 cols, technical_price 5 cols, price_volume_time
10 cols).

Behavior:
  - Single-ticker DataFrame in -> (features, valid_mask) out.
  - features.dtype == np.float64; valid_mask.dtype == np.bool_;
    both shape (n,) and (n, k) where k = len(FEATURE_SETS[feature_set]).
  - Column order in features verbatim matches FEATURE_SETS[feature_set]
    tuple; FEATURE_SETS mirrors CONFIG_SCREENING_FREEZE 1:1 and is
    locked by test against drift.
  - valid_mask is row-level AND: True iff every selected feature is
    finite at row t. Reserved for DERIVED-feature NaN (warmup,
    denominator-zero); raw OHLCV NaN/inf is rejected fail-loud by
    baseline_v1._validated_ohlcv before this mask is computed.
  - Wrapper-layer fail-fast checks (in order, before delegating):
    feature_set type and name, frame is DataFrame, all 7 required
    base columns present (ticker, timestamp, OHLCV), timestamp dtype
    is datetime64, timestamp tz-naive, no NaT, sorted ascending.
    Missing required columns raises ValueError listing ALL missing
    names sorted (so a missing ticker is a clean ValueError instead
    of a downstream KeyError).
  - n=0 short-circuits AFTER the schema + dtype + sort checks pass,
    BEFORE delegating to baseline_v1; otherwise
    _require_single_ticker_frame would raise on nunique=0 != 1.
  - Single-ticker uniqueness and OHLCV value sanity (high >= low,
    open/close in [low, high], positive prices, non-negative volume,
    raw OHLCV NaN/inf) are delegated to baseline_v1 as the single
    source of truth.

Three frozen aliases (build_price_action_core_features,
build_technical_price_features, build_price_volume_time_features)
wrap the single generic build_features(frame, *, feature_set). The
Stage 0 default (price_volume_time) is the alias most callers want.

Updates src/intraday_research/data/__init__.py docstring to record
that features.py has now arrived (alongside labels and raw_bars).

Tests in tests/data/test_features.py cover the section 4 contract on
synthetic single-ticker DataFrames (36 tests across 8 categories):
cross-check column-by-column against baseline_v1 (anti-drift gate;
authoritative numerical anchor), feature_set shape + column order +
alias equivalence, warmup qualitative ordering (price_action_core
reaches valid no later than technical_price and price_volume_time;
no numerical bar-count threshold is locked here), output dtypes and
shapes, wrapper-layer guards (non-DataFrame, missing ticker/timestamp/
multiple columns, wrong dtype, tz-aware, NaT, unsorted, invalid
feature_set name and type), baseline_v1-delegated guards (multi-ticker,
high<low, non-positive price, negative volume), and edge cases (n=0
short-circuit happy path, n=0 with missing column still raises, raw
close NaN raises via baseline_v1, constant-close derived bollinger
NaN keeps valid_mask=False without raising, FEATURE_SETS locked
verbatim to the freeze document).

No changes to:
  - baseline_v1.py (delegated to as single source of truth)
  - contract module
  - stage Python module
  - models/deep_sequence/ (controls + folds still as implemented in
    #5A / #5B)
  - labels.py (#5C-1) and raw_bars.py (#5C-3) unchanged
  - notebook content / design doc / configs

Verified:
  - pytest tests/stages/models = 80 passed (no regression)
  - pytest N08 face + tests/data = 381 passed
  - check_n08_resume_gate.{sh,ps1} exits 0; GATE PASSED

Spec: docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md
Plan: docs/superpowers/plans/2026-06-07-n08-data-features-price-volume-time-implementation-plan.md
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
- Test counts before/after (345 → 381).
- Resume Gate state.
- Updated task list (#12 / `#5C-2` → completed).
- Suggest next: push (user authorization required) and/or open #5C-4
  (split markers) per the planned 5-piece breakdown.

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
- `381 passed`
- `GATE PASSED`, exit 0
- Three changes: `M` `src/intraday_research/data/__init__.py`, `??` for
  `src/intraday_research/data/features.py` and `tests/data/test_features.py`

If any one fails, STOP and report. Do NOT debug under
brainstorming/writing-plans gates without re-engaging the user.

---

## Out of Scope

Explicitly NOT in this plan:

- The other pieces of #5C (#5C-1 labels done, #5C-3 raw_bars done;
  #5C-4 splits, #5C-5 window builder are siblings).
- Editing or refactoring `baseline_v1.add_baseline_v1_features` or any
  of its helpers (`grouped_rolling`, `grouped_wilder_ewm`,
  `continuous_ewm`, etc.).
- Train-only scaler fitting — that lives in
  `baseline_v1.fit_train_only_scaler` and is called by #5C-4 splits or
  by the orchestrator.
- Pooled multi-ticker handling — caller iterates per-ticker after
  `#5C-3 load_ticker_bars` returns the pooled DataFrame.
- Pushing the commit (push is a separate user-authorized step per
  AGENTS.md §9).
- `tests/__init__.py` / `tests/data/__init__.py` (pytest auto-discovery
  handles this; matches the existing `tests/data/test_labels.py` and
  `tests/data/test_raw_bars.py` conventions).
- Locking numerical bar-count thresholds for warmup completion (the
  cross-check test is the authoritative numerical anchor).

---

## Known Risks

1. **Raw OHLCV NaN vs derived feature NaN — distinct channels.** The
   spec §"Error handling" makes this explicit, but the implementation
   detail matters: raw `close = NaN` (or any OHLCV NaN/inf) is
   rejected fail-loud by `baseline_v1._validated_ohlcv` before
   `valid_mask` is computed. `valid_mask=False` ONLY signals derived-
   feature NaN (warmup, denominator-zero from constant close, etc.).
   Test 7.1 `test_raw_close_nan_is_fail_loud_via_baseline_v1_not_valid_mask`
   and test 7.1 `test_constant_close_produces_bollinger_nan_so_valid_mask_false`
   lock this distinction.

2. **MACD denominator behavior under constant close.** A previous
   draft of the spec claimed `normalized_macd_hist` also becomes NaN
   under constant close. This is INCORRECT: `baseline_v1` computes
   `normalized_macd_hist = (macd - signal) / ema_26.replace(0.0,
   np.nan)`, and a constant `close = 100.0` gives `ema_26 = 100.0`
   (not zero) so the denominator is finite and the result is
   `(0 - 0) / 100 = 0`. Only `bollinger_pctb` becomes NaN under
   constant close (because `rolling_std_20 = 0` → band width = 0 →
   `bollinger_denom.replace(0.0, np.nan)` triggers). The
   constant-close test asserts ONLY the Bollinger behavior; do not
   broaden it to claim MACD also NaN.

3. **Warmup bar counts depend on baseline_v1 internals.** The cross-
   check test (Task 1) is the authoritative numerical anchor. The
   warmup tests in Task 3 deliberately do NOT lock specific bar-count
   thresholds (e.g. "row 22 is first valid for technical_price")
   because those numbers shift if baseline_v1 changes its EWM
   `min_periods` or `_validated_ohlcv` rounding. Qualitative
   ordering invariants are robust.

4. **pandas tz-aware dtype check — avoid the deprecated helper.** The
   implementation uses `isinstance(ts.dtype, pd.DatetimeTZDtype)` to
   detect tz-aware timestamps, NOT
   `pd.api.types.is_datetime64tz_dtype(ts)`. The latter is deprecated
   in recent pandas versions and emits a `DeprecationWarning`; the
   project's `pytest.ini` has
   `filterwarnings = error::Warning:intraday_research\..*`, which
   would promote that warning to a test error attributed to
   `intraday_research.data.features` via the warnings stacklevel
   attribution (same trap that bit #5C-3's `pd.to_datetime` UserWarning,
   caught at implementation time). The `isinstance` check is the
   supported idiom going forward and is warning-free.

5. **`n=0` short-circuit ordering.** The short-circuit MUST run AFTER
   the schema check (so missing columns on empty frames still raise)
   and BEFORE the baseline_v1 delegation (so an empty frame does not
   trip `_require_single_ticker_frame`). The Task 7 tests
   `test_empty_frame_with_full_schema_returns_empty_arrays` and
   `test_empty_frame_missing_required_column_still_raises` lock both
   ordering constraints.

---

## Self-Review (skill checklist, 4-way sync)

Compared against
`docs/superpowers/specs/2026-06-07-n08-data-features-price-volume-time-design.md`:

**1. Spec coverage**: every spec §4 test category and every spec §3
error mode maps to a task / step in this plan:

| Spec section | Plan task |
|---|---|
| §1 architecture (constant + generic + 3 aliases) | Task 1 step 1.4 (implementation) + Task 2 (subset / alias tests) |
| §2 data flow step 1 (feature_set validate) | Task 5 (invalid name / non-str tests) |
| §2 data flow step 2 (frame validate, all 7 required cols + dtype/tz/NaT/sort) | Task 5 (8 wrapper-layer guard tests) |
| §2 data flow step 3 (n=0 short-circuit) | Task 7 (n=0 happy + n=0 missing column) |
| §2 data flow step 4 (delegate to baseline_v1) | Task 6 (multi-ticker / OHLC / non-positive / negative volume) |
| §2 data flow step 5 (column subset selection) | Task 2 (parametrized shape + column order) |
| §2 data flow step 6 (to_numpy float64) | Task 4 (dtype tests) |
| §2 data flow step 7 (valid_mask = isfinite.all axis=1) | Task 1 (cross-check) + Task 4 + Task 7 |
| §2 data flow step 8 (return) | covered by every test |
| §3 errors (14 modes) | All addressed across Tasks 5, 6, 7 |
| §4 categories 1–8 | Tasks 1, 2, 3, 4, 5, 6, 7 (1:1 mapping) |

**2. Placeholder scan**: no "TBD" / "TODO" / "appropriate handling" /
"similar to" / undefined methods. All code blocks contain runnable
code. ✓

**3. Type consistency**: `build_features` signature, return type, and
parameter names are identical across Task 1 stub (Step 1.1), Task 1
implementation (Step 1.4), and all test calls (Tasks 2–7). `FEATURE_SETS`
type (`Mapping[str, tuple[str, ...]]`) is consistent. `_REQUIRED_BASE_COLUMNS`
private constant matches the spec's "required base columns" list
exactly. ✓

---

## Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-06-07-n08-data-features-price-volume-time-implementation-plan.md`.

Per the user's standing instruction, the plan is NOT auto-executed.
Awaiting explicit user review of this plan and authorization to begin
Task 1. Do not invoke an execution skill before that authorization.
