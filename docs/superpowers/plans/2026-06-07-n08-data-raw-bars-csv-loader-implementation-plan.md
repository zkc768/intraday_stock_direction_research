# N08 #5C-3 — `data/raw_bars.py` CSV Loader Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.
> Execute tasks in order; each task is a self-contained RED→GREEN→VERIFY
> cycle. Do NOT commit until Task 9 — every intermediate verification is
> read-only.
>
> **Shell assumption:** All commands assume **Git Bash on Windows** (the
> project's standard shell). The Task 9 heredoc for `git commit -m`
> requires Git Bash; from PowerShell, write the message to a file and
> use `git commit -F <file>`, or run the PowerShell sibling
> `scripts/check_n08_resume_gate.ps1` for the gate. All other commands
> use the explicit project-Python path
> (`E:/codex_workspace/_envs/py311_shared/python.exe`) and avoid shell
> env-var shorthands (no `$PYTHON`, no `head -1`).

**Goal:** Implement `src/intraday_research/data/raw_bars.py` as a
fail-loud CSV loader that produces a pooled, validated, pre-holdout
5-minute bar DataFrame consumable by #5C-1 labels / #5C-4 splits /
#5C-5 windows. Single commit, ~30 cross-checked tests.

**Architecture:** Single module-level constant `VAL_END = 2017-01-25`
plus one generic function
`load_ticker_bars(manifest: Mapping[str, str | Path], *, val_end=VAL_END)`.
Pre-loop manifest + `val_end` normalization, then a strict 9-step
per-ticker loop, then concat + sort + return pooled DataFrame.

**Tech Stack:** Python 3.11 / numpy / pandas / pytest. Project Python
`E:/codex_workspace/_envs/py311_shared/python.exe`.

**Reference commits:**
- #5C-1 `8ce2829` (labels.py wrapping baseline_v1) — same wrap-as-source-of-truth philosophy.
- #5B `e85b55e` (folds.py rolling-origin) — same fail-loud guard style.
- #5A `0616701` (controls.py LightGBM) — same scope discipline.

**Spec:** `docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md`
(committed in `1ae79c8`).

---

## Files

| Path | Action | Notes |
|---|---|---|
| `src/intraday_research/data/__init__.py` | unchanged | already exists from #5C-1 |
| `src/intraday_research/data/raw_bars.py` | create | 1 constant + 1 function, ~110 lines |
| `tests/data/test_raw_bars.py` | create | ~30 tests across 8 spec §4 categories |

No existing files modified.

---

## Task 1: Scaffold + first happy-path test (RED→GREEN)

**Files:**
- Create: `src/intraday_research/data/raw_bars.py`
- Create: `tests/data/test_raw_bars.py`

- [ ] **Step 1.1: Create the stub `raw_bars.py`**

The stub pre-defines `VAL_END` and the function signature so the test
file in Step 1.2 imports cleanly. RED comes from `NotImplementedError`
at call time, not from `ImportError`.

Write `src/intraday_research/data/raw_bars.py`:

```python
"""5-minute pre-aggregated raw-bar CSV loader for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


VAL_END: pd.Timestamp = pd.Timestamp("2017-01-25")
# Bars at or after VAL_END belong to the closed holdout/test partition
# (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md). The loader REFUSES to
# silently cap or drop; it raises ValueError so any contamination is
# loud, not hidden in dropped rows.


def load_ticker_bars(
    manifest: Mapping[str, str | Path],
    *,
    val_end: str | pd.Timestamp = VAL_END,
) -> pd.DataFrame:
    """See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md.

    Implementation lands in Task 1 step 1.4.
    """
    raise NotImplementedError("load_ticker_bars — Task 1 step 1.4")
```

- [ ] **Step 1.2: Write the happy-path test**

Write `tests/data/test_raw_bars.py`:

```python
"""Behavioral tests for ``intraday_research.data.raw_bars`` (N08 #5C-3).

Synthetic-CSV tests only. No raw bar I/O against real ``data/*.csv``
(those are gitignored), no fixture files committed to the repo, no
official validation, no holdout. Verifies the §4 contract documented
in ``docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday_research.data.raw_bars import VAL_END, load_ticker_bars


def _synthetic_bar_frame(
    n: int = 10,
    start: str = "2010-01-04 09:30:00",
    drift_bps_per_bar: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """5-min OHLCV bars that pass baseline_v1._validated_ohlcv."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="5min")
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close * 0.9995,
        "high": close * 1.0005,
        "low": close * 0.9990,
        "close": close,
        "volume": rng.integers(1000, 10_000, n),
    })


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_single_ticker_happy_path_returns_canonical_columns(tmp_path: Path):
    """1 ticker, 10 rows → pooled DataFrame with canonical column order."""
    path = _write_csv(_synthetic_bar_frame(n=10), tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert list(pooled.columns) == [
        "ticker", "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert len(pooled) == 10
    assert (pooled["ticker"] == "CSCO").all()
    assert str(pooled["timestamp"].dtype) == "datetime64[ns]"
    assert pooled["timestamp"].is_monotonic_increasing
```

- [ ] **Step 1.3: Run the test and verify it FAILS**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py::test_single_ticker_happy_path_returns_canonical_columns -v
```

Expected: `FAILED` with `NotImplementedError("load_ticker_bars — Task 1 step 1.4")`.

- [ ] **Step 1.4: Implement the full body**

Replace the body of `src/intraday_research/data/raw_bars.py` with the
full implementation:

```python
"""5-minute pre-aggregated raw-bar CSV loader for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from intraday_research.baseline_v1 import _validated_ohlcv


VAL_END: pd.Timestamp = pd.Timestamp("2017-01-25")
# Bars at or after VAL_END belong to the closed holdout/test partition
# (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md). The loader REFUSES to
# silently cap or drop; it raises ValueError so any contamination is
# loud, not hidden in dropped rows.

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)
_CANONICAL_COLUMN_ORDER: tuple[str, ...] = (
    "ticker", "timestamp", "open", "high", "low", "close", "volume",
)


def load_ticker_bars(
    manifest: Mapping[str, str | Path],
    *,
    val_end: str | pd.Timestamp = VAL_END,
) -> pd.DataFrame:
    """Return pooled pre-holdout 5-minute bars across all manifest tickers.

    Output columns (exactly, in this order):
        ["ticker", "timestamp", "open", "high", "low", "close", "volume"]

    Output sort: ascending by ``(ticker, timestamp)``.

    Type discipline:
        - ``val_end`` MUST be ``str`` or ``pd.Timestamp``; any other
          type (including ``int``, ``float``, ``datetime.date``) raises
          ``TypeError`` before any file I/O. This closes the
          ``pd.Timestamp(42)`` epoch-ns silent-parse hole.
        - All timestamps (CSV column and ``val_end``) MUST be
          timezone-naive; tz-aware timestamps raise ``ValueError``
          rather than being implicitly converted.

    Raises:
        See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md
        §"Error handling" for the full enumeration of 14 fail-loud modes.
    """
    # ---- Pre-loop A: manifest guards ----
    if not manifest:
        raise ValueError("manifest is empty; cannot load 0 tickers")
    normalized_manifest: list[tuple[str, Path]] = []
    seen_normalized: set[str] = set()
    for raw_ticker, raw_path in manifest.items():
        ticker = str(raw_ticker).strip()
        if not ticker:
            raise ValueError("manifest ticker key empty")
        if ticker in seen_normalized:
            raise ValueError(
                f"duplicate ticker after normalization: {ticker!r}"
            )
        seen_normalized.add(ticker)
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(
                f"manifest[{raw_ticker!r}] must be str or Path; "
                f"got {type(raw_path).__name__}"
            )
        normalized_manifest.append((ticker, Path(raw_path)))

    # ---- Pre-loop B: val_end normalization ----
    if isinstance(val_end, str):
        val_end_ts = pd.Timestamp(val_end)
    elif isinstance(val_end, pd.Timestamp):
        val_end_ts = val_end
    else:
        raise TypeError(
            f"val_end must be str or pd.Timestamp; "
            f"got {type(val_end).__name__}"
        )
    if val_end_ts.tzinfo is not None:
        raise ValueError(
            f"val_end must be timezone-naive; got tz={val_end_ts.tzinfo}"
        )

    # ---- Per-ticker loop ----
    frames: list[pd.DataFrame] = []
    for ticker, path in normalized_manifest:
        # Pre-step: file existence check
        if not path.exists():
            raise FileNotFoundError(f"ticker={ticker} path={path}")

        # Step 1: read CSV with ParserError wrap
        try:
            df = pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"ticker={ticker}: CSV parse failed path={path}"
            ) from exc

        # Step 2: normalize column names + dedup check
        normalized_cols = [str(c).strip().lower() for c in df.columns]
        col_counts: dict[str, int] = {}
        for c in normalized_cols:
            col_counts[c] = col_counts.get(c, 0) + 1
        dup_cols = sorted(c for c, count in col_counts.items() if count > 1)
        if dup_cols:
            raise ValueError(
                f"ticker={ticker}: duplicate column names "
                f"after normalization: {dup_cols}"
            )
        df.columns = normalized_cols

        # Empty CSV check (header-only)
        if len(df) == 0:
            raise ValueError(f"ticker={ticker}: CSV has zero data rows")

        # Step 3: assert required columns present
        missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(
                f"ticker={ticker}: CSV missing columns: {missing}"
            )

        # Step 4: inject ticker column
        df["ticker"] = ticker

        # Step 5: parse timestamp + NaT + tz-aware checks
        try:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="raise"
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"ticker={ticker}: timestamp parse failed: {exc}"
            ) from exc
        if df["timestamp"].isna().any():
            raise ValueError(
                f"ticker={ticker}: timestamp contains NaT"
            )
        if df["timestamp"].dt.tz is not None:
            raise ValueError(
                f"ticker={ticker}: timestamp is tz-aware "
                f"(tz={df['timestamp'].dt.tz}); strip timezone before loading"
            )

        # Step 6: sort within ticker by timestamp
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

        # Step 7: HOLDOUT CLOSURE CHECK (highest priority data check)
        contam_mask = df["timestamp"] >= val_end_ts
        if contam_mask.any():
            n_contam = int(contam_mask.sum())
            n_total = len(df)
            first_bad = df.loc[contam_mask, "timestamp"].iloc[0]
            raise ValueError(
                f"ticker={ticker}: holdout closure violated; "
                f"first contaminated timestamp={first_bad}; "
                f"rows={n_contam}/{n_total}"
            )

        # Step 8: intra-ticker timestamp uniqueness
        dup_ts_mask = df["timestamp"].duplicated()
        if dup_ts_mask.any():
            first_dup = df.loc[dup_ts_mask, "timestamp"].iloc[0]
            raise ValueError(
                f"ticker={ticker}: duplicate timestamp within ticker; "
                f"first duplicate={first_dup}"
            )

        # Step 9: OHLCV sanity + canonical column selection
        try:
            _validated_ohlcv(df)
        except ValueError as exc:
            raise ValueError(f"ticker={ticker}: {exc}") from exc
        df = df[list(_CANONICAL_COLUMN_ORDER)]
        frames.append(df)

    # ---- Post-loop: concat + sort + return ----
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(
        ["ticker", "timestamp"], kind="stable",
    ).reset_index(drop=True)
    return pooled
```

- [ ] **Step 1.5: Run the test and verify it PASSES**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py::test_single_ticker_happy_path_returns_canonical_columns -v
```

Expected: `1 passed`.

---

## Task 2: Multi-ticker pooling tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 2.1: Append 3 multi-ticker pooling tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_three_ticker_pooling_returns_pooled_frame(tmp_path: Path):
    manifest = {}
    for ticker in ("CSCO", "JPM", "KO"):
        path = _write_csv(_synthetic_bar_frame(n=5), tmp_path / f"{ticker}.csv")
        manifest[ticker] = path
    pooled = load_ticker_bars(manifest)
    assert len(pooled) == 15
    assert pooled.groupby("ticker").size().to_dict() == {
        "CSCO": 5, "JPM": 5, "KO": 5,
    }


def test_per_ticker_timestamps_remain_ascending_after_pooling(tmp_path: Path):
    manifest = {
        "CSCO": _write_csv(_synthetic_bar_frame(n=8), tmp_path / "CSCO.csv"),
        "JPM":  _write_csv(_synthetic_bar_frame(n=8), tmp_path / "JPM.csv"),
    }
    pooled = load_ticker_bars(manifest)
    for ticker, group in pooled.groupby("ticker"):
        assert group["timestamp"].is_monotonic_increasing, (
            f"timestamps for {ticker} not ascending"
        )


def test_pooled_sort_is_by_ticker_then_timestamp(tmp_path: Path):
    """Even if manifest order is reversed, output sort is (ticker, timestamp)."""
    manifest = {
        "WMT":  _write_csv(_synthetic_bar_frame(n=3), tmp_path / "WMT.csv"),
        "CSCO": _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv"),
    }
    pooled = load_ticker_bars(manifest)
    # First 3 rows must be CSCO (lexically before WMT), not WMT.
    assert (pooled["ticker"].iloc[:3] == "CSCO").all()
    assert (pooled["ticker"].iloc[3:] == "WMT").all()
```

- [ ] **Step 2.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `4 passed` (1 from Task 1 + 3 new).

---

## Task 3: Holdout closure tests (priority gate)

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 3.1: Append 5 holdout-closure tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_row_at_exact_val_end_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=3, start="2017-01-24 15:50:00")
    # n=3 with 5min freq starts at 15:50, 15:55, 16:00 — but 16:00 is OK
    # because >= val_end check uses val_end=2017-01-25. Force at-val_end row:
    frame.loc[2, "timestamp"] = pd.Timestamp("2017-01-25 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="holdout closure violated"):
        load_ticker_bars({"CSCO": path})


def test_holdout_error_message_carries_contamination_count(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5, start="2017-01-25 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "rows=5/5" in msg
    assert "first contaminated timestamp" in msg


def test_csv_entirely_before_val_end_passes(tmp_path: Path):
    frame = _synthetic_bar_frame(n=10, start="2016-12-01 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert len(pooled) == 10


def test_custom_val_end_string_overrides_default(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5, start="2018-06-01 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path}, val_end="2020-01-02")
    assert len(pooled) == 5


def test_val_end_int_42_raises_type_error_before_file_io(tmp_path: Path):
    """Locks the guard against pd.Timestamp(42) silently parsing to epoch ns.

    val_end=42 must raise TypeError BEFORE any file I/O happens; the
    manifest path is intentionally not even created so a TypeError is
    the only way the test can pass.
    """
    nonexistent = tmp_path / "does_not_exist.csv"
    with pytest.raises(TypeError, match="val_end must be str or pd.Timestamp"):
        load_ticker_bars({"CSCO": nonexistent}, val_end=42)
```

- [ ] **Step 3.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `9 passed` (4 + 5 new).

---

## Task 4: OHLCV validation delegation tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 4.1: Append 4 OHLCV validation tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_high_less_than_low_raises_with_ticker_prefix(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    # Break OHLC: high < low on row 2.
    frame.loc[2, "high"] = 50.0
    frame.loc[2, "low"] = 200.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "high must be >= low" in msg


def test_close_outside_high_low_range_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "close"] = frame.loc[2, "high"] + 10.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_non_positive_price_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "open"] = -1.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_negative_volume_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "volume"] = -100
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})
```

- [ ] **Step 4.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `13 passed` (9 + 4 new).

---

## Task 5: Schema + column-normalization tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 5.1: Append 5 schema + column-normalization tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_missing_volume_column_raises_with_missing_list(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5).drop(columns=["volume"])
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "missing columns" in msg
    assert "'volume'" in msg


def test_missing_timestamp_column_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5).drop(columns=["timestamp"])
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="'timestamp'"):
        load_ticker_bars({"CSCO": path})


def test_unparseable_timestamp_raises_nat_or_parse_error(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame["timestamp"] = "not a date"
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_case_and_whitespace_mixed_headers_succeed(tmp_path: Path):
    """Headers like 'Timestamp', 'OPEN ', 'High' normalize via strip().lower()."""
    frame = _synthetic_bar_frame(n=5)
    frame.columns = ["Timestamp", "OPEN ", " High", "Low", "Close", "VOLUME"]
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    # Output column order is canonical regardless of input casing.
    assert list(pooled.columns) == [
        "ticker", "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert len(pooled) == 5


def test_duplicate_columns_after_normalization_raise(tmp_path: Path):
    """If 'Open' and 'open' both appear, they collide on .lower()."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2010-01-04 09:30", periods=3, freq="5min"),
        "Open": [100.0, 100.5, 101.0],
        "open": [99.0, 99.5, 100.0],  # collides after normalization
        "high": [101.0, 101.5, 102.0],
        "low":  [98.0, 98.5, 99.0],
        "close": [100.0, 100.5, 101.0],
        "volume": [1000, 2000, 3000],
    })
    path = tmp_path / "CSCO.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate column names after normalization"):
        load_ticker_bars({"CSCO": path})
```

- [ ] **Step 5.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `18 passed` (13 + 5 new).

---

## Task 6: File / IO + ParserError wrapping tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 6.1: Append 3 file/IO tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_nonexistent_path_raises_file_not_found_with_ticker_context(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_ticker_bars({"CSCO": missing})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert str(missing) in msg


def test_zero_data_row_csv_raises(tmp_path: Path):
    path = tmp_path / "CSCO.csv"
    path.write_text("timestamp,open,high,low,close,volume\n")
    with pytest.raises(ValueError, match="zero data rows"):
        load_ticker_bars({"CSCO": path})


def test_garbled_csv_wraps_as_value_error_with_chained_cause(tmp_path: Path):
    """ParserError / UnicodeDecodeError → ValueError with __cause__ set."""
    path = tmp_path / "CSCO.csv"
    # Write malformed CSV with inconsistent quoting and binary noise.
    path.write_bytes(
        b'timestamp,open,high,low,close,volume\n'
        b'"unterminated quote,1,2,3,4,5\n'
        b'\x80\x81\x82,more,garbage,1,2,3\n'
    )
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "CSV parse failed" in msg
    # The original exception must be chained via "raise ... from exc".
    assert excinfo.value.__cause__ is not None
    assert isinstance(
        excinfo.value.__cause__,
        (pd.errors.ParserError, UnicodeDecodeError),
    )
```

- [ ] **Step 6.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `21 passed` (18 + 3 new).

---

## Task 7: Manifest guards + intra-ticker dup-timestamp tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 7.1: Append 5 manifest-guard + dup-timestamp tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_empty_manifest_raises():
    with pytest.raises(ValueError, match="manifest is empty"):
        load_ticker_bars({})


def test_empty_ticker_key_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "x.csv")
    with pytest.raises(ValueError, match="manifest ticker key empty"):
        load_ticker_bars({"": path})


def test_duplicate_ticker_after_strip_raises(tmp_path: Path):
    p1 = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "a.csv")
    p2 = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "b.csv")
    with pytest.raises(ValueError, match="duplicate ticker after normalization"):
        load_ticker_bars({"CSCO": p1, " CSCO ": p2})


def test_non_str_or_path_manifest_value_raises_type_error():
    with pytest.raises(TypeError, match="must be str or Path"):
        load_ticker_bars({"CSCO": 42})


def test_duplicate_timestamp_within_ticker_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    # Force a duplicate timestamp at rows 1 and 2.
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="duplicate timestamp within ticker"):
        load_ticker_bars({"CSCO": path})
```

- [ ] **Step 7.2: Run full test file**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `26 passed` (21 + 5 new).

---

## Task 8: Timezone semantics tests

**Files:**
- Modify: `tests/data/test_raw_bars.py` — append tests

- [ ] **Step 8.1: Append 4 timezone-semantics tests**

Append to `tests/data/test_raw_bars.py`:

```python
def test_tz_naive_csv_passes(tmp_path: Path):
    """All timestamps are tz-naive (the normal case)."""
    path = _write_csv(_synthetic_bar_frame(n=5), tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert pooled["timestamp"].dt.tz is None


def test_tz_aware_csv_timestamp_raises(tmp_path: Path):
    """Timestamps with embedded offset (e.g. -05:00) must be rejected."""
    df = pd.DataFrame({
        "timestamp": [
            "2010-06-01 09:30:00-05:00",
            "2010-06-01 09:35:00-05:00",
            "2010-06-01 09:40:00-05:00",
        ],
        "open":   [100.0, 100.5, 101.0],
        "high":   [101.0, 101.5, 102.0],
        "low":    [99.0,  99.5,  100.0],
        "close":  [100.5, 101.0, 101.5],
        "volume": [1000, 2000, 3000],
    })
    path = tmp_path / "CSCO.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="timestamp is tz-aware"):
        load_ticker_bars({"CSCO": path})


def test_tz_aware_val_end_string_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="val_end must be timezone-naive"):
        load_ticker_bars(
            {"CSCO": path},
            val_end="2017-01-25 00:00:00-05:00",
        )


def test_tz_aware_val_end_timestamp_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="val_end must be timezone-naive"):
        load_ticker_bars(
            {"CSCO": path},
            val_end=pd.Timestamp("2017-01-25", tz="UTC"),
        )
```

- [ ] **Step 8.2: Run full test file (final count)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q
```

Expected: `30 passed` (26 + 4 new = final).

---

## Task 9: Three-command verification + STOP + commit

**Files:**
- No new files; stages all prior changes.

- [ ] **Step 9.1: Run the models-tests gate (no regression in #5A/#5B)**

Command:
```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/stages/models -q
```

Expected: `80 passed`.

- [ ] **Step 9.2: Run the N08 face + data tests**

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

Expected: previous `313 passed` (#5C-1 baseline) + 30 new = `343 passed`.

- [ ] **Step 9.3: Run the Resume Gate**

Command:
```bash
bash scripts/check_n08_resume_gate.sh; echo "RESUME_GATE_EXIT=$?"
```

Expected:
```text
GATE PASSED. Substantive N08 work may proceed.
RESUME_GATE_EXIT=0
```

- [ ] **Step 9.4: Inventory the changes**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git status --short
git diff --stat HEAD
```

Expected (untracked):
```text
?? src/intraday_research/data/raw_bars.py
?? tests/data/test_raw_bars.py
```

- [ ] **Step 9.5: STOP and report to user for explicit commit authorization**

Before staging or committing, the agent reports:
- The three verification command outputs.
- Files staged.
- Proposed commit message (Step 9.7 below).

WAIT for the user's explicit `stage + commit` authorization. Do NOT
proceed without it (AGENTS.md §9).

- [ ] **Step 9.6: Stage files by name**

Command (only after user authorizes):
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git add \
  src/intraday_research/data/raw_bars.py \
  tests/data/test_raw_bars.py
git status --short
git diff --cached --stat
```

Expected: 2 `A` (added) lines, ~500 lines staged total.

- [ ] **Step 9.7: Commit**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git commit -m "$(cat <<'EOF'
feat(n08): implement load_ticker_bars in data/raw_bars.py (#5C-3)

Third piece of the #5C raw-data pipeline. Adds
src/intraday_research/data/raw_bars.py: a strict fail-loud loader that
reads 5-min pre-aggregated per-ticker CSVs into a single pooled
DataFrame matching the schema downstream pieces (#5C-1 labels, #5C-4
splits, #5C-5 windows) already expect.

Behavior:
  - Output columns exactly
    ["ticker", "timestamp", "open", "high", "low", "close", "volume"],
    sorted by (ticker, timestamp).
  - VAL_END = pd.Timestamp("2017-01-25") locked from the Stage 0
    freeze; rows at or after val_end raise ValueError rather than
    being silently capped (audit visibility).
  - val_end accepts only str or pd.Timestamp; int / float /
    datetime.date / numpy.datetime64 raise TypeError BEFORE any file
    I/O, closing the pd.Timestamp(42) epoch-ns silent-parse hole.
  - Both CSV timestamps and val_end must be timezone-naive; tz-aware
    raises ValueError rather than implicit conversion.
  - CSV column headers normalized via str.strip().lower(); duplicates
    after normalization raise; extras silently dropped only at the
    canonical-output step (data-structural, not data-content).
  - pd.errors.ParserError / UnicodeDecodeError wrap as ValueError
    with ticker + path context; chained via "raise ... from exc" so
    excinfo.value.__cause__ preserves the original.
  - OHLCV sanity delegated to baseline_v1._validated_ohlcv as single
    source of truth.
  - Intra-ticker duplicate timestamps raise; cross-ticker duplicates
    (different tickers, same bar time) are valid and pass through.

Tests in tests/data/test_raw_bars.py cover the section 4 contract on
synthetic tmp_path CSVs (30 tests across 8 categories): happy path
single + multi-ticker pooling, holdout closure with priority over
OHLCV checks (incl. val_end=42 TypeError lock), OHLCV validation
delegation, schema + column normalization (case-mixed headers,
duplicate-after-normalization), file/IO with ParserError wrap and
__cause__ chain, manifest guards (empty, empty-key, dup-after-strip,
non-str-or-Path, intra-ticker dup-timestamp), and timezone semantics
(tz-naive passes, tz-aware CSV / val_end string / val_end Timestamp
each raise).

No changes to:
  - baseline_v1.py (delegated to as single source of truth)
  - contract module
  - stage Python module
  - models/deep_sequence/ (controls + folds still as implemented in
    #5A / #5B)
  - labels.py (#5C-1 unchanged)
  - notebook content / design doc / configs

Verified:
  - pytest tests/stages/models = 80 passed (no regression)
  - pytest N08 face + tests/data = 343 passed
  - check_n08_resume_gate.{sh,ps1} exits 0; GATE PASSED

Spec: docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md
Plan: docs/superpowers/plans/2026-06-07-n08-data-raw-bars-csv-loader-implementation-plan.md
EOF
)"
```

- [ ] **Step 9.8: Post-commit verification**

Command:
```bash
cd E:/codex_workspace/projects/intraday_stock_direction_research
git log -1 --stat
git status --branch --short
```

Expected: commit SHA shown, working tree clean (no `M`/`R`/`A`/`??`
entries below the branch line), `[ahead N]` indicator showing the new
commit is local (not yet pushed).

- [ ] **Step 9.9: Report completion**

Report to user:
- Final commit SHA.
- Test counts before/after (313 → 343).
- Resume Gate state.
- Updated task list (#11 / `#5C-3` → completed).
- Suggest next: push (user authorization required) and / or open #5C-2
  (Features) per the planned 5-piece breakdown.

---

## Pre-Commit Checklist (Task 9 condensed)

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
- `343 passed`
- `GATE PASSED`, exit 0
- Two untracked entries: `src/intraday_research/data/raw_bars.py` and
  `tests/data/test_raw_bars.py`

If any one fails, STOP and report. Do NOT debug under
brainstorming/writing-plans gates without re-engaging the user.

---

## Out of Scope

Explicitly NOT in this plan:

- The other four pieces of #5C (labels are done in #5C-1; features
  #5C-2, splits #5C-4, window builder #5C-5 are siblings).
- Editing or refactoring `baseline_v1.py` (only consume the existing
  `_validated_ohlcv`).
- `.txt` 1-min raw bar ingestion or 5-min resample logic.
- Project-level `PROJECT_TICKER_FILES` constant or
  `load_project_raw_bars()` no-arg helper.
- RTH (regular-trading-hours) filtering.
- Lower-bound timestamp checks (TRAIN_START enforcement is #5C-4
  territory).
- Pushing the commit (push is a separate user-authorized step per
  AGENTS.md §9).
- `tests/__init__.py` / `tests/data/__init__.py` (pytest auto-discovery
  handles this; matches `tests/data/test_labels.py` convention).

---

## Known Risks

1. **`baseline_v1._validated_ohlcv` is "private" (leading underscore).**
   The plan imports it directly. Justification: it lives in the same
   package, the underscore signals "consumer should know what they're
   doing" not "external use forbidden", and the cross-check test (Task
   1) catches any drift in its return contract. Mitigation: if
   baseline_v1 is later refactored, the cross-check test will fail loud.

2. **pd.read_csv timestamp dtype.** Newer pandas may infer
   `datetime64[ns]` directly from ISO-format strings, skipping the
   `pd.to_datetime` step's tz check. The plan calls `pd.to_datetime`
   explicitly to normalize the path, so this is defensive even when
   pd.read_csv already parsed timestamps. Risk: low.

3. **pd.Timestamp("...-05:00") behavior.** A string like
   `"2017-01-25 00:00:00-05:00"` is parsed by `pd.Timestamp` into a
   tz-aware Timestamp; the loader catches this via `tzinfo is not None`
   in step B. Task 8's `test_tz_aware_val_end_string_raises` locks the
   behavior. Risk: low.

4. **Garbled-CSV test fragility (Step 6.1).** The synthetic
   binary-garbage CSV is designed to trigger `pd.errors.ParserError`
   or `UnicodeDecodeError`. Different pandas versions may pick
   different exception classes; the test accepts either via
   `isinstance(__cause__, (ParserError, UnicodeDecodeError))`. If
   neither is raised, the test FAILS loud rather than silently passing.

5. **`pd.errors.ParserError` chain on older pandas.** Some 1.x
   pandas versions chain `Exception` instead of `ParserError`. The
   project pins a recent pandas in `requirements.txt`; if a future
   pandas downgrade is needed, the test reveals it. Risk: low.

---

## Self-Review (skill checklist, 4-way sync)

Compared against
`docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md`:

**1. Spec coverage**: every spec §4 test category and every spec §3
error mode maps to a task / step in this plan:

| Spec section | Plan task |
|---|---|
| §2 pre-loop A (manifest guards) | Task 1 step 1.4 implementation + Task 7 manifest tests |
| §2 pre-loop B (val_end normalize) | Task 1 step 1.4 + Task 3 val_end tests |
| §2 step 1 (read + ParserError wrap) | Task 1 step 1.4 + Task 6 ParserError test |
| §2 step 2 (column normalize) | Task 1 step 1.4 + Task 5 case-mixed + dup-after-norm |
| §2 step 3 (required columns) | Task 5 missing-column tests |
| §2 step 4 (inject ticker) | covered by all happy-path tests asserting `ticker` column |
| §2 step 5 (timestamp parse + NaT + tz) | Task 5 (NaT) + Task 8 (tz) |
| §2 step 6 (sort) | Task 2 (per-ticker ascending) |
| §2 step 7 (holdout closure) | Task 3 (5 tests) |
| §2 step 8 (intra-ticker dup) | Task 7 dup-ts test |
| §2 step 9 (OHLCV + canonical cols) | Task 4 (4 tests) + Task 1 canonical-column assertion |
| §2 post-loop (concat + sort) | Task 2 pool-sort test |
| §3 errors #1–#14 | All addressed across Tasks 3, 5, 6, 7, 8 |
| §4 categories 1–8 | Tasks 1, 2, 3, 4, 5, 6, 7, 8 (1:1 mapping) |

**2. Placeholder scan**: no "TBD" / "TODO" / "appropriate handling" /
"similar to" / undefined methods. All code blocks contain runnable
code. ✓

**3. Type consistency**: `load_ticker_bars` signature, return type,
and parameter names are identical across Task 1 stub (Step 1.1), Task 1
implementation (Step 1.4), and all test calls (Tasks 2–8). `VAL_END`
type is `pd.Timestamp` in both the implementation and test imports.
`_REQUIRED_COLUMNS` and `_CANONICAL_COLUMN_ORDER` are private to the
implementation and not referenced from tests. ✓

---

## Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-06-07-n08-data-raw-bars-csv-loader-implementation-plan.md`.

Per the user's standing instruction, the plan is NOT auto-executed.
Awaiting explicit user review of this plan and authorization to begin
Task 1. Do not invoke an execution skill before that authorization.
