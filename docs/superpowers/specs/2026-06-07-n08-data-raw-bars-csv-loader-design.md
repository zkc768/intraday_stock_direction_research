# Design: N08 #5C-3 — `data/raw_bars.py` CSV loader

**Date**: 2026-06-07
**Scope**: validation_only
**Holdout/test contact**: false (the loader is the boundary that REJECTS holdout rows)
**Status**: draft, awaiting user review then writing-plans skill handoff

## Goal

Provide a `src/intraday_research/data/raw_bars.py` module that exposes
a thin generic function `load_ticker_bars(manifest)` returning a
pooled, validated, pre-holdout 5-minute bar DataFrame consumable by
the rest of the N08 #5C pipeline (#5C-1 labels, #5C-2 features,
#5C-4 splits, #5C-5 windows).

This is piece **#5C-3** of the 5-piece #5C subtask:

| piece | role | status |
|---|---|---|
| #5C-1 Labels | no-trade-band binary labels | committed (`8ce2829`) |
| **#5C-3 CSV loader** | raw 5-min bar I/O | this spec |
| #5C-2 Features | `price_volume_time` builder | sibling, separate spec |
| #5C-4 Split markers | calendar split + cross-split invalidation | sibling, separate spec |
| #5C-5 Window builder | `(X, y, timestamps, ticker_ids)` assembly | sibling, separate spec |

## Architecture

New files:

```
src/intraday_research/data/raw_bars.py
```

(`src/intraday_research/data/__init__.py` already exists from #5C-1; no
change.)

`raw_bars.py` exports a single module-level constant and one generic
function — no frozen-aliases pattern this time, because manifests are
caller-constructed at production time, not project-frozen:

```python
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
    """Return pooled pre-holdout 5-minute bars across all manifest tickers.

    Output columns (exactly, in this order):
        ["ticker", "timestamp", "open", "high", "low", "close", "volume"]

    Output sort: ascending by ``(ticker, timestamp)``.

    Type discipline:
        - ``val_end`` MUST be ``str`` or ``pd.Timestamp``; any other type
          (including ``int``, ``float``, ``datetime.date``) raises
          ``TypeError`` before any file I/O.
        - All timestamps (CSV column and ``val_end``) MUST be timezone-naive;
          tz-aware timestamps raise ``ValueError`` rather than being
          implicitly converted.

    Raises:
        ValueError if any loaded row has ``timestamp >= val_end``.
        See "Error handling" for the full enumeration.
    """
```

## Data flow

### Pre-loop normalization (runs once before any file I/O)

```
A. Manifest guard                    if not manifest: raise ValueError
                                     For each ticker key, str(key).strip();
                                     reject "" or duplicates post-strip.
                                     For each value, isinstance(v, (str, Path));
                                     else TypeError.

B. val_end normalization             if isinstance(val_end, str):
                                         val_end = pd.Timestamp(val_end)
                                     elif isinstance(val_end, pd.Timestamp):
                                         pass
                                     else:
                                         raise TypeError(
                                             "val_end must be str or pd.Timestamp;"
                                             f" got {type(val_end).__name__}"
                                         )
                                     if val_end.tzinfo is not None:
                                         raise ValueError(
                                             "val_end must be timezone-naive;"
                                             f" got tz={val_end.tzinfo}"
                                         )
```

### Per-ticker loop

For each `(ticker, path)` pair in `manifest.items()`, the loader runs
this strict 9-step sequence (no step is skipped, no silent fallback):

```
1. Read CSV at path                  try:
                                         df = pd.read_csv(path)
                                     except (pd.errors.ParserError,
                                             UnicodeDecodeError) as exc:
                                         raise ValueError(
                                             f"ticker={ticker}: CSV parse failed"
                                             f" path={path}"
                                         ) from exc
                                     - FileNotFoundError from open() bubbles up
                                       with ticker + path context (raised
                                       BEFORE pd.read_csv in a path.exists()
                                       precheck; see Error handling #6).

2. Normalize column names            df.columns = [
                                         str(col).strip().lower()
                                         for col in df.columns
                                     ]
                                     - if any duplicates after normalization:
                                       raise ValueError listing the ticker
                                       and the colliding column names.

3. Assert required columns           required = {"timestamp", "open", "high",
                                                 "low", "close", "volume"}
                                     - raise ValueError if any are missing
                                       from the normalized column set;
                                       message names the ticker AND the
                                       missing columns (sorted).
                                     - extra columns are allowed and will be
                                       dropped at step 9 below.

4. Inject ticker column              df["ticker"] = ticker

5. Parse timestamp                   df["timestamp"] = pd.to_datetime(
                                         df["timestamp"], errors="raise")
                                     - raise ValueError if any NaT post-parse
                                       (CSV had unparseable strings).
                                     - raise ValueError if the parsed dtype
                                       is tz-aware
                                       (df["timestamp"].dt.tz is not None);
                                       message names the ticker AND the tz.
                                       Caller MUST strip timezone before
                                       passing to the loader.

6. Sort within ticker by timestamp   df.sort_values("timestamp",
                                                    kind="stable")
                                                .reset_index(drop=True)

7. HOLDOUT CLOSURE CHECK             if (df["timestamp"] >= val_end).any():
   (highest-priority research            raise ValueError with:
   boundary; runs before any              - ticker name
   other data-quality check)              - first contaminated timestamp
                                          - n_contaminated / n_total

8. Intra-ticker timestamp            if df["timestamp"].duplicated().any():
   uniqueness                            raise ValueError with ticker + the
                                         first duplicate timestamp.

9. OHLCV sanity + canonical column   baseline_v1._validated_ohlcv(df)
                                     - delegate to the frozen
                                       single-source-of-truth validator;
                                       on failure, re-raise with
                                       f"ticker={ticker}: ..." prefix.
                                     - then select canonical column set
                                       and order:
                                       df = df[[
                                           "ticker", "timestamp", "open",
                                           "high", "low", "close", "volume"
                                       ]]
                                     - extra (non-required) columns are
                                       silently dropped here; this is the
                                       only place the loader is allowed to
                                       drop anything, and it is data-
                                       structural, not data-content.
```

### Post-loop finalization

```
10. Concat all ticker frames         pooled = pd.concat(frames,
                                                        ignore_index=True)

11. Re-sort by (ticker, timestamp)   pooled.sort_values(
                                         ["ticker", "timestamp"],
                                         kind="stable",
                                     ).reset_index(drop=True)

12. Return pooled                    return pooled
```

Step 11's outer sort is redundant in the single-ticker case but matters
when `manifest` has multiple tickers because `pd.concat` preserves input
order, not lexical ticker order.

### Downstream consumption pattern

```python
pooled = load_ticker_bars(manifest)
for ticker, group in pooled.groupby("ticker"):
    close = group["close"].to_numpy()
    timestamps = group["timestamp"].to_numpy()
    labels, valid_mask = build_h03_bps1p5_labels(close, timestamps)
    ...
```

Each consumer that wants per-ticker work iterates via `groupby("ticker")`.
Consumers that pool (fold builder, window builder) read `pooled` directly.

## Error handling

Fourteen canonical failure modes, all fail-loud, all carrying enough
context to debug without re-opening the source CSV. Order in the table
matches the data-flow priority: earlier errors short-circuit later ones.

| # | Trigger | Exception | Message anchors |
|---|---|---|---|
| 1 | `manifest` is an empty mapping | `ValueError` | `"manifest is empty; cannot load 0 tickers"` |
| 2 | Manifest contains an empty ticker key (`""` after `.strip()`) or two ticker keys that collide after `.strip()` | `ValueError` | `"manifest ticker key empty"` / `"duplicate ticker after normalization: <key>"` |
| 3 | Manifest value is not `str` or `Path` | `TypeError` | `"manifest[<ticker>] must be str or Path; got <type>"` |
| 4 | `val_end` is not `str` or `pd.Timestamp` | `TypeError` | `"val_end must be str or pd.Timestamp; got <type>"`. Explicitly: `int` (e.g. `42`), `float`, `datetime.date`, `numpy.datetime64` all rejected here. `pd.Timestamp(42)` would parse to epoch ns silently, which is why this guard is a `TypeError` BEFORE any `pd.Timestamp(...)` call. |
| 5 | `val_end` is a tz-aware `pd.Timestamp` (or a string that parses to one) | `ValueError` | `"val_end must be timezone-naive; got tz=<tz>"` |
| 6 | Path does not exist | `FileNotFoundError` | `"ticker=<ticker> path=<path>"` |
| 7 | CSV is empty (no data rows after header) | `ValueError` | `"ticker=<ticker>: CSV has zero data rows"` |
| 8 | CSV parse fails (corrupted binary, malformed quoting, unicode decode error, etc.) | `ValueError` | `"ticker=<ticker>: CSV parse failed path=<path>"` with `__cause__` set to the original `pd.errors.ParserError` or `UnicodeDecodeError` (use `raise ... from exc`) |
| 9 | Duplicate column names after `.strip().lower()` normalization | `ValueError` | `"ticker=<ticker>: duplicate column names after normalization: <colliding>"` |
| 10 | CSV missing one or more required columns (post-normalization) | `ValueError` | `"ticker=<ticker>: CSV missing columns: <sorted_missing>"` |
| 11 | Timestamp parse fails / produces NaT / parsed dtype is tz-aware | `ValueError` | One of: `"ticker=<ticker>: timestamp parse failed: <original_error>"`, `"ticker=<ticker>: timestamp contains NaT"`, `"ticker=<ticker>: timestamp is tz-aware (tz=<tz>); strip timezone before loading"` |
| 12 | Any timestamp `>= val_end` | `ValueError` | `"ticker=<ticker>: holdout closure violated; first contaminated timestamp=<ts>; rows=<n_contam>/<n_total>"` |
| 13 | Duplicate timestamp within a single ticker | `ValueError` | `"ticker=<ticker>: duplicate timestamp within ticker; first duplicate=<ts>"` |
| 14 | OHLCV sanity failure (delegated to `baseline_v1._validated_ohlcv`) | `ValueError` | `"ticker=<ticker>: <baseline_v1 message>"` |

Manifest-level errors (1–5) are raised before any file I/O. File errors
(6–14) are raised per-ticker in the order tickers appear in `manifest`;
the loader stops at the first failing ticker rather than accumulating.

### Explicitly out of scope for the loader

- Lower-bound timestamp checks (e.g. enforcing `TRAIN_START = 1998-01-02`)
  — that is #5C-4 split-marker territory.
- Minimum row-count or per-ticker sample-sufficiency checks — that is
  #5C-5 window-builder territory.
- RTH (regular-trading-hours) filtering — current local CSVs appear
  pre-trimmed; if a future raw source mixes pre/after-hours rows, add
  an explicit `validate_rth` helper as a separate task (not a silent
  drop in this loader).
- Multi-file / chunked reading for memory — files are ~24 MB each;
  full-file `pd.read_csv` is acceptable.

## Testing strategy

Test file: `tests/data/test_raw_bars.py` (matches the
`tests/data/test_labels.py` convention from #5C-1).

All tests use `tmp_path` plus `pd.DataFrame(...).to_csv(...)` to
synthesize CSVs; no `tests/data/fixtures/` files are checked into the
repo, and no test depends on the gitignored `data/*.csv`.

### Eight test categories, ~30–35 cases

1. **Happy path single ticker** (≈3 cases)
   - Write a 10-row 5-min bar CSV to `tmp_path`; call
     `load_ticker_bars({"CSCO": path})`.
   - Assert column order is exactly
     `["ticker", "timestamp", "open", "high", "low", "close", "volume"]`.
   - Assert length, ticker uniqueness, timestamp dtype is `datetime64[ns]`,
     and timestamps are monotonically increasing within the group.

2. **Multi-ticker pooling** (≈3 cases)
   - Write 3 tmp_path CSVs (CSCO/JPM/KO, 5 rows each).
   - Assert pooled length = 15, `groupby("ticker").size()` equals
     `{"CSCO": 5, "JPM": 5, "KO": 5}`.
   - Assert per-ticker timestamps remain ascending after pooling.

3. **Holdout closure (highest priority)** (≈5 cases)
   - CSV with one row at exactly `2017-01-25 09:30:00` → raises;
     message contains `first contaminated timestamp=2017-01-25`.
   - CSV with rows past `val_end` → raises; message contains
     `n_contaminated`/`n_total`.
   - CSV entirely before `val_end` → passes.
   - Custom `val_end="2020-01-02"` overrides the default and correctly
     accepts/rejects.
   - `val_end=42` (wrong TYPE) raises `TypeError` BEFORE any file I/O.
     Locks the guard against `pd.Timestamp(42)` silently parsing to epoch
     ns.

4. **OHLCV validation delegated to `baseline_v1`** (≈4 cases)
   - `high < low`, `close < low`, non-positive price, negative volume
     each raise; messages carry the ticker prefix and the underlying
     `_validated_ohlcv` text.

5. **Schema and column-normalization errors** (≈5 cases)
   - Missing `volume` column → ValueError lists `missing=['volume']`.
   - Missing `timestamp` column → ValueError lists `missing=['timestamp']`.
   - All-NaT timestamp column (e.g. unparseable strings post-`to_datetime`)
     → ValueError ("timestamp contains NaT").
   - Case-mixed and whitespace-padded headers (e.g.
     `" Timestamp ,Open ,High,Low,Close,Volume"`) → loader normalizes
     via `.strip().lower()` and succeeds; output column order is canonical.
   - Two columns that collide after normalization (e.g. `Open` and
     `open`) → ValueError ("duplicate column names after normalization").

6. **File / IO errors** (≈3 cases)
   - Non-existent path → `FileNotFoundError` with ticker + path.
   - Zero-data-row CSV (header only) → ValueError ("zero data rows").
   - Garbled binary file → ValueError ("CSV parse failed path=…");
     `excinfo.value.__cause__` is a `pd.errors.ParserError` or
     `UnicodeDecodeError` (verify the chained cause exists; the loader
     uses `raise ... from exc`).

7. **Manifest guards** (≈5 cases)
   - Empty manifest `{}` → ValueError.
   - `{"": path}` empty ticker key → ValueError.
   - `{"CSCO": p1, " CSCO ": p2}` strip-collision → ValueError
     ("duplicate ticker after normalization").
   - `manifest={"CSCO": 42}` non-str/Path → TypeError.
   - Duplicate timestamp within a single ticker → ValueError ("duplicate
     timestamp within ticker").

8. **Timezone semantics** (≈4 cases)
   - CSV with all timestamps tz-naive (e.g. `"2010-06-01 09:30:00"`) →
     passes.
   - CSV with all timestamps tz-aware (e.g. `"2010-06-01 09:30:00-05:00"`)
     → ValueError ("timestamp is tz-aware (tz=…); strip timezone before
     loading"); no silent conversion.
   - `val_end="2017-01-25 00:00:00-05:00"` (string that parses to a
     tz-aware Timestamp) → ValueError ("val_end must be timezone-naive").
   - `val_end=pd.Timestamp("2017-01-25", tz="UTC")` → ValueError.
   - Duplicate timestamp within a single ticker → ValueError ("duplicate
     timestamp within ticker").

**Fixture style**: every synthetic CSV is built inline via
`pd.DataFrame(...).to_csv(path, index=False)` with hand-computed expected
behavior so tests are deterministic and self-documenting.

**Estimated size**: 30–35 cases, ~400–500 lines of test, runtime well
under 2 seconds.

## What this spec does NOT cover

- `.txt` 1-minute raw bar support — N08 #5C-3 is CSV-only by user
  decision; the `.txt` ingestion + 5-min aggregation path lives in N02
  generator and is not duplicated here.
- The other four pieces of #5C (features, splits, window builder, or
  the implementation of run_stage) — each gets its own spec.
- Editing or extending `baseline_v1.py` (only consume its existing
  `_validated_ohlcv`).
- Project-level `PROJECT_TICKER_FILES` convenience constant or
  `load_project_raw_bars()` no-arg helper — user explicitly rejected
  adding these to keep the API minimal.
- Pushing the commit — push is a separate user-authorized step.

## Verification commands (post-implementation)

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_raw_bars.py -q

E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  tests/test_package_import.py \
  tests/stages \
  tests/data \
  -q

bash scripts/check_n08_resume_gate.sh
```

Expected:
- Per-file ≈ 30–35 passed; N08 face goes from 313 to ≈ 345 passed;
  `GATE PASSED` exit 0.

## Open questions

None at the spec stage. Decisions taken during brainstorming:

- CSV-only (no `.txt` 1-min ingestion).
- Pooled DataFrame output (caller `groupby("ticker")` for per-ticker work).
- Parameterized `val_end` with project default `VAL_END = 2017-01-25`.
- Single generic `load_ticker_bars(manifest)` (no project-level
  convenience function).
- OHLCV validation reuses `baseline_v1._validated_ohlcv` (single
  source of truth).
- val_end fail-loud takes priority over OHLCV validation in the
  per-ticker step order, because holdout closure is the higher-priority
  research boundary.
- All errors are loud; no warning/skip/silent-drop anywhere.
- Tests are tmp_path synthetic CSV only; no fixture files in repo, no
  dependency on gitignored `data/*.csv`.
- `val_end` accepts only `str` or `pd.Timestamp` (not `int` / `float` /
  `datetime.date` / `numpy.datetime64`); `int` rejection in particular
  closes the `pd.Timestamp(42)` epoch-ns silent-parse hole.
- Both CSV timestamps and `val_end` MUST be timezone-naive; tz-aware
  inputs raise rather than being implicitly converted.
- CSV column headers are normalized via `.strip().lower()` so
  case/whitespace variants of the canonical names succeed; duplicates
  after normalization raise; extra columns are silently dropped at the
  canonical-output step (the only place silent drop is allowed, and it
  is data-structural, not data-content).
- On CSV parse failure (`pd.errors.ParserError` / `UnicodeDecodeError`),
  the loader wraps as `ValueError` with ticker + path context and
  chains the original exception via `raise ... from exc`.
