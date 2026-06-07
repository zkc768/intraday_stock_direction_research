# Design: N08 #5C-4 — `data/splits.py` chronological split markers

**Date**: 2026-06-07
**Scope**: validation_only
**Holdout/test contact**: false (function fail-louds on any timestamp >= val_end)
**Status**: draft, awaiting user review then writing-plans skill handoff

## Goal

Provide a `src/intraday_research/data/splits.py` module that exposes a
pure-numpy function returning per-row partition codes (train /
validation) plus a `valid_mask` that encodes the "label horizon does
not cross a split boundary" constraint from AGENTS.md section 4.1.
Downstream consumers (#5C-5 window builder, the run_stage
orchestrator) AND the returned mask with the
labels/feature/window valid masks to filter samples.

This is piece **#5C-4** of the 5-piece #5C subtask:

| piece | role | status |
|---|---|---|
| #5C-1 Labels | no-trade-band binary labels | committed (`8ce2829`) |
| #5C-3 CSV loader | raw 5-min bar I/O | committed (`e540e68`) |
| #5C-2 Features | `price_volume_time` builder | committed (`ecfbc95`) |
| **#5C-4 Split markers** | chronological partition + cross-split mask | this spec |
| #5C-5 Window builder | `(X, y, timestamps, ticker_ids)` assembly | sibling, separate spec |

## Architecture

New files:

```
src/intraday_research/data/splits.py
```

(`src/intraday_research/data/__init__.py` already exists from #5C-1
and has been updated for #5C-3 / #5C-2; this commit updates it again
to record `splits` arrival.)

`splits.py` exports:

- Two integer-code constants for the two partitions:

  ```python
  PARTITION_TRAIN: np.int8 = np.int8(0)
  PARTITION_VALIDATION: np.int8 = np.int8(1)
  ```

- One frozen module-level boundary constant for the validation start
  (the train-end / validation-start join). `VAL_END` is intentionally
  NOT redefined here; it is imported from `raw_bars.py` to keep a
  single source of truth:

  ```python
  from intraday_research.data.raw_bars import VAL_END

  VALIDATION_START: pd.Timestamp = pd.Timestamp("2013-09-16")
  ```

- One generic function:

  ```python
  def apply_chronological_split(
      timestamps: np.ndarray,
      *,
      validation_start: pd.Timestamp,
      val_end: pd.Timestamp,
      horizon_k: int,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Returns (partition: int8, valid_mask: bool_)."""
  ```

- One Stage 0 frozen alias (parameterless except `horizon_k`):

  ```python
  def apply_stage0_chronological_split(
      timestamps: np.ndarray,
      *,
      horizon_k: int,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Project-frozen Stage 0 alias (validation_start=2013-09-16,
      val_end=2017-01-25)."""
      return apply_chronological_split(
          timestamps,
          validation_start=VALIDATION_START,
          val_end=VAL_END,
          horizon_k=horizon_k,
      )
  ```

**Implementation approach: pure numpy, no `baseline_v1` wrap.**
The cross-split check only depends on `(timestamps, horizon_k,
validation_start, val_end)`; it does not need OHLCV, ticker,
`future_cumulative_return`, or any other column. Wrapping
`baseline_v1.add_split_and_invalidate_boundaries` would require
faking a `future_cumulative_return` column and pulling in DataFrame
plumbing for no semantic gain. Anti-drift is tested by comparing
the partition assignment against `baseline_v1.assign_calendar_split`
on a per-timestamp basis (which is the relevant single-source-of-
truth function — `add_split_and_invalidate_boundaries` adds heavier
DataFrame-coupled boundary invalidation that is out of scope here).

## Data flow

The function performs strict input validation and then a small pure-
numpy compute:

```
Inputs:                                  Internal:                              Outputs:
─────────                                ───────────                            ─────────
timestamps: np.ndarray                   1. Validate validation_start +         partition: np.ndarray
  datetime64[ns], tz-naive,                 val_end:                              dtype int8, shape (n,)
  sorted ascending                          - both must be pd.Timestamp           values in
                                            - both must be tz-naive                {PARTITION_TRAIN=0,
validation_start: pd.Timestamp              - validation_start < val_end           PARTITION_VALIDATION=1}
val_end: pd.Timestamp
                                         2. Validate horizon_k:                 valid_mask: np.ndarray
horizon_k: int (positive)                   - is positive int                    dtype bool_, shape (n,)
                                                                                  True iff:
                                         3. Validate timestamps:                  (a) t + horizon_k < n
                                            - is 1D ndarray                        (label horizon row exists
                                            - dtype is datetime64[ns]              within the array), AND
                                            - is tz-naive                        (b) partition[t] ==
                                            - no NaT                               partition[t+horizon_k]
                                            - sorted ascending                     (label horizon stays in
                                                                                    same partition; no cross-
                                                                                    split contamination).
                                         4. HOLDOUT CLOSURE CHECK (fail-loud):
                                            if (timestamps >= val_end).any():
                                              raise ValueError with first
                                              contaminating timestamp +
                                              n_contaminated / n_total
                                            (mirrors raw_bars.load_ticker_bars
                                            holdout-closure discipline; in
                                            practice this never fires because
                                            raw_bars already rejects those rows,
                                            but #5C-4 is its own safety net.)

                                         5. n=0 short-circuit:
                                            if len(timestamps) == 0:
                                              return (
                                                np.empty((0,), dtype=np.int8),
                                                np.empty((0,), dtype=np.bool_),
                                              )

                                         6. Compute partition (pure numpy):
                                            partition = np.where(
                                              timestamps < validation_start,
                                              PARTITION_TRAIN,
                                              PARTITION_VALIDATION,
                                            ).astype(np.int8)

                                         7. Compute valid_mask:
                                            n = len(timestamps)
                                            valid_mask = np.zeros(n, np.bool_)
                                            if n > horizon_k:
                                              same_partition = (
                                                partition[:n - horizon_k]
                                                == partition[horizon_k:]
                                              )
                                              valid_mask[:n - horizon_k] = (
                                                same_partition
                                              )

                                         8. Return (partition, valid_mask)
```

### Key data-flow guarantees

- **Two-partition encoding**: `partition` only contains `0` or `1`. The
  `outside_defined_calendar` partition does NOT exist as a code because
  the val_end fail-loud check rejects `timestamp >= val_end`, and any
  `timestamp < validation_start` is classified as TRAIN with no lower
  bound (the `train_start = 1998-01-02` boundary from
  `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` is enforced upstream by
  `raw_bars.load_ticker_bars`'s data layout, not here).
- **No cross-day check**: `#5C-4` does NOT enforce
  `no_input_window_crosses_trading_day` or any related cross-day
  constraint. Cross-day label-horizon invalidation lives in
  `#5C-1 labels`; cross-day window-input invalidation lives in
  `#5C-5 windows`. Keeping `#5C-4`'s responsibility narrow ("split
  boundary only") prevents semantic overlap and double-coverage.
- **Output order = input order**: caller must pass timestamps already
  sorted ascending; the wrapper rejects unsorted input rather than
  silently reordering (matches `#5C-1` and `#5C-2` discipline).
- **n=0 returns empty arrays, does NOT raise**: an empty input frame
  is a legitimate edge case (matches `#5C-1` / `#5C-2`).
- **`n <= horizon_k` returns all-False `valid_mask`**: no row has a
  valid label-horizon successor in the array, so every row is
  invalid by definition.

## Error handling

Thirteen canonical failure modes, all fail-loud, all carrying enough
context to debug without re-opening the source frame. Order in the
table matches the data-flow priority: earlier errors short-circuit
later ones.

| # | Trigger | Exception | Message anchors |
|---|---|---|---|
| 1 | `validation_start` is not `pd.Timestamp` | `TypeError` | `"validation_start must be pd.Timestamp; got <type>"` |
| 2 | `validation_start` is tz-aware | `ValueError` | `"validation_start must be timezone-naive; got tz=<tz>"` |
| 3 | `val_end` is not `pd.Timestamp` | `TypeError` | `"val_end must be pd.Timestamp; got <type>"` |
| 4 | `val_end` is tz-aware | `ValueError` | `"val_end must be timezone-naive; got tz=<tz>"` |
| 5 | `validation_start >= val_end` | `ValueError` | `"validation_start ({...}) must be < val_end ({...})"` |
| 6 | `horizon_k` is not an `int` (incl. `bool`) or `<= 0` | `ValueError` | `"horizon_k must be a positive int; got {horizon_k!r}"` |
| 7 | `timestamps` is not a 1-D `np.ndarray` | `ValueError` | `"timestamps must be a 1-D ndarray; got shape <shape>"` |
| 8 | `timestamps.dtype` is not `datetime64[ns]` | `ValueError` | `"timestamps dtype must be datetime64; got <dtype>"` |
| 9 | `timestamps` is tz-aware (`isinstance(arr.dtype, pd.DatetimeTZDtype)`) | `ValueError` | `"timestamps must be timezone-naive; got tz=<tz>"` |
| 10 | `timestamps` contains any `NaT` | `ValueError` | `"timestamps contains NaT"` |
| 11 | `timestamps` is not monotonically non-decreasing | `ValueError` | `"timestamps must be sorted ascending"` |
| 12 | Any `timestamp >= val_end` (holdout closure) | `ValueError` | `"holdout closure violated; first contaminated timestamp=<ts>; rows=<n_contam>/<n_total>"` |
| 13 | `n == 0` (after all schema/dtype/sort checks pass) | NOT an error: returns `(np.empty((0,), int8), np.empty((0,), bool_))` |

**Key error-handling design points**:

- Argument-type errors (#1–#6) raise BEFORE any work on `timestamps`,
  so a bad call shape fails immediately.
- The tz-aware check uses `isinstance(arr.dtype, pd.DatetimeTZDtype)`
  rather than the deprecated `pd.api.types.is_datetime64tz_dtype`,
  matching the fix applied in `#5C-2` (the project's
  `pytest.ini` promotes DeprecationWarnings from `intraday_research.*`
  to errors).
- The val_end fail-loud check (#12) runs AFTER timestamp schema /
  dtype / sort validation, so we know the comparison is well-defined
  before applying it.
- `n=0` is handled by short-circuit AFTER the schema / dtype / sort /
  val_end checks pass. An empty frame still has to clear those gates.

## Testing strategy

Test file: `tests/data/test_splits.py` (matches the
`tests/data/test_*.py` conventions from `#5C-1` / `#5C-2` / `#5C-3`).

All tests use synthetic timestamps via
`pd.date_range("YYYY-MM-DD HH:MM", periods=N, freq="5min").to_numpy()`;
no fixture files committed to the repo, no `data/*.csv` dependency.

### Seven test categories, ~30–35 cases

1. **Cross-check against `baseline_v1.assign_calendar_split` per row**
   (≈2 cases, the anti-drift gate):
   - Build a synthetic timestamps array spanning the train /
     validation join (e.g. `2013-09-15` through `2013-09-17` with
     5-min bars).
   - Apply `apply_chronological_split(...)` and capture
     `partition`.
   - Loop each timestamp through
     `baseline_v1.assign_calendar_split(ts, splits=...)` (where
     `splits` is constructed with the three keys baseline_v1
     expects: `train`, `validation`, and a placeholder
     `closed_holdout_boundary_only` such as
     `(val_end, "2099-01-01")`).
   - Map string partition names back to our int8 codes
     (`"train"` → `0`, `"validation"` → `1`) and assert
     `partition[t]` equals the mapped baseline_v1 value at every row.
   - One case for default `VALIDATION_START` / `VAL_END`; one for a
     custom `validation_start="2015-01-05"` to exercise the
     parameter pass-through.

2. **Partition encoding + boundary precision** (≈4 cases):
   - Timestamp EXACTLY `validation_start` → `PARTITION_VALIDATION`
     (the `<` in step 6 of the data flow makes this validation, not
     train).
   - Timestamp one 5-min bar before `validation_start` →
     `PARTITION_TRAIN`.
   - Timestamp far before `validation_start` → `PARTITION_TRAIN`.
   - Timestamp near (but `<`) `val_end` → `PARTITION_VALIDATION`.

3. **`valid_mask` cross-split semantics** (≈5 cases):
   - All-TRAIN array (every timestamp before `validation_start`) with
     `horizon_k=3`: rows `[0, n-3)` are valid, the last 3 rows are
     invalid (no horizon successor).
   - All-VALIDATION array (every timestamp between `validation_start`
     and `val_end`) with `horizon_k=3`: rows `[0, n-3)` valid, last
     3 invalid.
   - **Mixed array with horizon spanning train→validation**: the
     last `horizon_k` train rows have `valid_mask=False` because
     `partition[t]=TRAIN` but `partition[t+horizon_k]=VALIDATION`.
   - Validation rows near (but `<`) `val_end` with insufficient
     remaining bars: last `horizon_k` rows have `valid_mask=False`
     because their horizon row does not exist.
   - `n <= horizon_k`: every row has `valid_mask=False`.

4. **Stage 0 alias equivalence** (≈3 cases):
   - `apply_stage0_chronological_split(timestamps, horizon_k=3)`
     produces identical `(partition, valid_mask)` to
     `apply_chronological_split(timestamps,
     validation_start=VALIDATION_START, val_end=VAL_END,
     horizon_k=3)`.
   - Same equivalence for `horizon_k=9` and `horizon_k=24` (matching
     the three Stage 0 label configs `h03_bps1p5`, `h09_bps3p0`,
     `h24_bps7p5` even though this function does not care about
     label semantics).

5. **Holdout closure fail-loud** (≈4 cases):
   - One row at exactly `val_end` (e.g.
     `pd.Timestamp("2017-01-25 00:00:00")`) → raises with
     `first contaminated timestamp=2017-01-25` and
     `rows=1/n`.
   - Multiple rows past `val_end` → raises with
     `rows=k/n` where k is the count.
   - All timestamps `< val_end` → passes (no raise).
   - Custom `val_end="2020-01-02"` overrides the default and
     correctly classifies a 2018 timestamp as VALIDATION instead of
     raising.

6. **Wrapper-layer input guards** (≈10 cases, may use parametrize):
   - `validation_start` non-Timestamp (e.g. `"2013-09-16"` string)
     → TypeError.
   - `validation_start` tz-aware → ValueError.
   - `val_end` non-Timestamp → TypeError.
   - `val_end` tz-aware → ValueError.
   - `validation_start >= val_end` → ValueError.
   - `horizon_k <= 0` parametrized `[0, -1, -5]` → ValueError.
   - `horizon_k=True` (bool subtype of int) → ValueError ("must be
     a positive int").
   - `timestamps` not 1-D (e.g. shape `(4, 3)`) → ValueError.
   - `timestamps` dtype `int64` → ValueError ("must be datetime64").
   - `timestamps` tz-aware → ValueError.
   - `timestamps` contains NaT → ValueError.
   - `timestamps` reverse-sorted → ValueError ("must be sorted
     ascending").

7. **Edge cases + frozen constant lock** (≈4 cases):
   - `n=0` timestamps with full validation → returns
     `(np.empty((0,), int8), np.empty((0,), bool_))`; does NOT raise.
   - `n == horizon_k` → every row's `valid_mask=False`
     (`n - horizon_k = 0`, so the inner same-partition slice is
     empty).
   - Constants lock: `PARTITION_TRAIN == np.int8(0)`,
     `PARTITION_VALIDATION == np.int8(1)`,
     `VALIDATION_START == pd.Timestamp("2013-09-16")`.
   - `VAL_END` import from `intraday_research.data.raw_bars`
     resolves and equals `pd.Timestamp("2017-01-25")` (locks the
     re-use of `raw_bars.VAL_END` as single source of truth).

**Estimated size**: 30–35 cases (some parametrized), ~350–450 lines
of test, runtime < 1 second.

## What this spec does NOT cover

- The other pieces of #5C (`#5C-1` labels done, `#5C-3` raw_bars
  done, `#5C-2` features done; `#5C-5` window builder is a sibling).
- Editing or refactoring `baseline_v1.assign_calendar_split` or
  `baseline_v1.add_split_and_invalidate_boundaries`.
- Cross-day boundary invalidation — that is `#5C-1` labels' and
  `#5C-5` windows' responsibility; `#5C-4` is intentionally narrow.
- Train-only scaler fitting or feature scaling — handled by the
  orchestrator using `baseline_v1.fit_train_only_scaler`.
- Pooled multi-ticker handling — caller iterates per-ticker after
  `#5C-3 load_ticker_bars` returns the pooled DataFrame.
- A `train_start` lower bound — anything `< validation_start` is
  TRAIN; the `train_start = 1998-01-02` boundary is enforced
  upstream by raw bar data availability.
- A `closed_holdout_boundary_only` partition code — rows past
  `val_end` are fail-loud rejected; they never reach the partition
  array.
- Pushing the commit — push is a separate user-authorized step.

## Verification commands (post-implementation)

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_splits.py -q

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
- Per-file ≈ 30–35 passed; N08 face goes from 381 to ≈ 414 passed;
  `GATE PASSED` exit 0.

## Open questions

None at the spec stage. Decisions taken during brainstorming:

- Pure-numpy implementation (no wrap of
  `baseline_v1.add_split_and_invalidate_boundaries`); anti-drift
  tested per-row against `baseline_v1.assign_calendar_split`.
- Single-ticker 1-D `timestamps` array in →
  `(partition: int8, valid_mask: bool_)` numpy tuple out (matches
  `#5C-1 labels` interface style).
- Two-partition int8 encoding (`PARTITION_TRAIN=0`,
  `PARTITION_VALIDATION=1`); no `outside_defined_calendar` code
  because the val_end fail-loud check guarantees every kept row
  belongs to one of the two partitions.
- `VAL_END` imported from `raw_bars.py` as single source of truth;
  `VALIDATION_START` defined locally because it is split-specific.
- One generic `apply_chronological_split` plus one Stage 0
  parameterless alias `apply_stage0_chronological_split`; horizon_k
  is a runtime parameter, not part of the alias name (the three
  Stage 0 horizons share the same split boundary).
- val_end fail-loud is the same discipline as
  `raw_bars.load_ticker_bars`, providing a safety net even though
  the loader has already rejected those rows.
- `valid_mask` only encodes cross-split safety; cross-day handling
  stays in `#5C-1 labels` and `#5C-5 windows`.
- Wrapper-layer fail-fast on missing / wrong-dtype / unsorted /
  tz-aware / NaT timestamps and on bad parameter types; `n=0`
  short-circuits AFTER those checks pass.
- Tests use synthetic `pd.date_range` arrays; no fixture files
  committed; no `data/*.csv` dependency.
