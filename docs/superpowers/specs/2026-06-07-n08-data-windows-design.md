# N08 #5C-5 — Sliding-Window Builder (`data/windows.py`) Design

> Status: design approved 2026-06-07. Terminal state of the brainstorming
> skill; next step is the writing-plans skill.
> Pipeline position: #5C-5, the final piece of the #5C data pipeline
> (#5C-1 labels, #5C-2 features, #5C-3 raw_bars, #5C-4 splits).

## 1. Goal & Scope

Provide a **numpy-faced, pure-geometry sliding-window builder** that turns the
per-row package arrays produced by #5C-1/#5C-2/#5C-4 into supervised
`(X, y, ...)` training windows for the N08 deep-sequence models and the
LightGBM control.

Two public functions:

- `build_windows_single_ticker(...)` — the **core** implementation. One ticker
  in, windows out. All window semantics live here.
- `build_windows(...)` — a **thin pooled wrapper**. Groups by `ticker_ids`,
  calls the core per ticker, concatenates, and remaps provenance indices.
  It contains **no second copy of the windowing logic**.

In-scope invariants (AGENTS.md §4.1):

- stride = 1, target = last row of each window (both frozen, not parameters);
- no window crosses a trading-day boundary;
- no window crosses a ticker boundary (core is single-ticker; pooled slices
  per ticker before calling the core);
- the chronological train/validation split boundary is **date-aligned**
  (`validation_start = 2013-09-16 00:00`, #5C-4), therefore every same-day
  window carries a single partition, fully described by `target_partition`.

Out of scope (see §6).

## 2. Interface

```python
# src/intraday_research/data/windows.py

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
    ...

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
    ...
```

### 2.1 Why keyword-only

`partition` / `labels` are both `int8`; `feature_valid_mask` /
`target_valid_mask` are both `bool`. Passing them positionally invites a
silent swap (a swapped mask = wrong drops or label leakage, with **no
exception**). Pushing the `*` after the unambiguous core triple
(`features` float64 2-D, `labels` int8, `timestamps` datetime64) forces the
four confusable arguments to be named. The core triple stays positional
because their dtypes make a swap fail loudly anyway.

### 2.2 Input array contract

| Param | dtype | shape | constraint |
|---|---|---|---|
| `features` | `float64` | `(n, F)`, `F >= 1` | finiteness is **not** re-checked (caller's job via `feature_valid_mask`) |
| `labels` | `int8` | `(n,)` | values in `{0, 1, -1}`; `-1` only where `target_valid_mask` is False |
| `timestamps` | `datetime64[ns]` | `(n,)` | tz-naive; **single-ticker:** nondecreasing |
| `ticker_ids` (pooled) | numeric / str / homogeneous object | `(n,)` | mixed incomparable object dtype → reject |
| `partition` | `int8` | `(n,)` | values in `{0, 1}` |
| `feature_valid_mask` | `bool_` | `(n,)` | — |
| `target_valid_mask` | `bool_` | `(n,)` | where True, `labels` must be in `{0, 1}` |
| `window_size` | `int` (not `bool`) | scalar | `> 0` |

### 2.3 Output schema

`build_windows_single_ticker` returns `dict[str, np.ndarray]`:

| key | dtype | shape | meaning |
|---|---|---|---|
| `X` | `float64` | `(W, window_size, F)` | input windows, column order verbatim from `features` |
| `y` | `int8` | `(W,)` | target label, always in `{0, 1}` |
| `target_partition` | `int8` | `(W,)` | partition of the target (== whole window) |
| `target_timestamps` | `datetime64[ns]` | `(W,)` | timestamp of the target row |
| `target_row_positions` | `int64` | `(W,)` | **positions into THIS call's input arrays** (row space, not window space) |

`build_windows` returns the same five keys **plus**:

| key | dtype | shape | meaning |
|---|---|---|---|
| `target_ticker_ids` | `== ticker_ids.dtype` | `(W,)` | ticker of the target row |

`target_row_positions` (row space, `0..n-1`) is for provenance/debug/ledger
alignment. It is **distinct** from the window-space indices (`0..W-1`) that
`models/deep_sequence/folds.py` consumes; do not conflate them.

## 3. Data Flow

### 3.1 `build_windows_single_ticker`

1. **Validate args** — types, dtypes, shapes/ndim, equal lengths,
   `F >= 1`, `window_size` (`int`, not `bool`, `> 0`). See §4.
2. **Validate order** — `np.diff(timestamps) >= np.timedelta64(0)` all True,
   else `ValueError`.
3. **Label-contract pre-pass (A4)** —
   `bad = target_valid_mask & ~np.isin(labels, np.array([0, 1], np.int8))`;
   if `bad.any()` → `ValueError` (a target-valid row carries a non-binary
   label, i.e. the caller built the mask wrong). Runs **before** windowing so
   it covers every potential target, including rows whose window is later
   dropped.
4. **Empty fast-path** — `n == 0` or `n < window_size` → empty schema (§3.3).
5. **Day key** — `dates = timestamps.astype("datetime64[D]")`
   (floor-to-date; tz-naive so no clock shift; matches baseline_v1's
   `.dt.date` grouping).
6. **Slide** — for `end_pos` in `range(window_size - 1, n)`, with
   `sl = slice(end_pos - window_size + 1, end_pos + 1)`:
   1. **same-day**: `(dates[sl] == dates[end_pos]).all()` else `continue`;
   2. **partition uniform (A1a, defensive)**:
      `(partition[sl] == partition[end_pos]).all()` else `ValueError`. This
      can only fire if the caller's `partition` is not date-aligned; with the
      real chronological split it never fires. Checked **after** same-day and
      **before** the validity filters, so a contract violation is never masked
      by a coincidental skip;
   3. **feature-valid**: `feature_valid_mask[sl].all()` else `continue`;
   4. **target-valid**: `target_valid_mask[end_pos]` else `continue`;
   5. **emit** `features[sl]`, `labels[end_pos]`, `partition[end_pos]`,
      `timestamps[end_pos]`, `end_pos`.
7. **Assemble** — stack emitted rows; cast to the §2.3 dtypes; if nothing was
   emitted, return the empty schema (§3.3) with `F` preserved.

> A day with fewer than `window_size` bars contributes zero windows naturally
> (the loop never forms a full same-day window); this is the general case of
> the §3.1-step-4 fast-path, not a separate code path.

### 3.2 `build_windows` (pooled)

1. **Validate args** — same as core, plus `ticker_ids` length and dtype
   homogeneity. Determining the unique tickers (`np.unique`) on a mixed
   incomparable object array raises `TypeError`; catch and re-raise as
   `ValueError` with a clear message.
2. **Empty fast-path** — `n == 0` → empty schema including
   `target_ticker_ids` with `dtype == ticker_ids.dtype`.
3. **Per ticker** in `np.unique(ticker_ids)` order (sorted, deterministic):
   1. `global_pos = np.where(ticker_ids == ticker)[0]` (ascending global
      index);
   2. slice every input array by `global_pos`;
   3. call `build_windows_single_ticker` on the slices. Wrap any `ValueError`
      it raises to prepend ticker context (mirrors #5C-3's loader-context
      pattern), e.g. `f"ticker {ticker!r}: {exc}"`;
   4. **remap provenance (A3)**:
      `block["target_row_positions"] = global_pos[block["target_row_positions"]]`
      so positions index the **original pooled** arrays;
   5. attach `target_ticker_ids = np.full(W_block, ticker, dtype=ticker_ids.dtype)`.
4. **Concatenate** blocks (per-ticker block order, block-internal ascending by
   target timestamp). If **all** blocks are empty, build the empty schema
   explicitly (never call `np.concatenate([])`).
5. **Return** the six-key pooled dict.

> The pooled wrapper enforces "no cross-ticker window" structurally: each
> ticker's rows are sliced out before the core ever sees them, so the core
> physically cannot form a window spanning two tickers. No extra check needed.

### 3.3 Empty schema (exact dtypes)

```python
{
    "X": np.empty((0, window_size, F), dtype=np.float64),
    "y": np.empty((0,), dtype=np.int8),
    "target_partition": np.empty((0,), dtype=np.int8),
    "target_timestamps": np.empty((0,), dtype="datetime64[ns]"),
    "target_row_positions": np.empty((0,), dtype=np.int64),
    # pooled only:
    "target_ticker_ids": np.empty((0,), dtype=ticker_ids.dtype),
}
```

## 4. Error Modes

| # | Condition | Function | Exception |
|---|---|---|---|
| 1 | any array arg not `np.ndarray` | both | `TypeError` |
| 2 | `features.ndim != 2` or other arrays not 1-D | both | `ValueError` |
| 3 | `features.shape[1] < 1` (`F == 0`) | both | `ValueError` |
| 4 | lengths of `features`/`labels`/`timestamps`/`partition`/masks (and `ticker_ids`) unequal | both | `ValueError` |
| 5 | `features.dtype != float64` | both | `ValueError` |
| 6 | `labels.dtype != int8` | both | `ValueError` |
| 7 | `timestamps.dtype != datetime64[ns]` (object/tz-aware Timestamp arrays included) | both | `ValueError` |
| 8 | `partition.dtype != int8` or values ∉ `{0, 1}` | both | `ValueError` |
| 9 | `feature_valid_mask`/`target_valid_mask` dtype not `bool_` | both | `ValueError` |
| 10 | `window_size` is `bool`, or not `int`, or `<= 0` | both | `TypeError` / `ValueError` |
| 11 | single-ticker timestamps not nondecreasing | core | `ValueError` |
| 12 | label-contract pre-pass: a `target_valid_mask` row has label ∉ `{0, 1}` | both | `ValueError` |
| 13 | same-day window with non-uniform partition (caller's partition not date-aligned) | both | `ValueError` |
| 14 | `ticker_ids` mixed incomparable object dtype | pooled | `ValueError` |

`bool`-before-`int` ordering for #10 reproduces the #5C-4 trap
(`isinstance(True, int) is True`): check `isinstance(window_size, bool)`
**first**.

Not raised on purpose (valid, returns empty or filtered output):
`n == 0`; `n < window_size`; every window filtered by mask/day-boundary;
a ticker block with `< window_size` rows (pooled).

## 5. Testing Strategy

Test file: `tests/data/test_windows.py` (no `__init__.py`, matching
`tests/data/test_*.py`).

### 5.1 Anti-drift cross-check vs `baseline_v1.build_windows_for_segment`

One fixture, fixed RNG seed (no `Math.random`-style nondeterminism):

- single ticker `"TEST"`, `F = 10`, timestamps on a 5-min grid spanning
  **3 trading days** with `> window_size` bars/day;
- `partition` **date-aligned**: day 1 → `0` (train), days 2–3 → `1`
  (validation). (A non-date-aligned fixture would diverge because baseline
  filters by split *before* windowing — see §1.)
- a few `feature_valid_mask=False` rows (warmup-like) and a few
  `target_valid_mask=False` rows; `labels` is `-1` exactly where
  `target_valid_mask` is False, else `{0, 1}`.

Build the baseline DataFrame (minimal columns only — baseline reads just the
`*_scaled` columns, `label`, `timestamp`, `ticker`, `split`):

```python
df["ticker"] = "TEST"
df["timestamp"] = timestamps
df["split"] = np.where(partition == 0, "train", "validation")
df["label"] = np.where(target_valid_mask, labels.astype(float), np.nan)
for i, name in enumerate(feature_names):
    df[f"{name}_scaled"] = np.where(feature_valid_mask, features[:, i], np.nan)
```

Reference = `build_windows_for_segment(df, "train", names, ws)` ⊕
`build_windows_for_segment(df, "validation", names, ws)`, concatenated and
sorted by `target_timestamp`. Ours = `build_windows_single_ticker(...)`,
already in global time order. Compare:

- `X`: **exact** equality (pure copy, no arithmetic);
- `y`: **values** equal (ours `int8` vs baseline `int64` → compare as int);
- `target_timestamps`: exact equality;
- `target_partition`: matches the split each reference window came from.

### 5.2 Unit categories (independent of baseline)

1. **Guards** — all 14 error modes (parametrize where natural;
   `bool` `window_size` trap explicit).
2. **Partition encoding & boundary** — `target_partition` value/dtype; window
   ending exactly on the first validation bar; window ending on the last
   train bar.
3. **valid_mask cross semantics** — feature-invalid anywhere in window drops
   it; target-invalid drops it; `target_valid=True` but the **target row**
   `feature_valid=False` → dropped (window needs the last row's features).
4. **Cross-day** — a window straddling a day boundary is dropped; `ws == 1`
   degenerate windows accepted; per-day short days contribute nothing while
   other days still emit.
5. **Pooled** — pooling concatenation order (per-ticker block, ascending
   within block); no cross-ticker window; `target_row_positions` provenance
   (`pooled_timestamps[result["target_row_positions"]] == result["target_timestamps"]`);
   ticker-context wrapping of a core `ValueError`; `target_ticker_ids` dtype
   passthrough (incl. empty `n == 0`).
6. **Edge** — `n == 0`; `n < window_size`; all windows filtered (empty `X`
   with shape `(0, ws, F)`); partition-uniformity fail-loud (construct a
   non-date-aligned same-day partition → expect `ValueError`).
7. **Locks** — empty-schema exact dtypes/shapes; output-key set;
   `target_row_positions` is `int64`, `y` is `int8`.

Estimated ~45–55 tests. **N08 face projection: ~416 (post-#5C-4) → ~460.**

## 6. Out of Scope

- **No feature scaling.** Windows carry raw `features`; train-only scaling is a
  separate preprocessing concern (AGENTS.md §4.1), applied upstream/within
  folds, never here.
- **No `stride` / `target_position` parameters.** Frozen `stride = 1`,
  `target = last row`. Not search axes for Stage 0.
- **No global chronological re-sort in the pooled wrapper.** It honors
  per-ticker order; cross-ticker global ordering is a downstream concern.
- **No generator / memmap / chunking.** Eager `np.stack`, matching
  `baseline_v1`. The full Stage-0 `X` is multi-GB
  (~`(1.8M, 20, 10)` float64 ≈ 2.9 GB before warmup/mask trimming); callers
  that hit memory pressure run per-ticker/per-split or cast downstream. A
  chunked variant is a future, separate piece if ever needed.
- **No float32.** `X` stays `float64` to anchor the cross-check and match
  `features`; downstream training may cast.

## 7. Dependencies

| Upstream | Provides | Maps to |
|---|---|---|
| #5C-1 `labels.py` | `labels` int8 `{0,1,-1}`, label `valid_mask` | `labels` + a term of `target_valid_mask` |
| #5C-2 `features.py` | `features` float64, `feature_valid_mask` | `features`, `feature_valid_mask` (passed through) |
| #5C-4 `splits.py` | `partition` int8 `{0,1}`, cross-split `valid_mask` | `partition` + a term of `target_valid_mask` |

`target_valid_mask` is assembled **by the caller** as
`label_valid & split_valid & partition_legal`, keeping label-validity and
split-validity out of the `labels` NaN channel (the labels array stays a clean
`{0,1,-1}` categorical).

| Downstream | Consumes |
|---|---|
| `models/deep_sequence/folds.py` | window-space arrays (`target_timestamps`, `target_ticker_ids`) → fold indices in `0..W-1` |
| run_stage / LightGBM control | `X`, `y`, `target_partition` to split train/validation window sets |

## 8. Files

- **Create** `src/intraday_research/data/windows.py`.
- **Create** `tests/data/test_windows.py`.
- **Modify** `src/intraday_research/data/__init__.py` line 7 docstring:
  `"splits, windows arrive in sibling commits #5C-4 / #5C-5."` →
  `"splits #5C-4 chronological partition / windows #5C-5 sliding-window builder."`
  No import re-export added.

## 9. Spec Self-Review

- **Placeholders:** none — every dtype, shape, and branch is concrete.
- **Internal consistency:** interface (§2) ↔ data flow (§3) ↔ error table
  (§4) ↔ tests (§5) cross-checked. The `target_partition`-characterizes-window
  claim (§1) is justified by the date-aligned split and enforced defensively
  (§3.1.6.2, error #13, test 5.2.6).
- **Scope:** single implementation plan; core + thin wrapper, one test file.
- **Ambiguity resolved:** `target_row_positions` is row space (§2.3),
  explicitly distinguished from folds' window space; label check is a
  pre-pass (§3.1.3); pooled order is per-ticker block (§3.2.4); empty dtypes
  are pinned (§3.3).
