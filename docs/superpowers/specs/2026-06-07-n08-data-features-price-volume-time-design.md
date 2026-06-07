# Design: N08 #5C-2 — `data/features.py` price_volume_time feature builder

**Date**: 2026-06-07
**Scope**: validation_only
**Holdout/test contact**: false
**Status**: draft, awaiting user review then writing-plans skill handoff

## Goal

Provide a `src/intraday_research/data/features.py` module that exposes a
numpy-faced wrapper around the frozen `baseline_v1.add_baseline_v1_features`
implementation, returning `(features, valid_mask)` arrays consumable by
#5C-5 window builder and `LastStepLightGBMControl` (#5A).

This is piece **#5C-2** of the 5-piece #5C subtask:

| piece | role | status |
|---|---|---|
| #5C-1 Labels | no-trade-band binary labels | committed (`8ce2829`) |
| #5C-3 CSV loader | raw 5-min bar I/O | committed (`e540e68`) |
| **#5C-2 Features** | `price_volume_time` builder | this spec |
| #5C-4 Split markers | calendar split + cross-split invalidation | sibling, separate spec |
| #5C-5 Window builder | `(X, y, timestamps, ticker_ids)` assembly | sibling, separate spec |

## Architecture

New files:

```
src/intraday_research/data/features.py
```

(`src/intraday_research/data/__init__.py` already exists from #5C-1; the
docstring will be updated to record `features` arrival in the same
implementation commit.)

`features.py` exports:

- One module-level constant mirroring the three frozen feature sets
  from `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`:

  ```python
  FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
      "price_action_core": (
          "log_return", "close_to_open_return", "high_low_range",
      ),
      "technical_price": (
          "log_return", "high_low_range", "rsi_14",
          "bollinger_pctb", "normalized_macd_hist",
      ),
      "price_volume_time": (
          "log_return", "close_to_open_return", "high_low_range",
          "rolling_volatility_20", "normalized_volume_20",
          "rsi_14", "bollinger_pctb", "normalized_macd_hist",
          "time_of_day_sin", "time_of_day_cos",
      ),
  }
  ```

- One generic function:

  ```python
  def build_features(
      frame: pd.DataFrame,
      *,
      feature_set: str,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Return (features, valid_mask)."""
  ```

- Three frozen aliases (thin one-liners around the generic):

  ```python
  def build_price_action_core_features(frame) -> ...    # 3 cols
  def build_technical_price_features(frame) -> ...      # 5 cols
  def build_price_volume_time_features(frame) -> ...    # 10 cols (Stage 0 frozen default)
  ```

Internally `build_features` wraps
`baseline_v1.add_baseline_v1_features` so the frozen Stage 0 feature
semantics are the single source of truth.

## Data flow

```
Inputs:                                       Internal:                            Outputs:
─────────                                     ───────────                          ─────────
frame: pd.DataFrame                           1. Validate feature_set name        features:
  columns:                                       (must be in FEATURE_SETS)          np.ndarray, float64,
    ticker (single ticker only)                                                     shape (n, k)
    timestamp (datetime64[ns],                2. Validate frame inputs:             where k = len(
                tz-naive, sorted ascending)      - is pd.DataFrame                  FEATURE_SETS[feature_set])
    open, high, low, close, volume              - timestamp column present          column order matches the
                                                 - timestamp dtype is               FEATURE_SETS[feature_set]
feature_set: str                                  datetime64                        tuple order verbatim
  one of:                                       - timestamp tz is None
    "price_action_core"  (3 cols)               - timestamp has no NaT            valid_mask:
    "technical_price"    (5 cols)               - timestamp is sorted               np.ndarray, bool_,
    "price_volume_time"  (10 cols)                ascending                         shape (n,)
                                                 - ALL required base columns       True iff ALL k features
                                                   present: ticker, timestamp,     at row t are finite
                                                   open, high, low, close,         (np.isfinite(features)
                                                   volume                          .all(axis=1))

                                              3. n=0 short-circuit:
                                                 if len(frame) == 0:
                                                     return (
                                                         np.empty((0, k), np.float64),
                                                         np.empty((0,), np.bool_),
                                                     )
                                                 # MUST happen before delegating
                                                 # because baseline_v1's
                                                 # _require_single_ticker_frame
                                                 # raises on n=0 (nunique=0 != 1).

                                              4. Delegate to baseline_v1:
                                                 enriched = (
                                                   baseline_v1.
                                                   add_baseline_v1_features(
                                                     frame
                                                   )
                                                 )
                                                 # baseline_v1 validates
                                                 # single-ticker via
                                                 # _require_single_ticker_frame
                                                 # and OHLCV via _validated_ohlcv
                                                 # (incl. raw NaN/inf fail-loud);
                                                 # we do not duplicate those.

                                              5. Select feature subset in
                                                 canonical order:
                                                 cols = FEATURE_SETS[feature_set]
                                                 features_df = enriched[list(cols)]

                                              6. Convert to ndarray:
                                                 features = features_df.to_numpy(
                                                     dtype=np.float64,
                                                 )

                                              7. Compute valid_mask:
                                                 valid_mask = (
                                                     np.isfinite(features)
                                                     .all(axis=1)
                                                 ).astype(np.bool_)

                                              8. Return (features, valid_mask)
```

### Key data-flow guarantees

- **Output order = input order**: caller must pass timestamps already
  sorted ascending; the wrapper rejects unsorted input rather than
  silently reordering (matches #5C-1 discipline). This avoids the
  ambiguity from baseline_v1's internal `sort_values("timestamp")`.
- **Column order = FEATURE_SETS[feature_set] tuple verbatim**: the
  numpy column at index `i` corresponds to feature
  `FEATURE_SETS[feature_set][i]`. Downstream consumers (#5C-5 window
  builder, LightGBM control) can rely on a stable index.
- **Row-level valid_mask, not per-feature**: a row is "valid" iff
  every selected feature is finite at that row. The caller does
  `features[valid_mask]` to get a clean numerical matrix. The
  `valid_mask` reflects DERIVED-feature NaN only (warmup periods,
  denominator-zero divisions, etc.); raw OHLCV NaN/inf is rejected
  fail-loud by `baseline_v1._validated_ohlcv` BEFORE this mask is
  computed, so `valid_mask=False` never means "raw input was corrupt".
- **n=0 short-circuits AFTER schema check**: an empty input frame is
  a legitimate edge case (e.g. a ticker with zero bars after
  filtering elsewhere). The wrapper still enforces the full required-
  column schema (`ticker`, `timestamp`, `open`, `high`, `low`,
  `close`, `volume`) and the `timestamp` dtype check, but then
  short-circuits to
  `(np.empty((0, k), float64), np.empty((0,), bool_))` WITHOUT
  delegating to `baseline_v1.add_baseline_v1_features`. Delegation
  cannot happen because
  `baseline_v1._require_single_ticker_frame` calls
  `frame["ticker"].nunique(dropna=True) != 1`, which evaluates to
  `0 != 1 == True` on an empty frame and would raise "Expected a
  single ticker frame." That diagnostic is wrong for an empty frame
  (the issue is "no rows", not "wrong ticker count"), so the wrapper
  handles the empty case itself.
- **Raw OHLCV NaN/inf is fail-loud, NOT a `valid_mask` event**: when
  raw `close`, `open`, `high`, `low`, or `volume` contains a NaN or
  inf, `baseline_v1._validated_ohlcv` raises
  `ValueError("Raw OHLC price columns must be finite.")` (or its
  volume sibling). The wrapper does NOT catch and convert this to
  `valid_mask=False`. The `valid_mask` channel is reserved for
  DERIVED-feature NaN — warmup periods, division-by-zero denominators
  (e.g. constant close → zero Bollinger band width → NaN
  `bollinger_pctb`), and similar finite-input-produces-non-finite-
  output cases.

## Error handling

| input condition | behavior | source |
|---|---|---|
| `feature_set` not in `FEATURE_SETS.keys()` | `ValueError(f"feature_set must be one of {sorted(FEATURE_SETS)}; got {feature_set!r}")` | wrapper |
| `feature_set` is not a `str` | `TypeError("feature_set must be a str")` | wrapper |
| `frame` is not a `pd.DataFrame` | `TypeError("frame must be pd.DataFrame")` | wrapper |
| `frame` missing any required base column from `{ticker, timestamp, open, high, low, close, volume}` | `ValueError(f"frame missing required columns: {sorted(missing)}")` | wrapper, fail-fast (so `ticker` missing is a clean `ValueError` with all missing columns listed, not a downstream `KeyError`) |
| `frame["timestamp"]` dtype is not `datetime64[ns]` | `ValueError("frame['timestamp'] must be datetime64; got <dtype>")` | wrapper |
| `frame["timestamp"]` is tz-aware | `ValueError("frame['timestamp'] must be timezone-naive; got tz=<tz>")` | wrapper |
| `frame["timestamp"]` contains NaT | `ValueError("frame['timestamp'] contains NaT")` | wrapper |
| `frame["timestamp"]` is not monotonically non-decreasing | `ValueError("frame['timestamp'] must be sorted ascending")` | wrapper, fail-fast |
| `n == 0` after schema + dtype + sort checks pass | returns `(np.empty((0, k), float64), np.empty((0,), bool_))` — NOT an error. Short-circuits before delegating to baseline_v1. |
| `frame` contains multiple tickers (n>0) | `ValueError("Expected a single ticker frame.")` | delegated to `baseline_v1._require_single_ticker_frame` |
| `frame["ticker"]` contains NaN | `ValueError("Expected a single non-null ticker frame.")` | delegated to `baseline_v1._require_single_ticker_frame` |
| OHLCV sanity failure (high < low, open/close ∉ [low, high], non-positive price, negative volume, raw OHLCV NaN/inf) | `ValueError("Raw OHLC sanity failed: ...")` or similar | delegated to `baseline_v1._validated_ohlcv` |
| Derived feature is non-finite (warmup, denominator-zero, etc.) | not raised; recorded as `valid_mask = False` at affected rows |

Wrapper-layer checks run BEFORE `baseline_v1.add_baseline_v1_features`
so that an invalid `feature_set` or a sort/tz violation fails fast
without paying the cost of computing features that would then be
discarded. baseline_v1 then handles single-ticker and OHLCV sanity
as the single source of truth.

## Testing strategy

Test file: `tests/data/test_features.py` (matches the
`tests/data/test_labels.py` and `tests/data/test_raw_bars.py`
conventions from #5C-1 and #5C-3).

Eight categories, approximately 30–35 cases:

1. **Cross-check against `baseline_v1.add_baseline_v1_features`** (most
   important): build a synthetic per-ticker DataFrame, call
   `baseline_v1.add_baseline_v1_features` directly, call
   `build_features(frame, feature_set="price_volume_time")`, and assert
   that every selected column matches the baseline_v1 output value-for-
   value at every valid row. This is the anti-drift gate.
2. **Three feature_set subsets selected correctly** (≈3 cases):
   - `price_action_core` → shape `(n, 3)`, column order matches the
     constant tuple.
   - `technical_price` → shape `(n, 5)`, column order matches.
   - `price_volume_time` → shape `(n, 10)`, column order matches.
3. **Three frozen aliases produce identical results to the generic**
   (≈3 cases): `build_price_action_core_features(frame)` equals
   `build_features(frame, feature_set="price_action_core")`; same for
   `technical_price` and `price_volume_time` aliases.
4. **Warmup NaN behavior reflects in `valid_mask`** (≈4 cases):
   - n=80 bars with up-drift to guarantee all features have warmed up
     by the end.
   - Initial rows have `valid_mask=False` (warmup not complete); later
     rows reach `valid_mask=True`. We do NOT lock specific bar-count
     thresholds because warmup depends on baseline_v1 internals
     (`rolling_volatility_20` needs ≥20 in-day shifted observations,
     `rsi_14` and `bollinger_pctb` need ≥14 / ≥20 in-day, the cross-day
     `normalized_macd_hist` cascades EWM with periods 12/26/9). The
     cross-check in test category 1 is the authoritative numerical lock;
     these tests only assert qualitative shape.
   - Relative-ordering invariant: for any synthetic frame,
     `valid_mask` from `price_action_core` becomes True no later than
     `valid_mask` from `technical_price`, which in turn becomes True no
     later than `valid_mask` from `price_volume_time`. (price_action_core
     has only same-day rolling/diff features with short warmup;
     technical_price and price_volume_time both contain the
     cross-day MACD with the longest warmup.)
   - `valid_mask` becomes True for at least one row by the end of the
     n=80 frame (sanity check that the test parameters actually exercise
     a valid window).
5. **Output dtypes and shapes** (≈3 cases):
   `features.dtype == np.float64`, `valid_mask.dtype == np.bool_`,
   `features.shape == (n, k)`, `valid_mask.shape == (n,)`.
6. **Wrapper-layer input guards** (≈9 cases):
   - Non-DataFrame `frame` (e.g. dict) → TypeError.
   - Missing `timestamp` column → ValueError listing
     `missing=['timestamp']`.
   - Missing `ticker` column → ValueError listing `missing=['ticker']`
     (proves the wrapper's required-column check runs BEFORE
     `_require_single_ticker_frame`, so callers get a clean
     `ValueError` instead of a downstream `KeyError`).
   - Missing multiple required columns at once → ValueError listing all
     of them sorted (e.g. `missing=['high', 'volume']`).
   - `timestamp` dtype is `int` → ValueError.
   - `timestamp` tz-aware → ValueError ("must be timezone-naive").
   - `timestamp` contains NaT → ValueError.
   - `timestamp` not sorted ascending → ValueError.
   - `feature_set="invalid_name"` → ValueError listing the valid
     choices.
   - `feature_set=42` (non-str) → TypeError.
7. **Delegated guards from baseline_v1** (≈4 cases):
   - Multi-ticker `frame` → ValueError from
     `_require_single_ticker_frame`.
   - Missing OHLCV column (`open` dropped) → ValueError from
     `_validated_ohlcv`.
   - OHLC sanity violation (`high < low`) → ValueError.
   - Non-positive price → ValueError.
8. **Edge cases + frozen constant lock** (≈5 cases):
   - n=0 frame with full required-column schema (and valid
     `timestamp` dtype) → returns empty arrays of correct dtypes and
     shapes, does NOT raise. Short-circuits before the baseline_v1
     delegation (which would otherwise raise "Expected a single ticker
     frame." via `nunique=0 != 1`).
   - n=0 frame MISSING a required column (e.g. `volume` dropped) →
     ValueError from the wrapper's required-column check, NOT the empty
     short-circuit (schema check precedes n=0 short-circuit per the
     §"Data flow" step ordering).
   - Raw `close` containing NaN → `ValueError("Raw OHLC ... must be
     finite.")` from `baseline_v1._validated_ohlcv`. The wrapper does
     NOT convert this to `valid_mask=False`; raw OHLCV NaN is fail-loud
     by design. (`valid_mask` only reflects DERIVED-feature NaN.)
   - Constant `close` (e.g. exactly 100.0 for every row, with high
     and low slightly different so OHLC sanity passes) → for
     `bollinger_pctb` the rolling-20 standard deviation is 0, the
     Bollinger band width (upper − lower) is 0, and
     `bollinger_denom = (upper − lower).replace(0.0, np.nan)` produces
     NaN; `bollinger_pctb = (close − lower) / NaN = NaN` for every
     row including AFTER the warmup completes. Therefore:
     - `price_action_core` (no `bollinger_pctb`): once warmup
       completes, `valid_mask` becomes True for the rest of the frame.
     - `technical_price` and `price_volume_time` (both include
       `bollinger_pctb`): `valid_mask` is False for EVERY row, even
       after warmup, because the Bollinger feature never becomes
       finite under a constant close. This proves the wrapper does
       NOT raise on derived NaN; the affected rows just record
       `valid_mask=False`. (This is the canonical "derived NaN →
       valid_mask" case that replaces the rejected "raw close NaN"
       test.)
   - `FEATURE_SETS` constant matches the
     `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` "Active feature
     sets" table exactly (lock against drift between the freeze
     document and this module).

Synthetic fixture style: 5-minute bar series built via
`pd.date_range("2025-01-02 09:30", periods=N, freq="5min")` and a
deterministic OHLCV series that satisfies `_validated_ohlcv`
(high≥low, open/close ∈ [low, high], positive prices, non-negative
volume). No raw bar I/O, no fixture file commits, runtime < 2 seconds
total.

## What this spec does NOT cover

- The other pieces of #5C (#5C-1 labels done, #5C-3 raw_bars done;
  #5C-4 splits, #5C-5 window builder are siblings). Each gets its own
  spec.
- Editing or refactoring `baseline_v1.add_baseline_v1_features` or
  any of its helpers (`grouped_rolling`, `grouped_wilder_ewm`,
  `continuous_ewm`, etc.).
- Train-only scaler fitting or feature scaling — that lives in
  `baseline_v1.fit_train_only_scaler` (called by #5C-4 split markers
  or by the orchestrator).
- Pooled multi-ticker handling — caller iterates per-ticker after
  `#5C-3 load_ticker_bars` returns the pooled DataFrame.
- Sample-sufficiency or row-count enforcement (warmup periods leave
  the first ≈20-35 rows invalid depending on the slowest feature in
  the selected set; this is the caller's concern to filter via
  `valid_mask`, and the exact threshold is intentionally not spec'd
  here because it depends on baseline_v1 internals).
- Pushing the commit — push is a separate user-authorized step.

## Verification commands (post-implementation)

```bash
E:/codex_workspace/_envs/py311_shared/python.exe -m pytest \
  tests/data/test_features.py -q

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
- Per-file ≈ 30–35 passed; N08 face goes from 345 to ≈ 378 passed;
  `GATE PASSED` exit 0.

## Open questions

None at the spec stage. Decisions taken during brainstorming:

- Single-ticker DataFrame in → `(features ndarray (n, k), valid_mask
  ndarray (n,))` out (per-ticker numpy style, matches #5C-1 labels).
- Wrap `baseline_v1.add_baseline_v1_features` as single source of
  truth; no reimplementation of the 10 feature formulas.
- Generic `build_features(frame, *, feature_set)` plus three frozen
  aliases (`build_price_action_core_features`,
  `build_technical_price_features`,
  `build_price_volume_time_features`) — same shape as #5C-1's H03_BPS1P5
  + `build_h03_bps1p5_labels`.
- `FEATURE_SETS` module-level constant mirrors the three frozen sets
  from `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` verbatim and is
  locked by a test against drift.
- Column order in the returned ndarray equals
  `FEATURE_SETS[feature_set]` tuple verbatim.
- `valid_mask` is row-level AND (a row is valid iff every selected
  feature is finite at that row); no per-feature mask.
- Wrapper-layer fail-fast on missing required base columns
  (`ticker`, `timestamp`, OHLCV), unsorted / tz-aware / NaT /
  wrong-dtype timestamps; single-ticker uniqueness and OHLCV value
  sanity (including raw OHLCV NaN/inf fail-loud) are delegated to
  baseline_v1 as single-source-of-truth.
- `n=0` short-circuits AFTER the wrapper's schema + dtype + sort
  checks pass, BEFORE the baseline_v1 delegation. baseline_v1 cannot
  be delegated to on an empty frame because
  `_require_single_ticker_frame` would raise on `nunique=0 != 1`.
- Raw OHLCV NaN/inf is fail-loud via
  `baseline_v1._validated_ohlcv`. The `valid_mask` channel is
  reserved for DERIVED-feature NaN only (warmup, denominator-zero
  divisions). Tests use constant `close` (causing Bollinger /
  MACD-denominator NaN) rather than raw-close NaN, since the latter
  raises rather than producing `valid_mask=False`.
- Warmup-bar count is intentionally not spec'd in test assertions;
  the cross-check test against `baseline_v1.add_baseline_v1_features`
  is the authoritative numerical lock. Warmup-related tests assert
  qualitative shape only (initial rows invalid, later rows valid,
  `price_action_core` valid no later than the other two sets).
- Tests cross-check the wrapper against
  `baseline_v1.add_baseline_v1_features` as the anti-drift gate; no
  fixture files, no raw bar I/O.
