# Design: N08 #5C-1 — `data/labels.py` for h03_bps1p5 no-trade-band labels

**Date**: 2026-06-07
**Scope**: validation_only
**Holdout/test contact**: false
**Status**: draft, awaiting user review then writing-plans skill handoff

## Goal

Provide a `src/intraday_research/data/labels.py` module that exposes a
numpy-faced wrapper around the frozen
`baseline_v1.make_no_trade_band_labels` implementation, returning
`(labels, valid_mask)` arrays consumable by `LastStepLightGBMControl`
(#5A) and the downstream pieces #5C-2 … #5C-5 of the N08 raw-data
pipeline.

This is piece **#5C-1** of the 5-piece #5C subtask:

| piece | role | status |
|---|---|---|
| **#5C-1 Labels** | no-trade-band binary labels | this spec |
| #5C-3 CSV loader | 5-ticker raw bar I/O | sibling, separate spec |
| #5C-2 Features | `price_volume_time` builder | sibling, separate spec |
| #5C-4 Split markers | calendar split + cross-split invalidation | sibling, separate spec |
| #5C-5 Window builder | `(X, y, timestamps, ticker_ids)` assembly | sibling, separate spec |

## Architecture

New files:

```
src/intraday_research/data/__init__.py
src/intraday_research/data/labels.py
```

`labels.py` exports:

- Three frozen configuration constants (matching
  `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` Candidate Space):

  ```python
  H03_BPS1P5 = {"horizon_k": 3, "threshold_bps": 1.5}
  H09_BPS3P0 = {"horizon_k": 9, "threshold_bps": 3.0}
  H24_BPS7P5 = {"horizon_k": 24, "threshold_bps": 7.5}
  ```

- One generic function:

  ```python
  def build_no_trade_band_labels(
      close: np.ndarray,
      timestamps: np.ndarray,
      *,
      horizon_k: int,
      threshold_bps: float,
  ) -> tuple[np.ndarray, np.ndarray]:
      """Return (labels, valid_mask)."""
  ```

- Three frozen aliases (thin one-liners around the generic):

  ```python
  def build_h03_bps1p5_labels(close, timestamps) -> ...
  def build_h09_bps3p0_labels(close, timestamps) -> ...
  def build_h24_bps7p5_labels(close, timestamps) -> ...
  ```

Internally `build_no_trade_band_labels` wraps
`baseline_v1.make_no_trade_band_labels` so the frozen Stage 0 semantics
are the single source of truth.

## Data flow

```
Inputs:                                       Internal:                          Outputs:
─────────                                     ───────────                        ─────────
close:      ndarray[float64], shape (n,)      1. Build temp DataFrame:           labels:     ndarray[int8], shape (n,)
                                                                                    values in {0, 1, -1}
                                                pd.DataFrame({                       -1 = invalid placeholder
                                                  "ticker": "_anon",                    (do not interpret as class 0
                                                  "timestamp": timestamps,              -- always pair with valid_mask)
                                                  "close": close,
timestamps: ndarray[datetime64[ns]],            })                                 valid_mask: ndarray[bool_], shape (n,)
            shape (n,)                                                                True iff ALL conditions:
                                              2. Call:                                  - future_cumulative_return is not
                                                 baseline_v1.                             NaN (covers both NaN at t and
                                                 make_no_trade_band_labels                NaN at t+h)
                                                                                        - bars at t+1..t+h are same
                                                                                          trading day as t
                                                                                        - future_return > +threshold
                                                                                          OR future_return < -threshold
                                                                                          (no-trade-band is the closed
                                                                                          interval [-threshold, +threshold];
                                              3. Extract from result frame:              boundary values are invalid)
                                                 - labels: column "label" (NaN -> -1,
                                                   else cast to int8 from {0.0, 1.0})
                                                 - valid_mask: ~np.isnan(frame.label)
```

### Key data-flow guarantees

- **Order preservation**: output arrays are in the same positional order
  as the input arrays. Caller must pass timestamps already sorted
  ascending; the wrapper does not reorder.
- **Sentinel discipline**: `-1` at invalid positions is a defensive
  placeholder. Downstream code MUST gate on `valid_mask` before treating
  a position as a class label. Tests verify the sentinel.
- **"Same trading day"** is `pd.Timestamp.dt.date` equality between t,
  t+1, …, t+h — the same timezone-naive calendar-date comparison
  `baseline_v1` uses. Caller is responsible for passing timestamps in
  a consistent timezone (typically the bar file's ET).
- **Out of scope here, in #5C-4**: cross-split (train/validation/holdout
  boundary) invalidation. `labels.py` only handles the three invalid
  reasons documented in `baseline_v1.make_no_trade_band_labels`:
  missing future, cross-day, and within-no-trade-band.

## Error handling

| input condition | behavior |
|---|---|
| `close` or `timestamps` not 1-D | `ValueError("must be 1-D arrays")` |
| `close.shape != timestamps.shape` | `ValueError("must be same length")` |
| `timestamps` not monotonically non-decreasing | `ValueError("timestamps must be sorted ascending")` — fail fast |
| `np.isnat(timestamps).any()` | `ValueError("timestamps contains NaT")` |
| `horizon_k <= 0` | `ValueError("horizon_k must be positive")` |
| `threshold_bps < 0` or non-finite | `ValueError("threshold_bps must be non-negative and finite")`. `threshold_bps == 0.0` is allowed and mirrors baseline_v1's degenerate no-trade-band (no band → any sign-classified return labels; the three frozen aliases all use positive thresholds so this branch only matters for future configs). |
| `close` contains NaN | not raised; NaN propagates through `future_cumulative_return` and registers as `invalid_missing_future` → `valid_mask` False |
| `n == 0` | returns empty `int8` and `bool_` arrays |
| `n < horizon_k + 1` | returns `labels = -1 * ones(n, int8)`, `valid_mask = zeros(n, bool_)` |
| input mixes multiple tickers | NOT detected — caller responsibility. Docstring warns. |

Requiring sorted timestamps avoids the hidden reorder bugs that would
otherwise be needed to remap baseline_v1's internally-sorted output
back to the caller's input order.

## Testing strategy

Test file: `tests/data/test_labels.py` (new directory mirroring
`src/intraday_research/data/`).

Six categories, approximately 18–22 cases:

1. **Cross-check against `baseline_v1.make_no_trade_band_labels`** (most
   important): build a synthetic per-ticker DataFrame, call baseline_v1
   directly, call our wrapper, assert label and valid_mask equivalence.
   This is the anti-drift gate.
2. **Three invalid reasons each activated separately**:
   - up-only price + small ε > threshold → valid samples label=1, last
     `horizon_k` invalid for missing future.
   - down-only → label=0.
   - within-band returns (|ret| <= threshold, boundary inclusive) →
     no-trade-band invalid.
   - prices spanning a trading-day boundary → cross-day invalid.
   - `threshold_bps == 0.0` (degenerate baseline_v1 case) → all valid
     non-cross-day non-missing-future samples receive a {0, 1} label
     by strict sign of return; no row is invalidated for being inside
     a zero-width no-trade-band.
3. **Three frozen aliases** (`h03_bps1p5`, `h09_bps3p0`, `h24_bps7p5`)
   produce identical results to calling the generic function with the
   matching `(horizon_k, threshold_bps)`.
4. **Output dtypes and shapes**: `labels.dtype == np.int8`,
   `valid_mask.dtype == np.bool_`, both `shape == close.shape`.
5. **Sentinel discipline**: at any invalid position labels == -1; at
   any valid position labels ∈ {0, 1}.
6. **Error handling**: non-1D / misaligned / unsorted timestamps / NaT
   / `horizon_k <= 0` / `threshold_bps < 0` / non-finite
   `threshold_bps` each raise `ValueError` with a matching message
   regex. `threshold_bps == 0.0` is explicitly accepted and tested
   (see item 2 above) since baseline_v1 accepts it. NaN-in-close
   passes through without raising and produces `valid_mask=False` at
   NaN positions.

Synthetic fixture style: 5-minute bar series built via
`pd.date_range("2025-01-02 09:30", periods=N, freq="5min")` and a
deterministic close series so expected labels can be computed by hand
in the test body. No raw bar I/O, no fixture file commits, runtime
< 1 second total.

## What this spec does NOT cover

- The other four pieces of #5C (loader, features, split markers, window
  builder) — each gets its own spec under
  `docs/superpowers/specs/` when its turn arrives.
- Real raw-bar CSV ingestion — that is #5C-3.
- Calendar partition assignment (`train` / `validation` /
  `closed_holdout_boundary_only`) — that is #5C-4.
- Window construction `(X, y, timestamps, ticker_ids)` — that is #5C-5.
- Implementation of the other two frozen label configs at a different
  function level (the aliases are thin and intentional; we are not
  building a dispatch registry).
- Editing the frozen `baseline_v1.py` implementation.

## Verification commands (post-implementation)

```bash
PYTHON=E:/codex_workspace/_envs/py311_shared/python.exe
"$PYTHON" -m pytest tests/data/test_labels.py -q
"$PYTHON" -m pytest tests/stages/models tests/contracts/test_deep_sequence_exploration_contract.py tests/notebooks/test_deep_sequence_exploration_static_gate.py tests/test_package_import.py tests/stages tests/data -q
bash scripts/check_n08_resume_gate.sh
```

## Open questions

None at the spec stage. Decisions taken during brainstorming:

- Wrap `baseline_v1` rather than reimplement (single source of truth).
- Return `(int8 labels with -1 sentinel, bool_ valid_mask)` not float-NaN.
- Cross-split invalidation deferred to #5C-4.
- Require sorted timestamps; fail fast on unsorted input.
- Test cross-checks against `baseline_v1` as the anti-drift gate; no
  raw-CSV fixtures at this layer.
