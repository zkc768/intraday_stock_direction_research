# N08 #5F-4 — 08X Data-Load Chain (windowed-index) Design

> Status: design 2026-06-08. Second half of the 08X real-data wiring; FINISHES
> sub-slice ② (real data). Chains the existing
> raw_bars→features→labels→splits→windows primitives into a windowed train-inner
> index, then wires it into `resolve_train_inner_index`'s production path so
> `RUN_08X_BUILD_TRAIN_INNER_FOLDS` runs on REAL data end-to-end. Tooling: inline
> design + `humanize:ask-codex` review (run
> `.humanize/skill/2026-06-08_01-24-01-905-93801c67/`).

## 1. Goal & Scope

After #5F-3 (the `.txt`→5min loader), the only thing between real data and the
fold builder is the chain that turns a 5-minute frame + frozen candidate into a
windowed sample index. This slice adds it and connects the production path.

**In scope**: new `data/windowed_index.py::build_train_inner_windowed_index`
(pure chain over a 5-min frame); `resolve_train_inner_index` production path in
`stages/deep_sequence_fold_build.py` (precedence + frozen-candidate validation);
synthetic + end-to-end tests.

**Out of scope** (Codex Q7): the Colab Drive-download cell / notebook regen; model
fit / trial loop / metrics; any change to raw_bars/features/labels/splits/windows.

## 2. Codex decisions absorbed
- Q2 (P0): runtime manifest from `config["data"]["txt_manifest"]` ({ticker:
  local_path}); frozen params from `config["frozen_candidate"]`. Stage does NOT
  parse Drive IDs (Colab downloads first, passes local paths).
- Q3 (P0): require explicit `horizon_k` + `threshold_bps`; cross-check against the
  frozen `label_config` mapping; **fail loud on mismatch** (auditable provenance).
- Q4 (P0): `feature_valid_mask = feat_valid`; `target_valid_mask = label_valid &
  split_valid`. No separate partition gate (build_windows enforces uniform-partition
  windows; resolve_train_inner_index filters PARTITION_TRAIN after).
- Q1 (P1): new `data/windowed_index.py`.
- Q5 (P1): row-aligned concat is the input contract (build_windows re-groups by
  ticker; output block order is sorted-ticker, which the fold builders re-sort).

## 3. New module — `data/windowed_index.py`

Pure data chain over a pooled 5-minute frame (no config / no I/O):

```python
def build_train_inner_windowed_index(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    horizon_k: int,
    threshold_bps: float,
    window_size: int,
) -> dict[str, np.ndarray]:
    """frame = pooled 5-min [ticker,timestamp,ohlcv] (sorted ticker,timestamp).
    Returns the build_windows dict (X,y,target_partition,target_timestamps,
    target_row_positions,target_ticker_ids)."""
```

Algorithm — per ticker group of `frame` (sorted by timestamp):
1. `feat, feat_valid = build_features(ticker_frame, feature_set=feature_set)`
2. `labels, label_valid = build_no_trade_band_labels(close, ts, horizon_k=, threshold_bps=)`
3. `partition, split_valid = apply_stage0_chronological_split(ts, horizon_k=)`
4. `target_valid = label_valid & split_valid`  (Codex Q4)

Concatenate per-ticker arrays IN frame row order →
`features` (n,F), `labels` (n,), `partition` (n,), `feature_valid_mask = feat_valid`,
`target_valid_mask = target_valid`; `ticker_ids = frame["ticker"].to_numpy()`,
`timestamps = frame["timestamp"].to_numpy()` (datetime64[ns]).
Then `build_windows(features, labels, timestamps, ticker_ids, partition=...,
feature_valid_mask=..., target_valid_mask=..., window_size=window_size)` and
return its dict.

Per-ticker `ts` must be `datetime64[ns]` ndarray (frame timestamps already are).
Guards: non-empty frame; required columns; `feature_set` / `window_size` valid
(delegated to the primitives' own fail-loud checks).

## 4. `resolve_train_inner_index` production path (fold_build.py)

Restructure so the windowed index is obtained, THEN the existing
PARTITION_TRAIN filter runs (unchanged). Precedence:

1. `injected_window_index` arg (tests) — as today.
2. `config["windowed_index"]` (tests / runner) — as today.
3. `config["data"]["txt_manifest"]` present → run the chain:
   `frame = load_ticker_bars_txt(manifest)` →
   `wi = build_train_inner_windowed_index(frame, **_resolve_chain_params(config))`.
4. else `NotImplementedError` (as today).

Then the existing domain-assert + `target_partition == PARTITION_TRAIN` filter →
`(target_timestamps[train], target_ticker_ids[train])`.

`_resolve_chain_params(config)` reads `config["frozen_candidate"]` and returns
`(feature_set, horizon_k, threshold_bps, window_size)`:
- require `feature_set` (in FEATURE_SETS), `window_size` (int>0), `horizon_k`
  (int>0), `threshold_bps` (num>=0);
- if `label_config` present, assert it maps (via the frozen
  `LABEL_CONFIGS = {h03_bps1p5:(3,1.5), h09_bps3p0:(9,3.0), h24_bps7p5:(24,7.5)}`,
  sourced from labels.H0*_*) to the explicit `(horizon_k, threshold_bps)`; fail on
  mismatch.
- `horizon_k` already equals the fold-purge horizon (#5F-2 provenance gate); this
  chain reuses the same value, so the purge and the labels share one horizon.

## 5. Tests

`tests/data/test_windowed_index.py` (synthetic 5-min frame, no real data):
1. build on a multi-ticker, multi-day frame straddling VALIDATION_START
   (2013-09-16): assert windowed dict keys, train+validation partitions both
   present, no cross-trading-day windows (target dates consistent), F == len(FEATURE_SETS[fs]).
2. mask effect: a row whose label is invalid (near a horizon/split edge) does not
   become a window target.
3. window_size respected (X shape (W, window_size, F)).

`tests/stages/test_deep_sequence_fold_build.py` (extend):
4. precedence: `injected_window_index` and `config["windowed_index"]` still bypass
   raw loading (no txt_manifest needed).
5. e2e: synthetic `.txt` files in tmp_path → `config["data"]["txt_manifest"]` +
   `config["frozen_candidate"]` (feature_set/window_size/horizon_k/threshold_bps/
   label_config) → `run_stage(RUN_08X_BUILD_TRAIN_INNER_FOLDS=True)` → real
   `08x_fold_results.csv` non-empty + passes validator.
6. label_config mismatch (h09 with horizon_k=3) → ValueError.
7. missing/empty txt_manifest AND no windowed_index → NotImplementedError.

## 6. Files touched
- NEW `src/intraday_research/data/windowed_index.py`
- MODIFY `src/intraday_research/stages/deep_sequence_fold_build.py`
  (`resolve_train_inner_index` production path + `_resolve_chain_params`)
- NEW `tests/data/test_windowed_index.py`
- MODIFY `tests/stages/test_deep_sequence_fold_build.py` (precedence + e2e + mismatch)
- NEW this spec

**Do NOT touch**: raw_bars/features/labels/splits/windows, the contract module,
other stages.

## 7. Acceptance criteria
1. `bash scripts/check_n08_resume_gate.sh` exits 0.
2. `pytest tests/data tests/stages -q` green (existing + new chain/e2e tests).
3. `pytest tests/stages/models -q` unchanged (419+2).
4. e2e: synthetic `.txt` → `run_stage` BUILD_FOLDS → non-empty fold_results passing
   `validate_08x_fold_results_frame(require_non_empty=True)`.
5. label_config↔(horizon_k,threshold_bps) mismatch fails loud.
6. injected/config windowed_index paths still bypass raw loading.
7. No real ticker file touched; no Drive API in package code.

## 8. After this slice
Sub-slice ② (real data) is COMPLETE: `RUN_08X_BUILD_TRAIN_INNER_FOLDS` runs on
real 5-ticker data end-to-end. Block 1 remaining: ③ search-space build
(`RUN_08X_SEARCH_SPACE_DRY_RUN`), then ④ `RUN_08X_QUICK_SEARCH` (first real model
fit, wiring the built model bodies + compute_trial_metrics). A separate notebook
regen slice adds the Colab Drive-download cell (Codex Q7 deferred).
