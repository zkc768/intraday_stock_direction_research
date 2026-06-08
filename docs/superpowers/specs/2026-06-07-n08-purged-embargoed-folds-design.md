# N08 #5E-1 — Purged + Embargoed Train-Inner Folds (`deep_sequence/folds.py`) Design

> Status: design 2026-06-07. First orchestration-layer piece — the two
> unimplemented §8.2 fold builders (`rolling_origin_folds` already shipped #5B).
> Tooling: inline design + `humanize:ask-codex` review. §4.1-CRITICAL: fold
> construction is a leakage red line.
> Coexistence: model package co-developed with a parallel Codex session (it owns
> the contract/orchestration surface); check `git status` clean + log before
> editing, commit only these files.

## 1. Goal & Scope

Implement `purged_time_series_folds` and `embargoed_train_inner_folds` (currently
`NotImplementedError` scaffolds) per the §8.2 contract documented in
`folds.py`'s module docstring, mirroring the shipped `rolling_origin_folds`
shape: per-ticker chronological split → pooled
`(train_inner_fit_idx, train_inner_val_idx)` pairs, `np.int64` positional indices,
sorted ascending, yielded in chronological fold order.

**Design doc §8.2 gives only high-level requirements** (split per ticker
chronologically; no fold trains on a label horizon overlapping its inner-val;
preprocessing on train-fit only). The precise fold-construction algorithm for
purged/embargoed is spec-introduced (like the GRU axes / fusion variants) and
flagged for Codex review (§7) — the leakage index math is the critical part.

**Out of scope:** data-agnostic index math only — no data loading, windows,
preprocessing, 08X/08F/08O, `run_stage`. The builder owns ONLY responsibility
layer 1 (the chronological split + purge/embargo); window/trading-day/ticker and
preprocessing invariants are upstream (layers 2-3, per the module docstring).

## 2. Distinction from `rolling_origin_folds`

`rolling_origin_folds` is forward-chaining / expanding-window: val = a fixed-size
TAIL block, train = everything strictly BEFORE val (early data is always train,
late data is always val). Only a LEADING purge is needed (train is all before
val).

`purged_time_series_folds` / `embargoed_train_inner_folds` are INTERIOR-block
K-fold (López de Prado style): the per-ticker series is tiled into `n_folds`
contiguous blocks; each block is val once; train = the OTHER blocks (BOTH before
AND after val). Because train now includes rows AFTER val, leakage is controlled
by purge (forward labels reaching val) + embargo (serial-correlation / window
overlap on both sides). No `inner_validation_size` (the block size is derived
from `n_folds`), matching the scaffold signatures.

## 3. Algorithm

### 3.1 Per-ticker blocks (shared)
Per ticker (sorted chronologically by `timestamps`, stable, exactly as
`rolling_origin_folds`): tile the `m` per-ticker positions into `n_folds`
contiguous blocks via `np.array_split` semantics (sizes differ by ≤ 1; earlier
blocks take the remainder). Block `i` = `[val_start_i, val_end_i)` (positional
within the ticker's chronological order).

### 3.2 `purged_time_series_folds(*, n_folds, label_horizon_k)`
For fold `i`, val = block `i` `[a, b)`. The **SYMMETRIC** López de Prado purge
(Codex P1) removes every train row whose label `[t, t+k]` overlaps the val
labels (which span `[a, b+k-1]`):
- train-BEFORE `[a-k, a)` — forward label reaches val (`t+k >= a`);
- train-AFTER `[b, b+k)` — the val rows near `b-1` have labels reaching into
  `[b, b+k)`, so a train-after row whose feature bar falls in that band sits
  inside a val label horizon (AGENTS §4.1.4 "label horizons must not cross
  train/validation boundaries"). Keeping all train-after rows (the earlier
  one-sided draft) was a real leak.

So exclude per ticker `[max(0, a-k), min(m, b+k))`:
`train = positions[:max(0, a-k)] ∪ positions[min(m, b+k):]`;
`val = positions[a:b]`.

### 3.3 `embargoed_train_inner_folds(*, n_folds, label_horizon_k, embargo_size)`
Symmetric purge PLUS an embargo gap of `embargo_size` on BOTH sides (Codex P1):
exclude per ticker `[max(0, a - k - embargo_size), min(m, b + k + embargo_size))`:
`train = positions[:max(0, a - k - e)] ∪ positions[min(m, b + k + e):]`.
i.e. `k + embargo_size` removed on each side (the `k` symmetric label-purge + the
`embargo_size` serial-correlation gap). Additive (`k + e`), not `max(k, e)` —
the conservative, leak-safe reading.

**Window-overlap contract (Codex P1/P2 — do NOT overclaim).** This layer-1 builder
has no `window_size`, so it CANNOT by itself guarantee freedom from input-window
overlap (a window of `window_size` could still straddle val). It guarantees only
the label-horizon purge + the requested `embargo_size` band. To also exclude
window overlap, the CALLER must size `embargo_size` so that
`k + embargo_size >= window_size - 1` (a layer-2 invariant owned upstream per the
module docstring). The `embargoed_train_inner_folds` docstring is corrected to
state this rather than claiming it removes window-overlap leakage outright.

### 3.4 Pool across tickers
Identical to `rolling_origin_folds`: fold `i` yields the union of each ticker's
fold-`i` train / val positions (global positional indices), each `np.sort`-ed
ascending, `np.int64`. Yield folds `0 .. n_folds-1` in chronological order.

## 4. Validation / guards (fail-loud, mirror `rolling_origin_folds`)
- `timestamps` / `ticker_ids` 1-D, equal length.
- `n_folds >= 2` (interior K-fold needs ≥ 2 blocks so every fold has a non-empty
  train from the other blocks; `n_folds=1` would leave fold 0 with no other
  block → reject, unlike rolling-origin which allows 1).
- `label_horizon_k >= 0`; (`embargoed`) `embargo_size >= 0`.
- Each ticker must have enough samples that EVERY fold has a non-empty val AND a
  non-empty train after purge/embargo. **Checked by SIMULATION (Codex P3), not a
  hand-derived formula:** iterate every ticker × fold, apply the exact exclusion
  interval, and `ValueError` (naming the ticker + fold) if any resulting train or
  val array is empty. This robustly catches fold 0 (no train-before), the last
  fold (no train-after), large `k`/`embargo`, and uneven `np.array_split` blocks.

## 5. Files
- **Modify** `src/intraday_research/models/deep_sequence/folds.py` (fill the two
  scaffolds; `rolling_origin_folds` untouched).
- **Create** `tests/stages/models/test_purged_embargoed_folds.py`.
- No other changes.

## 6. Testing (`tests/stages/models/test_purged_embargoed_folds.py`)
Mirror `test_rolling_origin_folds.py`:
- **Block tiling**: single ticker, n_folds blocks tile all positions; each block
  is val exactly once; val blocks are contiguous + disjoint + cover the series.
- **Interior train both sides**: an interior fold's train has rows BEFORE and
  AFTER val (distinguishes from rolling-origin); fold 0 has only train-after,
  fold n_folds-1 only train-before.
- **SYMMETRIC PURGE correctness (§4.1 red line)**: reusable per-ticker-rank
  assertion — for `purged`, EVERY train rank `r` satisfies `r < a-k or r >= b+k`
  (both the `[a-k, a)` and `[b, b+k)` bands absent); assert the just-outside rows
  (`a-k-1`, `b+k`) ARE present when in range.
- **EMBARGO correctness**: for `embargoed`, every train rank `r` satisfies
  `r < a-k-e or r >= b+k+e` (symmetric `k+e` bands absent); just-outside present.
- **Disjoint / sorted / int64 / n_folds pairs / per-ticker-chronological pool**
  (as rolling-origin, incl. the interleaved two-ticker fixture).
- **Guards**: shape mismatch, `n_folds<2`, negative k / embargo, insufficient
  per-ticker samples — each `ValueError`.
- **Import**: existing `test_folds_module_exports_three_named_builders` stays green.

## 7. Open Decisions for Review (Codex-resolved)
1. ~~Interior-block K-fold interpretation~~ — **confirmed** by Codex (names +
   signatures + §8.2 support it); documented as train-inner EXPLORATION folds, made
   leak-safe by the symmetric purge.
2. ~~Embargo additive vs max~~ — **resolved**: additive (`k + embargo_size` each
   side), Codex-confirmed as the leak-safe reading.
3. **`n_folds >= 2`** for the interior variants — Codex did not object; kept
   (interior K-fold needs ≥ 2 blocks so every fold trains on another block).
4. **Block tiling** via `np.array_split` — Codex confirmed fine.

## 8. Codex Design Review — absorbed (2026-06-07, gpt-5.5:high, 193s)
- **P1 (purged leak):** §3.2 now uses the SYMMETRIC purge `[a-k, b+k)` (removes
  the train-AFTER `[b, b+k)` band too — val labels reach into it). The one-sided
  draft was a real §4.1 leak.
- **P1 (embargo after-val):** §3.3 now `[a-k-e, b+k+e)` (symmetric `k+e` both
  sides), not the draft's `[a-k-e, b+e)`.
- **P1/P2 (window-overlap overclaim):** §3.3 + the `embargoed` docstring no longer
  claim window-overlap removal; they state the caller must size
  `embargo_size` so `k + embargo_size >= window_size - 1` (a layer-2 invariant).
- **P3 (guards by simulation):** §4 checks emptiness by simulating every
  ticker × fold exclusion, not a formula.
- **Test additions:** §6 uses per-rank symmetric-band assertions + the full
  edge-case matrix (k=0, e=0, fold 0/last, uneven split, unsorted, two interleaved
  tickers, insufficient samples).
- **Confirmed:** the corrected symmetric purge IS the textbook López de Prado
  purged K-fold; interior interpretation + additive embargo + `np.array_split`
  tiling are sound.
