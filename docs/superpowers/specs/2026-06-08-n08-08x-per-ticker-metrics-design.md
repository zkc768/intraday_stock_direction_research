# N08 #5F-7 — 08X Per-Ticker Train-Inner Metrics Design

> Status: design 2026-06-08. Focused foundational slice: quick-search emits
> per-ticker train-inner metrics (`08x_per_ticker.csv`), unblocking the §11.1
> quick→medium escalation gate (#5F-8 — "positive on ≥4 tickers"), §14.3
> concentration guardrails, and the §14.1 per_ticker_delta decision metric.
> Train-inner only; no official-validation / holdout (AGENTS §4.1). Tooling:
> inline design + `humanize:ask-codex` design review
> (`.humanize/skill/2026-06-08_03-18-52-869-5c12e923/`). No P0; 2 P1s folded in;
> contract addition SANCTIONED by the contract owner.

## 1. Goal & Scope
08X quick-search (#5F-6) emits only AGGREGATE per-trial metrics, and
`run_single_trial` discards `y_pred`, so per-ticker deltas cannot be recovered
post-hoc — they must be produced at trial time. §11.1's "positive on ≥4 tickers"
(`TIER_ESCALATION_QUICK_TO_MEDIUM_POSITIVE_TICKER_MIN=4`) needs them.

**In scope**: `compute_per_ticker_delta` (metrics.py, single-class-safe); a
`collect_per_ticker` tuple-return path on `run_single_trial`; per-ticker collection
+ aggregation + `08x_per_ticker.csv` in `run_quick_search`; the canonical 9th
OUTPUT_FILES_08X artifact + `validate_08x_per_ticker_frame` (contract owner
sanctioned) + the schema-smoke header.

**Out of scope (later)**: the §11.1 escalation gate itself + the lcb-vs-control
computation + `08x_tier_escalation` artifact (#5F-8); §14.3 concentration
guardrails; medium/aggressive tiers.

## 2. Research-safety invariants
- Per-ticker is computed ONLY from a completed trial's `y[val_idx]`, `y_pred`,
  `ticker_ids[val_idx]` (train-inner validation rows). No official/holdout read.
- A class-collapse trial (reclassified to `failed` AFTER `run_single_trial`
  returns) contributes NO per-ticker rows (Codex P1-1).
- The artifact is a COMPLETE `candidate × expected_ticker` grid so the
  "≥4 tickers" denominator is never hidden (Codex P1-2).

## 3. Codex design-review outcomes folded in
- **P1-1** tuple return, not a sink: the loop keeps per-ticker rows only when the
  post-class-collapse `fit_status == "completed"`.
- **P1-2** complete `candidate × expected_ticker` grid; missing/single-class ticker
  → NaN metrics, `positive_delta=False`, `coverage_status="insufficient"`.
- **Sanction**: `08x_per_ticker.csv` becomes the 9th OUTPUT_FILES_08X artifact.
- **Q6**: "positive on ≥4 tickers" basis = per-ticker `delta_macro_f1_vs_dummy > 0`
  (vs same-row stratified dummy, matching 08O); the control comparison is the
  separate #5F-8 condition.

## 4. Module changes

### 4.1 `metrics.py` — `compute_per_ticker_delta` (single-class-safe)
```python
def compute_per_ticker_delta(
    y_true: np.ndarray, y_pred: np.ndarray, ticker_ids: np.ndarray
) -> list[dict[str, Any]]:
    """Per-ticker macro_f1 + delta_macro_f1_vs_dummy on a fold's val rows.
    For each ticker: both_classes_present True -> (macro_f1, delta) via the same
    stratified-null logic as compute_trial_metrics; else NaN (NO raise)."""
```
Row: `{ticker, n_rows, both_classes_present, macro_f1, delta_macro_f1_vs_dummy}`.
Reuses `_stratified_null_macro_f1`; never raises on a single-class ticker slice
(returns `both_classes_present=False` + NaN). `compute_trial_metrics` is unchanged.

### 4.2 `run_single_trial` — optional tuple return (Codex P1-1 / Q1)
`run_single_trial(..., collect_per_ticker: bool = False)`. Default → returns the
29-col `dict` (unchanged; #5F-5/#5F-6 callers untouched). When `True` → returns
`(row, per_ticker_rows)`, where `per_ticker_rows` is `compute_per_ticker_delta(...)`
on `y[val_idx]/y_pred/ticker_ids[val_idx]` ONLY for a completed fit (else `[]`).
`y_pred` is hoisted so per-ticker is computed in the completed branch; a per-ticker
error can never flip a successful fit to failed.

### 4.3 `run_quick_search` — collect, aggregate, write
- Pass `collect_per_ticker=True`; after `_apply_class_collapse_guard(row)`, keep the
  trial's `per_ticker_rows` ONLY when `row["fit_status"] == "completed"`, tagged
  with `candidate_id / candidate_family / fold_id / seed`.
- `expected_tickers = sorted(unique(masked ticker_ids))` (the train-inner
  denominator).
- Aggregate to the complete `candidate × expected_ticker` grid. Per cell:
  - `n_trials_expected = n_folds * n_seeds`
  - `n_trials_contributing` = completed trials where this ticker had both classes
  - `n_rows_total` = Σ n_rows over contributing trials
  - `macro_f1_mean`, `delta_macro_f1_vs_dummy_mean` = mean over contributing trials
    (NaN if 0)
  - `coverage_rate = n_trials_contributing / n_trials_expected`
  - `positive_delta = (n_trials_contributing > 0) and (delta_mean > 0)`
  - `coverage_status = "ok" if n_trials_contributing > 0 else "insufficient"`
- Write `08x_per_ticker.csv` (overwrite). Add `per_ticker_sha256` to the run
  manifest provenance.

### 4.4 Contract (SANCTIONED) — `contracts/deep_sequence_exploration.py`
- `OUTPUT_FILES_08X += ("08x_per_ticker.csv",)` (9th artifact).
- `PER_TICKER_REQUIRED_COLUMNS` (set) + `validate_08x_per_ticker_frame(df, *,
  require_non_empty=False)`: columns present; `(candidate_id, ticker)` pairs unique
  + non-empty; integer dtype for `n_rows_total/n_trials_expected/n_trials_contributing`
  with `0 <= contributing <= expected`, `expected > 0`; `coverage_rate` numeric in
  `[0, 1]`; `positive_delta` boolean; `coverage_status ∈ {"ok","insufficient"}`. The
  empty branch returns unless `require_non_empty`.

### 4.5 `deep_sequence_schema_smoke.py`
- `PER_TICKER_COLUMNS` ordered tuple (matching the contract set) + a header writer +
  an entry in `SCHEMA_SMOKE_ARTIFACT_SPECS` + the import-time tuple/set drift check
  (mirroring `FOLD_RESULTS_COLUMNS`). `write_schema_smoke_artifacts` now lays the
  per-ticker header so every path (schema-smoke / build-folds / quick-search) keeps
  the 9-artifact bundle complete.

## 5. Test plan (synthetic only)
- `compute_per_ticker_delta`: both-class ticker → finite macro_f1/delta; single-class
  ticker → both_classes_present=False + NaN, NO raise; multi-ticker split correct.
- tuple return: `collect_per_ticker=True` on a completed fit → `(row, rows)` with one
  entry per ticker present in val; failed/skipped fit → `[]`; default call still
  returns a bare dict (regression guard).
- quick-search per-ticker artifact: complete `candidate × expected_ticker` grid;
  a ticker single-class in every fold → `coverage_status="insufficient"`,
  `positive_delta=False`; `validate_08x_per_ticker_frame` passes.
- class-collapse contamination: a collapsed (→failed) trial contributes NO
  per-ticker rows.
- schema-smoke / dispatcher: the 9-artifact bundle completeness still passes;
  `OUTPUT_FILES_08X` now has `08x_per_ticker.csv`.

## 6. Files
- M `src/intraday_research/models/deep_sequence/metrics.py`
- M `src/intraday_research/stages/deep_sequence_trial.py`
- M `src/intraday_research/stages/deep_sequence_quick_search.py`
- M `src/intraday_research/contracts/deep_sequence_exploration.py` (sanctioned)
- M `src/intraday_research/stages/deep_sequence_schema_smoke.py`
- M tests: test_trial_eval_metrics / test_deep_sequence_trial /
  test_deep_sequence_quick_search / schema-smoke + contract tests as needed.
