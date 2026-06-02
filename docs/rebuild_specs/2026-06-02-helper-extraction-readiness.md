# Helper Extraction Readiness Note

Date: 2026-06-02
Status: design note; followed by minimal P1 extraction.
Scope: P1 helper extraction readiness for `notebooks/04_ian_research_memo.ipynb`.

## Boundary

This note is based on static notebook inspection only. It does not execute the
notebook, train models, read closed holdout/test metrics, mutate raw data, or
import archived helper libraries.

Notebook cell numbers below are zero-based `nbformat` indexes.

The active notebook may keep logic inline until the helper boundary is proven by
tests. Any helper extraction must stay smaller than the old archived helper
library and must not restore old runner, PM, phase, or route-control machinery.

## Current Follow-Up

The tests-first P1 slice has since been implemented in:

```text
intraday_research/__init__.py
intraday_research/baseline_v1.py
tests/test_baseline_v1_helpers.py
```

The extraction is limited to label, split, scaler, window, and stratified dummy
helpers. It does not include data loading, feature engineering, plotting,
LightGBM, MS-DLinear+TCN, checkpoint readers, artifact readers, or archived
runner imports.

Post-review P1 fixes added single-ticker guards to the label, split-boundary,
and segment-window helpers. Pooled multi-ticker frames are rejected at these
single-ticker boundaries instead of relying only on caller discipline.

The active notebook has also been updated to call the tested helper module for
label construction, split-boundary invalidation, train-only preprocessing,
window construction, and stratified dummy evaluation. The notebook keeps data
loading, feature engineering, plotting, and interpretation inline.

Notebook integration is guarded by `tests/test_notebook_static_gate.py`, which
checks that saved outputs remain empty, `RUN_*` flags remain false, forbidden
imports stay absent, duplicate safety-critical helper definitions stay out of
the notebook, and raw-feature fallback code is not reintroduced.

## Candidate Functions

| Notebook Cell | Function | Candidate Status | Reason |
|---:|---|---|---|
| 7 | `find_timestamp_column` | keep inline for now | data-loading convenience, not safety-critical enough yet |
| 7 | `load_ticker_csv` | keep inline for now | path and notebook data contract are still memo-specific |
| 7 | `audit_ticker_frame` | keep inline for now | coverage display, not model safety logic |
| 7 | `audit_all_tickers` | keep inline for now | notebook diagnostics wrapper |
| 9 | `grouped_rolling` | extract only with feature tests | reusable causal rolling primitive, but API should be proven first |
| 9 | `grouped_ewm` | extract only with feature tests | reusable causal EWM primitive, but API should be proven first |
| 9 | `add_baseline_v1_features` | extract after feature-contract tests | central baseline contract; high value, high leakage risk |
| 11 | `make_no_trade_band_labels` | extract first | safety-critical label semantics and invalid-marker behavior |
| 13 | `assign_calendar_split` | extract first | split contract is shared by validation and future helpers |
| 13 | `add_split_and_invalidate_boundaries` | extract first | prevents label horizons crossing split boundaries |
| 15 | `fit_train_only_scaler` | extract first | train-only preprocessing invariant is critical |
| 15 | `transform_train_and_validation` | extract first | must forbid closed holdout/test transform in validation-only work |
| 17 | `build_windows_for_segment` | extract first | enforces per-split window construction and invalid-label skipping |
| 17 | `build_windows_by_ticker_and_split` | extract first | prevents windows spanning tickers or splits |
| 19 | `summarize_window_class_balance` | keep inline or extract later | reporting helper; not a prerequisite for safe validation |
| 21 | `evaluate_stratified_dummy` | extract after metric tests | required comparison baseline; low complexity but claim-sensitive |
| 21 | `pooled_train_validation_labels` | extract only if model harness needs it | adapter convenience, should follow model-harness design |

## Recommended First Extraction Slice

Design tests first for this minimal helper slice:

1. `make_no_trade_band_labels`
2. `assign_calendar_split`
3. `add_split_and_invalidate_boundaries`
4. `fit_train_only_scaler`
5. `transform_train_and_validation`
6. `build_windows_for_segment`
7. `build_windows_by_ticker_and_split`
8. `evaluate_stratified_dummy`

Do not extract LightGBM, MS-DLinear+TCN, archive runners, checkpoint readers, or
plotting/reporting utilities in the first helper pass.

## Required Tests Before Extraction

| Test Area | Required Assertion |
|---|---|
| label horizon | future cumulative return over the approved horizon leaves insufficient future rows invalid |
| no-trade band | returns inside the threshold become invalid markers, not class labels |
| split assignment | train and validation are chronological and closed holdout/test is not an evaluation split |
| split-boundary invalidation | labels whose horizon crosses a split boundary are invalidated before window construction |
| train-only scaler | scaler `.fit` sees train rows only, never validation or closed holdout/test |
| transform scope | validation-only transform applies only to train and validation rows |
| ticker isolation | windows never combine rows from different tickers |
| split isolation | windows never cross split boundaries |
| invalid labels | invalid labels are skipped, not filled or globally dropped before boundary checks |
| dummy baseline | stratified dummy uses train class probabilities and reports macro F1 plus balanced accuracy |

## Blocker Checks

No helper extraction should start if any item is true:

- notebook code imports `archive`, `legacy_model_runner`, `ml_utils`, or old
  runner utilities;
- smoke metric values are used as test expected values or decision rules;
- closed holdout/test rows are transformed, scored, summarized, or selected on;
- helper APIs would need checkpoint/artifact readers;
- helper extraction would require restoring a broad framework package.

Current blockers before extraction:

- `build_windows_for_segment` can fall back to raw feature columns when scaled
  columns are absent. Tests must either forbid that fallback for model windows
  or define the fallback as diagnostic-only.
- `summarize_window_class_balance` is still a reporting helper with a light
  schema. It should not be extracted before skipped-window and invalid-label
  accounting are made explicit.
- `evaluate_stratified_dummy` has a useful core, but its output schema and
  aggregate reporting contract should be tested before it becomes an active
  helper.

## Recommendation

Recommendation: design tests first, then extract the minimal safety-critical
slice.

The notebook has enough stable inline logic to justify a P1 helper design pass,
but direct extraction without tests would risk turning notebook memo code into a
new unverified framework. The first implementation task should create focused
tests for label, split, scaler, window, and dummy behavior using small synthetic
frames. Only after those tests exist should the matching functions move from the
notebook into a small active helper module.
