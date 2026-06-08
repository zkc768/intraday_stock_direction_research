# N08 #5E-2 — Trial-Eval Metrics (`deep_sequence/metrics.py`) Design

> Status: design 2026-06-07. First 08X-EXECUTION sub-piece (user-selected Option
> 1): a self-contained module computing the §8.3 trial-ledger METRIC columns from
> predictions. Tooling: inline design + `humanize:ask-codex` review.
> Coexistence: Codex owns the 08X CONTRACT surface (12 validators incl.
> `validate_trial_ledger_frame`, which validates the ledger SCHEMA, not values).
> This module PRODUCES the metric VALUES — complementary, no contract edit, no
> `run_stage` (gated). Check `git status` clean + log before editing.

## 1. Goal & Scope

Implement a new `src/intraday_research/models/deep_sequence/metrics.py` computing
the eight per-trial metric columns the contract's `REQUIRED_TRIAL_LEDGER_COLUMNS`
requires (design §8.3):
`macro_f1`, `balanced_accuracy`, `accuracy`,
`stratified_dummy_macro_f1_same_rows`, `delta_macro_f1_vs_dummy`,
`class0_pred_rate`, `class1_pred_rate`, `ticker_max_share`.

Pure, self-contained, DETERMINISTIC functions over `(y_true, y_pred,
ticker_ids)` returning a `dict[str, float]` keyed by exactly those column names,
so the future 08X harness builds a ledger row = bookkeeping cols (its concern) +
`compute_trial_metrics(...)`. **No seed** (Codex P1: the stratified null is
analytical/deterministic, not a seeded draw — see §3). **08X red line (design
§4.1):** these are TRAIN-INNER discovery metrics; nothing here reads or scores
official validation or holdout.

**Out of scope:** no trial loop, no search-space build, no ledger I/O, no
`run_stage`, no contract edit, no data loading. Inputs are the arrays the harness
already has per fold.

## 2. Metric definitions
Module location: `models/deep_sequence/metrics.py` (self-contained, mirrors
`losses.py` / `folds.py`). sklearn (1.4.2, available) via a DEFERRED import with
an explicit dependency error (mirrors `LastStepLightGBMControl`).

| column | definition |
|---|---|
| `macro_f1` | `sklearn.metrics.f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)` (explicit `labels=[0,1]` + zero_division=0 so a class-collapsed candidate scores F1=0 on the empty class, not a warning — Codex P3) |
| `balanced_accuracy` | `balanced_accuracy_score(y_true, y_pred)` |
| `accuracy` | `accuracy_score(y_true, y_pred)` |
| `class0_pred_rate` | `mean(y_pred == 0)` (feeds the §14.4 class-collapse guard, `class_collapse_pred_rate_min = 0.05`) |
| `class1_pred_rate` | `mean(y_pred == 1)` |
| `ticker_max_share` | `max_t count(ticker_ids == t) / n` — the largest single-ticker share of the scored rows (the §8.3 ledger column). NOTE (Codex P2): this is scored-row imbalance only; the fuller "did the gain come from one ticker" concentration analysis (per-ticker delta, positive_ticker_count, top-ticker gain share) is a SEPARATE per-ticker harness artifact, NOT this column — do not overload it. |
| `stratified_dummy_macro_f1_same_rows` | macro_f1 of a stratified null dummy on the SAME rows (see §3) |
| `delta_macro_f1_vs_dummy` | `macro_f1 - stratified_dummy_macro_f1_same_rows` |

All returned as Python `float`.

## 3. Stratified dummy — ANALYTICAL same-row null (Codex P1, resolved)
The candidate must beat a stratified NULL baseline; this delta is the core 08X
selection signal (§9.1 `min_train_inner_lcb_delta_macro_f1 = 0.005`). A seeded
single `DummyClassifier(strategy="stratified")` draw injects arbitrary noise into
the delta denominator — at a 0.005 LCB margin that noise can flip eligibility and
corrupt the §9.2 seed-stability score. So the dummy is computed
**ANALYTICALLY/DETERMINISTICALLY** (no seed):

Using the SAME-ROW class balance of `y_true` (`n = len`, `n_c = count(y_true==c)`,
`q_c = n_c / n`), the expected macro-F1 of a stratified null (predicts class `c`
with probability `q_c`) is, per class,
`F1_null_c = 2 * n_c * q_c / (n_c + n * q_c)` (= `q_c`), so
`stratified_dummy_macro_f1_same_rows = mean(F1_null_0, F1_null_1)` — which equals
**0.5** whenever both classes are present (the body requires that, §4), giving
`delta_macro_f1_vs_dummy = macro_f1 - 0.5`. The module computes the per-class
formula explicitly (transparent + robust) rather than hard-coding 0.5.

**Same-row (val) class balance** is the integrity-safe null (Codex P1-confirmed):
a train prior can be artificially weak under train-inner class-balance drift and
inflate the delta. Deriving ONLY the class balance from `y_true` is a no-feature
predeclared comparator, not leakage. (It is the expected score of a
same-distribution stratified guesser — not a model "fit on validation labels".)

## 4. Validation (fail-loud)
- `y_true` / `y_pred`: 1-D integer (non-bool) ndarrays, equal length, `n >= 1`,
  values ⊆ `{0, 1}`.
- `ticker_ids`: 1-D, same length; no missing/NaN/None entries (Codex P3 — so
  `ticker_max_share` is auditable).
- `y_true` must contain BOTH classes (a single-class val slice makes macro_f1 /
  the stratified-null delta ill-defined — the fold builders already guarantee
  non-degenerate folds; fail loud if violated here too).
- (No `stratified_dummy_seed` — the dummy is deterministic, §3.)

## 5. Files
- **Create** `src/intraday_research/models/deep_sequence/metrics.py`.
- **Create** `tests/stages/models/test_trial_eval_metrics.py`.
- No change to `__init__.py` (a module like `losses`/`folds`, imported as
  `from ...deep_sequence import metrics`), the contract, configs, or `run_stage`.

## 6. Testing (`tests/stages/models/test_trial_eval_metrics.py`)
- **Known values**: hand-computed `macro_f1` / `balanced_accuracy` / `accuracy` /
  `class*_pred_rate` / `ticker_max_share` on a tiny labelled example.
- **Schema match**: returned dict keys == the 8 contract metric columns exactly.
- **Deterministic stratified null**: `stratified_dummy_macro_f1_same_rows == 0.5`
  for any both-class `y_true` (balanced AND imbalanced); `delta_macro_f1_vs_dummy
  == macro_f1 - 0.5`; perfect predictions → `macro_f1 == 1.0`, delta `== 0.5`.
  No seed in the API; repeated calls return identical values.
- **class-collapse**: all-one-class predictions → that class's `pred_rate == 1`,
  the other `== 0` (feeds the 0.05 guard), macro_f1 with `labels=[0,1]`,
  zero_division=0.
- **ticker_max_share**: single ticker → 1.0; balanced 2 tickers → 0.5; skewed
  3:1 → 0.75.
- **Guards**: non-1D / length-mismatch / non-{0,1} / bool y / single-class
  y_true / NaN-or-None ticker id — each `ValueError`.
- Three gates stay green; no contract/__init__ change.

## 7. Open Decisions for Review (Codex-resolved)
1. ~~Stratified-dummy prior source~~ — **resolved**: same-row (val) class balance
   (Codex-confirmed integrity-safe; train prior can inflate the delta).
2. ~~Stratified-dummy estimator~~ — **resolved**: ANALYTICAL expected macro-F1
   (= 0.5 for both classes), deterministic, NO seed (Codex P1 — a seeded draw is
   noise that can flip §9.1 eligibility).
3. ~~Module location~~ — kept `models/deep_sequence/metrics.py` (Codex-sound;
   mirrors losses/folds; harness imports it).
4. ~~Single-class y_true rejected~~ — kept (fail loud).

## 8. Codex Design Review — absorbed (2026-06-07, gpt-5.5:high, 218s)
- **P1 (dummy estimator):** §3 now ANALYTICAL deterministic same-row stratified
  null (`F1_null_c = 2·n_c·q_c/(n_c+n·q_c)` → mean = 0.5 for both classes); the
  `stratified_dummy_seed` param is REMOVED.
- **P1 (prior source):** same-row (val) balance confirmed; prose reworded (a
  no-feature comparator, not a model "fit on val labels").
- **P2 (ticker_max_share scope):** kept as the §8.3 row-share column; documented
  that fuller per-ticker concentration (per-ticker delta / positive_ticker_count)
  is a SEPARATE harness artifact, not this column.
- **P2 (schema):** the 8 keys match the 08X ledger exactly; use the 08X name
  `delta_macro_f1_vs_dummy` (the `…_vs_stratified_dummy_same_rows` form is 08O).
- **P3:** `f1_score(labels=[0,1], …)`; reject NaN/None ticker ids.
- **Confirmed:** module location, single-class rejection, deferred-sklearn import,
  and the macro_f1/balanced_accuracy/accuracy/class-rate definitions.
