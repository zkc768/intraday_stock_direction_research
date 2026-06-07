# Model Family Screening Protocol - 2026-06-04

Version: 1.1
Frozen: 2026-06-04
Scope: validation_only
Target notebook: `notebooks/03_model_family_screening_colab.ipynb`

Changelog:

- 1.0 (2026-06-04): Initial protocol for Notebook 03 model-family screening.
- 1.1 (2026-06-04): Add completed Stage 0 context and interpretation caveats;
  no model panel, gate, label, feature, or window rule changed.

This is a technical research protocol, not a PM handoff, not a holdout/test
authorization, and not a thesis result claim. Its purpose is to lock the design
for the next validation-only notebook before Notebook 03 is written or run.

## 1. Purpose

Notebook 03 answers one narrow question:

```text
Given the Stage 0 validation-selected configuration candidates, does any
pre-registered model family show a stable validation-only signal over simple
dummy baselines?
```

Notebook 03 does not screen new labels, feature sets, windows, thresholds, or
architectures. It is a fixed-default model-family screen on Stage 0 candidates.

Weak, near-dummy, mixed, or failed model behavior is a valid outcome. Notebook
03 must not force a winner when the evidence does not support one.

## 2. Entry Conditions

Notebook 03 may start only after:

1. `notebooks/02_config_screening_colab.ipynb` has completed Stage 0A2.
2. Stage 0 has produced at least one final configuration candidate:
   `mean_candidate` and/or `lcb_candidate`.
3. Stage 0B has been run or explicitly reported as unavailable/blocking.
4. Stage 0 outputs have been saved from the Colab runtime.

If Stage 0 reports:

```text
do_not_decide_config
```

then Notebook 03 must not run.

Stage 0B is diagnostic only. A `deep_model_disagrees=True` flag does not block
Notebook 03; it must be carried into the Notebook 03 interpretation as context.

The final holdout/test remains closed. Notebook 03 must not read, transform,
window, score, summarize, or use holdout/test rows.

## 2.1 Completed Stage 0 Context

The completed Stage 0 desktop review selected:

```text
h03_bps1p5 + price_volume_time + window_size=20
```

Notebook 03 must carry the following Stage 0 context forward without changing
the fixed Notebook 03 model panel:

- Window 20 was selected by the frozen Stage 0A2 rule, but its margin over
  windows 10 and 5 was below 0.002 macro F1. Describe it as
  protocol-selected and defensible, not as empirically dominant.
- Stage 0B showed LogReg and LightGBM nearly tied, with frozen deep defaults
  below them and `deep_model_disagrees=False`. This supports explicit
  linear/tabular comparison in Notebook 03, but it is not evidence that deep
  model families are generally ineffective.
- Drive follow-up checked `stage0b_pooled__20260604T103737Z.csv`: all five
  Stage 0B LogReg pooled rows have `fit_status=converged`. The remaining issue
  is runtime cost, not convergence failure. Notebook 03 must still report
  convergence status for its own LogReg fits.
- The Stage 0 selected macro F1 is a validation-selected record, not expected
  holdout/test performance. Notebook 03 must not use the Stage 0 value as a
  promised future score.
- Drive follow-up checked the Stage 0 per-ticker CSVs for the selected
  window-20 LightGBM row. All five tickers have positive delta vs dummy, with
  JPM weakest but still positive. Notebook 03 must still report its own
  per-ticker results.

## 3. Candidate Inputs

The only allowed candidate inputs are the Stage 0 final candidate tuples:

```text
(label_config, feature_set, window_size)
```

If `mean_candidate == lcb_candidate`, Notebook 03 runs one candidate.

If `mean_candidate != lcb_candidate`, Notebook 03 runs both candidates. Each
candidate is analyzed independently:

- own model panel,
- own dummy baselines,
- own signal tags,
- own stop rule,
- own validation selection record.

Notebook 03 does not answer which Stage 0 candidate is globally better. If two
candidates both produce `candidate_signal`, both may be sent forward as
separate validation-selected branches for later controlled design review.

## 4. Source And Leakage Boundaries

Notebook 03 inherits the Stage 0 data and preprocessing boundary:

- raw input is the same five ticker files for `CSCO`, `JPM`, `KO`, `MSFT`, and
  `WMT`;
- no new source data are introduced;
- no new feature columns are introduced;
- no new label horizon or threshold is introduced;
- train/validation boundaries remain chronological;
- train-only preprocessing remains train-only;
- labels whose horizons cross split boundaries or trading-day boundaries are
  invalid;
- windows are generated per ticker and per split;
- no window may span tickers, splits, or trading days.

Notebook 03 should be raw-data-first and self-contained. Dataset preparation
functions should be copied from the active Stage 0 notebook implementation, not
imported from stale notebooks or project helper packages.

Any divergence between Notebook 02 and Notebook 03 in feature construction,
label construction, split-boundary invalidation, scaling, or window construction
is a protocol bug unless explicitly pre-registered before Notebook 03 runs.

## 5. Sample Alignment Invariant

All models must be evaluated on the exact same validation target rows in the
same order for a given candidate.

This invariant is the backbone of cross-model fairness:

```text
stratified_dummy y_validation
always_up_dummy  y_validation
tabular models   x_validation_flat, y_validation
sequence models  x_validation_seq,  y_validation
```

The tabular and sequence views must be equal-length and aligned by construction.
Notebook 03 must assert this before fitting models:

```text
len(y_validation_flat) == len(y_validation_seq)
all validation ticker/timestamp/label owners match across views
```

Any dataset-preparation change that breaks this invariant invalidates the
Notebook 03 comparison.

## 6. Fixed Model Panel

Notebook 03 uses the following fixed panel:

| model | role | selection role |
|---|---|---|
| `stratified_dummy` | stochastic dummy baseline from train label prior | gate baseline |
| `always_up_dummy` | deterministic constant-up baseline requested as sanity check | report-only baseline |
| `logreg` | linear tabular baseline | candidate model |
| `lightgbm` | tree tabular baseline and Ian-aligned model | candidate model |
| `vanilla_lstm` | recurrent sequence baseline | controlled baseline |
| `simple_gru` | recurrent sequence baseline inherited from Stage 0B | controlled baseline |
| `standalone_tcn` | convolutional sequence baseline | controlled baseline |
| `standard_dlinear` | simple DLinear-style reference baseline | completeness baseline |
| `ms_dlinear_tcn` | Ian-aligned multi-scale DLinear + residual TCN model | candidate model |

Do not add PatchTST, InceptionTime, DeepLOB, CNN-GRU, CNN-LSTM, NLP/news
features, sentiment models, or external market features to Notebook 03. Adding
a model after seeing Notebook 03 validation results is post-hoc expansion and
invalidates the fair-screening interpretation.

## 7. Fixed Hyperparameters

Notebook 03 does not tune model hyperparameters.

Tabular defaults:

```text
LogReg:
  inherit Notebook 02 LogisticRegression settings exactly

LightGBM:
  inherit Notebook 02 LGBMClassifier settings exactly
```

Torch defaults:

```text
epochs = 8
batch_size = 1024
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.10
optimizer = AdamW
loss = class-weighted CrossEntropyLoss using train labels
early_stopping = disabled
seeds = 101, 202, 303, 404, 505
```

Model-specific defaults:

```text
vanilla_lstm:
  hidden_dim = 32
  num_layers = 1
  external dropout after final recurrent output
  linear head to 2 logits

simple_gru:
  inherit Notebook 02 simple GRU architecture

standalone_tcn:
  new standalone classifier
  reuse Notebook 02 CausalConvBlock building block
  two causal blocks with channels=(32, 32), kernel_size=3, dropout=0.10
  linear head from final TCN timestep to 2 logits

standard_dlinear:
  moving_avg_kernel = 5
  individual = False
  linear head to 2 logits
  marked as completeness baseline, not expected to outperform

ms_dlinear_tcn:
  inherit Notebook 02 MS-DLinear+TCN architecture exactly
  moving_avg_kernels = (3, 5, 9, 15)
  TCN channels=(32, 32), kernel_size=3, dropout=0.10
```

If a Stage 0 candidate has `window_size=5`, `standard_dlinear` with
`moving_avg_kernel=5` degenerates to a window-level mean-centering boundary
case. This limitation must be noted in the notebook interpretation.

If a Stage 0 candidate has `window_size < 15`, some `ms_dlinear_tcn`
moving-average kernels exceed the input window and depend on replicate padding.
This limitation exists in the frozen Stage 0 model and must be reported rather
than silently altered in Notebook 03.

## 8. Baselines

`stratified_dummy` is the gate baseline. It must be run with the same five seeds
as the model panel and evaluated on the same validation rows.

`always_up_dummy` predicts label `1` for every validation row. It is
deterministic, so its cross-seed standard deviation is expected to be zero. This
is not a bug.

Always report both baselines, but only `stratified_dummy` participates in the
Notebook 03 signal gates.

## 9. Required Row-Level Output Schema

Notebook 03 pooled and per-ticker outputs should preserve Stage 0 column
semantics where possible.

Required row-level columns:

```text
stage
model
candidate_id
label_config
horizon_k
threshold_bps
feature_set
window_size
seed
scope
ticker_or_pooled
n
macro_f1
balanced_accuracy
accuracy
stratified_dummy_macro_f1
stratified_dummy_balanced_accuracy
delta_macro_f1_vs_stratified_dummy
delta_balanced_accuracy_vs_stratified_dummy
always_up_dummy_macro_f1
always_up_dummy_balanced_accuracy
delta_macro_f1_vs_always_up_dummy
delta_balanced_accuracy_vs_always_up_dummy
pred_up_pct
pred_down_pct
one_class_collapse
cm_tn
cm_fp
cm_fn
cm_tp
prep_seconds
fit_seconds
predict_seconds
total_seconds
fit_status
run_failed
failure_reason
```

`scope` must be:

```text
validation_only
```

Confusion matrices must be stored as four numeric columns, not as JSON strings.

## 10. Required Summary Schema

For each `(candidate_id, model, label_config, feature_set, window_size)` summary
cell, report:

```text
seed_count
n_failed_seeds
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
balanced_accuracy_std
stratified_dummy_macro_f1_mean
stratified_dummy_macro_f1_std
delta_macro_f1_vs_stratified_dummy_mean
delta_balanced_accuracy_vs_stratified_dummy_mean
always_up_dummy_macro_f1_mean
delta_macro_f1_vs_always_up_dummy_mean
n_mean
positive_ticker_count
top_ticker_gain_share
pred_up_pct_mean
pred_down_pct_mean
one_class_collapse_any
run_failed
failure_reason
signal_strength_tag
macro_f1_bootstrap_ci_lower
macro_f1_bootstrap_ci_upper
```

Use the same one-sided 95% seed-level lower confidence bound convention as
Stage 0:

```text
macro_f1_lcb_95 =
  macro_f1_mean - t_critical_one_sided_95(seed_count)
  * macro_f1_std / sqrt(seed_count)
```

For the default five seeds, `t_critical_one_sided_95 = 2.132`.

## 11. Stage 0-Compatible Gates

Notebook 03 uses Stage 0-compatible gates, with explicit `stratified_dummy`
names because Notebook 03 includes multiple dummy baselines.

Define:

```text
basic_gate :=
  delta_macro_f1_vs_stratified_dummy_mean > 0
  AND macro_f1_lcb_95 > stratified_dummy_macro_f1_mean
```

Define:

```text
lcb_eligible :=
  basic_gate
  AND delta_balanced_accuracy_vs_stratified_dummy_mean > 0
  AND top_ticker_gain_share < 0.50
  AND positive_ticker_count >= 3
```

Accuracy is auxiliary only and cannot choose, reject, rerank, or promote a
model.

## 12. Weak-Signal Protocol

Notebook 03 treats weak or near-dummy validation performance as a valid negative
result.

Define the minimum practical effect size:

```text
MIN_PRACTICAL_DELTA_MACRO_F1 = 0.005
```

This is not a formal statistical significance threshold. It is a practical
guardrail against promoting tiny validation deltas as model evidence.

Define collapse:

```text
one_class_collapse :=
  pred_up_pct > 0.95
  OR pred_down_pct > 0.95
```

Define seed instability when at least two successful seeds exist:

```text
seed_unstable :=
  macro_f1_std >= MIN_PRACTICAL_DELTA_MACRO_F1 / 2
```

Define mutually exclusive and exhaustive signal tags in this order:

```text
run_failed:
  n_failed_seeds >= 3

no_signal:
  n_failed_seeds < 3
  AND basic_gate failed

unstable_signal:
  n_failed_seeds < 3
  AND basic_gate passed
  AND (
    one_class_collapse_any
    OR top_ticker_gain_share >= 0.50
    OR positive_ticker_count < 3
    OR seed_unstable
  )

candidate_signal:
  n_failed_seeds < 3
  AND lcb_eligible passed
  AND delta_macro_f1_vs_stratified_dummy_mean >= MIN_PRACTICAL_DELTA_MACRO_F1
  AND one_class_collapse_any is False
  AND seed_unstable is False

near_dummy:
  all remaining non-failed rows
```

No model may proceed as a candidate solely because it is the least bad model in
the panel.

## 13. Bootstrap Confidence Intervals

Full-panel bootstrap is not part of the default Notebook 03 run.

Notebook 03 should expose these switches:

```text
BOOTSTRAP_CI_FULL_PANEL = False
BOOTSTRAP_CI_FOR_CANDIDATES = True
BOOTSTRAP_RESAMPLES = 1000
```

Default behavior:

1. Run the fixed model panel.
2. Compute summary gates and `signal_strength_tag`.
3. For each `(candidate_id, model)` summary row tagged `candidate_signal`,
   compute bootstrap confidence intervals for successful seed rows only.
4. Leave bootstrap CI columns as `NaN` for non-candidate rows unless
   `BOOTSTRAP_CI_FULL_PANEL=True`.

For a candidate-signal summary row:

- bootstrap each successful seed's validation predictions by resampling
  validation examples with replacement;
- compute macro F1 for each bootstrap resample;
- store per-seed bootstrap lower/upper values in the pooled output if retained;
- summarize conservatively as:

```text
macro_f1_bootstrap_ci_lower = min(seed_bootstrap_ci_lower)
macro_f1_bootstrap_ci_upper = max(seed_bootstrap_ci_upper)
```

Bootstrap CI is a sampling-level diagnostic. It does not replace the
pre-registered Stage 0-compatible gates.

## 14. Failure Handling

Every attempted model-seed run must leave an audit row.

Per seed:

- `run_failed=False` when metrics are valid.
- `run_failed=True` when fitting or prediction fails, loss becomes NaN,
  predictions contain NaN, runtime OOMs, or required inputs are empty.
- `failure_reason` must contain a concise reason when `run_failed=True`.
- failed metric fields are `NaN`.

Per `(candidate_id, model)` summary:

```text
n_failed_seeds <= 2:
  aggregate successful seeds only
  keep n_failed_seeds in the summary
  do not auto-mark the model as failed

n_failed_seeds >= 3:
  signal_strength_tag = run_failed
  exclude from selection

n_failed_seeds == 5:
  keep the summary row with metric fields NaN
```

Do not change batch size, epoch count, model architecture, or hyperparameters
after a failure inside Notebook 03. Any retry policy would require a new
pre-registered protocol.

## 15. Multiple-Comparison Guardrail

Notebook 03 may evaluate up to:

```text
9 models x 1-2 candidates x 5 seeds = 45-90 model-seed rows
```

With this many comparisons, at least one row can appear to beat dummy by chance.
Therefore:

- the full panel must be reported;
- all `signal_strength_tag` values must be shown;
- no result table may show only winners;
- post-hoc model additions are forbidden;
- post-hoc hyperparameter changes are forbidden;
- `candidate_signal` requires both Stage 0-compatible gates and the practical
  delta rule.

## 16. Selection And Stop Rules

Notebook 03 produces validation selection records, not holdout/test-ready
claims.

If no `(candidate_id, model)` summary row has:

```text
signal_strength_tag = candidate_signal
```

then Notebook 03 stops with:

```text
no_validation_signal_under_current_config
```

In that case:

- do not add models;
- do not tune LSTM, GRU, TCN, DLinear, LightGBM, or MS-DLinear+TCN;
- do not change labels, features, thresholds, or windows inside Notebook 03;
- do not open holdout/test;
- start any next feature/label/ticker branch only under a new pre-registration.

If at least one `candidate_signal` exists, write:

```text
notebook03_validation_selection.json
```

The JSON should contain validation-selected branches, not a single forced
winner:

```json
{
  "scope": "validation_only",
  "selection_status": "candidate_signal_found",
  "selected_branches": [
    {
      "candidate_id": "mean_candidate",
      "label_config": "...",
      "feature_set": "...",
      "window_size": 10,
      "model": "lightgbm",
      "signal_strength_tag": "candidate_signal",
      "fixed_params": {},
      "validation_macro_f1_mean": null,
      "validation_macro_f1_lcb_95": null,
      "delta_macro_f1_vs_stratified_dummy_mean": null
    }
  ],
  "holdout_test_authorized": false
}
```

If only a controlled baseline such as `vanilla_lstm`, `simple_gru`,
`standalone_tcn`, or `standard_dlinear` produces `candidate_signal`, the next
step is a design review before tuning. Ian's prior guidance treats LSTM-style
sequence models as baselines, not as an automatic tuning priority.

## 17. Output Artifacts

Notebook 03 writes outputs first to a local Colab runtime directory:

```text
/content/notebook03_model_family_screening_results
```

Required artifacts:

```text
notebook03_pooled.csv
notebook03_per_ticker.csv
notebook03_summary.csv
notebook03_validation_selection.json
notebook03_run_manifest.json
```

Copying results to Drive is allowed only through an explicit save cell, mirroring
the Stage 0 notebook style. Do not mount MyDrive in the default setup cell.

## 18. Interpretation Rules

Notebook 03 interpretation must include:

- whether each candidate produced any `candidate_signal`;
- whether each model was `no_signal`, `near_dummy`, `unstable_signal`,
  `candidate_signal`, or `run_failed`;
- pooled and per-ticker results;
- dummy baseline deltas;
- collapse diagnostics;
- sample counts;
- limitations from short windows and DLinear/MS-DLinear kernels;
- Stage 0 context caveats: marginal window-20 advantage, near-tie between
  LogReg and LightGBM, LogReg convergence status, and validation-selection
  bias in the Stage 0 macro F1;
- the explicit statement that holdout/test remains closed.

Allowed wording examples:

```text
This validation-only screen found no candidate_signal under the current Stage 0
configuration. This is a valid negative result and does not justify post-hoc
model expansion inside Notebook 03.
```

```text
This validation-only screen found candidate_signal for the listed branch under
fixed defaults. This does not authorize holdout/test evaluation or post-hoc
tuning; it only identifies a branch for controlled follow-up design.
```

Forbidden wording examples:

```text
The model works.
The model beats the market.
The best model is ready for holdout.
The weak delta is still promising because it is the best row.
```

## 19. Literature And Design Anchors

These references are design anchors only. They do not supply project-specific
performance evidence.

- Ojala and Garriga, "Permutation Tests for Studying Classifier Performance",
  JMLR 2010: null-control thinking for classifier performance.
  <https://jmlr.org/papers/v11/ojala10a.html>
- scikit-learn `DummyClassifier`: dummy predictors as first-class baselines.
  <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>
- scikit-learn `TimeSeriesSplit`: chronological split discipline for
  time-ordered data.
  <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>
- Bailey, Borwein, Lopez de Prado, and Zhu, "The Probability of Backtest
  Overfitting": multiple-trial guardrail for financial model screening.
  <https://escholarship.org/uc/item/4w1110bb>
- Bailey and Lopez de Prado, "The Deflated Sharpe Ratio": selection-bias and
  overfitting caution for financial backtests.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks":
  risk-coverage and reject-option framing.
  <https://arxiv.org/abs/1705.08500>
- "Trading via Selective Classification": financial abstention/no-trade
  framing.
  <https://arxiv.org/abs/2110.14914>
- Bai, Kolter, and Koltun, "An Empirical Evaluation of Generic Convolutional
  and Recurrent Networks for Sequence Modeling": TCN as a generic sequence
  baseline.
  <https://arxiv.org/abs/1803.01271>
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?":
  DLinear/linear-family baseline discipline.
  <https://arxiv.org/abs/2205.13504>

## 20. Non-Goals

Notebook 03 is not:

- a hyperparameter tuning notebook;
- a feature engineering notebook;
- a label redesign notebook;
- a holdout/test notebook;
- a trading strategy or PnL notebook;
- a paper-claim notebook;
- a broad model zoo.

Any work in those categories requires a separate pre-registered protocol.
