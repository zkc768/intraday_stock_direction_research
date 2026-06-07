# Config Screening Freeze - 2026-06-04

Scope: validation_only.

This freeze resets Stage 0 to a raw-data-first configuration screen. Active
candidate names and candidate sets are defined only in this file.

The goal is to choose a small number of validation-selected
`label_config + feature_set + window_size` candidates for the next model-family
screening notebook without turning validation into an unconstrained tuning
surface.

## Active Boundary

Earlier local screening notes and route files are not part of the active
project surface. Stage 0 starts from the raw ticker files and the candidate
space below.

The active notebook path is:

```text
notebooks/02_config_screening_colab.ipynb
```

The Colab runtime output path is:

```text
/content/stage0_config_screening_results
```

## Source Boundary

- Raw input comes only from the five Google Drive `.txt` files listed in the
  active notebook.
- The five tickers are `CSCO`, `JPM`, `KO`, `MSFT`, and `WMT`.
- Raw OHLCV columns are source data, not model features.
- Model features must be derived from current or prior completed bars only.
- No holdout/test rows may be read, transformed, windowed, scored, summarized,
  or used for wording decisions.

## Hard Boundary

Stage 0 is validation-only:

- Do not change train/validation boundaries.
- Do not add raw non-stationary features.
- Do not use any external feature-selection result to add, remove, rank, or describe
  active candidates.
- Do not tune deep-model hidden size, layers, epoch count, optimizer, or
  architecture inside Stage 0.
- Do not promote Stage 0 output to thesis evidence. Stage 0 only selects
  validation-selected candidates for later model-family screening.
- If runtime is too high, stop after the current stage and write down the
  blocker. Do not shrink seeds, remove models, or change the grid after seeing
  validation metrics.

## Stage 0 Overview

Stage 0 has four layers:

```text
Stage 0S  runtime/schema smoke
Stage 0A1 label-feature screen
Stage 0A2 window sensitivity
Stage 0B  deep-model second-view
```

The route follows a cheap-to-expensive order. LogReg and LightGBM are used for
wide screening. GRU and MS-DLinear+TCN are used only on short-listed
candidates.

## Candidate Space

The active label configs are new neutral protocol labels:

| label_config | horizon_k | threshold_bps | design role |
|---|---:|---:|---|
| `h03_bps1p5` | `3` | `1.5` | short horizon, narrow no-trade band |
| `h09_bps3p0` | `9` | `3.0` | middle horizon, moderate no-trade band |
| `h24_bps7p5` | `24` | `7.5` | longer horizon, wider no-trade band |

These labels are hypotheses, not result claims.

The active feature sets are rebuilt from raw OHLCV-derived, stationarity-safer
features:

| feature_set | features | design role |
|---|---|---|
| `price_action_core` | `log_return`, `close_to_open_return`, `high_low_range` | minimal price-action derived set |
| `technical_price` | `log_return`, `high_low_range`, `rsi_14`, `bollinger_pctb`, `normalized_macd_hist` | price-derived technical set |
| `price_volume_time` | `log_return`, `close_to_open_return`, `high_low_range`, `rolling_volatility_20`, `normalized_volume_20`, `rsi_14`, `bollinger_pctb`, `normalized_macd_hist`, `time_of_day_sin`, `time_of_day_cos` | broader derived set with volume and time encoding |

These are candidate input groups. Their names do not imply performance,
stability, or priority.

Feature timing boundary:

- Prediction is defined after the current five-minute bar has completed, so
  `open[t]`, `high[t]`, `low[t]`, `close[t]`, `volume[t]`, and timestamp-derived
  values at `t` are allowed.
- `bollinger_pctb` uses a same-day trailing window that includes `close[t]`;
  this is allowed under the current-completed-bar boundary and is not a
  look-ahead feature.
- `rolling_volatility_20` uses shifted same-day history, so its rolling
  volatility baseline excludes the current bar.
- `normalized_volume_20` is `log_volume[t]` minus the prior 20-bar same-day
  mean of shifted log volume. The baseline excludes the current bar, while the
  current completed bar's volume is intentionally present on the left side.
- `rsi_14` and `normalized_macd_hist` use single-pass causal EWM/Wilder state
  across the full per-ticker chronological series. This state carries overnight
  by design; it is not a same-day-only rolling feature.
- At the train/validation split boundary, validation RSI/MACD state may carry
  information from earlier train rows. This is allowed past-to-future state, not
  validation leakage, but validation should not be described as a
  self-contained technical-indicator state period.
- Future label fields, invalid markers, diagnostic fields, raw OHLCV columns,
  and split markers must not be included in `FEATURE_SETS`.

The active seeds are:

```text
101, 202, 303, 404, 505
```

## Model Input And Fixed Model Parameters

Tabular models use flattened windows. For LogReg and LightGBM, each sample is
the past `window_size` five-minute bars flattened into:

```text
window_size * n_features
```

The target is the label at the final bar of that input window. Tabular and
sequence windows must be built from the same per-ticker, per-split, per-day
candidate rows so that sample counts, labels, ticker owners, and timestamps
match.

This makes `window_size` a real model input hyperparameter for LogReg and
LightGBM, not only a sample-count side effect.

Fixed tabular model parameters:

```text
LogReg:
  estimator = sklearn LogisticRegression
  solver = liblinear
  class_weight = balanced
  max_iter = 2000
  random_state = seed

LightGBM:
  n_estimators = 200
  learning_rate = 0.03
  max_depth = 6
  num_leaves = 31
  subsample = 0.9
  subsample_freq = 1
  colsample_bytree = 0.9
  class_weight = balanced
  random_state = seed
```

## Stage 0S - Runtime And Schema Smoke

Stage 0S is optional but recommended before a full Stage 0 run.

```text
model        = LightGBM
label_config = h09_bps3p0
feature_set  = price_volume_time
window_size  = 10
seed         = 101
```

Purpose:

- verify Colab data loading,
- verify train-only preprocessing,
- verify validation-only output schema,
- measure rough runtime,
- catch missing adapters or Drive API issues.

Stage 0S is not a selection step. Its metrics must not choose, reject, or
rerank any Stage 0 candidate. If Stage 0S fails, fix the pipeline or data path
before Stage 0A. Do not inspect holdout/test to debug it.

## Stage 0A1 - Label-Feature Screen

Stage 0A1 fixes `window_size=10` and screens only label and feature set.

```text
models       = LogReg, LightGBM
label_config = h03_bps1p5, h09_bps3p0, h24_bps7p5
feature_set  = price_action_core, technical_price, price_volume_time
window_size  = 10
seeds        = 101, 202, 303, 404, 505
```

Expected rows:

```text
3 labels x 3 feature_sets x 1 window x 2 models x 5 seeds = 90 rows
```

Stage 0A1 outputs up to two label-feature candidates:

- `mean_label_feature`
- `lcb_label_feature`

If both names point to the same label-feature pair, keep only one pair. Do not
force a second candidate.

## Stage 0A2 - Window Sensitivity

Stage 0A2 evaluates window size only after Stage 0A1 has identified the
short-listed label-feature pair or pairs.

```text
model        = LightGBM
label_feature_pairs = union(mean_label_feature, lcb_label_feature)
window_size  = 5, 10, 20
seeds        = 101, 202, 303, 404, 505
```

Expected rows:

```text
15 rows if mean and lcb share one label-feature pair
30 rows if they are different pairs
```

Stage 0A2 outputs final configuration candidates:

- `mean_candidate`
- `lcb_candidate`

Each candidate is a full tuple:

```text
(label_config, feature_set, window_size)
```

## Stage 0B - Deep-Model Second-View

Stage 0B checks whether the selected candidate configurations remain credible
when viewed by sequence models.

```text
candidates = mean_candidate, lcb_candidate
models     = LogReg, LightGBM, simple GRU, MS-DLinear+TCN
seeds      = 101, 202, 303, 404, 505
```

Expected rows:

```text
2 candidates x 4 models x 5 seeds = 40 rows
```

Expected deep-model runs:

```text
2 candidates x 2 deep models x 5 seeds = 20 deep runs
```

Stage 0B is not a second selection pass on validation. It is a second-view
diagnostic. If GRU or MS-DLinear+TCN disagrees with a candidate, report
`deep_model_disagrees=True` and describe the disagreement. Do not use Stage 0B
to search new labels, windows, features, models, or hyperparameters.

## Metrics

Every output row must include:

```text
stage
model
label_config
horizon_k
threshold_bps
feature_set
window_size
seed
scope
macro_f1
balanced_accuracy
accuracy
dummy_macro_f1
dummy_balanced_accuracy
delta_macro_f1_vs_dummy
delta_balanced_accuracy_vs_dummy
n
ticker_or_pooled
prep_seconds
fit_seconds
predict_seconds
total_seconds
```

Per-ticker output must include:

```text
per_ticker_delta_macro_f1_vs_dummy
positive_ticker_count
top_ticker_gain_share
```

Accuracy is auxiliary only. It is reported, but it is not used to choose,
reject, rerank, or reopen a candidate.

## Summary Statistics

For each `(stage, model, label_config, feature_set, window_size)` cell:

```text
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
balanced_accuracy_std
dummy_macro_f1_mean
delta_macro_f1_vs_dummy_mean
delta_balanced_accuracy_vs_dummy_mean
n_mean
positive_ticker_count
top_ticker_gain_share
```

Use sample standard deviation with `ddof=1`.

Use a one-sided 95% t lower confidence bound keyed by the actual number of
distinct seeds in the summary cell:

```text
macro_f1_lcb_95 =
  macro_f1_mean - t_critical_one_sided_95(seed_count)
  * macro_f1_std / sqrt(seed_count)
```

For the default five seeds, `seed_count = 5` and
`t_critical_one_sided_95 = 2.132`. Do not use `1.96` for the seed-level lower
confidence bound. If a cell has only one seed, the LCB equals the mean because
there is no seed-level sample standard deviation.

## Gates And Candidate Selection

Define:

```text
basic_gate :=
  delta_macro_f1_vs_dummy_mean > 0
  AND macro_f1_lcb_95 > dummy_macro_f1_mean
```

If zero Stage 0A cells pass `basic_gate`, the Stage 0 result is:

```text
do_not_decide_config
```

In that case:

- do not relax thresholds,
- do not extend the grid,
- do not run Stage 0B,
- do not start model-family screening.

Define:

```text
lcb_eligible :=
  basic_gate
  AND delta_balanced_accuracy_vs_dummy_mean > 0
  AND top_ticker_gain_share < 0.50
  AND positive_ticker_count >= 3
```

`mean_candidate`:

```text
argmax over basic_gate cells of macro_f1_mean
```

`lcb_candidate`:

```text
argmax over lcb_eligible cells of macro_f1_lcb_95
```

If no cell passes `lcb_eligible`, report `lcb_candidate = null`. Do not invent
an LCB candidate from a failed cell.

If `mean_candidate` and `lcb_candidate` are the same tuple, keep one candidate
and record:

```text
candidate_count = 1
```

## Runtime Budget

The notebook must measure runtime instead of relying on estimates.

Rough planning expectation:

| stage | expected cost |
|---|---:|
| Stage 0S | 1 LightGBM seed run |
| Stage 0A1 | 90 tabular model-seed rows |
| Stage 0A2 | 15-30 LightGBM seed rows |
| Stage 0B | 40 model-seed rows, including 20 deep runs if two candidates survive |

If the runtime is materially higher than expected, stop after the current
stage, inspect timing columns, and decide whether a new freeze is needed. Do
not silently reduce seeds, remove models, or shrink the grid after seeing
performance metrics.

## Stage 1 After Stage 0

Only after Stage 0 has produced candidate configurations may planned notebook
03 start model-family screening.

Notebook 03 must take Stage 0 candidates as inputs. It must not introduce a
fixed primary lane before Stage 0 independently selects candidates under the
rules above.

## Validation Budget Log

| date | source | validation signal used | locked decision |
|---|---|---|---|
| 2026-06-04 | this freeze | none yet | raw-data-first Stage 0 candidate space, gates, metrics, and stop rules |
| 2026-06-04 | notebook/generator sync | none used | tabular flattened-window input, fixed LogReg/LightGBM parameters, and dynamic one-sided 95% LCB gate |
