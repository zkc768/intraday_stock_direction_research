# Diagnostic H0 Tabular Sweep Protocol - 2026-06-04

Version: 1.0
Frozen: 2026-06-04
Scope: validation_only_diagnostic
Target notebook: `notebooks/02_diagnostic_h0_tabular_sweep_colab.ipynb`

Changelog:

- 1.0 (2026-06-04): Initial diagnostic protocol for post-Stage 0 tabular
  window and LightGBM profile coverage.

This is a technical diagnostic protocol, not an official Stage 0 replacement,
not a holdout/test authorization, and not a thesis result claim. Its purpose is
to test whether the active Stage 0 route missed a materially better tabular
configuration because `window_size=24`, longer input windows, or a small number
of LightGBM capacity profiles were not included in the official Stage 0A2 grid.

## 1. Purpose

Diagnostic H0 answers three narrow questions:

```text
1. Does a nearby or longer input window such as 12, 24, 32, 48, or 64
   materially outperform the official Stage 0A2 baseline?
2. Does a small, neutral set of LightGBM capacity/regularization profiles
   materially outperform the official LightGBM default?
3. Does a simple linear model respond to window size in the same direction as
   LightGBM?
```

Diagnostic H0 must not:

- replace the official Stage 0A2 or Stage 0B output;
- change official Stage 0 candidate tuples in place;
- open, transform, window, score, summarize, or use holdout/test rows;
- add new labels, thresholds, feature sets, model families, or deep-model
  hyperparameter tuning;
- promote any validation-only diagnostic winner to thesis evidence.

Any H0 finding is only a candidate signal for a future separately
pre-registered branch.

## 2. Entry Conditions

Diagnostic H0 may start only after all of the following are true:

1. `notebooks/02_config_screening_colab.ipynb` has completed Stage 0A2.
2. Stage 0B has completed, or its unavailability has been explicitly recorded
   before H0 starts.
3. The Stage 0 output directory exists:

```text
/content/stage0_config_screening_results/
```

4. These Stage 0 files exist in that directory:

```text
stage0a2_pooled.csv
stage0a2_summary.csv
stage0_candidates.json
stage0b_summary.csv
```

5. The final holdout/test remains closed.

If Stage 0 reports `do_not_decide_config`, H0 must not run.

## 3. Baseline Definition

H0 must not hard-code a baseline macro F1 number.

The H0 notebook must read the official baseline from Stage 0 output files:

```text
h0_baseline_source = /content/stage0_config_screening_results/stage0a2_summary.csv
h0_candidate_source = /content/stage0_config_screening_results/stage0_candidates.json
```

The intended baseline cell is:

```text
model = lightgbm
label_config = h03_bps1p5
feature_set = price_volume_time
window_size = 20
seeds = 101, 202, 303, 404, 505
```

Before running H0, the notebook must verify that the intended baseline appears
as an official Stage 0A2 candidate or stop with a clear error. If Stage 0A2
selects a different official candidate, this H0 protocol must be revised before
any diagnostic sweep starts.

`h0_baseline_macro_f1` is the `macro_f1_mean` for the intended baseline cell,
read from `stage0a2_summary.csv`. The notebook must also verify from
`stage0a2_pooled.csv` that the baseline cell has exactly the expected seed rows.

### Extracted Artifact Fallback

If the H0 notebook is run in a fresh Colab runtime where the official Stage 0
runtime output directory is unavailable, it may read an extracted Stage 0
artifact folder instead. The artifact folder must contain:

```text
stage0a2_table1.csv
stage0b_table1.csv
stage0_decision_blocks.json
```

Expected Colab/Drive locations include:

```text
/content/stage0_desktop_02_config_screening_2026-06-04
/content/drive/MyDrive/intraday_stock_direction_research/artifacts/stage0_desktop_02_config_screening_2026-06-04
/content/drive/MyDrive/intraday_stock_direction_research/stage0_desktop_02_config_screening_2026-06-04
/content/drive/MyDrive/stage0_desktop_02_config_screening_2026-06-04
```

When the extracted artifact path is used:

```text
baseline_source = stage0_extracted_artifact:<path-to-stage0a2_table1.csv>
```

The notebook must verify that `stage0_decision_blocks.json` contains the
intended candidate:

```text
h03_bps1p5 + price_volume_time + window_size=20
```

It must also verify that the artifact `stage0a2_table1.csv` has exactly one
matching baseline row and `seed_count = 5`.

### Standalone Fresh Colab Fallback

If the H0 notebook is run in a fresh Colab runtime where the official Stage 0
CSV/JSON outputs and extracted Stage 0 artifacts are unavailable, the notebook
may enter:

```text
baseline_source = part0_standalone
```

In this fallback mode, Part 0 runs the intended baseline cell first and uses
that Part 0 `macro_f1_mean` as `h0_baseline_macro_f1` for the rest of the H0
diagnostic. This allows the diagnostic sweep to run without fabricating missing
Stage 0 files or hard-coding a baseline number.

Fallback limitation:

```text
part0_standalone cannot claim reproduction of official Stage 0A2 output.
It only defines the within-notebook H0 comparison baseline.
```

All H0 summaries must record `baseline_source` so later interpretation can
distinguish official reproduction from standalone diagnostic comparison.

## 4. Part 0 Sanity Check

Part 0 is mandatory and must pass before any sweep runs.

Run in the new H0 notebook:

```text
stage = Diagnostic H0 Part 0
model = lightgbm
label_config = h03_bps1p5
feature_set = price_volume_time
window_size = 20
params = LightGBM default from Stage 0
seeds = 101, 202, 303, 404, 505
```

Required result:

```text
abs(part0_macro_f1_mean - h0_baseline_macro_f1) <= 0.0001
```

If this tolerance fails, abort H0. A failure means implementation, dependency,
data, seed, or output drift must be debugged before continuing. Do not interpret
any H0 sweep result when Part 0 fails.

## 5. Fixed Data And Feature Contract

H0 uses the same raw-data-first, self-contained data path as the active Stage 0
notebook:

```text
tickers = CSCO, JPM, KO, MSFT, WMT
bar_interval = 5 minutes
label_config = h03_bps1p5
feature_set = price_volume_time
train/validation split = same as Stage 0
scaler = fit on pooled train rows only, transform train and validation
window construction = per ticker, per trading day
label horizon = must not cross trading day or split boundary
scope = validation_only_diagnostic
```

The H0 notebook must not import `intraday_research` or prior notebooks as the
active path. It should copy or generate the active Stage 0 data, feature, label,
split, scaling, and window logic directly.

## 6. Output Isolation

H0 outputs must be written only to:

```text
/content/diagnostic_h0_tabular_sweep/
```

H0 must not write into:

```text
/content/stage0_config_screening_results/
```

Required output files:

```text
diagnostic_h0_pre_committed_rules.json
diagnostic_h0_part0_sanity.csv
diagnostic_h0_part1_window_sweep.csv
diagnostic_h0_part2_lgbm_profiles.csv
diagnostic_h0_part3_confirmation.csv
diagnostic_h0_summary.csv
diagnostic_h0_per_ticker.csv
```

All rows must carry:

```text
scope = validation_only_diagnostic
diagnostic_name = diagnostic_h0_tabular_sweep
```

## 7. Part 1 Window Sweep

Part 1 tests sparse window anchors, not every integer window.

LightGBM default window sweep:

```text
model = lightgbm
params = Stage 0 default
window_size = 6, 12, 16, 20, 24, 28, 32, 48, 64
seeds = 101, 202, 303, 404, 505
```

LogReg window sweep:

```text
model = logreg
penalty = l2
C = 1.0
solver = liblinear
class_weight = balanced
window_size = 12, 20, 24, 32, 48
seeds = 101, 202, 303, 404, 505
```

LogReg `C` variation is intentionally deferred from H0. Part 1 only checks
whether a simple linear model responds to window length in the same broad
direction as LightGBM. If LogReg shows a meaningful window response, a separate
future diagnostic branch may test `C = 0.1, 1.0, 10.0` under a new
pre-registered protocol.

## 8. Round 2 Dense Window Sweep

Round 2 is conditional. It must not be run merely because the operator is
curious after seeing Round 1.

Round 2 trigger rule:

```text
Run Round 2 if and only if at least one of window_size in {24, 28, 32}
from Part 1 LightGBM satisfies all three conditions:

1. delta_macro_f1_vs_base >= 0.005
2. positive_ticker_count >= 4
3. top_ticker_gain_share <= 0.50
```

If the trigger is not met, stop after Part 1 and Part 2. Do not run Round 2.

If triggered, run:

```text
model = lightgbm
params = Stage 0 default
window_size = 18, 20, 22, 24, 26, 28, 30, 32, 36
seeds = 101, 202, 303, 404, 505
```

Round 2 inherits the same interpretation rules as Part 1. The top cells from
Round 1, Round 2, and Part 2 compete together for Part 3 confirmation.

## 9. Part 2 LightGBM Profile Sweep

Part 2 tests a small neutral set of LightGBM profiles. Profile names are
letters, not directional claims.

Run these profiles on:

```text
window_size = 20, 24, 32
seeds = 101, 202, 303, 404, 505
```

Profiles:

```text
profile_A:
  n_estimators = 150
  learning_rate = 0.05
  max_depth = 3
  num_leaves = 7
  subsample = 0.9
  subsample_freq = 1
  colsample_bytree = 0.9
  class_weight = balanced

profile_B:
  n_estimators = 200
  learning_rate = 0.03
  max_depth = 6
  num_leaves = 31
  subsample = 0.9
  subsample_freq = 1
  colsample_bytree = 0.9
  class_weight = balanced

profile_C:
  n_estimators = 300
  learning_rate = 0.02
  max_depth = 8
  num_leaves = 63
  subsample = 0.9
  subsample_freq = 1
  colsample_bytree = 0.9
  class_weight = balanced

profile_D1:
  same as profile_B plus min_child_samples = 100

profile_D2:
  same as profile_B plus reg_lambda = 1.0

profile_D3:
  same as profile_B plus min_child_samples = 100 and reg_lambda = 1.0
```

`profile_D3` intentionally combines two regularization axes after D1 and D2
isolate them. If D3 wins but D1 and D2 do not, report the result as a combined
profile interaction, not as evidence for either single axis.

## 10. Part 3 Confirmation

Part 3 reruns only the apparent winners with fresh seeds.

Select the top 3 diagnostic cells across:

```text
Part 1
Round 2 if triggered
Part 2
```

Ranking key:

```text
delta_macro_f1_vs_base
```

Only cells that satisfy all preliminary gates may enter Part 3:

```text
delta_macro_f1_vs_base >= 0.005
delta_macro_f1_vs_dummy > 0
positive_ticker_count >= 4
top_ticker_gain_share <= 0.50
```

Fresh seeds:

```text
606, 707, 808, 909, 1010
```

If a candidate's fresh-seed `delta_macro_f1_vs_base` drops below 0.005, treat
the apparent winner as a lottery result from multiple testing.

## 11. Metrics And Columns

Every pooled result row must include at least:

```text
diagnostic_name
part
round
model
profile
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
delta_macro_f1_vs_base
n_train
n_validation
fit_seconds
predict_seconds
fit_status
```

Every summary row must include at least:

```text
seed_count
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
dummy_macro_f1_mean
delta_macro_f1_vs_dummy_mean
delta_macro_f1_vs_base
positive_ticker_count
top_ticker_gain_share
preliminary_gate
confirmation_status
```

Per-ticker output must include per-ticker macro F1, dummy macro F1, and
delta-vs-dummy for each reported diagnostic cell.

## 12. Interpretation Rules

All H0 comparisons are against the official Stage 0A2 baseline defined in
Section 3.

```text
delta_macro_f1_vs_base < 0.005:
  indistinguishable from base; no action

0.005 <= delta_macro_f1_vs_base < 0.010:
  weak diagnostic preference; note only; do not replace official Stage 0

delta_macro_f1_vs_base >= 0.010:
  strong diagnostic signal; requires a new pre-registered branch before use
```

Additional required gates:

```text
delta_macro_f1_vs_dummy > 0
positive_ticker_count >= 4
top_ticker_gain_share <= 0.50
fresh-seed confirmation, if selected for Part 3
```

H0 uses `positive_ticker_count >= 4`, which is stricter than the official Stage
0 gate of 3. This is intentional. H0 is a post-Stage 0 diagnostic sweep with
higher multiple-testing risk, so the cost of a false positive is higher than
the cost of missing a marginal diagnostic signal.

The `0.005` indistinguishable threshold is treated as an approximate
multiple-testing noise floor for roughly 200 validation-screened cells under
the current data scale. Cells below this threshold are not actionable even if
they rank first.

## 13. Trial Accounting

Canonical planned model-seed rows:

```text
Part 0 sanity:                         5
Part 1 window sweep:                  70
Part 2 LightGBM profiles:             90
Part 3 confirmation:                  15
Total before optional Round 2:       180
Optional Round 2 dense window sweep:  45
Total if Round 2 runs:               225
```

Implementation may de-duplicate identical rows, such as `profile_B` at
`window_size=20`, because it is identical to the Stage 0 LightGBM default.
However, reporting and multiple-testing commentary must use the canonical
planned grid above.

## 14. Literature Rationale And Caveat

This diagnostic is motivated by prior time-series and financial ML literature
that treats input lookback/window length as a meaningful experimental
dimension:

- DLinear / LTSF-Linear analyzes lookback length and shows that simple linear
  models can benefit from longer histories on several long-horizon forecasting
  benchmarks.
- PatchTST uses patching to retain local temporal information while enabling
  longer lookback windows at lower attention cost.
- DeepLOB uses a long sequence input in a high-frequency finance setting, but
  on limit-order-book data rather than OHLCV bars.
- Stock trend work such as MLCA-LSTM motivates multi-scale sliding windows for
  noisy, non-stationary stock data.

Caveat: DLinear and PatchTST results are primarily long-horizon forecasting
results, not 5-minute intraday binary direction classification. DeepLOB uses
limit-order-book features, not OHLCV. These papers motivate H0's diagnostic
question; they do not prove that longer windows should work in this project.
H0 empirically tests that transferability under the current validation-only
setup.

References:

- DLinear / LTSF-Linear:
  <https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089>
- PatchTST:
  <https://arxiv.org/abs/2211.14730>
- DeepLOB:
  <https://arxiv.org/abs/1808.03668>
- MLCA-LSTM stock trend paper:
  <https://www.sciencedirect.com/science/article/pii/S0925231222008736>
- Deflated Sharpe / multiple-testing warning:
  <https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2460551_code87814.pdf?abstractid=2460551>

## 15. Allowed Wording

Allowed:

```text
Diagnostic H0 found a validation-only diagnostic preference for [cell], but the
result is post-Stage 0 and cannot replace the official Stage 0 selection without
a new pre-registered branch.
```

Allowed:

```text
Diagnostic H0 did not find any post-Stage 0 tabular sweep result that cleared
the pre-committed practical delta, ticker-breadth, concentration, and
fresh-seed confirmation gates.
```

Forbidden:

```text
Window 24 is the best configuration.
The official Stage 0 winner should be replaced.
This result is ready for holdout/test.
The thesis model is selected.
```

## 16. Stop Conditions

Stop H0 if any of these occur:

- required Stage 0 output files are missing;
- the intended baseline is not present in official Stage 0A2 outputs;
- Part 0 fails the baseline reproduction tolerance;
- any code path reads, transforms, windows, scores, or summarizes holdout/test;
- dummy baseline or per-ticker outputs are missing;
- window construction crosses ticker or trading-day boundaries;
- train-only scaler fitting cannot be verified;
- LogReg convergence warnings exceed 5 percent of LogReg cells.

If stopped, report the exact blocker path or failing condition and do not
interpret partial diagnostic metrics.
