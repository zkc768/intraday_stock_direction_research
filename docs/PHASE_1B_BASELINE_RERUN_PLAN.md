# Phase 1B Baseline Rerun Plan

Status: docs-only planning for P1B.8c baseline rerun.

## 1. Purpose

This plan defines the first no-trade-band LSTM rerun before adding TCN or DLinear models. The rerun should isolate the effect of changing the label definition from the legacy full-sample binary label to the no-trade-band retained-subset label.

The goal is to measure the label-change effect before changing model architecture. Architecture changes, including TCN, DLinear, multi-scale DLinear, and residual branches, must wait until this LSTM-only rerun is reviewed.

## 2. Non-goals

- No TCN.
- No DLinear.
- No multi-scale DLinear.
- No residual TCN.
- No Notebook 03.
- No heavy training in this step.
- No new `ml_utils` APIs in this step.

## 3. Profiling-informed candidate grid

Use the P1B.7 / P1B.7d profiling results as the source of truth for the first rerun grid.

| Candidate | window_size | label_horizon_k | threshold_bps | split | n_valid_windows | minority_pct |
|---|---:|---:|---:|---|---:|---:|
| Main | 12 | 12 | 5 | train | 213116 | 0.4921 |
| Main | 12 | 12 | 5 | val | 11903 | 0.4990 |
| Main | 12 | 12 | 5 | test | 19129 | 0.4531 |
| Secondary | 24 | 12 | 5 | train | 161397 | 0.4927 |
| Secondary | 24 | 12 | 5 | val | 8118 | 0.4963 |
| Secondary | 24 | 12 | 5 | test | 13395 | 0.4465 |
| Longer horizon | 12 | 24 | 5 | train | 88783 | 0.4880 |
| Longer horizon | 12 | 24 | 5 | val | 2960 | 0.4811 |
| Longer horizon | 12 | 24 | 5 | test | 6644 | 0.3975 |
| Long-context stress test | 60 | 12 | 5 | train | 36780 | 0.4965 |
| Long-context stress test | 60 | 12 | 5 | val | 1695 | 0.4956 |
| Long-context stress test | 60 | 12 | 5 | test | 2940 | 0.4650 |
| Sensitivity only | 12 | 12 | 10 | train | 56900 | n/a |
| Sensitivity only | 12 | 12 | 10 | val | 1018 | n/a |
| Sensitivity only | 12 | 12 | 10 | test | 4269 | n/a |

The `threshold_bps=10` validation sample count is near the lower bound, so it is not a main candidate.

Explicitly excluded from the first baseline rerun:

- `window_size=60`, `label_horizon_k=24`.
- `window_size=78`, `label_horizon_k=12`.
- `window_size=78`, `label_horizon_k=24`.
- `threshold_bps=0` as a main no-trade-band setting.
- `threshold_bps >= 15`.

## 4. Data and split contract

Use this data pipeline contract for the rerun:

```text
/content/drive/MyDrive/stockdata/Dow_30_1min/
```

- Raw data path: `/content/drive/MyDrive/stockdata/Dow_30_1min/`.
- Use the five Phase 1B tickers: `CSCO`, `JPM`, `KO`, `MSFT`, `WMT`.
- Resample OHLCV to the configured bar frequency with `open=first`, `high=max`, `low=min`, `close=last`, and `volume=sum`.
- Use lowercase columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- Do not forward fill.
- Split chronologically per ticker.
- Use the `DataConfig` default split ratio: `train=0.7`, `val=0.15`, `test=0.15`.
- Do not use the old TensorFlow notebook's `80/10/10` split.
- Build labels via `make_no_trade_band_labels`.
- Skip neutral labels through the existing `NaN` label path.
- Windows must not cross trading days.
- Future label horizons must not cross trading days.
- Fit the scaler on pooled train only, then transform validation and test.
- The final experiment should call `ml_utils`; it should not define independent notebook-only label, split, scaling, windowing, metric, or training logic.

## 5. Label contract

- Use `make_no_trade_band_labels`.
- `future_avg_r > +threshold` maps to label `1`.
- `future_avg_r < -threshold` maps to label `0`.
- `abs(future_avg_r) <= threshold` maps to `NaN` and is skipped by the windowed dataset.
- Exact threshold boundaries are neutral.
- Report retained coverage for each run.
- Report up/down balance after filtering.
- Include this selection-bias disclosure in any result table or writeup: metrics estimate conditional direction classification on retained samples, not full-market direction prediction.

## 6. Model contract

- Use the existing `LSTMClassifier` only.
- Do not change model architecture during this rerun.
- Do not tune TCN or DLinear.
- Seed handling should follow existing `ml_utils.seed` behavior.
- Checkpointing should follow existing `Trainer` and checkpoint behavior.

## 7. Seed policy

- The first rerun can use `seed=42`.
- If any candidate materially beats `dummy_stratified`, rerun the top candidate with seeds `[42, 43, 44]`.
- Do not claim robustness from one seed.

## 8. Metrics and baseline contract

Report one row per candidate, split, ticker scope, and seed when feasible. The result schema should include:

- `candidate_name`.
- `window_size`.
- `label_horizon_k`.
- `threshold_bps`.
- `split`.
- `n_valid_windows`.
- `retained_pct` or window coverage.
- `n_up`.
- `n_down`.
- `up_pct`.
- `down_pct`.
- `minority_pct`.
- `model_macro_f1`.
- `model_balanced_accuracy`.
- `model_precision_macro`.
- `model_recall_macro`.
- `dummy_stratified_macro_f1_mean`.
- `dummy_stratified_macro_f1_std`.
- `dummy_prior_macro_f1`.
- `always_up_macro_f1`.
- `always_down_macro_f1`.
- `delta_macro_f1_vs_dummy`.
- `confusion_matrix`.
- Per-ticker results and pooled summary if feasible.

The dummy baselines should be fit on the training labels and evaluated on the target evaluation split. Do not fit baselines on validation or test distributions.

## 9. Colab execution plan

Future Colab execution should be a thin orchestration layer that pulls the latest repo, mounts Drive, loads the configured data, calls `ml_utils`, and reports the rerun table.

The next execution artifact may be either:

- An updated dedicated rerun notebook.
- A thin Colab experiment notebook that imports `ml_utils`.

This document does not create that artifact. It also does not create Notebook 03.

## 10. Stopping rule

- The first execution should remain small and controlled for Colab.
- Run the main, secondary, longer-horizon, and long-context stress-test candidates first.
- Run the `threshold_bps=10` sensitivity candidate only after the main candidates finish.
- Do not proceed to TCN or DLinear until the LSTM rerun results are reviewed.

## 11. Acceptance criteria for moving to execution

Before the Colab rerun:

- This plan is reviewed.
- A docs-only commit is completed.
- Git is pushed.
- Colab pulls the latest repo.
- No production code changes are required.
- There is no unresolved mismatch between the profiling candidates and implementation capabilities.

## 12. Risks / review checklist

- `threshold_bps=10` has low validation sample count and is sensitivity only.
- The longer horizon `label_horizon_k=24` may reduce retained windows and class balance.
- `window_size=60`, `label_horizon_k=12` is a stress test, not the main baseline.
- Do not compare no-trade-band retained-subset metrics directly against legacy full-sample metrics without coverage disclosure.
- Avoid claiming Phase 1B model improvement before architecture changes.

## 13. P1B.9d label-alignment fixed Candidate A smoke result

P1B.9 first exposed label-window alignment leakage in the Candidate A smoke run. The leakage was fixed in:

```text
e2e2869 fix(dataset): align window labels to prediction point
```

After that fix, Candidate A was rerun in Colab as a smoke-only check. Do not treat this as a full A-D rerun and do not proceed to TCN or DLinear from this result alone.

### 13.1 Run configuration

```text
candidate_name = A_main_alignment_fixed_smoke
window_size = 12
label_horizon_k = 12
threshold_bps = 5
seed = 42
epochs = 2
repo_head = e2e2869 fix(dataset): align window labels to prediction point
output_dir = /content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed/
```

Colab guard status:

```text
Commit guard passed: e2e2869
P1B.9d guard passed: Candidate A only, seed=42, epochs=2, new output dir.
```

Selection-bias disclosure printed in Colab:

```text
No-trade-band binary classification estimates P(sign(r) | X, |r| > tau), not full-market P(sign(r) | X). Report coverage together with F1.
```

### 13.2 Input and label diagnostics

Raw 5-minute rows:

| ticker | rows | start | end | nan |
|---|---:|---|---|---:|
| CSCO | 444305 | 1998-01-02 | 2020-06-08 | 0 |
| JPM | 443589 | 1998-01-02 | 2020-06-05 | 0 |
| KO | 443273 | 1998-01-02 | 2020-06-08 | 0 |
| MSFT | 444322 | 1998-01-02 | 2020-06-05 | 0 |
| WMT | 443278 | 1998-01-02 | 2020-06-08 | 0 |

Technical-indicator and no-trade-band label diagnostics:

| ticker | 5m_rows | ti_rows | ti_drop | up | down | neutral | retained_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSCO | 444305 | 444285 | 20 | 42341 | 45174 | 289042 | 19.70% |
| JPM | 443589 | 443569 | 20 | 40039 | 41120 | 294694 | 18.30% |
| KO | 443273 | 443253 | 20 | 21254 | 20136 | 334135 | 9.34% |
| MSFT | 444322 | 444302 | 20 | 33708 | 34686 | 308192 | 15.39% |
| WMT | 443278 | 443258 | 20 | 26313 | 26060 | 323157 | 11.82% |

Label diagnostics dicts printed by Colab:

```text
CSCO {'n_total': 444285, 'n_tail': 12, 'n_cross_day': 67716, 'n_neutral': 289042, 'n_up': 42341, 'n_down': 45174}
JPM {'n_total': 443569, 'n_tail': 12, 'n_cross_day': 67704, 'n_neutral': 294694, 'n_up': 40039, 'n_down': 41120}
KO {'n_total': 443253, 'n_tail': 12, 'n_cross_day': 67716, 'n_neutral': 334135, 'n_up': 21254, 'n_down': 20136}
MSFT {'n_total': 444302, 'n_tail': 12, 'n_cross_day': 67704, 'n_neutral': 308192, 'n_up': 33708, 'n_down': 34686}
WMT {'n_total': 443258, 'n_tail': 12, 'n_cross_day': 67716, 'n_neutral': 323157, 'n_up': 26313, 'n_down': 26060}
```

The final summary cell did not recover separate train, validation, and test label distribution variables; it printed them as `None`. Window counts were printed by the training cell:

```text
train_windows=213,384
val_windows=11,903
test_windows=19,190
```

### 13.3 Training smoke output

```text
epoch 1 | train_loss=0.696941 | val_loss=0.694377 | val_macro_f1=0.450760 | best=0.450760
epoch 2 | train_loss=0.692010 | val_loss=0.695803 | val_macro_f1=0.419024 | best=0.450760
best_val_macro_f1 = 0.4508
best_epoch = 1
train_seconds = 103.8
```

### 13.4 Test split per-ticker results

| ticker | n_valid_windows | n_up | n_down | model_macro_f1 | dummy_stratified_macro_f1_mean | dummy_stratified_macro_f1_std | dummy_prior_macro_f1 | always_up_macro_f1 | always_down_macro_f1 | delta_macro_f1_vs_dummy | confusion_matrix labels=[0,1] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CSCO | 4697 | 2103 | 2594 | 0.5014 | 0.4990 | 0.0062 | 0.3558 | 0.3093 | 0.3558 | 0.0024 | `[[2043, 551], [1554, 549]]` |
| JPM | 4561 | 2056 | 2505 | 0.4310 | 0.5016 | 0.0074 | 0.3545 | 0.3107 | 0.3545 | -0.0706 | `[[2412, 93], [1879, 177]]` |
| KO | 2239 | 1006 | 1233 | 0.4936 | 0.4990 | 0.0090 | 0.3551 | 0.3100 | 0.3551 | -0.0053 | `[[1004, 229], [771, 235]]` |
| MSFT | 4856 | 2135 | 2721 | 0.3908 | 0.4991 | 0.0041 | 0.3591 | 0.3054 | 0.3591 | -0.1083 | `[[2687, 34], [2065, 70]]` |
| WMT | 2837 | 1374 | 1463 | 0.3621 | 0.5051 | 0.0048 | 0.3402 | 0.3263 | 0.3402 | -0.1430 | `[[1421, 42], [1337, 37]]` |

Ticker-mean summary printed by Colab:

```text
dummy_stratified_macro_f1_mean: 0.5007495641561069
dummy_stratified_macro_f1_std: 0.00629495738648912
test_macro_f1 ticker-mean: 0.4357976294876892
test_balanced_accuracy ticker-mean: 0.5163990677355137
delta_macro_f1_vs_dummy: -0.06495193466841775
```

Leakage check:

```text
LEAKAGE CHECK PASSED: test_macro_f1 below 0.90 threshold
```

This is a material change from the leaked run, where macro F1 was abnormally close to 1.0. The post-fix smoke result is near dummy baseline and does not support proceeding directly to broader architecture work. Review this result before any full A-D rerun.

### 13.5 Saved artifacts

Colab printed these saved paths:

```text
/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed/results_A_main_alignment_fixed_smoke.csv
/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed/A_main_alignment_fixed_smoke_summary.json
/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed/A_main_alignment_fixed_smoke_summary.csv
```
