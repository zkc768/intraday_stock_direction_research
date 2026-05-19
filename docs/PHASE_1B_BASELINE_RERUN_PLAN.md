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
