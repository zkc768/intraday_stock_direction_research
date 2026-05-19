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

| Role | window_size | label_horizon_k | threshold_bps |
|---|---:|---:|---:|
| Main | 12 | 12 | 5 |
| Secondary | 24 | 12 | 5 |
| Longer horizon | 12 | 24 | 5 |
| Long-context stress test | 60 | 12 | 5 |
| Sensitivity only | 12 | 12 | 10 |

Explicitly excluded from the first baseline rerun:

- `window_size=60`, `label_horizon_k=24`.
- `window_size=78`, `label_horizon_k=12`.
- `window_size=78`, `label_horizon_k=24`.
- `threshold_bps >= 15`.

## 4. Data and split contract

- Use the `DataConfig` default split ratio: `train=0.7`, `val=0.15`, `test=0.15`.
- Do not use the old TensorFlow notebook's `80/10/10` split.
- Use the five Phase 1B tickers: `CSCO`, `JPM`, `KO`, `MSFT`, `WMT`.
- Load data from the Colab Drive path:

```text
/content/drive/MyDrive/stockdata/Dow_30_1min/
```

- Resample and preprocess consistently with prior smoke notebooks if needed.
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

## 7. Metrics and baseline contract

Report the following for each candidate when feasible:

- `macro_f1`.
- `balanced_accuracy`.
- Confusion matrix.
- `dummy_stratified` mean plus/minus std.
- `dummy_prior`.
- `always_up`.
- `always_down`.
- `delta_macro_f1_vs_dummy`.
- `retained_pct`.
- `n_valid_windows`.
- Per-ticker results and pooled summary if feasible.

The dummy baselines should be fit on the training labels and evaluated on the target evaluation split. Do not fit baselines on validation or test distributions.

## 8. Colab execution plan

Future Colab execution should be a thin orchestration layer that pulls the latest repo, mounts Drive, loads the configured data, calls `ml_utils`, and reports the rerun table.

The next execution artifact may be either:

- An updated dedicated rerun notebook.
- A thin Colab experiment notebook that imports `ml_utils`.

This document does not create that artifact. It also does not create Notebook 03.

## 9. Acceptance criteria for moving to execution

Before the Colab rerun:

- This plan is reviewed.
- A docs-only commit is completed.
- Git is pushed.
- Colab pulls the latest repo.
- No production code changes are required.
- There is no unresolved mismatch between the profiling candidates and implementation capabilities.

## 10. Risks / review checklist

- `threshold_bps=10` has low validation sample count and is sensitivity only.
- The longer horizon `label_horizon_k=24` may reduce retained windows and class balance.
- `window_size=60`, `label_horizon_k=12` is a stress test, not the main baseline.
- Do not compare no-trade-band retained-subset metrics directly against legacy full-sample metrics without coverage disclosure.
- Avoid claiming Phase 1B model improvement before architecture changes.
