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

## P1B.11b Candidate A Strict Multi-Seed Verification

### Run identity

- phase: P1B.11b
- script commit: 9619a08 chore(phase1b): add P1B.11b candidate A multiseed script
- source of truth: P1B.10-derived script
- candidate: A / main
- window_size: 12
- label_horizon_k: 12
- threshold_bps: 5
- seeds: [42, 43, 44]
- output directory: `/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_p1b11b_candidate_a_multiseed/run_20260520T052850Z`

### Pipeline guards

- HEAD in Colab: 9619a08dd8fd18cf4432f4e2cf01ad82df775204
- required history commit 208d1e3 present
- e2e2869 label-alignment fix present
- fresh Colab clone ml_utils import guard passed
- no TCN/DLinear/Notebook 03 used
- suspicious metric guard passed

### Window-count guard

For every seed [42, 43, 44]:

- pooled train windows = 213384
- pooled val windows = 11903
- pooled test windows = 19190

### Per-seed summary

| seed | mean_model_macro_f1 | mean_dummy_stratified_macro_f1 | mean_delta_macro_f1_vs_dummy | positive_delta_tickers | n_tickers |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.5203 | 0.4990 | +0.0213 | 5 | 5 |
| 43 | 0.5028 | 0.4991 | +0.0038 | 2 | 5 |
| 44 | 0.4909 | 0.4998 | -0.0089 | 2 | 5 |

### Overall multi-seed summary

| mean_delta_across_seeds | std_delta_across_seeds | mean_model_macro_f1_across_seeds | std_model_macro_f1_across_seeds | positive_delta_tickers_total | total_seed_ticker_runs |
|---:|---:|---:|---:|---:|---:|
| +0.0054 | 0.0152 | 0.5047 | 0.0148 | 9 | 15 |

### Coverage / retained-subset reminder

- no-trade-band binary classification estimates P(sign(r) | X, |r| > tau), not P(sign(r) | X)
- retained-subset metrics must be interpreted together with coverage/window retention
- retained_pct by ticker in this run:
  - CSCO: 19.6979%
  - JPM: 18.2968%
  - KO: 9.3378%
  - MSFT: 15.3936%
  - WMT: 11.8155%

### Interpretation

- P1B.11b validates that P1B.10 Candidate A seed=42 was not caused by pipeline drift.
- However, the advantage is not stable across seeds.
- seed 42 is positive on all 5 tickers.
- seed 43 is only slightly positive overall and positive on 2/5 tickers.
- seed 44 is negative overall and positive on 2/5 tickers.
- Overall mean delta is only +0.0054 with std 0.0152.
- Therefore Candidate A should be described as weak / seed-sensitive / not robustly better than dummy.
- Do not use P1B.11b as evidence of a strong LSTM signal.
- This supports keeping LSTM as a weak baseline before TCN/DLinear comparisons.

### Non-actions

- no code changes
- no ml_utils changes
- no tests run
- no Notebook 03
- no TCN/DLinear
- no experiment outputs committed

## 14. P1B.18b / P1B.18c Notebook 03 minimal real-data smoke record

Date: 2026-05-21

P1B.18b was a minimal real-data smoke for Notebook 03 helper paths, not a full model comparison. P1B.18c reviewed the pasted P1B.18b output summary and accepted it as PASS WITH WARNINGS. This record does not claim independent Colab verification, independent Drive artifact inspection, robust model signal, or full Notebook 03 comparison readiness.

### 14.1 Scope

- CSCO only.
- Candidate A only.
- LSTM only.
- seed=42 only.
- max_raw_rows_per_ticker=20000.
- one epoch only.
- no artifacts.
- no `Trainer.fit`.
- no `run_model_comparison()`.
- no TCN.
- no DLinear.
- no full A-D comparison.
- no multi-seed run.

### 14.2 Reported pipeline and hygiene

The reported minimal pipeline completed:

```text
data load -> filter/resample/features -> no-trade-band labels -> windows -> dummy baselines -> one-epoch LSTM train_one_epoch -> evaluate
```

Reported notebook and repo hygiene:

- cell_count: 25
- bad_outputs: []
- bad_exec_counts: []
- FULL_RUN: False
- RUN_TRAINING: False
- final git status: clean
- final smoke verdict: PASS

Reported Colab smoke environment:

- python: 3.12.13
- torch: 2.10.0+cpu
- cuda_available: False

### 14.3 Reported data and window counts

- rows after cap: 20000
- rows after regular-hours filter: 19950
- rows after 5-minute resample: 4035
- rows after feature preparation: 4015
- rows after no-trade-band labeling: 4015
- train_windows: 641
- val_windows: 145
- test_windows: 161

### 14.4 Reported one-epoch model result

Reported test confusion matrix with `labels=[0, 1]`:

```text
[[ 0 97]
 [ 0 64]]
```

Reported metrics:

- test_macro_f1: 0.2844
- dummy_stratified test macro F1: 0.5112
- delta_macro_f1_vs_dummy: -0.2268

Interpretation:

- PASS WITH WARNINGS.
- The smoke supports that the Notebook 03 helper-driven minimal real-data path can connect to Drive data and complete the scoped smoke.
- The smoke does not prove robust model signal.
- The smoke does not prove full Notebook 03 comparison readiness.
- The smoke does not validate TCN or DLinear.
- The smoke does not validate the full A-D comparison.
- The smoke does not validate multi-seed robustness.

### 14.5 Warnings and next step

Warnings:

- The one-epoch LSTM collapsed to always-up.
- The reported delta versus dummy was negative.
- The Colab Python and torch environment differs from the local project environment.
- Drive artifact absence was not independently inspected; only the reported no-artifact non-actions are accepted here.
- The sample scope was intentionally tiny.

Do not go directly to a full run from this smoke. The recommended next step is P1B.19 full Notebook 03 readiness planning or a narrowly scoped readiness patch/review if blockers are found. Any full comparison must have an explicit execution plan, artifact policy, runtime guard policy, and scope approval.

## 15. P1B.20b guarded entrypoint narrow smoke record

Date: 2026-05-21

P1B.20b was a guarded Notebook 03 entrypoint smoke using:

```text
results = run_model_comparison()
```

This was not a helper-path smoke, not a full A-D run, and not evidence of model signal. It only reviewed whether the guarded `run_model_comparison()` entrypoint can complete a deliberately narrow real-data run without writing artifacts.

### 15.1 Scope

- Candidate A only.
- LSTM only.
- CSCO only.
- seed=42 only.
- max_raw_rows_per_ticker=20000.
- one epoch only.
- no artifacts.
- no TCN.
- no DLinear.
- no full A-D validation.
- no multi-seed robustness claim.

### 15.2 Execution config

| field | value |
|---|---|
| repo HEAD | `765bc41b0e4db0eff8cc1f585c7b008261f69b96` |
| Colab git status | `## master...origin/master` |
| FULL_RUN | `True` |
| RUN_TRAINING | `True` |
| WRITE_ARTIFACTS | `False` |
| ALLOW_OVERWRITE | `False` |
| SELECTED_CANDIDATES | `["A"]` |
| SELECTED_MODELS | `["lstm"]` |
| SELECTED_TICKERS | `["CSCO"]` |
| SELECTED_SEEDS | `[42]` |
| MAX_RAW_ROWS_PER_TICKER | `20000` |
| MAX_EPOCHS | `1` |

Although `FULL_RUN=True` and `RUN_TRAINING=True` were enabled to exercise the guarded entrypoint, the selection gates narrowed the execution to one candidate, one model, one ticker, one seed, and one epoch. This must not be described as full Notebook 03 comparison validation.

### 15.3 Results table

| field | value |
|---|---:|
| candidate_id | A |
| candidate_name | main |
| model_name | lstm |
| ticker | CSCO |
| seed | 42 |
| window_size | 12 |
| label_horizon_k | 12 |
| threshold_bps | 5.0 |
| split | test |
| n_train_windows | 641 |
| n_val_windows | 145 |
| n_test_windows | 161 |
| label_retained_pct | 0.304608 |
| model_macro_f1 | 0.278027 |
| model_balanced_accuracy | 0.484375 |
| dummy_stratified_macro_f1_mean | 0.511218 |
| dummy_stratified_macro_f1_std | 0.030821 |
| dummy_prior_macro_f1 | 0.375969 |
| always_up_macro_f1 | 0.284444 |
| always_down_macro_f1 | 0.375969 |
| delta_macro_f1_vs_dummy | -0.233191 |

Confusion matrix with `labels=[0, 1]`:

```text
[[0, 97], [2, 62]]
```

### 15.4 Diagnostics table

| field | value |
|---|---:|
| train_up_pct | 0.486739 |
| train_down_pct | 0.513261 |
| val_up_pct | 0.4 |
| val_down_pct | 0.6 |
| test_up_pct | 0.397516 |
| test_down_pct | 0.602484 |
| label_n_total | 4015 |
| label_n_retained | 1223 |
| label_n_up | 605 |
| label_n_down | 618 |
| label_n_neutral | 2174 |
| label_n_cross_day | 606 |
| label_n_tail | 12 |
| model_precision_macro | 0.194969 |
| model_recall_macro | 0.484375 |
| best_epoch | 1 |
| best_val_macro_f1 | 0.347277 |
| training_time_seconds | 0.592459 |
| suspicious_status | False |

### 15.5 Manifest summary

| field | value |
|---|---|
| notebook | `03_model_comparison.ipynb` |
| phase | `P1B.19b` |
| timestamp | `2026-05-21T05:55:44.324349+00:00` |
| git_commit_hash | `765bc41` |
| run_id | `notebook03_20260521T055543Z_765bc41` |
| planned artifact root | `/content/drive/MyDrive/stockdata/phase1b_notebook03_model_comparison/notebook03_20260521T055543Z_765bc41` |

### 15.6 Artifact absence evidence

| path/check | exists |
|---|---|
| run_dir | False |
| per_ticker_results | False |
| summary_by_model | False |
| summary_by_seed | False |
| run_manifest | False |

This supports the no-artifact contract for this scoped smoke because `WRITE_ARTIFACTS=False` and the planned artifact paths were absent.

### 15.7 Interpretation

- PASS WITH WARNINGS for guarded entrypoint readiness.
- The one-epoch LSTM underperformed `dummy_stratified` on test macro F1.
- The one-epoch LSTM mostly predicted class 1, with confusion matrix `[[0, 97], [2, 62]]`.
- No model signal claim is supported by this run.
- No TCN or DLinear validation is supported by this run.
- No full A-D validation is supported by this run.
- No multi-seed robustness claim is supported by this run.

### 15.8 Warnings and next step

Warnings:

- The manifest `phase` field still says `P1B.19b`, even though this execution is P1B.20b. Correct this in a future notebook patch.
- The local notebook file may be dirty from VSCode/Colab execution cells and should not be committed as part of this docs-only record.

Recommended next step:

```text
P1B.20d - clean notebook execution residue or review-only decide whether to discard notebook changes
```

Non-actions in this record:

- no git add
- no commit
- no push
- no notebook edits
- no `ml_utils` changes
- no tests changes
- no prompts changes
- no notebook execution
- no training
- no Colab
- no artifacts
- no further experiments

## 16. P1B.21 staged Notebook 03 readiness plan

Date: 2026-05-21

This is a docs-only plan for the next guarded Notebook 03 readiness run. It is not a Colab execution record and does not authorize a full run.

P1B.20b showed that the guarded `run_model_comparison()` entrypoint can complete a deliberately narrow A/LSTM/CSCO/seed-42/one-epoch smoke without writing artifacts. It also showed that the one-epoch LSTM underperformed `dummy_stratified`, so the next step is readiness coverage, not a model-signal claim.

### 16.1 Remaining unproven scope

The following are still unproven by P1B.20b:

- TCN entrypoint execution.
- DLinear entrypoint execution.
- Any result beyond Candidate A.
- Any result beyond CSCO.
- Any result beyond seed 42.
- Any result beyond one epoch.
- Artifact writing and overwrite safety.
- Full A-D candidate validation.
- Full five-ticker validation.
- Multi-seed robustness.
- Any model signal claim.

### 16.2 Recommended next rung: P1B.21a model-axis narrow smoke

The next execution should expand only the model axis while keeping all other axes fixed at the P1B.20b narrow scope.

Purpose:

- Validate that Notebook 03 can instantiate and execute all registered model paths through `run_model_comparison()`.
- Validate only narrow entrypoint readiness for LSTM, TCN, and DLinear.
- Avoid mixing model-axis validation with candidate, ticker, seed, epoch, or artifact expansion.

Execution scope:

| field | value |
|---|---|
| entrypoint | `results = run_model_comparison()` |
| FULL_RUN | `True` |
| RUN_TRAINING | `True` |
| WRITE_ARTIFACTS | `False` |
| ALLOW_OVERWRITE | `False` |
| SELECTED_CANDIDATES | `["A"]` |
| SELECTED_MODELS | `["lstm", "tcn", "dlinear"]` |
| SELECTED_TICKERS | `["CSCO"]` |
| SELECTED_SEEDS | `[42]` |
| MAX_RAW_ROWS_PER_TICKER | `20000` |
| MAX_EPOCHS | `1` |

Expected result shape:

- 3 result rows: one each for `lstm`, `tcn`, and `dlinear`.
- 3 diagnostics rows if diagnostics are emitted per result row.
- No artifact files or run directory.

### 16.3 Required pre-execution checks

Before P1B.21a execution:

- Local `master` must include the P1B.20c/P1B.20f commit.
- The commit must be pushed before Colab pulls.
- Colab must show the expected HEAD after pull.
- The notebook manifest `phase` should be updated to the P1B.21a execution label before the run, or the P1B.21a review must explicitly record any stale phase mismatch.
- Notebook source must be clean before execution.
- Outputs and execution counts must not be committed.

### 16.4 P1B.21a pass criteria

P1B.21a may be recorded as PASS WITH WARNINGS if all of the following hold:

- `run_model_comparison()` completes without exception.
- The result table contains exactly the intended model set: `lstm`, `tcn`, and `dlinear`.
- Each model row is limited to Candidate A, CSCO, seed 42, split `test`.
- `MAX_RAW_ROWS_PER_TICKER=20000` and `MAX_EPOCHS=1` are confirmed.
- `WRITE_ARTIFACTS=False` is confirmed.
- Planned artifact paths are independently checked as absent.
- `suspicious_status=False` for all emitted diagnostics, if present.
- Confusion matrices are recorded with `labels=[0, 1]`.
- The review explicitly states that this does not validate full A-D, all tickers, multi-seed robustness, or model signal.

### 16.5 Stop conditions

Stop and review before any broader run if:

- TCN or DLinear fails to instantiate.
- Any selected model silently drops from the output.
- Any result row appears outside Candidate A / CSCO / seed 42.
- Artifacts are written unexpectedly.
- The manifest phase is stale and not documented.
- The notebook becomes dirty with ad hoc execution cells.
- Any suspicious metric guard fires.
- Runtime or memory looks incompatible with scaling the next rung.

### 16.6 Anti-overclaiming rules

P1B.21a must not be described as:

- full Notebook 03 validation.
- full A-D validation.
- five-ticker validation.
- multi-seed validation.
- evidence that TCN or DLinear has signal.
- evidence that any model beats dummy.
- artifact policy validation.

The only intended claim is:

```text
Notebook 03 guarded entrypoint can execute the model registry paths for LSTM, TCN, and DLinear under the narrow A/CSCO/seed-42/one-epoch/no-artifact smoke scope.
```

### 16.7 Next rung after P1B.21a

If P1B.21a passes review, the next controlled expansion should choose exactly one axis:

- ticker-axis expansion: Candidate A, seed 42, one epoch, no artifacts, all five Phase 1B tickers; or
- candidate-axis expansion: A-D candidates, CSCO only, seed 42, one epoch, no artifacts; or
- artifact-policy smoke: Candidate A, LSTM only, CSCO only, seed 42, one epoch, artifacts enabled into a fresh run directory.

Do not combine these expansions in one step. Do not jump directly to A-D x three models x five tickers x multi-seed.

## 17. P1B.21a model-axis narrow smoke record

Date: 2026-05-21

P1B.21a was a guarded Notebook 03 model-axis narrow smoke using:

```text
results = run_model_comparison()
```

This run used a temporary execution override in Colab/runtime state. The committed notebook source kept safe defaults with `FULL_RUN=False` and `RUN_TRAINING=False`; those flags were flipped only for this execution.

This record supersedes the earlier stale-runtime summary that still showed the old P1B.19b/P1B.20b one-row LSTM result. The valid P1B.21a run is identified by manifest phase `P1B.21a`, manifest git hash `7aad9ad`, three result rows, and selected models `["lstm", "tcn", "dlinear"]`.

### 17.1 Scope

- Candidate A only.
- LSTM, TCN, and DLinear only.
- CSCO only.
- seed=42 only.
- max_raw_rows_per_ticker=20000.
- one epoch only.
- no artifacts.
- no full A-D validation.
- no five-ticker validation.
- no multi-seed robustness claim.
- no model-signal claim.

### 17.2 Execution config

| field | value |
|---|---|
| generated_at_local | `2026-05-21T06:59:52` |
| entrypoint | `results = run_model_comparison()` |
| manifest git_commit_hash | `7aad9ad` |
| FULL_RUN | `True` |
| RUN_TRAINING | `True` |
| WRITE_ARTIFACTS | `False` |
| ALLOW_OVERWRITE | `False` |
| SELECTED_CANDIDATES | `["A"]` |
| SELECTED_MODELS | `["lstm", "tcn", "dlinear"]` |
| SELECTED_TICKERS | `["CSCO"]` |
| SELECTED_SEEDS | `[42]` |
| MAX_RAW_ROWS_PER_TICKER | `20000` |
| MAX_EPOCHS | `1` |

### 17.3 Results table

| model_name | ticker | seed | n_train_windows | n_val_windows | n_test_windows | label_retained_pct | model_macro_f1 | model_balanced_accuracy | dummy_stratified_macro_f1_mean | dummy_stratified_macro_f1_std | dummy_prior_macro_f1 | always_up_macro_f1 | always_down_macro_f1 | delta_macro_f1_vs_dummy | confusion_matrix labels=[0,1] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lstm | CSCO | 42 | 641 | 145 | 161 | 0.304608 | 0.278027 | 0.484375 | 0.511218 | 0.030821 | 0.375969 | 0.284444 | 0.375969 | -0.233191 | `[[0, 97], [2, 62]]` |
| tcn | CSCO | 42 | 641 | 145 | 161 | 0.304608 | 0.516287 | 0.517880 | 0.511218 | 0.030821 | 0.375969 | 0.284444 | 0.375969 | 0.005069 | `[[55, 42], [34, 30]]` |
| dlinear | CSCO | 42 | 641 | 145 | 161 | 0.304608 | 0.464939 | 0.468025 | 0.511218 | 0.030821 | 0.375969 | 0.284444 | 0.375969 | -0.046279 | `[[62, 35], [45, 19]]` |

All rows share:

| field | value |
|---|---|
| candidate_id | A |
| candidate_name | main |
| window_size | 12 |
| label_horizon_k | 12 |
| threshold_bps | 5.0 |
| split | test |

### 17.4 Diagnostics table

| model_name | train_up_pct | train_down_pct | val_up_pct | val_down_pct | test_up_pct | test_down_pct | label_n_total | label_n_retained | label_n_up | label_n_down | label_n_neutral | label_n_cross_day | label_n_tail | model_precision_macro | model_recall_macro | best_epoch | best_val_macro_f1 | training_time_seconds | suspicious_status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lstm | 0.486739 | 0.513261 | 0.4 | 0.6 | 0.397516 | 0.602484 | 4015 | 1223 | 605 | 618 | 2174 | 606 | 12 | 0.194969 | 0.484375 | 1 | 0.347277 | 0.404462 | False |
| tcn | 0.486739 | 0.513261 | 0.4 | 0.6 | 0.397516 | 0.602484 | 4015 | 1223 | 605 | 618 | 2174 | 606 | 12 | 0.517322 | 0.517880 | 1 | 0.431498 | 0.326727 | False |
| dlinear | 0.486739 | 0.513261 | 0.4 | 0.6 | 0.397516 | 0.602484 | 4015 | 1223 | 605 | 618 | 2174 | 606 | 12 | 0.465646 | 0.468025 | 1 | 0.390291 | 0.253023 | False |

### 17.5 Manifest summary

| field | value |
|---|---|
| notebook | `03_model_comparison.ipynb` |
| phase | `P1B.21a` |
| timestamp | `2026-05-21T06:59:52.423369+00:00` |
| git_commit_hash | `7aad9ad` |
| run_id | `notebook03_20260521T065950Z_7aad9ad` |
| planned artifact root | `/content/drive/MyDrive/stockdata/phase1b_notebook03_model_comparison/notebook03_20260521T065950Z_7aad9ad` |

Manifest guards:

| guard | value |
|---|---|
| full_run | True |
| run_training | True |
| write_artifacts | False |
| allow_overwrite | False |

Manifest scope:

| scope field | value |
|---|---|
| selected_candidates | `A / main / window_size=12 / label_horizon_k=12 / threshold_bps=5` |
| selected_models | `["lstm", "tcn", "dlinear"]` |
| selected_tickers | `["CSCO"]` |
| selected_seeds | `[42]` |
| max_raw_rows_per_ticker | `20000` |
| max_epochs | `1` |

### 17.6 Artifact absence evidence

| path/check | exists |
|---|---|
| run_dir | False |
| per_ticker_results | False |
| summary_by_model | False |
| summary_by_seed | False |
| run_manifest | False |

Checked planned artifact paths under:

```text
/content/drive/MyDrive/stockdata/phase1b_notebook03_model_comparison/notebook03_20260521T065950Z_7aad9ad
```

This supports the no-artifact contract for this scoped smoke because `WRITE_ARTIFACTS=False` and the planned artifact paths were absent. It does not validate artifact-writing behavior.

### 17.7 Interpretation

- PASS WITH WARNINGS for model-axis guarded entrypoint readiness.
- Notebook 03 executed the LSTM, TCN, and DLinear registry paths under the narrow A/CSCO/seed-42/one-epoch/no-artifact scope.
- TCN had a small positive `delta_macro_f1_vs_dummy` of `0.005069`, but this is not a model-signal claim because the run is one ticker, one seed, one epoch, and 20k capped rows.
- LSTM and DLinear underperformed `dummy_stratified` on macro F1.
- All diagnostics reported `suspicious_status=False`.
- No full A-D validation is supported by this run.
- No five-ticker validation is supported by this run.
- No multi-seed robustness claim is supported by this run.
- No artifact-writing policy validation is supported by this run.

### 17.8 Warnings and next step

Warnings:

- A stale Colab runtime initially surfaced the older one-row LSTM result with manifest phase `P1B.19b`; that stale output was rejected and is not the P1B.21a record.
- Local notebooks may contain execution residue after Colab/VSCode activity and should not be committed as part of this docs-only record.
- This run validates narrow entrypoint execution only, not model quality.

Recommended next step:

```text
P1B.21c - clean notebook execution residue, then review/commit this docs-only record
```

Controlled expansion options after this record is reviewed:

- ticker-axis expansion: Candidate A, all three models, seed 42, one epoch, no artifacts, all five Phase 1B tickers; or
- candidate-axis expansion: A-D candidates, all three models, CSCO only, seed 42, one epoch, no artifacts; or
- artifact-policy smoke: Candidate A, one model only, CSCO only, seed 42, one epoch, artifacts enabled into a fresh run directory.

Do not combine these expansions in one step.
