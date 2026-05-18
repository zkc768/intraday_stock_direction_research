# SPRINT_LOG.md — hf_stock_clf / ml_utils

最近更新：2026-05-17

## 当前状态

最近完成:     W6 final full validation PASS WITH WARNINGS
当前阶段:     W6 complete and validated
下一步:       decide next sprint scope
备注:         86 tests passed; Notebook 01/02 tracked files clean; runtime artifacts ignored; Phase 1B files absent; one existing PyTorch scheduler warning tracked
## 已合并模块清单（Codex 可以安全 import）

- `ml_utils/config.py`
- `ml_utils/seed.py`
- `ml_utils/metrics.py`
- `ml_utils/dataset.py`
- `ml_utils/checkpoint.py`
- `ml_utils/models/lstm_classifier.py`
- `ml_utils/trainer.py`

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| 0 MVP_YES | PASS | SPRINT_PLAN §1 / plan v2 §6 |
| 1 AGENTS v4.2 + marker + §9.1 例外 | PASS | AGENTS.md L2 + §9.1 末尾 |
| 2 AGENTS_BUILD_LOG v4.2 | PASS | BUILD_LOG §2/§3/§4 |
| 3 plan_v2_patches 应用 + MVP_YES + patch4#7 | PASS | grep 词表 100% 通过 |
| 4 ENVIRONMENT 填实 | PASS | Python / pytest / imports / pip check passed |
| 5 requirements 锁定 | PASS | requirements.txt 5 行核心依赖，全部 ==，torch 保留 +cpu |
| 6 SPRINT_LOG 创建 | PASS | 本文件 |
| 7 whitelist audit | PASS | requirements 5 个核心依赖均在 AGENTS.md §6.1 白名单或 tests scope |
| 8 reference_excerpts ltsf_data_loader.py | PASS | vendored from local LTSF-Linear data_provider/data_loader.py |
| 9 atomic commit | PASS | Gate 9 commit created by Codex session; hash not written back |

## Workstream status

| Item | Status | Evidence |
|---|---|---|
| Gate 0-6 | PASS | 手工段已完成 |
| Gate 7-9 | PASS | Gate status 表中 7-9 均为 PASS |
| W0.1 readiness audit | PASS | readiness audit 已完成；项目可进入 §14.2 testing infrastructure |
| W1.1 testing infrastructure | PASS | pytest.ini 与 tests/conftest.py 已创建；commit 249b2d8 |
| W2.1 config-test | PASS | tests/test_config.py 已创建；commit e0773f2 |
| W2.2 config-impl | PASS | ml_utils/config.py 已创建；commit 2945c81；tests/test_config.py 26 passed；W2.3 review PASS |
| W3.1 seed-test | PASS | tests/test_seed.py 已创建；commit 7e6b1ab；collect-only 5 tests collected；生成的 tests/__pycache__/ 已清理且未提交 |
| W3.2 seed-impl | PASS | commit `db0baf3`; created `ml_utils/seed.py`; implemented `seed_everything`; `tests/test_seed.py` passed 5/5; `tests/test_config.py tests/test_seed.py` passed 31/31 |
| W3.3 seed-review | PASS | Fresh review found no implementation issues; final status recovery restored clean scope before commit |
| W4.1 metrics-test | PASS | commit `092331d`; created `tests/test_metrics.py`; lazy import used; metrics collect-only collected 10 tests; config+seed+metrics collect-only collected 41 tests; `ml_utils/metrics.py` not created |
| W4.2 metrics-impl | PASS | commit `53b8398`; `ml_utils/metrics.py`; `tests/test_metrics.py` 10 passed; config+seed+metrics regression 41 passed; No SPRINT_LOG.md edits were made during W4.2 finalization; No git push was run |
| W4.3 metrics-review | PASS | fresh review found no BLOCKER / WARNING / NIT |
| W4.B.1 dataset-test | PASS | commit `ceb7969`; added dataset leakage, label generation, and window boundary tests; `pytest --collect-only` collected 63 items; no production code modified |
| W4.B.2 dataset-impl | PASS | commit `9466d05`; `ml_utils/dataset.py`; implemented label generation, chronological splits, train-only scaling, split/day boundary invalid marking, and per-ticker window dataset; targeted finalization tests passed; collect-only collected 63 tests |
| W4.C.1 checkpoint-test | PASS | commit `a46afcf`; added `tests/test_checkpoint.py`; 9 lazy-import checkpoint tests collected; full collect-only collected 72 tests; `ml_utils/checkpoint.py` not created |
| W4.C.2 checkpoint-impl | PASS | commit `b2738ee`; `ml_utils/checkpoint.py`; implemented save/load checkpoint; `tests/test_checkpoint.py` 9 passed, 1 warning; collect-only collected 72 tests |
| W4.D.1 LSTM classifier tests | PASS | commit `702db98`; added `tests/test_models_shape.py`; 6 lazy-import LSTMClassifier tests collected; full collect-only collected 78 tests; `ml_utils/models/lstm_classifier.py` not created |
| W4.D.2 LSTM classifier implementation | PASS | commit `e13ad0f`; `ml_utils/models/lstm_classifier.py`; implemented LSTMClassifier; `tests/test_models_shape.py` 6 passed; collect-only collected 78 tests |
| W5.1 trainer smoke tests | PASS | commit `2c9a43c`; added `tests/test_trainer_smoke.py`; 8 lazy-import trainer tests collected; full collect-only collected 86 tests; `ml_utils/trainer.py` not created |
| W5.2 trainer implementation | PASS | commit `a52bf0a`; `ml_utils/trainer.py`; implemented trainer loop helpers and Trainer.fit; `tests/test_trainer_smoke.py` 8 passed; collect-only collected 86 tests |
| MVP full validation audit | PASS | Python 3.11.15; pytest 8.3.5; pip check passed; collect-only collected 86 tests; full pytest passed 86 tests with 1 non-blocking warning; scope audit passed |
| W6.4 Notebook 01 single-stock LSTM smoke | PASS WITH WARNING | Notebook 01 executed with CSCO 5000-row subset; 2 epochs completed; final compact table present; no tracked files changed; checkpoint directory empty warning tracked |
| W6.12 Notebook 02 pooled LSTM smoke | PASS WITH WARNINGS | Notebook 02 executed with 5 tickers x 5000 rows; 2 epochs completed; shuffled-label sanity PASS; no tracked files changed; pandas FutureWarning tracked |
| W6.15 Notebook smoke phase closure | PASS WITH WARNINGS | W6 close ready after log-only wrap-up; tracked warnings deferred or classified as environment noise; no notebook / data / test / `ml_utils` patch required |
| W6.17 Final full validation after notebook smoke closure | PASS WITH WARNINGS | Python / pip / imports passed; 86 tests passed with 1 existing scheduler warning; notebooks clean; runtime artifacts ignored; no tracked diff |

## MVP full validation audit

Validation session name: MVP full validation audit

Overall verdict: PASS

### Environment validation

- Python version: Python 3.11.15
- pytest version: pytest 8.3.5
- torch 2.12.0+cpu
- numpy 1.26.4
- pandas 2.2.2
- sklearn 1.4.2
- pip check: No broken requirements found

### Test validation

- collect-only: 86 tests collected in 1.13s
- full pytest: 86 passed, 1 warning in 3.05s

### Non-blocking warning observed

- PyTorch scheduler warning in `tests/test_checkpoint.py:96` about calling `lr_scheduler.step()` before `optimizer.step()`
- Classification: non-blocking warning / future hygiene candidate
- No test failure resulted from this warning

### Scope audit

MVP production files present:

- `ml_utils/config.py`
- `ml_utils/seed.py`
- `ml_utils/metrics.py`
- `ml_utils/dataset.py`
- `ml_utils/checkpoint.py`
- `ml_utils/models/lstm_classifier.py`
- `ml_utils/trainer.py`

Phase 1B files absent:

- `ml_utils/models/tcn_classifier.py`
- `ml_utils/models/dlinear_classifier.py`

Expected tests present:

- `tests/test_config.py`
- `tests/test_seed.py`
- `tests/test_metrics.py`
- `tests/test_label_generation.py`
- `tests/test_dataset_leakage.py`
- `tests/test_window_boundaries.py`
- `tests/test_checkpoint.py`
- `tests/test_models_shape.py`
- `tests/test_trainer_smoke.py`

### Notebook status

- `notebooks` directory produced no output
- no notebook created

### Cache cleanup

Removed pytest-generated untracked cache directories:

- `ml_utils/__pycache__/`
- `ml_utils/models/__pycache__/`
- `tests/__pycache__/`

### Git status

- clean before validation
- clean after cache cleanup
- no files modified
- no files committed during validation audit

### Explicit non-actions

- no production code changed
- no tests changed
- no `SPRINT_LOG.md` update during validation audit
- no notebook created
- no TCN / DLinear files created
- no git add
- no git commit
- no git push

## W6.4 Notebook 01 single-stock LSTM smoke - PASS

Date: 2026-05-17

Verdict: PASS WITH WARNING

### Execution target

- Input notebook: `notebooks/01_smoke_test_single_stock_lstm.ipynb`
- Executed notebook: `checkpoints/notebook_runs/01_smoke_test_single_stock_lstm.executed.ipynb`
- Data file: `data/CSCO.csv`
- Related notebook fix commits:
  - `9fd1dd5 notebook: fix single-stock smoke import path`
  - `416ec3c notebook: resolve smoke data path from project root`

### Execution environment

- Python: 3.11.15
- nbconvert: 7.16.6
- pip check: passed

### Smoke evidence

- Dataset / ticker: CSCO
- Rows used: 5000
- Date range: 1998-01-02 09:30:00 to 1998-04-03 11:20:00
- Split rows: train / val / test = 3500 / 750 / 750
- Dataset windows: train / val / test = 1936 / 400 / 396
- Model initialized: yes
- Model: `lstm_classifier`
- Parameters: 5122
- Device: cpu
- Training completed: yes
- Epochs: 2
- Best epoch: 2
- Best val macro F1: 0.347471
- Val macro F1: 0.347471
- Test macro F1: 0.405405
- Test balanced accuracy: 0.500000
- Final compact table: present, 10 rows including LSTM val/test plus dummy/always baselines

### Baseline comparison

- Val delta vs `dummy_stratified`: -0.142006 macro F1
- Test delta vs `dummy_stratified`: -0.078262 macro F1

### Git hygiene

- Tracked notebook unchanged after execution
- `ml_utils` unchanged
- `tests` unchanged
- Git status clean after smoke

### Warning

- `checkpoints/notebook_01_lstm_smoke/` empty
- No model checkpoint file generated
- Non-blocking for W6.4 orchestration smoke, but should be tracked before treating Notebook 01 as checkpoint-save validation

### Non-actions

- Did not create Notebook 02
- Did not start TCN / DLinear
- Did not modify `ml_utils` or `tests`
- Did not git add runtime artifacts

### Next step

- W6.8 Notebook 02 read-only planning / public API support audit
- Notebook 02 pooled/global-scaler workflow 暂缓；先确认 `dataset.py` public API 是否自然支持 pooled workflow，不创建 Notebook 02

## W6.12 Notebook 02 pooled LSTM smoke - PASS

Date: 2026-05-17

Verdict: PASS WITH WARNINGS

### Execution target

- Input notebook: `notebooks/02_smoke_test_pooled_lstm.ipynb`
- Executed notebook: `checkpoints/notebook_runs/02_smoke_test_pooled_lstm.executed.ipynb`
- Data files:
  - `data/CSCO.csv`
  - `data/JPM.csv`
  - `data/KO.csv`
  - `data/MSFT.csv`
  - `data/WMT.csv`
- Related notebook commit:
  - `a9a2b7e notebook: add pooled LSTM smoke test`

### Execution environment

- Python: 3.11.15
- nbconvert: 7.16.6
- pip check: passed

### Smoke evidence

- Tickers: CSCO, JPM, KO, MSFT, WMT
- Max rows per ticker: 5000
- Loaded rows per ticker: all 5000
- Split rows per ticker: train / val / test = 3500 / 750 / 750
- Global train frame rows: 17500
- Window counts:
  - CSCO: train / val / test = 1936 / 400 / 396
  - JPM: train / val / test = 1925 / 384 / 399
  - KO: train / val / test = 1932 / 400 / 392
  - MSFT: train / val / test = 1936 / 400 / 396
  - WMT: train / val / test = 1925 / 393 / 400
- Pooled dataset sizes: train / val / test = 9654 / 1977 / 1983
- Model initialized: yes
- Model: `lstm_classifier`
- Parameters: 1538
- Device: CPU
- Training completed: yes
- Epochs: 2
- Best epoch: 1
- Best val macro F1: 0.348814
- Val accuracy: 0.535660
- Val macro F1: 0.348814
- Val balanced accuracy: 0.500000
- Test accuracy: 0.616742
- Test macro F1: 0.381472
- Test balanced accuracy: 0.500000
- Final compact table: present

### Baseline comparison

- Val `dummy_stratified` macro F1: 0.497313
- Val delta vs `dummy_stratified`: -0.148498 macro F1
- Test `dummy_stratified` macro F1: 0.493808
- Test delta vs `dummy_stratified`: -0.112335 macro F1
- Final summary: LSTM underperformed `dummy_stratified` on val/test

### Shuffled-label sanity

- Status: PASS
- `shuffled_label_macro_f1`: 0.348814
- `dummy_stratified_macro_f1_mean`: 0.497313
- 1.10x threshold: 0.547044
- 1.20x threshold: 0.596775

### Checkpoint / artifacts

- `checkpoints/notebook_02_pooled_lstm_smoke/main/best.pt`
- `checkpoints/notebook_02_pooled_lstm_smoke/main/last.pt`
- `checkpoints/notebook_02_pooled_lstm_smoke/shuffled_labels/best.pt`
- `checkpoints/notebook_02_pooled_lstm_smoke/shuffled_labels/last.pt`

### Git hygiene

- Tracked Notebook 02 unchanged after execution
- Notebook 01 unchanged
- `ml_utils` unchanged
- `tests` unchanged
- Git status clean after smoke

### Warnings

- nbconvert emitted a Windows/zmq Proactor event loop warning; execution succeeded
- pandas `FutureWarning` from `ml_utils/dataset.py:187` during `transform_split`; non-blocking, but track before future pandas upgrades
- `checkpoints/notebook_02_pooled_lstm_shuffled/` empty because Notebook 02 used `checkpoints/notebook_02_pooled_lstm_smoke/shuffled_labels/`

### Non-actions

- Did not create Notebook 03
- Did not start TCN / DLinear
- Did not modify `ml_utils` or `tests`
- Did not git add runtime artifacts

### Next step

- W6 wrap-up / decide whether to record notebook smoke phase complete or address tracked warnings

## W6.15 Notebook smoke phase closure - PASS WITH WARNINGS

Date: 2026-05-17

Decision: W6_CLOSE_READY_AFTER_LOG_ONLY

Closure verdict: PASS WITH WARNINGS

### Notebook smoke status

- Notebook 01 single-stock LSTM smoke: PASS WITH WARNING
- Notebook 01 result recorded in commit `3b53ab8 docs(log): record Notebook 01 smoke pass`
- Notebook 02 pooled 5-stock LSTM smoke: PASS WITH WARNINGS
- Notebook 02 notebook commit: `a9a2b7e notebook: add pooled LSTM smoke test`
- Notebook 02 result recorded in commit `fc4b4e1 docs(log): record Notebook 02 smoke pass`

### W6.14 wrap-up audit

- Tracked notebooks clean
- Executed notebooks exist
- Notebook 02 checkpoint artifacts exist
- Git status clean

### Warning disposition

- Notebook 01 no checkpoint generated: TRACK_AND_DEFER; does not block W6 close
- Notebook 02 Windows/zmq Proactor warning: IGNORE_ENVIRONMENT_NOISE; does not block W6 close
- Notebook 02 pandas `FutureWarning` at `ml_utils/dataset.py:187`: TRACK_AND_DEFER; does not block W6 close
- Notebook 02 empty alternate shuffled checkpoint directory: TRACK_AND_DEFER; does not block W6 close

### Explicit non-actions

- No notebook patch required
- No `ml_utils` patch required
- No data patch required
- No test patch required
- No TCN / DLinear started

### Recommended next step

- Final full validation / decide next sprint scope

## W6.17 Final full validation after notebook smoke closure - PASS WITH WARNINGS

Date: 2026-05-17

Decision: FINAL_VALIDATION_PASS_WITH_WARNINGS_READY_TO_RECORD

Validation verdict: PASS WITH WARNINGS

### Environment validation

- Python: 3.11.15
- pytest: 8.3.5
- pip check: passed
- torch: 2.12.0+cpu
- numpy: 1.26.4
- pandas: 2.2.2
- sklearn: 1.4.2

### Test validation

- collect-only: 86 tests collected
- full pytest: 86 passed
- Warning: one existing PyTorch scheduler-order warning in `tests/test_checkpoint.py::test_scheduler_state_is_restored_when_scheduler_is_provided`

### Notebook hygiene

- Notebook 01 tracked version clean: 14 cells, outputs=0, execution_counts=0
- Notebook 01 PROJECT_ROOT bootstrap present
- Notebook 01 PROJECT_ROOT-relative `DATA_PATH` present
- Notebook 02 tracked version clean: 18 cells, outputs=0, execution_counts=0
- Notebook 02 `FULL_RUN=False` present
- Notebook 02 PROJECT_ROOT bootstrap present
- Notebook 02 PROJECT_ROOT-relative `DATA_DIR` present
- Notebook 02 remains LSTM only, no TCN / DLinear
- No Notebook 03

### Runtime / data validation

- Executed Notebook 01 exists
- Executed Notebook 02 exists
- Notebook 02 checkpoint artifacts exist
- All five processed CSVs exist
- `data/` and `checkpoints/` artifacts ignored by git

### Scope validation

- MVP production files present
- Phase 1B files absent:
  - `ml_utils/models/tcn_classifier.py` absent
  - `ml_utils/models/dlinear_classifier.py` absent

### SPRINT_LOG consistency

- Notebook 01 smoke recorded
- Notebook 02 smoke recorded
- W6 closure recorded
- Tracked W6 warnings still recorded

### Git hygiene

- Git status clean
- No tracked diff

### Explicit non-actions

- No notebook patch required
- No `ml_utils` patch required
- No test patch required
- No data / checkpoint patch required
- No TCN / DLinear started

### Recommended next step

- Decide next sprint scope
## 当前 git 状态

记录 MVP full validation audit 后，预期本次 docs/log step 只修改 `SPRINT_LOG.md`。

- 工作目录干净：待 Gate 0-6 commit 后确认
- 当前分支：待确认
- 与 origin：待确认

## Requirements whitelist audit

审计日期：2026-05-15
判据：AGENTS.md §6.1 依赖白名单

| Package | Version | Status | Source in AGENTS §6.1 |
|---|---|---|---|
| torch | 2.12.0+cpu | allowed | "torch、torch.nn、torch.optim、torch.utils.data" |
| numpy | 1.26.4 | allowed | "numpy、pandas" |
| pandas | 2.2.2 | allowed | "numpy、pandas" |
| scikit-learn | 1.4.2 | allowed | "sklearn.preprocessing、sklearn.metrics、sklearn.dummy、sklearn.base" |
| pytest | 8.3.5 | allowed (tests scope) | "tests/：可用 tempfile、pytest fixture" |

差异:
- requirements 有但白名单未列：无
- 白名单允许但 requirements 未列：无（核心第三方依赖均已列；`pandas.api.types` 由 pandas 提供，`sklearn.*` 由 scikit-learn 提供）

结论：PASS

## Reference excerpts

| File | Status | Source commit | License | Note |
|---|---|---|---|---|
| ltsf_data_loader.py | vendored | 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 | Apache-2.0 | MVP 用 |
| pytorch_tcn_core.py | deferred | n/a | n/a | Phase 1B 启动前再补 |
| ltsf_dlinear_model.py | deferred | n/a | n/a | Phase 1B 启动前再补 |

## 进行中 session 注意事项

- Gate 0-6 已由用户手工执行
- Gate 7-9 已完成并记录 PASS
- W0.1 readiness audit 已完成并记录 PASS
- W1.1 testing infrastructure 已完成并记录 PASS
- W2.1 config-test 已完成并记录 PASS
- W2.2 config-impl 已完成并记录 PASS
- W3.1 seed-test 已完成并记录 PASS
- W3.2 seed-impl 已完成并记录 PASS
- W3.3 seed-review 已完成并记录 PASS
- W4.1 metrics-test 已完成并记录 PASS
- W4.2 metrics-impl 已完成并记录 PASS；implementation commit 为 `53b8398 feat(metrics): add binary classification metrics`
- W4.2 committed file 仅 `ml_utils/metrics.py`
- W4.2 test evidence: `tests/test_metrics.py`: 10 passed
- W4.2 regression evidence: `tests/test_config.py tests/test_seed.py tests/test_metrics.py`: 41 passed
- W4.2 finalization 未修改 `SPRINT_LOG.md`
- W4.2 未运行 `git push`
- W4.3 metrics-review 已完成并记录 PASS
- W4.3 review evidence: Fresh review found no BLOCKER / WARNING / NIT
- W4.B.1 dataset-test 已完成并记录 PASS；test commit 为 `ceb7969 test: add W4.B.1 dataset tests`
- W4.B.1 files added: `tests/test_dataset_leakage.py`, `tests/test_label_generation.py`, `tests/test_window_boundaries.py`
- W4.B.1 label generation tests cover hand-computed future average return, zero boundary class 0, tail k NaN, row preservation, input immutability, invalid k, missing price column
- W4.B.1 leakage tests cover chronological splits, train-only scaler statistics, transform immutability, scaler_type standard/minmax, invalid scaler_type, no cross-ticker windows
- W4.B.1 window boundary tests cover split-boundary invalid markers, cross-trading-day horizon invalid markers, no cross-day input windows, skipping NaN label starts, tensor shape/dtype, stride behavior, invalid stride
- W4.B.1 validation evidence: `pytest --collect-only` collected 63 items
- W4.B.1 explicit non-actions: No full pytest was run; No production code was modified; `ml_utils/dataset.py` was not created or modified; `tests/conftest.py` was not modified; `pytest.ini` was not modified; No git push was run
- W4.B.1 label indexing clarification commit: `68923e1 test: clarify W4.B.1 label future-return indexing`
- W4.B.1 clarified convention: label at row t uses returns from t->t+1 through t+k-1->t+k
- W4.B.1 NaN convention: final k labels are NaN
- W4.B.1 rejected convention: do not use indexing that creates k+1 trailing NaN labels
- W4.B.1 clarification explicit non-actions: no production code modified; `ml_utils/dataset.py` still not implemented; no full pytest was run; no git push
- W4.B.1 zero-boundary test data fix commit: `fa2709b test: fix W4.B.1 zero-boundary label data`
- W4.B.1 zero-boundary fix summary: `test_zero_future_average_return_maps_to_non_up_class_zero` now uses prices `[100.0, 125.0, 93.75, 100.0]`; for k=2, returns are `+0.25` and `-0.25`, `future_avg = 0.0`, and exact zero maps to class 0
- W4.B.1 formula test adjustment: formula test data was moved away from zero-boundary artifacts; positive / negative labels are now unambiguous; zero boundary is tested only in the dedicated zero-boundary test
- W4.B.1 zero-boundary validation evidence: `tests/test_label_generation.py --collect-only` collected 7 tests; full collect-only collected 63 tests
- W4.B.1 zero-boundary explicit non-actions: no production code modified; `ml_utils/dataset.py` still not implemented; no full pytest was run; no git push
- W4.B.2 dataset implementation commit: `9466d05 feat(dataset): add W4.B.2 dataset implementation`
- W4.B.2 files changed: `ml_utils/dataset.py`
- W4.B.2 public API implemented: `make_binary_labels_from_future_avg_return`, `make_time_splits`, `fit_scaler_on_train`, `transform_split`, `trim_labels_at_split_boundary`, `WindowedClassificationDataset`
- W4.B.2 label convention: label at row t uses future k bar-to-bar returns from t->t+1 through t+k-1->t+k
- W4.B.2 label formula uses arithmetic mean only
- W4.B.2 exact zero future average return maps to class 0
- W4.B.2 final k rows remain NaN; no fill/drop
- W4.B.2 split behavior: chronological train/val/test split
- W4.B.2 scaler behavior: scaler fit uses train split only
- W4.B.2 transform behavior: `transform_split` returns a new DataFrame
- W4.B.2 split-boundary invalid labels are marked with NaN, not deleted
- W4.B.2 window behavior: no cross-ticker windows
- W4.B.2 window behavior: no cross-trading-day input windows
- W4.B.2 label horizon behavior: no cross-trading-day label horizons
- W4.B.2 validation behavior: duplicate timestamps rejected
- W4.B.2 timezone behavior: `timezone_policy` supports `"naive"` and `"utc"`
- W4.B.2 trim behavior: `trim_labels_at_split_boundary` uses explicit `label_col`, no binary-column scanning
- W4.B.3 initial review failed with 4 warnings: duplicate timestamp validation, timezone policy, unsafe label inference, trim ordering
- W4.B.2-fix addressed all 4 W4.B.3 review warnings
- W4.B.3 re-review PASS
- W4.B.3 remaining NITs only: future test coverage for unsorted/duplicate trim and `timezone_policy="utc"`
- W4.B.2 finalization validation evidence: `tests/test_label_generation.py`: 7 passed
- W4.B.2 finalization validation evidence: `tests/test_dataset_leakage.py`: 7 passed
- W4.B.2 finalization validation evidence: `tests/test_window_boundaries.py`: 8 passed
- W4.B.2 finalization validation evidence: `tests/test_config.py tests/test_seed.py tests/test_metrics.py`: 41 passed
- W4.B.2 finalization validation evidence: collect-only collected 63 tests
- W4.B.2 finalization details: commit `9466d05 feat(dataset): add W4.B.2 dataset implementation`
- W4.B.2 finalization details: commit stat `ml_utils/dataset.py | 318 insertions`
- W4.B.2 finalization details: `git diff --cached --check` had no output
- W4.B.2 finalization details: post-commit working tree clean
- W4.B.2 explicit non-actions: no tests modified
- W4.B.2 explicit non-actions: no SPRINT_LOG update was included in implementation commit
- W4.B.2 explicit non-actions: no full pytest was run
- W4.B.2 explicit non-actions: no git push
- W4.B.2 explicit non-actions: no files besides `ml_utils/dataset.py` committed
- W4.B.2 finalization retry cleanup: generated `__pycache__` files were cleaned during finalization retry only
- W4.C.1 checkpoint-test commit: `a46afcf test: add W4.C.1 checkpoint tests`
- W4.C.1 files changed: `tests/test_checkpoint.py`
- W4.C.1 tests added: 9 lazy-import checkpoint tests
- W4.C.1 checkpoint coverage: save/load model weight roundtrip
- W4.C.1 checkpoint coverage: optimizer state restore
- W4.C.1 checkpoint coverage: scheduler state restore
- W4.C.1 checkpoint coverage: metadata roundtrip
- W4.C.1 checkpoint coverage: `extra=None` roundtrip
- W4.C.1 checkpoint coverage: missing checkpoint path raises explicit error
- W4.C.1 checkpoint coverage: `weights_only=True` behavior
- W4.C.1 checkpoint coverage: raw checkpoint does not contain full model object
- W4.C.1 checkpoint coverage: `rng_state` keys exist
- W4.C.1 test-first / lazy import compliance: tests use lazy import for `ml_utils.checkpoint`
- W4.C.1 test-first / lazy import compliance: no top-level import of `ml_utils.checkpoint`
- W4.C.1 test-first / lazy import compliance: `ml_utils/checkpoint.py` was not created in W4.C.1
- W4.C.1 validation evidence: `tests/test_checkpoint.py --collect-only` collected 9 tests
- W4.C.1 validation evidence: all collect-only collected 72 tests
- W4.C.1 commit details: commit `a46afcf test: add W4.C.1 checkpoint tests`
- W4.C.1 commit details: commit stat `tests/test_checkpoint.py | 248 insertions`
- W4.C.1 commit details: `git diff --cached --check` had no output
- W4.C.1 commit details: post-commit working tree clean
- W4.C.1 explicit non-actions: no production code created
- W4.C.1 explicit non-actions: no `ml_utils/checkpoint.py` created
- W4.C.1 explicit non-actions: no existing tests modified
- W4.C.1 explicit non-actions: no full pytest was run
- W4.C.1 explicit non-actions: no git push
- W4.C.1 explicit non-actions: no `git add .`
- W4.C.1 explicit non-actions: no `git add -A`
- W4.C.2 checkpoint implementation commit: `b2738ee feat(checkpoint): add W4.C.2 checkpoint implementation`
- W4.C.2 files changed: `ml_utils/checkpoint.py`
- W4.C.2 implementation summary: implemented `save_checkpoint`
- W4.C.2 implementation summary: implemented `load_checkpoint`
- W4.C.2 implementation summary: saves `model.state_dict` only, not full model object
- W4.C.2 implementation summary: saves `optimizer_state_dict`
- W4.C.2 implementation summary: saves `scheduler_state_dict` when scheduler is not None
- W4.C.2 implementation summary: saves `epoch`
- W4.C.2 implementation summary: saves `best_metric`
- W4.C.2 implementation summary: saves `rng_state` keys `python`, `numpy`, `torch`, and `cuda`
- W4.C.2 implementation summary: preserves `extra=None` roundtrip
- W4.C.2 implementation summary: `load_checkpoint` raises explicit `FileNotFoundError` for missing path
- W4.C.2 implementation summary: `load_checkpoint` supports `weights_only=True` by loading model weights only
- W4.C.2 implementation summary: optimizer and scheduler restore only when `weights_only=False`
- W4.C.2 validation evidence: `tests/test_checkpoint.py`: 9 passed, 1 warning in 2.62s
- W4.C.2 validation evidence: all collect-only: 72 tests collected in 1.14s
- W4.C.2 validation evidence: `git diff --check` had no output
- W4.C.2 recovery detail: first W4.C.2 attempt stopped before commit because pytest generated untracked `ml_utils/__pycache__/` and `tests/__pycache__/`
- W4.C.2 recovery detail: recovery deleted only `ml_utils/__pycache__/` and `tests/__pycache__/`
- W4.C.2 recovery detail: recovery committed only `ml_utils/checkpoint.py`
- W4.C.2 explicit non-actions: tests not modified
- W4.C.2 explicit non-actions: `SPRINT_LOG.md` not modified during implementation session
- W4.C.2 explicit non-actions: no other production code modified
- W4.C.2 explicit non-actions: no full pytest run
- W4.C.2 explicit non-actions: no git clean
- W4.C.2 explicit non-actions: no `git add .`
- W4.C.2 explicit non-actions: no `git add -A`
- W4.C.2 explicit non-actions: no git push
- W4.C.2 explicit non-actions: no `.gitignore` change
- W4.D.1 LSTM classifier tests commit: `702db98 test(model): add W4.D.1 LSTM classifier shape tests`
- W4.D.1 files changed: `tests/test_models_shape.py`
- W4.D.1 tests added: 6 lazy-import `LSTMClassifier` tests
- W4.D.1 LSTM coverage: forward output shape input `(32, 60, 7)` returns logits `(32, 2)`
- W4.D.1 LSTM coverage: backward pass with `torch.nn.CrossEntropyLoss` produces gradients
- W4.D.1 LSTM coverage: `bidirectional=True` still returns `(batch, num_classes)`
- W4.D.1 LSTM coverage: single-layer dropout uses zero LSTM dropout / forward works
- W4.D.1 LSTM coverage: outputs are logits, not softmax/sigmoid probabilities
- W4.D.1 LSTM coverage: missing batch dimension raises `AssertionError` or `ValueError`
- W4.D.1 test-first / lazy import compliance: tests use lazy import for `ml_utils.models.lstm_classifier`
- W4.D.1 test-first / lazy import compliance: no top-level import of `LSTMClassifier`
- W4.D.1 test-first / lazy import compliance: `ml_utils/models/lstm_classifier.py` was not created in W4.D.1
- W4.D.1 MVP scope compliance: only `LSTMClassifier` tests were added
- W4.D.1 MVP scope compliance: no TCN tests
- W4.D.1 MVP scope compliance: no DLinear tests
- W4.D.1 MVP scope compliance: no TCN / DLinear production files created
- W4.D.1 MVP scope compliance: TCN / DLinear remain deferred to Phase 1B
- W4.D.1 validation evidence: `tests/test_models_shape.py --collect-only` collected 6 tests in 0.94s
- W4.D.1 validation evidence: all collect-only collected 78 tests in 1.09s
- W4.D.1 validation evidence: `git diff --check` had no output
- W4.D.1 commit details: commit `702db98 test(model): add W4.D.1 LSTM classifier shape tests`
- W4.D.1 commit details: commit stat `tests/test_models_shape.py | 128 insertions`
- W4.D.1 commit details: post-commit working tree clean
- W4.D.1 explicit non-actions: no production code created
- W4.D.1 explicit non-actions: no `ml_utils/models/lstm_classifier.py` created
- W4.D.1 explicit non-actions: no TCN / DLinear files created
- W4.D.1 explicit non-actions: no TCN / DLinear tests added
- W4.D.1 explicit non-actions: existing tests not modified
- W4.D.1 explicit non-actions: `SPRINT_LOG.md` not modified during test session
- W4.D.1 explicit non-actions: no normal pytest run
- W4.D.1 explicit non-actions: no git push
- W4.D.1 explicit non-actions: no `git add .`
- W4.D.1 explicit non-actions: no `git add -A`
- W4.D.2 LSTM classifier implementation commit: `e13ad0f feat(model): add W4.D.2 LSTM classifier implementation`
- W4.D.2 files changed: `ml_utils/models/lstm_classifier.py`
- W4.D.2 implementation summary: implemented `LSTMClassifier`
- W4.D.2 implementation summary: uses `nn.LSTM(batch_first=True)`
- W4.D.2 implementation summary: uses last time step output
- W4.D.2 implementation summary: applies `LayerNorm`
- W4.D.2 implementation summary: uses `Linear` classification head
- W4.D.2 implementation summary: returns raw logits only
- W4.D.2 implementation summary: no softmax
- W4.D.2 implementation summary: no sigmoid
- W4.D.2 implementation summary: supports `bidirectional=True`
- W4.D.2 implementation summary: single-layer LSTM uses internal `dropout=0.0`
- W4.D.2 implementation summary: validates `input_size > 0`
- W4.D.2 implementation summary: validates `hidden_size > 0`
- W4.D.2 implementation summary: validates `num_layers > 0`
- W4.D.2 implementation summary: validates `num_classes > 0`
- W4.D.2 implementation summary: validates `0 <= dropout < 1`
- W4.D.2 implementation summary: forward enforces 3D input shape
- W4.D.2 implementation summary: forward enforces last dimension equals `input_size`
- W4.D.2 reference note: `reference_excerpts/yutsuro_lstm_module.py` was not present
- W4.D.2 reference note: implementation followed `docs/ml_utils_construction_plan_v2.md` §5.4 spec
- W4.D.2 validation evidence: `tests/test_models_shape.py`: 6 passed in 1.05s
- W4.D.2 validation evidence: all collect-only: 78 tests collected in 1.07s
- W4.D.2 validation evidence: `git diff --check` had no output
- W4.D.2 commit details: commit `e13ad0f feat(model): add W4.D.2 LSTM classifier implementation`
- W4.D.2 commit details: commit stat `ml_utils/models/lstm_classifier.py | 72 insertions`
- W4.D.2 commit details: post-commit working tree clean
- W4.D.2 commit details: no cache cleanup needed; no `__pycache__` appeared
- W4.D.2 explicit non-actions: tests not modified
- W4.D.2 explicit non-actions: `SPRINT_LOG.md` not modified during implementation session
- W4.D.2 explicit non-actions: no TCN / DLinear files created
- W4.D.2 explicit non-actions: no trainer code created
- W4.D.2 explicit non-actions: no `.gitignore` change
- W4.D.2 explicit non-actions: no full pytest run
- W4.D.2 explicit non-actions: no git push
- W4.D.2 explicit non-actions: no `git add .`
- W4.D.2 explicit non-actions: no `git add -A`
- W5.1 trainer smoke tests commit: `2c9a43c test(trainer): add W5.1 trainer smoke tests`
- W5.1 files changed: `tests/test_trainer_smoke.py`
- W5.1 tests added: 8 lazy-import trainer tests
- W5.1 trainer coverage: `train_one_epoch` returns loss and accuracy
- W5.1 trainer coverage: `train_one_epoch` updates at least one model parameter
- W5.1 trainer coverage: `train_one_epoch` supports `grad_clip` smoke path
- W5.1 trainer coverage: `evaluate` returns metrics, `y_true`, and `y_pred`
- W5.1 trainer coverage: `evaluate` does not mutate model parameters
- W5.1 trainer coverage: `Trainer.fit` creates `best.pt` and `last.pt`
- W5.1 trainer coverage: `Trainer.fit` returns history with expected keys
- W5.1 trainer coverage: early stopping exits before max epochs on plateau
- W5.1 trainer coverage: invalid `monitor_mode` raises `ValueError`
- W5.1 trainer coverage: non-plateau scheduler `StepLR` is stepped per epoch
- W5.1 test-first / lazy import compliance: tests use lazy import for `ml_utils.trainer`
- W5.1 test-first / lazy import compliance: no top-level import of `train_one_epoch`, `evaluate`, or `Trainer`
- W5.1 test-first / lazy import compliance: `ml_utils/trainer.py` was not created in W5.1
- W5.1 validation evidence: `tests/test_trainer_smoke.py --collect-only` collected 8 tests in 1.17s
- W5.1 validation evidence: all collect-only collected 86 tests in 1.22s
- W5.1 validation evidence: `git diff --check` had no output
- W5.1 recovery detail: first W5.1 attempt completed collect-only validation but stopped before final git checks / commit because the approval system rejected further shell commands due usage limit
- W5.1 recovery detail: recovery reused prior collect-only validation
- W5.1 recovery detail: recovery confirmed only `tests/test_trainer_smoke.py` was present
- W5.1 recovery detail: recovery ran `git diff --check` with no output
- W5.1 recovery detail: recovery committed only `tests/test_trainer_smoke.py`
- W5.1 commit details: commit `2c9a43c test(trainer): add W5.1 trainer smoke tests`
- W5.1 commit details: commit stat `tests/test_trainer_smoke.py | 251 insertions`
- W5.1 commit details: post-commit working tree clean
- W5.1 explicit non-actions: no production code created
- W5.1 explicit non-actions: no `ml_utils/trainer.py` created
- W5.1 explicit non-actions: existing tests not modified
- W5.1 explicit non-actions: `SPRINT_LOG.md` not modified during test session
- W5.1 explicit non-actions: no normal pytest run
- W5.1 explicit non-actions: no full pytest run
- W5.1 explicit non-actions: no git clean
- W5.1 explicit non-actions: no git push
- W5.1 explicit non-actions: no `git add .`
- W5.1 explicit non-actions: no `git add -A`
- W5.2 trainer implementation commit: `a52bf0a feat(trainer): add W5.2 trainer implementation`
- W5.2 files changed: `ml_utils/trainer.py`
- W5.2 implementation summary: implemented `train_one_epoch`
- W5.2 implementation summary: implemented `evaluate`
- W5.2 implementation summary: implemented `Trainer.fit`
- W5.2 implementation summary: `train_one_epoch` returns loss and accuracy
- W5.2 implementation summary: `train_one_epoch` supports optional `grad_clip`
- W5.2 implementation summary: `evaluate` returns `metrics_dict`, `y_true`, and `y_pred`
- W5.2 implementation summary: `evaluate` does not update model parameters
- W5.2 implementation summary: `Trainer.fit` saves `best.pt`
- W5.2 implementation summary: `Trainer.fit` saves `last.pt`
- W5.2 implementation summary: `Trainer.fit` returns history with expected keys
- W5.2 implementation summary: supports early stopping
- W5.2 implementation summary: `ReduceLROnPlateau` is stepped with monitor value
- W5.2 implementation summary: non-plateau schedulers are stepped per epoch
- W5.2 implementation summary: invalid `monitor_mode` raises `ValueError`
- W5.2 implementation summary: no baseline calculation
- W5.2 implementation summary: no AMP
- W5.2 implementation summary: no multi-GPU
- W5.2 implementation summary: no logging framework
- W5.2 validation evidence: `tests/test_trainer_smoke.py`: 8 passed in 6.96s
- W5.2 validation evidence: all collect-only: 86 tests collected in 1.07s
- W5.2 validation evidence: `git diff --check` had no output
- W5.2 cache cleanup: generated `ml_utils/__pycache__/` was removed
- W5.2 cache cleanup: generated `tests/__pycache__/` was removed
- W5.2 cache cleanup: post-cleanup status only showed `ml_utils/trainer.py` before commit
- W5.2 commit details: commit `a52bf0a feat(trainer): add W5.2 trainer implementation`
- W5.2 commit details: commit stat `ml_utils/trainer.py | 267 insertions`
- W5.2 commit details: post-commit working tree clean
- W5.2 explicit non-actions: tests not modified
- W5.2 explicit non-actions: `SPRINT_LOG.md` not modified during implementation session
- W5.2 explicit non-actions: `checkpoint.py` not modified
- W5.2 explicit non-actions: `metrics.py` not modified
- W5.2 explicit non-actions: no model files modified
- W5.2 explicit non-actions: no TCN / DLinear files created
- W5.2 explicit non-actions: no notebook code
- W5.2 explicit non-actions: no full pytest run
- W5.2 explicit non-actions: no `.gitignore` change
- W5.2 explicit non-actions: no git push
- W5.2 explicit non-actions: no `git add .`
- W5.2 explicit non-actions: no `git add -A`
- 最近完成: W5.2 trainer implementation
- 当前阶段: W5.2 trainer implementation log update
- 下一步: return to ChatGPT for review before trainer implementation review or next module step

## Atomic commits

本项目 Gate 阶段采用两次 commit 边界：

- C1：Gate 0-6，由用户手工 commit
- C2：Gate 7-9，由 Codex session commit

| Commit | Gate range | Owner | Commit hash | Message |
|---|---|---|---|---|
| C1 | 0-6 | user manual | 待 Gate 0-6 commit 后生成 | chore(gate): apply Gate 0-6 sprint contract updates |
| C2 | 7-9 | Codex session | 不回填（按 Gate 9 规则，用 git log --oneline -1 验证） | chore(gate): land Gate 7-9 (whitelist audit + ltsf_data_loader vendor) for hf_stock_clf v4.1 |

## W7.3 — Record W7 cleanup and skills docs

- W7.2 pandas FutureWarning cleanup verdict: PASS WITH WARNINGS
- W7.2 commit: `0ff48fb fix(dataset): avoid pandas dtype warning in transform_split`
- W7.2 files changed:
  - `ml_utils/dataset.py`
  - `tests/test_dataset_leakage.py`
- W7.2 dataset focused test: 7 passed
- W7.2 full pytest: 86 passed, 1 existing warning
- W7.2 pandas FutureWarning from `transform_split`: fixed
- W7.2 remaining warning: existing PyTorch scheduler-order warning only
- Skills usage docs verdict: PASS WITH WARNINGS
- Skills usage docs commit: `80cfeea docs: add skills usage guide`
- Skills usage docs files changed:
  - `AGENTS.md`
  - `docs/SKILLS_USAGE.md`
- Skills usage docs scope: added standalone skills usage guide
- Skills usage docs scope: added short `AGENTS.md` reference section
- Skills usage docs scope: did not merge full skills guide into `AGENTS.md`
- W7.3 non-actions: no notebook execution
- W7.3 non-actions: no training
- W7.3 non-actions: no Phase 1B
- W7.3 non-actions: no Notebook 03
- W7.3 non-actions: no TCN/DLinear
- W7.3 non-actions: no `ml_utils` changes during skills docs step
- W7.3 non-actions: no tests changes during skills docs step
- Current status: W7 cleanup/logging status recorded
- Current status: git status expected clean
- Recommended next decision: choose between README/MVP documentation pass, checkpoint cleanup, or Phase 1B readiness audit
- Recommended next decision: do not directly implement Phase 1B without readiness/reference audit
