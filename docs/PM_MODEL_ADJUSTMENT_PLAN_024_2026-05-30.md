# PM Model Adjustment Plan 024

Date: 2026-05-30
Task: PM-MODEL-ADJUSTMENT-PLAN-024
Mode: docs-only planning; no training; no notebook execution; no commit

## PM Status After d6fa5d3

Latest confirmed commit:

```text
d6fa5d3 docs: record mentor clean runner gate review
```

Closed gates:

- Gate 1 runner harden completed.
  - PM-019 targeted route-safety tests: 4 passed.
  - Full `tests/test_phase1b_local_runner.py`: 81 passed.
  - Validation-only manifest output is test/holdout blind.
  - Existing run artifact directories now fail loudly instead of being reused.
- Gate 2 tiny validation-only protocol smoke completed and reviewed.
  - Route used `mentor_clean_v1`, `no_trade_band`, and explicit fixed `threshold_bps=5.0`.
  - Metadata recorded `report_scope=validation_only`, `test_metrics_embargoed=True`, and `test_metrics_used=False`.
  - Manifest/results review found no forbidden test/holdout exposure.
- Gate 3 review note committed.
  - The accepted claim is protocol/artifact readiness only.
  - No model-performance claim is supported by the tiny smoke.

Historical note: `docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md` is stale where it says LightGBM and MS-DLinear+TCN are blocked. Current-code inspection shows LightGBM has a locked validation-only runner path, and MS-DLinear+TCN is implemented and wired as an explicit torch model. Execution policy is still separately gated.

## Lane A: LightGBM

Current route status:

- Implemented and callable through `--model-family lightgbm`.
- Hard-locked to `--validation-only-report`.
- Rejects `--full-run`.
- Requires `--feature-set mentor_clean_v1`.
- Requires `--label-mode no_trade_band`.
- Requires explicit `--threshold-bps 5.0`.
- Uses lazy LightGBM import with an explicit dependency error.
- `lightgbm==4.6.0` is pinned in `requirements.txt` and available in the shared project interpreter inspected for this planning gate.
- Tests cover fake-module dependency handling, route dispatch, fixed PM args, validation-only embargo, and no test-split scoring.

Blockers:

- No code blocker found for a bounded validation-only LightGBM smoke.
- Dependency is not currently a blocker in the shared interpreter, but a future worker must still fail loudly if `load_lightgbm_module()` reports the package missing.
- Validation metrics from the smoke must not be promoted to model-performance evidence.

Exact allowed command class after separate approval:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --model-family lightgbm `
  --validation-only-report `
  --validation-only-per-ticker `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --tickers CSCO JPM KO MSFT WMT `
  --max-rows-per-ticker <small_cap> `
  --output-dir checkpoints\pm_lgbm_val_<task_id>
```

Route constraints:

- Use the runner's locked LightGBM defaults only: `objective=binary`, `n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=1`, `verbosity=-1`.
- Do not add a hyperparameter grid.
- Do not choose thresholds, feature sets, or model settings from validation or test outcomes.
- Keep the threshold source fixed/pre-registered by passing `--threshold-bps 5.0` explicitly.
- Treat validation metrics as diagnostic observability only.

Stop rules:

- Stop if the output run directory already exists.
- Stop if raw data is missing.
- Stop if LightGBM import fails.
- Stop if the CLI would require `--full-run`.
- Stop if the command would omit `mentor_clean_v1`, `no_trade_band`, or explicit `threshold_bps=5.0`.
- Stop if metadata does not record `report_scope=validation_only`, `test_metrics_embargoed=True`, `test_metrics_used=False`, `threshold_source=fixed_pre_registered_5bps`, `decision_time_policy=post_bar_close_completed_bar`, and `scaler_id=standard_pooled_train_only_v1`.
- Stop if manifest/results expose forbidden test or holdout fields.
- Stop before writing evidence matrix, wiki, Zotero, paper claim maps, or performance narrative.

Minimum next Lane A task:

```text
PM-LGBM-VAL-025 -- bounded LightGBM validation-only smoke and artifact review
```

## Lane B: MS-DLinear+TCN

Current route status:

- Model class exists at `ml_utils/models/ms_dlinear_tcn_classifier.py`.
- Runner imports `MultiScaleDLinearTCNClassifier`.
- Runner can build it through `build_model("ms_dlinear_tcn", ...)`.
- It is not in the default `--models` list; it requires explicit `--models ms_dlinear_tcn`.
- Existing tests cover constructor/shape and monkeypatched CLI dispatch without real training.
- Existing tests verify mentor-route metadata for dispatch: model name, post-bar-close decision policy, train-only scaler id, and fixed 5 bps threshold source.

Blockers:

- Current real torch runtime path calls `trainer.fit(...)`, loads a checkpoint, and evaluates `prepared.test_dataset` plus per-ticker test datasets.
- `--validation-only-report` currently rejects torch model family and is limited to sklearn/logreg or LightGBM.
- Therefore, a real `--models ms_dlinear_tcn --smoke` is not a validation-only smoke. It is a tiny train-plus-test-evaluation route and is not safe for the Ian model-adjustment gate.

Exact allowed command class at this planning stage:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile `
  ml_utils\models\ms_dlinear_tcn_classifier.py `
  scripts\phase1b_local\local_baseline_matrix.py `
  tests\test_phase1b_local_runner.py

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest `
  tests\test_phase1b_local_runner.py::test_build_model_constructs_ms_dlinear_tcn_with_seq_len_and_input_size `
  tests\test_phase1b_local_runner.py::test_torch_cli_can_select_ms_dlinear_tcn_without_real_training `
  -q
```

These commands are review/config checks only. They do not authorize runtime training, notebook execution, or test evaluation.

Minimum safe Lane B task:

```text
PM-MS-DLINEAR-TCN-VAL-026 -- design or add a true torch validation-only route before any runtime smoke
```

That follow-up should be a code/test gate, not a runtime gate, unless the user explicitly approves a tiny torch train/validation-only implementation path.

Stop rules:

- Stop if a proposed command calls the current real torch `run_model_once` path for MS-DLinear+TCN.
- Stop if the command would read or report test metrics.
- Stop if the command would run notebooks.
- Stop if the command would use `--full-run`.
- Stop if it requires threshold tuning or test-driven model selection.
- Stop if it changes scaler behavior, label semantics, threshold policy, feature set, or decision-time policy.
- Stop if it would create checkpoint artifacts before a true validation-only torch path exists.

## Shared Train/Validation/Test Policy

| Area | Train | Validation | Test / holdout |
| --- | --- | --- | --- |
| Weights | Fit model weights only on train. | Do not fit weights directly except through train-driven early stopping/monitoring in approved torch protocols. | Never fit. |
| Scaler | Fit pooled scaler only after per-ticker chronological train split. | Transform with train-fitted scaler only. | Transform with train-fitted scaler only. |
| Hyperparameters | Candidate configs may be declared before fitting. | May select configs only in separately approved train/validation protocol. | Never select configs. |
| Threshold | Fixed pre-registered `threshold_bps=5.0` for current mentor route. | No threshold tuning in this route. | Never tune or choose threshold. |
| Metrics | Training diagnostics only. | Validation-only diagnostics and artifact review. | Embargoed until a separately approved final evaluation gate. |
| Artifacts | Record train counts/windows for observability. | Record validation scope and diagnostics. | Do not expose concrete test/holdout fields in validation-only manifest/results. |
| Claims | No performance claim from train. | Protocol/artifact claim only unless a future protocol permits stronger language. | Final performance claims require separately approved test gate. |

Shared route invariants:

- `feature_set_id=mentor_clean_v1`
- `label_mode=no_trade_band`
- `threshold_bps=5.0`
- `threshold_source=fixed_pre_registered_5bps`
- `decision_time_policy=post_bar_close_completed_bar`
- `scaler_id=standard_pooled_train_only_v1`
- `scaler_fit_scope=pooled_train_after_per_ticker_chronological_split`

## Paper Route Usage

Screened papers and mentor guidance may justify route rationale, constraints, and cautious model-selection hypotheses. They cannot become local model-performance evidence. Do not write `evidence_matrix.csv`, wiki pages, Zotero records, paper claims, paper tables, or performance narrative from PLAN-024 or from a validation-only smoke.

## PM Decision

Run Lane A first after separate approval.

Reason:

- LightGBM has lower implementation risk.
- It already has a locked validation-only runner path.
- It does not require torch training or the current torch test-evaluation path.
- It has explicit PM route guards for `mentor_clean_v1`, `no_trade_band`, and fixed 5 bps.
- It can produce an artifact that is useful for protocol review while preserving test embargo.

Keep Lane B behind a separate gate.

Reason:

- MS-DLinear+TCN is implemented and dispatch-tested, but current real torch execution evaluates test.
- A true torch validation-only report path or explicit train/validation-only implementation gate must come before runtime.

## Next Minimal User Approval

Approve this exact next task if you want execution:

```text
PM-LGBM-VAL-025 -- bounded LightGBM validation-only smoke and artifact review

Allowed:
- Read hf_stock_clf/AGENTS.md and PM plan/review docs.
- Run one bounded LightGBM validation-only command using:
  --model-family lightgbm
  --validation-only-report
  --validation-only-per-ticker
  --feature-set mentor_clean_v1
  --label-mode no_trade_band
  --threshold-bps 5.0
  --max-rows-per-ticker <small_cap>
  --output-dir checkpoints\pm_lgbm_val_025_<timestamp>
- Read back metadata.json, manifest.csv, and results.csv.
- Report protocol/artifact status only.

Forbidden:
- No notebook execution.
- No torch model training.
- No MS-DLinear+TCN runtime.
- No full-run.
- No threshold tuning.
- No test-metric readout.
- No evidence_matrix/wiki/Zotero/paper-claim writes.
- No performance claims.
- No commit unless separately approved.

Stop if:
- LightGBM dependency is missing.
- The output directory already exists.
- Any metadata field violates the shared route invariants.
- Manifest/results expose forbidden test/holdout fields.
- Validation metrics are about to be interpreted as model performance.
```
