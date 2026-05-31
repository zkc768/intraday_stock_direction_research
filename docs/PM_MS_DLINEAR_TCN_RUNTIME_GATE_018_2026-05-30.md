# PM-MS-DLINEAR-TCN-RUNTIME-GATE-018

Date: 2026-05-30
Owner: PM route control
Status: runtime smoke blocked; runner gate required first

## Superseded / Current Live State

This runtime-gate packet is historical. It remains useful as the audit trail
for why the original torch smoke was blocked, but its "current" runner state is
superseded by later code/tests.

Current live state for later PM gates:

- `MultiScaleDLinearTCNClassifier` exists and is runner-wired as
  `ms_dlinear_tcn`.
- `--validation-only-report` now accepts the torch `ms_dlinear_tcn` route when
  the PM locks are explicit.
- The validation-only path must avoid test/holdout scoring and emitted
  test/holdout metric exposure; prior runner tests cover safe branch dispatch,
  unlocked-route rejection, validation-only result scope, and no test data
  materialization for this route.
- A future tiny smoke is permitted only after the current doc-reconcile commit
  gate passes and only as a train-on-train / report-validation diagnostic. It
  must not be read as permission for full-run, notebook execution, tuning,
  test/holdout scoring, or any model-performance claim.

## PM Answer

Ian's model-adjustment phase is close, but it is not ready to start yet.
There are four small gates left before using validation results to adjust the
combined MS-DLinear+TCN route:

1. Route-safety test gate: write focused tests that fail unless
   validation-only output is test-blind and artifact-safe.
2. Runner implementation gate: add the smallest torch validation-only route
   that satisfies the route-safety tests.
3. Tiny validation-only runtime smoke: run the bounded route only after the
   runner implementation gate passes.
4. Artifact review gate: inspect the smoke artifact and decide whether it is
   only protocol evidence or whether it can justify a later model-adjustment
   task.

Only after those gates pass should the project start model adjustment, such as
changing MS-DLinear+TCN capacity, sequence settings, training settings, or
feature/model comparisons.

## Current State

The MS-DLinear+TCN tests stage passed, but it does not authorize runtime smoke:

- `tests/test_ms_dlinear_tcn_classifier.py` verifies the model contract.
- `tests/test_phase1b_local_runner.py` verifies runner construction and CLI
  dispatch for `ms_dlinear_tcn`.
- The current torch runner path still trains and then evaluates the test split.
- `--validation-only-report` is currently restricted to `sklearn_logreg` and
  `lightgbm`.

Therefore, a command such as `--models ms_dlinear_tcn --smoke` is not a
validation-only smoke. It is a small torch training run followed by test-set
evaluation.

## Evidence Anchors

| Source | Current behavior | Gate decision |
| --- | --- | --- |
| `scripts/phase1b_local/local_baseline_matrix.py` lines 201-205 | `--validation-only-report` rejects torch model family. | Torch validation-only route does not exist. |
| `scripts/phase1b_local/local_baseline_matrix.py` lines 2028-2031 | `run_model_once` calls `trainer.fit(...)` and loads the best checkpoint. | Any current torch runtime smoke trains. |
| `scripts/phase1b_local/local_baseline_matrix.py` lines 2033-2057 | `run_model_once` evaluates `prepared.test_dataset` and ticker test datasets. | Current torch smoke reads test metrics. |
| `scripts/phase1b_local/local_baseline_matrix.py` lines 2078-2085 | `build_model("ms_dlinear_tcn", ...)` is wired. | Construction is available, but runtime policy is not safe. |
| `scripts/phase1b_local/local_baseline_matrix.py` lines 2265-2271 | validation-only fields exist for non-torch routes. | Reuse the field contract for a future torch validation-only path. |

## Options Considered

### Option A: Dispatch/Manifest-Only Smoke

Allowed behavior:

- Parse CLI.
- Prepare metadata.
- Confirm `ms_dlinear_tcn` dispatch target.
- Optionally build the model object.
- Write no training metrics and read no train/validation/test windows.

Pros:

- Safest and fastest.
- No training.
- No test-set exposure.

Cons:

- Does not prove the model can train or produce validation rows.
- Still needs a separate runtime validation task later.

Decision: acceptable only as a dispatch gate, not as validation evidence.

### Option B: True Torch Validation-Only Runtime Path

Allowed behavior:

- Split chronologically by ticker.
- Fit scaler only on pooled train after per-ticker split.
- Train only on train windows.
- Use validation windows for monitoring/reporting.
- Emit validation-only rows with `test_metrics_embargoed=True` and
  `test_metrics_used=False`.
- Never featurize, score, baseline, or report the test split.

Pros:

- Directly unblocks the model route toward Ian's model-adjustment stage.
- Produces the artifact shape needed for the next smoke review.

Cons:

- Requires runner changes and focused tests.
- It is training, even if tiny, so it needs explicit approval and small bounds.

Decision: recommended next implementation route.

### Option C: Defer Runtime And Add Route-Safety Tests First

Allowed behavior:

- Keep MS-DLinear+TCN as model/dispatch tested only.
- Do not run or implement a validation-only torch path.
- Add tests that define the validation-only embargo and artifact-safety
  contract before runner implementation.

Pros:

- Lowest risk now.
- Turns the current NO-GO into executable failing tests.
- Prevents the later implementation from accidentally using the normal torch
  test-evaluation path.

Cons:

- Does not produce a runtime artifact by itself.

Decision: recommended now.

## Agent Reconciliation

Two PM agents independently recommended blocking direct runtime work:

- Runner-gate scout: choose Option C now, then Option B later. Tests/spec must
  define torch validation-only behavior before implementation.
- Adversarial route checker: block runtime work because validation-only output
  can still expose holdout/test manifest fields, output directories can be
  overwritten, and route docs are stale relative to the current code.

The PM decision is therefore revised:

- Choose Option C now: write the route-safety tests/spec first.
- Defer Option B: implement the true torch validation-only path only after the
  tests are approved and fail for the current runner behavior.

## Next Minimal Task

```text
PM-MS-DLINEAR-TCN-RUNTIME-GATE-019 -- validation-only route-safety tests

Task type: test-first / tests-only / no runner implementation / no notebook / no real-data smoke.

Goal:
- Add focused tests that prove a future validation-only route is test-blind and
  artifact-safe before any Ian-style MS-DLinear+TCN model adjustment.
- The tests may fail until a later runner implementation task.

Allowed files:
- tests/test_phase1b_local_runner.py

Forbidden files/actions:
- No `scripts/**` edits in this task.
- No `ml_utils/**` edits.
- No docs edits after this lock packet.
- No notebook execution.
- No real-data smoke.
- No training.
- No data/ or checkpoints/ writes outside pytest `tmp_path`.
- No evidence_matrix/wiki/Zotero update.
- No model-quality or performance claim.
- No test metric readout.
- No threshold selection.
- No `git add .` or staging notebooks/.codegraph.

Required tests:
- Validation-only `main()` must not write manifest fields exposing holdout/test
  label distribution or counts.
- Forbidden manifest fields include `test_rows`, `test_retained_labels`,
  `test_nan_labels`, `n_test_windows`, `test_up_pct`, `holdout_start_ts`, and
  `holdout_end_ts`.
- Validation-only output must fail or stop before overwriting an existing
  run/artifact directory.
- Explicit mentor route metadata must require:
  `feature_set_id=mentor_clean_v1`,
  `label_mode=no_trade_band`,
  `threshold_bps=5.0`,
  `decision_time_policy=post_bar_close_completed_bar`,
  `scaler_id=standard_pooled_train_only_v1`,
  `scaler_fit_scope=pooled_train_after_per_ticker_chronological_split`,
  `threshold_source=fixed_pre_registered_5bps`.
- If a torch validation-only dispatch test is added, it must monkeypatch any
  training path and fail if `run_model_once` is called.

Validation commands:
- E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_phase1b_local_runner.py
- E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_phase1b_local_runner.py -q -p no:cacheprovider

Stop rules:
- Stop if the task starts patching the runner.
- Stop if the tests need `ml_utils` model changes.
- Stop if tests would execute real training or notebooks.
- Stop if validation-only mode would expose any concrete test/holdout field.
- Stop if CLI would allow validation-only mode without explicit
  `--label-mode no_trade_band --threshold-bps 5.0 --feature-set mentor_clean_v1`.
- Stop if the route would overwrite an existing artifact directory.
- Stop if the change requires broad trainer abstractions, callbacks, registries,
  or notebook edits.
```

## Model Adjustment Boundary

The future model-adjustment phase may start only after:

- PM-MS-DLINEAR-TCN-RUNTIME-GATE-019 adds and validates the route-safety tests.
- PM-MS-DLINEAR-TCN-RUNTIME-GATE-020 implements the true torch validation-only
  path and passes those tests.
- One tiny validation-only MS-DLinear+TCN artifact is generated under the locked
  `mentor_clean_v1` protocol.
- The artifact review confirms no test metrics were read or reported.

Until then, `ms_dlinear_tcn` is implementation-tested and runner-dispatch-tested,
but not validation-runtime-approved.
