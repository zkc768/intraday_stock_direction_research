# PM-MS-DLINEAR-TCN-VAL-026 Route Design

Date: 2026-05-30
Task: PM-MS-DLINEAR-TCN-VAL-026
Mode: docs-only design; no torch runtime; no notebook execution; no commit

## Superseded / Current Live State

This route-design packet is historical and was written before the current
torch validation-only implementation and tests existed. Preserve the content
below as the audit trail, but do not use its old "current real torch runner is
not validation-only" statements as live instructions.

Current live state for later PM gates:

- The combined `MultiScaleDLinearTCNClassifier` exists.
- Runner dispatch for `--model-family torch --models ms_dlinear_tcn` exists.
- The PM-locked torch validation-only route requires
  `--validation-only-report`, `--validation-only-per-ticker`,
  `--feature-set mentor_clean_v1`, `--label-mode no_trade_band`, and explicit
  `--threshold-bps 5.0`.
- In validation-only mode, emitted artifacts must carry
  `report_scope=validation_only`, `selection_scope=validation_only`,
  `test_metrics_embargoed=True`, and `test_metrics_used=False`, with no
  concrete test/holdout scoring metrics.
- For PM-SEQUENCE-045, the stricter current policy supersedes the historical
  allowance below for internal test materialization: the smoke must not build,
  score, expose, or select from test/holdout datasets or metrics.
- Calendar holdout timestamps may appear only as split-boundary metadata if
  the runner emits them; they are not scoring outputs, selection evidence, or
  test authorization.
- This packet does not authorize runtime by itself. Runtime is allowed only by
  a later PM smoke gate after doc reconciliation is committed and the tree/index
  are clean.

## PM Status

The LightGBM validation-only lane is closed as protocol and artifact evidence only. The review note was committed as:

```text
72c5c90 docs: record LightGBM validation-only review
```

MS-DLinear+TCN is implemented and dispatch-tested, but the current real torch runner is not validation-only:

- `--validation-only-report` rejects torch model family before data preparation.
- The normal torch branch calls `run_model_once(...)`.
- `run_model_once(...)` trains with `trainer.fit(...)`, loads `best.pt`, evaluates `prepared.test_dataset`, then evaluates per-ticker test datasets.
- The result rows from that path use `split="test"` and include test metrics, test distribution fields, confusion matrices, classification reports, and dummy-baseline deltas.

Therefore no current MS-DLinear+TCN runtime smoke is authorized by this gate.

## PM Decision

PM-MS-DLINEAR-TCN-VAL-026 is a docs-only route design gate.

Do not implement or run the route in this phase. The next safe action is a separate test-first gate that defines the torch validation-only contract before any runner implementation or runtime smoke.

## What A True Torch Validation-Only Route Means

The safer minimal route is a tiny train/validation-only torch branch, not a no-training dispatch-only route.

Rationale:

- Dispatch-only already exists as wiring evidence and does not prove the model can produce validation-only artifacts.
- A useful future smoke should fit weights on train windows and report validation diagnostics only.
- The route must never score, baseline, summarize, expose, or select from test data.
- Any validation diagnostics remain protocol/runtime observability only, not model-performance evidence.

The route may still materialize the test split internally if the governing policy remains the current LightGBM policy: no test scoring, no test metric exposure, and no test-driven selection. If a stricter future policy requires no holdout/test materialization at all, this design is insufficient and a larger data-prep branch is required.

## Minimal Safe Seam

Current unsafe path:

1. `main()` rejects `--validation-only-report` for torch.
2. The torch branch calls `run_model_once(...)`.
3. `run_model_once(...)` trains and then calls `evaluate_scope(...)` on `prepared.test_dataset`.
4. `evaluate_scope(...)` emits test-scoped metrics and fields.

Future minimal implementation should:

1. Add an explicit PM route validator for torch validation-only MS-DLinear+TCN.
2. Require `--models ms_dlinear_tcn`.
3. Require `--validation-only-report`.
4. Require `--feature-set mentor_clean_v1`.
5. Require `--label-mode no_trade_band`.
6. Require explicit `--threshold-bps 5.0`, so `threshold_source` is `fixed_pre_registered_5bps`, not only a default.
7. Branch away from the current test-evaluation path before any test scoring.
8. Train only on train windows, monitor/report validation only, and emit validation-only artifact fields.
9. Reuse the validation-only embargo field contract: `report_scope=validation_only`, `selection_scope=validation_only`, `test_metrics_embargoed=True`, `test_metrics_used=False`.
10. Avoid emitting `model_macro_f1`, generic `delta_macro_f1_vs_dummy`, `n_test_windows`, `test_up_pct`, confusion matrices, classification reports, or any `test_*` fields beyond the two embargo booleans.

The route should not modify label semantics, threshold policy, feature sets, scaler policy, model capacity, or paper/evidence claims.

## Required Test-First Gate

Next gate should be tests first, with no runtime smoke and no real torch training.

Allowed files for that gate:

- `tests/test_phase1b_local_runner.py`
- `scripts/phase1b_local/local_baseline_matrix.py` only after failing tests are accepted

Minimum failing tests before implementation:

1. Replace the current torch validation-only rejection expectation with a PM-locked torch validation-only dispatch test.
   - CLI includes `--model-family torch`, `--models ms_dlinear_tcn`, `--validation-only-report`, `--feature-set mentor_clean_v1`, `--label-mode no_trade_band`, `--threshold-bps 5.0`.
   - Monkeypatch current training path to fail if called before the safe validation-only branch.
   - Assert metadata carries the shared mentor route invariants.

2. Add a torch validation-only embargo test for the future branch.
   - Use `_toy_prepared_data()`.
   - Monkeypatch test-dataset loaders or evaluation helpers to fail if `prepared.test_dataset`, `prepared.test_datasets_by_ticker`, `prepared.y_test`, or `prepared.y_test_by_ticker` are scored or baselined.
   - Assert result rows use `split="validation"`.
   - Assert every result row passes `_assert_validation_only_no_test_exposure(row)`.

3. Add a per-ticker validation-only test if `--validation-only-per-ticker` remains in scope for torch.
   - Assert row tickers are `pooled` plus ticker rows from validation datasets.
   - Assert no test or holdout fields leak into those rows.

4. Add metadata guard tests for explicit PM route locks.
   - Missing `--threshold-bps 5.0` must fail for this route.
   - Wrong `--feature-set`, wrong `--label-mode`, or model list other than `ms_dlinear_tcn` must fail before training.

5. Add artifact schema tests.
   - Manifest and results must omit `test_rows`, `test_retained_labels`, `test_nan_labels`, `n_test_windows`, `test_up_pct`, `holdout_start_ts`, and `holdout_end_ts`.
   - If strict artifact embargo is adopted, also forbid `calendar_holdout_start_ts` and `calendar_holdout_end_ts` in metadata.

## Future Command Class

This command class is not approved for this gate. It is the target shape for a later smoke only after the test/code gate passes.

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --model-family torch `
  --models ms_dlinear_tcn `
  --validation-only-report `
  --validation-only-per-ticker `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --tickers CSCO JPM KO MSFT WMT `
  --max-rows-per-ticker <small_cap> `
  --max-epochs <tiny_bound> `
  --output-dir checkpoints\pm_ms_dlinear_tcn_val_<task_id>
```

Future smoke stop rules:

- Stop if the output directory already exists.
- Stop if route validation does not require explicit `mentor_clean_v1`, `no_trade_band`, and `threshold_bps=5.0`.
- Stop if the command enters the current test-evaluation path.
- Stop if test metrics, test baselines, test distribution fields, confusion matrices, or classification reports would be produced.
- Stop if validation diagnostics are used to select threshold, feature set, model family, sequence length, model capacity, seed, or a winner.
- Stop before writing evidence matrix, wiki, Zotero, or paper claims.

## Shared Policy Table

| Area | Allowed in future torch validation-only route | Forbidden |
| --- | --- | --- |
| Train | Fit model weights on train windows only. | Fit on validation or test windows. |
| Validation | Monitor/report validation diagnostics only. | Treat validation diagnostics as performance evidence. |
| Test | Embargoed from scoring, baselines, result rows, and selection. | Test metric readout, test baselines, confusion matrix, classification report, or model choice. |
| Scaler | Train-only pooled scaler after per-ticker chronological split. | Full-data scaler fit or test-informed scaling. |
| Threshold | Fixed pre-registered `5.0` bps only. | Threshold tuning or default-only threshold provenance. |
| Artifacts | Validation-only scope plus embargo booleans. | Forbidden holdout/test exposure fields. |
| Claims | Protocol and artifact readiness only. | Local model-performance claim. |

## Agent Findings Integrated

Torch runner path inspector:

- Confirmed current torch route rejects validation-only mode and otherwise falls through to `run_model_once(...)`.
- Confirmed `run_model_once(...)` trains and immediately evaluates pooled and per-ticker test datasets.
- Recommended a future validation-only branch before test evaluation, with validation dataset evaluation only.

Adversarial evaluation-policy reviewer:

- Marked current MS-DLinear+TCN runtime smoke as NO-GO.
- Flagged that the existing MS-DLinear+TCN CLI test is dispatch-only, not validation-only safety evidence.
- Flagged that torch lacks LightGBM-style explicit PM route validation.
- Flagged a metadata policy question for calendar holdout boundaries.

## Recommendation

Next minimal task:

```text
PM-MS-DLINEAR-TCN-VAL-027 -- test-first torch validation-only safety gate
```

Recommended scope:

- Tests first in `tests/test_phase1b_local_runner.py`.
- No runner implementation until the failing tests are reviewed.
- No runtime smoke.
- No real torch training.
- No notebook execution.
- No evidence, wiki, Zotero, or paper-claim writes.

Implementation can follow only after those tests define the route contract and fail for the current runner behavior.
