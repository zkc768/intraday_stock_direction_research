# PM-MS-DLINEAR-TCN-FULLVAL-PLAN-047

Date: 2026-05-31
Status: planning/template artifact only; no runtime
PM decision: prepare a future MS-DLinear+TCN full-input validation-only route
template after the tiny 045 smoke review; do not run the route in this gate.

## Optimized Hard-Rule Prompt

Plan-commit gate first, runtime gate second. Stage 1 may commit exactly
`docs/PM_MS_DLINEAR_TCN_FULLVAL_PLAN_047_2026-05-31.md` and nothing else.
Stage 2 may run exactly one full-input calendar-split MS-DLinear+TCN
validation-only diagnostic command if the Stage 1 commit is HEAD, tracked and
cached diffs are clean, and the output directory is fresh. Do not edit code,
runner tests, notebooks, KB evidence surfaces, claim maps, Zotero, thresholds,
features, labels, scalers, decision policy, or model capacity. Do not run
full-run, notebooks, smoke mode, row-capped mode, tuning/search, or more than
the single approved validation-only command. Stop if the command would enter
non-validation torch, use capped smoke semantics for a full-input route, omit
route locks, expose or score test/holdout metrics, or promote validation
diagnostics as performance evidence.

Claim-safety checklist:

- Treat all validation diagnostics as protocol/runtime observability only.
- Do not state or imply MS-DLinear+TCN works, improves results, is robust,
  profitable, publishable, test-ready, or an Ian-result success.
- Papers remain design rationale and constraints only, not local evidence.
- Calendar holdout timestamps are split-boundary metadata only, not scoring,
  selection, metric evidence, or test authorization.
- No validation metric may select thresholds, hyperparameters, features,
  labels, scaler policy, model capacity, seed, or a winner.

## Prior Gates

| Gate | Status | Planning implication |
| --- | --- | --- |
| PM-RUNTIME-DOC-RECONCILE-044 | committed as `0aca665 docs: reconcile MS-DLinear TCN runtime route` | Stale route docs are superseded by the live validation-only route state. |
| PM-MS-DLINEAR-TCN-CANDIDATE-SMOKE-045 | reviewed as tiny train/validation-only diagnostic | The route produced a protocol-observability artifact only, not model evidence. |
| PM-MS-DLINEAR-TCN-SMOKE-SYNC-COMMIT-046B | committed as `4e1e34e docs: record MS-DLinear TCN candidate smoke review` | The 045 artifact review is git-visible before this plan. |

## Split Semantics

Use the exam analogy in PM-048:

- Train = learn model weights and fit the pooled scaler.
- Validation = mock exam for diagnostics and protocol observability.
- Test/holdout = final unopened exam.

The future PM-048 route may train on train windows and emit validation-only
diagnostics. It must not materialize, score, baseline, summarize, expose, or
select from test/holdout datasets or metrics. Holdout timestamps may appear
only as calendar boundary metadata. The route must not make claims from
validation diagnostics.

## Locked Route

| Field | Locked value |
| --- | --- |
| `model_family` | `torch` |
| `models` | `ms_dlinear_tcn` |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

No PM-048 command may change these values.

## Runner Template Audit

Current runner inspection supports a future full-input validation-only command
only under the calendar split route:

- `--model-family torch --models ms_dlinear_tcn --validation-only-report`
  enters the PM-locked torch validation-only route.
- `validate_torch_validation_only_pm_route(...)` rejects `--full-run`.
- `--validation-only-report` makes `prepare_data(...)` run with
  `include_test_data=False`, so the validation-only data-prep branch uses only
  train and validation splits.
- `--split-mode calendar` requires all train, validation, and holdout boundary
  arguments.
- `--split-mode calendar` rejects `--max-rows-per-ticker`.
- With calendar split active, the runner sets `effective_max_rows_per_ticker`
  to `None`, which is the required full-input condition.
- Because `--full-run` is rejected, `resolve_run_mode(...)` still names the
  run internally as `smoke`; PM-048 must treat any `smoke` token in run IDs or
  diagnostic scope as a runner naming limitation, not capped data semantics.
- Because run mode remains internally `smoke`, PM-048 must pass
  `--tickers CSCO JPM KO MSFT WMT` explicitly. Otherwise the runner default
  collapses the ticker list to `CSCO`.

## Calendar Split Policy

PM-048 should reuse the LightGBM 041/042 strict common trading-date boundaries
as the pre-registered boundary policy:

| Split | Start | End |
| --- | --- | --- |
| train | `1998-01-02` | `2013-09-16` |
| validation | `2013-09-16` | `2017-01-25` |
| holdout metadata boundary | `2017-01-25` | `2020-06-06` |

The interval convention is half-open: start inclusive, end exclusive. Holdout
timestamps may be carried as metadata only. They must not become scoring
outputs, selection fields, evidence, or authorization to open test/holdout.

## Candidate Command Policy For PM-048

Future PM-048 must:

- Use `--model-family torch --models ms_dlinear_tcn`.
- Use `--validation-only-report --validation-only-per-ticker`.
- Use `--feature-set mentor_clean_v1 --label-mode no_trade_band
  --threshold-bps 5.0`.
- Use explicit five-ticker scope: `CSCO JPM KO MSFT WMT`.
- Use `--split-mode calendar` with the boundaries above.
- Use a fresh output directory and stop if it already exists.
- Omit row caps so `effective_max_rows_per_ticker=None`.
- Keep any epoch/batch settings fixed in the prompt, not searched.

Future PM-048 must omit:

- `--full-run`
- `--smoke`
- `--max-rows-per-ticker`
- `--shuffle-train-labels`
- `--manifest-only`
- any threshold, feature-set, label-mode, scaler, decision-time, seed-search,
  hyperparameter-search, or model-capacity variant

Suggested PowerShell command shape for the future gate:

```powershell
$PYTHON_INTERPRETER = "E:\codex_workspace\_envs\py311_shared\python.exe"
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUT = "checkpoints\pm_ms_dlinear_tcn_fullval_048_$STAMP"

if (Test-Path -LiteralPath $OUT) {
    throw "STOP: output path already exists: $OUT"
}

& $PYTHON_INTERPRETER scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir $OUT `
  --candidate A `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --model-family torch `
  --models ms_dlinear_tcn `
  --validation-only-report `
  --validation-only-per-ticker `
  --tickers CSCO JPM KO MSFT WMT `
  --seeds 42 `
  --max-epochs 1 `
  --batch-size 256 `
  --split-mode calendar `
  --train-start-ts 1998-01-02 `
  --train-end-ts 2013-09-16 `
  --val-start-ts 2013-09-16 `
  --val-end-ts 2017-01-25 `
  --holdout-start-ts 2017-01-25 `
  --holdout-end-ts 2020-06-06
```

The `--max-epochs 1` and `--batch-size 256` values are fixed runtime bounds
for the diagnostic gate, not hyperparameter choices. If parent PM wants a
different fixed training budget, it must be pre-registered before execution and
must not be chosen from validation metrics.

## Artifact Audit Checklist For PM-048

Inspect `metadata.json`, `results.csv`, and `manifest.csv` for:

| Check | Required result |
| --- | --- |
| Route fields | `torch`, `ms_dlinear_tcn`, `mentor_clean_v1`, `no_trade_band`, explicit `5.0`, `fixed_pre_registered_5bps`, `post_bar_close_completed_bar` |
| Split mode | calendar split boundaries match this plan |
| Data cap | `effective_max_rows_per_ticker=None` |
| Ticker scope | `CSCO`, `JPM`, `KO`, `MSFT`, `WMT` |
| Scaler scope | `pooled_train_after_per_ticker_chronological_split` |
| Report scope | `report_scope=validation_only`, `selection_scope=validation_only` |
| Test embargo | `test_metrics_embargoed=True`, `test_metrics_used=False` |
| Results split | every `results.csv` row has `split=validation` |
| Rows | pooled row plus per-ticker validation rows if per-ticker report succeeds |
| Forbidden columns | no concrete `test_*` or `holdout_*` scoring columns beyond approved embargo/status fields |
| Diagnostics | coverage, no-trade, and class-balance fields are present if emitted by the route |
| Claim scope | closeout frames the run as diagnostic/protocol observability only |

Residual caveat: artifact inspection can show that emitted files do not expose
forbidden test/holdout scoring metrics. The stronger no-test-materialization
claim rests on runner code/tests and the PM-048 preflight code audit.

## Stop Rules For PM-048

Stop before execution if:

- The committed PM-047 plan is absent, not the current route basis, or the
  history no longer contains Stage 046B commit `4e1e34e`.
- The command contains `--full-run`.
- The command omits `--validation-only-report`.
- The command omits `--validation-only-per-ticker`.
- The command omits any route lock.
- The command omits explicit `--tickers CSCO JPM KO MSFT WMT`.
- The command contains `--smoke` or `--max-rows-per-ticker`.
- The command does not use `--split-mode calendar`.
- Any calendar boundary differs from this plan without a new PM-approved
  boundary packet.
- The output directory already exists.
- The command changes threshold, feature set, label mode, scaler policy,
  decision-time policy, seed policy, hyperparameters, or model capacity.
- The command introduces notebook execution, a full-run baseline, grid search,
  validation-driven tuning, or paper-result transfer.
- The planned run requires more than the single approved validation-only
  command. No additional reruns are allowed after failure without parent PM
  approval.

Stop after execution if:

- Any artifact exposes concrete test/holdout scoring metrics or selection
  fields.
- Any artifact emits concrete `holdout_start_ts` or `holdout_end_ts` result or
  manifest columns. Only `calendar_holdout_*` metadata fields may carry
  holdout boundary timestamps.
- Any artifact lacks validation-only report/selection scope.
- Any result row is not `split=validation`.
- Calendar holdout metadata is treated as metrics, evidence, or test access.
- Effective row cap is not `None`.
- Output has fewer than pooled plus five per-ticker validation rows, unless the
  runner explicitly records a validation-only per-ticker reporting failure
  without test/holdout access.
- Ticker scope is missing any locked ticker.
- Validation diagnostics are written as model-performance evidence.
- The run cannot be audited without opening test/holdout results.

## Future PM-MS-DLINEAR-TCN-FULLVAL-048 Prompt

```text
PM-MS-DLINEAR-TCN-FULLVAL-048 -- MS-DLinear+TCN full-input validation-only diagnostic

Task type: one bounded full-input calendar-split validation-only runtime gate /
artifact audit. This is not model adjustment, not tuning, not evidence
promotion, and not test authorization.

Required preflight:
1. Read AGENTS.md first and docs/ENVIRONMENT.md.
2. Read docs/PM_MS_DLINEAR_TCN_FULLVAL_PLAN_047_2026-05-31.md.
3. Confirm the committed PM-047 plan is HEAD and the history contains Stage
   046B commit 4e1e34e.
4. Confirm tracked and staged diffs are clean before runtime.
5. Confirm output directory is fresh.
6. Confirm command includes validation-only locks and calendar split boundaries.
7. Confirm command omits --full-run, --smoke, and --max-rows-per-ticker.

Allowed runtime:
Run exactly one command matching the PowerShell template in PM-047, with a
fresh timestamped output directory. The command may train on train windows and
report validation diagnostics only. No additional reruns are allowed after
failure without parent PM approval.

Forbidden:
No code edits, runner/test edits, notebooks, local_baseline_matrix.py reruns
beyond the single approved command, full-run, smoke flag, row cap, threshold
tuning, feature tuning, label changes, scaler changes, decision-time changes,
hyperparameter search, model-capacity tuning, evidence_matrix update,
claim-map update, Zotero update, performance/effectiveness/robustness/
profitability/publishability/Ian-result/model-superiority/test-readiness
claim, or test/holdout scoring.

Required audit:
Inspect metadata.json, results.csv, and manifest.csv for route locks,
calendar split boundaries, effective_max_rows_per_ticker=None,
validation-only scope, embargo booleans, validation-only result rows, five
ticker scope plus pooled row, no concrete test/holdout metric columns, and
diagnostic/non-claim framing.

Stop rules:
Stop on dirty tracked/staged diff, stale or missing PM-047 plan, existing
output directory, missing route lock, missing calendar boundary, --full-run,
--smoke, --max-rows-per-ticker, non-validation torch path, test/holdout metric
exposure, validation metric promotion, or any need for code/test edits.

Final output:
Exact command, exit status, artifact path, PASS/BLOCK audit checklist, files
inspected, files generated, commands run, final git status/diff/cached diff,
and residual caveats. Do not start any follow-up sync or runtime gate.
```

## Success Definition For This Plan

PM-047 succeeds if it leaves a single planning doc that defines a safe future
PM-048 route template and blocker policy. It does not authorize runtime by
itself and does not establish model performance.
