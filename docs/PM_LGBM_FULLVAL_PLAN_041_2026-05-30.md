# PM-LGBM-FULLVAL-PLAN-041

Date: 2026-05-30
Status: planning/template artifact only
PM decision: prepare a conditional PM-LGBM-FULLVAL-042 command template; do not
run it in this gate

## Superseded / Current Live State

This plan remains the historical PM-042 execution template. The later
`docs/PM_LGBM_FULLVAL_042_ARTIFACT_REVIEW_2026-05-30.md` artifact review is
the current non-claim sync target for PM-042 status.

Current live state for later PM gates:

- PM-042 is protocol-observability only and must not be read as LightGBM model
  evidence, tuning permission, or test/holdout authorization.
- "fullval" in this packet means the reviewed uncapped calendar-split
  validation-only diagnostic route, not final validation success, endpoint
  performance, or test readiness.
- PM-042 does not authorize MS-DLinear+TCN tuning. It only supplies reusable
  artifact-audit discipline: validation-only rows, route-lock metadata, embargo
  booleans, coverage/no-trade/class-balance disclosure, and no concrete
  test/holdout metric exposure.

This plan defines the next LightGBM validation-only template after the PM-040
bounded diagnostic smoke. It does not edit code, run the runner, train a model,
execute notebooks, update evidence/wiki/claim-map/Zotero, tune thresholds or
hyperparameters, open test scoring, or create model-performance evidence.

## Prior Gate Status

| Gate | Status | Planning implication |
| --- | --- | --- |
| PM-LGBM-CANDIDATE-PLAN-COMMIT-039B | committed as `81bd07d` | The LightGBM candidate lane is documented as fixed-route, validation-only, and non-claim. |
| PM-LGBM-CANDIDATE-SMOKE-040 | parent-verified diagnostic artifact | The bounded artifact passed route-lock and no forbidden test/holdout column review, but remains protocol observability only. |

## Split Semantics

Use the exam analogy in every PM-042 prompt and closeout:

- Train = learn and fit the model and scaler.
- Validation = mock exam and protocol check.
- Test = final unopened exam.

PM-042 may fit the fixed LightGBM candidate on train windows and emit
validation-only diagnostics. It must not score, expose, materialize, select
from, or make decisions from test/holdout data. Validation diagnostic metrics
are for protocol observability, route-lock audit, coverage disclosure, and
class-balance disclosure only.

## Locked Route

| Field | Locked value |
| --- | --- |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `model_family` | `lightgbm` |
| `model_name` | `lightgbm_lgbm_classifier` |
| `objective` | `binary` |
| `n_estimators` | `100` |
| `learning_rate` | `0.05` |
| `num_leaves` | `31` |
| `random_state` | `42` |
| `n_jobs` | `1` |
| `verbosity` | `-1` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `feature_view` | `last_step` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

No PM-042 command may change these values. Papers, validation diagnostics, old
notebooks, or previous smoke artifacts cannot justify a threshold, feature,
label, scaler, or hyperparameter change.

## Runner Constraint

Current runner inspection found a route constraint that PM-042 must respect:

- `--model-family lightgbm` requires `--validation-only-report`.
- `--model-family lightgbm` rejects `--full-run`.
- `--model-family lightgbm` requires `--feature-set mentor_clean_v1`.
- `--model-family lightgbm` requires `--label-mode no_trade_band`.
- `--model-family lightgbm` requires explicit `--threshold-bps 5.0`.
- The default non-full run mode resolves internally to `smoke`.
- In that internal smoke mode, default tickers collapse to `CSCO` unless
  `--tickers` is explicit, and `effective_max_rows_per_ticker` defaults to
  `20000`.
- Calendar split mode bypasses the row cap because `max_rows_per_ticker` is set
  to `None` when calendar boundaries are provided.

Therefore, PM-042 must not use `--full-run`. If PM-042 needs an uncapped
full-input validation-only pass, it must use pre-registered calendar
boundaries. If those boundaries are not supplied before execution, PM-042 must
stop rather than run a capped ratio split and call it full validation.

## PM-042 Execution Template

Precondition: parent PM must provide exact calendar boundaries before execution:

- `TRAIN_START_TS`
- `TRAIN_END_TS`
- `VAL_START_TS`
- `VAL_END_TS`
- `HOLDOUT_START_TS`
- `HOLDOUT_END_TS`

PowerShell template:

```powershell
$PYTHON_INTERPRETER = "E:\codex_workspace\_envs\py311_shared\python.exe"
$OUT = "checkpoints\pm_lgbm_fullval_042_<YYYYMMDD_HHMMSS>"

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
  --model-family lightgbm `
  --validation-only-report `
  --validation-only-per-ticker `
  --feature-view last_step `
  --window-size 12 `
  --tickers CSCO JPM KO MSFT WMT `
  --seeds 42 `
  --split-mode calendar `
  --train-start-ts <TRAIN_START_TS> `
  --train-end-ts <TRAIN_END_TS> `
  --val-start-ts <VAL_START_TS> `
  --val-end-ts <VAL_END_TS> `
  --holdout-start-ts <HOLDOUT_START_TS> `
  --holdout-end-ts <HOLDOUT_END_TS>
```

Omit these on purpose:

- `--full-run`
- `--smoke`
- `--max-rows-per-ticker`
- `--shuffle-train-labels`
- `--manifest-only`
- any LightGBM hyperparameter override
- any threshold, feature-set, label-mode, scaler, or decision-time variant

## Output Directory Strategy

Use a fresh timestamped parent directory:

```text
checkpoints\pm_lgbm_fullval_042_<YYYYMMDD_HHMMSS>
```

Stop before execution if the parent directory already exists or contains any
artifact. Do not overwrite, merge into, or reuse PM-040 output directories.

The current runner may still create a child directory whose name contains
`smoke` because LightGBM cannot use `--full-run` and the internal non-full run
mode resolves to `smoke`. In PM-042, that child-directory token must be treated
as a runner naming limitation, not as permission to use smoke-sized data or to
weaken the calendar-split/full-input precondition.

## Allowed Actions For PM-042

- Read project rules and this PM-041 plan.
- Confirm the parent PM supplied exact calendar timestamps before execution.
- Confirm the output parent directory is fresh.
- Run exactly one LightGBM validation-only command matching the template.
- Inspect `metadata.json`, `results.csv`, and `manifest.csv`.
- Report route locks, validation-only split markers, fixed LightGBM settings,
  coverage/no-trade/class-balance fields, and absence of forbidden
  test/holdout scoring columns.

## Forbidden Actions For PM-042

- No code edits.
- No runner or test edits.
- No notebook execution.
- No `--full-run`.
- No hyperparameter grid, search, tuning loop, or validation-driven parameter
  choice.
- No threshold tuning or threshold selection.
- No feature-set, label, scaler, or decision-time policy change.
- No test/holdout scoring, metric exposure, materialization, or selection.
- No evidence_matrix/wiki/claim-map/Zotero update.
- No paper-result transfer into local project evidence.
- No model-performance, effectiveness, profitability, publishability, or
  trading-value claim.
- No commit of checkpoint artifacts.

## Artifact Audit Checklist For PM-042

Inspect `metadata.json`, `results.csv`, and `manifest.csv` for:

| Check | Required result |
| --- | --- |
| Route fields | `mentor_clean_v1`, `no_trade_band`, explicit `5.0`, `fixed_pre_registered_5bps`, `post_bar_close_completed_bar` |
| Scaler scope | `pooled_train_after_per_ticker_chronological_split` |
| LightGBM identity | `model_family=lightgbm`, `model_name=lightgbm_lgbm_classifier` |
| Fixed params | `objective=binary`, `n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=1`, `verbosity=-1` |
| Feature view | `feature_view=last_step` |
| Report scope | `report_scope=validation_only`, `selection_scope=validation_only` |
| Test embargo | `test_metrics_embargoed=True`, `test_metrics_used=False` |
| Results split | every `results.csv` row has `split=validation` |
| Coverage diagnostics | post-filter validation window counts, trade/no-trade counts and rates, class-balance counts and percentages are present |
| Forbidden columns | no concrete `test_*` or `holdout_*` scoring columns beyond approved embargo booleans |
| Claim scope | metadata and closeout frame the run as diagnostic/protocol observability only |

Residual caveat to keep in the closeout: emitted artifacts can show no
forbidden test/holdout scoring exposure, but artifacts alone do not prove
internal no-test/holdout materialization. That stronger claim rests on the
previously committed runner code/tests and any PM-042 preflight code audit.

## Stop Rules For PM-042

Stop before execution if:

- exact calendar boundaries are missing or not pre-registered;
- the output parent directory exists;
- the command contains `--full-run`;
- the command contains `--smoke` or `--max-rows-per-ticker`;
- the command omits any locked route flag;
- the command changes threshold, feature set, label mode, scaler policy,
  decision-time policy, or LightGBM fixed parameters;
- the command introduces hyperparameter search or validation-driven selection;
- the command would run a notebook or full-run baseline.

Stop after execution if:

- any artifact exposes concrete test/holdout scoring metrics or selection
  fields;
- any artifact lacks validation-only report/selection scope;
- any result row is not `split=validation`;
- coverage/no-trade/class-balance diagnostics are missing;
- validation diagnostics are written as model-performance evidence;
- the run cannot be audited without opening test/holdout results.

## Success Definition

PM-042 succeeds only if it emits a complete validation-only diagnostic artifact
under the locked LightGBM route, with fixed candidate settings,
coverage/no-trade/class-balance disclosure, and no forbidden test/holdout
scoring exposure.

Success does not mean LightGBM is effective, better, profitable, publishable,
tradable, paper-evidence-backed, or ready for test evaluation.

## Decision After PM-042

After PM-042, parent PM should perform artifact review first. If the artifact
passes, a separate non-claim sync gate may record it as protocol observability
only. Then parent PM can decide whether to move the MS-DLinear+TCN candidate
lane forward or request a smaller LightGBM cleanup gate.

PM-041 and PM-042 do not authorize test scoring, test exposure, threshold
selection, feature selection, hyperparameter search, or evidence-matrix
promotion.
