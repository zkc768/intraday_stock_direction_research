# PM-065B-CLI-ALIGNED-VALIDATION-ONLY-MODEL-TESTING-SPEC

Date: 2026-06-01

Status: validation-only model-testing spec / runtime may proceed only if
preflight passes / no evidence promotion / no test access

## Root-Cause Repair

PM-065 blocked because the parent prompt required unsupported runner flags. The
live runner exposes `--seeds`, not `--seed`, and does not expose
`--threshold-source`, `--decision-time-policy`, `--scaler-id`, or
`--scaler-fit-scope`.

PM-065B aligns the command contract to the current CLI by using `--seeds 42`
and by verifying route-lock fields from emitted metadata and result artifacts
instead of passing unsupported metadata-lock flags.

No runner code, model code, notebook, test, or route-control implementation is
edited by this spec.

## Protocol Question

Can the existing MS-DLinear+TCN route run one frozen-route validation-only
diagnostic under the current CLI, producing inspectable artifacts with
route-lock metadata, validation-only report scope, train-only scaling metadata,
and test/holdout embargo fields, without route drift?

This is model testing because it runs the existing model entrypoint once under
validation-only scope.

This is not model selection because MS-DLinear+TCN is chosen for neural-route
coverage after LightGBM PM-059/060 and because PM-047/048 already define an
accepted command shape. It is not chosen because of validation metric values,
model superiority, profitability, robustness, publishability, Ian-result
success, or test readiness.

## Frozen Route Locks

| Lock | Value | How enforced |
| --- | --- | --- |
| `candidate` | `A` | CLI-passed |
| `feature_set_id` | `mentor_clean_v1` | CLI-passed as `--feature-set mentor_clean_v1`; verify artifact field |
| `label_mode` | `no_trade_band` | CLI-passed as `--label-mode no_trade_band`; verify artifact field |
| `threshold_bps` | `5.0` | CLI-passed as `--threshold-bps 5.0`; verify artifact field |
| `model_family` | `torch` | CLI-passed as `--model-family torch`; verify artifact field |
| `models` | `ms_dlinear_tcn` | CLI-passed as `--models ms_dlinear_tcn`; verify artifact field |
| `feature_view` | `last_step` | CLI-passed as `--feature-view last_step`; verify artifact field if emitted |
| `window_size` | `12` | CLI-passed as `--window-size 12`; verify artifact field |
| `tickers` | `CSCO JPM KO MSFT WMT` | CLI-passed as explicit five-ticker scope; verify artifact field |
| `seeds` | `42` | CLI-passed as `--seeds 42`; verify artifact field |
| `max_epochs` | `1` | CLI-passed as `--max-epochs 1`; verify artifact field |
| `batch_size` | `256` | CLI-passed as `--batch-size 256`; verify artifact field |
| `split_mode` | `calendar` | CLI-passed as `--split-mode calendar`; verify artifact field |
| train interval | `[1998-01-02, 2013-09-16)` | CLI-passed boundaries; verify artifact fields |
| validation interval | `[2013-09-16, 2017-01-25)` | CLI-passed boundaries; verify artifact fields |
| holdout metadata interval | `[2017-01-25, 2020-06-06)` | CLI-passed boundaries; verify metadata only, not scoring |
| `threshold_source` | `fixed_pre_registered_5bps` | Metadata-verified only; do not pass unsupported CLI flag |
| `decision_time_policy` | `post_bar_close_completed_bar` | Metadata-verified only; do not pass unsupported CLI flag |
| `scaler_id` | `standard_pooled_train_only_v1` | Metadata-verified only; do not pass unsupported CLI flag |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | Metadata-verified only; do not pass unsupported CLI flag |
| `report_scope` | `validation_only` | Metadata/result verified from validation-only runner path |
| `selection_scope` | `validation_only` | Metadata/result verified from validation-only runner path |
| `test_metrics_embargoed` | `True` | Metadata/result verified |
| `test_metrics_used` | `False` | Metadata/result verified |

Ian guidance and papers are design rationale and blocker-check context only.
They are not local evidence, validation results, or permission to open
holdout/test.

## Exact Runtime Command

Precondition: all Phase A validation checks must pass, and the output parent
directory must not already exist.

Output parent directory:

```text
checkpoints\pm_065b_ms_dlinear_tcn_validation_only_20260601
```

Run exactly this command from `E:\codex_workspace\projects\hf_stock_clf`:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir checkpoints\pm_065b_ms_dlinear_tcn_validation_only_20260601 `
  --candidate A `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --model-family torch `
  --models ms_dlinear_tcn `
  --validation-only-report `
  --validation-only-per-ticker `
  --feature-view last_step `
  --window-size 12 `
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

The command intentionally omits `--full-run`, `--smoke`,
`--max-rows-per-ticker`, `--shuffle-train-labels`, `--manifest-only`,
LightGBM, any second model, seed search, threshold search, feature variant,
scaler variant, decision-time variant, label variant, split variant, broad
grid, notebook execution, and test/holdout scoring.

Do not add unsupported flags. Do not edit code to add flags.

## Phase A Validation

Before runtime, verify:

- this spec file exists;
- this spec file ends with LF;
- this spec file has no trailing whitespace;
- `git diff --check` is clean;
- the output parent directory does not already exist;
- tracked/cached diff before runtime is only this allowed spec doc or empty;
- cached diff is empty because this gate does not stage or commit.

Stop if any Phase A validation check fails.

## Phase B Runtime Scope

Allowed runtime action: run exactly one command, exactly as written above, only
if Phase A passes.

Runtime scope:

- one model family: `torch`;
- one model: `ms_dlinear_tcn`;
- one seed via `--seeds 42`;
- five tickers: `CSCO JPM KO MSFT WMT`;
- `max_epochs=1`;
- `batch_size=256`;
- validation-only report/per-ticker;
- fixed route locks and calendar boundaries;
- no row cap.

Stop if the command exits nonzero. Report the command and error without
broadening, retrying for modeling reasons, or changing the route.

## Phase C Immediate Integrity Checks

After runtime, inspect only enough generated artifacts to verify:

- output parent exists and has exactly one timestamped child run directory;
- required files exist: `metadata.json`, `results.csv`, `manifest.csv`;
- row counts for `results.csv` and `manifest.csv`;
- route fields match the frozen locks;
- split ranges match the frozen calendar locks;
- result rows are validation-only;
- no concrete forbidden `test_*` or `holdout_*` scoring metric columns appear,
  excluding allowed flags `test_metrics_embargoed` and `test_metrics_used`.

If the child path or metadata includes `smoke`, treat it as a runner naming
caveat only if row caps are null and validation-only fields hold. Otherwise
block.

Do not perform a full artifact review document in this task. The next PM gate
after PASS or PASS_WITH_CAVEAT should be a separate artifact review gate.

## Stop Rules

Stop before runtime if:

- `AGENTS.md` or `docs/ENVIRONMENT.md` cannot be read;
- git state is not inspectable;
- `HEAD` differs from `origin/master`;
- `HEAD` is not `0967a597574efeb60a1db340eb0068cbf46e115e`;
- tracked diff before writing is non-empty;
- cached diff before writing is non-empty;
- unknown risky untracked files appear;
- required route/control docs are missing or contradict the frozen locks;
- runner help does not support any flag in the corrected command;
- the output parent already exists;
- the runtime command would include forbidden flags or omit route locks.

Stop after runtime if:

- runtime exits nonzero;
- output creates more than one timestamped child run unexpectedly;
- required artifacts are missing;
- artifacts expose forbidden concrete test/holdout scoring columns;
- any wording would claim model quality, choose a model, or promote validation
  diagnostics as evidence.

## Claim Safety

Allowed language: validation-only diagnostic, protocol-observability artifact,
route-lock integrity check, non-metric runtime gate.

Forbidden language: model quality, improvement, robustness, profitability,
publishability, Ian-result success, test readiness, model selection, evidence
promotion, or holdout/test authorization.

Validation metrics must not be quoted, compared, ranked, or used to choose a
model or next action in this task.
