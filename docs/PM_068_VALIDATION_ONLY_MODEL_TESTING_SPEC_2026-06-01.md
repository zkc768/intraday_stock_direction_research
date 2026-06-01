# PM-068 Validation-Only Model-Testing Spec

Date: 2026-06-01

Status: GO / validation-only protocol question / runtime may proceed only if
pre-runtime checks pass / no evidence promotion / no test access

## Role And Scope

Role: PM plus agent coordinator with route, CLI contract, leakage, claim-scope,
artifact-integrity, and adversarial review roles.

This gate defines exactly one non-metric protocol question for a controlled
validation-only runtime after PM-067. It does not authorize model selection,
validation metric quotation, evidence promotion, final test/holdout access,
route changes, code edits, notebook execution, staging, commit, push, or KB
writes.

## Context

PM-067 accepted, committed, pushed, and synced the PM-065B/PM-066 docs at:

```text
202ae1bf3d4b05c8087b4fdb97ca5fd0bd46b758
```

PM-065B already repaired the unsupported-flag issue for the MS-DLinear+TCN
validation-only route by using `--seeds 42` and by verifying emitted metadata
for route-lock fields that are not CLI flags. PM-066 reviewed that artifact as
`PASS_WITH_CAVEAT` because `smoke` appears in runner naming while row caps are
null and validation-only locks hold.

PM-068 is not a repeat question about whether the CLI-aligned command shape was
once possible. Its distinct protocol question is whether the accepted and pushed
current HEAD can produce one fresh pre-registered validation-only artifact from
a clean tracked state, under the same frozen route locks, for a later PM-070
artifact review.

## Non-Metric Protocol Question

Can the current pushed HEAD `202ae1bf3d4b05c8087b4fdb97ca5fd0bd46b758` produce
one fresh, pre-registered MS-DLinear+TCN validation-only model-testing artifact
under the frozen route locks, using only supported CLI flags and emitted
metadata verification, without route drift, validation metric use, model
selection, or test/holdout scoring?

Decision: GO, if and only if all pre-runtime checks below pass.

This question is non-metric because success depends only on command contract,
route-lock preservation, validation-only artifact fields, file completeness,
and test/holdout embargo surfaces. It does not depend on validation metric
values, rankings, comparisons, or deltas.

## Frozen Route Locks

| Lock | Value | Enforcement |
| --- | --- | --- |
| `candidate` | `A` | CLI-passed |
| `feature_set_id` | `mentor_clean_v1` | CLI-passed and artifact-verified |
| `label_mode` | `no_trade_band` | CLI-passed and artifact-verified |
| `threshold_bps` | `5.0` | CLI-passed and artifact-verified |
| `threshold_source` | `fixed_pre_registered_5bps` | Artifact-verified only |
| `decision_time_policy` | `post_bar_close_completed_bar` | Artifact-verified only |
| `scaler_id` | `standard_pooled_train_only_v1` | Artifact-verified only |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | Artifact-verified only |
| train interval | `[1998-01-02, 2013-09-16)` | CLI-passed and artifact-verified |
| validation interval | `[2013-09-16, 2017-01-25)` | CLI-passed and artifact-verified |
| holdout metadata interval | `[2017-01-25, 2020-06-06)` | CLI-passed as metadata boundary only |
| `calendar_interval_convention` | `half_open_start_inclusive_end_exclusive` | Artifact-verified |
| `model_family` | `torch` | CLI-passed and artifact-verified |
| `models` | `ms_dlinear_tcn` | CLI-passed and artifact-verified |
| `feature_view` | `last_step` | CLI-passed and artifact-verified if emitted |
| `window_size` | `12` | CLI-passed and artifact-verified |
| `tickers` | `CSCO JPM KO MSFT WMT` | CLI-passed and artifact-verified |
| `seeds` | `42` | CLI-passed as `--seeds 42` and artifact-verified |
| `max_epochs` | `1` | CLI-passed and artifact-verified |
| `batch_size` | `256` | CLI-passed and artifact-verified |
| `split_mode` | `calendar` | CLI-passed and artifact-verified |
| `report_scope` | `validation_only` | Artifact-verified |
| `selection_scope` | `validation_only` | Artifact-verified |
| `test_metrics_embargoed` | `True` | Artifact-verified |
| `test_metrics_used` | `False` | Artifact-verified |

Ian guidance and papers remain rationale/constraint context only. They are not
local evidence, metric evidence, model-selection evidence, or test/holdout
authorization.

## Output Root

The PM-069 runtime, if allowed, must use exactly this output parent root:

```text
checkpoints\pm_069_controlled_validation_only_model_testing_20260601
```

The output parent root must not exist before runtime.

## Exact Runtime Command

Run exactly this command from `E:\codex_workspace\projects\hf_stock_clf` only
after all pre-runtime checks pass:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir checkpoints\pm_069_controlled_validation_only_model_testing_20260601 `
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

This command intentionally omits `--full-run`, `--smoke`,
`--max-rows-per-ticker`, `--shuffle-train-labels`, `--manifest-only`, LightGBM,
sklearn-logreg, any second model, any additional seed, threshold search,
feature search, scaler change, decision-time change, label change, split
variant, hyperparameter search, notebook execution, and test/holdout scoring.

Do not add unsupported CLI flags such as `--seed`, `--threshold-source`,
`--decision-time-policy`, `--scaler-id`, or `--scaler-fit-scope`.

## Pre-Runtime Checks

PM-069 may run only if all checks pass:

- `AGENTS.md` and `docs/ENVIRONMENT.md` are readable.
- Git state is inspectable.
- `HEAD == origin/master == 202ae1bf3d4b05c8087b4fdb97ca5fd0bd46b758`.
- Ahead/behind is `0 0`.
- Tracked diff before this spec was empty.
- Cached diff before this spec was empty.
- Current tracked diff is limited to this PM-068 spec doc.
- Cached diff remains empty.
- `git diff --check` is clean.
- Required PM-059/060 and PM-065B/066 docs exist and do not contradict the
  frozen locks above.
- Runner help/source confirms every CLI flag in the exact command is supported.
- The output parent root does not already exist.
- The command contains no forbidden flags and no unsupported metadata-lock
  flags.

Stop before runtime if any check fails.

## Immediate Runtime Integrity Checks

If the runtime executes and exits zero, inspect only enough generated artifacts
to verify:

- output parent root exists;
- exactly one child run directory exists under the output parent root;
- required files exist in the child run: `metadata.json`, `results.csv`,
  `manifest.csv`;
- `results.csv` and `manifest.csv` row counts are recorded;
- route-lock metadata matches this spec;
- result rows are validation-only;
- `report_scope=validation_only`;
- `selection_scope=validation_only`;
- `test_metrics_embargoed=True`;
- `test_metrics_used=False`;
- no forbidden concrete `test_*` or `holdout_*` scoring metric columns appear,
  excluding allowed boolean embargo fields.

If child path or metadata contains `smoke`, treat it as a runner naming caveat
only if row caps are null and validation-only fields hold. Otherwise block.

Do not quote validation metric values.

## Allowed Actions

- Read local rules, environment, prior PM docs, KB continuity entries, runner
  help, and minimal runner source needed for CLI contract verification.
- Write this PM-068 spec doc.
- If this spec is GO and pre-runtime checks pass, run exactly the command above
  once.
- If runtime runs, write exactly one PM-069 closeout doc.
- Run static checks and summarize live repo state.

## Forbidden Actions

- No final test/holdout scoring, exposure, selection, baseline, metric claim, or
  authorization.
- No model selection from validation metrics.
- No validation metric quotation, ranking, comparison, or claim.
- No evidence promotion, evidence matrix, claim map, Zotero, PDF/MinerU/source
  conversion, or paper claim work.
- No notebook execution.
- No code, script, test, route, threshold, feature, label, scaler,
  decision-time, model-capacity, or split changes.
- No broad grid search, hyperparameter search, seed search, feature search,
  threshold search, or model-family search.
- No staging, commit, push, or KB writes.
- Never use `git add .`.
- Do not modify or stage `.codegraph`, notebooks, raw data, unrelated
  checkpoints, or unrelated files.

## Stop Rules

Stop if any of these occur:

- `AGENTS.md` or `docs/ENVIRONMENT.md` cannot be read.
- Git state is not inspectable.
- `HEAD != origin/master` before writing or before runtime.
- Tracked or cached diff is non-empty before writing.
- Required PM-059/060 or PM-065B/066 docs are missing.
- Source docs disagree on route locks.
- The output parent root already exists.
- The runtime command requires unsupported CLI flags.
- This PM-068 spec is missing, unsafe, ambiguous, or not GO.
- Runtime exits nonzero.
- Artifacts expose forbidden concrete test/holdout scoring columns.
- Any step would tune, select, or claim from validation metrics.

## PM+Agent Role Checklist

| Role | PM-068 Result |
| --- | --- |
| Route Explorer | GO: distinct current-pushed-HEAD freshness/protocol-observability question defined without metric dependence. |
| CLI Contract Auditor | GO: use supported flags only; metadata-only locks are verified after emission. |
| Leakage/Test Embargo Auditor | GO if PM-069 artifacts preserve validation-only rows and no concrete test/holdout scoring columns. |
| Claim-Scope Auditor | GO if PM-069 closeout quotes no validation metrics and makes no model-quality claim. |
| Artifact Integrity Auditor | Pending PM-069 runtime output. |
| Final Adversarial Reviewer | GO with duplication-risk caveat: PM-065B/066 already proved the CLI-aligned route once; PM-068 is only justified as a fresh current-pushed-HEAD artifact gate after PM-067, not as new performance evidence. |

## Next Gate

If PM-069 runs and immediate integrity checks pass, the next gate is PM-070
artifact review. If PM-069 does not run, return to parent PM for route decision.
