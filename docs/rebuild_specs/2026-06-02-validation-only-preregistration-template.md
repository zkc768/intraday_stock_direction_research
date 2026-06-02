# Validation-Only Preregistration Template

Date: 2026-06-02
Status: template only; not an approval to run.
Scope: future validation-only preregistration.

## Purpose

Use this template before any future full validation command. A completed
preregistration must be written before execution and must not copy smoke metric
values into the decision rule.

Smoke reports may confirm pipeline mechanics only. They must not select or
reject features, thresholds, model families, hyperparameters, claim wording, or
holdout/test access.

## Required Fields

| Field | Required Value Before Run |
|---|---|
| preregistration_id | |
| notebook_or_entrypoint | |
| result_scope | `validation_only` |
| tickers | |
| split_scope | train plus validation only |
| closed_holdout_policy | closed; no scoring, transform, selection, or report readout |
| feature_set | |
| label_policy | |
| threshold_policy | |
| window_policy | |
| scaler_policy | fit on train only, transform approved splits only |
| model_family | |
| hyperparameters | |
| random_seeds | |
| metrics | macro F1, balanced accuracy, accuracy auxiliary, dummy delta |
| dummy_baseline_policy | stratified dummy baseline required |
| output_path | new non-overwriting path, if artifacts are approved |
| artifact_policy | |
| claim_language | |
| stop_rules | |
| approval_source | |

## Non-Decision Inputs

Do not use these as decision rules:

- `docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md`
- `docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md`
- any bounded smoke macro F1, balanced accuracy, or LightGBM diagnostic value
- archived checkpoint, artifact, or holdout/test result

## Minimum Stop Rules

Stop before execution if any item is true:

- the preregistration leaves the model family, feature set, threshold, metric,
  or output path ambiguous;
- the command would read, transform, score, summarize, or select from closed
  holdout/test;
- the run would overwrite an existing output path;
- the notebook or entrypoint imports archived runner/helper code without a
  completed migration audit;
- preprocessing fits on validation or closed holdout/test rows;
- label horizons or input windows can cross ticker, day, or split boundaries;
- the report cannot include stratified dummy baseline and dummy deltas.

## Completion Check

Before running, verify:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short --branch
rg -n "RUN_MODEL_VALIDATION = True|RUN_TRAINING = True|holdout|test_metrics|archive|legacy_model_runner|ml_utils" notebooks\04_ian_research_memo.ipynb
```

Expected:

- dirty tree is understood and scoped;
- no unapproved archived runner/helper import is present;
- closed holdout/test appears only in boundary or prohibition language;
- run guards are enabled only by the approved validation-only task.
