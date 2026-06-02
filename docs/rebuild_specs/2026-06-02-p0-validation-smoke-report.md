# P0 Validation Smoke Report

Date: 2026-06-02
Scope: validation-only smoke, not evidence-ready.

## Boundary

This smoke executed `notebooks/04_ian_research_memo.ipynb` in memory and then
used the notebook-defined functions on CSCO train/validation rows only.

No notebook output was saved. No checkpoint, artifact, raw data, source data, or
holdout/test metric was written or read. The LightGBM run was a small last-step
feature smoke, not a tuned model result.

## Command Shape

The smoke was run through the project Python:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -c "exec(open(...temporary smoke script...).read())"
```

The temporary script was deleted after the run.

## Dataset Slice

| Item | Value |
|---|---:|
| Ticker | CSCO |
| Rows used | 377522 |
| Train rows | 310905 |
| Validation rows | 66617 |
| Valid train labels | 76155 |
| Valid validation labels | 4618 |
| Train windows | 25645 |
| Validation windows | 1100 |

Only train and validation splits were used. The closed holdout boundary remained
closed and was not scored.

## Dummy Baseline Smoke

| Seed | Macro F1 | Balanced Accuracy | Validation n |
|---:|---:|---:|---:|
| 41 | 0.492645 | 0.493122 | 1100 |
| 42 | 0.484473 | 0.484921 | 1100 |
| 43 | 0.524542 | 0.524669 | 1100 |

Mean dummy macro F1: `0.500553`.

Mean dummy balanced accuracy: `0.500904`.

## LightGBM Last-Step Smoke

| Item | Value |
|---|---:|
| Train sample | 5000 |
| Validation sample | 1100 |
| Macro F1 | 0.362615 |
| Balanced accuracy | 0.497354 |

This only proves the guarded notebook functions can feed a small LightGBM
validation-only smoke. It does not support a model-performance claim.

## Non-Decision Policy

This report has scope `validation_only_smoke_not_evidence`.

Smoke metrics may be reported only as pipeline diagnostics. They must not be
used to select or reject features, thresholds, model families, hyperparameters,
claim wording, or holdout/test access.

Future full validation must be independently pre-registered. The smoke values
in this report must not be copied into that decision.

## Interpretation

The P0 notebook pipeline is executable in memory for CSCO train/validation:
data loading, causal features, no-trade labels, chronological split,
train-only scaling, per-split windows, dummy baseline, and a tiny LightGBM
last-step smoke all completed.

The LightGBM smoke is weak relative to dummy in this slice and should be treated
as a diagnostic. It is not evidence-ready and does not justify feature, model,
threshold, or claim changes.

## Next Step

Run a multi-ticker validation-only smoke after explicitly choosing the approved
sample size and whether to write a new non-overwriting output directory.
