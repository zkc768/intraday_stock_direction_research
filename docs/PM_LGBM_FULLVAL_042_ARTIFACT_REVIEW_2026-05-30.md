# PM-LGBM-FULLVAL-042 Artifact Review

## Current Live State

This review is now part of the git-visible PM route trail for the next
MS-DLinear+TCN gate. It records LightGBM PM-042 as protocol observability only:
not model evidence, not a performance claim, not a tuning signal, and not
permission to open test/holdout scoring. The artifact's calendar holdout
timestamps are split-boundary metadata only and must not be reused as scoring,
selection, or metric evidence.

Date: 2026-05-30
Status: diagnostic/protocol-observability only; non-claim
PM decision: record the artifact review without promoting validation metrics

This review records the PM-LGBM-FULLVAL-042 LightGBM validation-only diagnostic
artifact. It verifies route locks, validation-only scope, artifact disclosure
fields, and embargo indicators. It does not report model effectiveness,
trading value, paper readiness, generalization strength, or endpoint results.

## Optimized Hard-Rule Prompt

Create one PM review note for a validated LightGBM artifact and optionally sync
only KB protocol-observability targets. Keep the task non-claim. Do not edit
code, runners, tests, notebooks, model settings, thresholds, features, labels,
scalers, or checkpoints. Do not run training, notebooks, reruns, Zotero, PDF
downloads, staging, commits, or test/holdout access. Stop if the artifact is
missing, the audit disagrees with PM-042, the target doc already has unrelated
content, the KB schema cannot safely encode non-claim status, text implies
performance evidence or test readiness, or any write would leave the allowed
file list.

Allowed write targets:

- `docs/PM_LGBM_FULLVAL_042_ARTIFACT_REVIEW_2026-05-30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\hf_stock_clf_protocol_artifacts_2026_05_25.csv`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\wiki\log.md`

Claim-safety checklist:

- Status must say diagnostic/protocol-observability only and non-claim.
- Validation-only metrics must not be written as performance evidence.
- `fullval` means uncapped calendar-split validation-only artifact, not final
  performance.
- `smoke` wording in the child run id is a runner naming limitation only.
- No test/holdout authorization, evidence promotion, threshold selection,
  feature selection, hyperparameter selection, or model ranking is implied.

## Source Artifact

Artifact directory:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_lgbm_fullval_042_20260530_220434\phase1b_local_no_trade_band_smoke_20260530_220643
```

Key files:

| File | Review use |
| --- | --- |
| `metadata.json` | Route locks, scope flags, split boundaries, and caveats |
| `results.csv` | Validation-only result rows and disclosure-field schema |
| `manifest.csv` | Split, label, and window manifest schema |

Command pointer: the PM-042 command follows the locked PowerShell template in
`docs/PM_LGBM_FULLVAL_PLAN_041_2026-05-30.md`, using the calendar split
boundaries recorded below, `--model-family lightgbm`,
`--validation-only-report`, `--validation-only-per-ticker`, and
`--feature-view last_step`. The command intentionally omits `--full-run`,
`--smoke`, `--max-rows-per-ticker`, `--shuffle-train-labels`,
`--manifest-only`, LightGBM hyperparameter overrides, and route variants.

## Route Locks

| Field | Locked value |
| --- | --- |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `model_family` | `lightgbm` |
| `model_name` | `lightgbm_lgbm_classifier` |
| `feature_view` | `last_step` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

Fixed LightGBM settings are present and unchanged across the validation rows:
`objective=binary`, `n_estimators=100`, `learning_rate=0.05`,
`num_leaves=31`, `random_state=42`, `n_jobs=1`, and `verbosity=-1`.

## Calendar Split Boundaries

| Split | Start | End |
| --- | --- | --- |
| train | 1998-01-02 | 2013-09-16 |
| validation | 2013-09-16 | 2017-01-25 |
| holdout | 2017-01-25 | 2020-06-06 |

The split convention is half-open, start inclusive and end exclusive, as
recorded by the artifact metadata.

## Artifact Audit Result

PASS for protocol-observability artifact review only.

Observed artifact properties:

- `results.csv` has 6 validation rows: `pooled`, `CSCO`, `JPM`, `KO`, `MSFT`,
  and `WMT`.
- Coverage, no-trade, and class-balance disclosure fields are present.
- `results.csv` and `manifest.csv` have no concrete test or holdout scoring
  metric columns. The `test_metrics_embargoed` and `test_metrics_used` fields
  are embargo/status disclosure fields, not performance outputs.
- `max_rows_per_ticker` and `effective_max_rows_per_ticker` are null in
  metadata and blank in CSV fields.
- The child artifact directory and metadata contain internal `smoke` wording;
  this is a runner naming limitation, not evidence of a row-capped smoke.

Metric columns, where present, are diagnostic fields only and are not promoted
as performance evidence in this review or in the KB sync.

## Required Caveats

- Validation-only diagnostics are not performance evidence.
- No test or holdout authorization is granted by this artifact review.
- No threshold, hyperparameter, feature, label, scaler, or model-capacity
  change is authorized by this artifact review.
- Artifact inspection can confirm absence of concrete test/holdout scoring
  columns in emitted files, but it does not by itself prove internal
  no-materialization.

## KB Sync Decision

KB protocol-observability sync is safe only for the protocol-artifact index and
wiki log. The CSV schema can represent a non-claim artifact row if `status`,
`role`, and `key_stats` explicitly preserve diagnostic-only wording and omit
validation metric values.

Deferred targets:

- `indexes/evidence_matrix.csv`
- claim-map files
- Zotero records

These targets remain deferred because this artifact is not evidence for model
effectiveness, trading value, paper claims, generalization strength, or
endpoint results.

## Next Gate Recommendation

After parent PM review and commit, proceed to
PM-MS-DLINEAR-TCN-CANDIDATE-PLAN-043 as read-only planning. Do not start any
runtime from this sync gate.
