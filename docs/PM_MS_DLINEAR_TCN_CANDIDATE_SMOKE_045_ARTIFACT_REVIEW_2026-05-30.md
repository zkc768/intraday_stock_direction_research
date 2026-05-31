# PM-MS-DLINEAR-TCN-CANDIDATE-SMOKE-045 Artifact Review

Date: 2026-05-30
Status: tiny train/validation-only diagnostic/protocol-observability only; non-claim
PM decision: record the artifact review without promoting validation metrics

This review records the PM-MS-DLINEAR-TCN-CANDIDATE-SMOKE-045 tiny
MS-DLinear+TCN validation-only diagnostic artifact. It verifies route locks,
validation-only scope, artifact disclosure fields, embargo indicators, and the
duplicate-run quarantine decision. It does not report model effectiveness,
robustness, trading value, profitability, publishability, paper readiness,
generalization strength, or endpoint results.

## Optimized Hard-Rule Prompt

Create one HF artifact-review note and optionally sync only KB
protocol-observability targets. Keep the task non-claim. Do not edit code,
runners, tests, notebooks, model settings, thresholds, features, labels,
scalers, checkpoints, evidence matrices, claim maps, or Zotero. Do not run
training, notebooks, runtime reruns, PDF downloads, staging, commits, or
test/holdout access. Stop if the artifact is missing, the active root has more
than one child run, the quarantine path is missing or inside the active root,
the audit disagrees with PM-045, the target doc already exists with unrelated
content, KB schema cannot safely encode non-claim status, text implies
performance evidence or test readiness, or any write would leave the allowed
file list.

Allowed write targets:

- `docs/PM_MS_DLINEAR_TCN_CANDIDATE_SMOKE_045_ARTIFACT_REVIEW_2026-05-30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\hf_stock_clf_protocol_artifacts_2026_05_25.csv`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\wiki\log.md`

Claim-safety checklist:

- Status must say tiny train/validation-only diagnostic, protocol-observability
  only, and non-claim.
- Validation metrics must not be written as performance evidence.
- The artifact must not imply MS-DLinear+TCN works, improves results, is robust,
  is profitable, is publishable, or is ready for test/holdout.
- No threshold, feature, label, scaler, hyperparameter, model-capacity, or
  decision-policy change is authorized.
- No test/holdout authorization, evidence promotion, paper claim, threshold
  selection, feature selection, hyperparameter selection, or model ranking is
  implied.

## Prior Gate

Stage 1 doc-reconcile commit:

```text
0aca665 docs: reconcile MS-DLinear TCN runtime route
```

The committed doc-reconcile gate made the MS-DLinear+TCN route-state packet
git-visible before the tiny validation-only smoke was inspected.

## Source Artifact

Active artifact root:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_ms_dlinear_tcn_candidate_smoke_045
```

Active artifact directory:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_ms_dlinear_tcn_candidate_smoke_045\phase1b_local_no_trade_band_smoke_20260530_225414
```

Key files:

| File | Review use |
| --- | --- |
| `metadata.json` | Route locks, scope flags, diagnostic/non-claim flags, and command metadata |
| `results.csv` | Validation-only result rows and disclosure-field schema |
| `manifest.csv` | Split, label, and window manifest schema |

Duplicate-run quarantine:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_ms_dlinear_tcn_candidate_smoke_045_quarantine_duplicate_runs\phase1b_local_no_trade_band_smoke_20260530_225340
```

Reason: an earlier duplicate child run was preserved outside the active root to
restore exactly-one active artifact semantics without deleting evidence.

## Command Pointer

PM-045 ran one bounded tiny torch command using the locked route:

```text
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --model-family torch --models ms_dlinear_tcn --validation-only-report --validation-only-per-ticker --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --smoke --max-epochs 1 --seeds 42 --batch-size 256 --max-rows-per-ticker 5000 --output-dir checkpoints\pm_ms_dlinear_tcn_candidate_smoke_045
```

This command trains the tiny model on train windows and reports validation
diagnostics only. It is not a no-runtime command, not a full validation-scale
run, and not a full-run/test gate.

## Route Locks

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

## Artifact Audit Result

PASS for protocol-observability artifact review only.

Observed artifact properties:

- The active artifact root has exactly one child run directory after duplicate
  quarantine.
- The quarantined duplicate run was preserved outside the active root, not
  deleted.
- `results.csv` has 2 validation rows: `pooled` and `CSCO`.
- `metadata.json` records validation-only report/selection scope.
- `metadata.json` records `diagnostic_only=True` and `non_claim=True`.
- `metadata.json` records `test_metrics_embargoed=True` and
  `test_metrics_used=False`.
- `results.csv` rows use `split=validation`.
- `results.csv` and `manifest.csv` have no concrete test or holdout scoring
  metric columns. The `test_metrics_embargoed` and `test_metrics_used` fields
  are embargo/status disclosure fields, not performance outputs.
- Coverage, no-trade, and class-balance disclosure fields are present as
  validation diagnostic fields.

Metric columns, where present, are diagnostic fields only and are not promoted
as performance evidence in this review or in the KB sync.

## Required Caveats

- This is a tiny train-on-train / validation-only runtime diagnostic.
- It is not no-runtime.
- It is not full validation scale.
- It is not performance evidence.
- It does not prove MS-DLinear+TCN is effective, robust, profitable,
  publishable, or ready for test/holdout.
- It does not authorize test/holdout scoring, exposure, selection, or access.
- It does not authorize threshold, hyperparameter, feature, label, scaler,
  model-capacity, or decision-policy changes.
- It does not authorize validation-driven model selection or candidate
  superiority claims.
- The duplicate run was quarantined outside the active root, not deleted.

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
effectiveness, trading value, paper claims, generalization strength, or endpoint
results.

## Next Gate Recommendation

After parent PM review and commit, proceed to
PM-MS-DLINEAR-TCN-FULLVAL-PLAN-047 as a planning gate. Do not start full
validation runtime from this sync gate.
