# PM-LGBM-VAL-SMOKE-036 Artifact Review

Date: 2026-05-30
Status: passed for protocol-observability artifact review only
PM decision: do not promote to model-performance evidence

This review records the bounded LightGBM validation-only diagnostic smoke for
the locked `mentor_clean_v1` route. It verifies artifact metadata, report
scope, coverage observability fields, and test-metric embargo flags. It does
not validate model quality, feature value, LightGBM superiority, trading value,
or paper-ready performance.

## Optimized Hard-Rule Prompt

Record PM-LGBM-VAL-SMOKE-036 as a non-claim diagnostic/protocol-observability
artifact. Do not run code, train models, execute notebooks, update Zotero, tune
thresholds or hyperparameters, open test scoring, or write model-performance
claims. Prefer an hf review doc and KB protocol-artifact/log sync; defer
`evidence_matrix.csv` if the schema cannot encode `diagnostic_only` and
`non_claim` safely.

## Source Artifact

Command:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --model-family lightgbm --validation-only-report --validation-only-per-ticker --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --smoke --seeds 42 --max-rows-per-ticker 5000 --output-dir checkpoints\pm_lgbm_val_smoke_036
```

Artifact directory:

```text
checkpoints/pm_lgbm_val_smoke_036/phase1b_local_no_trade_band_smoke_20260530_193348/
```

Artifact files:

| File | Review use |
| --- | --- |
| `metadata.json` | protocol metadata source |
| `manifest.csv` | split, label, and window manifest |
| `results.csv` | validation-only report schema and diagnostics |

## Protocol Checks

| Field | Observed value | Decision |
| --- | --- | --- |
| `feature_set_id` | `mentor_clean_v1` | pass |
| `label_mode` | `no_trade_band` | pass |
| `threshold_bps` | `5.0` | pass |
| `threshold_source` | `fixed_pre_registered_5bps` | pass |
| `decision_time_policy` | `post_bar_close_completed_bar` | pass |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | pass |
| `model_family` | `lightgbm` | pass |
| `report_scope` | `validation_only` | pass |
| `selection_scope` | `validation_only` | pass |
| `test_metrics_embargoed` | `true` | pass |
| `test_metrics_used` | `false` | pass |
| `claim_scope` | `smoke_observation_not_performance_claim` | pass |
| `diagnostic_only` / `non_claim` | `true` / `true` | pass |

## Report Observability

`results.csv` contains two rows, both marked `split=validation`.

Observed report properties:

- `report_scope` is `validation_only` for every result row.
- `selection_scope` is `validation_only` for every result row.
- `test_metrics_embargoed` is `True`.
- `test_metrics_used` is `False`.
- Validation coverage fields are present, including post-filter validation
  window counts, trade/no-trade counts, trade/no-trade rates, and validation
  class-balance counts and percentages.
- No concrete test or holdout scoring metric columns were found in
  `results.csv` or `manifest.csv` beyond the allowed embargo booleans.

Validation metric columns are diagnostic columns only. Metric values are
intentionally not promoted in this review or in the KB sync.

## Evidence Sync Decision

Allowed sync language:

- A bounded LightGBM validation-only diagnostic smoke artifact exists.
- The artifact records the locked `mentor_clean_v1 + no_trade_band + fixed 5
  bps` route.
- The artifact records post-bar-close decision timing and train-only scaler
  scope after per-ticker chronological split.
- The artifact records validation-only report/selection scope and test-metric
  embargo flags.
- The artifact exposes validation coverage/class-balance observability fields.

Forbidden sync language:

- LightGBM improved performance.
- LightGBM beats a baseline.
- `mentor_clean_v1` features work.
- Validation diagnostics prove signal.
- Fixed 5 bps is optimal.
- The artifact supports model ranking, trading value, paper claims, or
  holdout/test opening.

## Residual Caveat

The emitted artifact files show no concrete test/holdout scoring exposure, but
the files alone cannot prove that the runner avoided all internal test/holdout
materialization. That stronger no-materialization claim remains supported by
the committed code/tests from previous gates, not by this artifact by itself.

## KB Sync

This gate may sync one non-claim row to
`stock_ml_knowledge_base/indexes/hf_stock_clf_protocol_artifacts_2026_05_25.csv`
and one short `stock_ml_knowledge_base/wiki/log.md` entry.

`stock_ml_knowledge_base/indexes/evidence_matrix.csv` is deferred because its
current schema has no explicit `diagnostic_only`, `non_claim`,
`test_metrics_embargoed`, or `test_metrics_used` fields. A deferred row would
need to remain `smoke_status` / protocol-observability only and omit validation
metric values.

No claim-map update was made because no exact diagnostic/protocol-observability
claim-map target was identified that would avoid performance-evidence
ambiguity.

## Next PM Recommendation

PM-EVIDENCE-SYNC-037 should receive review/commit approval for this doc and KB
sync. A separate PM task should then decide whether to open MS-DLinear+TCN
adjustment planning or another LightGBM planning lane. This review does not
start that next task.
