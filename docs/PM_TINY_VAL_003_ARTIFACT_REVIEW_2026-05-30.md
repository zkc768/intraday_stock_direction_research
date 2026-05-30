# PM-TINY-VAL-003 Artifact Review

Date: 2026-05-30
Status: passed for protocol-artifact evidence only
PM decision: do not promote to performance evidence

This review closes the minimal validation-only smoke for Ian's
`mentor_clean_v1` route. It verifies that the existing runner can produce a
bounded report under the locked no-trade-band protocol. It does not validate
model quality, feature value, LightGBM, or MS-DLinear+TCN.

## Optimized Hard-Rule Prompt

Task:

- Run one tiny validation-only smoke for `mentor_clean_v1`.
- Confirm fixed 5 bps no-trade labeling.
- Confirm train-only scaler metadata.
- Confirm the report is validation-only and test metrics remain embargoed.

Allowed:

- Use the existing `sklearn_logreg` validation-only path.
- Write only a new checkpoint/output directory under `checkpoints/`.
- Review the generated artifact files.
- Record this docs-only PM artifact review.

Forbidden:

- No `ml_utils` edits.
- No runner edits.
- No notebook execution.
- No torch training.
- No LightGBM claim.
- No MS-DLinear+TCN claim.
- No evidence-matrix, wiki, Zotero, or claim-map update from this artifact.
- No threshold, feature, or model choice based on test data.

Stop rules:

- Stop if the runner cannot emit `report_scope=validation_only`.
- Stop if `threshold_source` is not fixed/pre-registered.
- Stop if scaler metadata does not state train-only fit after chronological
  per-ticker split.
- Stop if test metrics are read or used for model selection.
- Stop if the artifact is missing protocol fields needed for a non-claim
  review.

## Source Artifact

Command:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --model-family sklearn_logreg --validation-only-report --validation-only-per-ticker --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --tickers CSCO JPM KO MSFT WMT --max-rows-per-ticker 5000 --output-dir checkpoints\mentor_clean_v1_tiny_validation_2026-05-30
```

Artifact directory:

```text
checkpoints/mentor_clean_v1_tiny_validation_2026-05-30/phase1b_local_no_trade_band_smoke_20260530_063513/
```

Artifact files:

| File | Rows | Review use |
| --- | ---: | --- |
| `metadata.json` | n/a | protocol metadata source |
| `manifest.csv` | 6 | split, label, window, and coverage manifest |
| `results.csv` | 6 | validation-only report schema and observability |

The manifest and results files record git commit hash
`2fdf04149d313e9a43b2b5a996c0a7cf510b2bad`.

## Protocol Checks

| Field | Observed value | Decision |
| --- | --- | --- |
| `feature_set_id` | `mentor_clean_v1` | pass |
| `label_mode` | `no_trade_band` | pass |
| `threshold_bps` | `5.0` | pass |
| `threshold_source` | `fixed_pre_registered_5bps` | pass |
| `decision_time_policy` | `post_bar_close_completed_bar` | pass |
| `scaler_id` | `standard_pooled_train_only_v1` | pass |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | pass |
| `model_family` | `sklearn_logreg` | pass |
| `report_scope` | `validation_only` | pass |
| `selection_scope` | `validation_only` | pass |
| `test_metrics_embargoed` | `true` | pass |
| `test_metrics_used` | `false` | pass |
| `claim_scope` | `smoke_observation_not_performance_claim` | pass |
| `diagnostic_only` / `non_claim` | `true` / `true` | pass |

## Split Discipline

The artifact follows the project split rule:

```text
training set -> validation set -> test set
```

The training set is for fitting the model and scaler. The validation set is for
protocol smoke review and future hyperparameter selection if separately
approved. The test set is final holdout only and must not be used to choose a
threshold, feature set, model, or checkpoint.

Per-ticker manifest chronology:

| Ticker | train end <= val start | val end <= test start | Train windows | Val windows | Test windows |
| --- | --- | --- | ---: | ---: | ---: |
| CSCO | true | true | 342 | 86 | 44 |
| JPM | true | true | 266 | 47 | 37 |
| KO | true | true | 139 | 12 | 53 |
| MSFT | true | true | 201 | 48 | 48 |
| WMT | true | true | 319 | 35 | 41 |

The pooled row is an aggregate across tickers, so its min/max timestamps overlap
and must not be used as the chronology check. The per-ticker rows are the
chronological split evidence.

## Report Observability

`results.csv` contains six rows: pooled plus CSCO, JPM, KO, MSFT, and WMT.

Observed report properties:

- `report_scope` is `validation_only` for every row.
- `selection_scope` is `validation_only` for every row.
- `test_metrics_used` is `False` for every row.
- The only test-like fields are `test_metrics_embargoed` and
  `test_metrics_used`; no concrete test performance metric columns were found.
- Validation macro F1, validation balanced accuracy, validation delta versus
  dummy, and validation dummy-stratified macro F1 fields are present.
- Window counts and validation class-balance observability are present through
  manifest/results fields such as `n_train_windows`, `n_val_windows`,
  retained-label counts, and `val_up_pct`.

Gap:

- The report does not expose explicit `val_support_class_0` and
  `val_support_class_1` fields. Class support can be inferred only from
  retained-label counts and `val_up_pct`. This is acceptable for a PM tiny
  smoke review, but it blocks any stronger evidence sync or performance claim
  until the schema is reviewed or extended in a separate approved task.

## Evidence Sync Decision

Allowed sync language:

- A bounded validation-only `mentor_clean_v1` smoke artifact exists.
- The artifact records fixed 5 bps no-trade labeling.
- The artifact records post-bar-close decision timing.
- The artifact records train-only pooled scaler fit after chronological
  per-ticker split.
- The artifact records validation-only scope and test-metric embargo.

Forbidden sync language:

- `mentor_clean_v1` improves performance.
- Cleaned features work.
- LightGBM is implemented or validated.
- MS-DLinear+TCN is implemented or validated by this smoke.
- Validation metrics prove signal.
- Fixed 5 bps is optimal.
- Test performance supports the route.

PM-EVIDENCE-SYNC status:

- Do not update `evidence_matrix.csv` from this artifact.
- Do not update `wiki/*` from this artifact.
- Do not update the paper claim map from this artifact.
- The knowledge-base protocol-artifact index may carry one non-claim
  artifact-review row because its schema tracks local pipeline artifacts rather
  than paper/model claims.
- The synced row must omit validation scores and point back to this review doc
  as its non-wiki `sync_target`.

## Next Gate

The next minimum PM task is:

```text
PM-EVIDENCE-SYNC-005 -- protocol-artifact index only

Allowed:
- Append one protocol-artifact index row that points to the latest tiny
  validation artifact.
- Record protocol fields, artifact path, row counts, and non-claim status.

Forbidden:
- No evidence_matrix.csv update.
- No wiki update.
- No paper claim-map update.
- No validation-score performance claim.
- No test metric use.
```

PM-EVIDENCE-SYNC-005 result:

- `stock_ml_knowledge_base/indexes/hf_stock_clf_protocol_artifacts_2026_05_25.csv`
  now includes one `validation_artifact_summary` row for
  `mentor_clean_v1_tiny_validation_smoke_20260530`.
- `status=artifact_review_PASS` means the artifact review passed for protocol
  observability only.
- `key_stats` intentionally omits validation scores and records
  `no_performance_claim`.
