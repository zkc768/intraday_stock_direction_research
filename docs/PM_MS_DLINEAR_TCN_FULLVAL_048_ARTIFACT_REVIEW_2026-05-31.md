# PM-MS-DLINEAR-TCN-FULLVAL-048 Artifact Review

Date: 2026-05-31
Status: full-input calendar-split train/validation-only diagnostic/protocol-observability only; non-claim

## Optimized Hard-Rule Prompt

Create exactly one HF artifact review doc and, only if schema-safe, update the
KB protocol-observability CSV/log. Do not edit code, runner/tests, notebooks,
evidence matrix, claim maps, Zotero, PDFs, thresholds, features, labels,
scalers, decision policy, or model capacity. Do not stage, commit, train,
rerun `local_baseline_matrix.py`, execute notebooks, move artifacts, delete
artifacts, or quote validation metrics as evidence. Stop if the active
artifact is missing, the root has anything other than one child run, artifacts
disagree with the parent PM audit, text implies performance/test readiness, or
any write would leave the allowed HF doc plus optional KB protocol CSV/log.

Claim-safety checklist:

- Validation diagnostics are protocol/runtime observability only.
- This is not model performance evidence, test authorization, or a result
  claim.
- No validation metric may select thresholds, hyperparameters, features,
  labels, scaler policy, model capacity, seed, or a winner.
- Calendar holdout timestamps are split-boundary metadata only.
- Papers remain design rationale and constraints only, not local evidence.

## Prior Gate

| Gate | Pointer |
| --- | --- |
| PM-MS-DLINEAR-TCN-FULLVAL-PLAN-COMMIT-047B | `b5df040cd0a05baa802c8980dc54e9eb593e8ac5 docs: plan MS-DLinear TCN full validation-only route` |

## Artifact Paths

| Item | Path |
| --- | --- |
| Active artifact root | `E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_ms_dlinear_tcn_fullval_048_20260531_133642` |
| Active run directory | `E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_ms_dlinear_tcn_fullval_048_20260531_133642\phase1b_local_no_trade_band_smoke_20260531_134242` |
| Metadata | `metadata.json` |
| Results | `results.csv` |
| Manifest | `manifest.csv` |

The active root contains exactly one child run directory.

## Command Summary

The PM-048 command used the project Python interpreter and the locked
MS-DLinear+TCN validation-only route:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir checkpoints\pm_ms_dlinear_tcn_fullval_048_20260531_133642 `
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

The command omitted `--full-run`, `--smoke`, and `--max-rows-per-ticker`.

## Route Locks

| Field | Artifact value |
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
| `split_mode` | `calendar` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

## Runtime Bounds

| Field | Artifact value |
| --- | --- |
| `seeds` | `[42]` |
| `max_epochs` | `1` |
| `batch_size` | `256` |
| `max_rows_per_ticker` | `None` |
| `effective_max_rows_per_ticker` | `None` |
| `tickers` | `CSCO`, `JPM`, `KO`, `MSFT`, `WMT` |

The child run name still includes `smoke` because of the runner naming
limitation. The row-cap fields above show full-input/no-cap semantics for this
calendar-split validation-only diagnostic.

## Calendar Split

Interval convention: half-open, start-inclusive/end-exclusive.

| Split boundary | Timestamp |
| --- | --- |
| `calendar_train_start_ts` | `1998-01-02T00:00:00` |
| `calendar_train_end_ts` | `2013-09-16T00:00:00` |
| `calendar_val_start_ts` | `2013-09-16T00:00:00` |
| `calendar_val_end_ts` | `2017-01-25T00:00:00` |
| `calendar_holdout_start_ts` | `2017-01-25T00:00:00` |
| `calendar_holdout_end_ts` | `2020-06-06T00:00:00` |

Calendar holdout timestamps are boundary metadata only. They are not scoring,
selection, metric evidence, or authorization to open test/holdout.

## Artifact Audit Result

| Check | Result |
| --- | --- |
| Active root child count | PASS: exactly one child run |
| Required files | PASS: `metadata.json`, `results.csv`, `manifest.csv` present |
| Result rows | PASS: 6 validation rows, pooled plus five tickers |
| Result split values | PASS: every row is `validation` |
| Ticker rows | PASS: `pooled`, `CSCO`, `JPM`, `KO`, `MSFT`, `WMT` |
| Report and selection scope | PASS: validation-only |
| Test embargo flags | PASS: `test_metrics_embargoed=True`, `test_metrics_used=False` |
| Row cap | PASS: `effective_max_rows_per_ticker=None` |
| Results forbidden columns | PASS: no concrete test/holdout scoring columns |
| Manifest forbidden columns | PASS: no concrete test/holdout scoring columns |
| Diagnostic flags | PASS: `diagnostic_only=True`, `non_claim=True` |

## Required Caveats

- This run trains on train windows and reports validation diagnostics only.
- This is not no-runtime.
- This is not test/holdout authorization.
- This is not performance evidence.
- Calendar holdout timestamps are boundary metadata only.
- The run directory includes `smoke` because of runner naming limitation; row
  cap fields show full-input/no-cap semantics.
- No threshold, hyperparameter, feature, label, scaler, model-capacity, or
  decision-policy change is authorized.
- No validation metric values are promoted here or treated as evidence.

## Next Gate Recommendation

After parent PM review/commit, proceed to a separate route-freeze or
model-adjustment planning gate. Do not start evidence promotion, tuning,
threshold selection, model-capacity search, or test/holdout access from this
sync gate.
