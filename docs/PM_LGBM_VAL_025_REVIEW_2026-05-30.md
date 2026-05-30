# PM-LGBM-VAL-025 Review

Date: 2026-05-30

## Scope

This note records a bounded LightGBM validation-only route under the Ian mentor-clean model-adjustment lane.

This is protocol and artifact evidence only. It is not model-performance evidence, and it must not be written to the evidence matrix, wiki, Zotero, or paper-claim materials.

## Inputs

- Latest planning commit before the run: `9d7109d docs: plan mentor clean model adjustment route`.
- Route lock: `mentor_clean_v1`, `no_trade_band`, fixed pre-registered `5.0` bps threshold.
- Tickers: `CSCO JPM KO MSFT WMT`.
- Row cap chosen before execution: `--max-rows-per-ticker 5000`.
- Output root: `checkpoints\pm_lgbm_val_025_20260530_001`.

## Command

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --model-family lightgbm --validation-only-report --validation-only-per-ticker --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --tickers CSCO JPM KO MSFT WMT --max-rows-per-ticker 5000 --output-dir checkpoints\pm_lgbm_val_025_20260530_001
```

Runtime result: command exited successfully and wrote 6 result rows.

## Artifacts Reviewed

- `checkpoints\pm_lgbm_val_025_20260530_001\phase1b_local_no_trade_band_smoke_20260530_155410\metadata.json`
- `checkpoints\pm_lgbm_val_025_20260530_001\phase1b_local_no_trade_band_smoke_20260530_155410\manifest.csv`
- `checkpoints\pm_lgbm_val_025_20260530_001\phase1b_local_no_trade_band_smoke_20260530_155410\results.csv`

## Protocol Checklist

| Check | Status |
| --- | --- |
| `report_scope == validation_only` | PASS |
| `selection_scope == validation_only` | PASS |
| `test_metrics_embargoed == True` | PASS |
| `test_metrics_used == False` | PASS |
| `feature_set_id == mentor_clean_v1` | PASS |
| `label_mode == no_trade_band` | PASS |
| `threshold_bps == 5.0` | PASS |
| `threshold_source == fixed_pre_registered_5bps` | PASS |
| `decision_time_policy == post_bar_close_completed_bar` | PASS |
| `scaler_id == standard_pooled_train_only_v1` | PASS |
| `scaler_fit_scope == pooled_train_after_per_ticker_chronological_split` | PASS |
| `diagnostic_only == True` | PASS |
| `non_claim == True` | PASS |
| `manifest.csv` has no forbidden holdout/test exposure columns | PASS |
| `results.csv` has no forbidden holdout/test exposure columns beyond allowed embargo booleans | PASS |

## Adversarial Review Note

The read-only adversarial reviewer found no blocker for the route if the PM policy means no test scoring, exposure, or selection. It flagged one caveat: the current runner still materializes the holdout/test split internally during data preparation, although this route does not score or expose test metrics. Treat this as acceptable under the current validation-only artifact gate, but stop if a future gate requires no holdout/test materialization at all.

## Recommendation

PM-LGBM-VAL-025 can be closed as a validation-only protocol/artifact smoke for the LightGBM lane. Do not promote any validation diagnostic columns to model-performance evidence. Do not update evidence matrix, wiki, Zotero, or paper claims from this smoke.

Next minimal approval: authorize a separate PM closeout commit for this docs-only review note, or open the next bounded planning gate for the MS-DLinear+TCN lane without running it.
