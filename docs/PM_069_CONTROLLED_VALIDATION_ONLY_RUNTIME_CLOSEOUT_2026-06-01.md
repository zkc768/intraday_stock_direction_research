# PM-069 Controlled Validation-Only Runtime Closeout

Date: 2026-06-01

Status: immediate integrity PASS_WITH_CAVEAT / validation-only runtime
closeout / no evidence promotion / no test access

## Scope

PM-069 executed exactly one PM-068-pre-registered validation-only runtime command
after PM-068 produced a GO decision and pre-runtime checks passed.

This closeout records command execution and immediate artifact integrity only.
It does not quote, compare, rank, or interpret validation metric values. It does
not select a model, promote evidence, authorize claims, open test/holdout,
execute notebooks, edit code, stage, commit, push, or update the KB.

## PM-068 Protocol Question

Can the current pushed HEAD `202ae1bf3d4b05c8087b4fdb97ca5fd0bd46b758` produce
one fresh, pre-registered MS-DLinear+TCN validation-only model-testing artifact
under the frozen route locks, using only supported CLI flags and emitted
metadata verification, without route drift, validation metric use, model
selection, or test/holdout scoring?

PM-068 decision: GO.

## Runtime Command

The runtime command was executed exactly once from
`E:\codex_workspace\projects\hf_stock_clf`:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --data-dir data --output-dir checkpoints\pm_069_controlled_validation_only_model_testing_20260601 --candidate A --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --model-family torch --models ms_dlinear_tcn --validation-only-report --validation-only-per-ticker --feature-view last_step --window-size 12 --tickers CSCO JPM KO MSFT WMT --seeds 42 --max-epochs 1 --batch-size 256 --split-mode calendar --train-start-ts 1998-01-02 --train-end-ts 2013-09-16 --val-start-ts 2013-09-16 --val-end-ts 2017-01-25 --holdout-start-ts 2017-01-25 --holdout-end-ts 2020-06-06
```

Runtime exit: zero.

Runtime output message:

```text
wrote result rows: 6
```

## Artifact Paths

| Item | Path | Status |
| --- | --- | --- |
| Output parent root | `checkpoints\pm_069_controlled_validation_only_model_testing_20260601` | Present |
| Child run | `checkpoints\pm_069_controlled_validation_only_model_testing_20260601\phase1b_local_no_trade_band_smoke_20260531_192407` | Present; exactly one child |
| Metadata | `metadata.json` | Present |
| Results | `results.csv` | Present |
| Manifest | `manifest.csv` | Present |

## Immediate Integrity Checks

| Check | Result |
| --- | --- |
| Output parent root exists | PASS |
| Output parent root has exactly one child run | PASS |
| Required files exist | PASS |
| `results.csv` row count | PASS: 6 rows |
| `manifest.csv` row count | PASS: 6 rows |
| Result rows use validation split only | PASS |
| `report_scope=validation_only` | PASS |
| `selection_scope=validation_only` | PASS |
| `test_metrics_embargoed=True` | PASS |
| `test_metrics_used=False` | PASS |
| Forbidden concrete `test_*` scoring columns absent except allowed embargo flags | PASS |
| Forbidden concrete `holdout_*` scoring columns absent | PASS |
| Row caps are null | PASS |

## Route-Lock Metadata Checks

| Field | Expected | Observed |
| --- | --- | --- |
| `feature_set_id` | `mentor_clean_v1` | PASS |
| `label_mode` | `no_trade_band` | PASS |
| `threshold_bps` | `5.0` | PASS |
| `threshold_source` | `fixed_pre_registered_5bps` | PASS |
| `decision_time_policy` | `post_bar_close_completed_bar` | PASS |
| `scaler_id` | `standard_pooled_train_only_v1` | PASS |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | PASS |
| train interval | `[1998-01-02, 2013-09-16)` | PASS |
| validation interval | `[2013-09-16, 2017-01-25)` | PASS |
| holdout metadata interval | `[2017-01-25, 2020-06-06)` | PASS |
| interval convention | `half_open_start_inclusive_end_exclusive` | PASS |
| `model_family` | `torch` | PASS |
| `models` | `[ms_dlinear_tcn]` | PASS |
| `tickers` | `[CSCO, JPM, KO, MSFT, WMT]` | PASS |
| `seeds` | `[42]` | PASS |
| `max_epochs` | `1` | PASS |
| `batch_size` | `256` | PASS |
| `window_size` | `12` | PASS |
| `feature_view` | `last_step` | PASS |
| `diagnostic_only` | `True` | PASS |
| `non_claim` | `True` | PASS |

## Runner Naming Caveat

Verdict: PASS_WITH_CAVEAT.

The child run path contains `smoke`, matching the prior runner naming caveat
recorded in PM-066. This closeout treats the token as a naming caveat only
because row caps are null, the command did not include `--smoke`, the route
locks hold, and validation-only/test-embargo fields hold.

## PM+Agent Audit Results

| Role | Finding | Result |
| --- | --- | --- |
| Route Explorer | PM-068 defined a current-pushed-HEAD freshness protocol question that does not depend on metric values. | PASS |
| CLI Contract Auditor | Runtime used supported CLI flags only and omitted unsupported metadata-lock flags. | PASS |
| Leakage/Test Embargo Auditor | Output rows are validation-only and no forbidden concrete test/holdout scoring columns appeared. | PASS |
| Claim-Scope Auditor | This closeout quotes no validation metric values and makes no model-quality, selection, or test-readiness claim. | PASS |
| Artifact Integrity Auditor | One child run and required files are present; row counts and route-lock fields pass immediate checks. | PASS |
| Final Adversarial Reviewer | PASS_WITH_CAVEAT: artifact is safe for PM-070 review, but duplicate-work risk remains bounded by the narrow current-pushed-HEAD protocol question and the runner naming caveat remains visible. |

## Next Gate

If parent PM accepts this immediate closeout, the next gate is PM-070 artifact
review. PM-069 does not itself authorize evidence promotion, model selection,
test/holdout access, KB sync, staging, commit, or push.
