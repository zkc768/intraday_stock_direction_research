# PM-065B MS-DLinear+TCN Validation-Only Artifact Review

Date: 2026-06-01

Status: validation-only diagnostic / protocol-observability only / non-claim / no evidence promotion

## PM+Agent Inheritance

- Parent PM authority thread: `019e7feb-8537-76a2-a247-3dbe2e2b4e5b`
- PM-065B execution thread: `019e80d5-3f32-72b2-a32f-12f27dcaa9c3`
- Original parent PM thread: `019e782e-2a33-7ff3-938d-67239f0222e8`

## Root-Cause Repair Summary

PM-065 blocked because the prior prompt used unsupported runner flags, including singular `--seed` and metadata-lock flags not exposed by the current CLI. PM-065B repaired the command contract by using the supported `--seeds 42` argument and verifying emitted metadata locks from the generated artifacts instead of trying to pass unsupported lock flags.

No runner code, notebook, test, threshold, feature, label, scaler, decision-time, or model-capacity change is authorized or reviewed here.

## Artifact Paths

- Spec: `E:\codex_workspace\projects\hf_stock_clf\docs\PM_065B_CLI_ALIGNED_VALIDATION_ONLY_MODEL_TESTING_SPEC_2026-06-01.md`
- Artifact root: `E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_065b_ms_dlinear_tcn_validation_only_20260601`
- Child run: `E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_065b_ms_dlinear_tcn_validation_only_20260601\phase1b_local_no_trade_band_smoke_20260531_184326`
- Required files present: `metadata.json`, `results.csv`, `manifest.csv`

## Runtime Scope Summary

The artifact set reflects one validation-only diagnostic route with one model family `torch`, one model `ms_dlinear_tcn`, one seed `42`, five tickers `CSCO`, `JPM`, `KO`, `MSFT`, `WMT`, `max_epochs=1`, `batch_size=256`, validation-only report/per-ticker scope, and no row cap.

This review does not quote, compare, rank, or interpret validation metric values.

## Frozen Route Locks

| Field | Expected lock | Observed audit result |
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
| `calendar_interval_convention` | `half_open_start_inclusive_end_exclusive` | PASS |
| `model_family` | `torch` | PASS |
| `models` | `[ms_dlinear_tcn]` | PASS |
| `seeds` | `[42]` | PASS |
| `tickers` | `[CSCO, JPM, KO, MSFT, WMT]` | PASS |
| `max_epochs` | `1` | PASS |
| `batch_size` | `256` | PASS |
| `window_size` | `12` | PASS |
| `feature_view` | `last_step` | PASS |
| `max_rows_per_ticker` | `None` | PASS |
| `effective_max_rows_per_ticker` | `None` | PASS |
| `report_scope` | `validation_only` | PASS |
| `selection_scope` | `validation_only` | PASS |
| `test_metrics_embargoed` | `True` | PASS |
| `test_metrics_used` | `False` | PASS |
| `non_claim` | `True` | PASS |
| `diagnostic_only` | `True` | PASS |

## Artifact Audit Results

| Check | Result |
| --- | --- |
| Artifact root exists | PASS |
| Artifact root has exactly one child run | PASS |
| Child run matches parent-observed run id | PASS |
| Required files are present | PASS |
| `results.csv` row count | PASS: 6 rows |
| `manifest.csv` row count | PASS: 6 rows |
| Result rows use validation-only report and selection scope | PASS |
| Metadata reports train-only pooled scaler scope | PASS |
| Route and split locks match frozen PM-065B locks | PASS |
| Forbidden concrete `test_*` metric columns absent, except allowed `test_metrics_embargoed` and `test_metrics_used` booleans | PASS |
| Forbidden concrete `holdout_*` metric columns absent | PASS |

## Caveats

- The child path and metadata contain `smoke` naming, but `max_rows_per_ticker` and `effective_max_rows_per_ticker` are null and validation-only fields hold. This is treated as a runner naming caveat, not as a row-capped smoke result.
- This validation-only diagnostic does not prove model quality, robustness, profitability, publishability, Ian-result success, or test readiness.
- No holdout/test authorization is granted.
- No evidence promotion, model selection, threshold change, hyperparameter change, feature change, label change, scaler change, decision-time change, or model-capacity change is authorized.
- No notebook execution, code edit, runtime rerun, staging, commit, push, or KB sync is part of this review.

## PM+Agent Role Findings

- Artifact Structure Auditor: PASS.
- Route-Lock Auditor: PASS.
- Leakage/Test Embargo Auditor: PASS.
- Claim-Scope Auditor: PASS.
- Git Scope Auditor: PASS for this write path only, pending final post-write git verification.
- Final Adversarial Reviewer: PASS_WITH_CAVEAT because the `smoke` naming is visible, but it is bounded by null row caps and validation-only route locks.

## Recommended Next PM Gate

Run a separate exact-path docs commit and necessary KB sync gate for the PM-065B spec and this PM-066 review, followed by a separate push gate if that commit gate passes. Do not perform that gate inside this artifact review.
