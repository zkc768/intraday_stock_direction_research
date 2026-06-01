# PM-075 Final Holdout Artifact Review

Date: 2026-06-01

Status: PASS / artifact integrity accepted with bounded caveats

## Reviewed Artifact

Root:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_074_final_holdout_ms_dlinear_tcn_20260601
```

Child:

```text
phase1b_local_no_trade_band_full_20260531_201555
```

Expected files are present:

- `metadata.json`
- `manifest.csv`
- `results.csv`
- `ms_dlinear_tcn_seed_42\best.pt`
- `ms_dlinear_tcn_seed_42\last.pt`

The PM-074 root contains exactly one child run. `results.csv` contains six rows,
and `manifest.csv` contains six rows.

## Integrity Checks

| Check | Result |
|---|---|
| Exactly one timestamped child run | PASS |
| Expected artifact files present | PASS |
| `results.csv` row count | PASS; six rows |
| `manifest.csv` row count | PASS; six rows |
| Result scopes | PASS; `pooled`, `CSCO`, `JPM`, `KO`, `MSFT`, `WMT` |
| Result split | PASS; all rows are `test` |
| Model name | PASS; all rows are `ms_dlinear_tcn` |
| Seed | PASS; all rows use seed `42` |
| Concrete metrics present | PASS |
| Validation-only fields absent from result rows | PASS |
| Suspicious status | PASS; all rows are `ok` |

## Route-Lock Verification

Metadata matches the frozen route:

| Field | Observed |
|---|---|
| `run_mode` | `full` |
| `model_family` | `torch` |
| `models` | `[ms_dlinear_tcn]` |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `tickers` | `[CSCO, JPM, KO, MSFT, WMT]` |
| `seeds` | `[42]` |
| `max_epochs` | `1` |
| `batch_size` | `256` |
| `window_size` | `12` |
| `feature_view` | `last_step` |
| `split_mode` | `calendar` |
| train interval | `[1998-01-02T00:00:00, 2013-09-16T00:00:00)` |
| validation interval | `[2013-09-16T00:00:00, 2017-01-25T00:00:00)` |
| holdout-test interval | `[2017-01-25T00:00:00, 2020-06-06T00:00:00)` |

## Mode Verification

This is not a validation-only artifact.

Evidence:

- `metadata.json` has `run_mode=full`.
- `claim_scope=full_run_performance_evaluation`.
- `diagnostic_scope=full_run_candidate_evaluation`.
- `diagnostic_only=false`.
- `non_claim=false`.
- all six result rows have `split=test`.
- result rows contain concrete held-out metrics and dummy deltas.
- validation-only report fields such as `report_scope`, `selection_scope`,
  `test_metrics_embargoed`, and `test_metrics_used` are absent from the final
  result rows.

## Final Metrics

| ticker | macro F1 | balanced accuracy | delta macro F1 vs dummy | delta macro F1 vs ticker dummy | test up pct | n test windows |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 0.5090765910439399 | 0.5199419698841798 | 0.005936494650455426 | 0.005936494650455426 | 0.45519761775852735 | 7388 |
| CSCO | 0.552588176595744 | 0.5614058163886396 | 0.05249391250349544 | 0.05184517983267278 | 0.46319365798414497 | 1766 |
| JPM | 0.488508942637383 | 0.4971468909935879 | -0.005339713730280471 | -0.005462236376601259 | 0.438324727481354 | 1743 |
| KO | 0.48839508574021845 | 0.4972965854867695 | -0.0020172258275721333 | -0.017087005170071945 | 0.44745762711864406 | 885 |
| MSFT | 0.5093173440837355 | 0.522188461909108 | 0.006805483062818807 | 0.006338418897394615 | 0.46125265392781317 | 1884 |
| WMT | 0.48693956235978486 | 0.5045219638242894 | -0.01331217888324332 | -0.013413587754212086 | 0.4648648648648649 | 1110 |

## Artifact Interpretation

The pooled held-out result is weakly positive against the dummy baseline:
`delta_macro_f1_vs_dummy=0.005936494650455426`, with balanced accuracy
`0.5199419698841798`.

Ticker-level results are mixed:

- CSCO is positive versus dummy.
- MSFT is slightly positive versus dummy.
- JPM, KO, and WMT are negative versus dummy on macro-F1 delta.

This is not a strong or broad result. It is accepted as the exact final
held-out artifact for the frozen route, not as proof of general model quality.

## No-Retune Review

PASS. The artifact records one model, one seed, one feature set, one threshold,
one scaler policy, one model family, and one calendar holdout interval. No
artifact evidence indicates retuning, reselection, rerun, route fallback, seed
search, threshold change, feature change, label change, scaler change, or
validation-metric rescue after seeing holdout-test results.

The validation columns present in result rows, including `best_val_macro_f1` and
`val_delta_macro_f1_vs_dummy`, are treated only as frozen training/checkpoint
metadata. They do not authorize post-holdout selection or reinterpretation.

## Caveats

- The run metadata records the known untracked files present at runtime:
  `.codegraph/` and three notebooks. This is a reproducibility caveat, not a
  route-change or retune finding.
- Historical KB entries contain older diagnostic or legacy test metric records.
  PM-075 therefore does not claim that no test metrics ever existed anywhere in
  project history. The claim is narrower: no prior final PM-074 frozen-route
  artifact existed before this authorized run.
- Checkpoint tensors were not loaded or re-executed during review.

## PM-075 Verdict

PM-075 verdict: PASS.

The final holdout-test artifact is internally consistent, route-locked,
non-validation-only, exactly one child run, and suitable for PM-076 claim-scope
closeout. The result is weak/mixed and must be described with narrow wording.
