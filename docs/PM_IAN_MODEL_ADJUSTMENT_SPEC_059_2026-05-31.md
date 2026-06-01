# PM-IAN-MODEL-ADJUSTMENT-SPEC-059

Date: 2026-05-31

Status: model-adjustment spec / validation-only runtime may proceed only if all
gates pass / no evidence promotion / no test access

This document selects one bounded Ian-guided validation-only runtime lane after
the PM-050/051/052/053 route-control sequence and PM-056/057/058 stale route-doc
cleanup. It does not authorize evidence promotion, model selection, test/holdout
access, route semantic changes, hyperparameter search, seed search, notebook
execution, code edits, staging, commit, or push.

## Live Repo State Before Spec Creation

| Check | Live value |
| --- | --- |
| `HEAD` | `ab02262 docs: mark stale route-readiness blockers superseded` |
| `origin/master` relationship | `origin/master...HEAD = 0 2` |
| branch | `master...origin/master [ahead 2]` |
| tracked diff | empty |
| cached diff | empty |
| cached name list | empty |
| known untracked | `.codegraph/` and three notebooks |

## Frozen Route Lock Table

| Lock | Value |
| --- | --- |
| train meaning | learns/fits model weights and scaler |
| validation meaning | diagnostic/mock exam/protocol observability only |
| test/holdout meaning | final unopened exam; no scoring, exposure, selection, baseline, metric, claim, or tuning |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| train interval | `[1998-01-02, 2013-09-16)` |
| validation interval | `[2013-09-16, 2017-01-25)` |
| holdout metadata interval | `[2017-01-25, 2020-06-06)` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |
| Ian guidance and papers | design rationale and blocker checks only; not local evidence or results |

## Source Consistency Check

| Source | Finding | Status |
| --- | --- | --- |
| PM-050 | Prepared Ian-guided adjustment and required route-freeze first. | PASS |
| PM-051 | Selected `AXIS_RAW_ROUTE_CONTRACT`; did not authorize runtime or model selection. | PASS |
| PM-052 | Raw-feature contract PASS with caveats; active `mentor_clean_v1` model inputs exclude raw OHLCV, raw volume, and raw MACD-family columns. | PASS |
| PM-053 | Closed the previous non-runtime fastpath as route-control only; current PM-059 is a new parent-authorized spec/runtime gate. | PASS |
| PM-056/057/058 | Stale route-readiness text was planned, patched, accepted, and committed as cleanup/provenance only. | PASS |
| PM-042 LightGBM | Existing LightGBM full-input calendar-split validation-only route is protocol observability only. | PASS |
| PM-048 MS-DLinear+TCN | Existing MS-DLinear+TCN full-input calendar-split validation-only route is protocol observability only. | PASS |
| Protocol lock | Route locks match this spec; lower historical blocker text is now marked superseded. | PASS |
| Runner help | Existing `local_baseline_matrix.py` exposes the required validation-only, calendar-split, LightGBM, torch, model, ticker, seed, and split-boundary flags. | PASS |

## Candidate Lane Table

| lane_id | model_family | allowed_adjustment | existing_entrypoint? | requires_code_edit? | requires_route_semantic_change? | leakage_risk | runtime_risk | selection_risk | why_allowed_or_blocked | recommended? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `LGBM_FIRST_VALIDATION_ONLY` | `lightgbm` | One fixed-route calendar-split validation-only diagnostic using the existing LightGBM entrypoint, fixed 5 bps no-trade band, `mentor_clean_v1`, `last_step` feature view, five tickers, seed 42, and no hyperparameter overrides. | Yes: `scripts/phase1b_local/local_baseline_matrix.py --model-family lightgbm --validation-only-report`. | No | No | Low if calendar split, train-only scaler, post-bar-close features, and validation-only output checks hold. | Low-medium: one fixed tabular command; no epoch/batch training budget. | Medium unless framed strictly as protocol observability; no validation metric values may choose the lane or any next action. | Allowed because it is the shortest existing runtime surface and is operationally simpler than neural training. This is not a claim that LightGBM is better. | Yes |
| `MS_DLINEAR_TCN_FIRST_VALIDATION_ONLY` | `torch` / `ms_dlinear_tcn` | One fixed-route calendar-split validation-only diagnostic using the existing MS-DLinear+TCN entrypoint, max-epochs 1, batch-size 256, five tickers, seed 42. | Yes: `scripts/phase1b_local/local_baseline_matrix.py --model-family torch --models ms_dlinear_tcn --validation-only-report`. | No | No | Low if PM-048 locks are preserved. | Medium: neural lane has epoch/batch runtime knobs and a larger artifact surface. | Medium-high unless framed strictly as protocol observability and not architecture/capacity tuning. | Safe in principle, but broader than the LightGBM-first lane for this single runtime gate. | No |
| `BLOCKED_NEEDS_SEPARATE_IMPLEMENTATION_OR_PROTOCOL_GATE` | none | Stop after spec if no single exact command is safe. | Not applicable | Not applicable | Not applicable | Not applicable | Not applicable | Low | Not selected because a safe existing LightGBM validation-only command is identifiable without code edits or route changes. | No |

## Selected Lane

Selected lane: `LGBM_FIRST_VALIDATION_ONLY`.

Rationale: this lane has the shortest and simplest existing runtime surface for
one controlled validation-only diagnostic. The choice is based on route safety,
operational simplicity, and Ian/design rationale around checking a tabular
baseline under the cleaned feature contract. It is not based on validation
metric values, model superiority, profitability, robustness, publishability,
Ian-result success, or test readiness.

## Stage B Exact Runtime Command

Precondition: the output parent directory must not already exist.

Expected output parent:

```text
checkpoints\pm_ian_model_adjust_spec_059_lgbm_20260531
```

Exact command to run from `E:\codex_workspace\projects\hf_stock_clf`:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir checkpoints\pm_ian_model_adjust_spec_059_lgbm_20260531 `
  --candidate A `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --model-family lightgbm `
  --validation-only-report `
  --validation-only-per-ticker `
  --feature-view last_step `
  --window-size 12 `
  --tickers CSCO JPM KO MSFT WMT `
  --seeds 42 `
  --split-mode calendar `
  --train-start-ts 1998-01-02 `
  --train-end-ts 2013-09-16 `
  --val-start-ts 2013-09-16 `
  --val-end-ts 2017-01-25 `
  --holdout-start-ts 2017-01-25 `
  --holdout-end-ts 2020-06-06
```

The command intentionally omits `--full-run`, `--smoke`,
`--max-rows-per-ticker`, `--shuffle-train-labels`, `--manifest-only`, any
LightGBM hyperparameter override, threshold search, feature variant, scaler
variant, decision-time variant, seed search, grid search, and test/holdout
access.

Expected runner naming caveat: the LightGBM route cannot use `--full-run`.
PM-041 and PM-042 record that this existing runner path may still emit
`run_mode=smoke` and a child run id containing `smoke` even when the command
omits `--smoke` and uses calendar boundaries. In PM-059 this is acceptable only
as a runner naming limitation if `max_rows_per_ticker` and
`effective_max_rows_per_ticker` are null/`None`, the command omitted `--smoke`,
and all report/selection/test embargo fields remain validation-only and
non-claim.

## Stage B Stop Rules

Stop before runtime if:

- the output parent directory exists;
- tracked or cached diffs contain unexpected files beyond this spec;
- the command would edit code, scripts, tests, notebooks, data, or route docs;
- the command omits any frozen route lock or calendar boundary;
- the command includes `--full-run`, `--smoke`, `--max-rows-per-ticker`,
  `--shuffle-train-labels`, `--manifest-only`, a hyperparameter override, a
  seed search, a broad grid, a second model family, or a notebook;
- the lane justification depends on validation metric values.

Stop after runtime if:

- the command exits nonzero;
- output creates more than one timestamped child run unexpectedly;
- required artifact files are missing;
- any artifact lacks validation-only report/selection scope;
- any result row is not validation-only;
- any concrete `test_*` or `holdout_*` scoring metric or selection column is
  emitted;
- any artifact changes feature, label, threshold, scaler, decision-time, split,
  model-capacity, hyperparameter, seed, or route semantics;
- validation diagnostics are framed as evidence, performance, selection, or
  test readiness.

Do not treat a `smoke` token in the run id or metadata as a blocker by itself if
the command omitted `--smoke` and the artifact confirms uncapped calendar-split
validation-only semantics. Treat it as a named runner limitation that must be
reported.

## Stage C Immediate Integrity Checks

Inspect only enough generated artifacts to verify:

- output parent exists and has exactly one child run directory;
- `metadata.json`, `results.csv`, and `manifest.csv` exist;
- route fields match this spec;
- `report_scope=validation_only`;
- `selection_scope=validation_only`;
- `test_metrics_embargoed=True`;
- `test_metrics_used=False`;
- split ranges match this spec;
- result rows are validation-only;
- no concrete `test_*` or `holdout_*` scoring metric columns are present;
- diagnostic/non-claim flags or wording are present if emitted.
- if `run_mode=smoke` or a `smoke` token appears, verify this is the known
  LightGBM runner naming limitation and not a row cap or explicit smoke flag.

Do not perform a full artifact review, evidence sync, claim-map update, KB sync,
commit, push, or follow-up runtime in PM-059. If runtime succeeds, the next
parent-PM gate should be artifact review.

## Claim-Safety Language

Any PM-059 runtime is a validation-only protocol diagnostic/mock-exam
observability artifact. It is not local evidence, not model performance proof,
not model selection, not feature selection, not threshold selection, not tuning,
not a paper claim, not an Ian-result success claim, not robustness,
profitability, publishability, or test-readiness evidence, and not authorization
to open test/holdout.
