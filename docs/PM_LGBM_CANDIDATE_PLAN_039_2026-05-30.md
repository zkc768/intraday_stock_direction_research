# PM-LGBM-CANDIDATE-PLAN-039

Date: 2026-05-30
Status: planning artifact only
PM decision: prepare a future LightGBM validation-only candidate execution gate;
do not run it in this gate

This plan defines the smallest safe LightGBM candidate lane after the
validation-only coverage-reporting and diagnostic-smoke review gates. It does
not edit code, train a model, execute notebooks, run a smoke, update
`evidence_matrix.csv`, update claim maps, or create model-performance evidence.

## Prior Gate Status

| Gate | Status | Planning implication |
| --- | --- | --- |
| PM-COVERAGE-REPORTING-034 | committed as coverage/reporting hardening | Future validation-only rows must keep post-filter coverage and class-balance diagnostics. |
| PM-LGBM-ADJUST-035 | review-only outcome | Current LightGBM route was considered locked enough for a diagnostic smoke. |
| PM-LGBM-VAL-SMOKE-036 | diagnostic smoke completed and reviewed | Use as protocol-observability only; do not treat validation metrics as performance evidence. |
| PM-EVIDENCE-SYNC-037 | non-claim sync completed | Evidence matrix and claim map remained deferred. |
| PM-SYNC-COMMIT-038 | committed as `c600f01` | The 036 HF review doc is now tracked; KB sync remains non-git state. |

## Split Semantics

Use the exam analogy in every future execution prompt:

- Train = learn and fit the model and scaler.
- Validation = select or check the candidate under the locked protocol.
- Test = final unopened exam.

For this lane, LightGBM may fit on train data and report validation-only
diagnostics. It must not score, expose, select from, or materialize test/holdout
data in validation-only paths beyond the already-approved embargo booleans and
route metadata.

## Locked Route

| Field | Locked value |
| --- | --- |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

These fields are not adjustable in PM-LGBM-CANDIDATE-SMOKE-040. Any change to
threshold, feature set, label semantics, decision-time policy, or scaler policy
requires a separate PM protocol gate before execution.

## Candidate Policy

The candidate surface is the existing fixed LightGBM route, not a search space.

Pre-registered candidate settings:

| Setting | Value |
| --- | --- |
| `model_family` | `lightgbm` |
| `model_name` | `lightgbm_lgbm_classifier` |
| `objective` | `binary` |
| `n_estimators` | `100` |
| `learning_rate` | `0.05` |
| `num_leaves` | `31` |
| `random_state` | `42` |
| `n_jobs` | `1` |
| `verbosity` | `-1` |

Allowed:

- Use the existing LightGBM validation-only runner path.
- Fit LightGBM on training windows only.
- Emit validation-only rows and coverage/class-balance diagnostics.
- Emit candidate metadata showing the fixed settings above.
- Use validation diagnostics for protocol checking inside the future gate.

Forbidden:

- No hyperparameter grid, search, tuning loop, or validation-driven parameter
  selection.
- No threshold selection or threshold change.
- No feature-set change.
- No label semantic change.
- No scaler-policy change.
- No test/holdout scoring, metric exposure, or selection.
- No local performance, effectiveness, profitability, publishability, or
  paper-evidence claim.

## Runner Feasibility

Read-only runner inspection found a feasible future command surface:

- `--model-family lightgbm` is a supported model family.
- LightGBM requires `--validation-only-report`.
- LightGBM rejects `--full-run`.
- LightGBM requires `--feature-set mentor_clean_v1`.
- LightGBM requires `--label-mode no_trade_band`.
- LightGBM requires explicit `--threshold-bps 5.0`.
- Result rows carry `split=validation` and validation-only report fields.
- Fixed LightGBM parameters are emitted through result metadata.

This inspection did not execute the runner.

## Paper-Use Policy

Papers may enter this lane only as design rationale, constraints, and blockers.
They cannot become local evidence or results.

Allowed paper uses:

- Explain why a tabular LightGBM candidate is worth a bounded validation-only
  check after feature cleaning.
- Motivate route constraints such as feature cleaning, trailing/normalized
  inputs, coverage disclosure, and leakage-safe validation.
- Identify blockers that require future source, table, code, or protocol audit.

Forbidden paper uses:

- Do not choose thresholds from paper values.
- Do not choose hyperparameters from paper results inside this gate.
- Do not treat paper results as local `hf_stock_clf` results.
- Do not use papers to open holdout/test scoring.
- Do not promote raw-material-only or parsed-unverified sources to evidence.

Current KB route maps remain design-only. They explicitly block
`evidence_matrix.csv` and wiki promotion until later Gate 3 claim audits create
claim-level anchors.

Design constraints carried from the KB route maps:

| Area | Constraint for the future LightGBM lane |
| --- | --- |
| Feature route | Main candidate must use `mentor_clean_v1`; raw OHLCV, raw volume, and raw MACD-family features stay out of the main run. `technical_v1` is control-only. |
| Timing | Features using current-bar close, high, low, or volume are allowed only after the completed 5-minute bar. Any pre-close route is blocked. |
| Scaling | Scaler must be fit on train only after chronological per-ticker split. |
| Rolling features | Rolling volatility, volume denominators, Bollinger, RSI, and MACD normalization must be trailing-only and point-in-time. |
| Threshold | Papers cannot choose or tune the no-trade threshold; fixed 5 bps remains locked. |
| No-trade layer | Coverage, no-trade rate, and class-balance disclosure are required diagnostics; literature does not prove local no-trade benefit. |
| Costs | Any trading-usefulness wording remains blocked unless turnover, cost/slippage, gross/net, and false-positive trade-rate reporting are approved later. |

Useful paper-route roles remain narrow:

| Paper or route | Allowed use | Blocker |
| --- | --- | --- |
| `1912.07165` | Feature-recipe rationale for 5-minute returns, liquidity, and indicators | No exact results or tradability claims without table/protocol audit. |
| `1907.09452` | Rolling normalization and imbalanced-metric guardrail | LOB-to-bar transfer and table/code audit remain blocked. |
| `Geifman_2017` | Selective-risk and coverage vocabulary | Non-finance transfer only; no effectiveness claim. |
| `Noh_2021` | No-trade protocol design after provenance repair | Key/title/authors mismatch and no backtest superiority claim. |
| `2605.23959` | Decision-time leakage checklist | Checklist only; no local alpha/performance transfer. |
| `1807.02787` / `2004.10178` | Cost, rolling split, and return-feature guardrails | No PnL or tradability claim. |
| `2104.05413` | Indicator ablation caution | Cannot say indicators help locally. |
| `2502.18177` | Shifted normalized volume and time-of-day inspiration | Shift and denominator unaudited. |
| `2501.07580` / `2107.11972` | LightGBM framing or rationale only | Unreviewed or transfer-risk; not implementation readiness. |

## Future Execution Gate

Recommended next task:

```text
PM-LGBM-CANDIDATE-SMOKE-040 -- LightGBM train/validation-only candidate smoke

Task type: bounded validation-only execution / artifact audit.
Goal: run one pre-registered LightGBM candidate under the locked
mentor_clean_v1 + no_trade_band + fixed 5 bps protocol, then audit artifacts as
diagnostic/protocol observability only.

Required route:
- feature_set_id=mentor_clean_v1
- label_mode=no_trade_band
- threshold_bps=5.0
- threshold_source=fixed_pre_registered_5bps
- decision_time_policy=post_bar_close_completed_bar
- scaler_fit_scope=pooled_train_after_per_ticker_chronological_split
- model_family=lightgbm
- objective=binary
- n_estimators=100
- learning_rate=0.05
- num_leaves=31
- random_state=42
- n_jobs=1
- verbosity=-1
- report_scope=validation_only
- selection_scope=validation_only
- test_metrics_embargoed=True
- test_metrics_used=False

Allowed:
- Read project rules and this plan.
- Run exactly one bounded validation-only LightGBM command if the output
  directory is fresh.
- Inspect metadata.json, manifest.csv, and results.csv.
- Report route locks, coverage fields, and absence of concrete test/holdout
  metric exposure.

Forbidden:
- No code edits.
- No runner/test edits.
- No notebook execution.
- No full-run.
- No hyperparameter grid/search.
- No threshold, feature, label, decision-time, or scaler change.
- No evidence_matrix, wiki, claim-map, Zotero, or paper-result update.
- No model-performance, profitability, or publishability claim.

Stop rules:
- Stop on any test/holdout scoring or exposure.
- Stop on threshold/feature/label/scaler drift.
- Stop on hyperparameter search not pre-registered here.
- Stop if validation metrics are written as performance claims.
- Stop if output path would overwrite an existing artifact.
- Stop if the command would require notebook execution or a full run.

Final report:
- Exact command and exit status.
- Generated artifact path and files inspected.
- PASS/BLOCK checklist for locked route fields.
- Coverage/no-trade/class-balance diagnostics present or missing.
- No-test/holdout exposure audit.
- Residual caveats, especially whether artifacts alone prove internal
  no-materialization.
```

## Success Definition For 040

Success means the future candidate emits a complete validation-only diagnostic
artifact under the locked route with coverage/class-balance disclosure and no
test/holdout metric exposure. It does not mean LightGBM is effective, better,
profitable, publishable, or paper-evidence-backed.

## Stop Conditions For This Plan

This planning gate should stop rather than broaden if:

- a useful next step requires code edits;
- candidate selection would require validation/test metric peeking in this
  planning gate;
- paper sources would need to be used as evidence claims;
- a safe doc target becomes ambiguous;
- the next task would require threshold, feature, label, decision-time, or
  scaler-policy changes.
