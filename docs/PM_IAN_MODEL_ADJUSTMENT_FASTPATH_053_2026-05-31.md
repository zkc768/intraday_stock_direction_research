# PM-IAN-MODEL-ADJUSTMENT-FASTPATH-053

Date: 2026-05-31

Status: planning/control closeout and next-route prompt generator only / no runtime intended

Parent acceptance context: PM-052 was accepted by the parent PM and committed in Stage 1 of this sequence as `59c85d4 docs: record mentor clean raw-route contract audit`. Parent clarification says the fastest safe non-runtime path is preferred, expected to be closeout/synthesis if PM-052 raw-contract PASS and existing PM-042/048 artifacts are sufficient. The parent clarification does not mean choosing the Stage 2 table's LightGBM option by label.

Process note: PM-053 did not authorize runtime, training, notebook execution, code edits, evidence promotion, or test/holdout access. One delegated Feature Contract Auditor unexpectedly ran a targeted pytest contract check while performing read-only review. That was outside the intended no-runtime scope, did not alter files, did not run training/notebooks/`local_baseline_matrix.py`, and is not used here as a selection signal or authorization.

## Frozen Route Table

| Lock | Value |
|---|---|
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
| calendar split | train `[1998-01-02, 2013-09-16)`, validation `[2013-09-16, 2017-01-25)`, holdout metadata `[2017-01-25, 2020-06-06)` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |
| Ian guidance and papers | design rationale and blocker checks only, not local evidence or results |

## Current Artifact/Gate Map

| Gate | Current state | PM-053 use |
|---|---|---|
| PM-042 LightGBM | Full-input calendar-split validation-only diagnostic artifact review recorded as protocol-observability only and non-claim. | Sufficient as LightGBM route observability; not model evidence, not selection, not tuning permission. |
| PM-048 MS-DLinear+TCN | Full-input calendar-split train/validation-only diagnostic artifact review recorded as protocol-observability only and non-claim. | Sufficient as MS-DLinear+TCN route observability; not model evidence, not model superiority, not tuning permission. |
| PM-050 Ian model-adjustment plan | Planning-only gate that selected route-freeze audit before model-specific adjustment. | Confirms no model adjustment had started and no model family was selected from validation diagnostics. |
| PM-051 route-freeze audit | Planning/audit-only gate that selected `AXIS_RAW_ROUTE_CONTRACT`. | Confirms the next minimal question was feature-contract preservation, not runtime or model-specific tuning. |
| PM-052 raw-route contract audit | Planning/spec read-only contract audit; PASS with caveats. | Confirms active `mentor_clean_v1` model `feature_columns` exclude raw OHLCV, raw volume, raw MACD, raw MACD signal, and raw MACD histogram. |

## Fastpath Option Table

| option_id | description | why_safe_or_blocked | time_to_model_debug_completion | risk | requires_runtime? | requires_semantic_feature_change? | recommended? |
|---|---|---|---|---|---|---|---|
| A | Ian model-adjustment closeout/synthesis. Define whether the current Ian-guided debugging trail is complete under validation-only constraints. | Safe and preferred. PM-042/048 already provide cross-family protocol-observability. PM-051/052 freeze the route and raw-feature contract. No model-specific action is needed to answer the current control question. | Fastest: one planning/control closeout doc and a later commit gate if accepted. | Low, if it stays non-claim and does not quote validation metric values. | No | No | Yes |
| B | LightGBM-first model-specific planning/spec gate. | Blocked as the next fastest path for this stage. PM-042 already supplies the needed LightGBM protocol-observability artifact; a LightGBM spec would duplicate or widen scope unless a future parent PM opens a new lane. | Slower: adds a model-specific branch before closeout. | Medium: can be mistaken as model selection from validation diagnostics. | No in spec form | No unless it proposes feature semantics changes | No |
| C | MS-DLinear+TCN-first model-specific planning/spec gate. | Blocked as the next fastest path for this stage. PM-048 already supplies the needed MS-DLinear+TCN protocol-observability artifact; a neural spec would widen scope unless a future parent PM opens a new lane. | Slower: adds model-specific planning before closeout. | Medium-high: easier to drift into capacity, architecture, seed, or hyperparameter decisions. | No in spec form | No unless it proposes feature semantics changes | No |
| D | Narrow stale-doc cleanup spec. | Deferred. PM-052 identified stale route-readiness text as cleanup-only, not a blocker to closeout/synthesis. Cleanup may be useful later, but it is not the fastest path to current Ian model-adjustment debug completion. | Slower for this objective: touches many status surfaces before answering the control question. | Low if scoped tightly, but broader file surface. | No | No | No |

## Recommended Next Path

Choose option A: Ian model-adjustment closeout/synthesis.

Rationale: PM-052 raw-contract PASS removes the immediate feature-contract blocker without authorizing semantic feature changes. PM-042 and PM-048 already supply the two model-family protocol-observability artifacts needed for a validation-only control closeout. Selecting LightGBM-first or MS-DLinear+TCN-first now would add model-specific scope without a route-lock need. Broad stale-doc cleanup is not a blocker because PM-052 classifies stale text as superseded cleanup-only drift.

This recommendation is based on scope, risk, and route completion. It is not based on validation metric values, model comparison, profitability, robustness, publishability, or test readiness.

## Ian Model-Adjustment Debug Completion Definition

Under validation-only constraints, "Ian model-adjustment debug completion" means:

- The route has a frozen feature, label, threshold, decision-time, scaler, calendar-split, report-scope, and test-embargo contract.
- Both LightGBM and MS-DLinear+TCN have protocol-observability artifacts under that route.
- The raw-feature contract has been audited and active model `feature_columns` exclude raw OHLCV, raw volume, and raw MACD-family columns.
- Remaining normalized-volume and normalized-MACD caveats are known protocol-change blockers, not hidden model-adjustment permissions.
- No model family is selected, tuned, or declared better from validation diagnostics.
- No test/holdout scoring, evidence promotion, or local-result claim is authorized.

By this definition, the current Ian-guided model-adjustment debugging trail is complete as a non-claim route-control sequence. The next task should be a closeout/commit gate, not runtime or model-specific adjustment.

## Exact Next PM Prompt

```text
PM-IAN-MODEL-ADJUSTMENT-CLOSEOUT-COMMIT-053B -- commit accepted Ian fastpath closeout

Task type: docs commit / no runtime.

Goal:
Commit exactly docs/PM_IAN_MODEL_ADJUSTMENT_FASTPATH_053_2026-05-31.md if accepted by parent PM, and nothing else.

Hard rules:
- Read hf_stock_clf/AGENTS.md and docs/ENVIRONMENT.md first.
- Verify HEAD is the PM-052B commit or descendant.
- Verify tracked diff is empty, cached diff is empty, and the only staged file after exact-path staging is docs/PM_IAN_MODEL_ADJUSTMENT_FASTPATH_053_2026-05-31.md.
- Verify the doc remains planning/control closeout only and does not quote validation metric values or claim performance, effectiveness, robustness, profitability, publishability, Ian-result success, model superiority, or test readiness.
- Commit message: docs: close Ian model-adjustment fastpath
- Do not run notebooks, training, smoke/full validation, local_baseline_matrix.py, evidence/KB updates, code/test edits, feature/threshold/label/scaler/decision-time/calendar/model-capacity/hyperparameter/seed changes, or test/holdout access.
```

## Caveat

PM-053 performed no code edits, runner/test edits, notebook execution, training, smoke/full validation, model adjustment, test/holdout access, evidence-matrix or claim-map update, Zotero/PDF/source-conversion/checkpoint work, feature/threshold/label/scaler/decision-time/boundary/model-capacity/hyperparameter/seed change, model selection from validation diagnostics, or evidence promotion. Runtime was not authorized; the targeted pytest command reported by one delegated auditor is recorded above as a process deviation and is not a basis for any claim or selection.
