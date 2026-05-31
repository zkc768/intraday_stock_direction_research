# PM-IAN-MODEL-ADJUSTMENT-PLAN-050

Date: 2026-05-31

Status: planning-only / route-freeze / no runtime.

This document prepares the next Ian-guided model-adjustment lane after the LightGBM and MS-DLinear+TCN full-input validation-only gates. It does not start model adjustment, does not run training or inference, does not choose a model from validation diagnostics, and does not authorize test or holdout access.

## Current Gate State

| Gate | State | Scope decision |
| --- | --- | --- |
| LightGBM full-input validation-only review | Available in `docs/PM_LGBM_FULLVAL_042_ARTIFACT_REVIEW_2026-05-30.md` | Protocol observability only; not model-performance evidence. |
| MS-DLinear+TCN full-input validation-only review | Committed in PM-049B as `1ba8f1a docs: record MS-DLinear TCN full validation-only review` | Protocol observability only; not model-performance evidence. |
| Test/holdout | Unopened for scoring, selection, baselines, metrics, and claims | Boundary timestamps may remain split metadata only. |
| Evidence promotion | Not authorized | No evidence_matrix, claim-map, Zotero, or performance-claim promotion. |
| Model adjustment | Not started | PM-050 only selects the safest next planning gate. |

## Frozen Route Table

| Field | Frozen value |
| --- | --- |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |
| Calendar train interval | `1998-01-02` to `2013-09-16`, half-open |
| Calendar validation interval | `2013-09-16` to `2017-01-25`, half-open |
| Calendar holdout metadata interval | `2017-01-25` to `2020-06-06`, half-open metadata only |

Any future adjustment plan must preserve the route locks unless a separate PM-approved protocol-change gate explicitly authorizes a different route. Validation diagnostics remain mock-exam observability; they cannot be used to tune thresholds, pick a winning model, or open the final exam.

## Paper And Ian Guidance Use Policy

Ian guidance and papers may be used only as design rationale, constraints, and blocker checks. In the current route family, the allowed use is to motivate a bounded spec around stationarity, decision-time-safe feature construction, selective/no-trade reporting, and leakage-safe chronological evaluation.

Forbidden uses:

- Do not treat papers, advisor guidance, LightGBM validation diagnostics, or MS-DLinear+TCN validation diagnostics as local evidence.
- Do not select a threshold, model family, model capacity, hyperparameter, seed, feature set, scaler, or decision policy from papers or validation metrics.
- Do not frame a future adjustment as improving, robust, profitable, publishable, test-ready, or an Ian-result success.
- Do not update evidence_matrix, claim maps, Zotero, or paper evidence tables from this planning lane.

## Candidate Lane Comparison

| lane_id | goal | allowed_changes | forbidden_actions | evidence_use | risk | stop_rules | next_prompt_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ROUTE_FREEZE_AUDIT_051` | Freeze exactly one Ian-guided adjustment axis before any model-specific spec. | One planning/audit doc that reconciles route locks, Ian feature guidance, source docs, and allowed future adjustment axes. | No code, runtime, notebooks, KB evidence writes, threshold changes, feature implementation, scaler changes, or model selection. | Design constraints only. | Lowest scope risk; slower because it inserts one extra audit gate. | Stop if source docs disagree on route locks; stop if any lane depends on validation metric values; stop if an adjustment axis would alter threshold, label, scaler, decision time, or boundaries without a separate PM gate. | `PM-IAN-ROUTE-FREEZE-AUDIT-051` |
| `LGBM_ADJUST_SPEC_051` | Draft a bounded LightGBM adjustment spec first. | One planning/spec doc for a future LightGBM-only adjustment candidate, scoped to Ian-guided feature/stationarity questions. | No runtime, hyperparameter search, metric-based selection, threshold change, test access, or evidence promotion. | Design constraints only; route diagnostics only establish runner observability. | Medium; simpler later runtime surface, but could be mistaken as choosing LightGBM from validation diagnostics. | Stop if the recommendation cites validation metrics, alters route locks, or widens into tuning/search. | `PM-LGBM-ADJUST-SPEC-051` |
| `MSDLTCN_ADJUST_SPEC_051` | Draft a bounded MS-DLinear+TCN adjustment spec first. | One planning/spec doc for a future neural adjustment candidate under the same route locks. | No runtime, architecture/capacity tuning, seed search, threshold change, test access, or evidence promotion. | Design constraints only; route diagnostics only establish runner observability. | Higher; neural changes are easier to confuse with model-capacity tuning and validation-driven optimization. | Stop if any capacity, kernel, horizon, loss, threshold, or hyperparameter change is justified by validation diagnostics. | `PM-MS-DLINEAR-TCN-ADJUST-SPEC-051` |

## PM Recommendation

Choose `ROUTE_FREEZE_AUDIT_051` as the next minimal lane.

Reason: both model families have now produced full-input validation-only artifacts, but those artifacts are protocol-observability records only. They do not justify selecting LightGBM, selecting MS-DLinear+TCN, tuning either model, or changing the route. The lowest-risk next step is a cross-model route-freeze audit that chooses exactly one allowable Ian-guided adjustment axis before any model-specific spec. This keeps the decision grounded in scope control and leakage prevention rather than validation metric values.

Important caveat: axes such as raw-feature exclusion, normalized MACD handling, normalized volume handling, or trailing-only normalization are safe audit topics only. Any implementation that changes feature semantics under the unchanged `mentor_clean_v1` identifier is blocked until a separate protocol-change gate either preserves the route with documented no-op semantics or creates a new feature-set identifier.

No model adjustment has started in PM-050.

## Future Prompt Text For Separate PM-051 Dispatch

Do not execute this prompt inside PM-050. The block below is prompt-ready text for a later PM-051 window only.

PM-IAN-ROUTE-FREEZE-AUDIT-051

Create an active goal. You are PM + multi-agent coordinator + adversarial reviewer.

First compress this prompt into hard rules, allowed/forbidden actions, stop rules, exact outputs, and a claim-safety checklist. Report the compressed hard-rule prompt in final closeout, then execute it.

Context:

- Repo: `E:\codex_workspace\projects\hf_stock_clf`
- KB: `E:\codex_workspace\projects\stock_ml_knowledge_base`
- Prior gates:
  - LightGBM full-input validation-only artifact review is protocol-observability only.
  - MS-DLinear+TCN full-input validation-only artifact review is protocol-observability only.
  - PM-050 recommended a cross-model route-freeze audit before any model-specific adjustment.
- Current task is planning/audit only. It must not run training, smoke tests, full validation, notebooks, or model code.

Goal:

Define exactly one Ian-guided adjustment axis that can be safely carried into a later model-specific spec gate without changing the frozen route by accident. If no single axis is safe, output blockers and the minimal repair task instead.

Hard route locks:

- `feature_set_id=mentor_clean_v1`
- `label_mode=no_trade_band`
- `threshold_bps=5.0`
- `threshold_source=fixed_pre_registered_5bps`
- `decision_time_policy=post_bar_close_completed_bar`
- `scaler_id=standard_pooled_train_only_v1`
- `scaler_fit_scope=pooled_train_after_per_ticker_chronological_split`
- calendar split: train `1998-01-02` to `2013-09-16`, validation `2013-09-16` to `2017-01-25`, holdout metadata `2017-01-25` to `2020-06-06`, half-open intervals
- `report_scope=validation_only`
- `selection_scope=validation_only`
- `test_metrics_embargoed=True`
- `test_metrics_used=False`

Allowed actions:

- Read `AGENTS.md` first and `docs/ENVIRONMENT.md`.
- Inspect only route docs, artifact reviews, KB route maps, and recent KB log entries needed for Ian-guided route-freeze analysis.
- Create at most one planning doc: `docs/PM_IAN_ROUTE_FREEZE_AUDIT_051_2026-05-31.md`.
- Compare candidate adjustment axes such as raw-feature removal, normalized MACD handling, normalized volume handling, trailing-only normalization policy, or route documentation cleanup.
- Use papers and Ian guidance only as design constraints and rationale.

Forbidden actions:

- No code edits, runner/test edits, notebook execution, runtime command, training, smoke, full validation, or `local_baseline_matrix.py`.
- No evidence_matrix, claim-map, Zotero, or PDF work.
- No threshold, feature implementation, label, scaler, decision-time, boundary, model-capacity, hyperparameter, or seed changes.
- No model selection from validation metrics.
- No test/holdout scoring, metric exposure, selection, or authorization.
- No performance, effectiveness, robustness, profitability, publishability, Ian-result, or test-readiness claim.

Required agents:

- Protocol Freeze Auditor: verify frozen route locks and contradictions.
- Ian Constraint Mapper: map Ian guidance to constraints, not local evidence.
- Candidate Axis Planner: compare 2-3 adjustment axes and recommend exactly one.
- Leakage/Test Embargo Auditor: challenge scaler, threshold, feature, label, decision-time, split, and test/holdout drift.
- Claim-Scope Auditor: block metric promotion and performance wording.
- Final Adversarial Reviewer if available.

Required output:

- Goal status.
- Compressed hard-rule prompt used.
- Agents used and PASS/BLOCK findings.
- Files inspected.
- Chosen adjustment axis or blocker.
- Candidate-axis comparison table.
- Exact next PM prompt for the chosen subsequent planning gate, planning-only unless separately authorized.
- Validation results for the new doc: newline, trailing whitespace, `git diff --check`.
- Final git status/diff/cached diff.
- Explicit caveat that no model adjustment, runtime, test access, or evidence promotion occurred.

Stop rules:

- Stop if source docs disagree on route locks.
- Stop if a candidate axis requires code/runtime changes before a separate spec gate.
- Stop if any recommendation depends on validation metric values.
- Stop if papers or Ian guidance are framed as local evidence.
- Stop if the plan would alter threshold, labels, scaler, decision policy, boundaries, model capacity, or test/holdout access.

## PM-050 Stop Rules Carried Forward

- Stop if source docs disagree on route locks in a way that affects allowed next actions.
- Stop if planning would require reading or scoring test/holdout.
- Stop if any recommendation depends on validation metric values or declares a winning model.
- Stop if papers are used as local evidence.
- Stop if the next task would require code/runtime changes without a separate PM approval.
- Stop if the plan would alter threshold, features, labels, scaler, decision policy, boundaries, or model capacity before a dedicated spec gate.
