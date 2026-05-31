# PM-FEATURE-ROUTE-DOC-STALE-CLEANUP-SPEC-056

Date: 2026-05-31

Status: planning/spec-only / no source cleanup yet / no runtime

## Live Repo State

Before this spec was created, `hf_stock_clf` was pushed and synchronized:

- `HEAD == origin/master == de81701 docs: close Ian model-adjustment fastpath`
- `git rev-list --left-right --count origin/master...HEAD` was `0 0`
- `git status --short --branch` showed `## master...origin/master` plus only known untracked `.codegraph/` and three notebooks
- tracked diff was empty
- cached diff was empty
- cached name list was empty

This task may create only this planning/spec document. It does not edit stale
source files, stage, commit, push, run runtime, or start model work.

## Current Route Truth Summary

| gate | current route-control status | non-claim interpretation |
| --- | --- | --- |
| PM-042 LightGBM full validation-only artifact review | Available as `docs/PM_LGBM_FULLVAL_042_ARTIFACT_REVIEW_2026-05-30.md`. | Protocol observability only; not model-quality evidence, not a model-family selection, and not test/holdout authorization. |
| PM-048 MS-DLinear+TCN full validation-only artifact review | Available as `docs/PM_MS_DLINEAR_TCN_FULLVAL_048_ARTIFACT_REVIEW_2026-05-31.md`. | Protocol observability only; not model-quality evidence, not a model-family selection, and not test/holdout authorization. |
| PM-050 Ian model-adjustment plan | Planning-only document that routed to a route-freeze audit. | No model adjustment started and no validation diagnostic was promoted. |
| PM-051 Ian route-freeze audit | Selected `AXIS_RAW_ROUTE_CONTRACT`. | Chose a feature-contract audit lane, not runtime, tuning, or model selection. |
| PM-052 raw-route contract audit | Raw-feature contract PASS with caveats. | Active `mentor_clean_v1` model features exclude raw OHLCV, raw volume, raw MACD, raw MACD signal, and raw MACD histogram. Normalized volume/MACD semantics are fixed unless a separate protocol-change gate creates a new feature-set decision. |
| PM-053 Ian fastpath closeout | Closed the current Ian-guided debugging trail as route-control/protocol documentation. | No model family was selected, tuned, proven better, or authorized for test/holdout. |

Current frozen route locks remain:

| lock | value |
| --- | --- |
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

## Stale-Source Inventory

| source_path | line_or_section | stale_text_summary | why_superseded | cleanup_class | allowed_future_edit | forbidden_edit | needs_parent_acceptance? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hf_stock_clf/docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md` | Lines 50-58 and lower decision sections around lines 149-188 | Lower historical text says LightGBM and combined MS-DLinear+TCN paths were not found or were blocked, while `mentor_clean_v1` needed implementation. | PM-042, PM-048, PM-050, PM-051, PM-052, and PM-053 now define the current route-control state. The top of this file already starts to mark the lower text as historical, but the lower sections still need safer local markers. | `cleanup_only_mark_superseded` | Add short supersession notes above affected historical sections and point to PM-042/048/050/051/052/053. Preserve the original historical text. | Do not rewrite the lock values, claim model quality, or remove no-test/no-evidence caveats. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_route_goal_status_dashboard_2026_05_30.md` | Lines 34-35, 76, 79-84, 93-105 | Status rows say LightGBM readiness, MS-DLinear+TCN readiness, normalized MACD, and route implementation pieces are blocked or missing. | PM-042 and PM-048 later supplied validation-only protocol-observability artifacts, and PM-052 records the raw-feature contract PASS with normalized MACD/volume caveats. | `cleanup_only_replace_status_row` | Replace stale status cells with a superseded-status note and current non-claim route-control pointer. Keep evidence/wiki and test/holdout blockers current. | Do not mark any model as effective, selected, test-ready, or evidence-ready. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_clean_v1_project_path_audit_2026_05_30.md` | Lines 35-36, 44, 47, 49-53, 73-77 | Read-only path audit says LightGBM path, combined MS-DLinear+TCN, normalized MACD, normalized volume, clean feature completion, and raw-feature removal were blocked or not confirmed. | PM-042/048/052 supersede route-readiness and raw-feature contract status. The audit remains useful as historical path-state provenance. | `cleanup_only_mark_superseded` | Add a top supersession banner and, if needed, status-row annotations that point readers to PM-042/048/052/053. | Do not erase the historical audit, claim no-trade improvement, or imply validation diagnostics prove feature quality. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_route_downloaded_only_reconciliation_2026_05_30.md` | Line 42 | Text says to keep LightGBM path-not-found and combined MS-DLinear+TCN path-not-found as active blockers. | Later PM route-control docs supersede those missing-path blockers for route-readiness status only. The downloaded-only paper queue findings remain historical. | `cleanup_only_mark_superseded` | Add a localized note that the missing-path blocker sentence is superseded for route-readiness by PM-042/048/052/053. | Do not modify paper-source quality decisions or promote any S2 row to evidence. | Yes |
| `stock_ml_knowledge_base/indexes/pm_shard_a_source_protocol_repair_reconciliation_2026_05_30.md` | Lines 17, 79, 141-157, 199-233 | PM shard text says LightGBM and combined MS-DLinear+TCN are not verified local routes and remain blocked until guardrail validation. | Later PM-042 and PM-048 provide validation-only protocol-observability route artifacts; PM-053 closes the current Ian debugging trail as route-control only. | `cleanup_only_mark_superseded` | Add supersession notes for local-route missing-path blockers; retain raw-feature exclusion constraints where they are still current. | Do not change paper-source repair conclusions, raw-feature exclusions, or evidence-promotion blockers. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_clean_v1_pm_best_route_execution_2026_05_30.md` | Lines 33, 61, 110, 148-155 | Route execution note says model routes remain implementation-blocked and lists blocked claims. | PM-042/048/052/053 supersede route-readiness blockers, while blocked-claim wording remains current. | `cleanup_only_replace_status_row` | Replace route-readiness status rows only; keep blocked-claim rows as current. | Do not convert a route being observable into a claim that either model is better or ready for test. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_route_goal_completion_audit_2026_05_30.md` | Lines 68, 76-81, 87-90 | Completion audit says baseline planning remains blocked without project-code approval and LightGBM/combined MS-DLinear+TCN readiness are blocked. | The paper-reading/design goal completion remains current, but route-readiness blockers are superseded by PM-042/048/052/053. | `cleanup_only_replace_status_row` | Add a note that paper-reading completion remains bounded while route-readiness blocker text is superseded. | Do not turn this paper-reading completion audit into evidence/wiki completion or implementation-readiness proof. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_aligned_literature_gap_queue_2026_05_30.md` | Lines 21-26 | Gap rows say `mentor_clean_v1`, normalized volume/MACD, LightGBM, and no-trade threshold were not implemented or not found. | PM-052 supersedes raw/normalized feature-contract status; PM-042/048 supersede missing model-route status; fixed no-trade threshold locks are current. | `cleanup_only_replace_status_row` | Replace status cells with superseded route-control pointers and retain any literature-gap rows that remain design-only or source-quality limited. | Do not treat literature gaps as closed evidence or claim that Ian guidance was empirically confirmed. | Yes |
| `stock_ml_knowledge_base/indexes/mentor_clean_v1_literature_to_experiment_map_2026_05_30.md` | Line 96 and related guardrail rows | Hypothesis row says `mentor_clean_v1` was not implemented and feature timestamp tests were missing. | PM-052 now records current raw-feature contract PASS with caveats, but no-op assertions or future tests remain separate gates. | `cleanup_only_replace_status_row` | Split the stale implementation status from still-future assertion/test candidates. Mark implementation-status text superseded, and keep test/assertion work as future gated work. | Do not authorize tests, code edits, or runtime from this cleanup spec. | Yes |
| `stock_ml_knowledge_base/NEXT_WINDOW_HANDOFF.md` | Current PM-054 section around lines 140-154 | Current handoff records PM-053 closeout and says stale-doc cleanup may be opened separately. | This is already the current state, not stale source text. | `still_current_no_edit` | No direct cleanup edit needed. Later handoff can be updated only by a separate accepted sync gate. | Do not rewrite handoff to imply runtime/model work is next by default. | No |
| `stock_ml_knowledge_base/wiki/log.md` | Current PM-054 entry around lines 595-598 | Current log records PM-051/052/053 commits and PM-053 closeout as route-control only. | This is already current and claim-safe. Older log entries are historical and should not be rewritten by a cleanup patch. | `still_current_no_edit` | No direct cleanup edit needed. If parent wants a log marker, append a new historical-note entry in a separate KB sync gate. | Do not rewrite old log history or promote validation diagnostics as evidence. | No |

Inventory result: no `blocker_stop_parent_review` item was found in this
planning/spec pass. The stale items are cleanup-only route-readiness drift.
Evidence, test/holdout, runtime, and model-quality blockers remain current.

## Future Cleanup Patch Gate Plan

A later cleanup patch gate may edit only exact paths accepted by the parent PM.
The candidate allowed paths are:

- `E:\codex_workspace\projects\hf_stock_clf\docs\MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_route_goal_status_dashboard_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_clean_v1_project_path_audit_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_route_downloaded_only_reconciliation_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\pm_shard_a_source_protocol_repair_reconciliation_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_clean_v1_pm_best_route_execution_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_route_goal_completion_audit_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_aligned_literature_gap_queue_2026_05_30.md`
- `E:\codex_workspace\projects\stock_ml_knowledge_base\indexes\mentor_clean_v1_literature_to_experiment_map_2026_05_30.md`

The later cleanup patch gate should not edit `NEXT_WINDOW_HANDOFF.md` or
`wiki/log.md` unless the parent PM explicitly opens a separate KB continuity
sync. Those files already contain the current PM-054 handoff/log state.

Forbidden paths and surfaces for a later cleanup patch gate:

- `ml_utils/**`
- `scripts/**`
- `tests/**`
- `notebooks/**`
- `checkpoints/**`
- `.codegraph/**`
- `data/**`
- `indexes/evidence_matrix.csv`
- claim-map files
- Zotero exports/imports
- PDF, MinerU, source-conversion, or paper-card files
- raw data or artifact outputs

## Future Cleanup Wording Templates

Allowed template for historical sections:

```text
Superseded route-readiness note: this section is retained for historical
context. Current non-claim route-control state is recorded in PM-042, PM-048,
PM-050, PM-051, PM-052, and PM-053. Those later docs do not authorize runtime,
model selection, evidence promotion, or test/holdout access.
```

Allowed template for status table cells:

```text
superseded_by_PM_042_048_052_053_route_control; current status is
protocol-observability/contract-recorded only, non-claim
```

Allowed template for rows where part of the old blocker remains current:

```text
route-readiness blocker superseded; evidence/test/model-quality blocker remains
active unless a separate parent-PM gate authorizes it
```

Forbidden cleanup wording:

- Do not say any model is better, effective, robust, profitable, publishable,
  test-ready, or selected.
- Do not say Ian guidance was confirmed by local results.
- Do not say validation diagnostics are evidence.
- Do not quote validation metric values.
- Do not say test/holdout can be opened.
- Do not say stale cleanup proves implementation correctness.

## Stop Rules For Later Cleanup Patch

Stop if any target file disagrees with the frozen route locks rather than merely
containing stale route-readiness text.

Stop if a proposed edit changes route semantics, feature semantics, thresholds,
labels, scaler policy, decision-time policy, split boundaries, model capacity,
hyperparameters, seeds, or test/holdout access.

Stop if a proposed edit touches any forbidden path or requires runtime, tests,
notebook execution, model work, evidence promotion, Zotero, PDFs, MinerU, or
source conversion.

Stop if a proposed edit promotes validation diagnostics, Ian guidance, or paper
guidance into local evidence or model-quality claims.

Stop if a proposed edit rewrites historical records without preserving
provenance. Prefer a supersession note or status-row replacement over deletion.

## PM+Agent Findings

Planned agent lanes for this spec:

- Stale-Source Auditor: identify exact stale statements and classify cleanup
  class.
- Route-Lock Auditor: verify the frozen route locks remain consistent.
- Claim-Scope Auditor: block evidence, model-quality, Ian-result, and
  test-readiness wording.
- Final Adversarial Reviewer: block source cleanup, runtime authorization,
  overbroad path scope, staging, commit, or push.

Main PM reconciliation: stale route-readiness text can be mapped for a later
exact-path cleanup patch. This spec itself is not that patch.

## Explicit Caveat

PM-056 did not edit KB/index/wiki source files, did not run runtime, did not run
tests, did not execute notebooks, did not train or adjust a model, did not access
test/holdout scoring, did not promote evidence, and did not stage, commit, or
push.
