# PM-IAN-MODEL-ADJUSTMENT-SYNTHESIS-NEXT-LANE-064

Date: 2026-06-01

Status: synthesis / next-lane decision only / no runtime / no evidence promotion / no test access

This document decides the next parent-PM lane after PM-062 push completion.
It does not run models, rerun validation, train, tune, execute notebooks, edit
code, promote evidence, select a model, or authorize test/holdout access.

## Starting State

PM-062 push gate is complete. `hf_stock_clf` is synced at:

| check | value |
| --- | --- |
| `HEAD` | `84a6e4ef1bc1769a18d0c477c19b07eccd1bcc28` |
| `origin/master` | `84a6e4ef1bc1769a18d0c477c19b07eccd1bcc28` |
| short commit | `84a6e4e` |
| branch relationship | `origin/master...HEAD = 0 0` |
| tracked diff before PM-064 doc | empty |
| cached diff before PM-064 doc | empty |
| known untracked | `.codegraph/` and three notebooks |

The KB handoff was stale before PM-063 because it still treated parent
acceptance of PM-061 and a network/push gate as the next active decision.
PM-063 updates the non-git KB handoff/log/index to record the completed push
gate before this synthesis is treated as active.

## Brief Lineage

| gate | pointer | PM-064 interpretation |
| --- | --- | --- |
| PM-050 | `docs/PM_IAN_MODEL_ADJUSTMENT_PLAN_050_2026-05-31.md` | Planned Ian-guided model-adjustment route control and chose route-freeze before model-specific action. |
| PM-051 | `docs/PM_IAN_ROUTE_FREEZE_AUDIT_051_2026-05-31.md` | Preserved route locks and selected `AXIS_RAW_ROUTE_CONTRACT`; no runtime or model selection. |
| PM-052 | `docs/PM_FEATURE_ROUTE_PROTOCOL_052_2026-05-31.md` | Raw-feature contract PASS with caveats; active `mentor_clean_v1` features exclude raw OHLCV, raw volume, and raw MACD-family columns. |
| PM-053 | `docs/PM_IAN_MODEL_ADJUSTMENT_FASTPATH_053_2026-05-31.md` | Closed the earlier Ian debugging trail as route-control/protocol documentation only. |
| PM-056 | `docs/PM_FEATURE_ROUTE_DOC_STALE_CLEANUP_SPEC_056_2026-05-31.md` | Planned stale route-readiness cleanup as documentation/provenance only. |
| PM-058 | `ab02262 docs: mark stale route-readiness blockers superseded` | Accepted cleanup committed; no route semantic change and no evidence promotion. |
| PM-059 | `docs/PM_IAN_MODEL_ADJUSTMENT_SPEC_059_2026-05-31.md` | Selected exactly one LightGBM validation-only diagnostic lane under existing route locks; no evidence promotion or test access. |
| PM-060 | `docs/PM_IAN_LGBM_ADJUST_059_ARTIFACT_REVIEW_060_2026-05-31.md` | Artifact review verdict `PASS_WITH_CAVEAT`; runner `smoke` naming caveat is non-blocking and non-claim. |
| PM-061 | `84a6e4e docs: record Ian LightGBM adjustment artifact review` | Accepted and committed PM-059/060 docs through the latest pushed commit. |
| PM-062 | push gate | Push completed; `HEAD == origin/master == 84a6e4e`. |

## Frozen Route Locks

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
| Ian guidance and papers | design rationale and blocker checks only, not local evidence or results |

## Source Consistency Check

| check | result | note |
| --- | --- | --- |
| Route-lock agreement across PM-050/051/052/053/056/059/060 and protocol lock | PASS | The route-lock values above remain consistent. |
| PM-062 pushed state | PASS | Live state is synced at `84a6e4e` with ahead/behind `0 0`. |
| Stale KB active push-gate wording after Stage A | PASS | PM-063 supersedes the active "next push gate" wording and records push completion. |
| PM-060 caveat visibility | PASS | The LightGBM runner naming caveat remains visible and non-blocking. |
| Validation metric use | PASS | This synthesis does not quote or rely on validation metric values. |
| Evidence/test boundary | PASS | No evidence matrix, claim map, model selection, or test/holdout access is authorized. |

No BLOCK-level source contradiction was found.

## Synthesis Question

Can the current Ian-guided debugging close as validation-only route-control, or
is a new preregistered validation-only lane needed?

Answer: close the current Ian model-adjustment trail as non-claim
route-control/protocol-observability. A new validation-only runtime lane is not
needed unless a future parent-PM gate identifies a non-metric protocol question
that cannot be answered by the existing PM-042, PM-048, PM-059, and PM-060
artifacts.

## Next-Lane Options

| option_id | decision | why safe or blocked | allowed future shape |
| --- | --- | --- | --- |
| `CLOSEOUT_ROUTE_CONTROL_ONLY` | SELECTED | Shortest safe path. PM-062 push is complete, route locks are frozen, PM-052 raw contract is PASS with caveats, PM-060 artifact review is PASS_WITH_CAVEAT, and all current outputs are non-claim protocol observability. | Parent-PM acceptance plus exact-path commit/KB-sync gate for this PM-064 synthesis. |
| `PREREGISTER_MS_DLINEAR_TCN_VALIDATION_ONLY` | NOT SELECTED | Safe only if a future non-metric protocol question exists. Current PM-048 already provides MS-DLinear+TCN validation-only protocol observability. Starting another neural lane now would widen scope. | Separate future spec gate only; no runnable command in this synthesis. |
| `PREREGISTER_LIGHTGBM_FOLLOWUP_VALIDATION_ONLY` | NOT SELECTED | Safe only if it answers a protocol or runner-caveat question, not model quality. Current PM-059/060 already cover the bounded LightGBM follow-up and record the runner naming caveat. | Separate future spec gate only if parent PM defines a non-metric question. |
| `BLOCKED_TEST_OR_EVIDENCE_PROMOTION` | BLOCKED | Test/holdout scoring, evidence promotion, model selection, Ian-result success claims, and performance wording remain embargoed. | No future gate unless a separate parent-PM protocol changes the embargo. |

## Recommendation

Recommended next parent-PM gate:

`PM-IAN-MODEL-ADJUSTMENT-CLOSEOUT-COMMIT-KB-SYNC-064B`

Choose exactly `CLOSEOUT_ROUTE_CONTROL_ONLY`. The current Ian model-adjustment
sequence should close as validation-only route-control/protocol-observability.
Do not open another runtime/spec lane from this synthesis. If a future lane is
needed, it must first be a separate preregistered validation-only spec justified
by a non-metric protocol question.

## Exact Next Prompt Text

Do not execute this prompt inside PM-064. It is prompt-ready text for a later
parent-PM gate only.

```text
PM-IAN-MODEL-ADJUSTMENT-CLOSEOUT-COMMIT-KB-SYNC-064B

Task type: parent-PM acceptance, exact-path commit, and KB sync verification only / no runtime.

Goal:
Accept PM-064 synthesis if the parent PM agrees, commit exactly
docs/PM_IAN_MODEL_ADJUSTMENT_SYNTHESIS_NEXT_LANE_064_2026-06-01.md in
hf_stock_clf, and verify the non-git KB handoff/log/index continuity sync.

Required first reads/checks:
- Read hf_stock_clf/AGENTS.md first.
- Read hf_stock_clf/docs/ENVIRONMENT.md.
- Verify HEAD is 84a6e4ef1bc1769a18d0c477c19b07eccd1bcc28 or a direct
  descendant accepted by the parent PM.
- Verify tracked and cached diffs before staging.
- Verify the only staged hf_stock_clf path is
  docs/PM_IAN_MODEL_ADJUSTMENT_SYNTHESIS_NEXT_LANE_064_2026-06-01.md.

Allowed:
- Stage and commit exactly the PM-064 synthesis doc if accepted.
- Commit message: docs: close Ian model-adjustment synthesis
- Verify KB non-git continuity files mention PM-062 push completion and PM-064
  closeout recommendation.
- Optionally update KB handoff/log/index only if the parent PM says the current
  sync is stale.

Forbidden:
- No runtime, rerun, training, smoke/full validation, local_baseline_matrix.py,
  notebook execution, code/script/test edits, evidence matrix, claim maps,
  Zotero, PDF/MinerU/source conversion, model selection, validation-metric
  promotion, Ian-result success claim, or test/holdout access.
- Never use git add .
- Do not stage notebooks, .codegraph, checkpoints, KB files, artifacts, or
  unrelated files in the hf_stock_clf commit.

Validation:
- Check LF endings and trailing whitespace for changed files.
- If the KB CSV is touched, parse it and confirm the row-width set remains [10].
- Run git diff --check.
- Report final git status, HEAD, origin/master, ahead/behind, tracked diff,
  cached diff, and exact staged/committed paths.
```

## PM+Agent Auditor Findings

| role | finding | result |
| --- | --- | --- |
| Route-State Auditor | PM-062 push completed and the stale KB push-gate wording is superseded by Stage A. | PASS |
| Protocol Freeze Auditor | No feature, label, threshold, decision-time, scaler, split, report-scope, or test-embargo drift found. | PASS |
| Ian Constraint Mapper | Ian guidance maps only to design constraints, route caveats, and blocker checks. | PASS |
| Next-Lane Planner | Closeout route-control is the shortest safe next gate; future runtime lanes require separate non-metric specs. | PASS |
| Claim-Scope/Test Embargo Auditor | No metric promotion, model selection, evidence promotion, or test/holdout access is authorized. | PASS |
| Final Adversarial Reviewer | The merged task remains docs/KB synthesis only and does not become runtime, evidence, or model selection. | PASS |

## Explicit Caveat

PM-064 did not run models, tune, select, promote evidence, open test/holdout,
authorize claims, execute notebooks, edit code/scripts/tests, stage, commit,
push, or start any follow-up runtime/spec lane.
