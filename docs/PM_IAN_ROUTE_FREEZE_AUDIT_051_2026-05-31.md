# PM-IAN-ROUTE-FREEZE-AUDIT-051

Date: 2026-05-31

Status: planning/audit-only / no runtime.

This document defines one Ian-guided adjustment axis for a later planning/spec
gate. It does not start model adjustment, runtime, notebook execution,
training, feature implementation, evidence promotion, or test/holdout access.

## Prior Gate State

| Gate | State | PM-051 interpretation |
| --- | --- | --- |
| PM-050 | Committed as `06d4436 docs: plan Ian route-freeze adjustment gate` | The safest next lane is route-freeze audit before any model-specific adjustment. |
| LightGBM 042 | Reviewed as full-input calendar-split validation-only diagnostic | Protocol-observability only; not model-performance evidence, tuning permission, or test/holdout authorization. |
| MS-DLinear+TCN 048 | Reviewed as full-input calendar-split train/validation-only diagnostic | Protocol-observability only; not model-performance evidence, tuning permission, or test/holdout authorization. |
| Test/holdout | Unopened for scoring, selection, baselines, metrics, and claims | Holdout timestamps may appear only as split-boundary metadata. |
| Ian guidance and papers | Design rationale and blocker checks only | Not local evidence, not result claims, and not threshold/model/feature selection authority. |

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
| Calendar train interval | `1998-01-02` to `2013-09-16`, half-open |
| Calendar validation interval | `2013-09-16` to `2017-01-25`, half-open |
| Calendar holdout metadata interval | `2017-01-25` to `2020-06-06`, half-open metadata only |
| `report_scope` | `validation_only` |
| `selection_scope` | `validation_only` |
| `test_metrics_embargoed` | `True` |
| `test_metrics_used` | `False` |

These locks remain active for any future prompt unless a separate PM-approved
protocol-change gate explicitly changes them. No validation metric value may
select a threshold, model family, model capacity, feature, scaler policy, seed,
or winning route.

## Source Consistency Check

| Source area | Check | Result |
| --- | --- | --- |
| PM-050 route freeze | Frozen route values match this PM-051 table. | PASS |
| PM-041/042 LightGBM trail | LightGBM full validation-only route is recorded as protocol-observability only. | PASS |
| PM-047/048 MS-DLinear+TCN trail | MS-DLinear+TCN full validation-only route is recorded as protocol-observability only. | PASS |
| Protocol lock current live state | Current live-state section preserves the route locks and non-claim interpretation. | PASS |
| KB protocol artifact CSV/log | KB rows and log entries for PM-042 and PM-048 preserve non-claim protocol-observability wording. | PASS |
| Older blocker/readiness text | Some older protocol/KB sections still say route pieces were blocked or `mentor_clean_v1` needed implementation. Later PM-042/048 and PM-050 supersede that readiness state. | PASS with stale-text caveat |
| Normalized-volume policy | Current lock permits completed-bar normalization including current row; KB design notes raise a stricter prior-only denominator question. This is an audit tension, not a current route-lock contradiction. | PASS with future-gate caveat |

No BLOCK-level source disagreement was found for carrying the frozen route into
the next planning/spec gate.

## Ian Constraint Mapping

| Focus | Allowed use in PM-051 | Forbidden use |
| --- | --- | --- |
| Raw feature exclusion | Treat Ian's raw OHLCV/raw volume/raw MACD removal guidance as a feature-contract audit constraint. | Do not claim local improvement or silently change `mentor_clean_v1`. |
| Normalized MACD | Keep normalized MACD handling as a formula/documentation audit topic. | Do not alter denominator, timing, or feature semantics in this gate. |
| Normalized volume | Record completed-bar versus strict prior-only normalization as a later protocol question. | Do not switch denominator/window semantics under unchanged `mentor_clean_v1`. |
| Trailing-only normalization | Require any future feature-internal statistics to be train-only or past/trailing only. | No centered, future-inclusive, validation/test-aware, or full-series normalization. |
| Decision time | Preserve `post_bar_close_completed_bar`. | No pre-close prediction without lagged/rebuilt features and a new gate. |
| No-trade band | Preserve fixed 5 bps no-trade labels and coverage/no-trade disclosure. | No paper-, notebook-, validation-, or test-selected threshold. |
| Literature route maps | Use as design-only/checklist-only route material. | No evidence_matrix, claim-map, Zotero, exact paper-result, or local-result promotion. |

## Candidate Axis Comparison

| axis_id | description | allowed_future_change | requires_new_feature_set_id? | leakage_risk | route_drift_risk | paper/Ian use | stop_rules | recommended? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AXIS_RAW_ROUTE_CONTRACT` | Raw-feature exclusion and route documentation cleanup: freeze what `mentor_clean_v1` means, distinguish active route truth from stale blocker text, and classify raw OHLCV/raw volume/raw MACD as excluded or quarantined. | Next gate may create a planning/spec doc and perform read-only code/metadata contract inspection. Later tests may assert the existing contract if they are separately approved. No runtime, formula change, model choice, or metric use. | No for documentation or no-op contract tests. Yes or BLOCK if inspection shows actual feature semantics must change under `mentor_clean_v1`. | Low: reduces accidental raw-feature leakage without introducing new transforms. | Low: preserves the frozen route and clarifies stale route-readiness text. | Ian/papers motivate audit constraints only: remove raw non-stationary inputs, preserve no-trade and chronological/train-only protocol. | Stop if current `mentor_clean_v1` contains raw features and fixing it would change semantics without a protocol-change gate; stop if validation metrics or papers are used as local evidence. | Yes |
| `AXIS_MACD_NORM_SPEC` | Normalized MACD handling: freeze exact formula, denominator, and completed-bar timing. | Later planning/spec may document or audit the existing formula only. Any denominator or timing change requires a separate protocol gate. | Likely yes if the formula changes; no only for documented no-op semantics. | Medium: MACD depends on price history and normalization can leak if denominator/fitting scope is underspecified. | Medium: protocol lock says ready for v1 while KB design material keeps formula questions open. | Ian motivates normalized MACD as a design question only. | Stop if denominator is selected from validation metrics, paper tables, test/holdout, or full-sample statistics. | No |
| `AXIS_VOLUME_TRAILING_POLICY` | Normalized volume and trailing-only policy: decide whether completed-bar rolling mean is sufficient or strict prior-only denominator is required. | Later planning/spec may define a prior-only rule. Any implementation or semantics change needs a separate gate and likely a new feature-set identifier. | Yes if changing current `normalized_volume_20` semantics; no only for documenting completed-bar policy. | Medium-high: rolling volume normalization is a common same-bar, full-period, and cross-day leakage surface. | High: broad policy could affect volume, volatility, indicators, and neural normalization at once. | Ian/papers motivate the audit question only. | Stop if policy expands to scaler changes, pre-close prediction, threshold changes, broad feature refactor, or runtime. | No |

## Recommendation

Choose `AXIS_RAW_ROUTE_CONTRACT` for the next planning/spec gate.

Reason: it is the smallest route-freeze follow-up that can reduce accidental
feature-route drift before any model-specific adjustment. It does not require
runtime, model choice, hyperparameter search, feature implementation, or a
validation-metric rationale. It also directly addresses the active ambiguity
found in the source audit: older protocol and KB text contains stale readiness
or blocker wording, while later PM-042, PM-048, and PM-050 carry the current
non-claim route state.

This recommendation is based on risk, scope, leakage control, and route
identity discipline. It is not based on validation metric values.

## Future Next Prompt Text

Do not execute this prompt inside PM-051. It is prompt-ready text for a later
parent-PM dispatch only.

```text
PM-FEATURE-ROUTE-PROTOCOL-052 -- mentor_clean_v1 raw-route contract audit

Task type: planning/spec and read-only contract audit only / no runtime.

Create an active goal. You are PM + multi-agent coordinator + adversarial
reviewer.

Goal:
Define the exact `mentor_clean_v1` raw-feature contract before any model-specific
adjustment. Determine whether raw OHLCV, raw volume, raw MACD, raw MACD signal,
and raw MACD histogram are already excluded or quarantined in the active route.
Classify every finding as one of:

- current contract already satisfies Ian raw-feature exclusion;
- documentation/stale-status cleanup only;
- no-op test/spec assertion candidate;
- BLOCK: semantic change required, needing a separate protocol-change gate and
  likely a new feature-set identifier.

Allowed:
- Read `AGENTS.md` first and `docs/ENVIRONMENT.md`.
- Inspect route docs, the runner/feature-set resolver, metadata schemas, and
  KB route maps needed to verify the feature contract.
- Create at most one planning/spec doc under `docs/`.
- Use Ian guidance and papers only as design constraints and blocker checks.

Forbidden:
- No code edits, runner/test edits, notebook execution, runtime command,
  training, smoke, full validation, or `local_baseline_matrix.py`.
- No evidence_matrix, claim-map, Zotero, PDF, source conversion, checkpoint,
  or KB evidence work.
- No threshold, label, scaler, decision-time, split-boundary, model-capacity,
  hyperparameter, seed, or feature implementation changes.
- No model selection from validation metrics.
- No test/holdout scoring, metric exposure, selection, or authorization.
- No performance, effectiveness, robustness, profitability, publishability,
  Ian-result, model-superiority, or test-readiness claim.

Frozen locks:
- `feature_set_id=mentor_clean_v1`
- `label_mode=no_trade_band`
- `threshold_bps=5.0`
- `threshold_source=fixed_pre_registered_5bps`
- `decision_time_policy=post_bar_close_completed_bar`
- `scaler_id=standard_pooled_train_only_v1`
- `scaler_fit_scope=pooled_train_after_per_ticker_chronological_split`
- train `1998-01-02` to `2013-09-16`, validation `2013-09-16` to
  `2017-01-25`, holdout metadata `2017-01-25` to `2020-06-06`, half-open
- `report_scope=validation_only`
- `selection_scope=validation_only`
- `test_metrics_embargoed=True`
- `test_metrics_used=False`

Stop rules:
- Stop if the audit discovers that preserving `mentor_clean_v1` requires a
  semantic feature change.
- Stop if any future change would alter threshold, labels, scaler, decision
  policy, calendar boundaries, model capacity, or test/holdout access.
- Stop if papers, Ian guidance, validation diagnostics, or old notebook
  outputs are framed as local evidence.
- Stop if validation metric values are quoted or used for selection.
- Stop if the task needs runtime, code edits, notebook execution, or evidence
  promotion.

Required output:
- PASS/BLOCK contract audit.
- Raw-feature contract table.
- Stale-doc cleanup list, if any.
- No-op assertion candidates versus protocol-change blockers.
- Recommended next PM prompt, planning-only unless separately authorized.
- Files inspected, commands run, validation checks, and final git status.
```

## PM-051 Caveats

- PM-051 did not start model adjustment, runtime, notebook execution, test
  access, evidence promotion, threshold selection, feature implementation,
  scaler changes, label changes, decision-policy changes, model-capacity
  changes, hyperparameter search, or seed search.
- PM-051 did not choose LightGBM or MS-DLinear+TCN from validation diagnostics.
- PM-051 did not quote or promote validation metric values.
- PM-051 did not use papers or Ian guidance as local evidence.
- PM-051 leaves the new planning doc uncommitted for parent PM review.
