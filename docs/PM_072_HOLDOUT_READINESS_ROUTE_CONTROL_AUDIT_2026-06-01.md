# PM-072 Holdout-Readiness Route-Control Audit

Date: 2026-06-01

Status: PASS_WITH_CAVEAT / route-control readiness only / no final holdout or
test access / no evidence promotion

## Scope

PM-072 audits whether the accepted validation-only lineage has a frozen,
non-claim route that can be considered by the parent PM for a future, separately
authorized final holdout/test execution gate.

This audit did not run runtime, training, smoke/full validation, notebooks, or
`local_baseline_matrix.py`. It did not expose, score, select from, or authorize
final test/holdout data. It did not quote validation metric values, rank models,
promote evidence, update claim maps, or change code.

## Compressed Hard-Rule Prompt Used

Role: PM plus narrow internal auditors for lineage, route freeze,
validation-leakage, test embargo, claim scope, CLI feasibility, exact-path git,
KB sync, and adversarial review.

Context: PM-070/071 was accepted as `PASS_WITH_CAVEAT` at
`1f053ed0680171a1a18b9491193ca208df8f145f`. The current lineage under review is
PM-059/060, PM-065B/066, and PM-068/069/070, with current PM-069 artifact root
`checkpoints/pm_069_controlled_validation_only_model_testing_20260601` and child
`phase1b_local_no_trade_band_smoke_20260531_192407`.

Allowed actions: read local rules and environment; verify git state; inspect
relevant prior PM docs, KB sync entries, artifact metadata/results/manifest, and
runner source only enough to assess readiness and future command feasibility;
write exactly the PM-072 and PM-073 docs if not blocked; validate, exact-path
stage/commit/push only those docs if both docs are PASS/PASS_WITH_CAVEAT; then
sync only the three allowed KB files.

Forbidden actions: no runtime, training, notebooks, code edits, runner edits,
route-lock changes, validation metric quotation, model ranking, model selection
from metrics, evidence promotion, claim-map/evidence-matrix work, broad search,
final test/holdout execution, or PM-074 authorization.

Stop rules: stop on unreadable rules/environment, uninspectable git, pre-write
tracked/cached diff, `HEAD != origin/master`, missing prior docs/artifacts,
source route-lock disagreement, forbidden concrete test/holdout scoring columns,
route identification that depends on validation metrics, PM-073 criteria that
authorize test without a separate PM-074 parent gate, unsupported future command
requirements, non-exact staged paths, commit/push failure, or unsafe KB schema.

Output schema: report verdicts, route locks, artifact lineage, caveats, PM-073
criteria outcome, exact git/KB state, files inspected/changed, commands run,
role audits, unresolved issues, and explicit no-runtime/no-holdout caveats.

Drift-control checklist: re-check live git before writing and before final;
separate committed repo state from KB state; stage exact paths only; quote no
metrics; preserve validation-only/non-claim language; keep PM-074 separate.

## Live Pre-Write State

| Check | Result |
|---|---|
| Branch | `master` |
| `HEAD` | `1f053ed0680171a1a18b9491193ca208df8f145f` |
| `origin/master` | `1f053ed0680171a1a18b9491193ca208df8f145f` |
| ahead/behind | `0 0` |
| tracked diff | empty |
| cached diff | empty |
| `git diff --check` | clean |
| known untracked | `.codegraph/` and three notebooks |

This satisfied the PM-072 pre-write gate.

## Accepted Lineage

| Lineage item | Audit result |
|---|---|
| PM-059/060 Ian LightGBM adjustment artifact review | `PASS_WITH_CAVEAT`; validation-only/non-claim; no model selection, evidence promotion, or test/holdout authorization. |
| PM-065B/066 MS-DLinear+TCN validation-only model-testing artifact review | `PASS_WITH_CAVEAT`; CLI-aligned validation-only torch route; row caps null; `smoke` naming caveat recorded as non-blocking. |
| PM-068/069/070 controlled validation-only MS-DLinear+TCN artifact review | `PASS_WITH_CAVEAT`; current pushed docs lineage at `1f053ed`; route-lock and embargo checks passed. |
| KB PM-070/071 sync | Present in `NEXT_WINDOW_HANDOFF.md`, `wiki/log.md`, and protocol CSV row `pm_069_controlled_validation_artifact_review_070_20260601`. |

The current readiness candidate is the PM-068/069/070 MS-DLinear+TCN route, not
the Ian LightGBM diagnostic. The route is identifiable because PM-068 framed a
non-metric current-pushed-HEAD freshness question for the existing torch
`ms_dlinear_tcn` route after PM-065B/066, not because validation metric values
were compared or ranked.

## Frozen Route Locks

| Field | Readiness state |
|---|---|
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| calendar split | train `[1998-01-02, 2013-09-16)`, validation `[2013-09-16, 2017-01-25)`, holdout metadata `[2017-01-25, 2020-06-06)` |
| interval convention | half-open start-inclusive/end-exclusive |
| `model_family` | `torch` |
| `models` | `[ms_dlinear_tcn]` |
| `tickers` | `[CSCO, JPM, KO, MSFT, WMT]` |
| `seeds` | `[42]` |
| `max_epochs` | `1` |
| `batch_size` | `256` |
| `window_size` | `12` |
| `feature_view` | `last_step` |
| row caps | `max_rows_per_ticker=None`; `effective_max_rows_per_ticker=None` |
| validation scope | `report_scope=validation_only`; `selection_scope=validation_only` |
| test embargo | `test_metrics_embargoed=True`; `test_metrics_used=False` |
| claim flags | `diagnostic_only=True`; `non_claim=True` |

The PM-069 artifact metadata stores the calendar fields as `calendar_*`
timestamps and matches these locks. Source docs inspected did not disagree on
the route locks.

## Artifact Completeness

| Artifact surface | Result |
|---|---|
| Parent root | `checkpoints/pm_069_controlled_validation_only_model_testing_20260601` exists. |
| Child run count | Exactly one child run exists. |
| Child run | `phase1b_local_no_trade_band_smoke_20260531_192407`. |
| Required files | `metadata.json`, `results.csv`, and `manifest.csv` exist. |
| Row completeness | `results.csv` and `manifest.csv` each contain six rows, matching pooled plus five tickers. |
| Row scope | `results.csv` rows are validation rows only. |
| Validation-only fields | `report_scope=validation_only`, `selection_scope=validation_only`, `test_metrics_embargoed=True`, and `test_metrics_used=False` hold. |
| Forbidden columns | No forbidden concrete `test_*` or `holdout_*` scoring columns were found; allowed embargo booleans are present. |

Metric columns exist only as validation diagnostic fields. This audit did not
quote, interpret, rank, or select from their values.

## Leakage And Claim-Scope Audit

| Question | Result |
|---|---|
| Did accepted route selection depend on validation metric values? | No. PM-068 defines a non-metric protocol-observability question for the already specified `ms_dlinear_tcn` route. |
| Is the exact final route/model identifiable without using validation metrics? | Yes: `model_family=torch`, `models=[ms_dlinear_tcn]`, five tickers, seed `42`, and the fixed route locks above. |
| Were validation diagnostics promoted as evidence? | No evidence matrix, claim map, paper card, Zotero, PDF/MinerU/source conversion, or paper-claim work was part of the accepted lineage or this audit. |
| Was final test/holdout accessed so far? | No concrete final test/holdout scoring columns were exposed in PM-069 surfaces; holdout appears only as metadata interval boundaries. |
| Did this audit run or authorize final holdout/test? | No. |

## Known Caveats

1. The child run path and emitted run mode contain `smoke`. This remains a
   runner naming caveat, not a row-capped smoke blocker, because the PM-069
   command omitted `--smoke`, calendar split makes row caps null, route locks
   hold, and validation-only/test-embargo fields hold.
2. PM-069 artifacts are protocol-observability artifacts. They are not
   model-quality evidence, tuning evidence, publishability evidence, robustness
   evidence, or test-readiness proof.
3. Current artifact review proves the inspected artifact surfaces only. A
   future final holdout/test execution still requires a separate PM-074 gate,
   a fresh exact command review, and post-artifact review.

## PM+Agent Audit Results

| Role | Result |
|---|---|
| Lineage Auditor | PASS: accepted PM-059/060, PM-065B/066, and PM-068/069/070 lineage is present; current pushed repo lineage starts at `1f053ed`. |
| Route-Freeze Auditor | PASS: route locks are complete and consistent across inspected current docs and artifact surfaces. |
| Validation-Leakage Auditor | PASS: route identification does not depend on validation metric ranking or comparison. |
| Test-Embargo Auditor | PASS: no concrete final test/holdout scoring columns were found in the inspected validation-only artifact surfaces. |
| Claim-Scope Auditor | PASS: wording remains non-claim and no validation metric values are quoted. |
| CLI/Execution Feasibility Auditor | PASS_WITH_CAVEAT: current runner source exposes calendar holdout boundaries and non-validation-only evaluation paths, but future execution must be separately reviewed under PM-074 before use. |
| Exact-Path Git Auditor | PASS pre-write: tracked/cached diffs were empty and only two PM docs are authorized for this stage. |
| KB Sync Auditor | PASS pre-write: PM-070/071 sync is present; further KB sync is allowed only after exact-path docs commit and push. |
| Final Adversarial Reviewer | PASS_WITH_CAVEAT: no hidden test exposure or metric-selection blocker found; the `smoke` naming caveat and separate PM-074 requirement must remain explicit. |

## PM-072 Verdict

PM-072 verdict: PASS_WITH_CAVEAT.

The project has a sufficiently frozen, non-claim, validation-only-reviewed
MS-DLinear+TCN route to consider a future final holdout/test authorization gate.
This is not authorization to run final holdout/test. It is only readiness for
the parent PM to decide whether to open a separate PM-074 execution gate.
