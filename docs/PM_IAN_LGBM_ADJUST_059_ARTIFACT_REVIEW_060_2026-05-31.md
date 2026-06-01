# PM-IAN-LGBM-ADJUST-059-ARTIFACT-REVIEW-060

Date: 2026-05-31

Status: artifact-review only / validation-only diagnostic / no evidence promotion / no test access / no runtime rerun

This review records a static artifact audit of the PM-059 LightGBM validation-only diagnostic output. It does not rerun the command, train, tune, execute notebooks, stage, commit, push, update KB/evidence files, select a model family, promote validation diagnostics as evidence, or authorize test/holdout access.

## Optimized Hard-Rule Prompt

Role: PM + agent coordinator + adversarial reviewer. Review only the PM-059 LightGBM artifact and create exactly one review doc at `docs/PM_IAN_LGBM_ADJUST_059_ARTIFACT_REVIEW_060_2026-05-31.md` if safe. Read `AGENTS.md` and `docs/ENVIRONMENT.md` first, verify live git state, inspect PM-059 route/spec docs, and statically inspect `metadata.json`, `results.csv`, and `manifest.csv`. Forbidden: runtime rerun, training, smoke/full validation rerun, `local_baseline_matrix.py`, notebooks, code/script/test edits, KB/evidence/claim-map/Zotero/PDF/MinerU work, staging, commit, push, route-lock changes, model selection, validation metric promotion, or test/holdout scoring/exposure. Stop if artifacts are missing, output root has multiple child runs, route locks disagree, forbidden test/holdout scoring columns appear, review requires rerun/edit/deletion, or verdict depends on validation metric values. Output a PASS/PASS_WITH_CAVEAT/BLOCKED artifact review with route-lock, completeness, runtime-scope, split/scaler, test-embargo, claim-scope, and runner-naming audits.

## Artifact Paths

| item | path | status |
| --- | --- | --- |
| PM-059 spec | `docs/PM_IAN_MODEL_ADJUSTMENT_SPEC_059_2026-05-31.md` | Present; selected `LGBM_FIRST_VALIDATION_ONLY`. |
| Output root | `checkpoints/pm_ian_model_adjust_spec_059_lgbm_20260531` | Present. |
| Child run | `checkpoints/pm_ian_model_adjust_spec_059_lgbm_20260531/phase1b_local_no_trade_band_smoke_20260531_170100` | Present; exactly one child run found. |
| Metadata | `metadata.json` | Present. |
| Results | `results.csv` | Present; 6 rows. |
| Manifest | `manifest.csv` | Present; 6 rows. |

## Review Scope

Allowed scope:

- Static inspection of route docs and artifact files.
- Verification of frozen route-lock fields, validation-only scope, row/file completeness, and absence of forbidden concrete test/holdout scoring columns.
- Claim-scope review that treats metrics as diagnostic fields only and does not quote metric values.

Forbidden scope:

- No runtime rerun, training, smoke/full validation rerun, notebook execution, code edits, source cleanup, artifact deletion, staging, commit, push, KB sync, evidence matrix update, claim-map update, model selection, or test/holdout access.

## Frozen Route Lock Audit

| field | expected | observed | result |
| --- | --- | --- | --- |
| `feature_set_id` | `mentor_clean_v1` | `mentor_clean_v1` | PASS |
| `label_mode` | `no_trade_band` | `no_trade_band` | PASS |
| `threshold_bps` | `5.0` | `5.0` | PASS |
| `threshold_source` | `fixed_pre_registered_5bps` | `fixed_pre_registered_5bps` | PASS |
| `decision_time_policy` | `post_bar_close_completed_bar` | `post_bar_close_completed_bar` | PASS |
| `scaler_id` | `standard_pooled_train_only_v1` | `standard_pooled_train_only_v1` | PASS |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` | `pooled_train_after_per_ticker_chronological_split` | PASS |
| train interval | `[1998-01-02, 2013-09-16)` | `1998-01-02T00:00:00` to `2013-09-16T00:00:00` | PASS |
| validation interval | `[2013-09-16, 2017-01-25)` | `2013-09-16T00:00:00` to `2017-01-25T00:00:00` | PASS |
| holdout metadata interval | `[2017-01-25, 2020-06-06)` | `2017-01-25T00:00:00` to `2020-06-06T00:00:00` | PASS |
| interval convention | half-open | `half_open_start_inclusive_end_exclusive` | PASS |
| `report_scope` | `validation_only` | `validation_only` | PASS |
| `selection_scope` | `validation_only` | `validation_only` | PASS |
| `test_metrics_embargoed` | `True` | `True` | PASS |
| `test_metrics_used` | `False` | `False` | PASS |
| model family | `lightgbm` | `lightgbm` | PASS |
| feature view | `last_step` | `last_step` | PASS |
| window size | `12` | `12` | PASS |
| split mode | `calendar` | `calendar` | PASS |

Observed active feature columns are derived or normalized route features. Raw OHLCV, raw volume, raw MACD, raw MACD signal, and raw MACD histogram do not appear as active model feature columns in the metadata inspected for this artifact.

## Artifact Completeness

| check | observed | result |
| --- | --- | --- |
| Output root exists | Yes | PASS |
| Child run count | 1 | PASS |
| Child run name | `phase1b_local_no_trade_band_smoke_20260531_170100` | PASS_WITH_CAVEAT; see runner naming audit. |
| `metadata.json` exists | Yes | PASS |
| `results.csv` exists | Yes | PASS |
| `manifest.csv` exists | Yes | PASS |
| `results.csv` row count | 6 | PASS |
| `manifest.csv` row count | 6 | PASS |
| Output command reference | PM-059 Stage B command in `docs/PM_IAN_MODEL_ADJUSTMENT_SPEC_059_2026-05-31.md`; artifact metadata does not store the exact command text. | PASS_WITH_CAVEAT |

## Runtime Scope Audit

| check | observed | result |
| --- | --- | --- |
| Runtime rerun during PM-060 | None performed | PASS |
| Output roots reviewed | One PM-059 root | PASS |
| Child runs reviewed | Exactly one child run | PASS |
| Model family | LightGBM only | PASS |
| Model entry | One LightGBM classifier entry | PASS |
| Seed scope | Metadata records seed list `[42]`; result rows expose no additional seed values. | PASS |
| Broad sweep/grid/seed search | Not visible in inspected artifact metadata or PM-059 command. | PASS |
| Notebook execution | None performed in this review. | PASS |
| Code edits | None performed in this review. | PASS |
| Row caps | `max_rows_per_ticker=None`; `effective_max_rows_per_ticker=None` | PASS |

The PM-059 command reference includes `--model-family lightgbm`, `--validation-only-report`, `--validation-only-per-ticker`, `--feature-set mentor_clean_v1`, `--feature-view last_step`, five explicit tickers, seed `42`, calendar split boundaries, and the fixed no-trade band threshold. It intentionally omits `--smoke`, `--full-run`, `--max-rows-per-ticker`, label shuffling, grid search, and seed search according to the PM-059 spec.

## Split And Scaler Audit

| item | finding | result |
| --- | --- | --- |
| Calendar split | Metadata contains locked train, validation, and holdout metadata boundaries with half-open convention. | PASS |
| Holdout boundary use | Holdout timestamps appear only as metadata boundaries in the inspected metadata, not as scored output columns. | PASS |
| Scaler scope | Metadata records pooled train-only scaler fit after per-ticker chronological split. | PASS |
| Decision time | Metadata records post-bar-close completed-bar policy. | PASS |
| Feature contract | Metadata feature columns are derived/normalized active features under `mentor_clean_v1`. | PASS |

The timestamp fields are stored as ISO timestamps at midnight for split boundaries. This is consistent with the route docs' date-boundary lock and half-open interval intent.

## Test And Holdout Embargo Audit

| file | concrete forbidden `test_*` / `holdout_*` scoring columns | allowed embargo flags | result |
| --- | --- | --- | --- |
| `results.csv` | None found | `test_metrics_embargoed`, `test_metrics_used` | PASS |
| `manifest.csv` | None found | No test/holdout scoring flags required in this file | PASS |

Rows inspected in `results.csv` are validation split rows. The allowed embargo flags preserve `test_metrics_embargoed=True` and `test_metrics_used=False`. This review does not claim internal proof beyond the inspected artifact surfaces.

## Claim-Scope Audit

| check | finding | result |
| --- | --- | --- |
| Validation metric quotation | No metric values are quoted in this review. | PASS |
| Model-quality comparison | No comparison to PM-042, PM-048, or another model family is made as model quality. | PASS |
| Evidence promotion | Artifact is framed as validation-only diagnostic/protocol observability, not evidence. | PASS |
| Ian-result claim | No Ian-result success claim is made. | PASS |
| Test readiness | No test/holdout opening or readiness claim is made. | PASS |

Metric columns may exist in the artifact as diagnostic fields, but this review does not use them for evidence, selection, ranking, profitability, robustness, publishability, or test-readiness claims.

## Runner Naming Caveat

Verdict: PASS_WITH_CAVEAT.

The PM-059 spec says the command omitted `--smoke`, and it names the LightGBM `smoke` token as a known runner naming limitation. The reviewed artifact contains `run_mode='smoke'` and the child run directory contains `smoke`, but the metadata also records `max_rows_per_ticker=None`, `effective_max_rows_per_ticker=None`, explicit calendar split fields, validation-only report and selection scope, and test metric embargo fields. PM-041 and PM-042 document the same LightGBM naming limitation: the non-full LightGBM route can carry internal `smoke` naming while still representing an uncapped calendar-split validation-only diagnostic when row caps are null and validation-only fields hold.

This is non-blocking for artifact acceptance, but it must remain visible in any commit/sync handoff. The artifact metadata does not store the exact command text, so command omission is verified from the PM-059 spec and parent PM context rather than from metadata alone.

## PM+Agent Auditor Findings

| role | finding | result |
| --- | --- | --- |
| Artifact Completeness Auditor | Required root, one child run, `metadata.json`, `results.csv`, and `manifest.csv` are present; row counts are 6 and 6. | PASS |
| Route-Lock/Leakage Auditor | Frozen route fields match metadata; active feature columns are derived/normalized route features. | PASS |
| Test Embargo Auditor | No forbidden concrete test/holdout scoring columns found in `results.csv` or `manifest.csv`; allowed embargo flags hold. | PASS |
| Runtime Scope Auditor | One LightGBM diagnostic artifact, no broad sweep visible, no rerun performed; `smoke` naming caveat remains. | PASS_WITH_CAVEAT |
| Claim-Scope Auditor | Review uses non-claim wording and does not quote metric values or promote validation diagnostics. | PASS |
| Final Adversarial Reviewer | No blocker found; runner naming limitation should prevent an unqualified PASS. | PASS_WITH_CAVEAT |

## Final Verdict

PASS_WITH_CAVEAT.

The PM-059 LightGBM artifact is acceptable as a validation-only diagnostic/protocol-observability artifact under the inspected route locks and embargo fields. The caveat is the LightGBM runner naming limitation: `run_mode='smoke'` and the child path include `smoke` despite PM-059 command text omitting `--smoke`. Because row caps are null, validation-only fields hold, and prior route docs identify this naming behavior, the caveat is recorded as non-blocking.

This verdict is not model evidence, not a model-quality claim, not model selection, not an Ian-result success claim, not robustness/profitability/publishability/test-readiness evidence, and not authorization to open test/holdout.

## Recommended Next Parent-PM Gate

If the parent PM accepts this review, open a separate exact-path docs commit and optional KB-sync decision for:

- `docs/PM_IAN_MODEL_ADJUSTMENT_SPEC_059_2026-05-31.md`
- `docs/PM_IAN_LGBM_ADJUST_059_ARTIFACT_REVIEW_060_2026-05-31.md`

Do not start push, runtime rerun, model-specific selection, evidence promotion, or test/holdout work from this review. A push gate should be separate and only after parent PM acceptance of the relevant commit state.

## Explicit Caveat

PM-060 performed no runtime rerun, no training, no smoke/full validation rerun, no notebook execution, no code/script/test edits, no artifact deletion, no KB/evidence/claim-map update, no model selection, no test/holdout access, no staging, no commit, and no push.
