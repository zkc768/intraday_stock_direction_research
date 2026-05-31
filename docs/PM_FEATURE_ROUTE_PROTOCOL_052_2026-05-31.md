# PM-FEATURE-ROUTE-PROTOCOL-052

Date: 2026-05-31

Status: planning/spec and read-only contract audit only / no runtime

Parent acceptance context: PM-051 was accepted by the parent PM and was committed in Stage 1 of this sequence as `78272ca docs: record Ian route-freeze audit`. PM-051 selected `AXIS_RAW_ROUTE_CONTRACT` and recommended `PM-FEATURE-ROUTE-PROTOCOL-052 -- mentor_clean_v1 raw-route contract audit`.

This document is a route-contract audit. It does not start model adjustment, edit code, execute notebooks, run training, run smoke/full validation, open test/holdout, update evidence artifacts, or promote validation diagnostics.

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

## Source Consistency Check

| Source area | Finding | Status |
|---|---|---|
| Local rules | `AGENTS.md` and `docs/ENVIRONMENT.md` were read before this write. | PASS |
| PM-051 route handoff | PM-051 selected `AXIS_RAW_ROUTE_CONTRACT`, required no semantic feature change under unchanged `mentor_clean_v1`, and routed to this planning/spec gate. | PASS |
| Protocol lock | The lock names `mentor_clean_v1`, post-bar-close completed-bar semantics, train-only pooled scaler policy, validation-only reporting, and test/holdout embargo. Older lower-section blocker text is superseded by later PM-042/048/050/051 state. | PASS with stale-doc cleanup |
| Runner feature resolver | `MENTOR_CLEAN_V1_FEATURES` contains only derived/normalized features; `FEATURE_SETS["mentor_clean_v1"]` maps to that tuple; active `feature_cols` are derived from `FEATURE_SETS[feature_set_id]`. | PASS |
| Final model input gate | Dataset construction uses `ordered[feature_cols]` as the model matrix, so retained raw source columns are not model inputs unless included in `feature_cols`. | PASS |
| Existing route tests | Existing tests assert that `MENTOR_CLEAN_V1_FEATURES` excludes raw `open`, `high`, `low`, `close`, `volume`, `macd`, `macd_signal`, and `macd_hist`. | PASS |
| KB route maps/logs | KB route maps preserve Ian raw-feature exclusion as a design constraint, but several 2026-05-30 status notes are stale relative to PM-042/048/050/051. | PASS with cleanup-only drift |
| Leakage/test embargo | Same-bar source use is consistent only with `post_bar_close_completed_bar`; scaler scope and test/holdout embargo remain locked. | PASS with caveats |
| Claim scope | No validation metric value is quoted or used for selection; Ian/papers are not framed as local evidence. | PASS |

## Raw-Feature Contract Table

| item | expected_contract | observed_source | observed_status | classification | blocker_or_next_action |
|---|---|---|---|---|---|
| raw open/high/low/close | Excluded from active model `feature_columns`; allowed only as completed-bar source inputs for derived features under `post_bar_close_completed_bar`. | `scripts/phase1b_local/local_baseline_matrix.py` defines `OHLCV_FEATURES`, validates raw inputs, computes derived features, and keeps `MENTOR_CLEAN_V1_FEATURES` free of raw OHLC columns. `ml_utils/dataset.py` gates model input through `feature_cols`. | Raw OHLC columns are source inputs and retained frame columns, not active model features for `mentor_clean_v1`. | current contract already satisfies Ian raw-feature exclusion | Keep the distinction between source columns and model `feature_columns`. BLOCK any future raw OHLC inclusion under unchanged `mentor_clean_v1`. |
| raw volume | Excluded from active model `feature_columns`; allowed only as completed-bar source input for normalized volume. | `MENTOR_CLEAN_V1_FEATURES` includes `normalized_volume_20`, not raw `volume`; input validation still requires `volume`. | Raw `volume` is a source input, not an active model feature. | current contract already satisfies Ian raw-feature exclusion | Keep current completed-bar caveat. A strict prior-only normalization change would be semantic drift and needs a separate protocol-change gate, likely with a new feature-set identifier. |
| raw MACD | Excluded from active model `feature_columns`; not required as input for `mentor_clean_v1`. | The `technical_v1` path can create `macd`, but the `mentor_clean_v1` path computes normalized MACD math internally and does not add raw `macd` as an active feature. | Raw `macd` is not an active `mentor_clean_v1` feature. | current contract already satisfies Ian raw-feature exclusion | BLOCK any future raw `macd` inclusion under unchanged `mentor_clean_v1`. |
| raw MACD signal | Excluded from active model `feature_columns`; not required as input for `mentor_clean_v1`. | `_normalized_macd_hist_one_group` computes signal as a local intermediate and returns a normalized histogram value. | Raw `macd_signal` is not an active `mentor_clean_v1` feature. | current contract already satisfies Ian raw-feature exclusion | BLOCK any future raw `macd_signal` inclusion under unchanged `mentor_clean_v1`. |
| raw MACD histogram | Excluded from active model `feature_columns`; replaced by a normalized equivalent. | `MENTOR_CLEAN_V1_FEATURES` includes `normalized_macd_hist`, not `macd_hist`; tests assert raw `macd_hist` exclusion. | Raw `macd_hist` is not an active `mentor_clean_v1` feature. | current contract already satisfies Ian raw-feature exclusion | BLOCK any future raw `macd_hist` inclusion under unchanged `mentor_clean_v1`. |
| normalized MACD histogram or equivalent active normalized MACD feature | Allowed as the active MACD-family feature if route timing/normalization semantics stay fixed. | `MENTOR_CLEAN_V1_FEATURES` includes `normalized_macd_hist`; `_normalized_macd_hist_one_group` computes MACD and signal locally and returns `(macd - macd_signal) / close`. | Active feature is present and normalized by current close under completed-bar semantics. | current contract already satisfies Ian raw-feature exclusion; no-op test/spec assertion candidate | Keep formula/timing caveat. Any denominator, lag, or timing change is semantic feature drift and needs a separate protocol-change gate. |
| normalized volume feature | Allowed as the active volume-family feature if route timing/normalization semantics stay fixed. | `MENTOR_CLEAN_V1_FEATURES` includes `normalized_volume_20`; current implementation uses `log1p(volume)` minus a rolling mean through the completed current row. | Active feature is present; it is completed-bar/current-row, not strict prior-only. | current contract already satisfies Ian raw-feature exclusion; no-op test/spec assertion candidate | Keep current caveat. Changing to strict prior-only behavior under unchanged `mentor_clean_v1` is BLOCK and needs a protocol-change gate. |

## Stale-Doc Cleanup List

Cleanup is documentation/status-only. It must not change code, feature semantics, route locks, or evidence artifacts in this PM-052 task.

- `docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md`: lower sections still contain historical blocker/readiness text saying LightGBM or MS-DLinear+TCN paths were not found or were blocked. The top live-state notes and later PM-042/048/050/051 docs supersede that text.
- `stock_ml_knowledge_base/indexes/mentor_route_goal_status_dashboard_2026_05_30.md`: stale route-readiness rows still present missing-path blockers.
- `stock_ml_knowledge_base/indexes/mentor_clean_v1_project_path_audit_2026_05_30.md`: stale path-audit conclusions still say current route paths are missing.
- `stock_ml_knowledge_base/indexes/mentor_route_downloaded_only_reconciliation_2026_05_30.md`: stale language keeps missing route paths as active blockers.
- `stock_ml_knowledge_base/indexes/pm_shard_a_source_protocol_repair_reconciliation_2026_05_30.md`: stale language says LightGBM/MS-DLinear+TCN are not verified local routes.
- `stock_ml_knowledge_base/indexes/mentor_clean_v1_pm_best_route_execution_2026_05_30.md`: stale route-readiness rows remain.
- `stock_ml_knowledge_base/indexes/mentor_route_goal_completion_audit_2026_05_30.md`: stale readiness blockers remain.
- `stock_ml_knowledge_base/indexes/mentor_aligned_literature_gap_queue_2026_05_30.md`: stale language says `mentor_clean_v1` is not implemented and LightGBM is not found.
- `stock_ml_knowledge_base/indexes/mentor_clean_v1_literature_to_experiment_map_2026_05_30.md`: stale language says `mentor_clean_v1` is not implemented.

## No-Op Assertion Candidates

These are candidates only. PM-052 does not edit tests or code.

- Preserve or restate the existing assertion that `MENTOR_CLEAN_V1_FEATURES` excludes raw `open`, `high`, `low`, `close`, `volume`, `macd`, `macd_signal`, and `macd_hist`.
- Preserve or restate the metadata assertion that emitted `feature_columns` equals the `mentor_clean_v1` feature tuple.
- Add a future spec-only assertion that raw source columns may be required for derived feature computation while remaining forbidden as model inputs.
- Add a future spec-only assertion that post-bar-close same-bar features require `decision_time_policy=post_bar_close_completed_bar`.

## Protocol-Change Blockers

The following are not cleanup items. Each requires a separate protocol-change gate and likely a new feature-set identifier.

- Include raw OHLCV, raw volume, raw MACD, raw MACD signal, or raw MACD histogram in active model `feature_columns`.
- Change `normalized_volume_20` from completed-bar/current-row normalization to strict prior-only normalization under the unchanged `mentor_clean_v1` identifier.
- Change `normalized_macd_hist` denominator, lag, or timing semantics under the unchanged `mentor_clean_v1` identifier.
- Change from post-bar-close completed-bar prediction to any pre-close route without lagging/rebuilding same-bar features.
- Change threshold, label mode, scaler scope, decision-time policy, calendar boundaries, model capacity, hyperparameters, seed policy, or test/holdout access.
- Use validation diagnostics, Ian guidance, papers, or old notebook outputs as local evidence for model quality or feature selection.

## Recommendation

Raw-feature contract outcome: PASS with caveats. The current `mentor_clean_v1` active-column contract excludes raw OHLCV, raw volume, and raw MACD-family columns. Raw OHLCV/volume are source inputs, not model features. Raw MACD/signal/histogram are not active model features; MACD-family math is used only to produce `normalized_macd_hist`.

Next minimal PM gate:

`PM-FEATURE-ROUTE-PROTOCOL-COMMIT-052B -- commit accepted PM-052 raw-route contract audit`

After that commit gate, the next planning-only cleanup gate can be:

`PM-FEATURE-ROUTE-DOC-STALE-CLEANUP-SPEC-053 -- mark superseded route-readiness text without changing route semantics`

Do not start either gate from PM-052.

## Explicit Caveat

PM-052 performed no runtime, no code edits, no runner/test edits, no notebook execution, no training, no smoke/full validation, no model adjustment, no test/holdout access, no evidence-matrix or claim-map update, no Zotero/PDF/source-conversion/checkpoint work, no feature/threshold/label/scaler/decision-time/boundary/model-capacity/hyperparameter/seed change, no model selection from validation diagnostics, and no evidence promotion.
