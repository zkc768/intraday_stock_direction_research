# Notebook 07 Validation Synthesis And Gap Audit Technical Design - 2026-06-06

Scope: `validation_only` technical design.

Target notebook:

```text
notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb
```

This document designs Notebook 07 as a synthesis, robustness, explanation, validation-budget ledger, and gap-audit notebook over frozen Notebook 05 and Notebook 06 artifacts. It is not a new search notebook, not a model-selection notebook, not a selective-threshold search notebook, and not holdout/test authorization.

## Optimized Prompt

Design Notebook 07 for the intraday stock direction research route.

Inputs:

- active project rules in `AGENTS.md`;
- active workflow and Stage 0 freeze in `docs/RESEARCH_WORKFLOW.md` and `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`;
- Notebook 05 LightGBM tuning protocol and artifacts;
- Notebook 06 selective/no-trade protocol, technical design, and artifacts;
- research notes on 06/07 literature, calibration, robustness, explainability, permutation, null controls, and concentration guardrails.

Task:

Create a durable technical design for `notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb` that defines what 07 must read, what it may compute, what it must output, what it must not decide, and how it hands off unresolved gaps to 08X/08F/08O without becoming another validation search surface.

Failure modes to prevent:

- validation reuse disguised as fresh evidence;
- official-validation-best replacement of the train-inner Notebook 05 choice;
- model zoo expansion after seeing 05/06 validation results;
- selective threshold or coverage fishing;
- SHAP, permutation importance, ECE, AURC, or null-control outputs used as selection gates;
- concentration in one ticker, date, time bucket, or seed hidden by pooled results;
- same-row dummy baseline missing for final reported rows;
- holdout/test rows read, transformed, scored, summarized, plotted, or used for wording.

Output format:

- one linear Colab notebook with all `RUN_*` switches defaulting to `False`;
- tables and JSON manifests as primary artifacts;
- plots only as explanatory artifacts;
- every result row includes `scope`;
- every hard-stop reports the exact missing path, missing column, or invalid key.

Acceptance standard:

- 07 can support cautious paper wording about locked validation-only evidence;
- 07 cannot change model, label, feature, window, HPO profile, threshold, coverage point, or final holdout/test policy;
- 07 produces an explicit gateway memo for 08X/08F/08O instead of directly selecting exploratory candidates.

## Skills And Review Lenses Used

- Prompt optimization style: reformulated the user request into task, inputs, failure modes, output format, and acceptance standard before designing sections.
- `brainstorming-research-ideas`: applied problem-first framing, tension/contradiction, simplicity test, failure/boundary probing, and composition/decomposition. The resulting design treats 07's core problem as validation reuse risk, not lack of more models.
- `academic-pipeline`: classified 07 as a research-to-paper synthesis and integrity-boundary artifact. It sits before paper-ready claims and after validation-only model work; it is not final holdout evidence.
- `deep-research`: decomposed 07 by method, evidence, risk, and deliverable: final table, ledger, robustness, explainability appendix, permutation/null-control appendix, gap audit, and paper-ready synthesis.
- `academic-paper-reviewer`: simulated reviewer attacks around validation reuse, model zoo behavior, post-hoc explanation, threshold fishing, causal overclaim, concentration, and inadequate dummy baselines; defenses are written into gates and wording rules.

## Notebook 07 Scope And Non-Scope

In scope:

- read frozen Notebook 05 official-validation artifacts;
- read frozen Notebook 06 selective/no-trade artifacts if and only if 06 passed its artifact contract and decision record;
- produce a final validation-only comparison table over locked rows;
- reconcile metric names and dummy deltas from 05 and 06 into one comparison surface;
- produce a validation-budget ledger that records every search, confirmation, diagnostic, and decision surface;
- summarize per-ticker, per-seed, and concentration robustness;
- optionally add locked-model explainability, permutation-importance, and null-control appendices when required artifacts and dependencies already exist or are explicitly approved;
- write a gap audit that routes unresolved questions into 08X/08F/08O without choosing new candidates inside 07;
- write conservative paper-ready text with allowed and forbidden wording.

Non-scope:

- no new model family;
- no new HPO;
- no new feature subset, label horizon, no-trade threshold, coverage threshold, or window-size selection;
- no official-validation-best replacement;
- no calibration fitting on official validation;
- no holdout/test access, scoring, summary, plotting, or wording;
- no PnL, Sharpe, deployment, live-trading, or evidence-ready claim;
- no installation of SHAP, MAPIE, or other packages unless a later explicit operator approval changes the implementation boundary.

## Relationship Map

| Notebook | Role | Selection Surface | Output To 07 | Hard Boundary |
|---|---|---:|---|---|
| 02 | Stage 0 config screening from raw data | yes, validation-only Stage 0A | official Stage 0 candidate context | raw-data-first; no holdout/test |
| 03 | model-family screening after Stage 0 | yes, validation-only model-family screen | model-family context rows if generated | cannot reopen Stage 0 candidate |
| 04 | controlled follow-up / Exit A routing | diagnostic and route gate | 04D decision context for 05 | not final selection evidence |
| 05 | LightGBM train-inner HPO plus official-validation confirmation | train-inner HPO only; official validation confirms | required 05 artifacts and predictions | official-validation-best is not selected |
| 06 | selective/no-trade fixed-grid readout | no threshold search | optional frozen 06 artifacts | fixed coverage grid only |
| 07 | validation synthesis, ledger, robustness, explanation, gap audit | none | paper-ready validation-only synthesis and gap report | no new search degree of freedom |
| 08X | post-Stage-0 exploratory extension | exploratory only | future candidate ideas, not 07 claims | must be separately designed and labeled exploratory |
| 08F | future fixed follow-up / finalist confirmation route | only if pre-registered later | future fixed readout artifacts | cannot inherit 07 diagnostics as selectors |
| 08O | optional operational/out-of-sample or external validation planning route | none unless separately authorized | protocol and readiness notes | no hidden holdout/test reopening |
| 09 | manuscript / thesis evidence packaging | wording and reporting only | final tables, caveats, limitations | cannot revise methods based on paper needs |

07 is the bridge from validation-only notebooks to thesis wording. It does not generate the next candidate. It records why a next exploratory route may be needed.

## Pre-registration Constants Table

All numeric thresholds used by Notebook 07 are listed below for reader auditability.
Any change to a value here MUST be accompanied by a new freeze document and a new
notebook07_lockfile_scope_gate.json hash.

| constant | value | first appearance | source |
|---|---|---|---|
| improvement_threshold_delta_macro_f1_lcb_95 | 0.005 | §07B | AGENTS.md §4.2.5a |
| improvement_threshold_positive_ticker_count_min | 4 | §07B | AGENTS.md §4.2.5a |
| weak_signal_band_upper | 0.005 | §07B | this freeze |
| weak_signal_band_lower | 0.0 | §07B | this freeze |
| concentration_warning_top_ticker_share_max | 0.35 | §07D | this freeze |
| concentration_warning_positive_ticker_count_min | 4 | §07D | this freeze |
| weak_seed_evidence_count_threshold | 5 | §07D | this freeze |
| null_control_alpha_total | 0.05 | §07F | this freeze |

## Source Artifact Requirements

07 must start with `07A lockfile/scope gate`. The gate reads only metadata, manifests, CSVs, JSONs, and prediction/model artifacts needed for locked readout. Any missing or invalid artifact is a hard stop.

Required project documents:

```text
AGENTS.md
docs/RESEARCH_WORKFLOW.md
docs/CONFIG_SCREENING_FREEZE_2026-06-04.md
docs/BASELINE_REFERENCE.md
docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md
docs/research_notes/06_07_literature_materials_2026-06-05.md
docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md
docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md
```

Required Notebook 05 artifact directory:

```text
/content/notebook05_lightgbm_tuning_results/
```

Required Notebook 05 files:

```text
notebook05_entry_decision.json
notebook05_hpo_search_manifest.csv
notebook05_inner_fold_manifest.csv
notebook05_inner_hpo_results.csv
notebook05_inner_hpo_summary.csv
notebook05_finalists.csv
notebook05_official_validation_pooled.csv
notebook05_official_validation_per_ticker.csv
notebook05_official_validation_summary.csv
notebook05_decision_record.json
notebook05_run_manifest.json
```

Required Notebook 05 prediction artifacts when 07E/07F use prediction-level diagnostics:

```text
notebook05_probability_predictions_manifest.csv
predictions/*.npz
```

Required `.npz` arrays for prediction-level 07 sections:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence
```

Recommended Notebook 05 row fields:

```text
profile_id
profile_role
seed
label_config
horizon_k
threshold_bps
feature_set
window_size
scope
macro_f1
balanced_accuracy
accuracy
stratified_dummy_macro_f1
stratified_dummy_balanced_accuracy
delta_macro_f1_vs_stratified_dummy
always_up_dummy_macro_f1
delta_macro_f1_vs_always_up_dummy
train_n
validation_n
train_class0_n
train_class1_n
train_positive_rate
positive_ticker_count
top_ticker_gain_share
validation_sample_id_hash
prediction_artifact
run_failed
failure_reason
```

Optional Notebook 06 artifact directory:

```text
/content/notebook06_selective_no_trade_calibration_results/
```

Notebook 06 files allowed as frozen 07 inputs:

```text
notebook06_artifact_contract_check.json
notebook06_prediction_frame_manifest.csv
notebook06_probability_diagnostics.csv
notebook06_reliability_bins.csv
notebook06_coverage_grid.csv
notebook06_same_row_baselines.csv
notebook06_random_abstention_baselines.csv
notebook06_risk_coverage_summary.csv
notebook06_concentration_guardrails.csv
notebook06_per_ticker_coverage.csv
notebook06_decision_record.json
notebook06_run_manifest.json
```

Hard-stop examples:

- missing required path;
- `scope != "validation_only"`;
- `holdout_test_authorized != false`;
- `selective_threshold_selected != false`;
- `selected_profile_source` indicates official-validation-best replacement;
- prediction path contains `holdout` or `test`;
- required 05 row lacks same-row dummy metric;
- required 06 coverage row lacks same-row dummy and random-abstention comparison;
- sample-id hash mismatch;
- duplicate or missing `validation_sample_id`;
- missing `train_class0_n`, `train_class1_n`, or `train_positive_rate` when same-row stratified dummy reconstruction is needed.

## Proposed 07 Sections

### 07A - Lockfile And Scope Gate

Purpose:

- verify that 05 and optional 06 artifacts are frozen;
- verify that the official candidate remains `h03_bps1p5 + price_volume_time + window_size=20`;
- verify that every input artifact has `scope = validation_only`;
- verify that `holdout_test_authorized == false`;
- verify that `selective_threshold_selected == false`;
- verify that 07 will not read raw OHLCV or holdout/test rows.
- revalidate this gate at the entry of every subsequent `RUN_07*` phase; treat any artifact-hash change between phases as a hard stop until a new freeze is signed.

Output:

```text
notebook07_lockfile_scope_gate.json
```

The lockfile should include artifact paths, SHA-256 hashes, selected profile ids, candidate tuple, sample-id hashes, accepted run switches, and a `contract_passed` boolean. All hashes use canonical CSV form (utf-8, LF line endings, lexicographic column order, 6-decimal rounding for float columns) before sha256 to ensure cross-platform and cross-pandas-version stability; the lockfile records the `hash_input_normalization` recipe verbatim so future revalidations can reproduce the bytes that were hashed.

### 07B - Final Validation-Only Comparison

Purpose:

- build one normalized comparison table for locked 05 rows and frozen 06 fixed-grid rows;
- show full-coverage and selective rows without treating selective rows as threshold-selected winners;
- report same-row dummy deltas for every comparable row.

Recommended columns (grouped REQUIRED / RECOMMENDED / OPTIONAL; downstream
static tests assert REQUIRED columns present, RECOMMENDED columns present if
the corresponding row class exists, and ignore OPTIONAL):

```text
# REQUIRED — every row MUST have these:
artifact_source
notebook_stage
model
profile_id
profile_role
label_config
horizon_k
threshold_bps
feature_set
window_size
coverage
coverage_source
seed_count
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
balanced_accuracy_std
accuracy_mean
dummy_macro_f1_mean
dummy_balanced_accuracy_mean
delta_macro_f1_vs_dummy_mean
delta_balanced_accuracy_vs_dummy_mean
always_up_dummy_macro_f1_mean
delta_macro_f1_vs_always_up_dummy_mean
random_abstention_macro_f1_mean
delta_macro_f1_vs_random_abstention_mean
positive_ticker_count
top_ticker_gain_share
validation_n
retained_n
abstained_n
scope
decision_source
allowed_wording_tag

# RECOMMENDED — present when row class supports them
# (profile_id, profile_role, label_config, horizon_k, threshold_bps,
#  feature_set, window_size, macro_f1_std, balanced_accuracy_std,
#  dummy_balanced_accuracy_mean, delta_balanced_accuracy_vs_dummy_mean,
#  always_up_dummy_macro_f1_mean, delta_macro_f1_vs_always_up_dummy_mean,
#  diagnostic_only)
diagnostic_only

# OPTIONAL — selective / coverage rows only
# (coverage, coverage_source, retained_n, abstained_n,
#  random_abstention_macro_f1_mean, delta_macro_f1_vs_random_abstention_mean)
```

Interpretation:

- `delta_macro_f1_vs_dummy_mean <= 0`: no validation signal.
- `0 < delta_macro_f1_vs_dummy_mean < 0.005`: weak/small validation-only signal.
- `delta_macro_f1_vs_dummy_mean >= 0.005`: practical validation-only signal, not holdout/test evidence.
- concentration caveat applies if `positive_ticker_count < 4`, `top_ticker_gain_share > 0.35`, or 06 concentration guardrails trigger warning/severe status.

Decision rule per band (pre-registered; binds §07G gap routing and §07H wording):

| band | §07G gap category routed to 08X | §07H wording allowed (per AGENTS.md §4.2.5a) |
|---|---|---|
| `<= 0` | `metric_gap` + `generalization_gap` | "no detected validation-only signal" |
| `0 < x < 0.005` | `metric_gap` (exploratory) | "weak" or "mixed" — `improvement` is forbidden |
| `>= 0.005 AND positive_ticker_count >= 4` | `paper_wording_gap` only (if any) | `improvement` allowed |
| `>= 0.005 AND positive_ticker_count < 4` | `concentration_gap` | "weak / concentration-limited" — `improvement` is forbidden |

### 07C - Validation-Budget Ledger

Purpose:

- make validation reuse visible;
- separate train-inner decisions, official-validation confirmation, official-validation diagnostics, and post-hoc interpretation;
- prevent 07 from becoming an untracked selection layer.

Ledger columns:

```text
artifact
notebook_stage
decision_made
decision_timing
decision_surface
model_families_considered
profiles_or_trials_considered
seeds_used
thresholds_or_coverages_considered
official_validation_rows_inspected
cumulative_official_validation_inspections_across_notebooks
train_inner_only_decision
official_validation_informed_decision
diagnostic_only_readout
holdout_test_contact
allowed_wording
forbidden_wording
risk_note
appended_by_notebook
appended_at_utc
```

**Cross-notebook append-only rule**: `notebook07_validation_budget_ledger.csv`
is the project-level append-only ledger. Any downstream notebook (08X, 08F, 08O,
or future thesis chapters) that reads any official-validation metric MUST append
new rows to this ledger BEFORE the read, with `cumulative_official_validation_inspections_across_notebooks`
incremented. A read without prior append is a contract violation detected by
static gate.

07 should explicitly count at least:

- Stage 0 label, feature, window, model, and seed grid;
- Notebook 05 train-inner HPO trials, finalists, and official-validation rows;
- Notebook 06 fixed coverage-grid rows and random-abstention repeats;
- 07 explanation/permutation/null-control diagnostics if executed.

### 07D - Per-Ticker And Seed Robustness

Purpose:

- test whether pooled evidence hides one-ticker, one-seed, or one-time-region behavior;
- present robustness as diagnostic, not selection.

Diagnostics:

- per-ticker `delta_macro_f1_vs_dummy`;
- seed-level mean/std/LCB;
- matched-seed comparison where applicable;
- `positive_ticker_count`;
- `top_ticker_gain_share`;
- retained share by ticker for selective rows;
- concentration warnings from 06;
- note that sliding windows are overlapping and seeds are not independent market samples.

Recommended outputs:

```text
notebook07_per_ticker_robustness.csv
notebook07_seed_robustness.csv
notebook07_concentration_summary.csv
```

### 07E - Explainability Appendix

Purpose:

- explain locked LightGBM predictions;
- support paper interpretation without changing features.

Default behavior (appendix-only): on `RUN_07E_EXPLAINABILITY_APPENDIX = True`
the notebook MUST emit ONLY items 1-2 (split + gain importance). Promoting any
of items 3-5 to active output requires an explicit operator JSON override
recorded in `notebook07_decision_and_wording_record.json` under
`explainability_upgrade_record = { "promoted_items": [...], "reason": "..." }`,
plus a new `EXPECTED_DESIGN_DOC_SHA256` covering this revision. Items 3-5 are
appendix-only and never influence §07B / §07H wording.

Allowed low-dependency order:

1. LightGBM split importance.
2. LightGBM gain importance.
3. LightGBM `pred_contrib=True` if the locked model artifact and validation matrix are available.
4. SHAP/TreeSHAP only if `shap` is already available or explicitly approved later.
   SHAP approval gate (all three conditions required, recorded in
   `notebook07_decision_and_wording_record.json` before any SHAP call):
   (a) a new `notebook07_lockfile_scope_gate.json` signed with the SHAP package
       version pinned in `hash_input_normalization`;
   (b) an explicit operator acknowledgement
       `OPERATOR_ACKNOWLEDGES_SHAP_APPROVAL = True` plus the design-doc sha256
       of the revision that authorized SHAP;
   (c) a no-selection clause: SHAP outputs cannot retire features, add
       features, reweight features, or alter §07B / §07H wording.
5. Feature-group summary for price/return, volume, time-of-day, and technical-indicator groups.

Required caveats:

- feature importance is model-specific;
- correlated features can redistribute importance;
- SHAP values explain predictions under a specified background/perturbation assumption;
- no feature is added, removed, reweighted, or promoted from this appendix.

Recommended outputs:

```text
notebook07_lightgbm_importance_gain.csv
notebook07_lightgbm_importance_split.csv
notebook07_lightgbm_pred_contrib_summary.csv
notebook07_explainability_manifest.json
```

### 07F - Permutation / Null-Control Appendix

Purpose:

- stress-test whether the observed validation-only signal is larger than a chronology-aware diagnostic null;
- inspect feature reliance without feature reselection.

Alpha-spending policy (pre-registered; total alpha budget shared across all 07F reads):

- `null_control_alpha_total = 0.05` (see Pre-registration Constants Table).
- 07A lockfile MUST allocate `alpha_share_<design_name>` to each null design BEFORE any 07F read; the sum of allocations cannot exceed `null_control_alpha_total`; allocations cannot be revised after a 07F read.
- §07C ledger gains two virtual columns `alpha_consumed_after_row` and `alpha_remaining_after_row`; once `alpha_remaining` ≤ 0, no further null-control read is allowed even if a new design is proposed inside 07F.
- 08X may re-fund alpha only via a separately frozen 08F record (with a new alpha policy block), not from 07.
- the chosen null-design counts and allocations are frozen in `notebook07_lockfile_scope_gate.json.null_control_alpha_policy`.

Allowed permutation-importance design:

- locked model only;
- locked validation sample ids only;
- macro F1 and balanced accuracy only;
- feature-group permutation preferred over naive single-column permutation for correlated feature families;
- repeated with fixed seeds;
- report mean/std degradation;
- `scope = diagnostic`.

Allowed null-control designs from safest to riskiest:

1. read-only reporting of an existing pre-registered null-control artifact;
2. day-block or ticker-day block label permutation;
3. circular within-block shift;
4. feature-family permutation within ticker/day blocks.

Required null-control columns:

```text
null_design
permutation_unit
n_permutations
score
observed_score
null_score_mean
null_score_p95
p_value_one_sided
n_permutations
multiplicity_corrected
dependency_caveat
scope = diagnostic
```

Strict wording:

- Allowed: the observed validation-only delta was larger than the selected chronology-aware diagnostic null distribution under this design.
- Forbidden: the model is statistically proven to generalize; this passes holdout/test; this proves market profitability.

### 07G - Gap Audit For 08X

Purpose:

- convert 07 weaknesses into explicit future-work questions;
- document why 08X may be needed;
- avoid selecting any 08X candidate inside 07.

Gap categories:

```text
artifact_gap
metric_gap
concentration_gap
dependency_gap
explanation_gap
null_control_gap
generalization_gap
paper_wording_gap
```

Each gap row should include:

```text
gap_id
gap_category
evidence_source
observed_issue
why_it_matters
allowed_next_route
target_route ∈ {08X, 08F, 08O, none}  # 08X consumer asserts target_route=="08X" rows are routed
forbidden_in_07
minimum_pre_registration_needed
priority ∈ {must, useful, optional}
requires_extra_preregistration: bool  # was previously "risky"; now an explicit
                                        # boolean so static gates can check it
                                        # alongside priority instead of overloading
                                        # the priority enum
scope
```

Gateway rule:

- If the gap suggests a new feature, model, threshold, coverage, label, window, calibration method, or null design, the row goes to 08X/08F/08O planning. It does not alter 07 outputs.

### 07H - Paper-Ready Synthesis

Purpose:

- produce concise thesis-ready language that separates supported claims, caveats, and unresolved gaps.

Allowed wording template:

```text
Under the locked chronological validation-only route, the selected LightGBM
configuration produced [positive/mixed/no] delta over same-row stratified dummy
baselines across the five-ticker panel. Robustness diagnostics summarize
per-ticker, per-seed, concentration, and optional explanation/null-control
behavior. Because the official validation partition was reused for confirmation
and diagnostics, these results support cautious validation-only thesis wording,
not holdout/test, deployment, or profitability claims.
```

Forbidden wording:

```text
The model is final.
The model is holdout-ready.
The selective threshold is final.
The model is tradable or profitable.
SHAP proves the causal driver.
Permutation importance selects the final feature set.
ECE/AURC chooses the final threshold.
The 07 null-control appendix proves generalization.
```

Forbidden-phrase regex (07H MUST refuse to emit any paragraph that matches
the case-insensitive regex below; the kit's
`forbidden_phrases_blocked_at_runtime` list records every match it blocked):

```text
\b(final|production|deploy(?:ed|able|ment)?|tradable|live|sharpe|alpha)\b
```

This is a belt-and-suspenders layer over the explicit "Forbidden wording" list:
the explicit list catches exact phrases, the regex catches close variants
(e.g. "production-grade", "near-final", "ready for live", "alpha trading").

Falsification Conditions (pre-registered; recorded in
`notebook07_thesis_paragraph_kit.json.falsification_conditions`):

The paragraph kit MUST encode at least the following conditions. If any becomes
true under future N05/N06 same-candidate data, the corresponding paper claim is
no longer supported and an erratum is required — without rewriting the original
results:

- if one additional year of data drops `lcb_delta_macro_f1_vs_dummy` below 0.003 for the locked candidate, the `improvement` wording is retracted (per AGENTS.md §4.2.5a);
- if any single ticker's per-ticker delta becomes negative for two consecutive monitoring windows, the per-ticker positivity claim is retracted;
- if a future replication on a separately frozen feature set fails to clear the AGENTS.md §4.2.5a threshold, the generalization claim is downgraded to "specific to this feature set";
- if 06 concentration guardrails trigger severe status retroactively under updated bucket definitions (e.g., new date / time-of-day / volatility-regime buckets), the concentration caveat is upgraded to a hard limitation.

### 07J - Post-Publication Monitoring Plan

Purpose:

- record what 07 will monitor after paper publication so that the §07H falsification conditions are a concrete process, not just paper prose;
- 07J emits the plan; the actual monitoring runs happen in future sessions and are out of scope for 07 itself.

Plan:

- monitoring cadence: at least one re-run per calendar quarter for the first year after publication, then per year;
- monitoring scope per cycle: rerun §07B band comparison and §07D concentration on the same frozen N05 LightGBM candidate using newly arrived N05 / N06 same-candidate data; do NOT change the candidate, feature set, label, window, or threshold;
- monitoring outcome routing: if any §07H falsification condition becomes true under monitor data, the monitor MUST write an `erratum_recommended.json` next to the original paper artifacts; the original paper text is NOT retroactively edited (transparency erratum, not revision);
- monitoring is read-only diagnostic: it cannot select a new model, threshold, feature, or wording. Any change requires a new pre-registered protocol (08X / 08F / 08O or beyond);
- 07J does NOT authorize holdout/test access at any cycle.

Required outputs (when `RUN_07J_WRITE_MONITORING_PLAN = True`):

```text
notebook07_post_publication_monitoring_plan.json
notebook07_monitoring_cycle_<YYYY-MM-DD>.csv  (one per future cycle, append-only family)
notebook07_erratum_recommended.json           (only when a falsification condition triggers)
```

## Run Switches And Defaults

All switches default to `False`.

```python
RUN_07A_LOCKFILE_SCOPE_GATE = False
RUN_07B_FINAL_VALIDATION_COMPARISON = False
RUN_07C_VALIDATION_BUDGET_LEDGER = False
RUN_07D_ROBUSTNESS_AND_CONCENTRATION = False
RUN_07E_EXPLAINABILITY_APPENDIX = False
RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX = False
RUN_07G_GAP_AUDIT_FOR_08X = False
RUN_07H_PAPER_READY_SYNTHESIS = False
RUN_07I_BACKUP_TO_GOOGLE_DRIVE = False
RUN_07J_WRITE_MONITORING_PLAN = False
BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE = False
```

Operator acknowledgements should also default to `False`:

```python
OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH = False
OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False
OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS = False
OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH = False
EXPECTED_DESIGN_DOC_SHA256 = ""  # sha256 of the 2026-06-06 design at freeze time;
                                  # 07A asserts hashlib.sha256(open(__file__).read()) ==
                                  # EXPECTED_DESIGN_DOC_SHA256 before any RUN_07* may set True
```

If a run-all copy is ever created, it must be separately named and must not replace the canonical notebook.

## Output Artifact Manifest

Default output directory:

```text
/content/notebook07_validation_synthesis_and_gap_audit_results/
```

Required outputs:

```text
notebook07_lockfile_scope_gate.json
notebook07_input_artifact_manifest.csv
notebook07_final_validation_comparison.csv
notebook07_validation_budget_ledger.csv
notebook07_per_ticker_robustness.csv
notebook07_seed_robustness.csv
notebook07_concentration_summary.csv
notebook07_gap_audit_for_08x.csv
notebook07_paper_ready_synthesis.md
notebook07_thesis_paragraph_kit.json
notebook07_decision_and_wording_record.json
notebook07_run_manifest.json
```

`notebook07_thesis_paragraph_kit.json` schema (one-shot import into thesis chapter):

```text
{
  "results_paragraph": "<machine-filled from §07B band selection>",
  "robustness_paragraph": "<machine-filled from §07D>",
  "limitation_paragraph": "<machine-filled from §07G unresolved gaps>",
  "caveat_phrases_used": [...],
  "forbidden_phrases_blocked_at_runtime": [...],
  "reproducibility_pointers": [
    {"sentence_id": "<sha256-first-12-hex of normalized sentence>", "artifact": "<allowed path>", "row_filter": "<pandas query>", "expected_value_summary": "<short string or null>"}
  ],
  "reproducibility_pointer_rules": {
    "every_sentence_must_have_at_least_one_pointer": true,
    "every_pointer_sentence_id_must_be_unique_within_kit": true,
    "artifact_path_allowlist_globs": [
      "notebook07_*.csv",
      "notebook07_*.json",
      "notebook05_official_validation_*.csv",
      "notebook06_*.csv"
    ]
  },
  "improvement_wording_applied": <bool>,
  "improvement_threshold_check": {
    "delta_macro_f1_vs_dummy_lcb_95": <float>,
    "positive_ticker_count": <int>,
    "passed_per_AGENTS_md_4_2_5a": <bool>
  }
}
```

Optional outputs:

```text
notebook07_lightgbm_importance_gain.csv
notebook07_lightgbm_importance_split.csv
notebook07_lightgbm_pred_contrib_summary.csv
notebook07_permutation_importance_diagnostic.csv
notebook07_null_control_diagnostic.csv
notebook07_explainability_manifest.json
notebook07_drive_backup_manifest.json
```

Required schema fields for `notebook07_run_manifest.json`:

```text
scope
created_utc
notebook07_version
run_switches
operator_acknowledgements
input_artifacts
input_artifact_hashes
output_files
official_candidate
notebook05_result_dir
notebook06_result_dir
notebook05_decision_record_sha256
notebook05_run_manifest_sha256
notebook06_decision_record_sha256
notebook06_run_manifest_sha256
holdout_test_authorized
selective_threshold_selected
diagnostic_only_sections
gateway_to_08x
```

All CSV tables must include `scope`. Rows produced by 07E/07F must use `scope = diagnostic`; final comparison rows use `scope = validation_only`.

## Metrics And Decision / Wording Rules

Primary classifier metrics:

- macro F1;
- balanced accuracy.

Auxiliary metrics:

- accuracy;
- Brier score;
- ECE;
- AURC/E-AURC;
- log loss if already available;
- retained coverage and abstention counts;
- concentration metrics.

Baseline rules:

- Every full-coverage model row needs same-row stratified dummy and `delta_macro_f1_vs_dummy`.
- Every selective row needs same-row stratified dummy on retained rows.
- Every selective row should include ticker-stratified random abstention when available.
- `always_up_dummy` is auxiliary and cannot replace stratified dummy.

Seed aggregation:

- report mean, std, seed count, and one-sided 95% LCB for macro F1 and key deltas;
- if only one seed exists, LCB equals the mean and the row must be flagged as weak seed evidence;
- additionally, `seed_count < 5` for any final-reporting row sets `weak_seed_evidence = True` and the row's `allowed_wording_tag` MUST downgrade to "weak" (per AGENTS.md §4.2.5a);
- do not treat seed repeats over overlapping validation windows as independent market samples.

Concentration guardrails:

- report `positive_ticker_count`;
- report `top_ticker_gain_share`;
- report retained share by ticker for selective rows;
- include date/time/ticker concentration warnings from 06;
- downgrade wording if positive evidence is concentrated in fewer than four tickers or dominated by one ticker/date/time bucket.

Diagnostic-only metrics:

- ECE;
- Brier;
- AURC/E-AURC;
- SHAP/TreeSHAP;
- LightGBM split/gain importance;
- permutation importance;
- null-control empirical ranks;
- ablation-style summaries over already-approved artifacts.

These diagnostics can explain, caveat, or route future work. They cannot select a new candidate.

## Material Preparation Checklist

Before Notebook 07 implementation starts:

- Notebook 05 canonical or run-all Colab copy has produced the required result directory.
- `notebook05_decision_record.json` states `scope = validation_only`.
- `notebook05_decision_record.json` and `notebook05_run_manifest.json` state `holdout_test_authorized == false`.
- `notebook05_decision_record.json` and `notebook05_run_manifest.json` state `selective_threshold_selected == false`.
- Notebook 05 selected profile source is train-inner HPO or default fallback, not official-validation-best replacement.
- Notebook 05 official pooled/per-ticker/summary CSVs exist.
- Notebook 05 rows include same-row dummy deltas.
- Notebook 05 prediction artifacts include stable `validation_sample_id` if 07E/07F will use prediction-level diagnostics.
- Notebook 05 pooled rows include `train_class0_n`, `train_class1_n`, and `train_positive_rate` when same-row dummy reconstruction is needed downstream.
- Notebook 06 has been executed only if its artifact contract passed.
- Notebook 06 `notebook06_artifact_contract_check.json` exists and has `contract_passed == true`.
- Notebook 06 fixed coverage grid exists if selective rows are included in 07B.
- Notebook 06 decision record exists and states no final threshold was selected.
- No 07 section requires installing a dependency.
- If SHAP is desired, either `shap` is already available or the SHAP appendix is deferred.
- If null controls are desired, the exact diagnostic design is frozen before running 07F.

## What 07 Must Not Do

- Do not train a new model.
- Do not refit LightGBM.
- Do not run HPO.
- Do not add deep models.
- Do not add a model zoo.
- Do not search a feature subset.
- Do not choose a label, threshold, window, coverage point, or calibrator.
- Do not use official validation to replace Notebook 05's train-inner selection.
- Do not treat 06 selective rows as a final trading rule.
- Do not touch holdout/test.
- Do not read raw OHLCV unless a later implementation explicitly proves it is needed for locked-model explanation and still excludes holdout/test; default 07 should be artifact-first.
- Do not use SHAP, permutation importance, ECE, Brier, AURC, or null-control results as selection gates.
- Do not claim causal feature effects from SHAP or permutation importance.
- Do not report profitability, PnL, Sharpe, deployment readiness, or evidence-ready conclusions.

## Conversion / Gateway To 08X

07 may justify an 08X plan only through `notebook07_gap_audit_for_08x.csv`.

Allowed 07 gateway output:

- "A future 08X exploratory route should test whether [gap] can be addressed under a separately frozen design."
- "This gap cannot be resolved inside 07 because it would require a new search degree of freedom."
- "08F/08O require separate pre-registration before any new validation or external evidence is read."

Forbidden 07 gateway output:

- "08X should use candidate X because 07 diagnostic Y looked best."
- "SHAP ranked feature X first, so 08X should select it."
- "Coverage 0.60 looked best, so 08X should use it as final."
- "The null-control appendix passed, so holdout/test can be opened."

08X should be post-Stage-0 exploratory extension. 08F should be a later fixed follow-up if a candidate is frozen before readout. 08O should be an operational or out-of-sample planning route only if separately authorized. 07 only records the gateway.

## Reviewer Risk Audit And Defenses

| Reviewer Attack | Risk | 07 Defense | Static check ID |
|---|---|---|---|
| Validation reuse | 05/06/07 all inspect official validation | `notebook07_validation_budget_ledger.csv` counts every decision/readout and downgrades wording | `tests/test_notebook07_artifact_contract.py::test_ledger_append_only_monotonic` + `::test_ledger_rejects_regressing_cumulative_counter` |
| Model zoo | New models after seeing validation | 07A forbids new model families and requires frozen input artifacts | `tests/test_notebook07_static_gate.py::test_design_forbids_diagnostics_as_selection_gates` |
| Official-validation-best replacement | 05 confirmation becomes selection | 07A fails if `selected_profile_source` indicates official-validation-best | TBD (depends on `selected_profile_source` field landing per CN-005); planned `tests/test_notebook07_artifact_contract.py::test_lockfile_rejects_official_val_best_source` |
| Selective threshold fishing | 06 coverage curves choose threshold | 07 only reads fixed 06 coverage grid and keeps `selective_threshold_selected == false` | `tests/test_notebook06_artifact_contract.py` (existing) |
| Post-hoc explanation | SHAP/permutation used to change features | 07E/07F marked diagnostic-only; no feature changes allowed | `tests/test_notebook07_static_gate.py::test_design_forbids_diagnostics_as_selection_gates` |
| Causal overclaim | SHAP/permutation interpreted causally | required caveats say model-specific association/explanation only | `tests/test_notebook07_artifact_contract.py::test_thesis_kit_rejects_improvement_wording_without_lcb` |
| Concentration | pooled gain comes from one ticker/date/time | 07D requires per-ticker/seed/concentration tables and wording downgrade | `tests/test_notebook07_artifact_contract.py::test_thesis_kit_rejects_improvement_wording_without_breadth` |
| Dummy baseline insufficiency | model compared to pooled dummy only | every final row needs same-row stratified dummy; selective rows should include random abstention | `tests/test_notebook06_artifact_contract.py` (existing) |
| Null-control overclaim | diagnostic p-value treated as generalization | 07F wording forbids holdout/test/generalization claims | `tests/test_notebook07_static_gate.py::test_design_forbids_diagnostics_as_selection_gates` |
| Paper pressure | thesis narrative changes methods | 07H separates supported claims, caveats, and gaps without altering upstream choices | `tests/test_notebook07_artifact_contract.py::test_thesis_kit_rejects_claim_disagreement` |

## Static Tests / Validation Plan For Future Implementation

This design does not implement or execute Notebook 07. Future implementation should add tests before trusting generated notebook output.

Suggested static gate:

```text
tests/test_notebook07_static_gate.py
```

Required static checks:

1. Generated notebook parses as JSON.
2. All code cells AST-parse.
3. Outputs are empty.
4. Execution counts are `None`.
5. All `RUN_07*` switches default to `False`.
6. Operator acknowledgements default to `False`.
7. Active code contains `NOTEBOOK07_SCOPE = "validation_only"`.
8. Active code rejects holdout/test paths and flags.
9. Active code contains no active `drive.mount`.
10. Active code does not import `intraday_research`, `baseline_helpers`, stale notebooks, or project helper packages as Colab runtime dependencies.
11. Active code contains lockfile/scope gate checks.
12. Active code writes every required 07 output artifact.
13. Active code contains validation-budget ledger logic.
14. Active code contains same-row dummy requirements for final rows.
15. Active code contains forbidden-wording guardrails.
16. Active code contains no assignments or functions named `select_threshold`, `best_threshold`, `optimal_threshold`, `optimal_coverage`, `select_feature_subset`, `run_hpo`, or `train_new_model`.
17. Active code contains no `exec(`, `eval(`, `compile(`, `__import__(`, or `getattr(..., "select_*"|"best_*"|"optimal_*"|"run_hpo")` calls (defense against the item-16 list being bypassed via dynamic dispatch); the static gate AST-walks all code cells and rejects matches.

Suggested artifact-contract tests:

```text
tests/test_notebook07_artifact_contract.py
```

Required contract checks:

1. Minimal valid fake 05 bundle passes.
2. Missing 05 decision record fails with exact path.
3. `holdout_test_authorized == true` fails.
4. `selective_threshold_selected == true` fails.
5. Official-validation-best source fails.
6. Missing same-row dummy metric fails for a final comparison row.
7. Missing `validation_sample_id` fails when prediction-level diagnostics are requested.
8. Prediction path containing `holdout` or `test` fails.
9. Minimal valid fake 06 bundle can be included after `contract_passed == true`.
10. 06 bundle with `contract_passed == false` is excluded or blocks selective sections.
11. Final table preserves `scope`.
12. Diagnostic sections preserve `scope = diagnostic`.
13. Gap audit rows cannot set `selected_candidate`.
14. Wording record rejects forbidden phrases.

Use project Python only:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\create_validation_synthesis_and_gap_audit_colab_notebook.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook07_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook07_artifact_contract.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook07_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook07_artifact_contract.py
```

Do not execute the notebook as part of static validation.

## Acceptance Criteria

This technical design is acceptable when:

1. It is stored as one durable project-local Markdown artifact.
2. It keeps 07 validation-only and artifact-first.
3. It distinguishes 07 from 05, 06, 08X, 08F, 08O, and 09.
4. It specifies required 05/06 artifacts, schemas, and hard-stop conditions.
5. It defines 07A-07H sections.
6. It keeps all run switches default-off.
7. It defines output manifests and scope fields.
8. It preserves macro F1, balanced accuracy, same-row dummy deltas, random abstention where relevant, seed mean/std/LCB, and concentration guardrails.
9. It marks ECE, AURC, SHAP, permutation, null controls, and ablations as diagnostic-only.
10. It forbids new models, HPO, threshold search, feature reselection, holdout/test contact, and explanation-driven selection.
11. It creates a gateway to 08X without selecting an 08X candidate.
12. It includes reviewer-risk defenses and future static/artifact-contract tests.

## Source Anchors

Project-local anchors:

- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`
- `docs/BASELINE_REFERENCE.md`
- `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`
- `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md`
- `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md`
- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md`
- `docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md`
- `scripts/create_lightgbm_tuning_colab_notebook.py`
- `scripts/notebook06_contract.py`
- `scripts/create_selective_no_trade_calibration_colab_notebook.py`
- `tests/test_notebook05_static_gate.py`
- `tests/test_notebook06_static_gate.py`
- `tests/test_notebook06_artifact_contract.py`

Primary or official literature / documentation anchors already identified in the project notes:

- LightGBM paper: <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boost>
- Random search HPO: <https://jmlr.org/papers/v13/bergstra12a.html>
- Cawley and Talbot model-selection overfitting: <https://jmlr.org/papers/v11/cawley10a.html>
- SHAP paper: <https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>
- SHAP TreeExplainer documentation: <https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html>
- scikit-learn permutation importance: <https://scikit-learn.org/stable/modules/permutation_importance.html>
- Geifman and El-Yaniv selective classification: <https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks>
- scikit-learn probability calibration: <https://scikit-learn.org/stable/modules/calibration.html>
- Brier score documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html>
- Probability of Backtest Overfitting: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659>

## Task Report

Files inspected:

- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`
- `docs/BASELINE_REFERENCE.md`
- `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`
- `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md`
- `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md`
- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md`
- `docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md`
- current `notebooks/`, `scripts/`, and `tests/` file inventory related to 05/06/07, LightGBM, selective/no-trade, artifact contract, static gate, validation ledger, explainability, and robustness.

Files changed:

- `docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md`

Commands run:

- local memory lookup with `rg`;
- local `AGENTS.md` read;
- skill instruction reads for prompt optimization style, `brainstorming-research-ideas`, `academic-pipeline`, `deep-research`, and `academic-paper-reviewer`;
- `git status --short`;
- `git diff --stat`;
- `rg --files docs notebooks scripts tests`;
- `rg -n` inventory search for 05/06/07-related terms;
- project-Python read-only extraction of required document headings, key lines, protocol sections, and notebook/script/test inventories.

Validation results:

- This document is a static technical-design artifact only.
- No notebook was executed.
- No training was run.
- No dependency was installed.
- No holdout/test artifact was read, scored, summarized, or touched.
- No commit, push, or branch operation was performed.

Unresolved issues:

- Actual 07 implementation still needs a generator, static gates, and artifact-contract tests.
- 07 execution should wait until valid 05 and optional 06 result bundles exist and pass their contracts.
- SHAP should remain deferred unless the runtime already has `shap` or the operator explicitly approves dependency installation.
- Null controls require a separately frozen diagnostic design before execution.
