# Notebook 08 Deep Sequence Exploration, Freeze, And Readout Technical Design - 2026-06-06

Scope: `technical_design`

Result scope authorized by this document: `exploratory`, `diagnostic`,
`validation_only`

Holdout/test authorization: `false`

This document defines a future Notebook 08 route for aggressive deep sequence
exploration without corrupting the existing validation-only research route. It
does not authorize training in this task, notebook execution, dependency
installation, holdout/test access, commits, pushes, or changes to Notebooks
02-06.

The recommended route is:

```text
08X = aggressive deep sequence exploration lab / discovery / failure map
08F = freeze record / conversion gate / candidate compression
08O = official validation readout of one frozen candidate
```

The central rule is simple: 08X may discover candidates, but 08X results are
not official evidence. A candidate becomes eligible for official-validation
readout only after 08F freezes the architecture, preprocessing, loss, HPO
budget, seed protocol, metrics, wording rules, and artifact contract. 08O then
performs one frozen official-validation readout.

---

## 1. Optimized Prompt

### 1.1 Task

Design Notebook 08 as a post-Stage-0 extension that investigates whether
aggressive deep sequence models can produce defensible validation-only evidence
for the already frozen Stage 0 configuration:

```text
label_config = h03_bps1p5
feature_set = price_volume_time
window_size = 20
scope = validation_only
```

The route must turn exploratory deep sequence search into a paper-safe
validation-only readout by separating discovery, freeze, and readout:

```text
08X: train-inner exploration only
08F: candidate compression and freeze gate
08O: one frozen official-validation readout
```

### 1.2 Failure Modes To Prevent

- Model zoo fishing: many architectures are tried and only the best validation
  story is reported.
- Data snooping: official validation is repeatedly reused for architecture,
  loss, threshold, HPO, ensemble, or wording decisions.
- HPO budget unfairness: deep models receive far more budget than LightGBM,
  LogReg, or dummy baselines without transparent accounting.
- Loss/threshold fishing: focal loss, class-balanced loss, calibration, or
  selective coverage are tuned after seeing official validation.
- Complexity without benefit: a deep sequence model wins only by tiny,
  unstable, ticker-concentrated margins over simple baselines.
- Exploratory-to-official leakage: 08X best rows are written as paper evidence
  without an 08F freeze record.
- Backward narrative rewrite: 08X results are used to alter the meaning of
  Notebooks 02-06 or the Stage 0 candidate selection.
- Holdout/test contamination: closed holdout/test rows are read, transformed,
  windowed, scored, summarized, or used in wording.

### 1.3 Output Format

The future Notebook 08 route should produce machine-readable artifacts first,
and prose only from those artifacts:

```text
08X outputs: search space, trial ledger, fold results, failure map
08F outputs: freeze record, candidate compression table, paper_safe_score
08O outputs: official-validation readout, per-ticker readout, run manifest
```

Every row must carry `scope`, `candidate_id`, `stage`, `data_partition`,
`selection_role`, and `holdout_test_authorized`.

### 1.4 Acceptance Standard

Notebook 08 is acceptable only if:

1. 08X never reads official validation for selection.
2. 08F freezes exactly one primary candidate and at most one fallback candidate
   before 08O.
3. 08O reads official validation once for the frozen candidate only.
4. Same-row dummy baselines are present for all official-validation comparisons.
5. Pooled, per-ticker, seed mean/std/LCB, concentration, and failure rows are
   reported.
6. All run switches default to `False`.
7. The route keeps holdout/test closed.

---

## 2. Skill And Review Lenses Used

This design used these lenses:

| Lens | How it was used |
|---|---|
| Prompt optimization style | Rewrote the task into task, failure modes, output format, and acceptance criteria. No Arize or ax CLI was used. |
| `brainstorming-research-ideas` | Applied problem-first framing, tension/contradiction, simplicity test, failure/boundary probing, and composition/decomposition to split 08 into 08X/08F/08O. |
| `academic-pipeline` | Classified 08X as development, 08F as integrity checkpoint, and 08O as validation-only readout. None of the three is final holdout evidence. |
| `deep-research` | Structured the design by method, evidence, risk, deliverable, and source hierarchy. |
| `academic-paper` | Used structure/claim-evidence discipline so the document can serve as a durable project artifact. |
| `academic-paper-reviewer` | Simulated reviewer attacks around model zoo, data snooping, validation reuse, HPO unfairness, loss/threshold fishing, complexity, selective filtering, ticker concentration, and post-hoc storytelling. |

---

## 3. Source Materials Inspected

### 3.1 Required Project Documents

The design is based on these local source artifacts:

```text
AGENTS.md
docs/RESEARCH_WORKFLOW.md
docs/CONFIG_SCREENING_FREEZE_2026-06-04.md
docs/BASELINE_REFERENCE.md
docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md
docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md
docs/research_notes/06_07_literature_materials_2026-06-05.md
docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md
docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md
```

The 07 technical design was present by final validation and was read after the
initial 08 draft. It confirms that Notebook 07 is an artifact-first 05/06
synthesis, validation-budget ledger, robustness/explainability appendix, and
gap audit. It is not a new model-selection stage. It may route unresolved gaps
to 08 only through a gateway such as `notebook07_gap_audit_for_08x.csv`; it
cannot select 08 candidates, choose thresholds, authorize holdout/test, or turn
diagnostics into official evidence.

### 3.2 Current Related File Inventory

Relevant current notebooks:

```text
notebooks/03_model_family_screening_colab.ipynb
notebooks/04_controlled_followup_colab.ipynb
notebooks/04_ian_research_memo.ipynb
notebooks/05_lightgbm_tuning_colab.ipynb
notebooks/05_lightgbm_tuning_colab_run_all.ipynb
notebooks/05_06_chained_validation_colab.ipynb
notebooks/05_06_chained_validation_colab_resume06.ipynb
notebooks/06_selective_no_trade_calibration_colab.ipynb
notebooks/evidence/feature_selection_2026-06-02/results/
```

Relevant current scripts:

```text
scripts/create_config_screening_colab_notebook.py
scripts/create_model_family_screening_colab_notebook.py
scripts/create_diagnostic_h0_tabular_sweep_colab_notebook.py
scripts/create_controlled_followup_colab_notebook.py
scripts/create_lightgbm_tuning_colab_notebook.py
scripts/create_selective_no_trade_calibration_colab_notebook.py
scripts/notebook06_contract.py
scripts/extract_stage0_notebook_outputs.py
scripts/review_extracted_stage0_outputs.py
```

Relevant current tests:

```text
tests/test_validation_pipeline.py
tests/test_notebook_static_gate.py
tests/test_notebook03_static_gate.py
tests/test_notebook04_static_gate.py
tests/test_notebook05_static_gate.py
tests/test_notebook06_static_gate.py
tests/test_notebook06_artifact_contract.py
```

08 should inherit the static-gate style used by 03-06: notebook parses, code
cells AST-parse, outputs are empty, execution counts are `None`, default run
switches are inert, forbidden strings are absent, and artifact manifests are
schema-checked before runtime trust.

---

## 4. Notebook 08 Route Scope

### 4.1 08X - Aggressive Deep Sequence Exploration Lab

Scope:

- Explore whether deep sequence variants have any train-inner evidence worth
  freezing.
- Use only the official training partition, split into inner chronological
  folds.
- Produce a failure map, not a paper result.
- Record every trial, skipped trial, failed trial, seed, fold, config, runtime
  budget, and stop reason.
- Compare all candidates against simple baselines on the same inner-fold rows.

Non-scope:

- No official validation model selection.
- No official-validation threshold selection.
- No holdout/test access.
- No paper claim that a deep model beats LightGBM.
- No replacement of Stage 0 candidate selection.

Allowed result language:

```text
08X found exploratory train-inner evidence that candidate <id> is worth freezing for 08F review.
```

Forbidden result language:

```text
08X proves the deep model is better.
08X selects the official model.
08X validates the sequence route.
```

### 4.2 08F - Freeze Record / Conversion Gate / Candidate Compression

Scope:

- Read only 08X artifacts.
- Reject invalid, under-budgeted, unstable, or overfit-looking candidates.
- Compress the exploration set into exactly one `primary_candidate` and at most
  one `fallback_candidate`.
- Freeze architecture, feature input, window size, loss, class weighting, HPO
  result, seeds, official-validation readout plan, metrics, wording rules, and
  artifact schemas.
- Write a signed freeze record before 08O is allowed.

Non-scope:

- No official-validation read.
- No new architecture search.
- No editing 02-06 decisions.
- No selection based on post-hoc visual appeal.

### 4.3 08O - Official Validation Readout

Scope:

- Read the 08F freeze record.
- Rebuild the frozen candidate from raw-data-first logic or from an audited
  frozen train-inner artifact contract.
- Train/refit only as specified in the freeze record.
- Score official validation exactly once for the frozen primary candidate.
- Score fallback only if 08F pre-registers an explicit primary failure path
  that occurs before official-validation metrics are inspected.
- Report pooled, per-ticker, per-seed, same-row dummy deltas, concentration,
  and failure rows.

Non-scope:

- No official-validation HPO.
- No threshold fishing.
- No selective/no-trade operating threshold selection unless the threshold was
  frozen before 08O.
- No holdout/test.
- No feature reselection or label reselection.
- No claim that 08O is final evidence-ready holdout performance.

---

## 5. Relationship Map

| Stage | Role | 08 relationship |
|---|---|---|
| 02 | Raw-data-first Stage 0 configuration screening. Selects at most two `label_config + feature_set + window_size` candidates. | 08 must inherit the frozen Stage 0 candidate, not rerun or reinterpret 02. |
| 03 | Model-family screening on Stage 0 candidates. | 08 is later than 03 and should not backfill 03 selection. |
| 04 | Controlled follow-up and operator routing, including fresh-seed and selective diagnostics around the fixed candidate. | 08 may use 04 as context for gaps, not as permission to rerank via 08X. |
| 05 | LightGBM train-inner tuning plus official-validation confirmation. | 08 should mirror 05's train-inner HPO versus official-validation readout separation. |
| 06 | Selective/no-trade calibration and coverage diagnostics over frozen 05 probability artifacts. | 08 may reuse 06 diagnostic vocabulary, but must not select thresholds on official validation. |
| 07 | Frozen 05/06 synthesis, validation-budget ledger, robustness/explainability appendix, and gap audit. | 07 is not model selection. It can motivate 08 as a gap response only through a recorded gateway, but cannot select 08 candidates or authorize 08O directly. |
| 08X | Deep sequence exploration lab. | Development and failure mapping only. |
| 08F | Freeze/conversion gate. | Integrity checkpoint converting exploration into one frozen candidate. |
| 08O | Official-validation readout. | Validation-only evidence for the frozen candidate, not final holdout evidence. |
| 09 | Possible later manuscript/result synthesis. | 09 can summarize 08O only with validation-only wording and all caveats. |

Notebook 08 is a post-Stage-0 extension and gap response. It must not倒灌 08X
results into the narrative of 02-06.

Note on naming: the 07 technical design uses 08X/08F/08O as a future gateway
vocabulary, with 08O described generically as a later operational or external
planning route. This 08 document uses the user-requested narrower definition:
`08O = official validation readout of one frozen candidate`. This is still
`validation_only`; it does not authorize holdout/test or external validation.

---

## 5.5. Pre-registration Constants Table

All numeric thresholds used by Notebook 08 are listed below for reader auditability.
Any change MUST be accompanied by a new freeze document and a new 08x_search_space.json
sha256 stamp.

| constant | value | first appearance | source |
|---|---|---|---|
| improvement_threshold_delta_macro_f1_lcb_95 | 0.005 | §10.4 | AGENTS.md §4.2.5a |
| improvement_threshold_positive_ticker_count_min | 4 | §10.4 | AGENTS.md §4.2.5a |
| fusion_min_lcb_advantage_over_components | 0.003 | §7.4 | this freeze |
| candidate_eligibility_min_train_inner_lcb_delta | 0.005 | §9.1 | this freeze |
| paper_safe_score_weight_lcb_delta | 0.35 | §9.2 | this freeze |
| paper_safe_score_weight_mean_delta | 0.20 | §9.2 | this freeze |
| paper_safe_score_weight_seed_stability | 0.15 | §9.2 | this freeze |
| paper_safe_score_weight_fold_consistency | 0.10 | §9.2 | this freeze |
| paper_safe_score_weight_per_ticker | 0.10 | §9.2 | this freeze |
| paper_safe_score_penalty_complexity | -0.05 | §9.2 | this freeze |
| paper_safe_score_penalty_compute | -0.05 | §9.2 | this freeze |
| class_collapse_pred_rate_min | 0.05 | §14.4 | this freeze |
| total_trial_budget_cap_across_all_families | 250 | §11 | this freeze |

## 6. Source Artifact Requirements

### 6.1 08X Inputs

08X should start from:

```text
docs/CONFIG_SCREENING_FREEZE_2026-06-04.md
docs/RESEARCH_WORKFLOW.md
notebooks/02_config_screening_colab.ipynb
scripts/create_config_screening_colab_notebook.py
notebooks/03_model_family_screening_colab.ipynb
scripts/create_model_family_screening_colab_notebook.py
notebooks/04_controlled_followup_colab.ipynb
scripts/create_controlled_followup_colab_notebook.py
notebook07_gap_audit_for_08x.csv (REQUIRED if 07G has emitted gap rows)
```

For each row in `notebook07_gap_audit_for_08x.csv` whose `target_route == "08X"`,
08X MUST either (a) include a corresponding entry in `08x_search_space.json`
under a `routed_from_07g_gap_id: <gap_id>` field, or (b) record a documented
deferral with `08x_search_space.json.deferred_07g_gaps[<gap_id>] = "<reason>"`.
08F MUST refuse to freeze a candidate that silently ignores a 07G-routed gap
without either route entry or deferral.

08X MAY read `notebook05_decision_record.json` and `notebook06_decision_record.json`
to extract `{label_config, feature_set, window_size, candidate_id}` as the
frozen Stage 0 anchor. 08X MUST NOT read `notebook05_official_validation_*.csv`
or `notebook06_*_coverage.csv` for selection; those are official-validation
artifacts.

Minimum frozen candidate fields:

```text
label_config
horizon_k
threshold_bps
feature_set
window_size
tickers
train/validation boundary dates
closed holdout/test boundary marker
same-day window rule
label-horizon invalidation rule
train-only preprocessing rule
dummy baseline rule
```

Preferred 08X implementation posture:

- raw-data-first and self-contained in Colab;
- no active import from `intraday_research`, prior notebooks, or stale helper
  packages;
- duplicate only the safety-critical feature, label, split, and window logic
  needed by Notebook 08;
- all heavy cells guarded by `RUN_08X_* = False` by default.

### 6.2 08F Inputs

08F requires completed 08X outputs:

```text
08x_search_space.json
08x_trial_ledger.csv
08x_fold_results.csv
08x_failure_ledger.csv
08x_seed_summary.csv
08x_candidate_compression_table.csv
08x_run_manifest.json
```

08F must refuse to freeze if any 08X output is missing, lacks schema fields,
contains official-validation selection columns, or omits failed/skipped trials.

### 6.3 08O Inputs

08O requires:

```text
08x_run_manifest.json
08x_trial_ledger.csv
08x_candidate_compression_table.csv
08f_candidate_freeze_record.json
08f_candidate_freeze_record.md
08f_static_gate_report.json
```

08O must verify these before scoring:

```text
scope == validation_only
holdout_test_authorized == false
official_validation_used_for_selection == false
candidate_id matches 08F primary_candidate_id
feature/label/window config matches active Stage 0 freeze
run switches match the freeze record
```

---

## 7. 08X Exploration Search Space

08X is allowed to be aggressive only inside train-inner folds. It should be
explicitly finite and logged before execution.

### 7.1 Architecture Families

Candidate families:

```text
ms_dlinear_tcn
dlinear_only
tcn_only
shallow_gru
shallow_lstm
last_step_mlp_sequence_ablation
last_step_lightgbm_control
```

Deep families are compared against controls, not assumed superior.

### 7.2 DLinear Parameters

Allowed DLinear search axes:

```text
moving_avg_kernel: 3, 5, 7, 11
individual_channels: false, true
linear_head: shared, per_channel
seasonal/trend dropout: 0.0, 0.05, 0.10
input_projection: none, linear_bottleneck
```

The DLinear role is decomposition and linear sequence bias. It cannot justify a
complex architecture unless it beats `dlinear_only` and `last_step` controls
under the same budget.

### 7.3 TCN Parameters

Allowed TCN search axes:

```text
num_blocks: 2, 3, 4
channels: [16,16], [32,32], [32,32,32], [64,32,16]
kernel_size: 2, 3, 5
dilation_base: 2
dropout: 0.0, 0.05, 0.10, 0.20
residual: true
gating: false, true
normalization: none, weight_norm, layer_norm
causal: true
head: last_step, attention_pooling_pre_frozen
```

Any non-causal convolution, future-aware padding, or window that crosses ticker
or trading-day boundaries is forbidden.

### 7.4 Fusion Variants

Allowed fusion variants:

```text
dlinear_trend_plus_tcn_residual
dlinear_logits_plus_tcn_logits
late_average_probabilities
small_fusion_mlp
```

Fusion must be justified by train-inner results against both components alone.
Concretely: `lcb_delta_macro_f1` of the fused candidate vs the better of
{`dlinear_only`, `tcn_only`} on the same train-inner folds MUST exceed
`fusion_min_lcb_advantage_over_components` (see §5.5 Pre-registration
Constants Table). If fusion wins only by increasing complexity with unstable
seeds or by less than this threshold, 08F should reject it.

### 7.5 Loss And Class-Imbalance Variants

Allowed losses, if predeclared:

```text
cross_entropy
weighted_cross_entropy_train_prior
focal_loss
class_balanced_loss_effective_number
balanced_softmax
```

Loss variants are diagnostic unless train-inner selection was predeclared.
Official validation may not be used to pick a loss.

### 7.6 HPO Methods

Allowed HPO methods:

```text
random_search
tpe
successive_halving
asha
hyperband
bohb
```

PBT is risky in this project because it mutates training policy during training
and can obscure trial accounting. It should remain optional/research-only unless
08X writes full lineage records for exploit/explore events, inherited weights,
and schedule mutations.

### 7.7 Ensembles

Allowed only if frozen before official validation:

```text
seed_mean_probability_ensemble
top_k_train_inner_lcb_ensemble
snapshot_ensemble_with_predeclared_epochs
```

Forbidden:

```text
ensemble chosen from official-validation ranking
snapshot chosen from official-validation curve
seed excluded after official-validation failure
```

### 7.8 Calibration And Selective Diagnostics

08X may compute diagnostic calibration/selective curves on train-inner validation
folds:

```text
brier_score
log_loss
ece_uniform_bins
ece_quantile_bins
risk_coverage_curve
aurc
fixed_coverage_grid
same_row_dummy_at_coverage
random_abstention_at_coverage
```

These are diagnostic-only unless 08F freezes a specific use. ECE/AURC cannot be
turned into selection gates after results are seen.

**08F MUST NOT consume any 08X §7.8 calibration/selective diagnostic as an
input to `paper_safe_score` (§9.2) or to the candidate eligibility check
(§9.1).** If a future design wants to use calibration/selective metrics for
candidate compression, that requires a separately frozen 08F protocol and a
new freeze record; the current freeze does not authorize it. This isolates
the "compute on train-inner" surface from the "decide which candidate wins"
surface so that selective fishing cannot leak from §7.8 into §9.2.

---

## 8. 08X Validation Design

### 8.1 Data Partition

08X may use only the official training partition. It must create train-inner
folds with chronological ordering:

```text
outer official train partition
  -> train_inner_fit rows
  -> train_inner_validation rows
official validation partition
  -> not read by 08X
closed holdout/test
  -> boundary marker only, never read/scored
```

### 8.2 Fold Design

Recommended fold modes:

```text
rolling_origin_folds
purged_time_series_folds
embargoed_train_inner_folds
```

Requirements:

- split per ticker chronologically before pooling;
- no fold may train on a label horizon that overlaps its inner-validation
  interval;
- no input window may cross trading-day boundaries;
- no input window may cross ticker boundaries;
- preprocessing statistics fit on train-inner-fit rows only;
- train-inner validation rows may be scored only for 08X discovery.

### 8.3 Trial Ledger

Every trial should produce one row even when it fails:

```text
trial_id
candidate_family
candidate_id
config_hash
fold_id
seed
budget_tier
max_epochs
actual_epochs
early_stop_reason
fit_status
failure_type
failure_message
train_inner_fit_n
train_inner_validation_n
macro_f1
balanced_accuracy
accuracy
stratified_dummy_macro_f1_same_rows
delta_macro_f1_vs_dummy
class0_pred_rate
class1_pred_rate
ticker_max_share
actual_wall_clock_seconds
peak_memory_mb
gpu_seconds_or_null
scope = exploratory
official_validation_used = false
holdout_test_authorized = false
```

The `actual_wall_clock_seconds`, `peak_memory_mb`, and `gpu_seconds_or_null`
fields are required so the `compute_penalty` term in §9.2 can be reconstructed
from the ledger without trusting hand-edited summaries; `gpu_seconds_or_null`
may be `null` on CPU-only runs.

### 8.4 Failure Map

08X should treat failures as evidence:

```text
class_collapse
unstable_seed_variance
ticker_concentration
date_concentration
time_of_day_concentration
training_divergence
timeout
memory_error
artifact_schema_failure
official_validation_boundary_violation
insufficient_same_row_dummy
no_improvement_over_last_step_control
```

---

## 9. 08F Candidate Compression Gate

### 9.1 Candidate Eligibility

08F MUST be operated in a separate Colab session by a non-08X-author, OR with
`dmc_attestation.json` present in the 08F input directory containing:

```text
{
  "dmc_role": "data_monitoring_committee_proxy",
  "reviewer_identifier": "<name or stable id>",
  "reviewed_08x_run_manifest_sha256": "<sha256 of 08x_run_manifest.json>",
  "reviewed_at_utc": "<ISO 8601>",
  "attestation_statement": "<short text>"
}
```

08F MUST refuse to proceed if neither condition holds.

A candidate may enter 08F only if:

1. It has complete trial rows for all required folds and seeds, or explicit
   failure rows for missing runs.
2. It beats same-row stratified dummy on train-inner validation by a predeclared
   margin.
3. It beats the simple sequence controls under the same budget.
4. It avoids class collapse.
5. It avoids severe ticker concentration.
6. It has no official-validation selection contact.
7. It has a reproducible config hash and frozen implementation path.

### 9.2 Candidate Scoring Rule

08F should calculate a transparent `paper_safe_score`, not just pick the highest
mean metric:

```text
paper_safe_score =
    0.35 * lcb_delta_macro_f1_vs_dummy
  + 0.20 * mean_delta_balanced_accuracy_vs_dummy
  + 0.15 * seed_stability_score
  + 0.10 * fold_consistency_score
  + 0.10 * per_ticker_consistency_score
  - 0.05 * complexity_penalty
  - 0.05 * compute_penalty
```

Suggested penalty definitions:

```text
seed_stability_score = max(0, 1 - seed_std_macro_f1 / seed_std_scale)
   where seed_std_scale = 2.0 * median(N06_per_candidate_seed_std_macro_f1)
   (data-driven; computed once over N06 frozen artifacts before 08X starts,
    sha256-stamped into 08x_search_space.json; falls back to 0.02 only if
    N06 unavailable and the fallback is recorded in the freeze record)
fold_consistency_score = share of folds with positive delta_macro_f1_vs_dummy
per_ticker_consistency_score = share of tickers with non-negative delta
complexity_penalty = z(log10(parameter_count)) + z(ensemble_size) + z(fusion_depth)
   where z = z-score normalization computed over all 08X completed trials in
   the current run; pre-Freeze constants are recorded in 08x_search_space.json
compute_penalty = z(actual_wall_clock_seconds) + z(failed_trial_count)
   same z-score scope as complexity_penalty; uses §8.3 trial-ledger fields directly
```

The constants above are defaults for a future implementation plan. If changed,
08F must freeze the replacement before reading 08X compression outputs.

### 9.3 Primary And Fallback

08F writes:

```text
primary_candidate_id
fallback_candidate_id or null
fallback_activation_rule
```

The fallback may be used in 08O only if the primary fails before official
validation metrics are inspected, for example:

```text
primary implementation cannot reproduce train-inner checksum
primary model fails deterministic shape/static gate
primary training produces NaN before scoring official validation
primary artifact contract fails before official validation is read
```

Fallback cannot be activated because the primary official-validation metrics are
weak.

**Definition of "inspected"** (binds §9.3 fallback rule and §10.2 readout): a
field counts as inspected the moment any of the following happens:

- any of `macro_f1`, `balanced_accuracy`, `accuracy`, or any `delta_*` column
  is read from `08o_validation_readout.csv` / `08o_validation_per_ticker.csv`
  / `08o_seed_summary.csv` into memory by user code (including pandas read,
  cell display, `print`, logging, `describe()`, plotting, or assignment to a
  variable subsequently referenced);
- any aggregate of the above (mean, std, LCB, percentile, comparison) is
  computed;
- the operator visually reads the value from a notebook output or downstream
  artifact.

Reading the file's row count, schema, or `validation_n` alone does NOT count
as inspection. Loading `notebook07_validation_budget_ledger.csv` to append the
08O intent row does NOT count as inspection. The fallback decision MUST be
finalized before any inspection event above.

### 9.4 Hard-Stop Criteria

08F must stop and write `08f_no_candidate_freezable` if:

- no candidate has positive train-inner LCB delta versus same-row dummy;
- deep candidates do not beat `last_step_lightgbm_control` or
  `last_step_mlp_sequence_ablation`;
- best candidates have severe ticker concentration;
- best candidates depend on official-validation feedback;
- loss/threshold/ensemble choices were changed after train-inner results
  without a predeclared rule;
- failed/skipped trials are missing from the ledger;
- the candidate cannot be reproduced from a config hash;
- the candidate would require holdout/test wording to sound meaningful.

---

## 10. 08O Official Validation Readout

### 10.1 Entry Gates

08O may start only when:

```text
08f_candidate_freeze_record.json exists
08f_candidate_freeze_record.md exists
08f_static_gate_report.json says pass
holdout_test_authorized == false
official_validation_used_for_selection == false
RUN_08O_OFFICIAL_VALIDATION_READOUT == True by explicit operator edit
```

### 10.2 Official-Validation Readout Rules

08O must:

0. Load `notebook07_validation_budget_ledger.csv`; append a new row recording the
   08O intent (cumulative_official_validation_inspections_across_notebooks += 1)
   BEFORE reading any official-validation metric. Verify the cumulative count
   does not exceed the project-level cap (if declared in AGENTS.md or freeze doc).
   Static gate MUST refuse 08O if the ledger append did not happen.
1. Read only the official validation rows defined by the active project split.
2. Build samples with the same label, feature, window, and boundary rules as the
   active Stage 0 route.
3. Fit preprocessing on training rows only.
4. Fit the frozen candidate exactly as specified by 08F.
5. Score official validation once per frozen seed.
6. Compute same-row stratified dummy and always-up dummy baselines.
7. Write pooled, per-ticker, per-seed, and seed-summary outputs.
8. Write failure rows instead of silently dropping failed seeds.

### 10.3 Metrics

Primary metrics:

```text
macro_f1
balanced_accuracy
delta_macro_f1_vs_stratified_dummy_same_rows
delta_balanced_accuracy_vs_stratified_dummy_same_rows
```

Required supporting metrics:

```text
accuracy
confusion matrix counts
validation_n
train_class0_n
train_class1_n
train_positive_rate
class0_pred_rate
class1_pred_rate
seed_mean
seed_std
seed_lcb_95
per_ticker_macro_f1
per_ticker_delta_macro_f1_vs_dummy
ticker_max_share
date/time concentration diagnostics
```

Diagnostic-only metrics:

```text
brier_score
log_loss
ece_uniform_bins
ece_quantile_bins
aurc
risk_coverage_curve
fixed_coverage_grid
permutation importance
SHAP/pred_contrib
```

Diagnostics must not become post-hoc selection gates.

### 10.4 Allowed Wording

Strongest allowed positive wording:

```text
Under a pre-frozen validation-only Notebook 08 protocol, the frozen deep sequence
candidate showed validation-only improvement over same-row dummy baselines on
the official validation partition, with the reported seed variance, per-ticker
distribution, and concentration caveats. This is not holdout/test evidence and
does not revise the Stage 0 selection narrative.
```

If weak/mixed:

```text
The frozen deep sequence candidate did not provide stable validation-only
evidence beyond simple baselines. 08X remains useful as a failure map, but 08O
does not support a stronger deep sequence claim.
```

Forbidden wording:

```text
The deep model is validated on holdout.
The official validation set selected the best architecture.
08X proves deep sequence models work.
The model is tradable/deployable/profitable.
The 08 result invalidates the 02-06 route.
```

---

## 11. Budget Tiers

These are planning estimates, not runtime claims. Actual time depends on Colab
hardware, data size, and implementation efficiency.

| Tier | Intended use | Approx configs/fits | Approx compute | Main failure modes |
|---|---:|---:|---|---|
| quick | Static smoke plus tiny train-inner feasibility | 4-8 configs, 1-2 folds, 1-2 seeds | minutes to under 1 hour | misleading variance, too little search, shape bugs hidden |
| medium | Candidate-family comparison | 20-40 configs, 3 folds, 3 seeds | a few GPU/CPU hours | HPO budget still small, unstable deep candidates |
| aggressive | Deep sequence discovery and failure map | 80-200 configs, 3-5 folds, 5 seeds | many hours to multi-session Colab | model zoo risk, failed trial accounting, compute bias |
| paper-safe subset | Freeze-eligible subset after 08X | 1 primary + 1 fallback, fixed folds/seeds | bounded by 08F | overfitting to train-inner if compression rule weak |

Budget accounting must count failed, timed-out, skipped, and early-stopped trials.
Deep models must not receive unreported hidden budget.

### 11.1 Tier Escalation Rule

A higher tier MUST NOT start without a pre-registered escalation check from the
prior tier:

| Current tier completed | Escalation allowed only if |
|---|---|
| quick | best deep candidate train-inner `lcb_delta_macro_f1` vs `last_step_lightgbm_control` >= 0.003 AND positive on >= 4 tickers |
| medium | additionally, seed_std_macro_f1 <= 0.01 on the leading candidate |
| (aggressive done) | (no further escalation; proceed to 08F or 08f_no_candidate_freezable.json) |

If the escalation check fails, 08X stops and writes `08x_tier_escalation_blocked.json`
documenting the gate value; 08F proceeds with the candidates from the highest
completed tier or writes `no_candidate_freezable`. Skipping an escalation check
is a contract violation.

---

## 12. Run Switches And Defaults

All future Notebook 08 switches default to `False`:

```python
RUN_08X_SCHEMA_SMOKE = False
RUN_08X_BUILD_TRAIN_INNER_FOLDS = False
RUN_08X_SEARCH_SPACE_DRY_RUN = False
RUN_08X_QUICK_SEARCH = False
RUN_08X_MEDIUM_SEARCH = False
RUN_08X_AGGRESSIVE_SEARCH = False
RUN_08X_AGGREGATE_FAILURE_MAP = False

RUN_08F_CONTRACT_GATE = False
RUN_08F_CANDIDATE_COMPRESSION = False
RUN_08F_WRITE_FREEZE_RECORD = False

RUN_08O_ENTRY_GATE = False
RUN_08O_OFFICIAL_VALIDATION_READOUT = False
RUN_08O_AGGREGATE_AND_WRITE_MANIFEST = False

BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False
```

The notebook should print that no work ran when all switches are false.

---

## 13. Output Artifact Manifest

Recommended local Colab output root:

```text
/content/notebook08_deep_sequence_results/
```

### 13.1 08X Artifacts

```text
08x_search_space.json
08x_trial_ledger.csv
08x_fold_results.csv
08x_seed_summary.csv
08x_failure_ledger.csv
08x_candidate_compression_table.csv
08x_run_manifest.json
08x_environment_manifest.json
```

Required 08X manifest fields:

```text
notebook08_version
stage = 08X
scope = exploratory
source_stage0_candidate
official_validation_used = false
holdout_test_authorized = false
train_inner_fold_policy
purge_policy
embargo_policy
search_budget_tier
trial_count_requested
trial_count_completed
trial_count_failed
trial_count_skipped
```

### 13.2 08F Artifacts

```text
08f_candidate_freeze_record.json
08f_candidate_freeze_record.md
08f_candidate_compression_audit.csv
08f_static_gate_report.json
08f_no_candidate_freezable.json
```

Required 08F freeze fields:

```text
stage = 08F
scope = diagnostic
primary_candidate_id
fallback_candidate_id
fallback_activation_rule
config_hash
architecture_family
frozen_architecture_params
frozen_loss
frozen_hpo_method
frozen_seed_list
frozen_metric_list
frozen_wording_rule
paper_safe_score
frozen_code_git_sha
frozen_python_env_hash
frozen_dependency_versions
official_validation_used_for_selection = false
holdout_test_authorized = false
```

`frozen_code_git_sha` MUST be the full sha of the project repo commit at freeze
time (no `-dirty` allowed; uncommitted changes block freeze). `frozen_python_env_hash`
is a sha256 over the sorted `pip freeze` output. `frozen_dependency_versions` is
a dict of `{package: version}` for at least pytorch, lightgbm, numpy, pandas,
scikit-learn, and any deep-model framework loaded by the candidate. 08O MUST
verify the runtime env still matches these three fields before any official-
validation read.

### 13.3 08O Artifacts

```text
08o_validation_readout.csv
08o_validation_per_ticker.csv
08o_seed_summary.csv
08o_same_row_baselines.csv
08o_concentration_guardrails.csv
08o_failure_rows.csv
08o_decision_record.json
08o_run_manifest.json
```

Required 08O manifest fields:

```text
stage = 08O
scope = validation_only
primary_candidate_id
freeze_record_sha256
official_validation_readout_started_at
official_validation_used_for_selection = false
holdout_test_authorized = false
same_row_dummy_present = true
per_ticker_present = true
seed_summary_present = true
allowed_wording_bucket
```

---

## 14. Metrics And Decision/Wording Rules

### 14.1 Primary Decision Metrics

08O decision language should be based on:

```text
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
delta_macro_f1_vs_stratified_dummy_same_rows_mean
delta_macro_f1_vs_stratified_dummy_same_rows_lcb_95
per_ticker_delta_macro_f1_vs_dummy
```

### 14.2 Same-Row Dummy Rule

Every model row must be paired with a dummy baseline on the same target rows.
For selective/coverage rows, the dummy must be scored on the retained rows, not
the full validation partition.

### 14.3 Concentration Guardrails

Downgrade or reject stronger wording if:

```text
one ticker dominates retained/evaluated rows;
positive delta appears in only one ticker;
open/close time buckets dominate gains;
date clusters dominate gains;
overlapping-window dependence makes row counts look stronger than they are.
```

### 14.4 Class Collapse Guardrails

Flag class collapse when:

```text
class0_pred_rate < 0.05 or class1_pred_rate < 0.05
balanced_accuracy near 0.50 despite macro F1 movement
per-ticker confusion matrices show one-class prediction
```

### 14.5 Complexity Penalty

Any deep model claim must survive a complexity penalty. A candidate with more
parameters, more tuning, or ensembles must show stable benefit over:

```text
same-row stratified dummy
always-up dummy
last-step LightGBM control
last-step MLP sequence ablation
dlinear_only or tcn_only component baselines
```

### 14.6 Diagnostic-Only Metrics

These can qualify discussion but cannot select the model after official
validation:

```text
ECE
AURC
Brier score
log loss
SHAP
permutation importance
bootstrap intervals
selective coverage curves
conformal/risk-control diagnostics
```

---

## 15. Material Preparation Checklist

### 15.1 Before 08X

- Confirm active Stage 0 candidate and freeze document.
- Confirm raw-data-first data manifest for five tickers.
- Confirm no holdout/test paths are referenced by active code.
- Confirm train/validation/closed-holdout boundary markers.
- Write 08X search-space JSON before running.
- Write train-inner fold policy before running.
- Write failure ledger schema before running.
- Write static gate tests before notebook generation.

### 15.2 Before 08F

- Confirm 08X ledger includes failed/skipped trials.
- Confirm no official-validation selection columns are present.
- Confirm all trial rows have config hashes.
- Confirm same-row dummy rows exist for train-inner fold comparisons.
- Confirm candidate compression rule was frozen before reading compression
  outputs.
- Confirm no candidate relies on holdout/test wording.

### 15.3 Before 08O

- Confirm 08F freeze record exists and hashes match.
- Confirm primary/fallback activation rule is explicit.
- Confirm static gate passes.
- Confirm all `RUN_08O_*` switches are default false until operator edit.
- Confirm official-validation readout is one-time and frozen.
- Confirm same-row dummy and per-ticker output schemas are present.
- Confirm holdout/test remains closed.

---

## 16. What 08 Must Not Do

Notebook 08 must not:

- touch holdout/test;
- use official validation for HPO;
- use official validation for architecture search;
- select a loss from official validation;
- select a confidence or no-trade threshold from official validation;
- refit a calibrator on official validation and evaluate on the same rows;
- change labels, features, windows, or Stage 0 candidate after seeing 08X;
- package exploratory 08X best rows as paper evidence;
- hide failed trials;
- compare high-confidence retained rows only to a full-row dummy;
- claim profitability, tradability, deployability, or final evidence;
- rewrite the 02-06 selection narrative.

---

## 17. Reviewer Risk Audit And Defenses

| Reviewer attack | Risk | Defense required | Static check ID |
|---|---|---|---|
| "This is a model zoo." | Many deep variants create chance wins. | Finite 08X search space, full trial ledger, 08F compression, one 08O readout. | `tests/test_notebook08_static_gate.py::test_design_has_pre_registration_constants_table` (covers `total_trial_budget_cap_across_all_families`) |
| "You reused validation too often." | Official validation becomes a selection set. | 08X uses train-inner folds only; 08F freezes before 08O; 08O is readout only. | `tests/test_notebook08_artifact_contract.py::test_08o_ledger_append_increments_counter` + `::test_08o_missing_append_is_rejected` + `::test_08o_rejects_tampered_prefix_row_value` |
| "Deep models got unfair HPO." | Budget asymmetry inflates deep results. | Report configs/fits/time/failures; compare against simple controls and dummy baselines. | `tests/test_notebook08_static_gate.py::test_design_has_pre_registration_constants_table` (covers `total_trial_budget_cap_across_all_families`) |
| "Loss and thresholds were fished." | Focal/class-balanced/selective choices can be post-hoc. | Freeze loss and threshold rules in 08F; diagnostics only unless predeclared. | `tests/test_notebook08_artifact_contract.py::test_freeze_record_rejects_fallback_rule_referencing_official_val` |
| "Complexity is unjustified." | A fragile deep win is not thesis-worthy. | Complexity penalty, component ablations, last-step controls, seed/fold/per-ticker stability. | `tests/test_notebook08_static_gate.py::test_design_has_pre_registration_constants_table` (covers `paper_safe_score_penalty_complexity`) |
| "Ticker concentration drives the result." | One ticker or time bucket explains gains. | Per-ticker deltas, ticker share, date/time guardrails, downgraded wording. | TBD: planned `tests/test_notebook08_static_gate.py::test_design_requires_per_ticker_concentration_diagnostic` |
| "Selective filtering manufactures gains." | Retained rows are easier by construction. | Same-row dummy and random-abstention baselines at every coverage level. | `tests/test_notebook08_static_gate.py::test_design_has_pre_registration_constants_table` |
| "Post-hoc storytelling." | Weak evidence is narratively polished. | Allowed wording buckets tied to artifacts; no stronger claim than validation-only readout. | `tests/test_notebook08_artifact_contract.py::test_freeze_record_rejects_various_official_val_fallback_phrasings` |
| "Financial data snooping." | Repeated searches over a single history. | Treat 08X as development; disclose search budget; cite data-snooping literature; no holdout reopening. | `tests/test_notebook08_artifact_contract.py::test_08o_*` family + `tests/test_notebook08_static_gate.py::test_design_keeps_holdout_test_closed` |

---

## 18. Static Tests / Validation Plan For Future Implementation

This task does not implement or execute Notebook 08. A future implementation
should add tests before trusting the notebook.

### 18.1 Static Notebook Gate

Expected future test file:

```text
tests/test_notebook08_static_gate.py
```

Checks:

1. Notebook file exists and parses with `nbformat`.
2. All code cells AST-parse.
3. No saved outputs.
4. All execution counts are `None`.
5. All `RUN_08X_*`, `RUN_08F_*`, and `RUN_08O_*` switches default to `False`.
6. No `drive.mount(` in default setup cells.
7. No `from intraday_research`.
8. No `baseline_helpers`.
9. No `train_test_split`.
10. No `holdout_test_authorized = True`.
11. No official-validation selection strings inside 08X code.
12. Required artifact filenames appear in active code.

### 18.2 Artifact Contract Tests

Expected future test file:

```text
tests/test_notebook08_artifact_contract.py
```

Checks:

1. 08X search-space schema validates.
2. 08X trial ledger requires failed/skipped trial rows.
3. 08X rows reject `official_validation_used == true`.
4. 08F refuses missing trial ledger.
5. 08F refuses missing same-row dummy columns.
6. 08F writes exactly one primary candidate.
7. 08F fallback activation cannot depend on official-validation metrics.
8. 08O refuses freeze records with `holdout_test_authorized == true`.
9. 08O refuses freeze records where official validation was used for selection.
10. 08O run manifest requires same-row dummy, per-ticker, and seed summary
    artifacts.

### 18.3 Manifest Schema Tests

Expected future schema files:

```text
schemas/notebook08/08x_search_space.schema.json
schemas/notebook08/08x_trial_ledger.schema.json
schemas/notebook08/08f_candidate_freeze_record.schema.json
schemas/notebook08/08o_run_manifest.schema.json
```

If schemas are not added, tests should validate the minimal field set directly.

### 18.4 Verification Commands

Use the project Python executable:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\create_deep_sequence_exploration_colab_notebook.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook08_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook08_artifact_contract.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook08_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook08_artifact_contract.py
```

Do not run the notebook as a validation step unless explicitly authorized.

---

## 19. Literature And Official Source Anchors

These sources support method vocabulary and risk controls. They do not by
themselves authorize Notebook 08 results.

### 19.1 HPO And Search Control

- Bergstra and Bengio, Random Search for Hyper-Parameter Optimization, JMLR
  2012: https://www.jmlr.org/papers/v13/bergstra12a.html
- Optuna `TPESampler` documentation for Tree-structured Parzen Estimator:
  https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- Li et al., Hyperband, JMLR 2018:
  https://www.jmlr.org/papers/v18/16-558.html
- Li et al., ASHA / massively parallel hyperparameter tuning:
  https://arxiv.org/abs/1810.05934
- Falkner, Klein, and Hutter, BOHB:
  https://arxiv.org/abs/1807.01774
- Jaderberg et al., Population Based Training:
  https://arxiv.org/abs/1711.09846

### 19.2 Sequence Models And Losses

- Zeng et al., Are Transformers Effective for Time Series Forecasting? /
  LTSF-Linear and DLinear:
  https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089
- Bai, Kolter, and Koltun, empirical TCN evaluation:
  https://arxiv.org/abs/1803.01271
- Lin et al., Focal Loss:
  https://arxiv.org/abs/1708.02002
- Cui et al., Class-Balanced Loss:
  https://arxiv.org/abs/1901.05555
- Ren et al., Balanced Softmax / Balanced Meta-Softmax:
  https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html

### 19.3 Selection Bias, Calibration, And Selective Prediction

- Cawley and Talbot, over-fitting in model selection:
  https://jmlr.org/papers/v11/cawley10a.html
- White, Reality Check for Data Snooping:
  https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152
- Harvey, Liu, and Zhu, cross-section of expected returns:
  https://academic.oup.com/rfs/article/29/1/5/1843824
- Geifman and El-Yaniv, Selective Classification for Deep Neural Networks:
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- Guo et al., calibration of modern neural networks:
  https://arxiv.org/abs/1706.04599
- Angelopoulos et al., Conformal Risk Control:
  https://arxiv.org/abs/2208.02814
- scikit-learn `TimeSeriesSplit`:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- scikit-learn `DummyClassifier`:
  https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- LightGBM `LGBMClassifier`:
  https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

---

## 20. Acceptance Criteria

This technical design is complete when:

1. It states the 08X/08F/08O split and prevents 08X-to-official leakage.
2. It maps 02-09 roles and keeps 08 as a post-Stage-0 extension.
3. It lists source artifacts and current related file inventory.
4. It defines the 08X search space, train-inner validation design, and ledger.
5. It defines 08F candidate compression, `paper_safe_score`, and hard stops.
6. It defines 08O entry gates, metrics, baselines, and allowed wording.
7. It includes budget tiers and default-off run switches.
8. It includes output artifact names and required schema fields.
9. It includes reviewer risk audit and defenses.
10. It includes a future static-test and artifact-contract validation plan.
11. It keeps holdout/test closed and does not authorize execution.

---

## 21. Task Report

Files inspected:

```text
AGENTS.md
docs/RESEARCH_WORKFLOW.md
docs/CONFIG_SCREENING_FREEZE_2026-06-04.md
docs/BASELINE_REFERENCE.md
docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md
docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md
docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md
docs/research_notes/06_07_literature_materials_2026-06-05.md
docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md
docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md
notebooks/
scripts/
tests/
```

Files changed:

```text
docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md
```

Commands run:

```text
Get-Content -LiteralPath AGENTS.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --stat
rg --files docs notebooks scripts tests
rg -n "<08/LightGBM/DLinear/TCN/selective/artifact/static-gate patterns>" notebooks scripts tests docs
Test-Path -LiteralPath docs\NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md
rg --files docs | rg "NOTEBOOK0(7|8)|08|07_VALIDATION|DEEP_SEQUENCE"
E:\codex_workspace\_envs\py311_shared\python.exe -c "<required-doc heading and existence summary>"
```

Validation results:

```text
No notebook was executed.
No training was run.
No dependency was installed.
No holdout/test data was read.
No commit or push was made.
Target 08 technical-design document did not previously exist.
07 technical-design document appeared in final git status, was read, and this 08 design was rechecked against its 07G/gateway rules.
```

Unresolved issues:

```text
Future implementation still needs actual 08 notebook generator, static gates,
artifact-contract tests, and schemas.
08 budget constants and paper_safe_score weights should be frozen again before
any future 08X execution.
If the 07 technical design changes later, 08 should be rechecked against its
latest gateway rules before implementation.
```
