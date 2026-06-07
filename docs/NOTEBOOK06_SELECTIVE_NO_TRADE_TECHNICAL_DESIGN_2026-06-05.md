# Notebook 06 Selective No-Trade Technical Design - 2026-06-05

Scope: `validation_only`
Version: `1.4`
Last revised: `2026-06-05`

Change log:

```text
1.0 initial technical design
1.1 fixed 05 `.npz` field contract: physical key is `prob_up`
1.2 split decision-record fields into hard-required and preferred
1.3 added testable contract module, run switches, and quantified decision rules
1.4 aligned with parent protocol files, expanded 06 numeric/test contracts
1.5 aligned stage names, run switches, and required outputs with the generated
    Notebook 06 implementation
```

This document is the implementation-facing design for
`notebooks/06_selective_no_trade_calibration_colab.ipynb`. It freezes the first
Notebook 06 version before code changes. It is not an execution result, not a
new experiment approval, and not a holdout/test authorization.

Notebook 06 must be a conservative artifact-only selective/no-trade readout
over already produced Notebook 05 official-validation probability artifacts.
It must not train a new model, refit LightGBM, fit a probability calibrator,
select a final confidence threshold, or read holdout/test rows.

## 1. Source Materials Used

The first implementation must use these local materials as the design boundary:

| Material | Role in Notebook 06 |
| --- | --- |
| `AGENTS.md` | project-level chronology, dummy-baseline, holdout/test, notebook, and Python rules |
| `docs/RESEARCH_WORKFLOW.md` | validation-only reporting shape and dummy-baseline requirement |
| `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md` | parent Notebook 05 contract and selective-threshold boundary |
| `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md` | main Notebook 06 protocol |
| `docs/research_notes/06_artifact_contract_implementation_materials_2026-06-05.md` | 05 -> 06 artifact contract and implementation notes |
| `docs/research_notes/06_07_literature_materials_2026-06-05.md` | selective prediction, no-trade, validation-reuse, and finance caveat sources |
| `docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md` | Brier, ECE, reliability, AURC, and E-AURC formulas |
| `docs/research_notes/06_07_intraday_concentration_guardrails_materials_2026-06-05.md` | ticker, date, time-of-day, and dependence guardrails |

The literature supports selective prediction and no-trade framing under
validation-only evidence. It does not support profitability, deployment,
holdout/test, or final trading-threshold claims.

## 2. Version-1 Design Decision

Notebook 06 version 1 is:

```text
artifact-only raw-probability selective readout
fixed coverage grid
same-row dummy comparison
ticker-stratified random-abstention comparison
probability diagnostics
risk-coverage diagnostics
concentration guardrails
validation_only wording
```

Notebook 06 version 1 is not:

```text
new model training
LightGBM refit
CalibratedClassifierCV fit
isotonic calibration fit
sigmoid calibration fit
MAPIE or conformal risk-control implementation
official-validation threshold search
coverage-point selection
PnL, Sharpe, transaction-cost, or backtest notebook
holdout/test readout
```

Reasoning:

1. Notebook 05 provides official-validation probabilities. Those rows are
   appropriate for a readout of pre-registered coverage levels.
2. Fitting a calibrator on official validation and then reporting the same rows
   would reuse validation labels for model/threshold design.
3. A fixed coverage grid protects Notebook 06 from selecting the best-looking
   official-validation confidence threshold after inspection.
4. Selective/no-trade evidence is meaningful only if the retained subset is
   compared against baselines on the exact same retained rows and is not
   concentrated in one ticker, date, or intraday bucket.

## 3. Notebook 05 Dependency

Notebook 06 may start only after a Notebook 05 artifact bundle passes the
contract in this section.

Preferred input is a clean Notebook 05 rerun produced by the repaired Notebook 05
source, where `validation_sample_id` is generated before 05D writes prediction
artifacts.

Legacy Notebook 05 artifacts produced by a runtime hotfix may be inspected only
if all of these conditions pass:

1. `validation_sample_id` exists in every selected LightGBM `.npz` prediction
   artifact.
2. `validation_sample_id_hash` is identical across all selected profile/seed
   artifacts.
3. `sample_id_mismatch_count == 0` in the official-validation pooled rows.
4. The sample-id formula is documented in the Notebook 05 run notes or decision
   record.
5. The decision record states `holdout_test_authorized == false` and
   `selective_threshold_selected == false`.

Notebook 06 must resolve the primary profile in this order:

1. If `notebook05_decision_record.json` contains
   `downstream_primary_profile_id`, use that field.
2. Else, if the decision record contains `retained_default_lgbm_04 == true` or
   `official_validation_status == "retain_default_lgbm_04"`, use
   `default_lgbm_04`.
3. Else, use `selected_profile_id`.
4. Stop if the resolved profile is empty or has no prediction artifacts.

If Notebook 05 records any non-promotion status that also sets
`downstream_primary_profile_id == "default_lgbm_04"`, Notebook 06 primary
analysis must use `default_lgbm_04`. Tuned train-inner finalists may be included
only as secondary diagnostic profiles and must not replace the primary profile.

## 4. Hard Entry Gates

The first Notebook 06 code cell that reads artifacts must enforce these gates.
Any failed gate stops execution before metrics are computed.

Required scope gates:

```text
scope == validation_only
holdout_test_authorized == false
selective_threshold_selected == false
```

Required Notebook 05 files:

```text
notebook05_entry_decision.json
notebook05_decision_record.json
notebook05_run_manifest.json
notebook05_official_validation_summary.csv
notebook05_official_validation_pooled.csv
notebook05_official_validation_per_ticker.csv
predictions/*.npz
```

Hard-required decision-record fields:

```text
scope
holdout_test_authorized
selective_threshold_selected
selected_profile_id
selected_profile_source
```

Preferred decision-record fields:

```text
official_validation_status
downstream_primary_profile_id
train_inner_selected_profile_id
retained_default_lgbm_04
promotion_checks
```

Preferred fields enrich profile resolution, but Notebook 06 must not require
them from older Notebook 05 artifacts if the hard-required fields and pooled CSV
contract pass. `validation_sample_id_hash` is required from
`notebook05_official_validation_summary.csv` or
`notebook05_official_validation_pooled.csv`, not from the decision record.
The repaired Notebook 05 source is expected to write the preferred fields. The
fallback order exists only to keep older timestamped validation-only artifacts
reviewable when their sample-id and pooled-artifact contracts pass.

Required official pooled fields:

```text
profile_id
profile_role
seed
ticker_or_pooled
train_n
validation_n
train_class0_n
train_class1_n
train_positive_rate
validation_sample_id_hash
sample_id_mismatch_count
prediction_artifact
macro_f1
balanced_accuracy
accuracy
stratified_dummy_macro_f1
delta_macro_f1_vs_stratified_dummy
always_up_dummy_macro_f1
delta_macro_f1_vs_always_up_dummy
scope
```

Notebook 05 currently writes `train_n`, but the clean 05 -> 06 contract must
also provide `train_class0_n`, `train_class1_n`, and `train_positive_rate` for
each official-validation LightGBM pooled row, or provide a train-derived dummy
prediction artifact with stable `validation_sample_id`. Notebook 06 must not
estimate stratified dummy class probabilities from validation labels.
`train_positive_rate` must equal
`train_class1_n / (train_class0_n + train_class1_n)` within `FLOAT_TOLERANCE`.
`holdout_test_authorized` is a run/decision-level gate checked from
`notebook05_decision_record.json`, `notebook05_run_manifest.json`, and, when
present, `notebook05_entry_decision.json`; it is not required as a repeated
official-validation CSV row field.

Required `.npz` arrays:

```text
y_true
prob_up
y_pred
validation_sample_id
ticker
timestamp
confidence
```

The `.npz` artifact uses the Notebook 05 field name `prob_up`. Notebook 06 may
rename it to canonical frame column `y_prob_up` after loading, but the artifact
contract must validate the physical `.npz` key as `prob_up`.

Hard-stop conditions:

```text
missing required file
missing required column or JSON key
selected primary profile has no prediction artifact
prediction artifact path points outside the Notebook 05 result bundle
prediction artifact path contains holdout or test
selected_profile_source is official_validation_best or otherwise states an
  official-validation-best replacement
train-derived stratified dummy reconstruction fields are absent
any required `.npz` array is missing
required `.npz` arrays have unequal lengths
validation_sample_id is missing
validation_sample_id is duplicated within one artifact
validation_sample_id order differs across primary-profile seed artifacts
validation_sample_id_hash differs across primary-profile seed artifacts
artifact confidence differs from max(prob_up, 1 - prob_up) by more than FLOAT_TOLERANCE
official pooled sample_id_mismatch_count is not zero
scope is not validation_only
holdout_test_authorized is not false
selective_threshold_selected is not false
```

Notebook 06 must not repair a missing `validation_sample_id` by using row
numbers. Missing sample ids mean the Notebook 05 artifact bundle is invalid for
Notebook 06.

## 5. Fixed Constants

Notebook 06 version 1 must declare these constants before reading prediction
artifacts:

```python
NOTEBOOK06_SCOPE = "validation_only"
COVERAGE_GRID = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
DECISION_COVERAGE_GRID = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
MIN_INTERPRETABLE_COVERAGE = 0.30
MIN_DECISION_DELTA_MACRO_F1 = 0.005
MIN_POSITIVE_SEED_COUNT = 4
MIN_POSITIVE_DECISION_COVERAGE_COUNT = 4
NOT_SUPPORTED_FAILURE_COVERAGE_COUNT = 4
INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MIN = 1
INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MAX = 3
INCONCLUSIVE_NOISE_COVERAGE_COUNT = 4
INCONCLUSIVE_MIXED_SEED_COVERAGE_COUNT = 3
INCONCLUSIVE_WARNING_COVERAGE_COUNT = 2
RANDOM_ABSTENTION_REPEATS = 100
RANDOM_ABSTENTION_BASE_SEED = 260606
CALIBRATION_BIN_COUNT = 20
CALIBRATION_PRIMARY_BINNING = "quantile"
CALIBRATION_SENSITIVITY_BINNING = "uniform"
PRIMARY_CONFIDENCE_COLUMN = "confidence"
FLOAT_TOLERANCE = 1e-9
PLOT_DPI = 300
PLOT_FIGSIZE = (8.0, 6.0)
T_CRITICAL_ONE_SIDED_95 = {
    1: 0.0,
    2: 6.314,
    3: 2.920,
    4: 2.353,
    5: 2.132,
    6: 2.015,
    7: 1.943,
    8: 1.895,
    9: 1.860,
    10: 1.833,
}
```

These constants are not tuned after reading official-validation artifacts.
The implementation must assert:

```python
assert min(COVERAGE_GRID) >= MIN_INTERPRETABLE_COVERAGE
assert set(DECISION_COVERAGE_GRID).issubset(set(COVERAGE_GRID))
```

Coverage below `0.30` is omitted from the first implementation. If a later
appendix includes lower coverage, it must be labeled low-coverage diagnostic and
must not drive the decision record.

`CALIBRATION_PRIMARY_BINNING = "quantile"` is the primary ECE readout because
LightGBM probabilities can cluster, and quantile bins reduce empty-bin problems.
Uniform bins are retained only as a sensitivity view. ECE remains diagnostic and
does not select a calibrator or coverage level.

Notebook 06 must use explicit run switches, all defaulting to `False`:

```python
RUN_06A_TO_06G_FULL_PIPELINE = False
RUN_06S_SCHEMA_SMOKE = False
RUN_06A_ARTIFACT_GATE = False
RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS = False
RUN_06C_FIXED_COVERAGE_GRID = False
RUN_06D_AGGREGATE_AND_RISK_COVERAGE = False
RUN_06E_CONCENTRATION_GUARDRAILS = False
RUN_06F_DECISION_RECORD = False
RUN_06G_BACKUP_TO_GOOGLE_DRIVE = False
BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE = False
```

Notebook 06 must also require explicit operator acknowledgement before any
artifact-readout stage runs:

```python
OPERATOR_ACKNOWLEDGES_NOTEBOOK05_DECISION = False
OPERATOR_ACCEPTS_06_AS_ARTIFACT_ONLY_READOUT = False
OPERATOR_ACKNOWLEDGEMENT_TEXT = """
Notebook 06 is validation-only and artifact-only. It does not fit a calibrator,
does not select a final confidence threshold, and does not authorize
holdout/test access.
""".strip()
```

The 06A entry gate must display `OPERATOR_ACKNOWLEDGEMENT_TEXT` and require both
operator acknowledgement flags to be `True` before reading Notebook 05
artifacts.

## 6. Prediction Frame Contract

Notebook 06 must build one canonical prediction frame per
`(profile_id, seed)`.

Required columns in the canonical frame:

```text
validation_sample_id
profile_id
profile_role
seed
ticker
timestamp
y_true
y_prob_up
y_pred
confidence
correct
source_prediction_artifact
scope
```

`y_prob_up` is the canonical frame name derived from the `.npz` key `prob_up`.
`source_prediction_artifact` should be stored as a path relative to the
Notebook 05 result bundle when possible, so the manifest is portable across
local, Colab, and Drive copies.
`validation_sample_id` must be cast to string before sorting, hashing joins, or
tie-breaking so behavior is stable across Python, NumPy, and platform versions.

Derived definitions:

```python
computed_confidence = max(prob_up, 1.0 - prob_up)
assert allclose(confidence, computed_confidence, atol=FLOAT_TOLERANCE, rtol=0.0)
confidence = computed_confidence
y_prob_up = prob_up
correct = int(y_pred == y_true)
```

Sorting rule for selective retained rows:

```text
sort by confidence descending
break ties by string-cast validation_sample_id ascending
retain ceil(coverage * n) rows
```

The same sorting and tie-breaking rule must be used for every profile, seed,
coverage, and output table.

## 7. Notebook Structure

The generated notebook should use these sections:

```text
06A - Artifact Gate
06B - Prediction Frames And Probability Diagnostics
06C - Fixed Coverage Grid, Same-Row Dummy, Random Abstention
06D - Aggregate Seed Metrics And AURC/E-AURC
06E - Concentration Guardrails
06F - Decision Record
06G - Optional Google Drive Backup
```

### 7.1 06A - Artifact Contract Gate

Responsibilities:

1. Read only Notebook 05 JSON, CSV, and `.npz` artifacts.
2. Check required files, fields, sample ids, scope, and holdout flags.
3. Resolve the primary profile using the Section 3 fallback order.
4. Compute `notebook05_decision_record_sha256` and
   `notebook05_run_manifest_sha256` from raw file bytes.
5. Assert `selected_profile_source` does not indicate official-validation-best
   replacement.
6. Write `notebook06_artifact_contract_check.json`.
7. Stop before metrics if any gate fails.

### 7.2 06B - Selected Profile Resolution And Prediction Frame

Responsibilities:

1. Load primary-profile prediction artifacts for all official-validation seeds.
2. Optionally load secondary diagnostic profiles only when they pass the same
   artifact contract.
3. Build canonical prediction frames.
4. Re-verify equal sample-id order across primary-profile seeds as
   defense-in-depth after canonical frames are built.
5. Save `notebook06_prediction_frame_manifest.csv`.

Secondary diagnostic profiles:

```text
included in prediction_frame_manifest: yes
included in probability_diagnostics: yes
included in reliability_bins: yes
included in coverage_grid: yes
included in random_abstention_baselines: yes
included in risk_coverage_summary: yes
included in concentration_guardrails: yes
used for 06F primary decision: no
allowed to replace primary_profile_id: no
```

The decision record may mention secondary profiles only as diagnostic context.
It must not select a different profile because a secondary profile looks better
on official validation.

### 7.3 06C - Full-Row Probability Diagnostics

Compute full-validation diagnostics for each `(profile_id, seed)`:

```text
brier_score
ece_prob_up_quantile_20
ece_prob_up_uniform_20
ece_top_label_quantile_20
ece_top_label_uniform_20
mean_confidence
confidence_p10
confidence_p50
confidence_p90
positive_rate_true
positive_rate_predicted
```

Save:

```text
notebook06_probability_diagnostics.csv
notebook06_reliability_bins.csv
```

These diagnostics describe probability quality. They are not selection gates and
do not choose a calibrator or coverage point.
ECE computation skips bins with `bin_count == 0`. Empty bins are still written
to `notebook06_reliability_bins.csv` with `bin_calibration_gap = NaN` for
transparency.

`notebook06_reliability_bins.csv` schema:

```text
profile_id
seed
ece_type
binning_method
bin_id
bin_lower_edge
bin_upper_edge
bin_count
bin_avg_score
bin_avg_outcome
bin_calibration_gap
scope
```

### 7.4 06C - Fixed Coverage-Grid Selective Metrics

For each `(profile_id, seed, coverage_target)`, compute:

```text
coverage_target
coverage_actual
retained_n
abstained_n
min_retained_confidence
retained_class0_n
retained_class1_n
retained_pred0_n
retained_pred1_n
selective_macro_f1
selective_balanced_accuracy
selective_accuracy
selective_error
same_row_stratified_dummy_macro_f1
delta_macro_f1_vs_same_row_stratified_dummy
same_row_always_up_dummy_macro_f1
delta_macro_f1_vs_same_row_always_up_dummy
```

The implied confidence threshold is descriptive only. It is not a selected final
threshold. It is computed as:

```text
min_retained_confidence = min(confidence among retained rows)
```

Notebook 06 should avoid calling this value an operating threshold outside
schema compatibility contexts.

Save:

```text
notebook06_coverage_grid.csv
```

### 7.5 06C - Same-Row Dummy And Random-Abstention Baselines

Same-row dummy requirement:

1. Derive stratified dummy class probabilities only from official training
   partition class counts saved by Notebook 05, or load a Notebook 05
   train-derived dummy prediction artifact if one exists.
2. For each official-validation seed, generate one deterministic stratified
   dummy prediction for every validation sample in canonical sample-id order
   using that same seed.
3. Pair every `(profile_id, seed)` selective row with the train-derived dummy
   predictions for the same seed.
4. Score the dummy on exactly the retained sample ids for each coverage row.
5. Report `delta_macro_f1_vs_same_row_stratified_dummy`.

Notebook 06 must not estimate stratified dummy class probabilities from full or
retained validation labels. If the 05 artifact bundle lacks train class counts
and lacks train-derived dummy predictions, 06F must write
`selective_no_trade_blocked_artifact_contract_failure` instead of computing
validation-informed dummy baselines.

Ticker-stratified random-abstention requirement:

1. For each model-selected retained subset, compute retained count by ticker.
2. Repeat `RANDOM_ABSTENTION_REPEATS` times.
3. Within each ticker, randomly retain the same number of rows as the
   model-selected subset.
4. Score the fixed model predictions on those randomly retained rows.
5. Report mean, standard deviation, p10, p50, and p90 of random-abstention
   macro F1 and balanced accuracy.

Repeat seeds are deterministic:

```python
repeat_seed = RANDOM_ABSTENTION_BASE_SEED + repeat_index
```

This tests whether confidence ranking selects better rows than random retention
with the same ticker mix.

Save:

```text
notebook06_random_abstention_baselines.csv
```

`notebook06_random_abstention_baselines.csv` schema:

```text
profile_id
seed
coverage_target
retained_n
repeat_count
random_macro_f1_mean
random_macro_f1_std
random_macro_f1_p10
random_macro_f1_p50
random_macro_f1_p90
random_balanced_accuracy_mean
random_balanced_accuracy_std
random_balanced_accuracy_p10
random_balanced_accuracy_p50
random_balanced_accuracy_p90
model_selective_macro_f1
delta_macro_f1_vs_random_abstention_mean
scope
```

### 7.6 06D/06E - Aggregate Risk-Coverage And Concentration Guardrails

Risk-coverage outputs:

```text
AURC
oracle_AURC
E_AURC
full_coverage_error
mean_confidence
```

AURC and E-AURC evaluate ranking behavior over coverage. They are not
probability-calibration scores and not trading metrics.

Formula:

```text
AURC = area under selective_error versus coverage when rows are sorted by model confidence
oracle_AURC = AURC when rows are sorted by `correct` descending
E_AURC = AURC - oracle_AURC
```

Lower AURC and lower E-AURC are better. E-AURC is not normalized in version 1.

Concentration metrics must be computed at every fixed coverage point:

```text
top_ticker_retained_share
ticker_entropy_norm
ticker_effective_n_hhi
positive_ticker_count
top_ticker_gain_share
top_day_selected_share
top_5_day_selected_share
date_entropy_norm
top_time_bucket_share
open_close_concentration_30
open_close_lift_30
unique_ticker_day_count
retained_one_class_collapse
predicted_one_class_collapse
```

Time-of-day buckets are 30-minute regular-session buckets derived from
`timestamp`. The expected regular session is 09:30 to 16:00 New York time after
whatever timestamp normalization is already present in the Notebook 05
artifacts. Notebook 06 must not create new market-calendar labels from
holdout/test data.

Open/close definitions:

```text
open_30 = rows with time >= 09:30 and time < 10:00
close_30 = rows with time >= 15:30 and time < 16:00
open_close_concentration_30 =
  (retained_n_open_30 + retained_n_close_30) / retained_n
open_close_lift_30 =
  open_close_concentration_30 /
  ((eligible_n_open_30 + eligible_n_close_30) / eligible_n)
```

Warning guardrails:

```text
retained_class0_n == 0 or retained_class1_n == 0
retained_pred0_n / retained_n > 0.95
retained_pred1_n / retained_n > 0.95
top_ticker_retained_share > 0.50
ticker_entropy_norm < 0.70
positive_ticker_count < 3
top_day_selected_share > max(5 / eligible_day_count, 0.10)
top_5_day_selected_share > 0.40
date_entropy_norm < 0.60
top_time_bucket_share > 0.25
open_close_lift_30 > 1.50
unique_ticker_day_count < 10
```

`top_time_bucket_share > 0.25` is about 3.25 times the uniform expectation for
13 regular-session 30-minute buckets. A retained subset above that level in one
time bucket should be treated as possible intraday-seasonality concentration,
not broad validation evidence.

Severe downgrade guardrails:

```text
top_ticker_retained_share > 0.65
ticker_entropy_norm < 0.50
top_ticker_gain_share > 0.70
top_day_selected_share > max(10 / eligible_day_count, 0.20)
top_5_day_selected_share > 0.60
open_close_concentration_30 > 0.50 and open_close_lift_30 > 2.00
positive delta exists only in one ticker
```

Guardrails qualify wording. They do not choose coverage levels.

Save:

```text
notebook06_risk_coverage_summary.csv
notebook06_concentration_guardrails.csv
notebook06_per_ticker_coverage.csv
```

`notebook06_risk_coverage_summary.csv` schema:

```text
profile_id
seed
AURC
oracle_AURC
E_AURC
full_coverage_error
mean_confidence
coverage_grid
scope
```

`notebook06_concentration_guardrails.csv` schema:

```text
profile_id
seed
coverage_target
guardrail_name
guardrail_value
guardrail_threshold
severity
triggered
wording_effect
scope
```

`notebook06_per_ticker_coverage.csv` schema:

```text
profile_id
seed
coverage_target
ticker
eligible_n
retained_n
retained_share_within_ticker
ticker_retained_share_within_coverage
selective_macro_f1
selective_balanced_accuracy
same_row_stratified_dummy_macro_f1
delta_macro_f1_vs_same_row_stratified_dummy
scope
```

### 7.7 06F - Decision Record And Allowed Wording

Notebook 06 must write:

```text
notebook06_decision_record.json
```

Allowed decision values:

```text
selective_no_trade_promising_validation_only
selective_no_trade_inconclusive_validation_only
selective_no_trade_not_supported_validation_only
selective_no_trade_blocked_artifact_contract_failure
```

Decision follow-up actions:

```text
selective_no_trade_blocked_artifact_contract_failure:
  repair or rerun Notebook 05 artifacts; do not compute 06 metrics

selective_no_trade_not_supported_validation_only:
  record weak/no selective evidence; do not open holdout/test

selective_no_trade_inconclusive_validation_only:
  keep holdout/test closed; use 07 only for frozen validation-only synthesis

selective_no_trade_promising_validation_only:
  keep holdout/test closed; carry frozen 06 artifacts into 07 robustness and
  explanation checks without selecting a new threshold
```

Cross-seed aggregation:

```text
seed_count_expected = 5
seed_count_actual = count(successful seed-level rows)
t_critical_one_sided_95 = T_CRITICAL_ONE_SIDED_95.get(seed_count_actual, 1.645)
metric_mean = mean(metric across seeds)
metric_std = sample standard deviation across seeds
metric_lcb_95 = metric_mean - t_critical_one_sided_95 * metric_std / sqrt(seed_count_actual)
positive_seed_count =
  count(seed where delta_macro_f1_vs_same_row_stratified_dummy > 0)
positive_random_seed_count =
  count(seed where delta_macro_f1_vs_random_abstention > 0)
```

If `seed_count_actual == 1`, `metric_lcb_95` equals `metric_mean`.
Notebook 06 reports LCB fields for variance visibility, but the validation-only
decision uses mean delta plus seed-consistency counts rather than LCB as a
selection gate. That is deliberate: Notebook 06 does not select a model,
calibrator, or threshold from official validation.

The primary decision table aggregates only `DECISION_COVERAGE_GRID`:

```text
0.90, 0.80, 0.70, 0.60, 0.50, 0.40
```

Full coverage `1.00` is reported as the no-abstention anchor. Coverage `0.30`
is reported as low-coverage diagnostic. Neither drives the primary Notebook 06
decision unless the result is already blocked by an artifact or guardrail
failure.

Per-coverage pass definitions:

```text
beats_same_row_dummy =
  delta_macro_f1_vs_same_row_stratified_dummy_mean >= MIN_DECISION_DELTA_MACRO_F1
  and positive_seed_count >= MIN_POSITIVE_SEED_COUNT

beats_random_abstention =
  delta_macro_f1_vs_random_abstention_mean >= MIN_DECISION_DELTA_MACRO_F1
  and positive_random_seed_count >= MIN_POSITIVE_SEED_COUNT

small_delta =
  abs(delta_macro_f1_vs_same_row_stratified_dummy_mean) < MIN_DECISION_DELTA_MACRO_F1

mixed_across_seeds =
  positive_seed_count < MIN_POSITIVE_SEED_COUNT

random_abstention_competitive =
  delta_macro_f1_vs_random_abstention_mean < MIN_DECISION_DELTA_MACRO_F1
  or positive_random_seed_count < MIN_POSITIVE_SEED_COUNT

ticker_evidence_not_concentrated =
  top_ticker_gain_share <= 0.50 at the median DECISION_COVERAGE_GRID level
  and positive_ticker_count >= 3 at >= MIN_POSITIVE_DECISION_COVERAGE_COUNT
      decision coverage levels

concentration_warning_without_severe =
  any warning guardrail triggers at >= INCONCLUSIVE_WARNING_COVERAGE_COUNT
      decision coverage levels
  and no severe downgrade triggers at any decision coverage level
```

Decision logic:

```text
Decision evaluation order is strict priority. The first matching outcome wins.

blocked:
  artifact contract fails before metrics

not_supported:
  beats_same_row_dummy is false at all decision coverage levels
  or beats_random_abstention is false at >= NOT_SUPPORTED_FAILURE_COVERAGE_COUNT of 6 decision coverage levels
  or severe concentration downgrade triggers

inconclusive:
  beats_same_row_dummy is true at INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MIN
      to INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MAX of 6 decision coverage levels
  or small_delta is true at >= INCONCLUSIVE_NOISE_COVERAGE_COUNT of 6 decision coverage levels
  or mixed_across_seeds is true at >= INCONCLUSIVE_MIXED_SEED_COVERAGE_COUNT of 6 decision coverage levels
  or random_abstention_competitive is true at >= INCONCLUSIVE_MIXED_SEED_COVERAGE_COUNT of 6 decision coverage levels
  or concentration_warning_without_severe is true

promising:
  beats_same_row_dummy is true at >= MIN_POSITIVE_DECISION_COVERAGE_COUNT of 6 decision coverage levels
  and beats_random_abstention is true at >= MIN_POSITIVE_DECISION_COVERAGE_COUNT of 6 decision coverage levels
  and ticker_evidence_not_concentrated is true
  and no severe concentration downgrade triggers
```

These decision rules are deliberately conservative. They convert Notebook 06
into a validation-only diagnostic decision, not a final operating threshold.

Allowed wording:

```text
Notebook 06 reports validation-only selective/no-trade diagnostics for the
frozen Notebook 05 probability artifacts. Coverage levels were fixed before
official-validation readout. No final trading threshold is selected, and
holdout/test remains closed.
```

Forbidden wording:

```text
coverage X is the final threshold
official validation selected the best threshold
selective no-trade is validated on holdout
the model is holdout-ready
the high-confidence subset is safe for deployment
ECE proves calibration
AURC proves the strategy is tradable
the strategy is profitable
LightGBM is globally superior to all other model families
the abstention rate is optimal
the model is calibrated at coverage X
the high-confidence rows generalize to holdout
the abstention threshold transfers to live trading
```

### 7.8 06G - Optional Google Drive Backup

If `RUN_06G_BACKUP_TO_GOOGLE_DRIVE == True` and
`BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE == True`, Notebook 06 may copy completed
local artifacts to Google Drive. Backup writes must use timestamped file names
and Drive `files().create(...)` semantics so an upload creates a new artifact
record rather than overwriting an earlier Drive copy.

## 8. Output Artifact Manifest

Notebook 06 should write all outputs under:

```text
/content/notebook06_selective_no_trade_calibration_results/
```

Required outputs:

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

If `BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE == True`, the notebook must also write:

```text
notebook06_drive_backup_manifest.json
```

Drive backup is optional and explicit. It must copy only Notebook 06
validation-only result files and must not include holdout/test material.
Uploaded Drive names must follow the Notebook 05 timestamp pattern:

```text
{YYYYMMDDTHHMMSSZ}__{reason}__{filename}
```

`notebook06_artifact_contract_check.json` schema:

```text
scope
created_utc
notebook05_result_dir
notebook05_entry_decision_path
notebook05_decision_record_path
notebook05_run_manifest_path
notebook05_entry_decision_sha256
notebook05_decision_record_sha256
notebook05_run_manifest_sha256
primary_profile_id
primary_profile_source
operator_acknowledgements
required_files_present
required_columns_present
required_npz_arrays_present
sample_id_hash
sample_id_mismatch_count
prediction_artifact_count
holdout_test_authorized
selective_threshold_selected
contract_passed
failure_reason
```

`notebook06_run_manifest.json` schema:

```text
scope
created_utc
notebook06_version
run_switches
operator_acknowledgements
constants
input_artifacts
output_files
primary_profile_id
secondary_profile_ids
notebook05_entry_decision_sha256
notebook05_decision_record_sha256
notebook05_run_manifest_sha256
holdout_test_authorized
selective_threshold_selected
```

Optional explanatory plots:

```text
notebook06_reliability_curve.png
notebook06_risk_coverage_curve.png
notebook06_delta_vs_dummy_by_coverage.png
notebook06_retained_share_by_ticker.png
```

Tables and JSON records are primary artifacts. Plots are explanatory,
validation-only, and do not participate in model or threshold selection.
Optional plots must use `PLOT_DPI` and `PLOT_FIGSIZE` from the fixed constants
block.

## 9. Testable Implementation Units

Create `scripts/notebook06_contract.py` before the notebook generator. This
module is the importable source of truth for the artifact contract and pure
metric helpers. The Colab notebook generator should inline the module source
into the notebook so Colab stays self-contained and does not import local
project packages at runtime.

Required functions:

```python
def assert_notebook06_artifact_contract(notebook05_dir: Path) -> dict: ...
def resolve_notebook06_primary_profile(decision_record: dict, pooled: pd.DataFrame) -> str: ...
def load_notebook06_prediction_artifact(path: Path) -> dict: ...
def build_canonical_prediction_frame(npz_payload: dict, metadata: dict) -> pd.DataFrame: ...
def calibration_bins(values: np.ndarray, outcomes: np.ndarray, n_bins: int, strategy: str) -> list[dict]: ...
def ece_from_bins(rows: list[dict]) -> float: ...
def risk_coverage_curve(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> pd.DataFrame: ...
def aurc_from_curve(curve: pd.DataFrame) -> float: ...
def selective_retained_indices(confidence: np.ndarray, validation_sample_id: np.ndarray, coverage_target: float) -> np.ndarray: ...
def same_row_stratified_dummy_predict(train_class0_n: int, train_class1_n: int, n_validation: int, seed: int) -> np.ndarray: ...
def ticker_stratified_random_abstention(retained_count_by_ticker: dict[str, int], ticker_array: np.ndarray, base_seed: int, repeat_count: int) -> np.ndarray: ...
def concentration_metrics(retained_frame: pd.DataFrame, eligible_frame: pd.DataFrame) -> dict: ...
def aggregate_across_seeds(per_seed_metrics: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame: ...
def evaluate_decision_outcome(per_coverage_aggregated: pd.DataFrame, guardrails: pd.DataFrame, constants: dict) -> dict: ...
```

`scripts/create_selective_no_trade_calibration_colab_notebook.py` is responsible
only for notebook assembly, run switches, setup cells, and output wiring.
The notebook generator must read `scripts/notebook06_contract.py` as text and
insert the module body into the generated notebook as a code cell. The generated
notebook must call the functions directly after inlining. It must not import
`scripts.notebook06_contract`, `intraday_research`, `baseline_helpers`, or any
Notebook 05 source at Colab runtime.

## 10. Static Tests To Write Before Notebook Code

Create `tests/test_notebook06_static_gate.py` before writing the Notebook 06
generator.

Required static checks:

1. Generated notebook parses as JSON and all code cells parse as Python.
2. All outputs are empty and all execution counts are `None`.
3. Heavy or result-producing run switches default to `False`.
4. `NOTEBOOK06_SCOPE == "validation_only"` appears in setup code.
5. `COVERAGE_GRID` equals `(1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)`.
6. `DECISION_COVERAGE_GRID` equals `(0.90, 0.80, 0.70, 0.60, 0.50, 0.40)`.
7. Operator acknowledgement switches default to `False`.
8. Backup switch defaults to `False`.
9. `OPERATOR_ACKNOWLEDGEMENT_TEXT` exists and is displayed by the 06A entry
   gate.
10. No active code contains `drive.mount`.
11. No active code imports `intraday_research`, `baseline_helpers`, or stale
   helper packages.
12. Active code rejects holdout/test paths and flags.
13. Active code contains hard-stop checks for missing `validation_sample_id`.
14. Active code requires `.npz` key `prob_up`, not `y_prob_up`.
15. Active code generates same-row stratified dummy predictions only through
    `same_row_stratified_dummy_predict(`.
16. Active code contains same-row dummy and ticker-stratified random-abstention
    logic.
17. Active code contains dynamic `T_CRITICAL_ONE_SIDED_95` handling.
18. Active code contains strict decision-priority ordering.
19. Active code contains ECE, Brier, AURC, E-AURC, and concentration guardrail
    functions or calls.
20. Active code writes `notebook06_decision_record.json`.
21. Active code writes `notebook06_run_manifest.json`.
22. Active code contains forbidden-wording guardrails.

Create `tests/test_notebook06_artifact_contract.py` before wiring full notebook
generation.

Required unit checks:

1. Contract validation passes on a minimal valid fake Notebook 05 bundle.
2. Missing `prob_up` in `.npz` fails with the exact artifact path.
3. Missing `validation_sample_id` fails with the exact artifact path.
4. Duplicated `validation_sample_id` fails.
5. Differing sample-id order across two seed artifacts fails.
6. Differing sample-id hash across two seed artifacts fails.
7. `holdout_test_authorized == true` fails.
8. `selective_threshold_selected == true` fails.
9. Prediction paths containing `holdout` or `test` fail.
10. `downstream_primary_profile_id` is preferred over `selected_profile_id`.
11. `retain_default_lgbm_04` status resolves primary profile to
    `default_lgbm_04`.
12. Older decision records without preferred fields can still pass when hard
    required fields and pooled artifacts are valid.
13. Missing train class counts and missing dummy prediction artifacts block
    same-row stratified dummy computation.
14. A fake bundle that includes `train_class0_n` and `train_class1_n` can build
    deterministic train-derived same-row dummy predictions.
15. `selective_retained_indices` uses confidence-descending and string-cast
    sample-id ascending tie-breaks.
16. `same_row_stratified_dummy_predict` is deterministic for the same counts and
    seed.
17. `ticker_stratified_random_abstention` preserves retained counts by ticker in
    every repeat.
18. `calibration_bins` writes empty uniform bins with `bin_count == 0`, and
    `ece_from_bins` skips those bins.
19. Perfectly calibrated toy inputs return ECE near zero within
    `FLOAT_TOLERANCE`.
20. Actual AURC is greater than or equal to oracle AURC on toy inputs.
21. `evaluate_decision_outcome` applies strict priority, so blocked beats
    not_supported, not_supported beats inconclusive, and inconclusive beats
    promising when conditions overlap.
22. `evaluate_decision_outcome` treats secondary profiles as diagnostics only
    and never replaces `primary_profile_id`.

Use the project Python for local verification:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\notebook06_contract.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook06_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_notebook06_artifact_contract.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook06_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook06_artifact_contract.py
```

## 11. Implementation Order

Implement Notebook 06 in this exact order:

1. Add `scripts/notebook06_contract.py` with all Section 9 functions fully
   implemented. The module must pass `py_compile` before notebook generation.
2. Add `tests/test_notebook06_artifact_contract.py` with fake local artifacts.
3. Add `tests/test_notebook06_static_gate.py`.
4. Add `scripts/create_selective_no_trade_calibration_colab_notebook.py`.
5. Generate `notebooks/06_selective_no_trade_calibration_colab.ipynb` with
   `nbformat`.
6. Run static and artifact-contract tests locally with the project Python.
7. Inspect generated notebook for empty outputs and no execution counts.
8. Upload to Colab only after a Notebook 05 artifact bundle exists that passes
   every Section 4 hard entry gate, including `train_class0_n` and
   `train_class1_n` plus self-consistent `train_positive_rate` in official
   pooled rows.

This order prevents another Notebook 05-style runtime-only discovery. The
artifact contract is tested before the notebook is trusted.

## 12. Acceptance Criteria

Notebook 06 is implementation-ready only when all criteria pass:

1. The technical design in this file is still current.
2. Static tests pass locally.
3. Artifact-contract unit tests pass locally.
4. The generated notebook has no saved outputs.
5. The generated notebook has no execution counts.
6. The generated notebook contains no active `drive.mount` call.
7. The generated notebook contains no active holdout/test path or scoring logic.
8. The generated notebook treats missing sample ids as a hard stop.
9. The generated notebook writes the required output manifest.
10. `scripts/notebook06_contract.py` passes `py_compile`.
11. `scripts/create_selective_no_trade_calibration_colab_notebook.py` passes
    `py_compile`.
12. Contract module unit tests cover selective retention, train-derived dummy,
    ticker-stratified random abstention, ECE, AURC/E-AURC, concentration
    guardrails, and decision-outcome priority.
13. The generated notebook decision record writes
    `selective_threshold_selected == false`.
14. The generated notebook source contains no function or assignment named
    `select_threshold`, `best_threshold`, `optimal_threshold`, or
    `optimal_coverage`.

## 13. Carry Into Notebook 07

Notebook 06 may feed Notebook 07 only as a frozen validation-only artifact.
Notebook 07 may summarize:

```text
primary Notebook 05 profile used by 06
full-coverage Notebook 05 metrics
fixed coverage-grid Notebook 06 metrics
same-row dummy deltas
random-abstention comparison
concentration guardrails
validation-budget caveats
holdout/test remains closed
```

Notebook 07 should read these values from frozen 06 outputs:

```text
primary profile id -> notebook06_run_manifest.json
coverage-grid metrics -> notebook06_coverage_grid.csv
dummy deltas -> notebook06_coverage_grid.csv
random abstention comparison -> notebook06_random_abstention_baselines.csv
concentration guardrails -> notebook06_concentration_guardrails.csv
decision record -> notebook06_decision_record.json
```

Notebook 07 must not use Notebook 06 to add a new model, feature set, threshold,
coverage point, or holdout/test claim.
