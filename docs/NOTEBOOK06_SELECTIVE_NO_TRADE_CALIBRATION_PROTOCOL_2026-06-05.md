# Notebook 06 Selective No-Trade Calibration Protocol - 2026-06-05

Scope: `validation_only`

This protocol is a technical design note for planned Notebook 06:

```text
notebooks/06_selective_no_trade_calibration_colab.ipynb
```

Notebook 06 is not a model-tuning notebook, not a trading backtest, and not a
holdout/test authorization. Its role is to evaluate whether the already selected
Notebook 05 LightGBM profile has a useful prediction-time confidence ranking
under pre-registered coverage levels.

The first Notebook 06 version should be a raw-probability selective readout. It
should not fit a probability calibrator unless a separate, purged train-inner
out-of-fold probability source is available and explicitly authorized.

---

## 1. Source Materials

This document consolidates the current Notebook 06/07 research notes:

| material | role in this protocol |
| --- | --- |
| `docs/research_notes/06_07_literature_materials_2026-06-05.md` | broad selective/no-trade and validation-reuse literature map |
| `artifacts/research_packets/notebook06_07_selective_calibration_research_packet_2026-06-05.md` | initial 06/07 research packet and notebook outline |
| `docs/research_notes/06_07_purged_embargo_cv_materials_2026-06-05.md` | purged/embargoed fold guardrails for future calibration or OOF predictions |
| `docs/research_notes/06_ece_aurc_calibration_metrics_materials_2026-06-05.md` | ECE, Brier, AURC, E-AURC, and risk-coverage metrics |
| `docs/research_notes/06_artifact_contract_implementation_materials_2026-06-05.md` | Notebook 05 to Notebook 06 artifact contract and implementation guidance |
| `docs/research_notes/06_07_intraday_concentration_guardrails_materials_2026-06-05.md` | ticker, date, time-of-day, and overlapping-window concentration checks |
| `docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md` | later Notebook 07 robustness/explainability boundary |
| `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md` | parent protocol: 05 does not choose a selective threshold and keeps holdout/test closed |

Core external sources include:

- Chow (1970), reject option.
- El-Yaniv and Wiener (2010), selective classification foundations.
- Geifman and El-Yaniv (2017), risk-coverage curves and selective prediction.
- Chalkidis and Savani (2021), trading via selective classification.
- Niculescu-Mizil and Caruana (2005), Platt (1999), Zadrozny and Elkan (2002),
  Guo et al. (2017), Naeini et al. (2015), Nixon et al. (2019), and Kumar et al.
  (2018/2019), probability calibration and ECE caveats.
- Cawley and Talbot (2010), White (2000), Bailey et al. backtest-overfitting
  materials, and Lopez de Prado purging/embargoing, validation-reuse guardrails.
- Wood, McInish, and Ord (1985), Heston et al. intraday patterns, and related
  effective-sample-size materials for concentration and dependence caveats.

These sources support a validation-only selective diagnostic. They do not
support profitability, deployment, live trading, or holdout/test conclusions.

---

## 2. Research Question

Given the Notebook 05 selected LightGBM profile, can a pre-registered
prediction-time abstention rule improve directional reliability at an explicit
coverage cost, without choosing a threshold from official validation and without
using holdout/test?

Operationally:

```text
If rows are sorted by LightGBM confidence, do the retained high-confidence rows
show better validation-only macro F1 / balanced accuracy than same-row dummy and
random-abstention baselines, while avoiding concentration in a single ticker,
date, or time-of-day region?
```

---

## 3. Non-Goals

Notebook 06 must not answer:

- whether a final trading threshold has been selected;
- whether the model is profitable, tradable, deployable, or holdout-ready;
- whether selective coverage proves LightGBM is globally superior to LogReg or
  deep models;
- whether a new label, feature set, window size, or model family should replace
  the current route;
- whether conformal prediction guarantees hold for overlapping intraday
  time-series windows;
- whether a fitted calibrator improves official-validation performance, unless
  the calibrator is fit only on a separately authorized train-inner source.

Forbidden wording:

```text
coverage 0.60 is the final trading threshold
high-confidence rows are profitable trades
selective no-trade is validated on holdout
official validation selected the best threshold
conformal risk is guaranteed for this intraday time series
```

---

## 4. Entry Conditions

Notebook 06 may run only after Notebook 05 has been reviewed and has produced a
valid decision record.

Required checks:

1. Notebook 05 scope is `validation_only`.
2. `holdout_test_authorized == false`.
3. Notebook 05 selected profile source is train-inner HPO or default fallback,
   not official-validation-best replacement.
4. Notebook 05 states `selective_threshold_selected == false`.
5. Required official-validation prediction artifacts exist for the selected
   profile.
6. The prediction artifacts include stable sample identifiers.
7. Sample identifiers align with Notebook 05 pooled/per-ticker summary hashes.
8. No holdout/test rows are read, transformed, windowed, scored, summarized, or
   displayed.

If any check fails, Notebook 06 must stop and report the exact missing or
invalid path/key.

---

## 5. Notebook 05 To Notebook 06 Artifact Contract

Notebook 06 requires these Notebook 05 output files:

```text
/content/notebook05_lightgbm_tuning_results/notebook05_entry_decision.json
/content/notebook05_lightgbm_tuning_results/notebook05_decision_record.json
/content/notebook05_lightgbm_tuning_results/notebook05_run_manifest.json
/content/notebook05_lightgbm_tuning_results/notebook05_official_validation_pooled.csv
/content/notebook05_lightgbm_tuning_results/notebook05_official_validation_per_ticker.csv
/content/notebook05_lightgbm_tuning_results/predictions/{profile_id}__seed{seed}.npz
```

The `.npz` prediction artifact must contain:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence
```

Current static review of `scripts/create_lightgbm_tuning_colab_notebook.py`
shows that `save_probability_artifact_05(...)` intends to save:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence
```

However, static search did not find where `dataset["validation_sample_id"]` is
constructed before 05D uses it. This is a Notebook 06 readiness risk. Notebook
06 must treat missing `validation_sample_id` as a hard stop. It must not replace
missing sample ids with row numbers.

Recommended future 05 improvement, if needed before Notebook 06:

```text
validation_sample_id = f"{ticker}|{timestamp}|{window_start_timestamp}|{label_horizon_end_timestamp}"
```

The exact formula can be simpler if it is deterministic and stable, but it must
not include holdout/test values or future information beyond the already defined
label horizon metadata.

---

## 6. Calibration Policy

Notebook 06 first pass:

```text
calibrator fitting = none
probability source = raw Notebook 05 LightGBM prob_up
confidence         = max(prob_up, 1 - prob_up)
official validation = fixed-grid readout only
```

Rationale:

- Notebook 05 currently provides official-validation probabilities, not
  train-inner out-of-fold probabilities.
- Fitting `CalibratedClassifierCV` or isotonic/sigmoid calibration on official
  validation and then reporting the same official validation rows would create a
  validation-reuse problem.
- sklearn calibration APIs are useful references, but their default CV behavior
  is not sufficient for this project unless wrapped in custom chronological,
  per-ticker, purged fold logic.

If a future Notebook 06 calibration extension is authorized:

```text
OOF/calibration source = official training partition only
fold style             = purged chronological expanding-origin
event purge            = no train event interval overlaps calibration/OOF event interval
window policy          = no ticker/day/fold boundary crossing
calibrator fit         = train-inner OOF/calibration rows only
coverage thresholds    = train-inner calibrated confidence or pre-registered grid only
official validation    = readout only
```

Until that source exists, Notebook 06 should report raw probability diagnostics
and selective confidence-ranking diagnostics only.

---

## 7. Fixed Coverage Grid

Use a fixed coverage grid before reading Notebook 05 prediction artifacts:

```text
COVERAGE_GRID = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
MIN_INTERPRETABLE_COVERAGE = 0.30
```

Coverage below 0.30 may be plotted only as an exploratory visual appendix if it
is explicitly labeled as low-coverage diagnostic. It must not drive the decision
record.

For each `(profile_id, seed, coverage)`:

1. Sort rows by `confidence` descending.
2. Break ties by `validation_sample_id` ascending.
3. Retain `ceil(coverage * n)` rows.
4. Compute metrics only on retained rows.
5. Report the implied probability/confidence cutoff as descriptive context.
6. Do not choose a final threshold from the official-validation curve.

---

## 8. Metrics

### 8.1 Full-Row Probability Diagnostics

Compute on all official-validation rows for each selected profile/seed:

```text
brier_score
ece_positive_quantile_10
ece_positive_equal_width_10
top_label_ece_quantile_10
reliability_bin_table
```

Use Brier and ECE as diagnostics only. They are not the primary classifier
metric and are not selection gates.

Important caveats:

- Brier score mixes calibration, refinement/resolution, and class balance.
- ECE depends on binning choices and can hide score/class imbalance.
- ECE alone cannot prove probability calibration.
- Calibration metrics cannot prove trading profitability.

### 8.2 Selective Coverage Metrics

For every fixed coverage point:

```text
coverage_target
coverage_actual
retained_n
abstained_n
selective_macro_f1
selective_balanced_accuracy
selective_accuracy
selective_error = 1 - selective_accuracy
delta_macro_f1_vs_stratified_dummy_same_rows
delta_balanced_accuracy_vs_stratified_dummy_same_rows
delta_macro_f1_vs_always_up_dummy_same_rows
delta_macro_f1_vs_random_abstention
```

### 8.3 Risk-Coverage Summary

Compute:

```text
AURC
oracle_AURC
E_AURC = AURC - oracle_AURC
```

Interpretation:

- AURC/E-AURC evaluate confidence-ranking behavior over coverage.
- They are not probability calibration scores.
- They are not trading metrics.
- They should be reported alongside same-row dummy and random-abstention
  baselines.

---

## 9. Baselines Under Abstention

Notebook 06 must not compare selected high-confidence rows only against the
full-coverage dummy baseline.

For every retained subset:

1. Compute same-row stratified dummy on the retained target rows.
2. Compute same-row always-up dummy on the retained target rows.
3. Compute ticker-stratified random-abstention baselines:
   - retain the same count per ticker as the model retained;
   - repeat across fixed random seeds;
   - compare model-selected rows against randomly retained rows at the same
     coverage and ticker mix.

The same-row dummy answers:

```text
Does the selected subset beat a simple classifier on the exact retained rows?
```

The random-abstention baseline answers:

```text
Does the confidence ranking select better rows than randomly retaining the same
number of rows, preferably with the same ticker mix?
```

Both are required.

---

## 10. Concentration Guardrails

Selective gains are not useful if retained rows or gains are concentrated in a
single ticker, date, or clock-time segment.

Notebook 06 must report concentration diagnostics. These guardrails qualify
wording; they do not select coverage levels.

### 10.1 Ticker Concentration

Report:

```text
retained_n_by_ticker
coverage_by_ticker
delta_macro_f1_vs_dummy_by_ticker
top_ticker_retained_share
top_ticker_gain_share
ticker_entropy_norm
```

Suggested wording downgrades:

```text
warning if top_ticker_retained_share > 0.50
warning if ticker_entropy_norm < 0.70
severe downgrade if top_ticker_retained_share > 0.65
severe downgrade if ticker_entropy_norm < 0.50
severe downgrade if top_ticker_gain_share > 0.70
```

### 10.2 Date Concentration

Report:

```text
top_day_selected_share
top_5_day_selected_share
date_entropy_norm
date_effective_n_entropy
```

Suggested wording downgrades:

```text
warning if top_day_selected_share > max(5 / eligible_day_count, 0.10)
warning if top_5_day_selected_share > 0.40
warning if date_entropy_norm < 0.60
severe downgrade if top_day_selected_share > max(10 / eligible_day_count, 0.20)
```

### 10.3 Time-Of-Day Concentration

Use 30-minute buckets when possible:

```text
time_of_day_bucket
time_of_day_bucket_share
time_bucket_lift
top_time_bucket_share
open_close_concentration_30
open_close_concentration_60
open_close_lift_30
time_bucket_entropy_norm
```

Suggested wording downgrades:

```text
warning if open_close_concentration_30 > 0.30
warning if open_close_lift_30 > 1.5
severe downgrade if open_close_concentration_30 > 0.50 and open_close_lift_30 > 2.0
```

Do not remove open/close rows after seeing that they drive results. Report the
concentration and downgrade the claim.

### 10.4 Overlapping Windows

Notebook 06 should remind the reader that row counts are not independent sample
counts because input windows overlap heavily. Any uncertainty intervals should
be descriptive and preferably grouped by ticker/date/block. Do not present iid
row-bootstrap intervals as confirmatory proof.

---

## 11. Notebook Structure

Recommended stages:

```text
06A - Artifact gate
06B - Prediction frames and probability diagnostics
06C - Fixed coverage grid, same-row dummy, and random abstention
06D - Aggregate seed metrics and AURC/E-AURC
06E - Concentration guardrails
06F - Decision record and allowed wording
06G - Optional Google Drive backup
```

All heavy or result-producing switches should default to `False`:

```text
RUN_06A_ARTIFACT_GATE = False
RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS = False
RUN_06C_FIXED_COVERAGE_GRID = False
RUN_06D_AGGREGATE_AND_RISK_COVERAGE = False
RUN_06E_CONCENTRATION_GUARDRAILS = False
RUN_06F_DECISION_RECORD = False
RUN_06G_BACKUP_TO_GOOGLE_DRIVE = False
BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE = False
```

If a run-all copy is later created for Colab, it should be clearly named and
should require explicit operator acceptance.

---

## 12. Output Artifacts

Notebook 06 should write:

```text
notebook06_artifact_contract_check.json
notebook06_run_manifest.json
notebook06_probability_diagnostics.csv
notebook06_reliability_bins.csv
notebook06_coverage_grid.csv
notebook06_same_row_baselines.csv
notebook06_random_abstention_baselines.csv
notebook06_risk_coverage_summary.csv
notebook06_per_ticker_coverage.csv
notebook06_concentration_guardrails.csv
notebook06_decision_record.json
```

Optional plots:

```text
notebook06_risk_coverage_curve.png
notebook06_delta_vs_dummy_by_coverage.png
notebook06_reliability_diagram.png
notebook06_retained_share_by_ticker.png
notebook06_time_bucket_retained_share.png
```

Plots are explanatory. Tables and JSON records are the primary artifacts.

---

## 13. Decision Record

Notebook 06 should end with one of three validation-only decisions:

### Promising

Allowed only if:

```text
fixed coverage-grid rows show positive delta vs same-row stratified dummy;
model-selected rows beat ticker-stratified random abstention at meaningful coverage;
positive deltas are not concentrated in fewer than 3 of 5 tickers;
date/time/ticker concentration guards do not trigger severe downgrade;
probability diagnostics do not contradict the confidence-ranking story;
holdout_test_authorized remains false.
```

Allowed wording:

```text
Under pre-registered validation-only coverage levels, the selected LightGBM
profile shows a useful confidence-ranking pattern for prediction-time
abstention. This supports a cautious selective-analysis result, not a final
trading threshold or holdout/test claim.
```

### Inconclusive

Use if:

```text
delta vs dummy is mixed;
random-abstention baseline is competitive;
coverage gains appear only below 0.30;
probability diagnostics are weak;
concentration guardrails trigger warnings but not hard failure.
```

### Harmful / Not Supported

Use if:

```text
selected rows do not beat same-row dummy;
random-abstention baseline outperforms model confidence selection;
coverage gains concentrate in one ticker/date/time region;
probability artifacts fail alignment checks;
official validation would be needed to choose a threshold.
```

---

## 14. Relationship To Notebook 07

Notebook 06 feeds Notebook 07 only as a frozen validation-only artifact.

Notebook 07 may summarize:

- full-coverage model results from 05;
- fixed-grid selective rows from 06;
- per-ticker and per-seed robustness;
- validation-budget ledger;
- concentration caveats;
- optional explainability and null-control appendices.

Notebook 07 must not use 06 results to choose a new model, feature set, label,
window size, probability threshold, or coverage point.

---

## 15. Acceptance Criteria

A future Notebook 06 builder is acceptable only if:

1. It contains this protocol or cites it near the top.
2. It asserts `scope == validation_only`.
3. It asserts `holdout_test_authorized == false`.
4. It has no code path that reads, transforms, windows, scores, or summarizes
   holdout/test rows.
5. It treats missing `validation_sample_id` as a hard stop.
6. It uses fixed coverage levels declared before reading probability artifacts.
7. It reports full coverage plus all fixed coverage points.
8. It does not choose a final threshold from official validation.
9. It computes same-row dummy baselines for every retained subset.
10. It computes ticker-stratified random-abstention baselines.
11. It reports Brier, ECE, reliability bins, AURC/E-AURC, and concentration
    guardrails as diagnostics.
12. It does not fit a calibrator unless train-inner purged OOF probabilities are
    separately authorized and available.
13. It labels every result with scope.
14. It writes a decision record with allowed and forbidden wording.

---

## 16. Open Questions

1. Does the final Notebook 05 run actually produce `validation_sample_id` in
   05D prediction artifacts?
2. Should Notebook 05 be repaired before 06 if `validation_sample_id` is absent?
3. Should Notebook 06 include only the selected 05 train-inner profile, or also
   `default_lgbm_04` as context?
4. Should low coverage below 0.30 be omitted entirely or included as
   visualization-only?
5. Should a future calibration extension refit the frozen 05 profile inside the
   official training partition to create purged OOF probabilities?
6. Should Notebook 07 include CPCV/PBO as limitations language only, or as a
   separate diagnostic appendix after all choices are frozen?

Until these are resolved, the safest first Notebook 06 version is:

```text
artifact-only raw-probability selective readout
fixed coverage grid
same-row dummy and random-abstention baselines
probability diagnostics
concentration guardrails
no fitted calibrator
no final threshold
no holdout/test
```
