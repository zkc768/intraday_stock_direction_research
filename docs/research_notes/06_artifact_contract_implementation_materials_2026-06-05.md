# Notebook 06 Artifact Contract And Implementation Materials

Date: 2026-06-05

Scope: KB-ready research note and design material only. No training, no notebook
execution, no holdout/test access, no Notebook 05 edits, no dependency install.

Target follow-up notebook: `notebooks/06_selective_no_trade_calibration_colab.ipynb`

## Short Plan

1. Treat Notebook 06 as an artifact-only selective/no-trade calibration readout.
2. Require Notebook 05 validation probability artifacts before any 06 analysis.
3. Validate 05 -> 06 same-row alignment before computing metrics.
4. Start with dependency-free ECE, fixed coverage grid, same-row dummy,
   ticker-stratified random abstention, AURC, and concentration checks.
5. Defer sklearn calibration fitting, MAPIE, and net:cal unless a later
   pre-registered follow-up explicitly authorizes the extra dependency or
   train-inner calibration path.

## Local Context Inspected

- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`
- `scripts/create_lightgbm_tuning_colab_notebook.py`
- `notebooks/05_lightgbm_tuning_colab.ipynb` by static text search only
- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `artifacts/research_packets/notebook06_07_selective_calibration_research_packet_2026-06-05.md`
- `requirements.txt`
- `docs/ENVIRONMENT.md`

No notebooks were run.

## 05 Current Artifact Can Do

Static review of `scripts/create_lightgbm_tuning_colab_notebook.py` shows that
05D is intended to write LightGBM probability artifacts under:

```text
/content/notebook05_lightgbm_tuning_results/predictions/
```

For each LightGBM `profile_id` and `seed`, `save_probability_artifact_05(...)`
writes:

```text
{profile_id}__seed{seed}.npz
```

The saved `.npz` payload is intended to contain:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence = max(prob_up, 1 - prob_up)
```

The 05D pooled and per-ticker CSV rows also include:

```text
profile_id
profile_role
seed
label_config
horizon_k
threshold_bps
feature_set
window_size
scope = validation_only
validation_sample_id_hash
sample_id_mismatch_count
prediction_artifact
same-row dummy metrics
positive_ticker_count
top_ticker_gain_share
```

If 05D completes successfully and the `.npz` files exist, Notebook 06 can use
the current design for:

- full validation probability diagnostics on LightGBM rows;
- fixed coverage-grid selective metrics;
- same-row dummy comparisons using `y_true` plus retained sample ids;
- ticker concentration checks using `ticker`;
- timestamp-aware display and basic chronological sanity checks using
  `timestamp`;
- profile/seed linkage by joining `.npz` files to
  `notebook05_official_validation_pooled.csv` through `prediction_artifact`, or
  by parsing the `{profile_id}__seed{seed}.npz` filename.

## 05 Current Artifact Cannot Do Yet

The current static code has two important contract gaps:

1. `validation_sample_id` is read by `save_probability_artifact_05(...)`, but
   static search did not find any assignment of `dataset["validation_sample_id"]`
   in `get_dataset(...)`. In the generated 05 notebook, `get_dataset(...)`
   returns `validation_owner` and `validation_timestamp`, but not
   `validation_sample_id`. If no hidden runtime mutation adds this key, 05D
   probability saving will fail with an exact missing key:

```text
dataset["validation_sample_id"]
```

2. The `.npz` payload does not store `profile_id`, `profile_role`, `seed`,
   `label_config`, `feature_set`, `window_size`, or `scope` inside the file.
   Those fields are available in 05D CSV rows, but the `.npz` is not
   self-describing if copied alone.

Because the user boundary forbids modifying Notebook 05 here, this note treats
both as Notebook 06 stop conditions rather than applying a repair.

Notebook 06 also cannot perform train-inner calibration from current 05D
official-validation `.npz` files alone. It would need train-inner out-of-fold
probabilities or a separately pre-registered calibration split. Fitting a
calibrator on official validation and reporting the same official validation
rows would violate the route's selection discipline.

## Minimum 05 -> 06 Artifact Contract

Notebook 06 must require these paths:

```text
/content/notebook05_lightgbm_tuning_results/notebook05_decision_record.json
/content/notebook05_lightgbm_tuning_results/notebook05_run_manifest.json
/content/notebook05_lightgbm_tuning_results/notebook05_official_validation_pooled.csv
/content/notebook05_lightgbm_tuning_results/notebook05_official_validation_per_ticker.csv
/content/notebook05_lightgbm_tuning_results/notebook05_official_validation_summary.csv
/content/notebook05_lightgbm_tuning_results/predictions/
```

Minimum fields in `notebook05_decision_record.json`:

```text
scope == validation_only
holdout_test_authorized == false
selective_threshold_selected == false
selected_profile_id
selected_profile_source
```

Minimum fields in `notebook05_run_manifest.json`:

```text
scope == validation_only
holdout_test_authorized == false
official_validation_lightgbm_rows_completed
official_validation_dummy_rows_completed
```

Minimum fields in `notebook05_official_validation_pooled.csv`:

```text
profile_id
profile_role
seed
scope
label_config
horizon_k
threshold_bps
feature_set
window_size
ticker_or_pooled
n
validation_n
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
```

Minimum fields in each LightGBM prediction `.npz`:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence
```

Preferred self-describing fields for future 05 artifacts, not required if the
CSV join is reliable:

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
holdout_test_authorized
```

Alignment checks:

- every `prediction_artifact` path for LightGBM pooled rows must exist;
- every `.npz` must have equal-length arrays for all required arrays;
- `y_true` must be binary 0/1 with no NaN-like object values;
- `prob_up` must be finite and in `[0, 1]`;
- `confidence` must equal `max(prob_up, 1 - prob_up)` within a small tolerance;
- `validation_sample_id` must be unique within each artifact;
- all compared profile/seed artifacts must have the same ordered
  `validation_sample_id` array or must be inner-joined by sample id before
  any same-row metric;
- `timestamp` must parse as datetime and be nondecreasing within each ticker;
- `scope` from the CSV and manifests must be `validation_only`;
- no 06 input path may include holdout/test prediction artifacts.

Stop conditions:

- stop if any required 05 path is missing and report the exact missing path;
- stop if `holdout_test_authorized` is not explicitly false;
- stop if `selective_threshold_selected` is true in 05;
- stop if `selected_profile_id` is empty or absent;
- stop if selected profile has no prediction artifact rows;
- stop if any `.npz` is missing required fields;
- stop if `validation_sample_id` is missing, duplicated inside one artifact, or
  cannot be aligned across profile/seed rows;
- stop if `prob_up` is not finite or is outside `[0, 1]`;
- stop if selected rows at any coverage have too few total rows or too few rows
  for at least three tickers under the pre-registered guard;
- stop if any code path tries to read, transform, window, score, summarize, or
  display holdout/test rows.

Exact missing-path reporting format:

```text
Notebook 06 cannot start because required Notebook 05 artifact is missing:
<exact path>
```

Exact missing-field reporting format:

```text
Notebook 06 cannot start because required field is missing:
artifact=<exact path>
field=<field name>
```

Google Drive backup recommendation:

- keep 05 outputs first in local runtime:
  `/content/notebook05_lightgbm_tuning_results`;
- if backing up, upload required CSV/JSON files and all prediction `.npz` files;
- include a Drive backup manifest with local path, Drive file id, Drive file
  name, file size, and created UTC;
- do not rely on mounted Drive folders as the authoritative input in the
  default 06 setup;
- do not copy raw OHLCV, train matrices, validation feature matrices, or any
  holdout/test material as part of the 06 prediction-artifact backup.

## Dependency And API Fit

Pinned local dependencies from `requirements.txt` and live metadata:

```text
scikit-learn==1.4.2
lightgbm==4.6.0
mapie=MISSING
netcal=MISSING
```

### Suitable Now

- `lightgbm.LGBMClassifier.predict_proba`: suitable for generating `prob_up`.
  LightGBM 4.6.0 documents `predict_proba` as returning predicted probability
  for each class. Because 05 uses `class_weight="balanced"`, raw probabilities
  should be treated as confidence scores until calibration diagnostics are
  checked; LightGBM warns that class weighting can produce poor individual
  probability estimates and suggests calibration.
- `sklearn.metrics.brier_score_loss`: available in sklearn 1.4.2. Suitable as
  a probability-quality diagnostic for binary `y_true` and `prob_up`.
- `sklearn.calibration.calibration_curve`: available in sklearn 1.4.2. Suitable
  for reliability-table data, with explicit bin counts added locally.
- `sklearn.calibration.CalibrationDisplay`: available in sklearn 1.4.2.
  Suitable for plots from predictions.
- dependency-free ECE: recommended for Notebook 06 first pass because sklearn
  does not provide a core ECE metric.
- dependency-free risk-coverage/AURC and random-abstention baseline: recommended
  because they only need numpy/pandas/sklearn metrics already pinned.

### Conditionally Suitable

- `CalibratedClassifierCV`: available in sklearn 1.4.2, but do not use default
  `cv=None` for this project. The docs say default binary/multiclass CV uses
  5-fold `StratifiedKFold`, which violates the project's chronological split
  discipline. It is suitable only with explicit chronological train-inner splits
  or `cv="prefit"` where the estimator training data and calibrator data are
  provably disjoint. Isotonic calibration should be avoided with small
  calibration samples because sklearn warns it can overfit with too few rows.
- `TimeSeriesSplit`: available in sklearn 1.4.2 and useful as a starting point,
  but insufficient by itself for pooled multi-ticker windows. Notebook 06 would
  still need ticker ownership, day-boundary, split-boundary, and label-horizon
  guards.

### Not Suitable Under Current Pins

- `TunedThresholdClassifierCV`: not available in local `scikit-learn==1.4.2`.
  It appears in sklearn 1.5 docs as added in version 1.5. Even if upgraded
  later, default CV uses stratified K-fold and would be unsafe for the current
  time-series route unless replaced with explicit chronological splits.
- MAPIE: not installed. Do not add it silently. Even if installed later, MAPIE
  risk-control/conformal methods require careful exchangeability/time-series
  caveats and should be an optional appendix, not the first 06 implementation.
- net:cal: not installed. It offers calibration metrics and plotting, but the
  current project can implement the needed ECE and reliability tables without
  adding a new dependency.

## Must / Useful / Optional / Risky Source List

### Must

- scikit-learn 1.4 probability calibration user guide:
  https://scikit-learn.org/1.4/modules/calibration.html
  Use for reliability diagrams, calibrator-data separation, sigmoid/isotonic
  calibration, and Brier caveats.
- scikit-learn 1.4 `CalibratedClassifierCV`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  Use as an API reference only with custom chronological CV or disjoint
  `prefit` calibration data.
- scikit-learn 1.4 `CalibrationDisplay`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.CalibrationDisplay.html
  Use for reliability diagrams from predictions.
- scikit-learn 1.4 `calibration_curve`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.calibration.calibration_curve.html
  Use for reliability table values; add bin counts locally.
- scikit-learn 1.4 `brier_score_loss`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.brier_score_loss.html
  Use for binary probability-quality diagnostics, not as the primary
  directional metric.
- LightGBM 4.6.0 `LGBMClassifier`:
  https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html
  Use for `predict_proba` behavior and the class-weight probability caveat.
- scikit-learn 1.4 `DummyClassifier`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.dummy.DummyClassifier.html
  Use for same-row stratified dummy baseline convention.
- El-Yaniv and Wiener, selective classification / risk-coverage:
  https://jmlr.org/papers/v11/el-yaniv10a.html
- Geifman and El-Yaniv, selective classification for deep neural networks:
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- Chalkidis and Savani, trading via selective classification:
  https://arxiv.org/abs/2110.14914

### Useful

- scikit-learn 1.4 probability calibration example:
  https://scikit-learn.org/1.4/auto_examples/calibration/plot_calibration.html
  Useful tutorial for sigmoid/isotonic comparison and Brier reporting.
- scikit-learn 1.4 comparison of calibration classifiers:
  https://scikit-learn.org/1.4/auto_examples/calibration/plot_compare_calibration.html
  Useful for probability histograms plus calibration curves.
- scikit-learn 1.4 `TimeSeriesSplit`:
  https://scikit-learn.org/1.4/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
  Useful conceptually, but wrap with project-specific ticker/day/horizon guards.
- Niculescu-Mizil and Caruana, probability calibration:
  https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
- Guo et al., calibration of modern neural networks:
  https://proceedings.mlr.press/v70/guo17a.html
- MAPIE binary risk-control docs:
  https://mapie.readthedocs.io/en/latest/generated/mapie.risk_control.BinaryClassificationController.html
  Useful for future appendix only after dependency approval.

### Optional

- net:cal framework:
  https://github.com/efs-opensource/calibration-framework
  Optional plotting/metric framework; not needed for first 06.
- net:cal scaling API:
  https://efs-opensource.github.io/calibration-framework/build/html/_autosummary/netcal.scaling.html
- Conformal risk control:
  https://arxiv.org/abs/2208.02814
- MAPIE theoretical binary classification docs:
  https://mapie.readthedocs.io/en/latest/theoretical_description_binary_classification.html

### Risky For Current Scope

- sklearn 1.5+ `TunedThresholdClassifierCV`:
  https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
  Risky because it is not in the current pinned sklearn 1.4.2 environment and
  its default threshold-tuning CV is not chronology-safe.
- sklearn current threshold-tuning guide:
  https://scikit-learn.org/stable/modules/classification_threshold.html
  Useful warning material, but the examples are ordinary supervised threshold
  tuning and should not be copied into 06.
- arbitrary official-validation threshold search over many probability cutoffs.
  This is the main failure mode Notebook 06 must avoid.
- installing MAPIE or net:cal during 06 without explicit authorization.

## Recommended 06 First-Version Implementation Route

### 06A - Artifact Load And Gate

- Load only 05 CSV/JSON/NPZ artifacts.
- Assert `scope == validation_only`.
- Assert `holdout_test_authorized == false`.
- Assert `selective_threshold_selected == false`.
- Resolve the selected profile from `notebook05_decision_record.json`.
- Load only prediction artifacts for the selected profile plus optionally
  `default_lgbm_04` for context.
- Stop on missing exact paths or fields.

### 06B - Same-Row Prediction Frame

Create one long dataframe with:

```text
sample_id
profile_id
profile_role
seed
ticker
timestamp
y_true
y_pred
prob_up
confidence
source_prediction_artifact
scope
```

If `.npz` lacks `profile_id` and `seed`, derive them from the joined pooled CSV
row or filename and record:

```text
metadata_source = "notebook05_official_validation_pooled_csv"
```

If `sample_id` is missing, stop. Do not silently replace it with row number
unless a new Notebook 05 repair or separate protocol explicitly authorizes a
deterministic id formula.

### 06C - Probability Diagnostics

Compute full-row diagnostics per `(profile_id, seed)` and pooled over seeds:

```text
brier_score
ece_equal_frequency_10
ece_equal_width_10
reliability_bins_equal_frequency
reliability_bins_equal_width
probability_histogram_by_correctness
```

ECE formula:

```text
ece = sum_over_bins((bin_n / n) * abs(bin_accuracy - bin_confidence_mean))
```

Use equal-frequency bins as the primary diagnostic because LightGBM probabilities
may cluster tightly; include equal-width bins as a secondary view when enough
nonempty bins exist.

### 06D - Fixed Coverage Grid

Pre-register:

```text
coverage_grid = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
minimum_coverage_for_interpretation = 0.30
minimum_tickers_with_rows = 3
minimum_rows_per_ticker_for_interpretation = pre-registered constant
```

For each `(profile_id, seed, coverage_target)`:

1. sort rows by `confidence` descending;
2. retain `ceil(coverage_target * n)` rows;
3. compute selected-set metrics on retained rows;
4. compute the same-row stratified dummy and always-up dummy on exactly those
   retained sample ids;
5. report the implied confidence/probability cutoff, but do not select it as a
   final threshold.

Required output columns:

```text
profile_id
seed
coverage_target
coverage_actual
confidence_threshold
n_selected
n_abstained
pooled_macro_f1_selected
pooled_balanced_accuracy_selected
pooled_accuracy_selected
dummy_macro_f1_selected
delta_macro_f1_vs_dummy_selected
always_up_macro_f1_selected
delta_macro_f1_vs_always_up_selected
random_abstention_macro_f1_mean
delta_macro_f1_vs_random_abstention
per_ticker_positive_delta_count
top_ticker_selected_share
top_ticker_gain_share
selected_row_ticker_entropy
ece_selected_equal_frequency_10
brier_selected
scope = validation_only
```

### 06E - Ticker-Stratified Random-Abstention Baseline

For each model-selected retained count by ticker, sample the same number of rows
uniformly at random within each ticker from the full validation prediction frame.
Repeat with fixed seeds, for example:

```text
RANDOM_ABSTENTION_REPEATS = 100
RANDOM_ABSTENTION_BASE_SEED = 260606
```

Score the fixed model predictions on the randomly retained rows. This tests
whether high-confidence row selection beats a ticker-matched random retained
subset, not whether the model beats a dummy classifier.

Also keep the same-row dummy baseline because it answers a different question:
whether the selected subset beats a naive label generator on the exact same
target rows.

### 06D/06E - AURC And Concentration

Risk definition for AURC:

```text
selective_error = 1 - selected_accuracy
```

Compute AURC as trapezoidal area under `selective_error` versus
`coverage_actual` over the pre-registered coverage grid. Lower AURC means the
confidence ranking retains lower-error samples earlier. Report that AURC is a
ranking/selective-risk diagnostic, not a profitability metric.

Concentration guards:

```text
per_ticker_selected_n
per_ticker_coverage
per_ticker_delta_macro_f1_vs_dummy
per_ticker_positive_delta_count
top_ticker_selected_share
top_ticker_gain_share
selected_row_ticker_entropy
```

Interpretation guard:

```text
Do not make a positive selective claim if gains are driven by fewer than three
tickers or if selected rows concentrate in one ticker beyond the pre-registered
limit.
```

### 06F - Decision Record

Write:

```text
notebook06_artifact_contract_check.json
notebook06_probability_diagnostics.csv
notebook06_coverage_grid.csv
notebook06_same_row_baselines.csv
notebook06_per_ticker_coverage.csv
notebook06_random_abstention_baselines.csv
notebook06_risk_coverage_summary.csv
notebook06_concentration_guardrails.csv
notebook06_decision_record.json
notebook06_run_manifest.json
```

Decision record allowed outcomes:

```text
selective_analysis_not_started_missing_05_probability_artifact
selective_analysis_blocked_artifact_contract_failure
selective_analysis_inconclusive
selective_analysis_harmful
selective_analysis_promising_validation_only
```

Allowed wording:

```text
Notebook 06 evaluates prediction-time abstention under validation_only scope
using pre-specified coverage levels and same-row baselines. It does not select
a final trading threshold and does not use holdout/test data.
```

Forbidden wording:

```text
The selective threshold is final.
The high-confidence subset is tradable.
The strategy is profitable.
The threshold is holdout-ready.
The model is safe for deployment.
```

### 06G - Optional Google Drive Backup

Drive backup is an optional artifact-copy stage after the local result folder is
complete. It should create timestamped Drive files and a backup manifest rather
than overwriting earlier artifacts.

## Open Issues

1. Notebook 05 probability saving appears to require
   `dataset["validation_sample_id"]`, but static review did not find that field
   created in `get_dataset(...)`. Notebook 06 should fail closed until the 05
   artifact exists and passes field checks.
2. Current `.npz` artifacts are not self-describing for profile/seed/scope.
   Notebook 06 can join through `notebook05_official_validation_pooled.csv`, but
   future 05 artifacts should put metadata inside the `.npz` too.
3. Current 05 probability artifacts are official-validation predictions only.
   They support raw probability diagnostics and fixed-grid selective readout,
   but not calibration fitting without a separate train-inner calibration source.
4. MAPIE and net:cal are not installed under the pinned local environment.
   They should stay deferred unless explicitly approved.

## KB Summary

Notebook 06 should begin as a conservative artifact-only selective
classification notebook. The strongest first implementation is dependency-free:
validate 05 probability artifacts, compute reliability/ECE/Brier diagnostics,
evaluate a fixed coverage grid, compare selected rows against same-row dummy and
ticker-stratified random abstention, report AURC and concentration, and stop
without selecting a final threshold. Under current pinned dependencies,
`CalibratedClassifierCV` is available but unsafe with default CV, sklearn
`TunedThresholdClassifierCV` is unavailable, and MAPIE/net:cal are not
installed. The immediate blocker to a clean 05 -> 06 contract is that static
review does not find `validation_sample_id` creation in the 05 dataset, even
though 05D probability saving requires it.
