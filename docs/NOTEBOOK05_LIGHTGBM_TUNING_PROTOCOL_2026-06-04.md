# Notebook 05 LightGBM Tuning Protocol

Date: 2026-06-04

Scope: `validation_only`

This is a technical research protocol, not a holdout/test authorization, not a
paper-result claim, and not a model-family expansion plan. It freezes the
Notebook 05 design after Notebook 04D operator routing selected:

```text
Exit A - Proceed To 05 LightGBM Tuning
```

Notebook 05 integrates the Notebook 04D manual decision as its entry gate. It
then runs LightGBM hyperparameter search only inside the training partition and
uses the official validation partition only for a small, pre-registered
confirmation surface.

Version log:

- 1.0 (2026-06-04): Initial Notebook 05 protocol with Notebook 04D entry gate,
  train-inner LightGBM HPO, and official-validation finalist confirmation.

---

## 1. Purpose

Notebook 05 answers one narrow question:

```text
Given the official Stage 0 candidate and the Notebook 04D Exit A decision, can
LightGBM obtain a stable validation-only improvement from pre-registered
train-inner hyperparameter tuning without repeatedly searching the official
validation partition?
```

Notebook 05 does not answer:

- whether any final holdout/test result improved;
- whether a new label, threshold, feature set, or window should replace the
  official candidate;
- whether MS-DLinear+TCN, TCN, LSTM, GRU, RandomForest, XGBoost, CatBoost,
  PatchTST, DeepLOB, NLP/news, or external-market models should be added;
- whether selective confidence thresholds should become a final strategy;
- whether tuned LightGBM is generally superior to LogReg or deep sequence
  models.

---

## 2. Required 04D Entry Gate Inside Notebook 05

Notebook 05 begins with a read-only 04D import and operator-acceptance gate.
This moves the manual 04D decision into the 05 workflow without rewriting
Notebook 04 artifacts.

Required Notebook 04 artifacts:

```text
notebook04_context_checks.json
notebook04_summary.csv
notebook04_selective_coverage.csv
notebook04_decision_matrix.csv
notebook04_run_manifest.json
```

Notebook 05 must expose these entry variables near the top:

```text
RUN_05A_04D_ENTRY_GATE = False
OPERATOR_SELECTED_EXIT = ""
OPERATOR_ACCEPTS_EXIT_A = False
```

The operator may proceed only by setting:

```text
RUN_05A_04D_ENTRY_GATE = True
OPERATOR_SELECTED_EXIT = "Exit A - Proceed To 05 LightGBM Tuning"
OPERATOR_ACCEPTS_EXIT_A = True
```

05A must verify all of the following before any HPO or model fit:

```text
notebook04_context_checks.scope == "validation_only"
notebook04_context_checks.holdout_test_authorized == false
notebook04_context_checks.official_candidate ==
  h03_bps1p5 + price_volume_time + window_size=20
notebook04_summary contains exactly one lightgbm summary row
notebook04_summary.lightgbm.basic_gate_pass == true
notebook04_summary.lightgbm.fresh_seed_stability_tag in
  {"confirmed_or_improved", "marginal_drop_note_only"}
notebook04_decision_matrix includes the Exit A row
all Notebook 04D exits have holdout_test_authorized == false
OPERATOR_SELECTED_EXIT is exactly Exit A
OPERATOR_ACCEPTS_EXIT_A is true
```

05A writes:

```text
notebook05_entry_decision.json
```

with at least:

```json
{
  "scope": "validation_only",
  "entry_source": "notebook04_04d_decision_gate",
  "operator_selected_exit": "Exit A - Proceed To 05 LightGBM Tuning",
  "operator_accepts_exit_a": true,
  "holdout_test_authorized": false,
  "hpo_authorized": true,
  "authorized_model_family": "lightgbm",
  "authorized_candidate": {
    "label_config": "h03_bps1p5",
    "horizon_k": 3,
    "threshold_bps": 1.5,
    "feature_set": "price_volume_time",
    "window_size": 20
  }
}
```

If any assertion fails, Notebook 05 must stop before fitting any model. Do not
catch and ignore entry-gate failures. Do not silently choose another exit.

---

## 3. Fixed Candidate And Data Boundary

Notebook 05 uses only the official candidate:

```text
label_config  = h03_bps1p5
horizon_k     = 3
threshold_bps = 1.5
feature_set   = price_volume_time
window_size   = 20
```

The five tickers remain:

```text
CSCO, JPM, KO, MSFT, WMT
```

Notebook 05 inherits the active raw-data-first Colab boundary:

1. Download the five approved raw ticker files into the local Colab runtime.
2. Do not import `intraday_research`, prior notebooks, `baseline_helpers`, or
   archived helper packages as the active path.
3. Do not read, transform, window, score, summarize, display, or otherwise use
   holdout/test rows.
4. Holdout/test may appear only as a closed boundary marker for invalidating
   validation-edge labels.
5. Fit preprocessing statistics on training rows only.
6. Build windows per ticker, per split, and per trading day.
7. Input windows and label horizons must not cross ticker, split, or
   trading-day boundaries.
8. Features may use only current or earlier completed bars.

Notebook 05 must not change:

- label construction;
- no-trade threshold;
- feature columns;
- feature timing;
- scaler or imputer policy;
- split dates;
- window size;
- model family;
- evaluation metrics.

---

## 4. Baseline And Model-Family Boundary

Notebook 02 through Notebook 04 already rebuilt the baseline and model-family
surface for the active route.

Notebook 05 keeps only these roles:

| role | model/profile | use in Notebook 05 |
|---|---|---|
| gate baseline | `stratified_dummy` | mandatory comparison on the same official validation rows |
| sanity baseline | `always_up_dummy` | report-only directional baseline |
| tuning baseline | `default_lgbm_04` | fixed default from Notebook 04 |
| tuned candidates | `lightgbm_trial_*` | train-inner HPO finalists only |

Notebook 05 may read Notebook 04 results for context, but it must not refit or
retune:

- `logreg`;
- `standalone_tcn`;
- `ms_dlinear_tcn`;
- `vanilla_lstm`;
- `simple_gru`;
- `standard_dlinear`.

Do not add:

- RandomForest;
- XGBoost;
- CatBoost;
- PatchTST;
- DeepLOB;
- external market features;
- news, sentiment, or NLP features.

Those belong only to a later, separately pre-registered comparison or ablation
notebook if the operator explicitly approves one.

---

## 5. Default LightGBM Baseline

Notebook 05 must always re-run or import a same-row default LightGBM baseline
for comparison. The default profile is the Notebook 04 LightGBM profile:

```text
profile_id       = default_lgbm_04
boosting_type    = gbdt
n_estimators     = 200
learning_rate    = 0.03
max_depth        = 6
num_leaves       = 31
subsample        = 0.9
subsample_freq   = 1
colsample_bytree = 0.9
class_weight     = balanced
input_view       = flattened window
```

All tuned profiles must be compared against `default_lgbm_04`, `stratified_dummy`,
and `always_up_dummy` on identical official validation sample ids.

---

## 6. Overall 05 Structure

Notebook 05 has five parts:

```text
05A  04D entry gate and context import
05B  train-inner chronological HPO
05C  finalist selection from train-inner HPO only
05D  official-validation confirmation of default + finalists
05E  decision record and allowed wording
```

Default switches:

```text
RUN_05A_04D_ENTRY_GATE = False
RUN_05B_TRAIN_INNER_HPO = False
RUN_05C_SELECT_FINALISTS = False
RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION = False
RUN_05E_DECISION_RECORD = False
BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE = False
```

No cell should fit a model unless the corresponding `RUN_*` switch is true and
all earlier required artifacts exist.

---

## 7. Train-Inner Fold Design

Notebook 05 uses train-inner validation only for HPO. It does not use the
official validation partition to choose hyperparameters.

Fixed train-inner design:

```text
INNER_FOLD_COUNT = 3
INNER_FOLD_STYLE = chronological_expanding_origin
INNER_SPLIT_UNIT = calendar_date_per_ticker
INNER_PURGE_HORIZON_BARS = 3
INNER_WINDOW_CROSS_BOUNDARY_POLICY = forbidden
```

Each inner fold must be constructed inside the official training partition:

```text
fold 1: earliest train dates -> next train-inner validation block
fold 2: larger train prefix  -> next train-inner validation block
fold 3: largest train prefix -> final train-inner validation block
```

Rules:

1. Split each ticker chronologically first.
2. Assign inner train/inner validation blocks by calendar date within each
   ticker.
3. Pool rows only after per-ticker fold assignment.
4. Fit preprocessing on inner-train rows only for each fold.
5. Transform only inner-train and inner-validation rows for that fold.
6. Do not use official validation rows in inner folds.
7. Do not let input windows or label horizons cross inner train/inner validation
   boundaries.
8. Drop or invalidate boundary samples whose label horizon reaches into the
   next fold.
9. Report fold-level sample counts before fitting.

If any inner fold has only one class in its validation target, Notebook 05 must
stop and report the fold id and exact class counts. Do not merge folds after
seeing performance.

---

## 8. HPO Method And Budget

Notebook 05 uses one fixed random-search budget:

```text
HPO_METHOD = random_search
HPO_BUDGET = 100
HPO_RNG_SEED = 260605
INNER_FOLD_COUNT = 3
MAX_FIT_ROWS_BEFORE_CONFIRMATION = 300
```

The budget means:

```text
100 hyperparameter trials x 3 train-inner folds = 300 train-inner LightGBM fits
```

Do not extend the budget after seeing results. Do not shrink the budget after
seeing early weak results. If runtime fails before all 100 trials finish, the
run is incomplete unless the failure is deterministic and a new protocol
explicitly changes the budget before further execution.

Trial ids must be generated before fitting:

```text
trial_id = lgbm_hpo_000 ... lgbm_hpo_099
```

The full sampled search manifest must be written before the first trial fit:

```text
notebook05_hpo_search_manifest.csv
```

---

## 9. Search Space

Fixed parameters:

```text
boosting_type = gbdt
objective = binary
input_view = flattened window
class_weight = balanced
verbosity = -1
```

Search parameters:

| parameter | distribution |
|---|---|
| `learning_rate` | log-uniform in `[0.005, 0.08]` |
| `max_depth` | categorical `{3, 4, 5, 6, 8, -1}` |
| `num_leaves` | categorical `{7, 15, 31, 63}` |
| `min_child_samples` | categorical `{20, 50, 100, 200, 400}` |
| `subsample` | uniform in `[0.50, 1.00]` |
| `subsample_freq` | fixed `1` |
| `colsample_bytree` | uniform in `[0.50, 1.00]` |
| `reg_alpha` | zero-or-log-uniform: `0` or log-uniform in `[1e-4, 10]` |
| `reg_lambda` | zero-or-log-uniform: `0` or log-uniform in `[1e-4, 20]` |
| `min_split_gain` | uniform in `[0.0, 0.10]` |
| `max_estimators` | categorical `{400, 800, 1200, 1600, 2000}` |
| `early_stopping_rounds` | fixed `50` during inner folds only |

Constraint:

```text
if max_depth > 0:
  num_leaves <= 2 ** max_depth
```

If a sampled trial violates the constraint, resample that trial before fitting
and keep the same `trial_id`.

Official validation must not be used for early stopping. For official
validation confirmation, the final number of estimators for a finalist is:

```text
final_n_estimators =
  median(best_iteration across the successful inner folds for that finalist)
```

rounded to the nearest positive integer and clipped to the sampled
`max_estimators`.

---

## 10. Inner-HPO Metrics And Selection Criterion

Each `(trial_id, inner_fold_id)` row must report:

```text
trial_id
inner_fold_id
profile_id
fold_train_n
fold_validation_n
macro_f1
balanced_accuracy
accuracy
stratified_dummy_macro_f1
stratified_dummy_balanced_accuracy
delta_macro_f1_vs_stratified_dummy
always_up_dummy_macro_f1
delta_macro_f1_vs_always_up_dummy
best_iteration
fit_seconds
predict_seconds
run_failed
failure_reason
scope = train_inner_validation
```

Each `trial_id` summary must report:

```text
inner_macro_f1_mean
inner_macro_f1_std
inner_macro_f1_min
inner_macro_f1_max
inner_balanced_accuracy_mean
inner_stratified_dummy_macro_f1_mean
inner_delta_macro_f1_vs_stratified_dummy_mean
inner_delta_macro_f1_vs_stratified_dummy_min
inner_lcb_macro_f1
inner_positive_fold_count
inner_successful_fold_count
median_best_iteration
```

Primary train-inner ranking key:

```text
inner_lcb_macro_f1
```

The fold-level LCB is:

```text
inner_lcb_macro_f1 =
  inner_macro_f1_mean
  - t_critical_one_sided_95(successful_fold_count)
    * inner_macro_f1_std / sqrt(successful_fold_count)
```

Eligibility gate for finalists:

```text
inner_successful_fold_count == 3
inner_delta_macro_f1_vs_stratified_dummy_mean > 0
inner_delta_macro_f1_vs_stratified_dummy_min >= -0.002
inner_positive_fold_count >= 2
median_best_iteration >= 20
```

Tie break:

```text
1. higher inner_lcb_macro_f1
2. lower inner_macro_f1_std
3. smaller num_leaves
4. smaller max_depth when max_depth > 0
5. lower median_best_iteration
6. smaller trial_id lexical order
```

---

## 11. Finalist Rule

Notebook 05 selects finalists only from train-inner HPO:

```text
N_FINALISTS = 5
FINALIST_RULE = top 5 eligible trials by the tie-broken train-inner ranking
PRIMARY_SELECTION = train_inner_winner
```

The `train_inner_winner` is the rank-1 finalist by the train-inner rule.

Official validation may reject or qualify the train-inner winner, but it must
not replace it with the official-validation-best finalist. Reporting the
official-validation-best row is allowed only as calibration context.

If fewer than five trials pass the finalist gate:

- keep all eligible trials as finalists;
- report `n_finalists_available`;
- do not loosen the gate after seeing results.

If zero trials pass the finalist gate:

- skip official-validation tuned finalist confirmation;
- run or import only `default_lgbm_04` plus dummy baselines for context;
- write `notebook05_decision_record.json` with
  `decision = "no_train_inner_hpo_candidate"`.

---

## 12. Official-Validation Confirmation

05D evaluates only:

```text
default_lgbm_04
train_inner_winner
the remaining finalists up to N_FINALISTS = 5
stratified_dummy
always_up_dummy
```

Fresh seeds:

```text
606, 707, 808, 909, 1010
```

Canonical official-validation contact:

```text
6 LightGBM profiles x 5 seeds = 30 fitted LightGBM validation rows
2 dummy baselines x 5 seeds = 10 baseline rows
```

If `default_lgbm_04` is identical to one sampled finalist, implementation may
de-duplicate the physical fit. Reporting must still show the default profile
and the finalist profile as separate comparison roles.

Official validation confirmation must fit each LightGBM profile on the full
official training partition only, using `final_n_estimators` from train-inner
HPO for tuned finalists. It must evaluate on the official validation partition.
It must not use official validation for early stopping.

Each official-validation row must report:

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
positive_ticker_count
top_ticker_gain_share
fit_seconds
predict_seconds
run_failed
failure_reason
```

Summary rows must report:

```text
profile_id
profile_role
seed_count
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
stratified_dummy_macro_f1_mean
delta_macro_f1_vs_stratified_dummy_mean
always_up_dummy_macro_f1_mean
delta_macro_f1_vs_always_up_dummy_mean
delta_macro_f1_vs_default_lgbm_04
positive_ticker_count
top_ticker_gain_share
official_validation_rank_by_macro_f1
selected_by_train_inner
selected_by_official_validation
```

`selected_by_train_inner` is true only for `train_inner_winner`.
`selected_by_official_validation` may identify the highest official-validation
row for transparency, but it must not define the selected tuned model.

---

## 13. Promotion, Rejection, And Stop Rules

The primary selected candidate is always:

```text
train_inner_winner
```

Official validation assigns one of these statuses:

```text
promote_train_inner_winner
retain_default_lgbm_04
no_practical_tuning_gain
validation_rejects_train_inner_winner
no_train_inner_hpo_candidate
```

Promote `train_inner_winner` only if all conditions hold:

```text
train_inner_winner official-validation run_failed = False for all five seeds
delta_macro_f1_vs_stratified_dummy_mean > 0
macro_f1_lcb_95 > stratified_dummy_macro_f1_mean
delta_macro_f1_vs_default_lgbm_04 >= 0.001
positive_ticker_count >= 4
top_ticker_gain_share <= 0.35
macro_f1_std <= max(0.0025, 3 * default_lgbm_04_macro_f1_std)
```

Interpretation:

```text
delta_macro_f1_vs_default_lgbm_04 < 0.001:
  no practical tuning gain; retain default_lgbm_04

0.001 <= delta_macro_f1_vs_default_lgbm_04 < 0.005:
  small validation-only tuning gain; report cautiously

delta_macro_f1_vs_default_lgbm_04 >= 0.005:
  practical validation-only tuning gain; still not holdout/test evidence
```

If official validation's best finalist is not the `train_inner_winner`, report
the disagreement as:

```text
official_validation_ranking_disagrees_with_train_inner = true
```

Do not switch the selected configuration to the official-validation-best
finalist unless a new protocol is written before any further result use.

The decision record file `notebook05_decision_record.json` MUST contain the
following minimum fields so downstream notebooks (N06, N07, N08) can
deterministically verify the selection lineage without re-reading official
validation rows:

```text
selected_profile_id
selected_profile_role
selected_profile_source ∈ {
  "train_inner_winner",
  "default_lgbm_04_fallback",
  "no_train_inner_hpo_candidate"
}
official_validation_status ∈ {
  "promote_train_inner_winner",
  "retain_default_lgbm_04",
  "no_practical_tuning_gain",
  "validation_rejects_train_inner_winner",
  "no_train_inner_hpo_candidate"
}
selected_by_official_validation = false  // MUST remain false; transparency-only
official_validation_ranking_disagrees_with_train_inner: bool
scope = "validation_only"
holdout_test_authorized = false
selective_threshold_selected = false
notebook05_protocol_sha256
notebook05_design_doc_sha256
```

`selected_profile_source = "official_validation_best"` is **FORBIDDEN** by this
protocol and is also rejected by the N07 §07A lockfile check. If a future
protocol revision ever proposes to allow it, N07 / N08 design documents and
their static gates MUST be updated in lockstep before that revision is read
by any downstream notebook.

---

## 14. Trial Accounting

Notebook 05 must write both train-inner and official-validation budget records.

Train-inner HPO:

```text
100 trials x 3 inner folds = 300 train-inner LightGBM fits
```

Official-validation confirmation:

```text
30 fitted LightGBM validation rows
10 dummy baseline validation rows
```

The official-validation contact count must be appended to the cumulative
validation-trial budget tracker before any future holdout/test authorization
note is considered.

Notebook 05 must report:

```text
train_inner_hpo_trial_count
train_inner_hpo_fit_count_planned
train_inner_hpo_fit_count_completed
official_validation_lightgbm_rows_planned
official_validation_lightgbm_rows_completed
official_validation_dummy_rows_completed
```

---

## 15. Output Artifacts

Required files:

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

Optional files:

```text
notebook05_probability_predictions_manifest.csv
predictions/*.npz
notebook05_drive_backup_manifest.json
```

Prediction artifacts may include validation sample ids, labels, predictions,
probabilities, ticker ids, and timestamps. They must not include raw OHLCV rows,
train feature matrices, validation feature matrices, or holdout/test data.

---

## 16. Failure Behavior

Notebook 05 must fail fast if:

- any required 04D artifact is missing;
- `OPERATOR_ACCEPTS_EXIT_A` is not true;
- the selected exit is not exactly Exit A;
- `holdout_test_authorized` is not explicitly false;
- the official candidate differs from
  `h03_bps1p5 + price_volume_time + window_size=20`;
- any code path reads, transforms, windows, scores, or summarizes holdout/test
  rows;
- an inner fold has insufficient class coverage;
- official validation is used for early stopping;
- search budget changes after any HPO result is visible;
- official validation is used to replace the train-inner winner;
- required dummy baselines are missing on the same validation sample ids.

Do not silently substitute a different model, seed set, label, feature set,
window size, fold design, metric, or search budget.

---

## 17. Allowed Wording

Allowed:

```text
Notebook 05 performs validation-only LightGBM hyperparameter tuning under a
train-inner chronological HPO design. The official validation partition is used
only to confirm the pre-selected train-inner winner and a fixed number of
finalists.
```

```text
The selected tuned configuration is the train-inner HPO winner, not the
official-validation-best finalist.
```

```text
Notebook 05 does not authorize holdout/test evaluation.
```

If tuning gain is small:

```text
The train-inner selected LightGBM profile produced only a small validation-only
gain over the Notebook 04 default. This supports cautious reporting or retaining
the default, not a claim of robust model superiority.
```

Forbidden:

```text
The tuned model is final.
The tuned model is holdout-ready.
The tuned model significantly beats LogReg.
The tuned model proves LightGBM is superior to deep learning.
The official-validation-best finalist is selected.
Selective coverage is now the final trading threshold.
```

---

## 18. Literature Rationale

This protocol uses established model-selection and financial-time-series
guardrails:

- Cawley and Talbot, "On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation": model selection can overfit the
  selection criterion, motivating separated selection and evaluation surfaces.
  <https://jmlr.org/papers/v11/cawley10a.html>
- Bergstra and Bengio, "Random Search for Hyper-Parameter Optimization":
  random search is a valid HPO baseline when the search domain and budget are
  fixed in advance.
  <https://jmlr.org/papers/v13/bergstra12a.html>
- Bergmeir and Benitez, "On the use of cross-validation for time series
  predictor evaluation": blocked/time-aware CV is preferable to shuffled CV for
  time-series predictor evaluation.
  <https://www.sciencedirect.com/science/article/pii/S0020025511006773>
- Bailey, Borwein, Lopez de Prado, and Zhu, "The Probability of Backtest
  Overfitting": repeated financial strategy/model trials increase overfitting
  risk, motivating fixed trial accounting and cautious wording.
  <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659>
- Lopez de Prado, "Advances in Financial Machine Learning": purging and
  embargo-style thinking for financial labels whose outcomes span future bars.

These references support large HPO inside train-inner validation. They do not
authorize repeated official-validation selection or holdout/test reuse.

---

## 19. Acceptance Criteria For The Notebook Builder

The Notebook 05 builder is acceptable only if:

1. The generated notebook is standalone and raw-data-first.
2. All heavy `RUN_*` switches default to false.
3. 05A imports and verifies the 04D decision artifacts before any fit.
4. 05A requires an explicit operator Exit A acceptance flag.
5. The official candidate is fixed to
   `h03_bps1p5 + price_volume_time + window_size=20`.
6. Train-inner HPO uses exactly 100 trials and 3 chronological folds.
7. Official validation confirms only default + at most five finalists.
8. The selected tuned profile is the train-inner winner.
9. The official-validation-best finalist is reported but not selected.
10. Dummy baselines are computed on the same validation sample ids.
11. Every output row states its scope.
12. No holdout/test data is read, transformed, windowed, scored, summarized, or
    used for wording.
13. The notebook writes all required artifacts.
14. The notebook explicitly states `holdout_test_authorized = false`.

---

## 20. Next Notebook Boundary

Notebook 05 should be created only after this protocol is accepted.

The expected notebook name is:

```text
notebooks/05_lightgbm_tuning_colab.ipynb
```

Notebook 05 is not a final comparison notebook and is not a selective trading
strategy notebook.

The planned follow-up for Ian's second-layer no-trade/abstention question is:

```text
notebooks/06_selective_no_trade_calibration_colab.ipynb
```

That notebook is reserved for prediction-time abstention / high-confidence
coverage calibration after Notebook 05 has been reviewed. It must be separately
pre-registered, must keep holdout/test closed, and must not convert Notebook
04C's diagnostic coverage table into a post-hoc final threshold.

A later comparison notebook may compare the selected LightGBM profile against
default LightGBM, LogReg, standalone TCN, and MS-DLinear+TCN only if that
comparison is separately pre-registered and keeps holdout/test closed.
