# Notebook 06/07 Selective Calibration Research Packet

Date: 2026-06-05

Scope: `validation_only` design research only. This packet does not authorize
holdout/test use, notebook execution, training, threshold changes, feature
changes, or result reinterpretation.

Project fit:

- Current task: prepare literature, code, tutorial, and experiment-design
  material for planned Notebook 06 and Notebook 07.
- Notebook 06 expected path:
  `notebooks/06_selective_no_trade_calibration_colab.ipynb`.
- Notebook 06 role: prediction-time abstention / no-trade / high-confidence
  coverage calibration on already selected validation-only model outputs.
- Notebook 07 role: final validation-only comparison, ablation, robustness, and
  paper-ready synthesis, without opening holdout/test.
- Hard boundaries: chronological train/validation only, train-only
  preprocessing/calibration fitting, same-row dummy baseline,
  `delta_macro_f1_vs_dummy`, macro F1 and balanced accuracy as main metrics,
  per-ticker reporting, no final holdout/test contact.

## Search Design

The search was split into three evidence lanes:

1. Selective classification, abstention, conformal prediction, and probability
   calibration.
2. Financial no-trade, selective trading, transaction-cost-aware evaluation,
   and backtest overfitting.
3. Reusable code/tutorials for calibration, thresholding, conformal prediction,
   risk-coverage curves, and classifier comparison statistics.

The goal is not to maximize citation count. The goal is to find material that
can actually shape Notebook 06 and Notebook 07 while preserving the current
validation-only research route.

## Top 10 Must-Read Or Most Practical Sources

| priority | source | category | why it matters | supports | does not support | code/tutorial |
|---:|---|---|---|---|---|---|
| 1 | Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS 2017), https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks | classic selective classification | Defines reject-option / coverage tradeoff and risk-coverage framing. | Notebook 06 risk-coverage table/curve, fixed coverage grid, high-confidence subset evaluation. | Does not prove finance profitability or time-series validity. | Reference code: https://github.com/geifmany/selective_deep_learning |
| 2 | Chalkidis and Savani, "Trading via Selective Classification" (arXiv:2110.14914), https://arxiv.org/abs/2110.14914 | financial selective trading | Directly maps classifier abstention to taking no trading position. | Notebook 06 vocabulary: abstain = no-trade at prediction time; coverage as trade frequency proxy. | Does not validate this project's data, labels, transaction costs, leakage safety, or profitability claims. | No mature general-purpose library found; use as design anchor. |
| 3 | scikit-learn probability calibration user guide, https://scikit-learn.org/stable/modules/calibration.html | official implementation guidance | Explains reliability diagrams, calibration curves, Brier/log loss caveats, and probability calibration workflow. | Notebook 06 probability diagnostics: Brier, reliability diagram, ECE-style table. | Brier alone cannot prove calibration improvement because it mixes calibration, resolution, and uncertainty. | `CalibrationDisplay`, `calibration_curve`, `CalibratedClassifierCV`. |
| 4 | Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017), https://proceedings.mlr.press/v70/guo17a.html | calibration classic | Strong practical anchor for post-hoc calibration and temperature scaling. | Notebook 06 calibration rationale if deep-model probabilities are used. | Does not imply LightGBM or time-series probabilities are well calibrated after tuning. | Concepts portable; use sklearn calibration APIs when possible. |
| 5 | scikit-learn `CalibratedClassifierCV`, https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV | direct tool | Official API for sigmoid, isotonic, and temperature calibration. Warns that isotonic overfits with small calibration samples. | Notebook 06 train-inner or validation-split calibration experiments if pre-registered and disjoint from threshold selection. | Must not fit calibrator and choose thresholds on the same official validation rows without a nested plan. | Direct sklearn API. |
| 6 | MAPIE classification and risk-control docs, https://mapie.readthedocs.io/en/latest/api.html | conformal/code | Provides split/cross conformal classifiers, LAC/APS/RAPS conformity scores, and binary risk-control API. | Optional Notebook 06 appendix or conservative variant: set-valued / risk-controlled abstention. | Classical conformal guarantees rely on exchangeability; current intraday time-series setting weakens those guarantees. | `SplitConformalClassifier`, `BinaryClassificationController`, coverage metrics. |
| 7 | Romano, Sesia, and Candes, "Classification with Valid and Adaptive Coverage" (NeurIPS 2020), https://arxiv.org/abs/2006.02544 | conformal classification | Adaptive prediction sets for classification with marginal coverage framing. | Notebook 06 extension idea if output can be framed as prediction sets rather than one-label trade/no-trade. | Not a direct trading profitability or chronological-validation guarantee. | MAPIE implements related APS methods. |
| 8 | Cawley and Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation" (JMLR 2010), https://jmlr.org/papers/v11/cawley10a.html | model-selection guardrail | Explains why optimizing finite validation criteria can overfit selection. | Notebook 06/07 rule: pre-register threshold grid and final comparison before reading results; do not threshold-scrape. | Does not provide a finance-specific test by itself. | Design reference. |
| 9 | Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" / multiple-testing backtest overfitting material, https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf and https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3177057 | financial overfitting | Financial-model warning that repeated trials inflate false discoveries. | Notebook 07 multiple-comparison accounting, validation-budget table, cautious wording. | Sharpe/DSR should not be the main metric here unless a separately defined trading/PnL notebook exists. | Design reference; not necessary to implement DSR for current classifier-only route. |
| 10 | Ojala and Garriga, "Permutation Tests for Studying Classifier Performance" (JMLR 2010), https://jmlr.org/papers/v11/ojala10a.html | null-control/statistics | Provides classifier null-testing vocabulary via label/feature permutation. | Notebook 07 optional null-control appendix if implemented with chronological/block constraints. | Naive row-wise permutation is unsafe for autocorrelated intraday windows. | sklearn has `permutation_test_score`, but default CV/permutation must be adapted or avoided. |

## Additional Useful Sources

| source | classification | use |
|---|---|---|
| scikit-learn `TunedThresholdClassifierCV`, https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html | directly useful but risky | Useful warning that threshold tuning on the same data used for fitting is overfitting-prone. Do not use default stratified CV for this time series route. |
| Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR 2006), https://www.jmlr.org/papers/v7/demsar06a.html | cautious reference | Useful for reporting multiple classifiers over multiple tickers/seeds, but current project has five related tickers, not independent datasets. Prefer descriptive paired deltas plus bootstrap over strong significance claims. |
| MAPIE Sadinle example, https://mapie.readthedocs.io/en/stable/examples_classification/3-scientific-articles/plot_sadinle2019_example.html | tutorial | Good compact tutorial for split conformal classification and coverage/ambiguity ideas. |
| Angelopoulos et al., "Conformal Risk Control" (arXiv:2208.02814), https://arxiv.org/abs/2208.02814 | advanced | Strong future-facing risk-control framing. Use only as cautious optional design; not needed for a clean Notebook 06 first pass. |
| DeepLOB, https://arxiv.org/abs/1808.03668 | contextual | Useful high-frequency deep-learning context, but LOB data are not this project's 5-minute OHLCV bars. |
| LOBFrame / Deep Limit Order Book Forecasting, https://arxiv.org/abs/2403.09267 | cautious benchmark | Strong warning that high ML metrics need not equal actionable trading signal. Use for Notebook 07 discussion, not as a model template. |
| net:cal calibration framework, https://github.com/EFS-OpenSource/calibration-framework | possible plotting/tooling | Rich calibration diagrams and ECE-style tools, but adding a new dependency may be unnecessary. Prefer sklearn unless a specific plot is needed. |

## Notebook 06 Design Recommendation

Notebook 06 should be a post-prediction analysis notebook, not a new model
selection notebook.

### Entry Conditions

Required inputs should be validation-only prediction artifacts from Notebook 05:

- validation sample ids;
- ticker;
- timestamp;
- true label;
- predicted class;
- predicted probability or score;
- model/profile id;
- seed;
- dummy baseline predictions on the same rows;
- explicit `holdout_test_authorized = false`.

If Notebook 05 did not save probability artifacts, Notebook 06 should stop and
report the missing exact artifact path. It should not refit Notebook 05 models
unless a separate pre-registered 06 builder explicitly permits it.

### Main 06 Analysis

Recommended fixed coverage grid:

```text
coverage_grid = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
minimum_coverage = 0.30
minimum_rows_per_ticker = pre-registered, e.g. 500 or a data-derived guard
score = max(predicted_probability, 1 - predicted_probability)
abstain if score is below the threshold needed for target coverage
```

Use coverage grid first, not free-form probability thresholds. A coverage grid
keeps the number of comparisons finite, makes cross-model comparison easier, and
reduces threshold scraping. For interpretability, Notebook 06 can still report
the implied probability threshold for each coverage level.

Required table columns:

```text
model_or_profile
seed
coverage_target
coverage_actual
probability_threshold
n_selected
n_abstained
pooled_macro_f1_selected
pooled_balanced_accuracy_selected
dummy_macro_f1_selected
delta_macro_f1_vs_dummy_selected
random_abstention_macro_f1_mean
delta_macro_f1_vs_random_abstention
per_ticker_positive_delta_count
top_ticker_selected_share
selected_row_ticker_entropy
ece_selected
brier_selected
scope = validation_only
```

Recommended plots:

- risk-coverage curve: x = coverage, y = selected-set error or
  `1 - macro_f1` proxy;
- selected-set `delta_macro_f1_vs_dummy` by coverage;
- reliability diagram for full validation rows and selected rows;
- coverage concentration by ticker;
- probability-score histogram by correctness and ticker.

### Baselines For Abstention

Notebook 06 should not compare only against full-coverage dummy. It needs
same-row abstention baselines:

1. Same-row stratified dummy on selected rows at each coverage level.
2. Random-abstention baseline: randomly select the same number of validation
   rows as the model-selected subset, stratified by ticker and seed, then score
   the already fixed model predictions or dummy predictions depending on the
   question.
3. Always-up dummy on selected rows as a sanity check.

The strongest selective claim should be:

```text
At pre-registered coverage levels, the high-confidence selected subset retained
positive delta vs same-row dummy and exceeded a random-abstention baseline,
under validation_only scope.
```

Forbidden wording:

```text
The selective threshold is final.
The strategy is profitable.
The model is safer in deployment.
The threshold is holdout-ready.
```

### Calibration Diagnostics

Use Brier score, ECE-style binned calibration, and reliability diagrams as
diagnostics, not selection gates unless a calibration sub-split is introduced
before the run.

Recommended ECE implementation:

- 10 equal-frequency bins for pooled reliability, because predicted
  probabilities may cluster tightly.
- Also show equal-width bins if enough rows exist.
- Report bin counts; empty or tiny bins should be visible, not hidden.

Calibration fitting options:

1. No calibration fitting: first-pass Notebook 06 evaluates raw probabilities.
2. Train-inner calibration only: fit sigmoid/temperature calibrator inside
   training-derived folds, then apply to official validation once.
3. Validation split calibration: split official validation chronologically into
   calibration and analysis halves. This costs validation power and should be
   pre-registered. It must not choose thresholds on the same rows used to
   report selective performance.

Recommended first pass: option 1. If raw probabilities are poorly calibrated,
write a bounded follow-up plan instead of patching calibration into the same
result pass.

### Threshold-Scraping Guardrails

- Fix the coverage grid before loading Notebook 05 probabilities.
- Do not search arbitrary thresholds like 0.51, 0.52, ..., 0.99 and report the
  best.
- Do not choose the best coverage level as the "final" level. Show the whole
  curve and identify pre-registered operating zones only.
- Do not let a tiny selected subset dominate interpretation.
- Require positive delta across at least 3 of 5 tickers and concentration guard
  such as `top_ticker_selected_share <= 0.50`.
- Treat coverage below 0.30 as visualization-only unless separately justified.

## Notebook 07 Design Recommendation

Notebook 07 should be a validation-only synthesis notebook. It should not add a
new model family or tune a new threshold after seeing Notebook 06.

### Candidate 07 Purpose

Recommended Notebook 07 question:

```text
Under the frozen validation-only route, which conclusions remain after comparing
the selected LightGBM/default alternatives, selective coverage behavior,
same-row dummy baselines, per-ticker breadth, and pre-registered robustness
checks?
```

### Main Sections

1. Import frozen artifacts from 02/03/04/05/06.
2. Verify artifact manifests and `holdout_test_authorized = false`.
3. Build a validation-budget ledger:
   Stage 0 rows, Notebook 03 rows, Notebook 05 HPO/confirmation rows, Notebook
   06 coverage grid rows.
4. Define one final validation-only comparison table:
   default LightGBM, tuned LightGBM if promoted by 05, LogReg, selected sequence
   baseline if relevant, dummy baselines, and selective variants from 06.
5. Report pooled and per-ticker metrics on aligned rows.
6. Report ablation/robustness checks that were pre-registered before 07:
   no selective threshold search, no new labels/features, no new model zoo.
7. Write paper-ready methods wording and limitations.

### Robustness Checks Suitable For 07

Recommended checks:

- Per-ticker direction: every final row must show `delta_macro_f1_vs_dummy` by
  ticker.
- Seed stability: summarize mean/std/LCB over matched seeds.
- Coverage stability: for selective variants, show whether coverage and delta
  are concentrated in one ticker.
- Paired bootstrap over validation examples within ticker/day blocks, reported
  as descriptive CI only.
- Blocked day-level bootstrap for macro F1 and delta vs dummy if implementation
  time allows.
- Null-control appendix: label permutation or circular/day-block permutation
  only if chronology-safe; otherwise skip and state why.
- Multiple-comparison ledger: count every validation model/threshold/coverage
  contact before any conclusion.

Avoid:

- Row-wise iid bootstrap over overlapping sliding windows as a strong
  uncertainty claim.
- Random shuffled CV, default `StratifiedKFold`, or row-wise permutation tests.
- Reporting a p-value as thesis proof under a single reused official validation
  period.
- Treating five tickers as five independent datasets for strong Demsar-style
  claims.

### Statistical Tests And Reporting

Use a hierarchy:

1. Primary: effect sizes against same-row dummy:
   `delta_macro_f1_vs_dummy`, `delta_balanced_accuracy_vs_dummy`.
2. Secondary: paired differences by seed and ticker.
3. Diagnostic uncertainty: day-block bootstrap intervals for deltas.
4. Optional: McNemar-style paired error comparison on aligned rows, clearly
   labeled as approximate because windows are autocorrelated.
5. Optional: chronology-safe permutation null, only if blocks preserve temporal
   dependence enough to avoid a fake null.

Recommended wording:

```text
All comparisons are validation_only. Because this route has used one official
validation period for model, tuning, and selective analyses, intervals and
tests are diagnostic rather than independent confirmation.
```

## Code And Tutorial Shortlist

| tool/tutorial | URL | use | current recommendation |
|---|---|---|---|
| sklearn calibration guide | https://scikit-learn.org/stable/modules/calibration.html | Reliability diagrams, Brier/log loss caveats, calibration workflow. | Use first. Stable and low dependency burden. |
| sklearn `CalibratedClassifierCV` | https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV | Sigmoid/isotonic/temperature calibration. | Use only with disjoint chronological calibration data. |
| sklearn `TunedThresholdClassifierCV` | https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html | Threshold-tuning API and overfitting warning. | Do not use default CV. Treat as cautionary/API reference. |
| sklearn `calibration_curve` | https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html | Build reliability data. | Useful for Notebook 06 plots. |
| MAPIE API | https://mapie.readthedocs.io/en/latest/api.html | Conformal classification and risk control. | Optional, after checking install availability; beware exchangeability. |
| MAPIE Sadinle example | https://mapie.readthedocs.io/en/stable/examples_classification/3-scientific-articles/plot_sadinle2019_example.html | Compact conformal classification tutorial. | Useful for KB notes, not necessary for first 06. |
| Geifman selective code | https://github.com/geifmany/selective_deep_learning | Reference implementation of selective DNN risk bound. | Read for concepts only; CIFAR/VGG code is not directly reusable. |
| net:cal | https://github.com/EFS-OpenSource/calibration-framework | Rich calibration metrics/plots. | Do not add dependency unless sklearn plots are insufficient. |

## Not Recommended Or Easy To Misuse

| route | reason |
|---|---|
| Post-hoc best threshold from official validation | Converts 06 into validation scraping and selection bias. Use fixed coverage grid instead. |
| Turning selective coverage into a trading/PnL claim | Current route is classification validation-only; no transaction cost, slippage, liquidity, order execution, or position sizing model is authorized. |
| Conformal guarantees stated as valid for intraday time series | Classical conformal methods rely on exchangeability. Use as heuristic/risk-control framing unless a time-series-specific conformal design is pre-registered. |
| Default sklearn CV for threshold/calibration | Defaults are usually stratified/random-style, not the project's chronological split discipline. |
| Row-wise iid bootstrap over overlapping windows | Overlapping intraday windows are autocorrelated. Prefer ticker/day/block resampling and label it diagnostic. |
| Adding PatchTST/DeepLOB/LOBFrame models in 07 | These would expand the model zoo after validation results and often assume LOB data, not 5-minute OHLCV bars. |
| DSR/Sharpe as main Notebook 07 metric | This is not a PnL backtest notebook. Use DSR/PBO literature as overfitting caution, not as a metric retrofit. |
| Treating selected-row macro F1 as deployment safety | High-confidence subset performance is conditional on validation selection; it is not a deployment guarantee. |

## Suggested 06 Notebook Outline

```text
Research question and scope
Frozen source artifacts and no-holdout assertion
Load Notebook 05 validation predictions
Validate same-row alignment and required columns
Define fixed coverage grid and abstention score
Full-row probability diagnostics
Coverage-grid selective metrics
Same-row dummy and random-abstention baselines
Per-ticker coverage and delta analysis
Risk-coverage and reliability plots
Decision record: what selective analysis supports and does not support
```

Default heavy switches should remain false if the notebook needs any fitting:

```text
RUN_06_LOAD_ARTIFACTS = False
RUN_06_SELECTIVE_ANALYSIS = False
RUN_06_OPTIONAL_CALIBRATION_FOLLOWUP = False
```

## Suggested 07 Notebook Outline

```text
Research question and validation-only scope
Import artifact manifests from 02-06
No-holdout/test and same-row alignment checks
Validation-budget ledger
Final comparison table: pooled + per ticker
Final selective/no-selective comparison
Ablation and robustness summaries
Statistical diagnostics: paired deltas and block bootstrap if available
Paper-ready methods paragraph
Paper-ready limitations paragraph
Decision record and next-step boundary
```

## Paper-Ready Methods Language Seed

Use wording like:

```text
We evaluated all models under a chronological validation-only protocol. Input
windows and label horizons were constructed per ticker and per split, and
preprocessing statistics were fit on training rows only. Model comparisons used
macro F1 and balanced accuracy as primary metrics and included same-row
stratified dummy baselines. Selective/no-trade analysis was performed as a
post-prediction validation-only diagnostic over pre-specified coverage levels;
it did not use the closed holdout/test interval and did not select a deployment
threshold.
```

Limitations wording:

```text
The validation period was reused for a bounded sequence of model, tuning, and
selective diagnostics. Therefore, Notebook 07 conclusions should be framed as
validation-only synthesis rather than independent generalization evidence.
Selective coverage improves conditional metrics only for acted-upon rows and
does not imply trading profitability without a separately pre-registered
transaction-cost and execution model.
```

## Source-Check Notes

Sources checked live through web search/open on 2026-06-05. The project-local
rules, workflow, config screening freeze, Notebook 05 protocol, Notebook 03
protocol, and Stage 0 desktop review summary were inspected before writing this
packet. No notebooks were run, no training was launched, and no holdout/test
artifact was accessed.
