# Notebook 06/07 Literature And Materials Draft - 2026-06-05

Scope: research note only. No notebook execution, no training, no holdout/test
access, no Notebook 05 edits.

Project context used:

- Notebook 04 completed validation-only controlled follow-up, and 04D operator
  routing selected Exit A into Notebook 05 LightGBM tuning.
- Notebook 05 is a narrow LightGBM lane: train-inner chronological HPO plus
  small official-validation confirmation. It does not select a selective
  threshold and does not authorize holdout/test.
- Notebook 06 should be a separately pre-registered prediction-time abstention /
  high-confidence coverage / second-layer no-trade analysis after Notebook 05
  review.
- Notebook 07 should be a final validation-only comparison / ablation /
  robustness / synthesis notebook, with all model and threshold choices frozen
  before reading its comparison tables.

## Short Verdict

Notebook 06 is a standard and defensible idea if framed as selective
classification / reject-option prediction: the model may abstain when its
confidence is low, so the reported trade-off is risk versus coverage, not just
"better accuracy." This is common in the ML literature and has a direct finance
analogue: abstention maps to no position.

Notebook 06 becomes tuning / data snooping if it uses official-validation curves
to choose the final confidence threshold after seeing the results. The safe
route is:

1. Freeze the 05-selected LightGBM profile after 05 review.
2. Choose calibration and selective thresholds from train-inner / calibration
   rows only, or pre-register a small coverage grid before looking at Notebook
   06 official-validation outputs.
3. Evaluate official validation once as a readout: full coverage plus all
   pre-listed coverage points.
4. Report every selective result with coverage, retained sample count,
   per-ticker retained count, dummy baseline on the same retained rows, and
   scope = validation_only.

Notebook 07 should not be a new search notebook. It should consolidate locked
models and locked variants from 05/06, quantify robustness, and write cautious
validation-only claims. Because the final holdout/test has already been opened
once and is now closed, 07 should avoid evidence-ready wording unless a separate
pre-registered holdout policy is approved outside this note.

## Recommended Notebook 06 Design

Research question:

> Given the Notebook 05 selected LightGBM candidate, can a pre-registered
> prediction-time abstention rule improve directional reliability at a stated
> coverage cost, without using holdout/test and without choosing the threshold
> from official-validation results?

Suggested stages:

- 06A - Input and pre-registration check: read only Notebook 05 decision records
  and probability prediction manifests after 05 review. Fail if 05 did not
  authorize a final LightGBM profile or if probability outputs are missing.
- 06B - Train-inner calibration design: use train-inner out-of-fold predictions,
  or a chronological train-fit / train-calibration split, to fit probability
  calibration and freeze confidence thresholds.
- 06C - Official-validation readout: evaluate full coverage and a fixed grid,
  for example coverage = 1.00, 0.80, 0.60, 0.40. Do not choose the best coverage
  as the final strategy.
- 06D - Reliability and no-trade interpretation: report risk-coverage, AURC,
  ECE, Brier score, reliability diagrams, retained-count concentration, and
  per-ticker behavior.
- 06E - Decision record: state whether selective abstention is promising,
  inconclusive, or harmful under validation-only evidence. It must not authorize
  holdout/test or declare a final trading rule.

Metrics to include:

- coverage, retained_n, abstained_n
- selective_macro_f1, selective_balanced_accuracy, selective_accuracy
- dummy_macro_f1 on the same retained rows
- delta_macro_f1_vs_dummy on the same retained rows
- selective_error = 1 - selective_accuracy, for risk-coverage vocabulary
- AURC over the pre-registered curve
- Brier score and ECE for probability quality
- per-ticker retained_n, per-ticker coverage, per-ticker delta vs dummy
- top_ticker_retained_share and top_ticker_gain_share

Do not:

- use 04C diagnostic coverage thresholds as final thresholds;
- select a coverage point because official validation looked good there;
- describe abstained rows as "bad trades" unless a trading-cost model is
  separately defined and tested;
- convert validation-only selective metrics into profit, Sharpe, or live-trading
  claims.

## Recommended Notebook 07 Design

Research question:

> Across locked validation-only candidates from 05 and 06, what result is robust
> enough to summarize for a thesis without reopening holdout/test or adding new
> search degrees of freedom?

Suggested stages:

- 07A - Lockfile check: read 05 and 06 decision records only. Fail if any model,
  threshold, seed set, label config, feature set, or window size is not frozen.
- 07B - Final comparison table: default LightGBM, tuned LightGBM, LogReg, and
  any pre-registered TCN / MS-DLinear+TCN comparison rows. Include selective
  variants only if 06 froze them before 07.
- 07C - Ablation and robustness: feature-set, ticker, seed, and metric
  robustness using already-approved candidates. No new feature search.
- 07D - Uncertainty: seed-level mean/std/LCB, per-ticker table, and descriptive
  bootstrap confidence intervals where dependence caveats are explicit.
- 07E - Multiple-comparison and validation-reuse audit: count the number of
  models, profiles, thresholds, seeds, and diagnostics tried. State which
  decisions used validation and why the wording remains validation_only.
- 07F - Paper-ready synthesis: write what the project supports and what it does
  not support.

Do not:

- add fresh thresholds, HPO ranges, or model families inside 07;
- rank many models as if each row were an independent confirmatory test;
- use statistical tests that assume iid row-level samples without caveats;
- let one strong ticker override pooled and cross-ticker evidence.

## Source Cards

### 1. Chow, "On Optimum Recognition Error and Reject Tradeoff" (1970)

- Link: https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff
- DOI: https://doi.org/10.1109/TIT.1970.1054406
- Topic: reject option; error-reject trade-off.
- Why useful: This is the classic reject-option anchor. It makes abstention a
  normal decision-theoretic object rather than an ad hoc trading trick.
- Use in 06/07: Cite for the idea that a classifier may trade error against a
  reject/no-decision rate.
- Do not use: Do not claim Chow justifies this project's exact confidence rule,
  finance setup, or LightGBM probabilities.
- Priority: Must.

### 2. El-Yaniv and Wiener, "On the Foundations of Noise-free Selective Classification" (JMLR, 2010)

- Link: https://jmlr.org/papers/v11/el-yaniv10a.html
- PDF: https://jmlr.csail.mit.edu/papers/volume11/el-yaniv10a/el-yaniv10a.pdf
- Topic: selective classification; risk-coverage trade-off.
- Why useful: It names the risk-coverage framing that Notebook 06 should use.
- Use in 06/07: Define coverage, selective risk, and risk-coverage curves.
- Do not use: Do not import noise-free assumptions into noisy stock data claims.
- Priority: Must.

### 3. Cortes, DeSalvo, and Mohri, "Learning with Rejection" (ALT, 2016)

- Link: https://research.google/pubs/learning-with-rejection/
- PDF: https://cs.nyu.edu/~mohri/pub/rej.pdf
- Topic: learning a classifier plus rejection function.
- Why useful: It separates confidence-threshold rejection from learning a
  rejection function, which helps justify keeping 06 narrow.
- Use in 06/07: Mention as theory background; current project should start with
  post-hoc confidence abstention because it is simpler and lower scope.
- Do not use: Do not implement a new rejection learner in 06 unless separately
  pre-registered; that would widen the search space.
- Priority: Useful.

### 4. Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks" (NeurIPS, 2017)

- Link: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- arXiv: https://arxiv.org/abs/1705.08500
- Code: https://github.com/geifmany/selective_deep_learning
- Topic: selective classifier from a trained model; desired risk / coverage.
- Why useful: This is the most practical modern risk-coverage anchor and includes
  a confidence-based selection implementation.
- Use in 06/07: Use the risk-coverage curve and confidence sorting idea. For
  LightGBM, confidence can be `max(p_up, 1 - p_up)` after calibration.
- Do not use: Do not copy ImageNet risk guarantees to time-series finance. Do
  not claim high-probability risk control unless the exact method and assumptions
  are implemented.
- Priority: Must.

### 5. Geifman and El-Yaniv, "SelectiveNet" (2019)

- Link: https://arxiv.org/abs/1901.09192
- Topic: neural architecture with integrated reject option.
- Why useful: It shows that abstention can be trained end-to-end, but that is a
  different research path.
- Use in 06/07: Background only. It can motivate future deep-model work after a
  simple confidence-based 06.
- Do not use: Do not add SelectiveNet-style losses to 06; that would be new
  model design and not a LightGBM follow-up.
- Priority: Optional / Risky for current scope.

### 6. Chalkidis and Savani, "Trading via Selective Classification" (ICAIF/arXiv, 2021)

- Link: https://arxiv.org/abs/2110.14914
- RePEc: https://ideas.repec.org/p/arx/papers/2110.14914.html
- Topic: selective classification mapped to trading and no-position decisions.
- Why useful: Directly connects abstention to no-position, and also discusses
  binary versus ternary selective classifiers.
- Use in 06/07: Cite for the financial interpretation: abstention means no
  position. Use as a design anchor for reporting coverage and no-position rate.
- Do not use: Do not import their commodity futures backtest, profitability, or
  slippage results as evidence for this stock project.
- Priority: Must.

### 7. Magill and Constantinides, "Portfolio Selection with Transactions Costs" (1976)

- Link: https://econpapers.repec.org/article/eeejetheo/v_3a13_3ay_3a1976_3ai_3a2_3ap_3a245-263.htm
- PDF mirror from author site: https://dornsife.usc.edu/michael-magill/wp-content/uploads/sites/385/2023/12/portfolio_transaction_costs_JET_76.pdf
- Topic: transaction costs and no-trade regions.
- Why useful: Gives classical finance context that transaction costs can make
  "do nothing" rational.
- Use in 06/07: Conceptual support for no-trade as a legitimate action under
  costs.
- Do not use: This is portfolio control, not classifier threshold selection; it
  does not validate our no-trade band or confidence rule.
- Priority: Useful.

### 8. Niculescu-Mizil and Caruana, "Predicting Good Probabilities with Supervised Learning" (ICML, 2005)

- Link: https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf
- Topic: probability calibration; Platt scaling; isotonic regression.
- Why useful: It explains why strong classifiers can have poor probabilities and
  why calibration needs separate data.
- Use in 06/07: Justify evaluating Brier/ECE/reliability diagrams before using
  confidence thresholds.
- Do not use: Do not use isotonic if calibration sample size is too small; it can
  overfit.
- Priority: Must.

### 9. Platt, "Probabilistic Outputs for Support Vector Machines..." (1999)

- Bibliographic link: https://cir.nii.ac.jp/crid/1370851344281644289
- Topic: sigmoid probability calibration.
- Why useful: Classical source for sigmoid calibration.
- Use in 06/07: Background for sigmoid/Platt calibration when using
  `CalibratedClassifierCV(method="sigmoid")`.
- Do not use: Platt scaling is not automatically correct for LightGBM; validate
  calibration quality.
- Priority: Useful.

### 10. Guo et al., "On Calibration of Modern Neural Networks" (ICML, 2017)

- Link: https://proceedings.mlr.press/v70/guo17a
- arXiv: https://arxiv.org/abs/1706.04599
- Topic: calibration error, reliability diagrams, temperature scaling.
- Why useful: Modern calibration anchor and a clear reminder that accuracy and
  calibration are different.
- Use in 06/07: Use ECE/reliability-diagram vocabulary. For deep comparison
  models in 07, do not assume softmax confidence is calibrated.
- Do not use: Temperature scaling is mainly a neural-logit method; LightGBM
  should start with sklearn sigmoid/isotonic calibration.
- Priority: Must.

### 11. scikit-learn Probability Calibration docs

- User guide: https://scikit-learn.org/stable/modules/calibration.html
- CalibratedClassifierCV: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV
- CalibrationDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html
- Topic: practical probability calibration and reliability diagrams.
- Why useful: Official implementation path that can be used without inventing
  calibration code.
- Use in 06/07: Use `CalibratedClassifierCV` only with chronological/custom
  train-inner splits, not random CV. Use `CalibrationDisplay.from_predictions`
  for official-validation readout.
- Do not use: Do not let default CV silently create shuffled or non-chronological
  folds.
- Priority: Must.

### 12. LightGBM `LGBMClassifier` docs

- Link: https://lightgbm.readthedocs.io/en/v4.4.0/pythonapi/lightgbm.LGBMClassifier.html
- Older docs with explicit warning: https://lightgbm.readthedocs.io/en/v3.3.5/pythonapi/lightgbm.LGBMClassifier.html
- Topic: LightGBM probability outputs and class weights.
- Why useful: The docs warn that class-weighting can lead to poor individual
  probability estimates and suggest probability calibration.
- Use in 06/07: Because Notebook 05 uses class balancing, treat raw
  `predict_proba` confidence as a score first, not a reliable probability until
  calibrated and checked.
- Do not use: Do not call raw LightGBM probabilities "true chance of up move."
- Priority: Must.

### 13. scikit-learn Brier score docs

- Link: https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.brier_score_loss.html
- Topic: proper scoring rule for probability forecasts.
- Why useful: Brier score is a compact probability-quality metric for 06.
- Use in 06/07: Report Brier alongside ECE/reliability plots for the positive
  direction probability.
- Do not use: Do not make Brier the primary directional classifier metric; keep
  macro F1 and balanced accuracy as project primaries.
- Priority: Must.

### 14. MAPIE documentation and paper

- Docs: https://mapie.readthedocs.io/en/v1.1.0/theoretical_description_risk_control.html
- GitHub: https://github.com/scikit-learn-contrib/MAPIE
- Paper: https://arxiv.org/abs/2207.12274
- Topic: conformal prediction and risk control implementation.
- Why useful: MAPIE is a scikit-learn-compatible path for uncertainty sets and
  risk-control experiments.
- Use in 06/07: Consider as an optional 06 appendix if dependencies are already
  available or explicitly approved. Good for learning the method and checking
  whether conformal prediction sets collapse to no-trade behavior.
- Do not use: Do not install MAPIE silently. Do not claim conformal validity
  under time-series dependence without a clear exchangeability caveat or a
  time-aware calibration design.
- Priority: Useful, but dependency-gated.

### 15. Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction..." (2021)

- Link: https://arxiv.org/abs/2107.07511
- Topic: conformal prediction tutorial.
- Why useful: Best practical entry point for distribution-free uncertainty
  quantification.
- Use in 06/07: Use to explain what conformal prediction can and cannot promise.
- Do not use: Do not treat exchangeability assumptions as automatically true for
  intraday time series.
- Priority: Useful.

### 16. Angelopoulos et al., "Conformal Risk Control" (2022)

- Link: https://arxiv.org/abs/2208.02814
- Code: https://github.com/aangelopoulos/conformal-risk
- Topic: controlling monotone risks beyond simple coverage.
- Why useful: It is relevant if 06 wants target-risk language instead of only
  coverage-grid language.
- Use in 06/07: Optional advanced method for a future preregistered risk-control
  notebook.
- Do not use: Do not bolt this onto 06 unless the implementation and assumptions
  are reviewed; it can become a new tuning surface.
- Priority: Optional / Risky for current scope.

### 17. Vovk, Gammerman, and Shafer, "Algorithmic Learning in a Random World"

- Book site: https://alrw.net/
- Springer: https://link.springer.com/book/10.1007/978-3-031-06649-8
- Topic: conformal prediction foundations.
- Why useful: Foundational reference for reliability guarantees.
- Use in 06/07: Cite only for conformal background if MAPIE/conformal methods
  are included.
- Do not use: Too broad for the main 06 path; avoid turning 06 into a conformal
  theory notebook.
- Priority: Optional.

### 18. Bailey et al., "Pseudo-Mathematics and Financial Charlatanism..." / Backtest Overfitting

- Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- Topic: backtest overfitting and trying too many strategy configurations.
- Why useful: Strong finance-specific warning against selecting thresholds,
  variants, or strategies after repeated validation/backtest inspection.
- Use in 06/07: Use as a caution for validation reuse and threshold searching.
- Do not use: This project is not yet doing trading PnL backtests; keep the
  analogy to selection risk, not profitability claims.
- Priority: Must for 07 discussion.

### 19. Sullivan, Timmermann, and White, "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"

- Link: https://econpapers.repec.org/RePEc%3Aehl%3Alserod%3A119144
- PDF: https://researchonline.lse.ac.uk/id/eprint/119144/1/dp303.pdf
- Topic: data snooping in technical trading rule search.
- Why useful: Directly relevant to "do not pick the best validation threshold
  after looking."
- Use in 06/07: Use as finance/econometrics support for counting the tried
  rule/threshold/model universe.
- Do not use: White's Reality Check is more than a casual bootstrap; do not
  claim it was applied unless implemented.
- Priority: Must.

### 20. Hansen, "A Test for Superior Predictive Ability" (2005)

- Link: https://ideas.repec.org/a/bes/jnlbes/v23y2005p365-380.html
- Topic: superior predictive ability under multiple comparisons.
- Why useful: Follow-up to data-snooping tests; useful for 07's multiple-model
  caution.
- Use in 06/07: Cite as optional advanced test if 07 has many locked model
  variants.
- Do not use: Do not run SPA as a checkbox without designing the loss series and
  dependence handling.
- Priority: Useful / Optional.

### 21. Diebold and Mariano, "Comparing Predictive Accuracy" (1995)

- NBER: https://www.nber.org/papers/t0169
- Journal DOI: https://doi.org/10.1080/07350015.1995.10524599
- Topic: pairwise forecast accuracy comparison with dependent forecast errors.
- Why useful: Relevant if 07 compares two locked predictors on paired validation
  loss sequences.
- Use in 06/07: Consider for secondary paired comparison of locked models using
  a pre-specified loss, not as the primary project claim.
- Do not use: Do not apply DM blindly to macro F1 rows; it is a forecast-loss
  differential test and needs careful setup.
- Priority: Useful.

### 22. Cawley and Talbot, "On Over-fitting in Model Selection..." (JMLR, 2010)

- Link: https://jmlr.org/papers/v11/cawley10a.html
- PDF: https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf
- Topic: selection bias from model selection.
- Why useful: Central source for why 07 cannot treat a selected validation result
  as an unbiased final performance estimate.
- Use in 06/07: Use in 07's validation-reuse audit and allowed wording.
- Do not use: Do not imply nested CV fully fixes the already-consumed official
  validation budget.
- Priority: Must.

### 23. Varma and Simon, "Bias in error estimation when using cross-validation for model selection" (2006)

- Link: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-7-91
- DOI: https://doi.org/10.1186/1471-2105-7-91
- Topic: biased performance estimates after model selection.
- Why useful: Another clear source for separating selection from evaluation.
- Use in 06/07: Support 05's train-inner HPO design and 07's caution about
  selected profiles.
- Do not use: Biomedical context differs; cite the evaluation principle, not
  domain conclusions.
- Priority: Must.

### 24. Bergmeir, Hyndman, and Koo, "A note on the validity of cross-validation for autoregressive time series prediction" (2018)

- Link: https://cbergmeir.com/publications/2018-01-01_bergmeir2018note/
- Topic: time-series cross-validation caveats.
- Why useful: Provides a nuanced source: not all CV is forbidden, but dependence
  structure matters.
- Use in 06/07: Use to justify time-aware validation rather than random splits.
- Do not use: Do not use it to permit random splits for this project; AGENTS
  forbids random train/validation splits.
- Priority: Useful.

### 25. Tashman, "Out-of-sample tests of forecasting accuracy" (2000)

- ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/S0169207000000650
- RePEc: https://ideas.repec.org/a/eee/intfor/v16y2000i4p437-450.html
- Topic: fixed origin, rolling origin, multiple test periods.
- Why useful: Classic forecasting-evaluation source for 07 robustness framing.
- Use in 06/07: Cite for rolling-origin / multiple-period robustness as a better
  diagnostic than one lucky period.
- Do not use: Do not create new rolling-origin selection in 07 unless locked
  before reading results.
- Priority: Useful.

### 26. Hyndman and Athanasopoulos, "Forecasting: Principles and Practice" / `tsCV`

- Book: https://otexts.com/fpp3/
- `tsCV`: https://pkg.robjhyndman.com/forecast/reference/tsCV.html
- Blog: https://robjhyndman.com/hyndsight/tscv/
- Topic: time-series cross-validation and rolling forecast origin.
- Why useful: Practical tutorial source for chronological evaluation designs.
- Use in 06/07: Use for explanation and diagrams, not as the primary finance
  claim.
- Do not use: The examples are forecasting-centric; adapt carefully to
  classification and per-ticker windows.
- Priority: Useful.

### 27. scikit-learn `TimeSeriesSplit`

- Link: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Topic: official time-ordered split implementation.
- Why useful: Useful for train-inner calibration/HPO if wrapped with project
  split-boundary and ticker/day constraints.
- Use in 06/07: Use only as a starting point for train-inner indices; still
  enforce no cross-ticker, no cross-day, and no label-horizon crossing.
- Do not use: Do not use default splits directly on pooled windows if that mixes
  tickers or violates project sample ownership.
- Priority: Useful.

### 28. SciPy `bootstrap`

- Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Topic: bootstrap confidence intervals.
- Why useful: Practical implementation for descriptive CIs.
- Use in 06/07: Use for row-level or ticker-level descriptive intervals only
  with explicit autocorrelation/dependence caveats. Prefer seed-level summaries
  where already frozen.
- Do not use: Do not present iid row-bootstrap CIs as confirmatory proof for
  overlapping intraday windows.
- Priority: Useful.

### 29. Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets" (JMLR, 2006)

- Link: https://www.jmlr.org/papers/v7/demsar06a.html
- Topic: comparing classifiers across multiple datasets.
- Why useful: Gives conservative language for comparing multiple algorithms.
- Use in 06/07: Helpful if per-ticker results are treated as multiple related
  datasets, but only as a secondary robustness view.
- Do not use: Five tickers are not enough to make broad algorithmic superiority
  claims.
- Priority: Optional.

### 30. Nadeau and Bengio, "Inference for the Generalization Error" (2003)

- Link: https://mlanthology.org/mlj/2003/nadeau2003mlj-inference/
- DOI: https://doi.org/10.1023/A:1024068626366
- Topic: uncertainty of generalization error estimates under resampling.
- Why useful: Warns against underestimated variance in resampled model
  comparison.
- Use in 06/07: Use as caution when interpreting multi-seed or resampled
  validation rows.
- Do not use: The exact corrected tests may not match this project's
  chronological validation design.
- Priority: Optional.

### 31. Bartlett and Wegkamp, "Classification with a Reject Option using a Hinge Loss" (JMLR, 2008)

- Link: https://www.jmlr.org/papers/v9/bartlett08a.html
- Topic: binary classification with an explicit rejection cost.
- Why useful: This is a clean theory source for treating rejection as a
  cost-bearing decision, not a free metric improvement.
- Use in 06/07: Cite when explaining why abstention/no-trade has to be
  pre-registered with either a target coverage or an explicit cost proxy.
- Do not use: Do not claim hinge-loss reject-option theory proves the project's
  no-trade band or confidence threshold is optimal.
- Priority: Useful.

### 32. Zadrozny and Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates" (KDD, 2002)

- DOI: https://doi.org/10.1145/775047.775151
- Topic: probability calibration from classifier scores.
- Why useful: Classic calibration reference behind isotonic and score-to-probability
  methods.
- Use in 06/07: Cite as calibration background if Notebook 06 compares raw and
  calibrated LightGBM probabilities.
- Do not use: Do not calibrate on official-validation rows and then report the
  same rows as unbiased confirmation.
- Priority: Useful.

### 33. Fisch, Jaakkola, and Barzilay, "Calibrated Selective Classification" (2022)

- arXiv: https://arxiv.org/abs/2208.12084
- Code: https://github.com/ajfisch/calibrated-selective-classification
- Topic: calibration quality for selective classification.
- Why useful: Direct bridge between abstention and calibration on accepted
  predictions.
- Use in 06/07: Optional vocabulary if 06 reports calibration separately for
  covered samples and full-coverage samples.
- Do not use: Do not import selector networks or DRO training into the current
  LightGBM validation-only lane.
- Priority: Optional.

### 34. White, "A Reality Check for Data Snooping" (Econometrica, 2000)

- Link: https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
- DOI: https://doi.org/10.1111/1468-0262.00152
- Topic: data snooping from reusing the same time-series data for model or rule
  search.
- Why useful: This is the strongest direct econometrics anchor for the project's
  validation-budget caution.
- Use in 06/07: Keep a model/rule/threshold search ledger. Report that many
  official-validation looks raise selection risk.
- Do not use: Do not say White's Reality Check was performed unless 07 actually
  implements the bootstrap procedure and benchmark null.
- Priority: Must.

### 35. Bailey, Borwein, Lopez de Prado, and Zhu, "The Probability of Backtest Overfitting" (2015/2016)

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Related accessible warning paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- Topic: probability of backtest overfitting from trying many strategies.
- Why useful: Finance-specific language for why strong historical results can
  be artifacts of repeated selection.
- Use in 06/07: Use in limitations to explain why validation-only evidence after
  several notebooks cannot be written as final live-trading evidence.
- Do not use: Do not claim PBO or CSCV was estimated unless implemented.
- Priority: Must.

### 36. Harvey, Liu, and Zhu, "... and the Cross-Section of Expected Returns" (RFS, 2016)

- PDF: https://academic.oup.com/rfs/article-pdf/29/1/5/24450796/hhv059.pdf
- DOI: https://doi.org/10.1093/rfs/hhv059
- Topic: multiple testing in empirical asset pricing.
- Why useful: Gives a finance-specific multiple-comparison warning for a project
  that has tried labels, features, windows, profiles, and confidence rules.
- Use in 06/07: Cite for stronger evidence thresholds and conservative wording
  after many tried comparisons.
- Do not use: Do not transfer their factor-return thresholds directly to this
  5-minute classification task.
- Priority: Must.

### 37. Benjamini and Hochberg, "Controlling the False Discovery Rate" (1995)

- DOI record: https://colab.ws/articles/10.1111%2Fj.2517-6161.1995.tb02031.x
- Topic: false discovery rate control.
- Why useful: If 07 reports many p-values for model/ticker/ablation comparisons,
  FDR vocabulary is more appropriate than treating each test independently.
- Use in 06/07: Either adjust families of p-values or explicitly state that
  tests are descriptive and unadjusted.
- Do not use: FDR does not fix validation reuse, temporal dependence, or model
  selection bias by itself.
- Priority: Useful.

### 38. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" (Ecography, 2017)

- PDF: https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file
- DOI: https://doi.org/10.1111/ecog.02881
- Topic: structured cross-validation under dependence.
- Why useful: Broad method source for why ordinary random CV is inappropriate
  when observations are structured.
- Use in 06/07: Cite for blocked/structured evaluation principles alongside the
  project's stricter chronological split rules.
- Do not use: Ecology examples are not financial-market evidence.
- Priority: Useful.

### 39. Politis and Romano, "The Stationary Bootstrap" (JASA, 1994)

- Link: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870
- DOI: https://doi.org/10.1080/01621459.1994.10476870
- Topic: bootstrap for weakly dependent stationary observations.
- Why useful: Better source than iid bootstrap for uncertainty under temporal
  dependence.
- Use in 06/07: If reporting bootstrap intervals, prefer block/stationary
  bootstrap or ticker/day-level resampling over iid row resampling.
- Do not use: Do not claim overlapping intraday windows are iid after bootstrap.
- Priority: Must for CI caveats.

### 40. Takahashi et al., "Confidence interval for micro-averaged F1 and macro-averaged F1 scores" (2021)

- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC8936911/
- DOI: https://doi.org/10.1007/s10489-021-02635-5
- Topic: uncertainty intervals for F1 metrics.
- Why useful: Specific to F1, which is one of the project's primary metrics.
- Use in 06/07: Reference if adding confidence intervals for macro F1.
- Do not use: Pair with temporal-dependence caveats; F1 CI methods alone do not
  solve overlapping-window dependence.
- Priority: Useful.

### 41. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (2021)

- arXiv: https://arxiv.org/abs/2103.03098
- Topic: variance sources in ML benchmarking.
- Why useful: Supports reporting seed mean/std/LCB and not relying on one strong
  seed or one model fit.
- Use in 06/07: Use for seed-variance and benchmark-variance discussion.
- Do not use: Five seeds do not cover all variance sources such as data sampling,
  HPO randomness, implementation differences, or market regime changes.
- Priority: Must.

### 42. Reimers and Gurevych, "Reporting Score Distributions Makes a Difference" (2017)

- arXiv: https://arxiv.org/abs/1707.09861
- Topic: random-seed variability in neural NLP experiments.
- Why useful: Simple citation for reporting distributions rather than a single
  run, especially if 07 includes deep model rows.
- Use in 06/07: Use as supporting evidence for multi-seed reporting.
- Do not use: Do not transfer NLP LSTM variance magnitudes to stock models.
- Priority: Useful.

### 43. scikit-learn `DummyClassifier`

- Link: https://scikit-learn.org/dev/modules/generated/sklearn.dummy.DummyClassifier.html
- Topic: simple baseline classifiers.
- Why useful: Official support for the project's stratified dummy baseline
  convention.
- Use in 06/07: Cite for dummy baseline implementation, then keep
  `delta_macro_f1_vs_dummy` as a required column.
- Do not use: Beating dummy is not proof of tradable edge or robust signal.
- Priority: Must.

### 44. Lipton and Steinhardt, "Troubling Trends in Machine Learning Scholarship" (2018)

- arXiv: https://arxiv.org/abs/1807.03341
- Topic: empirical ML claim discipline and ablation interpretation.
- Why useful: Useful writing guardrail for 07's synthesis and ablation sections.
- Use in 06/07: Cite when separating observed ablation deltas from speculative
  causal explanations.
- Do not use: It is not a statistical test and not finance-specific.
- Priority: Optional.

### 45. scikit-learn Threshold Tuning docs / `TunedThresholdClassifierCV`

- User guide: https://scikit-learn.org/stable/modules/classification_threshold.html
- API: https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
- Topic: decision-threshold tuning.
- Why useful: It documents the common supervised threshold-tuning pattern, which
  is exactly the dangerous surface for Notebook 06 if used carelessly.
- Use in 06/07: Use as a cautionary implementation reference. Any threshold
  tuning must be train-inner and chronological, not official-validation tuning.
- Do not use: Do not use default CV behavior out of the box for this time-series
  route.
- Priority: Risky but useful.

### 46. Novy-Marx and Velikov, "A Taxonomy of Anomalies and Their Trading Costs" (2016)

- NBER: https://www.nber.org/papers/w20721
- DOI: https://doi.org/10.3386/w20721
- Topic: trading costs can erase apparent anomaly profitability.
- Why useful: Strong finance warning that classification success does not equal
  economic tradability.
- Use in 06/07: Cite when stating that transaction costs and execution are out
  of scope unless separately registered.
- Do not use: Do not infer costs for this project's 5-minute equities from their
  anomaly taxonomy.
- Priority: Must for trading-claim caveats.

### 47. Frazzini, Israel, and Moskowitz, "Trading Costs"

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719
- DOI: https://doi.org/10.2139/ssrn.3229719
- Topic: empirical trading costs from institutional execution data.
- Why useful: Practical grounding for why no-trade/coverage reductions are not
  automatically profitable.
- Use in 06/07: Mention as motivation for a future execution-cost notebook, not
  a current metric.
- Do not use: Do not assume zero slippage, midpoint fills, or constant bps costs.
- Priority: Useful.

### 48. Almgren and Chriss, "Optimal Execution of Portfolio Transactions" (2001)

- DOI: https://doi.org/10.21314/JOR.2001.041
- Reference page: https://research.amanote.com/publication/I5Ut2XMBKQvf0Bhiqr53/optimal-execution-of-portfolio-transactions
- Topic: market impact and optimal execution.
- Why useful: Canonical source showing execution is a separate modeling problem.
- Use in 06/07: Cite only to justify not making PnL or execution claims in 06.
- Do not use: Do not implement a casual execution model inside the selective
  classification notebook.
- Priority: Optional.

## Addendum - Notebook 07 Defense Sources

These additions fill reviewer-facing weak spots for Notebook 07. They should
be treated as source support for synthesis, caveats, and future-work language,
not as authorization to add a new model, threshold, coverage level, or
holdout/test result.

### 49. Traub et al., "Overcoming Common Flaws in the Evaluation of Selective Classification Systems" (2024)

- arXiv: https://arxiv.org/abs/2407.01032
- DOI: https://doi.org/10.48550/arXiv.2407.01032
- Topic: selective-classification metric flaws, risk-coverage evaluation, and
  AUGRC.
- Use in 06/07: Cite when explaining why 06 AURC/E-AURC and fixed coverage-grid
  gains are diagnostics, not operational safety proof.
- Cannot support: Do not add AUGRC post hoc to the already-run 06 decision or
  claim that 06 satisfied the paper's full metric recommendations.
- Priority: Must for Notebook 07 AURC/coverage caveats.

### 50. Hanczar, "Performance Visualization Spaces for Classification with Rejection Option" (2019)

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0031320319302870
- DOI: https://doi.org/10.1016/j.patcog.2019.106984
- Topic: rejection-option visualization, error-reject and cost-reject trade-off
  framing.
- Use in 06/07: Support no-trade/reject-option visualization language if 07
  includes coverage/error trade-off figures.
- Cannot support: It does not define this project's cost function or validate
  any fixed coverage level as economically optimal.
- Priority: Must for 07 visualization caveats.

### 51. Hooker, Mentch, and Zhou, "Unrestricted Permutation Forces Extrapolation" (2021)

- Springer: https://link.springer.com/article/10.1007/s11222-021-10057-z
- arXiv: https://arxiv.org/abs/1905.03151
- Topic: dependent-feature caveats for permutation-based variable importance.
- Use in 07: Justify grouped, block-aware, or clearly caveated permutation
  diagnostics for correlated OHLCV/time features.
- Cannot support: It does not make permutation importance causal, and it does
  not allow permutation diagnostics to reselect features.
- Priority: Useful.

### 52. Fisher, Rudin, and Dominici, "All Models are Wrong, but Many are Useful" (2019)

- JMLR: https://jmlr.csail.mit.edu/beta/papers/v20/18-760.html
- arXiv: https://arxiv.org/abs/1801.01489
- Topic: model class reliance and feature-importance uncertainty across good
  models.
- Use in 07: Explain why feature importance from a single locked LightGBM model
  is a model-specific diagnostic rather than feature truth.
- Cannot support: Do not implement a new model-class-reliance experiment in 07
  unless the model class and budget are pre-registered.
- Priority: Useful.

### 53. Zhang, Zohren, and Roberts, "DeepLOB" (2019)

- arXiv: https://arxiv.org/abs/1808.03668
- DOI: https://doi.org/10.1109/TSP.2019.2907260
- Topic: high-frequency stock movement prediction with limit order book data.
- Use in 07: Contextualize why richer microstructure data can matter for
  intraday direction prediction.
- Cannot support: It does not justify adding DeepLOB inside 07, and it does not
  validate this project's five-minute OHLCV-only result.
- Priority: Context only.

### 54. Aas, Jullum, and Loland, "Explaining Individual Predictions When Features Are Dependent" (2021)

- arXiv: https://arxiv.org/abs/1903.10464
- DOI: https://doi.org/10.1016/j.artint.2021.103502
- Topic: dependence-aware approximations to Shapley values.
- Use in 07: Caveat any SHAP or LightGBM contribution plots for correlated
  OHLCV-derived features and feature families.
- Cannot support: It does not make SHAP causal or allow validation-time
  feature re-selection.
- Priority: Must for dependent-feature SHAP caveats.

### 55. Janzing, Minorics, and Bloebaum, "Feature Relevance Quantification in Explainable AI: A Causal Problem" (2020)

- PMLR: https://proceedings.mlr.press/v108/janzing20a.html
- arXiv: https://arxiv.org/abs/1910.13413
- Topic: causal and non-causal interpretations of feature relevance.
- Use in 07: Support wording that feature-importance diagnostics describe the
  fitted predictor, not a market mechanism.
- Cannot support: It does not invalidate explanation plots and does not justify
  new feature pruning after official-validation readout.
- Priority: Useful.

### 56. Franc, Prusa, and Voracek, "Optimal Strategies for Reject Option Classifiers" (2023)

- arXiv: https://arxiv.org/abs/2101.12523
- JMLR: https://jmlr.org/papers/v24/21-0048.html
- Topic: cost-based reject-option classifier strategies.
- Use in 06/07: Clarify that fixed coverage-grid diagnostics are not an
  optimized no-trade policy unless a rejection cost and optimization target are
  pre-registered.
- Cannot support: It does not prove the current confidence score or coverage
  grid is optimal for five-minute stock data.
- Priority: Useful.

### 57. Huddleston, Liu, and Stentoft, "Intraday Market Predictability: A Machine Learning Approach" (2023)

- RePEc/Oxford record: https://ideas.repec.org/a/oup/jfinec/v21y2023i2p485-527..html
- DOI: https://doi.org/10.1093/jjfinec/nbab007
- Topic: five-minute equity market-return predictability with machine learning.
- Use in 07: Strong finance context for intraday predictability while keeping
  data, predictor, and economic-claim boundaries explicit.
- Cannot support: It does not validate this project's five-stock OHLCV-only
  classifier, no-trade result, Sharpe ratio, or transaction-cost conclusion.
- Priority: Must for finance-context limitations.

### 58. Kong, Zhu, and Azencott, "Predicting Intraday Jumps in Stock Prices Using Liquidity Measures and Technical Indicators" (2019)

- arXiv: https://arxiv.org/abs/1912.07165
- Topic: intraday stock jump prediction with technical and liquidity-style
  features.
- Use in 07: Context for intraday prediction and for separating jump labels,
  direction labels, OHLCV, and richer liquidity inputs.
- Cannot support: It does not validate this project's five-stock direction
  labels or locked LightGBM artifact.
- Priority: Useful context.

### 59. Briola, Bartolucci, and Aste, "Deep Limit Order Book Forecasting" (2024)

- arXiv: https://arxiv.org/abs/2403.09267
- DOI: https://doi.org/10.48550/arXiv.2403.09267
- Topic: recent deep learning for limit order book forecasting.
- Use in 07: Boundary source showing that high-frequency ML claims can depend
  on data modality and market microstructure inputs.
- Cannot support: It does not justify adding LOB models or using LOB results as
  evidence for five-minute OHLCV-only classification.
- Priority: Risky / context only.

### 60. Ait-Sahalia, Fan, Xue, and Zhu, "How and When Are High-Frequency Stock Returns Predictable?" (2022/2025)

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4095405
- DOI: https://doi.org/10.2139/ssrn.4095405
- Topic: high-frequency return predictability using richer trades/quotes style
  information and timing.
- Use in 07: Context for data-modality limitations and why five-minute OHLCV
  results should be worded conservatively.
- Cannot support: Its ultra-high-frequency trades/quotes setting is not
  comparable to this project's five-minute OHLCV features.
- Priority: Risky but useful context.

## Practical Implementation Notes

### Probability calibration for LightGBM

Preferred low-scope path:

1. Use Notebook 05 selected LightGBM profile.
2. Generate train-inner out-of-fold probabilities using chronological folds from
   the training partition only.
3. Fit a calibrator on train-inner probability scores and labels.
4. Freeze coverage thresholds from train-inner calibrated confidence.
5. Apply to official validation once.

Recommended calibrators:

- sigmoid/Platt first, because it has fewer degrees of freedom;
- isotonic only if train-inner calibration rows are large enough and the result
  is treated as exploratory or pre-registered.
- avoid official-validation calibration: official validation can evaluate
  calibration quality, but should not fit the calibrator or choose thresholds.

### Risk-coverage and AURC

For each row:

```text
confidence = max(p_up_calibrated, 1 - p_up_calibrated)
```

Sort validation rows by confidence descending. For each pre-registered coverage
point, retain the top `ceil(coverage * n)` rows. Compute selective metrics on
retained rows and report retained sample counts. AURC can be computed as the
area under selective error versus coverage. AURC evaluates confidence ranking,
not probability calibration.

Useful diagnostics:

- Brier score: probability quality for the binary target.
- Log loss: proper probabilistic loss, useful but sensitive to extreme
  probabilities.
- ECE: implement locally; scikit-learn does not expose a core ECE metric.
- reliability diagram: use scikit-learn `CalibrationDisplay` or
  `calibration_curve`.

### Dummy baseline under abstention

Every selective row needs a dummy baseline on the same retained target rows:

```text
model_selects_rows -> retained sample ids
dummy predicts on retained sample ids using train-label class distribution
delta_macro_f1_vs_dummy = selective_macro_f1 - dummy_macro_f1
```

This is not a perfect trading benchmark because the model controls which rows
are retained, but it is still necessary to avoid reporting a filtered metric
without a same-row reference.

### Validation-budget protection

Use 06 official validation to evaluate pre-frozen coverage points, not to choose
one. The final 06 decision can say:

- "selective abstention improves reliability at lower coverage under
  validation-only evidence";
- "selective abstention is inconclusive because gains concentrate in one ticker
  or one seed";
- "selective abstention is harmful because delta vs dummy does not survive
  coverage/ticker checks."

It cannot say:

- "coverage 0.62 is the final trading threshold" if that was found from the
  official-validation curve;
- "the high-confidence subset is tradable" without transaction-cost and
  execution assumptions;
- "this proves live profitability."

### Notebook 07 comparison discipline

Notebook 07 should include a validation-reuse ledger:

```text
artifact
decision made
models/profiles/seeds/thresholds tried
official-validation rows inspected
whether the decision was train-inner-only or official-validation-informed
allowed wording
```

Recommended uncertainty hierarchy:

1. Seed-level mean/std/LCB for frozen model profiles.
2. Per-ticker robustness and concentration checks.
3. Ticker/day/block or stationary-bootstrap intervals only as descriptive
   robustness, with dependence caveats.
4. Avoid iid row-level bootstrap as confirmatory evidence because windows
   overlap and intraday samples are autocorrelated.

## Proposed Priority Reading Order

1. El-Yaniv and Wiener 2010.
2. Geifman and El-Yaniv 2017 plus the code repository.
3. Chalkidis and Savani 2021.
4. scikit-learn probability calibration docs.
5. LightGBM `LGBMClassifier` probability warning.
6. Niculescu-Mizil and Caruana 2005.
7. Cawley and Talbot 2010.
8. White 2000 and Bailey et al. on data snooping / backtest overfitting.
9. Harvey, Liu, and Zhu 2016 for finance multiple-testing caution.
10. Tashman 2000 or Hyndman time-series CV materials.
11. Politis and Romano if 07 adds dependent-data bootstrap CIs.
12. MAPIE / conformal materials only if 06 wants an optional appendix and
    dependency availability is checked first.

## Open Questions For The Main Thread

- After Notebook 05 review, will 05 produce train-inner out-of-fold
  probabilities suitable for calibration, or only official-validation
  probabilities?
- Should Notebook 06 choose fixed coverage points, a target selective risk, or
  both? Fixed coverage is simpler and safer for the current project.
- Should conformal/MAPIE be included as an appendix, or deferred to avoid adding
  dependencies and assumptions?
- Should Notebook 07 include any deep model rows, or only rows already frozen
  before 07 starts?
