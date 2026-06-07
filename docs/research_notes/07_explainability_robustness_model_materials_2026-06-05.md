# Notebook 07 Explainability, Robustness, And Model-Persuasion Materials - 2026-06-05

Scope: KB-ready research note only. No notebook execution, no training, no
holdout/test access, no Notebook 05 edits, no dependency installation, no commit,
and no push.

Approved write target for this task:

```text
docs/research_notes/07_explainability_robustness_model_materials_2026-06-05.md
```

Project context read before writing:

- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`
- `artifacts/stage0_desktop_02_config_screening_2026-06-04/stage0_review_summary.md`
- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `artifacts/research_packets/notebook06_07_selective_calibration_research_packet_2026-06-05.md`

## Short Plan

1. Treat Notebook 07 as a validation-only synthesis notebook, not a new search
   notebook.
2. Use LightGBM, SHAP/TreeSHAP, permutation importance, null controls, and
   variance literature to improve explanation quality and thesis credibility.
3. Keep all interpretation diagnostic: feature importance, SHAP, permutation
   importance, ablation, and null-control outputs must not reselect feature set,
   model family, window size, no-trade threshold, coverage threshold, or final
   wording after official-validation results are visible.
4. Add a validation-budget ledger and conservative allowed wording so the paper
   can say what is supported without overstating final generalization.

## Current Project Boundary For Notebook 07

Notebook 07 can improve model/paper persuasiveness by showing:

- the selected LightGBM evidence is not a one-seed or one-ticker artifact;
- all final rows beat same-row dummy baselines by `delta_macro_f1_vs_dummy`;
- feature and explanation diagnostics are directionally coherent across gain,
  split, permutation, and SHAP views;
- null controls do not reproduce the observed validation signal under a
  chronology-aware design;
- the final thesis wording accounts for official-validation reuse and
  multiple-comparison risk.

Notebook 07 would break the current validation-only route if it:

- adds a new model family after seeing 05/06/07 validation evidence;
- uses SHAP, permutation importance, ablation, or null-control outcomes to change
  feature columns, model profile, window size, label horizon, threshold, or
  selective-coverage rule;
- uses official validation for early stopping, calibration fitting, threshold
  choice, or finalist replacement;
- treats a p-value, SHAP ranking, or permutation-importance plot as independent
  holdout/test evidence;
- claims live trading, PnL, Sharpe, or deployment reliability without a separate
  pre-registered execution-cost and holdout/test policy.

## Must Sources

### 1. LightGBM official parameter tuning documentation

- URL: https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
- Type: official documentation.
- Why it matters: Provides the official vocabulary for tuning `num_leaves`,
  `min_data_in_leaf`, `max_depth`, `num_iterations`, `learning_rate`, early
  stopping, bagging, and feature subsampling.
- Use in this project: Cite to explain why Notebook 05's train-inner random HPO
  over tree complexity and regularization is technically reasonable.
- Cannot support: It does not authorize increasing Notebook 05's budget after
  seeing results, using official validation for early stopping, or retuning in
  Notebook 07.
- Category: Must.

### 2. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

- NeurIPS page: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boost
- PDF: https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- Type: primary LightGBM paper.
- Why it matters: Establishes LightGBM as an efficient GBDT implementation with
  histogram-based learning, GOSS, and EFB.
- Use in this project: Cite for why LightGBM is a defensible tabular baseline for
  stationarity-safer 5-minute OHLCV-derived features.
- Cannot support: It does not show that LightGBM has financial alpha on this
  dataset, that tuned LightGBM beats all model families, or that selected
  validation performance generalizes to closed holdout/test.
- Category: Must.

### 3. Bergstra and Bengio, "Random Search for Hyper-Parameter Optimization"

- JMLR: https://jmlr.org/papers/v13/bergstra12a.html
- PDF: https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
- Type: primary HPO paper.
- Why it matters: Gives the standard justification for pre-specified random
  search as a strong HPO baseline when only some hyperparameters matter.
- Use in this project: Cite for Notebook 05's fixed random-search manifest and
  budget.
- Cannot support: It does not justify extending search after seeing validation
  results or replacing a train-inner-selected profile with the official
  validation-best profile.
- Category: Must.

### 4. Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions"

- NeurIPS page: https://papers.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- PDF: https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
- arXiv: https://arxiv.org/abs/1705.07874
- Type: primary SHAP paper.
- Why it matters: Defines SHAP as additive feature attribution for individual
  predictions.
- Use in this project: Use in a Notebook 07 explanation appendix to describe
  which features contributed to locked LightGBM predictions.
- Cannot support: SHAP values are explanations of a fitted model, not proof of
  causal market mechanisms, not feature-selection authorization, and not
  independent validation evidence.
- Category: Must.

### 5. Lundberg et al., "Consistent Individualized Feature Attribution for Tree Ensembles"

- arXiv: https://arxiv.org/abs/1802.03888
- Type: primary TreeSHAP paper.
- Why it matters: Gives the tree-ensemble-specific SHAP basis for fast feature
  attribution on LightGBM-like models.
- Use in this project: Use for a LightGBM SHAP appendix if Notebook 05 exports
  the final locked booster and validation prediction artifacts.
- Cannot support: It does not remove dependence/correlation caveats, and it does
  not make SHAP rankings safe for post-hoc feature pruning.
- Category: Must.

### 6. SHAP `TreeExplainer` documentation

- URL: https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html
- Type: official tool documentation.
- Why it matters: Documents support for LightGBM and notes that feature
  perturbation assumptions affect how feature dependence is handled.
- Use in this project: If `shap` is already available or explicitly approved,
  use `TreeExplainer` only on locked models and validation samples as a
  diagnostic appendix.
- Cannot support: Do not install `shap` silently. Do not treat a SHAP background
  choice as innocuous if it uses official-validation rows in a way that creates a
  new tuning surface.
- Category: Must if SHAP is included.

### 7. LightGBM `pred_contrib` / feature contribution documentation

- Latest LGBMClassifier docs: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- Versioned docs with explicit `pred_contrib` return note: https://lightgbm.readthedocs.io/en/v3.3.5/pythonapi/lightgbm.LGBMClassifier.html
- Type: official tool documentation.
- Why it matters: LightGBM can return per-sample feature contributions via
  `pred_contrib=True`, with an extra expected-value column.
- Use in this project: Prefer this as a dependency-light SHAP-style diagnostic if
  the already-used LightGBM version supports it.
- Cannot support: It does not provide SHAP interaction values and does not
  license feature re-selection in Notebook 07.
- Category: Must if a no-new-dependency explanation path is needed.

### 8. scikit-learn permutation importance documentation

- User guide: https://scikit-learn.org/stable/modules/permutation_importance.html
- API: https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
- Correlated-feature example: https://sklearn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
- Type: official documentation.
- Why it matters: Permutation importance measures how much shuffling a feature
  degrades a fitted model's score on a dataset, and the docs explicitly warn
  about model-specific interpretation and correlated features.
- Use in this project: Use as a validation-only diagnostic over locked
  validation predictions and fixed scoring metrics, preferably grouped by
  feature families when features are correlated.
- Cannot support: Naive row-wise permutation can break temporal and feature
  dependence. It cannot prove intrinsic feature value or justify dropping
  features after official validation.
- Category: Must.

### 9. Ojala and Garriga, "Permutation Tests for Studying Classifier Performance"

- JMLR: https://jmlr.org/papers/v11/ojala10a.html
- PDF: https://jmlr.org/papers/volume11/ojala10a/ojala10a.pdf
- Type: primary null-control paper.
- Why it matters: Provides classifier-performance permutation-test vocabulary:
  label permutation for class-structure nulls and restricted feature
  permutations for feature-dependence questions.
- Use in this project: Use for a Notebook 07 null-control appendix, but only with
  a chronology-aware design such as day-block, ticker-block, or circular
  within-block permutations that preserve the time-series contract as much as
  possible.
- Cannot support: Do not run default iid row-wise label shuffling and report it
  as a valid time-series p-value. Do not say a null test was performed unless the
  permutation design, score, number of permutations, and dependence caveats are
  actually implemented.
- Category: Must.

### 10. Cawley and Talbot, "On Over-fitting in Model Selection..."

- JMLR: https://jmlr.org/papers/v11/cawley10a.html
- PDF: https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf
- Type: primary model-selection bias paper.
- Why it matters: Explains why optimizing finite validation criteria can overfit
  model selection and bias subsequent performance evaluation.
- Use in this project: Anchor Notebook 07's validation-reuse ledger and cautious
  allowed wording.
- Cannot support: It does not retroactively make official-validation-selected
  records unbiased.
- Category: Must.

### 11. White, "A Reality Check for Data Snooping"

- Econometric Society: https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
- DOI: https://doi.org/10.1111/1468-0262.00152
- Type: primary econometrics paper.
- Why it matters: Strong direct source for data-snooping risk when the same
  time-series data are reused across model/rule searches.
- Use in this project: Justify the validation-budget ledger and the statement
  that repeated official-validation looks reduce claim strength.
- Cannot support: Do not claim White's Reality Check was run unless Notebook 07
  implements the actual bootstrap procedure and benchmark null.
- Category: Must.

### 12. Bailey et al., "The Probability of Backtest Overfitting"

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Type: finance overfitting paper.
- Why it matters: Finance-specific warning that repeated strategy/configuration
  trials inflate false discoveries.
- Use in this project: Use in limitations and thesis discussion to explain why
  validation-only classifier evidence is not live-trading proof.
- Cannot support: Do not report PBO, CSCV, Sharpe, or DSR unless a separately
  pre-registered trading/backtest notebook is built.
- Category: Must.

### 13. Harvey, Liu, and Zhu, "... and the Cross-Section of Expected Returns"

- DOI: https://doi.org/10.1093/rfs/hhv059
- PDF: https://academic.oup.com/rfs/article-pdf/29/1/5/24450794/hhv059.pdf
- Type: primary finance multiple-testing paper.
- Why it matters: Finance-specific multiple-testing caution for empirical
  searches across many candidate effects.
- Use in this project: Cite when explaining why labels, windows, features,
  models, seeds, HPO profiles, and selective thresholds must be accounted for in
  the validation ledger.
- Cannot support: Do not transfer their exact factor-return thresholds to this
  binary 5-minute classification task.
- Category: Must.

### 14. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks"

- arXiv: https://arxiv.org/abs/2103.03098
- Type: ML benchmark variance paper.
- Why it matters: Documents that benchmark conclusions are affected by data
  sampling, initialization, hyperparameter choices, and other variance sources.
- Use in this project: Support seed mean/std/LCB reporting and the statement that
  five seeds are useful but incomplete.
- Cannot support: It does not turn a five-seed official-validation summary into
  full uncertainty quantification across market regimes.
- Category: Must.

### 15. Reimers and Gurevych, "Reporting Score Distributions Makes a Difference"

- arXiv: https://arxiv.org/abs/1707.09861
- DOI: https://doi.org/10.48550/arXiv.1707.09861
- Type: seed-variance paper.
- Why it matters: Clear source for reporting score distributions instead of one
  lucky run.
- Use in this project: Support matched-seed tables and plots for locked
  LightGBM, LogReg, and any frozen sequence-model rows.
- Cannot support: Do not transfer NLP LSTM variance magnitudes to stock
  direction models.
- Category: Must for seed reporting.

## Useful Sources

### 16. LightGBM feature-importance documentation

- Plot docs: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.plot_importance.html
- Booster docs: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
- Type: official documentation.
- Use in this project: Report both `split` and `gain` importance for locked
  LightGBM as a low-cost model-inspection appendix.
- Cannot support: Split/gain importance is model-internal. It should not be used
  as a causal feature ranking or a feature-pruning instruction after validation.
- Category: Useful.

### 17. scikit-learn `permutation_test_score`

- API: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
- Example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_permutation_tests_for_classification.html
- Type: official documentation.
- Use in this project: Useful as implementation vocabulary for null tests, but
  only after replacing default random CV/permutation assumptions with the
  project's chronological/blocking constraints.
- Cannot support: Default `cv=None` / random permutations are not safe for this
  intraday route.
- Category: Useful but must be adapted.

### 18. Kumar et al., "Problems with Shapley-value-based explanations as feature importance measures"

- PMLR: https://proceedings.mlr.press/v119/kumar20e.html
- PDF: https://proceedings.mlr.press/v119/kumar20e/kumar20e.pdf
- Type: primary caution paper.
- Use in this project: Cite in the SHAP appendix caveats to avoid saying SHAP
  "explains the market" or provides causal feature truth.
- Cannot support: It does not mean SHAP is useless; it means interpretation must
  specify the value function/background and stay model-specific.
- Category: Useful.

### 19. Politis and Romano, "The Stationary Bootstrap"

- DOI: https://doi.org/10.1080/01621459.1994.10476870
- Publisher page: https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870
- Type: primary bootstrap paper.
- Use in this project: Background for dependent-data bootstrap caveats if 07 adds
  descriptive intervals.
- Cannot support: Do not claim overlapping 5-minute windows are iid after
  bootstrap. Do not use bootstrap intervals as independent holdout evidence.
- Category: Useful.

### 20. Takahashi et al., "Confidence interval for micro-averaged F1 and macro-averaged F1 scores"

- DOI: https://doi.org/10.1007/s10489-021-02635-5
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC8936911/
- Type: F1 uncertainty paper.
- Use in this project: Cite if 07 reports macro-F1 intervals.
- Cannot support: F1-specific CI methods do not by themselves solve temporal
  dependence or validation reuse.
- Category: Useful.

### 21. Diebold and Mariano, "Comparing Predictive Accuracy"

- DOI: https://doi.org/10.1080/07350015.1995.10524599
- NBER: https://www.nber.org/papers/t0169
- Type: predictive-accuracy comparison paper.
- Use in this project: Optional paired-loss comparison of locked predictors if
  Notebook 07 defines a per-sample loss sequence before looking.
- Cannot support: Do not apply it blindly to aggregate macro-F1 rows.
- Category: Useful / optional.

### 22. scikit-learn `DummyClassifier`

- API: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- Type: official documentation.
- Use in this project: Cite for the stratified dummy implementation and preserve
  `delta_macro_f1_vs_dummy` in every final row.
- Cannot support: Beating dummy is necessary for credibility but not sufficient
  for tradability, causality, or evidence-ready claims.
- Category: Useful and mandatory in output schema.

## Optional Sources

### 23. SHAP LightGBM example notebook

- URL: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html
- Type: official tutorial.
- Use in this project: Useful for plotting patterns only if `shap` is already
  available or explicitly approved.
- Cannot support: Tutorial data and plots are not project evidence.
- Category: Optional.

### 24. Conformal / MAPIE materials from the 06/07 packet

- MAPIE docs: https://mapie.readthedocs.io/en/latest/api.html
- Angelopoulos and Bates tutorial: https://arxiv.org/abs/2107.07511
- Conformal risk control: https://arxiv.org/abs/2208.02814
- Use in this project: Defer unless Notebook 06 explicitly includes a
  pre-registered selective-risk or set-valued prediction appendix.
- Cannot support: Do not install MAPIE silently or claim exchangeability-based
  guarantees for intraday time series without a time-aware design.
- Category: Optional / dependency-gated.

## Risky Or Easy To Misuse

### A. SHAP/permutation/importance as feature selection

- Risk: It would convert Notebook 07 into post-hoc feature search.
- Allowed: Use as locked-model explanation and appendix diagnostics.
- Forbidden: Dropping or adding features because a validation SHAP,
  split/gain, ablation, or permutation chart looked good.

### B. Null controls with iid row shuffling

- Risk: Row shuffling violates temporal dependence, overlapping windows, and
  trading-day structure.
- Allowed: Pre-specified ticker/day/block/circular nulls with caveats.
- Forbidden: Default iid permutation p-values reported as thesis proof.

### C. Official-validation best-profile replacement

- Risk: Official validation becomes a second HPO loop.
- Allowed: Report official-validation-best row as calibration context if
  Notebook 05 already permits it.
- Forbidden: Replacing the train-inner winner or default profile based on the
  official validation ranking unless a new protocol is written before any
  further result use.

### D. Adding new model families in 07

- Risk: Model-zoo expansion after seeing validation results.
- Allowed: Include only locked rows from 05/06 or already frozen prior
  validation-only artifacts.
- Forbidden: Adding XGBoost, CatBoost, RandomForest, PatchTST, DeepLOB, new TCN,
  NLP/news, external-market features, or broad architecture sweeps inside 07.

### E. Trading/PnL/Sharpe retrofit

- Risk: Converts classification validation into unsupported trading evidence.
- Allowed: Mention transaction costs and execution as future work/limitations.
- Forbidden: Sharpe, PnL, live trading, execution-cost, or deployment claims
  without a separate pre-registered trading notebook.

## Defense Literature Addendum For Notebook 07

These sources fill gaps that are most likely to matter in reviewer discussion.
They are added for Notebook 07 synthesis and wording defense. They must not
open a new model, feature, threshold, or holdout/test decision path.

### Must Add - Traub et al., "Overcoming Common Flaws in the Evaluation of Selective Classification Systems"

- arXiv: https://arxiv.org/abs/2407.01032
- DOI: https://doi.org/10.48550/arXiv.2407.01032
- Why it matters: Directly critiques common selective-classification evaluation
  patterns and proposes AUGRC as a generalized risk-coverage metric.
- Use in this project: Cite in Notebook 07 when qualifying 06 AURC/E-AURC and
  fixed coverage-grid results. It supports wording that AURC is a diagnostic
  summary, not proof of operational safety.
- Cannot support: Do not retrofit AUGRC into the already-run 06 decision unless
  a later notebook freezes the formula and role before reading results.
- Category: Must for AURC/coverage interpretation caveats.

### Must Add - Sullivan, Timmermann, and White, "Data-Snooping, Technical Trading Rule Performance, and the Bootstrap"

- FMG page: https://www.fmg.ac.uk/publications/discussion-papers/data-snooping-technical-trading-rule-performance-and-bootstrap
- PDF: https://eprints.lse.ac.uk/119144/1/dp303.pdf
- Why it matters: Finance-specific data-snooping warning for technical trading
  rule searches and bootstrap adjustment across a large rule universe.
- Use in this project: Cite in the Notebook 07 validation-budget ledger and
  limitations section when counting model, configuration, seed, and coverage
  degrees of freedom.
- Cannot support: Do not claim White's Reality Check or the paper's bootstrap
  test was run unless Notebook 07 implements a pre-registered compatible test.
- Category: Must for validation-reuse and data-snooping discussion.

### Useful - Hansen, "A Test for Superior Predictive Ability"

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569
- DOI: https://doi.org/10.2139/ssrn.264569
- Why it matters: SPA is a standard econometric response to multiple forecast
  comparisons and improves on the Reality Check in some settings.
- Use in this project: Mention as a future confirmatory route if locked
  per-sample loss sequences are defined before testing.
- Cannot support: Do not apply SPA to macro-F1 summary rows or present it as
  completed in Notebook 07.
- Category: Useful / optional.

### Must Add - Hanczar, "Performance Visualization Spaces for Classification with Rejection Option"

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0031320319302870
- DOI: https://doi.org/10.1016/j.patcog.2019.106984
- Why it matters: Provides reject-option visualization vocabulary and warns
  that rejection/error trade-offs must be interpreted jointly.
- Use in this project: Cite when presenting 06/07 no-trade coverage trade-offs,
  especially if plotting error-reject or cost-reject style figures.
- Cannot support: It does not validate this project's fixed coverage levels or
  trading utility costs.
- Category: Must for reject-option visualization and wording.

### Useful - Hooker, Mentch, and Zhou, "Unrestricted Permutation Forces Extrapolation"

- Springer: https://link.springer.com/article/10.1007/s11222-021-10057-z
- arXiv: https://arxiv.org/abs/1905.03151
- Why it matters: Explains why unrestricted permutation importance can create
  unrealistic extrapolation, especially when features are dependent.
- Use in this project: Cite in the permutation-importance appendix to justify
  grouped or block-aware feature perturbations and cautious wording.
- Cannot support: It does not make any specific feature causally important.
- Category: Useful for dependent-feature importance caveats.

### Must Add - Aas, Jullum, and Loland, "Explaining Individual Predictions When Features Are Dependent"

- arXiv: https://arxiv.org/abs/1903.10464
- DOI: https://doi.org/10.1016/j.artint.2021.103502
- Why it matters: Shows that Shapley approximations depend on assumptions about
  feature dependence, which is central for correlated OHLCV-derived features.
- Use in this project: Cite in any SHAP or LightGBM contribution appendix when
  explaining why background distributions and grouped dependent variables
  matter.
- Cannot support: It does not make SHAP causal, and it does not remove the
  need to keep explanation diagnostics locked to already-selected artifacts.
- Category: Must for dependent-feature SHAP caveats.

### Useful - Janzing, Minorics, and Bloebaum, "Feature Relevance Quantification in Explainable AI: A Causal Problem"

- PMLR: https://proceedings.mlr.press/v108/janzing20a.html
- arXiv: https://arxiv.org/abs/1910.13413
- Why it matters: Separates feature relevance and causal explanation, reducing
  the risk of over-interpreting SHAP or permutation plots.
- Use in this project: Cite when stating that feature-importance diagnostics
  describe a locked predictor, not a market mechanism.
- Cannot support: It does not invalidate all model explanations and does not
  authorize new feature selection in Notebook 07.
- Category: Useful for explanation caveats.

### Useful - Fisher, Rudin, and Dominici, "All Models are Wrong, but Many are Useful"

- JMLR: https://jmlr.csail.mit.edu/beta/papers/v20/18-760.html
- arXiv: https://arxiv.org/abs/1801.01489
- Why it matters: Introduces model class reliance and shows feature importance
  can vary across equally good models.
- Use in this project: Cite to qualify LightGBM feature importance, SHAP, and
  permutation diagnostics as model-specific explanation, not feature truth.
- Cannot support: Do not implement model-class reliance in Notebook 07 unless a
  separate locked model class and computational plan are pre-registered.
- Category: Useful for explanation uncertainty.

### Useful - Franc, Prusa, and Voracek, "Optimal Strategies for Reject Option Classifiers"

- arXiv: https://arxiv.org/abs/2101.12523
- JMLR: https://jmlr.org/papers/v24/21-0048.html
- Why it matters: Gives a modern cost-based reject-option framing that helps
  distinguish fixed coverage-grid diagnostics from an optimized rejection
  policy.
- Use in this project: Cite if Notebook 07 explains why 06 reports fixed
  coverage levels rather than claiming an economically optimal no-trade rule.
- Cannot support: It does not prove the project's confidence score or coverage
  grid is optimal for noisy five-minute stock data.
- Category: Useful for reject-option theory.

### Context Only - Zhang, Zohren, and Roberts, "DeepLOB"

- arXiv: https://arxiv.org/abs/1808.03668
- DOI: https://doi.org/10.1109/TSP.2019.2907260
- Why it matters: Well-known high-frequency stock direction paper using limit
  order book data and deep models.
- Use in this project: Cite only as background that high-frequency direction
  prediction often relies on richer LOB microstructure inputs.
- Cannot support: It does not justify adding DeepLOB to Notebook 07, and its LOB
  setting does not validate this project's five-minute OHLCV results.
- Category: Context only.

### Context Only - Huddleston, Liu, and Stentoft, "Intraday Market Predictability: A Machine Learning Approach"

- RePEc/Oxford record: https://ideas.repec.org/a/oup/jfinec/v21y2023i2p485-527..html
- DOI: https://doi.org/10.1093/jjfinec/nbab007
- Why it matters: Strong finance context for five-minute equity market
  predictability using machine learning.
- Use in this project: Cite as nearby intraday-market context while making clear
  that their predictors and economic-performance setting differ from this
  five-stock OHLCV-only classifier.
- Cannot support: Do not borrow their Sharpe, transaction-cost, or
  cross-sectional constituent-return conclusions.
- Category: Context only / finance boundary.

### Context Only - Kong, Zhu, and Azencott, "Predicting Intraday Jumps in Stock Prices Using Liquidity Measures and Technical Indicators"

- arXiv: https://arxiv.org/abs/1912.07165
- Why it matters: Provides another intraday prediction example using technical
  and liquidity-style features.
- Use in this project: Cite only as context for intraday prediction framing and
  the distinction between direction labels, jump labels, OHLCV, and richer
  liquidity inputs.
- Cannot support: It does not validate this project's five-stock directional
  labels or selected LightGBM artifact.
- Category: Context only.

### Risky Context - Briola, Bartolucci, and Aste, "Deep Limit Order Book Forecasting"

- arXiv: https://arxiv.org/abs/2403.09267
- DOI: https://doi.org/10.48550/arXiv.2403.09267
- Why it matters: Recent LOB-forecasting context and a useful boundary source
  for why high-frequency ML evidence can be data-modality specific.
- Use in this project: Cite only to separate rich LOB forecasting from
  five-minute OHLCV-only validation evidence.
- Cannot support: It does not justify adding LOB models or importing LOB
  performance claims into Notebook 07.
- Category: Risky / context only.

### Risky Context - Ait-Sahalia, Fan, Xue, and Zhu, "How and When Are High-Frequency Stock Returns Predictable?"

- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4095405
- DOI: https://doi.org/10.2139/ssrn.4095405
- Why it matters: Useful context that high-frequency predictability can depend
  on rich trades/quotes and timeliness.
- Use in this project: Cite only for limitations and data-modality boundaries.
- Cannot support: Its ultra-high-frequency trades/quotes setting is not
  comparable to five-minute OHLCV-only features.
- Category: Risky but useful context.

## Recommended Notebook 07 Experiment Modules

### 07A - Lockfile And Scope Gate

Required checks:

```text
scope = validation_only
holdout_test_authorized = false
official_candidate = h03_bps1p5 + price_volume_time + window_size=20
model/profile choices frozen before 07 final comparison
threshold/coverage choices frozen before 07 final comparison
same-row dummy baseline available for every reported comparison row
```

Fail fast if any artifact is missing, if a result row lacks same-row dummy
metadata, or if any file indicates holdout/test scoring.

### 07B - Final Validation-Only Table

Recommended columns:

```text
artifact_source
model
profile_id
profile_role
label_config
horizon_k
threshold_bps
feature_set
window_size
coverage
seed_count
macro_f1_mean
macro_f1_std
macro_f1_lcb_95
balanced_accuracy_mean
accuracy_mean
dummy_macro_f1_mean
delta_macro_f1_vs_dummy_mean
always_up_dummy_macro_f1_mean
delta_macro_f1_vs_always_up_dummy_mean
positive_ticker_count
top_ticker_gain_share
validation_n
scope
decision_source
allowed_wording_tag
```

Interpretation rule:

- `delta_macro_f1_vs_dummy_mean <= 0`: no validation signal.
- `0 < delta_macro_f1_vs_dummy_mean < 0.005`: weak/small validation-only signal.
- `delta_macro_f1_vs_dummy_mean >= 0.005`: practical validation-only signal, not
  holdout/test evidence.
- Concentration caveat applies if `positive_ticker_count < 4` or
  `top_ticker_gain_share > 0.35`.

### 07C - Validation-Budget Ledger

Ledger columns:

```text
artifact
notebook_stage
decision_made
decision_timing
model_families_considered
profiles_or_trials_considered
seeds_used
thresholds_or_coverages_considered
official_validation_rows_inspected
train_inner_only_decision
official_validation_informed_decision
holdout_test_contact
allowed_wording
```

Purpose:

- Make validation reuse visible.
- Distinguish train-inner HPO from official-validation confirmation.
- Prevent 07 from becoming an untracked selection layer.

### 07D - Per-Ticker And Seed Robustness

Recommended diagnostics:

- pooled summary by locked model/profile;
- per-ticker `delta_macro_f1_vs_dummy`;
- seed-level mean/std/LCB;
- heatmap: model/profile by ticker, colored by delta vs dummy;
- bar chart: per-ticker delta vs dummy for the final selected row;
- concentration metrics: `positive_ticker_count`, `top_ticker_gain_share`;
- note that shared chronological validation and overlapping windows mean seeds
  are not independent market samples.

### 07E - LightGBM Explainability Appendix

Allowed low-dependency path:

1. Report LightGBM `split` importance.
2. Report LightGBM `gain` importance.
3. If available from the existing LightGBM runtime, compute `pred_contrib=True`
   on locked validation samples.
4. If `shap` is already available or explicitly approved, add TreeSHAP summary
   plots for locked validation samples only.
5. Group related features where possible, for example price/return, volume,
   time-of-day, and technical-indicator groups.

Required caveats:

- Importance ranks are model-specific diagnostics.
- Correlated features can dilute or redistribute importance.
- SHAP values explain predictions under a specified background/perturbation
  assumption.
- No feature is added, removed, or reweighted based on this appendix.

### 07F - Permutation Importance Appendix

Allowed design:

- Run only on locked fitted models and locked validation sample ids.
- Use the same scoring metric family as the final table: macro F1 and balanced
  accuracy.
- Prefer feature-group permutation where features are correlated or represent the
  same concept.
- Repeat permutations with fixed seeds and report mean/std of score degradation.
- Label the appendix `diagnostic`.

Forbidden use:

- Do not use permutation importance to choose a new feature subset.
- Do not use iid row-wise permutation in a way that claims valid time-series
  inference.

### 07G - Null-Control Appendix

Allowed designs, from safest to riskiest:

1. Read-only reporting of an existing pre-registered null-control artifact if it
   already exists.
2. Day-block or ticker-day block label permutation, preserving within-day sample
   structure as much as possible.
3. Circular within-block shifts of labels/predictions as a diagnostic null.
4. Feature-family permutation within ticker/day blocks for feature-dependence
   diagnosis.

Required output:

```text
null_design
permutation_unit
n_permutations
score
observed_score
null_score_mean
null_score_p95
empirical_p_value_or_rank
dependency_caveat
scope = diagnostic
```

Strict wording:

- Allowed: "The observed validation-only delta was larger than the selected
  chronology-aware null-control distribution under this diagnostic design."
- Not allowed: "The model is statistically proven to generalize" or "this passes
  holdout/test."

### 07H - Paper-Ready Synthesis

Allowed wording:

> Under the locked chronological validation-only route, the selected LightGBM
> configuration produced a positive delta over same-row stratified dummy
> baselines across the five-ticker panel. Robustness diagnostics summarize
> seed-level variation, per-ticker breadth, feature-attribution consistency, and
> null-control behavior. Because the official validation period has been reused
> for a bounded sequence of screening, tuning, and diagnostic analyses, the
> conclusion remains validation-only and should not be read as independent
> holdout/test or live-trading evidence.

If 07 diagnostics are weak or mixed:

> The validation-only signal is positive but small and sensitive to the chosen
> diagnostic view. The result supports a cautious thesis claim about weak
> directional predictability under a locked protocol, not a claim of robust
> deployment readiness.

If null controls are not implemented:

> Null-control tests were scoped as a recommended appendix but not executed in
> this notebook. The absence of null-control results should be listed as a
> limitation rather than filled with speculative language.

## What Optimizes Model/Paper Persuasiveness Without Breaking The Route

High-value additions:

- final locked comparison table with dummy deltas;
- validation-budget ledger;
- per-ticker and seed robustness;
- feature-importance appendix with explicit caveats;
- SHAP/TreeSHAP appendix only if dependencies and artifacts already exist;
- permutation importance as locked-model inspection, not feature search;
- chronology-aware null-control appendix;
- conservative thesis wording.

Low-value or route-breaking additions:

- new model families;
- larger HPO budget after seeing results;
- official-validation-selected threshold or feature changes;
- iid permutation p-values;
- PnL/Sharpe retrofit;
- installing SHAP/MAPIE just to make 07 look richer.

## Open Issues For Main Thread

- Does Notebook 05 produce locked prediction artifacts with sample ids,
  probabilities, labels, timestamps, tickers, and seeds? If not, SHAP and
  permutation appendices should be limited or deferred.
- Is `shap` already available in the target Colab/runtime? If not, prefer
  LightGBM `pred_contrib=True` or skip SHAP unless dependency installation is
  explicitly approved.
- Does Notebook 06 freeze any selective coverage rule before 07? If not, 07
  should not include selective variants in the final comparison table.
- Should null controls be executed in 07, or listed as a separately
  pre-registered appendix? Execution would require a precise block-permutation
  design before any result is viewed.

## KB-Ready Tags

```text
project=intraday_stock_direction_research
notebook=07
scope=validation_only
topics=LightGBM,SHAP,TreeSHAP,permutation_importance,null_control,robustness,validation_reuse,benchmark_variance
hard_boundary=no_holdout_test,no_training_in_this_note,no_notebook_execution,no_feature_reselection,no_model_family_expansion
```
