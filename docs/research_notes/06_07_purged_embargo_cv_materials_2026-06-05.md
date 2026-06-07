# Notebook 06/07 Purged And Embargoed CV Materials - 2026-06-05

Scope: research note only. No notebook execution, no training, no dependency
installation, no Notebook 05 edits, no commit, no push, and no holdout/test
access.

This note supplements:

- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `artifacts/research_packets/notebook06_07_selective_calibration_research_packet_2026-06-05.md`
- `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`

It focuses only on purged/embargoed cross-validation, financial time-series
cross-validation, CPCV/CSCV, and backtest-overfitting guardrails for planned
Notebook 06 and Notebook 07.

## Short Plan

1. Use financial-ML CV sources to define purging, embargoing, CPCV/CSCV, and
   backtest-overfitting risk.
2. Map those ideas onto the active route:
   `5-minute OHLCV`, `window_size=20`, `horizon_k=3`, per-ticker/day windows,
   chronological train/validation, and closed holdout/test.
3. Decide whether Notebook 05 needs protocol backfill, whether Notebook 06/07
   need guardrails, and whether implementation should be purged inner folds or
   only a limitation note.

## Project-Specific Interpretation

The current route already has the most important first-order leakage controls:
chronological splits, per-ticker/per-day windows, train-only preprocessing,
invalid labels at split boundaries, and no holdout/test use. Purged/embargoed
CV does not override those rules. It adds a sharper vocabulary for train-inner
model selection and calibration folds when each labeled sample has an event
interval that extends into future bars.

For this project, each supervised sample should be treated as an event:

```text
sample target bar       = t
feature window interval = [t - window_size + 1, t]
label event interval    = [t, t + horizon_k]
active candidate        = window_size=20, horizon_k=3
```

Any inner-validation fold defines a set of test events. A train event should be
purged if its label event interval overlaps an inner-validation event interval.
Because the active notebooks build windows per ticker, per split, and per
trading day, window-boundary leakage is mostly handled by construction: initial
validation targets without a full in-fold 20-bar history are skipped, and train
targets whose 3-bar label horizon crosses into validation are invalidated.

The practical guardrail is:

```text
purge label overlap:       at least horizon_k = 3 bars around inner fold edges
window boundary exclusion: enforce no window crossing, which costs up to
                           window_size - 1 = 19 bars at the beginning of each
                           validation block
embargo after test fold:   required only if future-side train samples exist
                           after an inner-validation block
```

For the current expanding-origin design, future-side train samples after a
validation block should not exist. That means a classic post-test embargo is
less important than explicit event-overlap purging and no-cross-boundary window
construction. If any future Notebook 06/07 design uses KFold/CPCV-style folds
with train groups on both sides of a validation group, embargo becomes mandatory
and must be larger than the maximum feature/history or label-overlap risk being
allowed across folds.

## Source List

### Must

#### 1. Marcos Lopez de Prado, `Advances in Financial Machine Learning`

- URL: https://www.wiley-vch.de/de/fachgebiete/finanzen-wirtschaft-recht/advances-in-financial-machine-learning-978-1-119-48208-6
- O'Reilly listing: https://www.oreilly.com/library/view/advances-in-financial/9781119482086/
- Type: book; primary seed source.
- Relevant pieces: Chapter 7, cross-validation in finance; Chapter 9,
  hyperparameter tuning with CV; Chapter 11, dangers of backtesting; Chapter 12,
  backtesting through cross-validation.
- How to use in this project: cite as the conceptual source for treating each
  labeled financial sample as an event with a prediction/trade time and an
  event/end time, then purging overlapping train events from inner-validation
  folds. Use it to justify backfilling Notebook 05's `INNER_PURGE_HORIZON_BARS`
  definition into an explicit event-interval rule.
- Cannot support: it does not authorize non-chronological selection, repeated
  official-validation tuning, or reopening holdout/test. It also does not make
  CPCV automatically appropriate for a thesis route that has explicitly chosen
  chronological validation.

#### 2. mlfinlab documentation, Purged/Embargo CV and CPCV

- URL: https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html
- Type: implementation documentation.
- How to use in this project: use as a concrete explanation of purging
  training samples whose information overlaps test samples, embargoing
  observations adjacent to test folds, and CPCV's multiple-path view. Useful for
  notebook markdown diagrams and implementation checklists.
- Cannot support: documentation is not proof that a copied implementation is
  correct for this project. The local route must still enforce ticker/day/split
  boundaries, same-row dummy baselines, and no holdout/test contact.

#### 3. skfolio `CombinatorialPurgedCV`

- URL: https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html
- GitHub: https://github.com/skfolio/skfolio
- Type: maintained Python documentation and implementation.
- How to use in this project: cite as a modern, sklearn-style reference for
  `purged_size`, `embargo_size`, and combinatorial purged paths. Its API is a
  useful design reference if 07 later needs a robustness appendix.
- Cannot support: do not add `skfolio` as a dependency without explicit
  approval. Its portfolio-optimization context and observation-count
  purge/embargo parameters still need adaptation to this project's per-ticker
  5-minute sample/event intervals.

#### 4. Bailey, Borwein, Lopez de Prado, and Zhu, `The Probability of Backtest Overfitting`

- DOI: https://doi.org/10.21314/JCF.2016.322
- SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Metadata mirror: https://www.semanticscholar.org/paper/The-probability-of-back-test-over-fitting-Bailey-Borwein/b1233b4f5384f003e85c2e0eec1a2dfc08f624c5
- Type: peer-reviewed finance/backtesting paper.
- How to use in this project: use in Notebook 07 to justify a validation-budget
  ledger and cautious wording after multiple labels, feature sets, windows,
  models, HPO trials, seeds, and coverage thresholds have been examined.
- Cannot support: PBO/CSCV is defined for strategy/backtest selection surfaces.
  Unless this project implements the PBO procedure on an appropriate return or
  loss matrix, cite it as overfitting risk and trial-accounting rationale, not
  as a computed result.

#### 5. Cawley and Talbot, `On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation`

- URL: https://jmlr.org/papers/v11/cawley10a.html
- PDF: https://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf
- Type: peer-reviewed ML methodology paper.
- How to use in this project: supports Notebook 05's separation between
  train-inner HPO and official-validation confirmation, and supports Notebook
  06/07 warnings against selecting thresholds from the same validation curves
  later reported as evidence.
- Cannot support: it is not finance-specific and does not define purging or
  embargoing.

#### 6. scikit-learn `TimeSeriesSplit`

- URL: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Type: official implementation documentation.
- How to use in this project: use only as a basic chronological splitter
  reference. Its `gap` parameter is a useful analogy for a simple embargo/gap.
- Cannot support: default `TimeSeriesSplit` does not know about ticker
  ownership, trading-day boundaries, event end times, label horizons, same-row
  dummy baselines, or the closed holdout/test boundary. It is not sufficient by
  itself for Notebook 05/06.

#### 7. QuantInsti tutorial, `Cross Validation in Finance: Purging, Embargoing, Combinatorial`

- URL: https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/
- Type: practitioner tutorial.
- How to use in this project: useful for explanatory markdown and intuition:
  embargo handles feature-history/serial-dependence leakage near fold edges;
  purging handles label event overlap between train and test events.
- Cannot support: tutorial code is not a project dependency and should not be
  copied as authoritative implementation. It includes backtest/PnL examples
  outside the current validation-only classifier route.

#### 8. `timeseriescv` GitHub repository

- URL: https://github.com/sam31415/timeseriescv
- Type: open-source sklearn-style CV implementation.
- How to use in this project: useful design reference because its split API
  explicitly accepts `pred_times` and `eval_times`, matching the event-interval
  framing needed for purging.
- Cannot support: small/older library; do not install or import without
  approval. Treat as reference code only.

### Useful

#### 9. Bergmeir and Benitez, `On the use of cross-validation for time series predictor evaluation`

- DOI: https://doi.org/10.1016/j.ins.2011.12.028
- Metadata: https://colab.ws/articles/10.1016%2Fj.ins.2011.12.028
- Type: peer-reviewed time-series CV study.
- How to use in this project: cite for the broader point that blocked or
  time-aware CV can be useful for time-series model selection when dependence is
  handled explicitly.
- Cannot support: it does not justify shuffled splits, random train/test splits,
  or ignoring this project's stricter financial label/window constraints.

#### 10. Hyndman time-series cross-validation materials

- Book: https://otexts.com/fpp3/
- Blog: https://robjhyndman.com/hyndsight/tscv/
- R `tsCV`: https://pkg.robjhyndman.com/forecast/reference/tsCV.html
- Type: forecasting tutorial/reference.
- How to use in this project: useful for simple diagrams of rolling-origin
  evaluation and for explaining expanding-origin folds in Notebook 05/06.
- Cannot support: forecasting examples are not financial classification with
  overlapping label horizons. They need adaptation to sample event intervals.

#### 11. R `pbo` package

- RDocumentation: https://www.rdocumentation.org/packages/pbo/versions/1.3.4/topics/pbo
- rdrr overview: https://rdrr.io/cran/pbo/man/pbo-package.html
- Type: implementation package for Bailey et al. PBO/CSCV ideas.
- How to use in this project: cite as evidence that PBO can be operationalized
  when there is a proper strategy/result matrix.
- Cannot support: do not add an R dependency. It is not directly applicable to
  macro-F1 classifier rows unless 07 defines a valid candidate-by-time loss or
  return matrix.

#### 12. White, `A Reality Check for Data Snooping`

- DOI: https://doi.org/10.1111/1468-0262.00152
- Econometrica page: https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
- Type: peer-reviewed econometrics paper.
- How to use in this project: supports a 07 validation-reuse ledger and
  multiple-comparison caution when many candidate rules or thresholds have been
  tried.
- Cannot support: do not say Reality Check was applied unless a bootstrap
  procedure, benchmark null, and loss series are actually implemented.

#### 13. Sullivan, Timmermann, and White, `Data-Snooping, Technical Trading Rule Performance, and the Bootstrap`

- PDF: https://researchonline.lse.ac.uk/id/eprint/119144/1/dp303.pdf
- RePEc: https://ideas.repec.org/RePEc:ehl:lserod:119144
- Type: peer-reviewed/econometrics working-paper record.
- How to use in this project: cite for finance-specific threshold/rule-search
  danger, especially if 06 is tempted to pick a high-confidence threshold after
  reading validation curves.
- Cannot support: it does not validate any specific classifier or no-trade band.

#### 14. Hansen, `A Test for Superior Predictive Ability`

- URL: https://ideas.repec.org/a/bes/jnlbes/v23y2005p365-380.html
- DOI: https://doi.org/10.1198/073500105000000063
- Type: peer-reviewed econometrics paper.
- How to use in this project: optional statistical-method reference for 07 if a
  carefully defined paired loss sequence and candidate family exist.
- Cannot support: do not use SPA as a checkbox on macro-F1 summary rows.

#### 15. Harvey, Liu, and Zhu, `... and the Cross-Section of Expected Returns`

- DOI: https://doi.org/10.1093/rfs/hhv059
- PDF: https://academic.oup.com/rfs/article-pdf/29/1/5/24450796/hhv059.pdf
- Type: peer-reviewed empirical-finance paper.
- How to use in this project: supports stronger caution around many tried
  research degrees of freedom.
- Cannot support: factor-discovery t-stat thresholds do not transfer directly to
  5-minute directional classification.

#### 16. Roberts et al., `Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure`

- DOI: https://doi.org/10.1111/ecog.02881
- PDF: https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file
- Type: peer-reviewed structured-CV methodology.
- How to use in this project: good general citation that dependent/structured
  observations need blocked CV rather than iid random CV.
- Cannot support: ecology examples are not financial market evidence.

### Optional

#### 17. scikit-learn gap examples for time-related features

- Time-related feature engineering: https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- Lagged features example: https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html
- Type: official examples.
- How to use in this project: cite as implementation intuition for adding a gap
  between train and validation in time-ordered data.
- Cannot support: examples are not enough for event-overlap purging and do not
  handle ticker/day/window ownership.

#### 18. `mlfinlab` GitHub issue on PurgedKFold event overlap

- URL: https://github.com/hudson-and-thames/mlfinlab/issues/295
- Type: implementation bug report/discussion.
- How to use in this project: useful as a warning that purged-CV
  implementations are easy to get wrong; tests should assert event intervals
  do not overlap, not only that a class name says `PurgedKFold`.
- Cannot support: an open issue is not a peer-reviewed source and should not be
  cited as methodological authority.

#### 19. mlpack issue requesting PurgedKFoldCV

- URL: https://github.com/mlpack/mlpack/issues/3830
- Type: open-source issue.
- How to use in this project: optional evidence that purged/embargoed CV is
  recognized as a distinct requirement by ML practitioners.
- Cannot support: issue was closed as not planned and is not implementation
  guidance.

#### 20. AFML chapter-answer notebooks and community repos

- Example GitHub: https://github.com/WongYatChun/Advances-in-Financial-Machine-Learning/blob/master/cvFin.py
- Type: educational/community code.
- How to use in this project: read for pseudocode or sanity checks only.
- Cannot support: do not copy without audit; many AFML educational repos are
  partial, stale, or not tested against this project's event definitions.

### Risky

#### 21. CPCV as a direct replacement for chronological validation

- Reference source: `Advances in Financial Machine Learning`, Chapter 12.
- Implementation references: mlfinlab docs and skfolio `CombinatorialPurgedCV`.
- Why risky: CPCV recombines many train/test fold choices to produce multiple
  paths. That is useful for backtest-overfitting analysis, but it can train on
  later periods and test on earlier periods depending on split construction.
- Current recommendation: do not replace Notebook 05 train-inner expanding
  origin with CPCV. Do not use CPCV to choose 06 thresholds or 07 winners. At
  most, use CPCV as a later diagnostic appendix if all model/threshold choices
  are already frozen and every path keeps event purging, embargoing,
  ticker/day ownership, and no holdout/test.

#### 22. Default sklearn threshold/calibration CV

- Threshold docs: https://scikit-learn.org/stable/modules/classification_threshold.html
- `TunedThresholdClassifierCV`: https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
- Why risky: default CV behavior is not the active project's chronological,
  per-ticker, per-day, event-purged design.
- Current recommendation: use as warning/API reference only. Any actual
  threshold or calibration fitting in 06 must use custom purged chronological
  splits, not default random or stratified folds.

#### 23. Purged CV as authorization to keep searching official validation

- Why risky: purging reduces fold leakage; it does not eliminate selection bias
  from repeatedly looking at the same official validation period.
- Current recommendation: every Notebook 06 coverage point and every Notebook
  07 comparison must remain pre-registered and validation_only. Purged CV is
  not an authorization to reopen official validation selection loops or
  holdout/test.

## Concrete Recommendations

### Should Notebook 05 be backfilled?

Yes, but only as a protocol clarification or static guardrail. Do not edit or
rerun Notebook 05 in this task.

Notebook 05 already declares:

```text
INNER_FOLD_STYLE = chronological_expanding_origin
INNER_PURGE_HORIZON_BARS = 3
INNER_WINDOW_CROSS_BOUNDARY_POLICY = forbidden
```

That is directionally correct for `horizon_k=3`, but it should be made more
explicit before any future implementation review:

1. Define each sample's `event_start = target_timestamp` and
   `event_end = timestamp shifted + horizon_k bars` inside the same ticker and
   trading day.
2. For every inner fold, assert that no inner-train sample has a label event
   interval overlapping any inner-validation event interval.
3. Keep the existing rule that windows are constructed per ticker, per fold,
   and per trading day; this naturally removes the first `window_size - 1 = 19`
   validation targets in each fold if their history would cross the fold
   boundary.
4. If any future implementation allows future-side train samples after an
   inner-validation block, add explicit `embargo_bars >= max(horizon_k,
   feature_history_gap)` and explain why the value is sufficient.

For the current expanding-origin 05 route, implement purged expanding folds,
not CPCV.

### Does Notebook 06 need protocol guardrails?

Yes. Notebook 06 is the most important place to add purged/embargoed CV
guardrails because OOF probabilities and probability calibration can create a
new selection surface.

Recommended 06 guardrail:

```text
OOF/calibration source = official training partition only
fold style             = purged chronological expanding-origin
event purge            = no train event interval overlaps calibration/OOF event interval
window policy          = no ticker/day/fold boundary crossing
calibrator fit         = train-inner OOF/calibration rows only
coverage thresholds    = train-inner calibrated confidence or pre-registered grid only
official validation    = readout only
holdout/test           = closed
```

If 05 does not produce train-inner OOF probabilities, 06 should either:

1. stop and report the missing exact Notebook 05 probability artifact path, or
2. run a separately pre-registered 06A routine that refits the frozen 05 profile
   inside official training only to create purged OOF probabilities.

Option 2 would be fitting, so it requires explicit notebook protocol approval
before implementation. It is not authorized by this research note.

### Does Notebook 07 need protocol guardrails?

Yes. Notebook 07 should include a validation-reuse and cross-validation
limitations section.

Required 07 guardrails:

1. Do not use CPCV/PBO to select a new winner after seeing 05/06 results.
2. Use a validation-budget ledger that counts Stage 0, Notebook 03/04/05, and
   Notebook 06 coverage-grid contacts.
3. If reporting robustness intervals, prefer seed-level, ticker-level, or
   ticker/day/block resampling. Avoid iid row bootstrap over overlapping
   windows as confirmatory evidence.
4. If using CPCV/CSCV, label it diagnostic only unless its split design is
   pre-registered before results and does not touch holdout/test.
5. Keep all conclusions `validation_only`; do not write evidence-ready,
   holdout-ready, trading-ready, or deployment-ready claims.

### Implement purged inner folds or only write limitation?

Implement purged inner folds for any future train-inner HPO, OOF probability,
or calibration-fitting step. A limitation note alone is not enough when the
notebook uses folds to choose hyperparameters, calibrators, or thresholds.

For Notebook 05:

- recommended: purged chronological expanding folds;
- no CPCV;
- no official-validation early stopping;
- no official-validation finalist replacement;
- no holdout/test.

For Notebook 06:

- recommended first pass: no new calibration fitting; evaluate raw probabilities
  under fixed coverage grid if 05 artifacts already exist;
- if calibration is needed: train-inner purged OOF probabilities only;
- official validation is readout only.

For Notebook 07:

- recommended: validation-budget ledger plus robustness diagnostics;
- optional: CPCV/PBO literature discussion or diagnostic appendix;
- not recommended: CPCV as a model-selection engine.

## Minimal Implementation Sketch For Future Notebook Builders

This is pseudocode only. It is not executed in this task.

```python
def event_interval_for_sample(sample):
    return {
        "ticker": sample.ticker,
        "day": sample.trading_day,
        "start": sample.target_timestamp,
        "end": sample.horizon_end_timestamp,  # target + horizon_k bars
    }


def overlaps(a, b):
    if a["ticker"] != b["ticker"]:
        return False
    if a["day"] != b["day"]:
        return False
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def purge_train_against_validation(train_samples, validation_samples):
    validation_events = [event_interval_for_sample(s) for s in validation_samples]
    kept = []
    purged = []
    for sample in train_samples:
        event = event_interval_for_sample(sample)
        if any(overlaps(event, val_event) for val_event in validation_events):
            purged.append(sample)
        else:
            kept.append(sample)
    return kept, purged
```

Future implementation tests should assert:

```text
no train/validation event interval overlap by ticker/day
no window crosses ticker/day/fold boundary
no label horizon crosses ticker/day/fold boundary
no scaler/calibrator fit on validation or holdout/test rows
no holdout/test rows read except as closed boundary markers where already allowed
same-row dummy baseline exists for every validation/readout subset
```

## KB-Ready Takeaway

Purged/embargoed cross-validation is useful here, but the useful part is not a
new model-selection license. The useful part is a stricter fold-construction
contract: every sample has a feature window and a future label event interval,
and fold boundaries must remove training samples whose label events overlap
validation events. For the current `window_size=20`, `horizon_k=3` LightGBM
route, Notebook 05's expanding-origin HPO should be formalized as purged
expanding-origin folds. Notebook 06 should use the same purged fold logic if it
creates OOF probabilities or fits calibration. Notebook 07 should use CPCV/PBO
mainly to explain validation-reuse risk and, at most, as a diagnostic robustness
appendix after all choices are frozen. None of these methods permits reopening
official validation selection loops or holdout/test.

## Unresolved Questions

- Will Notebook 05 produce train-inner OOF probability artifacts, or only
  official-validation probabilities?
- If 06 needs calibration, will the project authorize refitting the frozen 05
  profile inside official training to create purged OOF probabilities?
- Should 07 include a CPCV/PBO diagnostic appendix, or keep CPCV/PBO as
  limitations language only?
- What exact artifact path will store 05 probabilities and sample ids for 06?

## Source-Check Notes

Sources were checked live through web search/open on 2026-06-05. Local project
rules and the required project materials were inspected before this note was
written. No notebooks were run, no training was launched, no dependencies were
installed, and no holdout/test artifact was accessed.
