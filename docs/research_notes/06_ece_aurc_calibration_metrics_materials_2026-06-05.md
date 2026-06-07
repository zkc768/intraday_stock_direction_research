# Notebook 06 ECE/AURC Calibration Metrics Materials - 2026-06-05

Scope: literature, code, tutorial, and method design research only. This note
does not authorize training, notebook execution, holdout/test access, Notebook
05 edits, dependency installation, commits, or pushes.

Project fit:

- Notebook 06 should be a validation-only post-prediction abstention /
  no-trade / calibration readout over the already selected Notebook 05
  LightGBM probability artifacts.
- Official validation must be a readout of a pre-registered full curve and
  fixed coverage grid. It must not be used to choose a best confidence
  threshold, best coverage, calibrator, feature, or model.
- ECE, Brier, AURC, and E-AURC are diagnostics for probability quality and
  selective-risk behavior. They are not selection gates for this project unless
  a separate train-inner calibration/selection split is frozen before reading
  official-validation results.

## Short Plan

1. Use Notebook 05 artifacts only after 05 review has frozen the LightGBM
   profile and produced validation prediction rows.
2. Compute full-row probability diagnostics on `prob_up`: Brier score,
   positive-class ECE, optional top-label ECE, and reliability tables.
3. Compute selective-risk diagnostics on `confidence = max(prob_up, 1 -
   prob_up)`: risk-coverage rows, AURC, and E-AURC.
4. Report the whole fixed coverage grid and the full risk-coverage curve.
   Do not choose an operating threshold from official validation.
5. Keep same-row dummy baselines and per-ticker retained counts in every
   selective table.

## Must Sources

### Guo et al. 2017 - On Calibration of Modern Neural Networks

- URL: https://proceedings.mlr.press/v70/guo17a.html
- arXiv: https://arxiv.org/abs/1706.04599
- Why it matters: Modern calibration anchor for reliability diagrams, ECE, and
  the distinction between accuracy and calibration.
- Use in this project: Cite for the need to evaluate probability calibration
  separately from macro F1 / balanced accuracy. Use its ECE/reliability
  vocabulary when describing Notebook 06 probability readout.
- Cannot support: It does not prove LightGBM probabilities are calibrated, does
  not validate time-series finance probabilities, and does not justify choosing
  a confidence threshold from official validation.

### Naeini, Cooper, and Hauskrecht 2015 - Bayesian Binning into Quantiles

- URL: https://ojs.aaai.org/index.php/AAAI/article/view/9602
- DOI: https://doi.org/10.1609/aaai.v29i1.9602
- Why it matters: Calibration paper tied to ECE/MCE-style histogram
  diagnostics and Bayesian binning into quantiles.
- Use in this project: Cite for the binning-based calibration-error family and
  for the reason quantile bins are attractive when predicted probabilities are
  clustered.
- Cannot support: BBQ itself is not required for Notebook 06 and should not be
  added as a new fitting method unless train-inner calibration data and a
  pre-registered comparison are available.

### Nixon et al. 2019 - Measuring Calibration in Deep Learning

- arXiv: https://arxiv.org/abs/1904.01685
- Why it matters: Practical critique of calibration metrics. It highlights that
  common ECE variants depend on binning choices and can hide class/score
  structure.
- Use in this project: Cite in the risk section: ECE is useful but not uniquely
  defined. Report binning method, bin counts, and sensitivity to equal-frequency
  versus equal-width bins.
- Cannot support: It does not make ECE a confirmatory metric, and it does not
  authorize selecting the ECE-minimizing calibrator on official validation.

### Kumar, Sarawagi, and Jain 2018 - Trainable Calibration Measures

- arXiv: https://arxiv.org/abs/1806.04490
- PMLR: https://proceedings.mlr.press/v80/kumar18a.html
- Why it matters: Shows that ordinary binned calibration errors are only one
  approximation and proposes kernel-based calibration measures.
- Use in this project: Cite as a caution that histogram ECE is an estimate with
  design choices, not the true calibration state of the model.
- Cannot support: Do not implement trainable calibration losses in Notebook 06;
  that would create a new model/optimization surface.

### Kumar, Liang, and Ma 2019 - Verified Uncertainty Calibration

- arXiv: https://arxiv.org/abs/1909.10155
- OpenReview: https://openreview.net/forum?id=SJx7F-4FvS
- Why it matters: Critiques finite-sample calibration evaluation and develops
  stronger uncertainty-calibration verification ideas.
- Use in this project: Cite for conservative wording around ECE estimates,
  especially when bins have limited samples.
- Cannot support: It does not provide a dependency-free Notebook 06
  implementation target, and it does not solve temporal dependence in intraday
  validation rows.

### Brier 1950 - Verification of Forecasts Expressed in Terms of Probability

- DOI: https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
- Why it matters: Original source for the Brier score, a proper score for
  probabilistic binary forecasts.
- Use in this project: Use Brier on `prob_up` versus `y_up` as a compact
  probability-quality diagnostic beside ECE/reliability diagrams.
- Cannot support: Brier is not the primary directional-classification metric
  for this project and should not replace macro F1, balanced accuracy, dummy
  baselines, or per-ticker reporting.

### scikit-learn probability calibration guide

- User guide: https://scikit-learn.org/stable/modules/calibration.html
- `calibration_curve`: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
- `CalibrationDisplay`: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html
- Why it matters: Official practical reference for reliability diagrams,
  calibration curves, Brier/log-loss caveats, sigmoid calibration, and isotonic
  calibration.
- Use in this project: Use the guide for plots and implementation vocabulary.
  If calibration fitting is later authorized, use only disjoint chronological
  train-inner calibration rows.
- Cannot support: Default CV or default examples do not satisfy this project's
  chronological, per-ticker, no-holdout rules by themselves.

### scikit-learn Brier score documentation

- URL: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html
- Why it matters: Official implementation of Brier score for binary
  probabilities.
- Use in this project: Compute `brier_score_loss(y_true, prob_up)` on full
  validation rows and optionally on retained rows for descriptive comparison.
- Cannot support: A lower Brier on a selected subset is conditional on the
  model's filtering and is not deployment evidence.

### Geifman and El-Yaniv 2017 - Selective Classification for Deep Neural Networks

- NeurIPS: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- arXiv: https://arxiv.org/abs/1705.08500
- Code: https://github.com/geifmany/selective_deep_learning
- Why it matters: Practical modern source for selective classification,
  coverage, selective risk, and risk-coverage curves.
- Use in this project: Use the risk-coverage vocabulary and the confidence
  sorting idea. For binary LightGBM, the score can be `confidence =
  max(prob_up, 1 - prob_up)` after deciding whether raw or calibrated
  probabilities are being read.
- Cannot support: It does not prove finance alpha, trading profitability,
  time-series validity, or holdout readiness.

### El-Yaniv and Wiener 2010 - Foundations of Selective Classification

- JMLR: https://jmlr.org/papers/v11/el-yaniv10a.html
- PDF: https://jmlr.csail.mit.edu/papers/volume11/el-yaniv10a/el-yaniv10a.pdf
- Why it matters: Foundation for the selective classifier, coverage, and
  selective-risk framing.
- Use in this project: Cite for definitions: coverage is the fraction of
  accepted predictions, and selective risk is error conditional on accepted
  predictions.
- Cannot support: The paper's theoretical assumptions do not convert noisy
  intraday stock classification into noise-free selective classification.

### Geifman and El-Yaniv 2019 - SelectiveNet

- arXiv: https://arxiv.org/abs/1901.09192
- Why it matters: Uses AURC and E-AURC as summary measures for
  risk-coverage behavior and provides a useful reference for E-AURC language.
- Use in this project: Cite for AURC/E-AURC terminology. Use E-AURC as
  `AURC - oracle_AURC` to describe excess area over an ideal ranking at the
  same error count.
- Cannot support: Do not implement SelectiveNet, selection losses, or target
  coverage training in Notebook 06. That is a new model family/path.

### TorchUncertainty classification metrics documentation

- Classification metrics: https://torch-uncertainty.github.io/auto_tutorials/Classification/tutorial_metrics.html
- Metrics API: https://torch-uncertainty.github.io/generated/torch_uncertainty.metrics.html
- GitHub: https://github.com/ENSTA-U2IS-AI/torch-uncertainty
- Why it matters: Practical reference implementation for uncertainty metrics,
  including calibration and selective-classification style metrics in a modern
  package.
- Use in this project: Read for formulas, naming, and sanity checks when
  writing dependency-free local code.
- Cannot support: Do not add TorchUncertainty as a dependency unless explicitly
  approved. Its torch-centric API is not needed for a LightGBM readout.

### Chalkidis and Savani 2021 - Trading via Selective Classification

- arXiv: https://arxiv.org/abs/2110.14914
- Why it matters: Directly maps selective classification to trading decisions
  where rejection corresponds to no position.
- Use in this project: Cite for the finance interpretation: coverage is an
  acted-upon fraction, and abstention can be described as no-trade at prediction
  time.
- Cannot support: It does not validate this project's stocks, 5-minute bars,
  labels, probability calibration, transaction costs, or profitability claims.

## Useful Sources

### Niculescu-Mizil and Caruana 2005 - Predicting Good Probabilities

- PDF: https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf
- Why it matters: Classic empirical source showing that classifiers with good
  accuracy may produce poor probabilities, and that calibration methods need
  separate data.
- Use in this project: Support the decision to evaluate probability quality
  before interpreting high-confidence LightGBM rows as reliable.
- Cannot support: It does not decide whether sigmoid or isotonic calibration is
  best for this data.

### Zadrozny and Elkan 2002 - Transforming Classifier Scores

- DOI: https://doi.org/10.1145/775047.775151
- Why it matters: Classic score-to-probability calibration reference.
- Use in this project: Background for calibration of model scores/probabilities.
- Cannot support: Do not fit calibration on official validation and report that
  same validation set as unbiased evidence.

### Platt 1999 - Probabilistic Outputs for Support Vector Machines

- Book chapter info: https://cir.nii.ac.jp/crid/1370851344281644289
- Why it matters: Classical sigmoid calibration source.
- Use in this project: Background only if Notebook 06 later uses sigmoid/Platt
  calibration on train-inner predictions.
- Cannot support: Platt scaling is not automatically valid for LightGBM or
  time-series data without a separate calibration split.

### LightGBM `LGBMClassifier` documentation

- Current API: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
- Versioned API: https://lightgbm.readthedocs.io/en/v4.4.0/pythonapi/lightgbm.LGBMClassifier.html
- Why it matters: Official API source for `predict_proba` and class-weight
  behavior.
- Use in this project: Treat `predict_proba` as the model's probability output
  to diagnose, not as automatically reliable market probability.
- Cannot support: The API docs do not prove the probabilities are calibrated in
  this project after class balancing or HPO.

### TorchMetrics calibration error documentation

- URL: https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html
- Why it matters: Compact implementation reference for expected, maximum, and
  RMS calibration error variants.
- Use in this project: Use as a formula cross-check if implementing local ECE.
- Cannot support: Do not add TorchMetrics as a dependency for a pandas/numpy
  LightGBM notebook unless separately approved.

### scikit-learn threshold tuning documentation

- User guide: https://scikit-learn.org/stable/modules/classification_threshold.html
- API: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
- Why it matters: Official documentation of threshold tuning as a supervised
  model-selection problem.
- Use in this project: Cite as a caution: threshold/coverage choice must be
  train-inner or pre-registered, not selected from official-validation readout.
- Cannot support: Do not use default threshold CV for this chronological
  intraday route.

### Cawley and Talbot 2010 - Over-fitting in Model Selection

- JMLR: https://jmlr.org/papers/v11/cawley10a.html
- Why it matters: Clear source for finite-validation selection bias.
- Use in this project: Support the rule that ECE/AURC curves are readout
  diagnostics unless selection was nested before official validation.
- Cannot support: It is not a finance-specific proof and does not define the
  project metrics.

### White 2000 - Reality Check for Data Snooping

- DOI: https://doi.org/10.1111/1468-0262.00152
- Publisher: https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
- Why it matters: Finance/econometrics anchor for repeated rule/model search on
  the same historical data.
- Use in this project: Cite in limitations and validation-budget discussion.
- Cannot support: Do not claim White's Reality Check was performed unless the
  exact bootstrap procedure and benchmark null are implemented.

## Optional Sources

### Fisch, Jaakkola, and Barzilay 2022 - Calibrated Selective Classification

- arXiv: https://arxiv.org/abs/2208.12084
- Code: https://github.com/ajfisch/calibrated-selective-classification
- Why it matters: Direct bridge between calibration and selective
  classification.
- Use in this project: Optional vocabulary for calibration on accepted samples
  versus full samples.
- Cannot support: Do not import DRO/selective training machinery into the
  current LightGBM Notebook 06 readout.

### MAPIE documentation

- API: https://mapie.readthedocs.io/en/latest/api.html
- Risk-control guide: https://mapie.readthedocs.io/en/stable/theoretical_description_risk_control.html
- GitHub: https://github.com/scikit-learn-contrib/MAPIE
- Why it matters: Practical conformal prediction and risk-control library.
- Use in this project: Optional future appendix if dependencies are already
  available or explicitly approved.
- Cannot support: Classical conformal guarantees rely on exchangeability; do
  not state conformal validity for intraday time series without a time-aware
  design.

### Angelopoulos and Bates 2021 - Gentle Introduction to Conformal Prediction

- arXiv: https://arxiv.org/abs/2107.07511
- Why it matters: Accessible conformal tutorial.
- Use in this project: Background for a possible future risk-control notebook.
- Cannot support: Not needed for the dependency-free first Notebook 06 metric
  implementation.

### Traub et al. 2024 - Overcoming Common Flaws in Selective Classification Evaluation

- arXiv: https://arxiv.org/abs/2407.01032
- NeurIPS PDF: https://papers.nips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf
- TorchUncertainty AUGRC docs: https://torch-uncertainty.github.io/generated/torch_uncertainty.metrics.classification.AUGRC.html
- Why it matters: Newer work critiques AURC-style aggregation and proposes
  AUGRC, the area under the generalized risk-coverage curve.
- Use in this project: Required caveat source for Notebook 06/07 wording when
  interpreting AURC/E-AURC. It supports treating AURC as a diagnostic summary
  and not as complete evidence that selective classification is robust.
- Cannot support: Do not add AUGRC to the first Notebook 06 unless its formula,
  purpose, and readout table are frozen before official-validation readout.

## Risky Or Easy-To-Misuse Materials

### Official-validation best threshold search

- Reference caution: https://scikit-learn.org/stable/modules/classification_threshold.html
- Use in this project: Treat as a prohibited pattern for official validation.
- Cannot support: It cannot justify selecting the best probability threshold or
  coverage level after seeing Notebook 06 curves.

### Isotonic calibration on small or reused samples

- Reference caution: https://scikit-learn.org/stable/modules/calibration.html
- Use in this project: Consider only with sufficient train-inner calibration
  rows and a frozen plan.
- Cannot support: It cannot be fit on official validation and then evaluated on
  the same official-validation rows as if independent.

### ECE-only claims

- Reference caution: https://arxiv.org/abs/1904.01685
- Use in this project: ECE should be one calibration diagnostic, with bin
  counts and sensitivity checks.
- Cannot support: ECE alone cannot prove selective no-trade safety, signal
  quality, or profitability.

### AURC-only claims

- Reference caution: https://arxiv.org/abs/1901.09192
- Use in this project: AURC summarizes confidence ranking over coverage.
- Cannot support: AURC is not a probability-calibration score and not a trading
  metric. Low AURC can come from ranking correct predictions earlier without
  making calibrated probability statements.

## Dependency-Free Implementation Guidance

### Input Columns Required From Notebook 05

Notebook 06 should stop with an exact missing-path or missing-column error if
any required artifact is absent.

```text
sample_id
ticker
timestamp
y_true
prob_up
pred_label
model_id
seed
scope
holdout_test_authorized
```

Optional but recommended:

```text
dummy_pred_label
split_name
label_config_id
feature_set_id
window_size
```

### Binary Probability And Confidence Definitions

Use two separate views:

```python
prob_up = np.asarray(prob_up, dtype=float)
y_true = np.asarray(y_true, dtype=int)  # 1 = up, 0 = down
pred_label = (prob_up >= 0.5).astype(int)
confidence = np.maximum(prob_up, 1.0 - prob_up)
correct = (pred_label == y_true).astype(int)
error = 1 - correct
```

Interpretation:

- `prob_up` is the positive-class probability used for Brier score and
  positive-class calibration.
- `confidence` is the top-label confidence used for selective sorting and
  risk-coverage.
- Raw LightGBM `prob_up` should be called a model probability output, not a
  true market probability.

### Brier Score

Dependency-free binary Brier:

```python
def brier_score_binary(y_true, prob_up):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob_up, dtype=float)
    if len(y) == 0:
        return np.nan
    return float(np.mean((p - y) ** 2))
```

Table columns:

```text
model_id
seed
row_scope = full_validation / selected_at_coverage
coverage_target
n
brier_prob_up
scope = validation_only
```

Risk note: Brier combines calibration, refinement/resolution, and class balance
effects. Do not use it as the primary classifier metric or a selection gate on
official validation.

### ECE: Equal-Frequency Versus Equal-Width Bins

Recommended first-pass reporting:

- Primary: positive-class ECE on `prob_up` versus `y_true`, with 10
  equal-frequency bins for pooled validation rows.
- Sensitivity: equal-width bins on `[0, 1]` if each non-empty bin has enough
  rows.
- Optional: top-label ECE on `confidence` versus `correct`, because selective
  analysis uses top-label confidence. Label it separately.

Dependency-free ECE helper:

```python
def calibration_bins(values, outcomes, n_bins=10, strategy="quantile"):
    v = np.asarray(values, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(v) & np.isfinite(o)
    v, o = v[mask], o[mask]
    if len(v) == 0:
        return []

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(v, quantiles)
        edges = np.unique(edges)
        if len(edges) <= 1:
            edges = np.array([v.min(), v.max()])
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"unknown bin strategy: {strategy}")

    rows = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            in_bin = (v >= left) & (v <= right)
        else:
            in_bin = (v >= left) & (v < right)
        if not np.any(in_bin):
            rows.append({
                "bin": i,
                "left": float(left),
                "right": float(right),
                "n": 0,
                "mean_score": np.nan,
                "empirical_rate": np.nan,
                "abs_gap": np.nan,
            })
            continue
        rows.append({
            "bin": i,
            "left": float(left),
            "right": float(right),
            "n": int(in_bin.sum()),
            "mean_score": float(v[in_bin].mean()),
            "empirical_rate": float(o[in_bin].mean()),
            "abs_gap": float(abs(v[in_bin].mean() - o[in_bin].mean())),
        })
    return rows

def ece_from_bins(rows):
    n_total = sum(row["n"] for row in rows)
    if n_total == 0:
        return np.nan
    return float(sum((row["n"] / n_total) * row["abs_gap"]
                     for row in rows if row["n"] > 0))
```

Positive-class ECE:

```python
prob_rows = calibration_bins(prob_up, y_true, n_bins=10, strategy="quantile")
ece_prob_up_quantile = ece_from_bins(prob_rows)
```

Top-label ECE:

```python
conf_rows = calibration_bins(confidence, correct, n_bins=10, strategy="quantile")
ece_top_label_quantile = ece_from_bins(conf_rows)
```

Report columns for each bin:

```text
model_id
seed
ece_type = positive_class_prob_up / top_label_confidence
bin_strategy = quantile / uniform
bin_index
bin_left
bin_right
n
mean_score
empirical_rate
abs_gap
scope = validation_only
```

Risk notes:

- Equal-frequency bins reduce empty-bin problems when `prob_up` clusters around
  a narrow range, which is likely in noisy intraday classification.
- Equal-width bins are easier to explain and reveal whether probabilities cover
  the full `[0, 1]` range, but can produce empty or tiny bins.
- Always report bin counts. Do not hide empty/tiny bins.
- Do not compare ECE values across different binning choices as if they were
  exact population quantities.

### Risk-Coverage, AURC, And E-AURC

Dependency-free curve:

```python
def risk_coverage_curve(y_true, prob_up):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_up, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    err = (pred != y).astype(float)

    order = np.argsort(-conf, kind="mergesort")
    err_sorted = err[order]
    conf_sorted = conf[order]
    k = np.arange(1, len(y) + 1)
    cumulative_errors = np.cumsum(err_sorted)
    coverage = k / len(y)
    selective_risk = cumulative_errors / k
    return {
        "order": order,
        "coverage": coverage,
        "selective_risk": selective_risk,
        "confidence_threshold": conf_sorted,
        "selected_n": k,
    }
```

Discrete AURC:

```python
def aurc_from_curve(selective_risk):
    r = np.asarray(selective_risk, dtype=float)
    if len(r) == 0:
        return np.nan
    return float(np.mean(r))
```

Oracle AURC and E-AURC:

```python
def oracle_aurc(y_true, prob_up):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_up, dtype=float)
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(float)
    oracle_err_sorted = np.sort(err)  # correct rows first, errors last
    k = np.arange(1, len(y) + 1)
    oracle_risk = np.cumsum(oracle_err_sorted) / k
    return aurc_from_curve(oracle_risk)

def e_aurc(y_true, prob_up):
    curve = risk_coverage_curve(y_true, prob_up)
    aurc = aurc_from_curve(curve["selective_risk"])
    return float(aurc - oracle_aurc(y_true, prob_up))
```

Recommended AURC/E-AURC summary table:

```text
model_id
seed
ticker_or_pooled
n
full_coverage_error
aurc
oracle_aurc
e_aurc
mean_confidence
scope = validation_only
```

Recommended fixed-grid readout table:

```text
model_id
seed
ticker_or_pooled
coverage_target
coverage_actual
confidence_threshold_implied
n_selected
n_abstained
selective_error
selective_accuracy
selective_macro_f1
selective_balanced_accuracy
dummy_macro_f1_same_rows
delta_macro_f1_vs_dummy
brier_selected_prob_up
ece_selected_prob_up_quantile
top_ticker_selected_share
scope = validation_only
```

Risk notes:

- AURC/E-AURC evaluate confidence ranking quality, not probability calibration.
- The selected subset is conditional on model confidence. It must be compared
  with same-row dummy and random-abstention baselines.
- If official validation is used only for readout, Notebook 06 may describe the
  curve shape but must not select a final threshold from it.
- Coverage below a pre-registered minimum, such as 0.30, should be
  visualization-only unless separately justified.

### Same-Row Dummy Baseline Under Abstention

For each coverage row, compute dummy baseline on exactly the retained sample
ids:

```python
selected_idx = curve["order"][:target_k]
y_selected = y_true[selected_idx]
dummy_selected_macro_f1 = score_dummy_on_same_rows(y_selected, train_label_rate)
delta_macro_f1_vs_dummy = selective_macro_f1 - dummy_selected_macro_f1
```

Risk note: The model chooses the retained rows, so same-row dummy is a
necessary baseline but not a complete fairness proof. Add random-abstention
baseline if implementation time allows.

## Recommended Notebook 06 Wording

Allowed:

```text
Notebook 06 reports probability calibration and selective-risk diagnostics for
the frozen Notebook 05 LightGBM probability artifacts under validation_only
scope. The coverage grid was fixed before official-validation readout. ECE,
Brier, AURC, and E-AURC are diagnostic summaries and are not used to choose a
deployment threshold.
```

Forbidden:

```text
The best coverage threshold is final.
ECE proves the model is calibrated.
AURC proves the strategy is tradable.
The high-confidence subset is safe for deployment.
The selective threshold is holdout-ready.
```

## Open Questions

- Does Notebook 05 save `prob_up` for the frozen model/profile/seed on official
  validation rows, with sample ids and ticker/timestamp alignment?
- Is train-inner out-of-fold probability output available for optional
  calibration fitting, or should Notebook 06 stay raw-probability readout only?
- What minimum per-ticker retained count should be frozen before official
  validation readout?
- Should top-label ECE be reported beside positive-class ECE, or kept as an
  appendix to avoid metric overload?

## Source Check Notes

Sources were checked by web search/open on 2026-06-05. Local project materials
inspected before writing this note:

- `AGENTS.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/research_notes/06_07_literature_materials_2026-06-05.md`
- `artifacts/research_packets/notebook06_07_selective_calibration_research_packet_2026-06-05.md`

No notebooks were run, no training was launched, no dependencies were
installed, no holdout/test artifact was accessed, and no files outside the
current project were modified.
