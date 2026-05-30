# Paper Innovation Roadmap - 2026-05-25

## Decision

The next project phase should support a paper narrative, not model expansion.

Current evidence keeps the PatchTST/new-model gate closed:

```text
canonical full-binary best pooled model: tcn, delta_macro_f1_vs_dummy = -0.002318
0bps diagnostic best pooled model:      tcn, delta_macro_f1_vs_dummy = +0.004890
5bps diagnostic best pooled model:      lstm, delta_macro_f1_vs_dummy = +0.001893
continuation threshold:                 +0.010000
```

The paper direction is therefore:

```text
Protocol-safe weak-signal evaluation for intraday stock direction
classification.
```

This is not a negative project outcome. It is a defensible research framing:
when strict protocol, coverage, ticker, seed, and dummy baselines are disclosed,
model complexity is not yet justified by the local evidence.

## Candidate Titles

1. Protocol-Safe Evaluation of Weak Signals in Intraday Stock Direction
   Classification
2. When Sequence Models Do Not Beat Dummy: A Leakage-Safe Study of Intraday
   Stock Direction Prediction
3. Beyond Model Stacking: Robust Baselines and Protocol Control for
   High-Frequency Stock Direction Classification

## One-Sentence Contribution

We introduce a leakage-safe evaluation harness for intraday stock direction
classification and show that, under chronological splits, train-only scaling,
dummy baselines, coverage disclosure, ticker-level analysis, and seed reporting,
standard LSTM, TCN, and DLinear sequence baselines do not robustly outperform a
stratified dummy classifier.

## Core Claims

Allowed claims:

- Under the current 5-minute, 5-stock, 3-seed, canonical full-binary setting,
  LSTM, TCN, and DLinear do not robustly outperform `dummy_stratified`.
- Diagnostic no-trade-band settings show that retained coverage and label
  semantics materially change interpretation.
- Current evidence supports protocol-first weak-signal analysis rather than
  architecture-first model expansion.
- The external-method gate is a reproducibility and contamination-control
  mechanism, not an accuracy improvement claim.

Blocked claims:

- The project discovered a profitable trading strategy.
- Stocks are unpredictable in general.
- Transformers or time-series foundation models fail on this task.
- CSCO/KO diagnostics prove a general alpha signal.
- 0bps or 5bps diagnostic results are equivalent to the canonical full-binary
  task.
- Complex models are useless. The supported claim is narrower: current evidence
  does not justify opening the new-model gate.

## Minimum Evidence Package

The minimum paper-ready package should contain:

| Artifact | Source | Purpose | Training Needed |
|---|---|---|---|
| Table 1: run gate summary | `run_summary.csv` | Show canonical/diagnostic regimes and gate status | No |
| Table 2: pooled model vs dummy | `pooled_by_model.csv` | Main canonical result and diagnostic comparison | No |
| Table 3: per-ticker dummy delta | `by_model_ticker.csv` | Show ticker-local heterogeneity | No |
| Table 4: coverage and label semantics | `coverage_by_ticker.csv` | Separate task semantics from retained subset effects | No |
| Figure 1: protocol firewall | docs/spec | Explain project-owned labels, splits, scaler, metrics, and adapter firewall | No |
| Figure 2: delta vs retained coverage | report CSVs | Show that diagnostic gains are below gate and coverage dependent | No |
| Figure 3: model x ticker heatmap | `by_model_ticker.csv` | Show all canonical per-ticker means are negative | No |
| Sanity table | existing shuffled-label smoke docs | Support leakage-safe workflow claim | No |

## Figure 1 Design

Figure 1 should be a protocol diagram, not a model architecture diagram.

Suggested layout:

```text
external papers / repos / local references
        |
        v
external-method review gate
        |
        +-- blocked: loaders, labels, splits, scalers, metrics, backtests
        |
        v
small adapter ideas only
        |
        v
hf_stock_clf-owned harness
        |
        +-- canonical binary labels
        +-- chronological splits
        +-- train-only scaler
        +-- no-cross-day windows/horizons
        +-- window-end label alignment
        +-- dummy baselines and macro F1
        |
        v
canonical table + diagnostic coverage tables + per-ticker deltas
```

Bottom-line caption:

```text
The harness treats external methods as hypotheses, not protocol owners.
```

## Related Work Groups

Use method-level grouping rather than paper-by-paper summaries:

1. Financial ML evaluation discipline:
   Gu/Kelly/Xiu, reliable stock-prediction evaluation, out-of-sample caution.
2. Time-series classification baselines:
   Great TSC Bake Off, ROCKET, MiniROCKET, InceptionTime.
3. Simple and linear time-series models:
   DLinear/LTSF-Linear, affine/normalization studies, stock-specific linear
   baselines.
4. Selective prediction and coverage:
   selective classification, abstention, risk-coverage, no-trade framing.
5. Large external time-series models and drift risk:
   PatchTST, iTransformer, Mantis/FM4MTSC, Chronos, MOMENT, Moirai.

Large external models should remain related-work context until an explicit
adapter gate is passed.

## No-Training Analysis Roadmap

These steps are the preferred next executable work. They use existing outputs
and do not touch `ml_utils`.

### Step 1 - Paper Tables

Input:

```text
checkpoints/phase1b_local_reports/table_records_20260525/run_summary.csv
checkpoints/phase1b_local_reports/table_records_20260525/pooled_by_model.csv
checkpoints/phase1b_local_reports/table_records_20260525/by_model_ticker.csv
checkpoints/phase1b_local_reports/table_records_20260525/coverage_by_ticker.csv
```

Output:

- manuscript-ready CSV or Markdown tables;
- numeric formatting with consistent precision;
- explicit `NA` for unavailable older-run fields.

Stop rules:

- Do not fabricate baseline balanced accuracy or confusion matrices for older
  runs.
- Do not compute confidence intervals unless the formula and assumptions are
  explicitly recorded.

### Step 2 - Ticker-Local Heterogeneity

Input:

```text
by_model_ticker.csv
completed run results.csv files if seed-level rows are needed
```

Output:

- heatmap-ready table: `run_id x model x ticker`;
- per-ticker mean/std of `delta_macro_f1_vs_ticker_dummy`;
- count of positive/negative ticker-level deltas by regime.

Stop rules:

- CSCO/KO positives must be marked post-hoc or hypothesis-generating.
- Every ticker-local figure must show the pooled result or cite it nearby.

### Step 3 - Threshold And Coverage Stability

Input:

- existing canonical, 0bps, and 5bps report outputs;
- existing threshold-sensitivity manifest, if available.

Output:

- threshold vs retained coverage;
- threshold vs test-window count;
- threshold vs test class balance;
- coverage fragility note for low-window regimes such as 10bps.

Stop rules:

- The canonical full-binary task remains the primary task.
- The best diagnostic threshold must not replace the canonical task.
- Manifest-first only unless a later prompt explicitly approves training.

### Step 4 - Risk-Coverage Proxy

Current limitation:

```text
No per-sample prediction confidence export exists yet.
```

Near-term proxy:

- treat threshold/no-trade retained coverage as an abstention proxy;
- plot coverage against macro F1, balanced accuracy, and dummy delta;
- label the plot as threshold-retention proxy, not confidence-based selective
  classification.

Stop rules:

- Do not call this a confidence curve.
- Do not infer calibration without logits/probabilities.

### Step 5 - Confidence-Based Risk-Coverage Spec

Future-only design:

```text
predictions.csv schema:
run_id, label_mode, model_name, ticker, seed, timestamp, y_true, y_pred,
logit_0, logit_1, prob_0, prob_1, confidence, split
```

Rules:

- prediction export must preserve project labels, splits, scaler, and metrics;
- confidence curves must be computed after model outputs are produced;
- no external model or dependency is required for this spec.

Stop rule:

- If this requires changing labels, split logic, scaler behavior, or metrics,
  stop and return to design review.

### Step 6 - Window/Horizon Stability Manifest

Preferred first step:

```text
manifest-only sweep
```

Candidate axes:

```text
window_size:     12, 24
label_horizon_k: 12, 24
threshold_bps:   0, 5, 10
```

Output:

- coverage heatmap;
- class-balance heatmap;
- test-window count heatmap;
- list of settings too narrow for model training.

Stop rules:

- No full training grid without explicit approval.
- Any runner CLI change must be scoped to the runner and runner tests.

### Step 7 - Sanity Evidence Integration

Input:

- existing shuffled-label smoke run notes;
- consolidated report tables.

Output:

- one small sanity table;
- one paragraph in methods or experiments explaining that shuffled labels did
  not produce a stable positive dummy delta.

Stop rules:

- Shuffled-label sanity supports leakage checking only.
- It does not prove that the unshuffled models are useful.

## Future Method Specs

### MiniROCKET / ROCKET

Status:

```text
future spec only
```

Why it matters:

- strong, simple time-series classification baseline;
- more aligned with current evidence than larger sequence models.

Blocked until:

- dependency/license review passes;
- train-only fit/transform plan is specified;
- metrics remain project-owned;
- no `aeon`, `sktime`, or official code is imported into `ml_utils`.

### Foundation Models

Status:

```text
related work and future work only
```

Allowed current use:

- source-review cards;
- mismatch tables for task, data modality, labels, metrics, and protocol.

Blocked claim:

```text
Foundation models fail on this project.
```

That claim requires a future approved adapter and local evaluation.

## Draft Paper Outline

1. Introduction
   - Problem: high-frequency stock direction work is vulnerable to leakage,
     baseline weakness, and protocol drift.
   - Claim: protocol-first evaluation changes interpretation before model
     expansion is justified.
2. Evaluation Harness
   - labels, splits, scaler, no-cross-day constraints, window-end alignment,
     metrics, dummy baselines.
3. Experimental Setup
   - data, tickers, features, models, seeds, canonical and diagnostic regimes.
4. Results
   - canonical full-binary table;
   - diagnostic no-trade-band comparison;
   - per-ticker heterogeneity;
   - coverage and label-semantics analysis.
5. Protocol Analysis
   - threshold-retention proxy;
   - shuffled-label sanity;
   - external-method gate.
6. Related Work
   - grouped by research function, not by chronological list.
7. Limitations
   - 5 tickers;
   - 3 seeds;
   - short local training;
   - OHLCV/technical features only;
   - no transaction costs, execution, or backtesting;
   - no claim about market predictability in general.
8. Conclusion
   - current evidence supports weak-signal evaluation and protocol control;
   - simple baselines and risk-coverage analysis should precede larger models.

## Next Concrete Implementation Candidate

If the next prompt asks for executable work, the safest candidate is:

```text
Create a paper-table/figure-prep script that reads the existing consolidated
report CSVs and writes paper-ready tables under a new report output directory.
```

Suggested write scope:

```text
target_file: scripts/phase1b_local/build_paper_tables.py
test_file:   tests/test_phase1b_paper_tables.py
output_dir:  checkpoints/phase1b_paper_tables_YYYYMMDD
```

This should not touch:

```text
ml_utils/
requirements.txt
reference_excerpts/
notebooks/
completed run directories
```

## Current Status

Proceed with paper-table and analysis automation before any new model work.

No heavy training is needed for the next step.
