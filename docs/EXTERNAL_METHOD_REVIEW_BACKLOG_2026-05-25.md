# External Method Review Backlog - 2026-05-25

## Purpose

This document records the current paper and external-code review backlog for
`hf_stock_clf`.

The project goal remains:

```text
Local-first, leakage-safe evaluation harness for high-frequency stock direction
classification, with external models integrated only as small adapters.
```

This document is not approval to implement any method. It is a review queue and
drift-control record.

## Non-Negotiable Protocol

External papers, repositories, and local reference excerpts must not change:

- canonical binary label:
  `label = 1 if future_avg_r > 0 else 0`;
- chronological split;
- train-only scaler fitting;
- no-cross-day input windows and horizons;
- window-end label alignment;
- project-owned macro F1, balanced accuracy, confusion matrix, and dummy
  baselines;
- coverage, ticker, seed, and label-semantics disclosure.

External code must not be copied or imported into `ml_utils`. Local
implementations must be first-principles and must pass an explicit gate before
becoming code.

## Agent Review Inputs

Read-only agents reviewed three complementary source pools:

- classic and high-impact literature;
- recent 2024-2026 papers and active repositories;
- local knowledge-base references and code excerpts.

All agents converged on the same control decision:

```text
PatchTST/new-model gate remains blocked.
The next work should be protocol analysis, reporting, and simple strong
baselines before larger neural architectures.
```

## Priority Deep-Reading Queue

### P0 - Immediate Deep Research

| Candidate | Source | Why It Matters | Allowed Near-Term Use | Code Status |
|---|---|---|---|---|
| Gu, Kelly, Xiu - Empirical Asset Pricing via Machine Learning | https://academic.oup.com/rfs/article/33/5/2223/5758276 | Finance ML discipline, out-of-sample caution, robust baseline reporting | Reporting frame and negative-result discipline | No code |
| Great Time Series Classification Bake Off | https://link.springer.com/article/10.1007/s10618-016-0483-9 | Classic TSC benchmark discipline and baseline comparison | Evaluation protocol review | No code |
| ROCKET | https://arxiv.org/abs/1910.13051 | Strong simple TSC classification baseline | Future baseline spec candidate | No code until gate |
| MiniROCKET / aeon MiniRocketClassifier | https://arxiv.org/abs/2012.08791 and https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.classification.convolution_based.MiniRocketClassifier.html | Very fast TSC baseline, close fit to binary classification | Isolated baseline spec candidate | No dependency/code until gate |
| Selective classification / risk-coverage framing | Local knowledge base notes | Turns no-trade/top-confidence ideas into abstention/risk-coverage analysis | Reporting and diagnostic design | No model code |
| Reliable Stock Prediction 2026 | Local knowledge base notes | Experiment audit checklist for leakage, scaling, baselines, and reporting | Methodology checklist | No model code |

### P1 - Strong Watchlist, Research Only

| Candidate | Source | Why It Matters | Risk |
|---|---|---|---|
| FM4MTSC | https://github.com/mlgig/FM4MTSC | Evaluates foundation models for multivariate TSC and helps resist "bigger is better" drift | Aggregates multiple methods and licenses; read conclusions only |
| Mantis / MantisV2 | https://arxiv.org/abs/2502.15637 and https://arxiv.org/abs/2602.17868 | Recent time-series classification foundation model work | External weights/API/trainer must not replace project harness |
| ELinear 2025 | Local knowledge base notes | Stock-specific linear-family baseline idea | Must audit label, split, scaler, and trading assumptions first |
| DLinear / LTSF-Linear | https://ojs.aaai.org/index.php/AAAI/article/view/26317 | Supports simple-model skepticism | Forecasting task must not redefine classification protocol |
| InceptionTime | https://arxiv.org/abs/1909.04939 | Classic deep TSC CNN baseline | New-model gate closed; future only |
| TSMixer | https://openreview.net/forum?id=wbpxTuXgm0 | Simpler MLP alternative to attention | New neural family blocked for now |
| ModernTCN | https://openreview.net/forum?id=vpJMJerXHU | Closest recent architectural cousin to current TCN | Watchlist only while current TCN does not beat dummy |

### P2 - Long-Term Watchlist Only

| Candidate | Source | Reason To Watch | Blocker |
|---|---|---|---|
| PatchTST | https://openreview.net/forum?id=Jbdc0vTOcol | Patch/channel-independent sequence modeling | Transformer gate blocked |
| iTransformer | https://openreview.net/forum?id=JePfAI8fah | Variates-as-tokens idea for future richer features | Attention/model expansion blocked |
| TimeMixer | https://openreview.net/forum?id=7oLshfEIC2 | Recent multiscale MLP idea | Not justified before simple baselines |
| MOMENT | https://arxiv.org/abs/2402.03885 | Foundation model with classification support | Heavy dependency/weight path; not a harness replacement |
| Moirai / Uni2TS | https://openreview.net/forum?id=Yd8eHMY1wz | High-quality universal forecasting model | Forecasting-first, not current binary classification |
| Chronos | https://arxiv.org/abs/2403.07815 | Active forecasting foundation model family | Forecasting-first and external pretraining concerns |
| StockMixer | https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/ | Stock-specific mixer idea | Forecasting/cross-stock assumptions and license need review |
| MASTER | https://arxiv.org/abs/2312.15235 | Market-guided stock Transformer | Transformer/Qlib/protocol drift risk |
| FinTSB | https://arxiv.org/abs/2502.18834 | Financial time-series benchmark and reporting ideas | Forecasting/backtesting drift risk; code license/deps need review |
| DeepLOB | https://arxiv.org/abs/1808.03668 | Classic high-frequency LOB model | Data modality mismatch and repo license risk |

## Local Knowledge-Base Boundaries

The local knowledge base is useful as a source triage pool, not as code source.

Allowed uses:

- source discovery;
- deep-reading notes;
- experiment audit checklists;
- architecture vocabulary;
- future spec drafting.

Blocked uses:

- importing from `reference_excerpts`;
- copying code blocks from local reference files;
- letting external loaders, labels, splits, scalers, runners, metrics, or
  checkpoints own project behavior;
- treating notebook or repo results as `hf_stock_clf` results.

Known high-risk local references:

- `Time-Series-Library`: benchmark/data-provider/runner assumptions are not
  project protocol;
- `mlfinlab`: concept-only because dependency and license risk are high;
- `DeepLOB`: limit-order-book domain does not match current 5-minute OHLCV
  data;
- `PatchTST`, `MASTER`, `HIST`, `FinGPT`, `StockNet`, `PriceSeer`: future
  literature/code review only;
- `pytorch-template`: trainer/checkpoint field inspiration only, not split
  logic;
- `yutsuro_tisc.py`: organization reference only, not evaluation logic.

## External Method Code Gate

Before any reviewed method becomes code, every item below must be answered
`PASS`.

1. Project preflight docs read and current new-model gate stated.
2. Method source recorded: paper, repo, access date, commit/version, license.
3. Scientific reason tied to current harness results.
4. Scope fit confirmed: binary high-frequency stock direction classification.
5. `hf_stock_clf` keeps ownership of labels, splits, scaler, metrics, runner,
   and result claims.
6. Adapter contract confirmed:
   `x` has shape `(batch, seq_len, features)` and output is raw logits.
7. No hidden device moves, global state, file reads, test-data access, or
   feature engineering.
8. Dependencies are already allowed or separately approved and pinned.
9. API signatures checked against current installed versions.
10. License/copy rule confirmed: no copied code and no imports from references.
11. Leakage review passed: no future features, cross-ticker windows, or
    cross-day windows/horizons.
12. Evaluation review passed: dummy stratified, dummy prior, always-up/down,
    macro F1, balanced accuracy, confusion matrix.
13. Robustness condition passed: not one ticker, one seed, or diagnostic mode.
14. Spec exists with exact target file, test files, line budget, and acceptance
    criteria.
15. User approves tests first, with lazy imports if target code does not exist.
16. Smoke plan exists before any full run.
17. Code-control confirms dirty tree and allowed files.
18. If any item fails, no code is written and a gate-failure memo is returned.

## Innovation Discovery Protocol

The paper may use multiple papers and codebases to discover innovation points,
but the project must avoid "method stitching." The correct research action is:

```text
paper/code source -> design hypothesis -> project-owned experiment -> bounded
claim
```

The incorrect action is:

```text
paper/code source -> copied component -> changed protocol -> strong claim
```

Innovation is allowed when it creates a testable claim under the existing
`hf_stock_clf` harness. It is not enough that two papers have not previously
been combined.

Each proposed innovation must record:

- source papers or repositories;
- the problem it addresses in this project;
- the exact project artifact needed before implementation;
- the experiment that could falsify it;
- the claim that would be allowed if the experiment succeeds;
- the claim that remains blocked even if the experiment succeeds.

## Paper-Ready Innovation Candidates

These candidates are research hypotheses, not implementation approvals.

### 1. Leakage-Safe Weak-Signal Evaluation Harness

Composition:

- Gu/Kelly/Xiu-style financial ML discipline;
- Great TSC Bake Off baseline discipline;
- project-owned chronological split, train-only scaler, dummy baselines,
  coverage disclosure, and per-ticker reporting.

Possible contribution sentence:

```text
We show that under a leakage-controlled local evaluation harness, standard
sequence models for high-frequency stock direction classification fail to
robustly outperform stratified dummy baselines, motivating protocol-first
evaluation over architecture-first model selection.
```

Needed evidence:

- canonical full-binary table;
- diagnostic no-trade-band tables;
- dummy, coverage, ticker, seed, and label-semantics disclosure;
- shuffled-label sanity check;
- clean-worktree rerun or clear dirty-tree disclosure.

Blocked claim:

```text
We discovered a profitable trading strategy.
```

### 2. Risk-Coverage Direction Classification

Composition:

- selective classification / abstention;
- no-trade-band diagnostics;
- project-owned coverage and retained-subset reporting.

Possible contribution sentence:

```text
We recast no-trade-band stock direction prediction as a risk-coverage problem,
separating abstention/coverage effects from genuine predictive improvement.
```

Needed evidence:

- coverage-vs-macro-F1 curves across thresholds or confidence bins;
- dummy baselines evaluated at the same retained coverage;
- per-ticker coverage and class-balance disclosure.

Blocked claim:

```text
Filtering uncertain samples proves market predictability.
```

### 3. Simple TSC Baseline Stress Test

Composition:

- ROCKET/MiniROCKET simple time-series classification baselines;
- current LSTM/TCN/DLinear baseline matrix;
- project-owned metrics and split protocol.

Possible contribution sentence:

```text
We test whether strong non-neural or shallow time-series classification
baselines explain apparent sequence-model gains in high-frequency stock
direction prediction.
```

Needed evidence:

- approved MiniROCKET or ROCKET baseline spec;
- dependency/license gate;
- train-only fit/transform plan;
- same dummy, ticker, seed, and coverage reporting as neural baselines.

Blocked claim:

```text
MiniROCKET is better for finance because external benchmarks say so.
```

### 4. Linear-Family Ablation For Stock Direction

Composition:

- DLinear / LTSF-Linear simplicity result;
- ELinear or stock-specific linear-family ideas from the knowledge base;
- current DLinear classifier baseline.

Possible contribution sentence:

```text
We isolate which parts of linear time-series models matter for stock direction
classification under strict chronological evaluation.
```

Candidate ablations:

- Linear vs NLinear-style normalization;
- DLinear decomposition on/off;
- per-ticker vs pooled linear heads;
- window length and horizon sensitivity.

Needed evidence:

- first-principles local implementation or existing project code only;
- tests proving unchanged label, split, scaler, and window-end semantics;
- comparison to dummy and existing DLinear under the same runner.

Blocked claim:

```text
Forecasting-paper gains transfer directly to stock direction classification.
```

### 5. Label, Window, Horizon, And Coverage Stability Map

Composition:

- project-owned canonical label and no-trade-band diagnostic modes;
- threshold sensitivity manifests;
- window-end label alignment;
- weak-signal reporting discipline.

Possible contribution sentence:

```text
We map how intraday stock direction results change across label threshold,
window length, horizon, coverage, ticker, and seed, separating stable signal
from protocol-induced artifacts.
```

Needed evidence:

- manifest-first sweeps before training;
- coverage and class-balance tables for each candidate setting;
- training only for pre-approved settings;
- no cherry-picking: canonical full-binary remains the main task.

Blocked claim:

```text
The best-performing threshold is the true stock prediction task.
```

### 6. Ticker-Local Weak-Signal Heterogeneity

Composition:

- per-ticker dummy comparisons;
- seed stability reporting;
- confusion matrices and class-balance disclosure;
- post-hoc CSCO/KO diagnostic caution.

Possible contribution sentence:

```text
We show that apparent model gains in intraday stock direction classification
are ticker-local and seed-sensitive, which changes the interpretation from
broad market predictability to hypothesis-generating heterogeneity.
```

Needed evidence:

- per-ticker `delta_macro_f1_vs_ticker_dummy`;
- seed-level and ticker-level summaries;
- explicit marking of post-hoc ticker selection;
- pooled result shown alongside every local positive result.

Blocked claim:

```text
Positive CSCO or KO diagnostics demonstrate a generalizable alpha signal.
```

### 7. Foundation-Model Skepticism For Financial TSC

Composition:

- FM4MTSC;
- Mantis/MantisV2;
- MOMENT/Chronos/Moirai watchlist;
- current negative baseline evidence.

Possible contribution sentence:

```text
We use recent time-series foundation-model evidence as a stress-test lens,
arguing that foundation models should not be introduced into financial
direction classification until simple leakage-safe baselines establish a stable
signal.
```

Needed evidence:

- source-review cards only;
- no code integration;
- explicit mismatch table: task, data modality, labels, metrics, and
  evaluation protocol.

Blocked claim:

```text
Foundation models fail on this project.
```

That claim would require an approved adapter, local experiments, and all gate
checks.

### 8. Adapter Firewall As A Reproducibility Contribution

Composition:

- external-code review findings;
- project adapter contract;
- strict ownership of labels, splits, scalers, metrics, and results.

Possible contribution sentence:

```text
We introduce an adapter firewall that allows external architectures to be tested
without letting external repositories alter the financial evaluation protocol.
```

Needed evidence:

- documented External Method Code Gate;
- one future approved adapter or dry-run spec;
- tests showing external model outputs raw logits only and cannot access data
  construction.

Blocked claim:

```text
The adapter firewall itself improves predictive performance.
```

Its value is reproducibility and contamination control, not accuracy.

## Innovation Ranking

Current strongest paper directions:

1. Leakage-safe weak-signal evaluation harness.
2. Risk-coverage direction classification.
3. Label/window/horizon stability map.
4. Ticker-local weak-signal heterogeneity.
5. Adapter firewall for external method reproducibility.
6. Simple TSC baseline stress test.
7. Linear-family ablation for stock direction.
8. Foundation-model skepticism as related-work framing.

The first five can support a paper even if results remain negative. The sixth
and seventh may become method contributions only after approved experiments.
The eighth is useful framing, but should stay in related work unless experiments
are later approved.

## Recommended Next Steps

1. Write deep-reading cards for P0 sources, starting with Gu/Kelly/Xiu,
   ROCKET/MiniROCKET, and Great TSC Bake Off.
2. Convert `Reliable Stock Prediction 2026` and selective-classification notes
   into a project audit checklist for reports.
3. Draft innovation cards for the six candidates above, each with falsifiable
   experiment requirements and blocked claims.
4. Draft a MiniROCKET isolated-baseline spec, but do not implement it until the
   dependency, fit/transform, and metrics ownership issues are resolved.
5. Keep MantisV2 and FM4MTSC as source-review tasks only; use them to challenge
   foundation-model enthusiasm, not to broaden the current implementation.
6. Keep PatchTST, iTransformer, ModernTCN, TimeMixer, and StockMixer on the
   watchlist until a simpler baseline or protocol diagnostic justifies
   reopening the new-model gate.

## Current Decision

Continue as an evaluation harness and weak-signal study.

Do not add PatchTST, attention, NLP, sentiment, RL, backtesting, or copied
external repositories based on current evidence.
