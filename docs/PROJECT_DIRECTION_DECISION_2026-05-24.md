# Project Direction Decision — 2026-05-24

## Decision

Continue the project, but change the goal.

The project should no longer be framed as "build a novel strongest stock model"
or "stitch together a large external repository." The best target is:

```text
Local-first, leakage-safe evaluation harness for high-frequency stock direction
classification, with external models integrated only as small adapters.
```

In other words, `hf_stock_clf` should own the experimental protocol:

- local 5-stock data loading;
- chronological split;
- train-only scaler;
- Phase 1B diagnostic no-trade-band subset;
- no cross-day input windows or label horizons;
- dummy baselines;
- macro F1, balanced accuracy, confusion matrix;
- coverage and retained-subset disclosure;
- multi-seed / multi-ticker comparison tables.

Canonical Phase 1 labels remain the AGENTS-defined full binary task:
`label = 1 if future_avg_r > 0 else 0`, with no no-trade-band filtering, unless
a separate spec change is explicitly approved.

External repositories may contribute model ideas, but not data handling,
splitting, labels, scaling, metrics, or evaluation claims.

## Why This Is The Best Goal

### Evidence From Local Code

- `ml_utils` already contains usable components:
  - `dataset.py`
  - `metrics.py`
  - `trainer.py`
  - `checkpoint.py`
  - `models/lstm_classifier.py`
  - `models/tcn_classifier.py`
  - `models/dlinear_classifier.py`
  - `profiling.py`
- Full local validation passed:

```text
142 passed, 1 warning
```

- Local data is available under `data/`:

```text
CSCO.csv, JPM.csv, KO.csv, MSFT.csv, WMT.csv
```

Each file has the expected columns:

```text
timestamp, open, high, low, close, volume
```

### Evidence From Local Data Smoke

A local Candidate A pipeline check succeeded without Colab:

```text
window_size = 12
label_horizon_k = 12
threshold_bps = 5
features = open/high/low/close/volume
```

Full 5-ticker local window counts:

```text
train = 213396
val   = 11903
test  = 19190
```

A small local CSCO model-axis smoke also completed:

```text
scope: CSCO, first 20000 raw rows, 1 epoch, LSTM/TCN/DLinear
train windows = 3146
val windows   = 1082
test windows  = 711
```

This proves that local execution can replace Colab for the next controlled
experiments. The smoke results are not model-quality evidence.

### Evidence From Research Direction PDF

The user-provided PDF, `Artificial Neural Networks, NLP and Stock Market
Performances`, is a broad research vision:

- high-frequency stock data;
- social-media/news text;
- LSTM, Transformer, GPT/LLM models;
- sentiment, topic modeling, portfolio selection, and reinforcement learning.

That vision is much broader than the current codebase. It should be treated as
long-term motivation, not as the next implementation scope.

The near-term publishable contribution is stronger if the project first answers:

```text
Under strict leakage-safe evaluation, how much signal remains in high-frequency
stock direction classification?
```

## What To Stop

Stop these until the local harness produces a stable baseline table:

- custom MS-DLinear + residual TCN model development;
- attention, CNN-LSTM, FinGPT, GPT, or sentiment fusion;
- reinforcement learning or trading strategy backtests;
- broad notebook expansion;
- copying external repositories into the project;
- treating Colab as required for the next step.

## What To Keep

Keep these as the project core:

- `ml_utils.dataset` as the source of truth for labels, splits, scaler, and
  windows;
- `ml_utils.metrics` as the source of truth for dummy baselines and evaluation;
- `ml_utils.trainer` for small local model runs;
- LSTM, TCN, and DLinear as baseline models;
- local `data/` CSVs as the first experimental dataset;
- `stock_ml_knowledge_base` as literature and source triage support.

## External Code Integration Rule

External models must enter only through a narrow adapter contract:

```python
model = CandidateAdapter(seq_len, input_size, num_classes=2, **params)
logits = model(x)  # x shape: (batch, seq_len, input_size)
```

Adapter rules:

- no CSV reads;
- no label generation;
- no train/validation/test splitting;
- no scaler fitting;
- no metric computation;
- no softmax/sigmoid in `forward`;
- no hidden access to test data;
- output raw logits only.

Preferred first external adapter candidate:

```text
PatchTST-style classifier adapter
```

But PatchTST should only be added after the local baseline matrix is stable.

## Agent Operating Model

Use a four-role workflow.

### 1. Design / Control Agent

Owns:

- project goal;
- experiment scope;
- stop rules;
- result interpretation;
- final decision reports.

Should not directly broaden the model set when results are weak.

### 2. Local Runner Agent

Owns:

- running small local checks;
- producing result CSVs/tables;
- reporting exact commands and configs.

Rules:

- may run local data/profile/model smokes;
- no heavy training without explicit scope;
- no notebook-only logic;
- no production-code changes unless assigned.

### 3. Review / Skeptic Agent

Owns:

- leakage review;
- baseline fairness;
- seed/ticker robustness;
- stale-document detection;
- overclaim prevention.

Must explicitly ask:

```text
Would this result survive if dummy, coverage, ticker, and seed are all reported?
```

### 4. Literature / Adapter Agent

Owns:

- mapping knowledge-base papers to candidate ideas;
- identifying external code worth adapting;
- license and dependency cautions;
- keeping NLP/LLM ideas out of the near-term path unless justified.

## Two-Week Minimal Plan

### Week 1: Local Harness Freeze

Goal: produce a trusted local baseline runner plan without changing model scope.

Tasks:

1. Reconcile current state:
   - record uncommitted `ml_utils/dataset.py` diff;
   - list untracked notebooks;
   - mark stale docs (`README`, `NEXT_SESSION_BRIEF`, `PROJECT_OVERVIEW`) as
     stale rather than relying on them.
2. Create or select one local runner entry point for:
   - Candidate A only;
   - 5 tickers;
   - seeds `[42, 43, 44]`;
   - models `[lstm, tcn, dlinear]`;
   - no artifacts unless an output directory is explicitly passed.
3. Add no new model ideas.
4. Run a tiny smoke first:
   - one ticker;
   - one seed;
   - one epoch;
   - capped rows.

### Week 2: Baseline Matrix

Goal: decide whether sequence models beat dummy under local strict evaluation.

Run:

```text
Candidate A
tickers = CSCO, JPM, KO, MSFT, WMT
seeds = [42, 43, 44]
models = [lstm, tcn, dlinear]
```

Report:

- macro F1;
- balanced accuracy;
- dummy stratified mean/std;
- delta macro F1 vs dummy;
- confusion matrix;
- retained coverage;
- per-ticker and pooled summaries.

## Decision Rules

### Continue With Existing Baselines

Continue if at least one model shows:

```text
mean delta_macro_f1_vs_dummy >= +0.01
```

and the gain is not isolated to one ticker or one seed.

### Add PatchTST Adapter

Add PatchTST only if:

- the local runner is stable;
- result table generation is automatic;
- LSTM/TCN/DLinear results are available as comparison;
- reviewer accepts no leakage or reporting gaps.

### Pivot To Evaluation Paper

Pivot if all sequence models are weak or seed-sensitive.

The paper angle becomes:

```text
Strict evaluation reveals that high-frequency stock direction prediction is a
weak-signal problem; model complexity is less important than leakage control,
coverage disclosure, and robust baselines.
```

### Stop The Project

Stop only if:

- local data cannot be made reproducible;
- the harness cannot produce stable metrics;
- the project cannot define a result table that survives review.

Current evidence does not support stopping.

## Current Recommendation

Do not abandon `hf_stock_clf`.

Freeze it as the local evaluation harness, run one controlled baseline matrix,
then decide whether the next contribution is:

1. a better adapter model, likely PatchTST; or
2. an evaluation/weak-signal paper built around strict protocol and negative
   evidence.

The project is not useless. It is too broad and needs a smaller target.

## Local Runner Update — 2026-05-24

A local runner was added after user approval:

```text
scripts/phase1b_local/local_baseline_matrix.py
```

Runner contract:

- Candidate A only: `window_size=12`, `label_horizon_k=12`,
  `threshold_bps=5`;
- Phase 1B diagnostic no-trade-band subset;
- pooled training with per-ticker and pooled test evaluation;
- scaler fit on pooled train only;
- outputs `manifest.csv`, `results.csv`, `metadata.json`, and checkpoints;
- default run mode is small smoke, not full training;
- full matrix requires explicit `--full-run`.

Validated commands:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\phase1b_local\local_baseline_matrix.py
```

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --manifest-only --tickers CSCO JPM KO MSFT WMT --feature-set technical_v1 --output-dir checkpoints\phase1b_local_manifest_technical
```

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --smoke --feature-set technical_v1 --models lstm tcn dlinear --tickers CSCO JPM KO MSFT WMT --max-rows-per-ticker 5000 --max-epochs 1 --batch-size 256 --output-dir checkpoints\phase1b_local_baseline_smoke_technical
```

`technical_v1` full local manifest completed without training:

```text
pooled train windows = 213265
pooled test windows  = 19182
retained coverage    = roughly 9.3% to 19.7% by ticker
```

The five-ticker `technical_v1` smoke completed:

```text
models = lstm, tcn, dlinear
seed = 42
max_rows_per_ticker = 5000
max_epochs = 1
result rows = 18
```

One small-sample warning appeared:

```text
TCN / KO / 95 test windows / macro_f1 = 0.736643
```

Follow-up leakage audit on the same smoke data found:

```text
train windows checked = 2631, bad = 0
val windows checked   = 518, bad = 0
test windows checked  = 415, bad = 0
train/test timestamp overlap = 0
```

Interpretation: the warning is most likely small-sample volatility, not proof of
signal. It should still block any strong claim until the full multi-seed matrix
is run.

Recommended next command, when heavy local CPU training is explicitly approved:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --full-run --models lstm tcn dlinear --seeds 42 43 44 --feature-set technical_v1 --max-epochs 3 --batch-size 512 --output-dir checkpoints\phase1b_local_baseline_full
```

## Full-Run Result — 2026-05-24

The local `technical_v1` full-run completed:

```text
run_dir = checkpoints/phase1b_local_baseline_full/phase1b_local_full_20260524_202110
rows = 54
models = lstm, tcn, dlinear
seeds = 42, 43, 44
scopes = pooled + CSCO/JPM/KO/MSFT/WMT
max_epochs = 3
batch_size = 512
```

Additional summary artifacts were written:

```text
summary_pooled_by_model.csv
summary_by_model_ticker.csv
```

Pooled multi-seed summary:

```text
model    macro_f1_mean  delta_mean  delta_std  balanced_accuracy_mean
dlinear  0.483400       -0.014646   0.045739   0.518245
lstm     0.499939        0.001893   0.028273   0.522107
tcn      0.494158       -0.003888   0.033868   0.526212
```

Per-ticker pattern:

```text
CSCO: LSTM and TCN positive delta
KO:   LSTM, TCN, and DLinear positive delta
JPM:  mostly negative delta
MSFT: negative delta
WMT:  negative delta
```

No full-run row triggered the suspicious `macro_f1 > 0.70` stop rule.

Interpretation:

- The pooled result does not meet the current continuation threshold of
  `mean delta_macro_f1_vs_dummy >= +0.01`.
- Positive evidence is ticker-local, mainly CSCO and KO, not broad enough to
  justify adding PatchTST yet.
- The next research move should be an evaluation/weak-signal analysis or a
  protocol diagnostic, not a larger model.
- This run is diagnostic, not a canonical Phase 1 baseline, because it uses the
  Phase 1B diagnostic no-trade-band subset and was run from a dirty local
  working tree.

## Project Control Update — Agent Workflow

The project is now being managed as:

```text
Manager / Control: decide scope, stop rules, and interpretation
Runner agent: run bounded experiments only
Reviewer agent: challenge leakage, protocol, and claims
Code-Control agent: inspect dirty tree, write scopes, and commit risk
```

Current gate status:

```text
dataset/test gate: ready
runner gate: usable for diagnostics
full-run evidence: diagnostic only, not canonical
PatchTST/new-model gate: blocked
```

What changed after review:

- `ml_utils/dataset.py` now reports invalid label, duplicate timestamp, and
  out-of-order timestamp schema errors with row/index details.
- `tests/test_window_boundaries.py` now covers multi-ticker trim grouping,
  cross-day label horizons, cross-day input windows, invalid labels, duplicate
  timestamps, and out-of-order timestamps.
- `scripts/phase1b_local/local_baseline_matrix.py` now reports per-ticker dummy
  comparisons in addition to pooled dummy comparisons.
- `scripts/phase1b_local/local_baseline_matrix.py` has a
  `--shuffle-train-labels` diagnostic flag.

Validation:

```text
dataset/window target tests: 44 passed
full non-integration tests: 148 passed, 1 existing checkpoint warning
shuffled-label smoke: completed, no stable positive delta
```

Next manager-level move:

1. Keep results framed as Phase 1B diagnostics.
2. Do not add PatchTST or another large model yet.
3. Prepare a clean rerun plan only after the current runner/docs/dataset/test
   changes are reviewed as a coherent patch set.
4. If continuing experiments, prioritize protocol diagnostics:
   - per-ticker baseline comparison on the full five-ticker run;
   - threshold sensitivity for 0/5/10 bps;
   - a notebook-02-compatible shuffled-label sanity presentation.

Threshold sensitivity status:

```text
0 bps: broad coverage, closest diagnostic analog to the canonical full binary task
5 bps: high-magnitude diagnostic subset, usable only with coverage disclosure
10 bps: very low coverage, exploratory only
```

Manager gate:

```text
Do not train a threshold grid yet. Rerun table-of-record diagnostics only after
runner row-level provenance and docs language are reviewed as a coherent patch.
```

## Experiment Update — Table-Of-Record Diagnostics

The runner provenance patch was reviewed and two table-of-record diagnostic
reruns completed:

```text
5bps high-magnitude diagnostic:
  checkpoints/phase1b_local_table_record_5bps/phase1b_local_full_20260524_215040

0bps strict-sign diagnostic analog:
  checkpoints/phase1b_local_table_record_0bps/phase1b_local_full_20260524_220040
```

The 5bps run remains the Phase 1B high-magnitude no-trade-band
table-of-record. It does not meet the continuation threshold:

```text
lstm     delta_macro_f1_vs_dummy = +0.001893
tcn      delta_macro_f1_vs_dummy = -0.003888
dlinear  delta_macro_f1_vs_dummy = -0.014646
```

The 0bps run has broad coverage but is still not the canonical Phase 1 full
binary task. It is a strict-sign diagnostic analog: exact-zero `future_avg_r`
rows are dropped as neutral/NaN, while canonical Phase 1 assigns exact-zero rows
to class 0 `non_up`. In this data the exact-zero count is small (`13` rows), but
the label semantics are different and must be disclosed.

0bps pooled diagnostic result:

```text
tcn      delta_macro_f1_vs_dummy = +0.004890
lstm     delta_macro_f1_vs_dummy = -0.017044
dlinear  delta_macro_f1_vs_dummy = -0.019760
```

The small TCN pooled gain at 0bps does not survive per-ticker dummy comparison:
all five TCN per-ticker deltas are negative.

Gate status at that point:

```text
dataset/test gate: ready
runner gate: usable for diagnostics
5bps Phase 1B diagnostic table: complete
0bps strict-sign diagnostic analog: complete
canonical Phase 1 full-binary table: not yet run at that point
PatchTST/new-model gate: blocked
```

Next decision from that stage:

```text
Do not add PatchTST, attention, NLP, or a larger copied repository yet.
If the project continues experimentally, the next table should be a canonical
Phase 1 full-binary run using make_binary_labels_from_future_avg_return, not
another no-trade-band threshold sweep.
```

## Experiment Update — Canonical Phase 1 Full-Binary

The canonical Phase 1 full-binary runner was added to the local baseline entry
point and validated:

```text
runner entry = scripts/phase1b_local/local_baseline_matrix.py
label_mode = legacy_binary
label_semantics = canonical_phase1_full_binary
label_formula = label = 1 if future_avg_r > 0 else 0
zero_return_policy = class_0_non_up
no_trade_band_enabled = false
```

Validation:

```text
runner py_compile: passed
runner semantic tests + label/config/window tests: 74 passed
full non-integration tests: 154 passed, 1 existing checkpoint warning
5-ticker canonical smoke: completed
```

The canonical full-run completed:

```text
run_dir = checkpoints/phase1_canonical_binary_full/phase1b_local_legacy_binary_full_20260524_230605
feature_set_id = technical_v1
tickers = CSCO, JPM, KO, MSFT, WMT
models = lstm, tcn, dlinear
seeds = 42, 43, 44
max_epochs = 3
batch_size = 512
rows = 54
suspicious rows = 0
pooled retained_pct = 0.847315
pooled test windows = 235333
pooled zero_return rows = 13
```

Pooled canonical result:

```text
tcn      delta_macro_f1_vs_dummy = -0.002318
lstm     delta_macro_f1_vs_dummy = -0.015578
dlinear  delta_macro_f1_vs_dummy = -0.023092
```

All model/ticker mean deltas versus the per-ticker dummy baseline are negative.
The best pooled model is TCN, but it is still slightly below the stratified
dummy baseline.

Updated gate status:

```text
dataset/test gate: ready
runner gate: supports diagnostics and canonical full-binary
5bps Phase 1B diagnostic table: complete
0bps strict-sign diagnostic analog: complete
canonical Phase 1 full-binary table: complete
PatchTST/new-model gate: blocked
```

Current project decision:

```text
Continue the project as an evaluation harness and weak-signal study.
Do not add PatchTST, attention, NLP, sentiment, RL, or a copied external
repository based on the current evidence.
Next work should prioritize reporting, protocol analysis, and simpler
non-sequence baselines before larger neural architectures.
```
