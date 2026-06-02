# Research Workflow

> Use this workflow to start a new `intraday_stock_direction_research` research notebook. The goal is
> a clean, linear analysis that Ian and future you can read without opening a
> pile of PM documents.
>
> Hard rules live in `AGENTS.md` Section 3. This file describes the normal
> notebook shape.

---

## 1. Notebook Naming

Use short, sortable, snake_case names:

```text
<nn>_<topic>_<scope>.ipynb
```

Examples:

```text
04_ian_research_memo.ipynb
05_lgbm_msdt_validation.ipynb
06_threshold_sensitivity.ipynb
07_selective_prediction.ipynb
```

Rules:

- Use lowercase snake_case.
- Keep the name short.
- Do not use question sentences.
- Do not routinely use `PM_###` names.
- Put dates, detailed research questions, and conclusions inside the notebook,
  not in the filename.

Archived notebooks under `notebooks/archive/` are provenance only. Do not use
them as templates for new research notebooks.

---

## 2. Start a New Experiment

At the top of the notebook, write:

```text
Research question:
Scope: exploratory / diagnostic / validation_only / evidence_ready
Changed from default:
Do not claim:
```

Then proceed linearly:

```text
setup
configuration
data loading
feature construction
label construction
chronological split
split-boundary invalidation
train-only preprocessing
per-ticker window construction
dummy baselines
validation-only models
comparison table and plots
honest interpretation
```

No extra spec, PM gate, handoff, or closeout document is needed for ordinary
research progress. Put the research memo, experiment cards, caveats, and next
step inside the notebook itself.

---

## 3. Notebook Skeleton

### 3.1 Setup

Import libraries, set seeds, and import stable project helpers only when needed.
Do not rewrite model architecture in the notebook.

```python
RUN_TRAINING = False
RUN_VALIDATION = False
```

Heavy cells should be guarded by explicit flags.

### 3.2 Research Question and Config

Use one markdown cell for the question and one code cell for constants:

```text
tickers
feature columns
label horizon
threshold
window size
calendar split
seed
result scope
```

Constants in the notebook are allowed. Silent drift is not allowed: if a default
changes, write down what changed.

### 3.3 Load Data

Load each ticker, parse timestamps, sort by time, and show a small diagnostic
table:

```text
ticker
row count
start timestamp
end timestamp
missing value summary
```

### 3.4 Build Features

Feature construction must be causal. For intraday rolling or return features,
group by trading day so overnight jumps do not enter the wrong calculation.

```python
day = df["timestamp"].dt.date
df["log_return"] = np.log(df["close"]).groupby(day, sort=False).transform(
    lambda s: s - s.shift(1)
)
```

Dropping rows with missing feature values is allowed after feature construction,
but label invalid markers must be handled separately.

### 3.5 Build Labels

Labels may be computed on each full ticker sequence, but this does not make all
labels usable. After chronological splits are defined, any label whose future
horizon reaches into the next split must be invalidated.

Core label idea:

```python
future_cumulative_return = df["close"].shift(-K) / df["close"] - 1.0

threshold = THRESHOLD_BPS / 10_000
label = pd.Series(np.nan, index=df.index)
label[future_cumulative_return > threshold] = 1.0
label[future_cumulative_return < -threshold] = 0.0
```

Trading-day horizon invalidation:

```python
dates = df["timestamp"].dt.date
crosses_day = dates != dates.shift(-K)
label[future_cumulative_return.notna() & crosses_day] = np.nan
```

Split-boundary invalidation:

```python
split_name = assign_split(df["timestamp"])
horizon_split = split_name.shift(-K)
crosses_split = split_name != horizon_split
label[future_cumulative_return.notna() & crosses_split] = np.nan
```

The exact implementation can differ, but the rule cannot: labels whose horizon
crosses train/validation or validation/holdout boundaries are invalid and must
not enter training, tuning, or evaluation.

Do not fill label NaNs. Do not globally `dropna` label rows before split,
trading-day, and window validity are enforced.

### 3.6 Split, Scale, and Window

The safe order is:

```text
1. assign chronological split per ticker
2. invalidate split-boundary labels
3. fit scaler on pooled train rows only
4. transform validation and holdout/test
5. build windows per ticker and per split
```

Window construction must run on an already split single-ticker segment:

```python
for start in range(len(part) - W - K + 1):
    target = start + W - 1
    horizon_end = target + K
    if np.isnan(labels[target]):
        continue
    if dates[start] != dates[target] or dates[target] != dates[horizon_end]:
        continue
    keep_window(part[start:start + W], labels[target])
```

Here `part` means one ticker and one split, already sorted by time. Do not build
windows on the full ticker and split afterward.

### 3.7 Dummy Baselines

Always compute the stratified dummy baseline on train labels and evaluate it on
the same validation target used for the model. Use multiple random states and
report the mean and variation.

The final comparison table must include:

```text
model
ticker or pooled
macro_f1
balanced_accuracy
dummy_macro_f1
delta_macro_f1_vs_dummy
n
scope
```

### 3.8 Models

Model work is validation-only unless the user explicitly authorizes a
pre-registered holdout/test run.

- LightGBM can use a last-step feature view.
- MS-DLinear+TCN can use full windows only after an active adapter/spec task is
  approved; archived references alone are not an active model path.
- Save probabilities if selective/no-trade analysis is relevant.
- Do not tune on holdout/test.

### 3.9 Compare and Interpret

End with:

- one compact results table
- one useful plot, usually delta vs dummy by ticker/model
- one markdown interpretation

The interpretation must say what the result does and does not support.
Weak/mixed results should be described as weak/mixed.

---

## 4. Thirty-Second Self-Check

- [ ] Split is chronological; no random split or shuffled validation.
- [ ] Label horizon does not cross train/validation or validation/holdout.
- [ ] Label horizon and input windows do not cross trading days.
- [ ] Windows are generated per ticker and per split.
- [ ] Scaler/imputer/normalizer fits train only.
- [ ] Model selection and threshold tuning use train plus validation only.
- [ ] Holdout/test metrics are not read in validation-only work.
- [ ] Results include dummy baseline and `delta_macro_f1_vs_dummy`.
- [ ] Claims state scope: exploratory, diagnostic, validation_only, or
  evidence_ready.
- [ ] The notebook can be read top to bottom without relying on hidden state.

---

## 5. When To Touch Other Project Areas

- **Archived helpers**: old helper library, tests, reference excerpts, and CLI
  scripts live under `archive/legacy_model_runner_reference/`. They are historical
  reference only.
- **New helpers**: rebuild a helper only after a notebook proves the logic is
  reused, safety-critical, and testable.
- **Tests**: write tests only for rebuilt reusable leakage, label, window, or
  metric helpers; do not test ordinary notebook cells.
- **Historical docs**: `PM_*`, `PHASE_1B_*`, transfer, manifest, route-control,
  and goal-mode docs are archive material. Do not scan or update them by default.
