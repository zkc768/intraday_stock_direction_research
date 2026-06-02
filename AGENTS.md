# AGENTS.md — intraday_stock_direction_research research project
<!-- AGENTS_VERSION: v5.1-research -->

> This is a research project, not a backend project and not a general ML
> framework. The default deliverable is a clean, readable, validation-only
> analysis notebook. The hard rules below protect scientific validity.
>
> Default reading order for new research work:
>
> 1. this file
> 2. `docs/RESEARCH_WORKFLOW.md`
> 3. the target notebook
>
> Historical PM docs, handoffs, route-control reviews, and old phase plans are
> archive/provenance material. Do not treat them as default instructions unless
> the user explicitly asks for historical reconstruction.

---

## 1. Project Identity

- **Project**: `intraday_stock_direction_research`, a Northeastern thesis project on high-frequency
  stock direction classification.
- **Research question**: Can 5-minute bar data support an honest directional
  classifier that beats simple baselines under chronological validation?
- **Current scope**: 5 stocks first, then possible expansion after the baseline
  setup is trustworthy.
- **Working style**: one research question at a time; one readable notebook per
  experiment; no extra process documents unless the user asks.
- **Role of the archived helper library**: historical reference only. Do not import it
  in active notebooks unless the user explicitly asks for historical
  reproduction or a new helper-library rebuild.

---

## 2. Default Experiment Shape

Use a linear notebook:

```text
research question
data loading
feature construction
label construction
chronological split
train-only preprocessing
window construction
dummy baselines
validation-only models
comparison table and plots
honest interpretation
```

Notebook code may be inline when that makes the analysis easier to read. Reused
logic can become a new helper only after the notebook evidence and test boundary
are clear.

New notebook names should be short, sortable, and snake_case:

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

Do not use question sentences or routine `PM_###` names for new research
notebooks. Existing `PM_*` notebooks and anything under `notebooks/archive/`
are historical references, not naming templates or active examples.

---

## 3. Hard Research Rules

Violating any item below can invalidate the research conclusion. Stop and ask
before continuing.

### 3.1 Chronology and Leakage

1. Train, validation, and holdout/test splits must be chronological. Random
   split, shuffled validation, and `train_test_split` style time-series splits
   are forbidden.
2. Preprocessing that learns statistics, including scaling, imputation, or
   normalization, must fit on train data only. For pooled multi-stock runs, split
   each ticker chronologically first, then fit the shared scaler on pooled train
   rows only, then transform validation and holdout/test.
3. Label horizons must not cross train/validation or validation/holdout
   boundaries. Any sample whose future label horizon reaches into the next split
   must be marked invalid and skipped.
4. Input windows and label horizons must not cross trading-day boundaries.
5. Multi-stock windows must be generated per ticker. No window may span tickers.
6. Features may use only the current completed bar and earlier completed bars.
   Forward-looking rolling means, returns, fills, or future-aware features are
   forbidden.
7. Invalid labels are markers, not missing-data cleanup targets. Do not fill
   them, and do not globally drop them before split-boundary, trading-day, and
   window validity have been enforced.

### 3.2 Evaluation Honesty

1. Model choice, thresholds, feature changes, and hyperparameters use train plus
   validation only.
2. The final holdout/test has already been opened once for this research line
   and is now closed. Reopening holdout/test requires a pre-registered note that
   states the exact model, metric, and decision rule before looking.
3. After holdout/test is viewed, do not change features, labels, thresholds,
   model architecture, or evaluation wording based on that result.
4. Main metrics are macro F1 and balanced accuracy. Accuracy is auxiliary.
5. Every model comparison must include stratified dummy baseline performance and
   `delta_macro_f1_vs_dummy`.
6. Report pooled results, per-ticker results, sample counts, and the result
   scope. Do not cherry-pick one strong ticker, seed, or chart as the conclusion.
7. Every result must be labeled with scope:

```text
exploratory
diagnostic
validation_only
evidence_ready
```

Most notebook work is `exploratory`, `diagnostic`, or `validation_only`.

### 3.3 Failure Behavior

- Do not fabricate metrics, file paths, model behavior, or experiment outcomes.
- Do not silently work around bugs.
- Do not catch and ignore exceptions.
- Do not change data, labels, thresholds, or metrics to make a result look
  better.
- If required data or code is missing, report the exact missing path.

---

## 4. Default Baseline, Not a Prison

The current reproducible baseline is:

| Item | Default |
|---|---|
| Feature set | `baseline_v1`: log_return, close_to_open_return, high_low_range, rolling_volatility_20, normalized_volume_20, rsi_14, bollinger_pctb, normalized_macd_hist, time_of_day_sin/cos |
| Label | no-trade band, threshold = 5 bps, horizon k = 12 |
| Window | window_size = 12, decision after current bar close |
| Split | calendar train/validation/holdout intervals already used in prior baseline work |
| Stocks | CSCO, JPM, KO, MSFT, WMT |
| Models | LightGBM and MS-DLinear+TCN |

These defaults are not a ban on research. Other features, windows, thresholds,
models, or stock scopes may be explored if the notebook states what changed and
the hard rules in Section 3 remain intact.

---

## 5. Notebook Style

- Keep notebooks linear and readable.
- Put a short research question and result scope near the top.
- Keep configuration values in one early cell.
- Prefer tables and a small number of useful plots over long print logs.
- Use `RUN_* = False` or equivalent guards for heavy training cells.
- Do not read or display holdout/test metrics in validation-only notebooks.
- Keep large outputs, raw tensor dumps, and long training logs out of shared
  notebooks.

---

## 6. What Not To Do By Default

These habits made the project feel like a backend/PM document machine. Do not
resume them as default research workflow.

- Do not create new `PM_NNN_*`, handoff, readiness, session-context, or
  closeout documents for ordinary research progress.
- Do not expand or restore archived CLI runners for new exploration.
- Do not test-first notebook exploration. Tests belong only to rebuilt stable
  helpers, not to ordinary memo notebooks.
- Do not build backend scaffolding for one-off analysis. Start in the notebook;
  extract to the library only after reuse is clear.
- Do not add plugin, hook, registry, callback, abstract-base-class, or
  "future extensibility" machinery.
- Do not move or rename historical docs/notebooks unless the user explicitly
  asks for a cleanup pass.

---

## 7. Environment and Git

- Use `E:\codex_workspace\_envs\py311_shared\python.exe` for local Python work.
- Do not use bare `python` or bare `pytest` for verification.
- Do not delete raw data.
- Do not commit, push, or create branches unless the user explicitly asks.
- Start and end scoped work by reporting `git status --short` / `git diff --stat`
  when file edits are involved.

---

## 8. One-Line Summary

This is a small research project. Default to one clear notebook, protect
chronology and validation honesty, compare against dummy baselines, label the
scope of every result, and do not recreate the PM document machine.
