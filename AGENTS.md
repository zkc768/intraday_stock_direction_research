# AGENTS.md - intraday_stock_direction_research rebuild route
<!-- AGENTS_VERSION: v6.2-raw-config-screening -->

This is a research project, not a backend project, not a general ML framework,
and not a process-document factory. The active route is a rebuild of the
research surface around one notebook-first configuration-screening lane.

Default reading order for new research work:

1. this file
2. `docs/RESEARCH_WORKFLOW.md`
3. `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`
5. the target notebook

Prior route-control reviews, handoffs, and phase plans are not part of the
active project surface. Do not recreate or rely on them for active notebook
work.

---

## 1. Project Identity

- **Project**: `intraday_stock_direction_research`, a Northeastern thesis
  project on high-frequency stock direction classification.
- **Research question**: Can 5-minute bar data support an honest directional
  classifier that beats simple baselines under chronological validation?
- **Current scope**: five stocks first: `CSCO`, `JPM`, `KO`, `MSFT`, `WMT`.
- **Current route**: rebuild the active research route around validation-only
  configuration screening before model-family screening.
- **Default deliverable**: one clean, readable, validation-only notebook for the
  active research question.
- **Active design surface**: N02–N08 technical-design documents in `docs/` are
  all current; active implementation lanes are `notebooks/02_config_screening_colab.ipynb`
  through `notebooks/06_selective_no_trade_calibration_colab.ipynb` (generated),
  while N07 / N08 exist as design-only and have not yet been generated as notebooks.
- **Active freezes**: `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` plus the
  per-notebook technical-design documents for N05 / N06 / N07 / N08 dated
  2026-06-04 through 2026-06-06; each downstream phase reads the freeze of the
  phase it depends on.
- **Notebook dependency boundary**: canonical research logic lives in the local
  package (`src/intraday_research/` once created). Generated Colab notebooks are
  thin execution/reporting interfaces and may import the package only when the
  install is pinned to an exact git commit and the run manifest records the
  package commit, config hash, notebook hash, inputs, outputs, and
  holdout/test-contact flag. Self-contained notebooks are optional archival
  snapshots, not the default source of truth.

The active route first screens `label_config + window_size + feature_set` from
raw ticker files, then runs model-family screening in the next notebook only
after Stage 0 produces candidates.

---

## 2. Active Research Route

The current route uses these scientific constraints:

- stationarity-safer features
- no-trade-band directional classification
- chronological train/validation validation
- train-only preprocessing
- dummy-baseline comparison
- no holdout/test use for model, feature, threshold, or wording decisions
- pre-registered decision rules before any validation result is used

The active Stage 0 lane is locked by
`docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`:

```text
Stage 0S = runtime/schema smoke, no selection
Stage 0A1 = label-feature screen with LogReg + LightGBM at window_size=10
Stage 0A2 = LightGBM window sensitivity on Stage 0A1 short list
Stage 0B = LogReg + LightGBM + simple GRU + MS-DLinear+TCN second-view
scope = validation_only
```

Stage 0 selects at most two validation-selected configuration candidates:
`mean_candidate` and `lcb_candidate`. Stage 0B is a deep-model second-view
only; it must not rerank or retract Stage 0A candidates.
Model-family screening lives in `notebooks/03_model_family_screening_colab.ipynb`
(designed and generated); validation-only follow-up routing is in
`notebooks/04_controlled_followup_colab.ipynb`; LightGBM train-inner tuning plus
official-validation confirmation is in
`notebooks/05_lightgbm_tuning_colab.ipynb`; selective / no-trade calibration
diagnostics are in `notebooks/06_selective_no_trade_calibration_colab.ipynb`;
validation synthesis and gap audit (N07) and deep sequence exploration / freeze /
readout (N08) are designed in `docs/NOTEBOOK07_...` and `docs/NOTEBOOK08_...`
2026-06-06 but have not yet been generated as notebooks.

If the active 02 notebook is missing, stale, or still points at a prior route,
repair or recreate the notebook before any screening run.

---

## 3. Default Notebook Shape

Use a linear notebook:

```text
research question
frozen protocol and scope
data loading
feature construction
label construction
chronological split
train-only preprocessing
window construction
dummy baselines
validation-only model panel
comparison table and plots
honest interpretation
```

Notebook code may be inline for setup, configuration, artifact display, and
short explanatory calculations. Canonical reusable research logic should live in
`src/intraday_research/` once the package-first migration begins. Reusable
helpers should be extracted when notebook evidence shows the logic is reused,
safety-critical, or testable.

New notebook names must be short, sortable, and snake_case:

```text
<nn>_<topic>_<scope>.ipynb
```

Use `02_config_screening_colab.ipynb` for the active screening lane. Do
not create new routine `PM_###`, handoff, readiness, session-context, or
closeout docs for ordinary progress.

Current notebook route:

```text
01_research_direction_colab.ipynb
02_config_screening_colab.ipynb
03_model_family_screening_colab.ipynb  # planned only, create after Stage 0
```

---

## 4. Hard Research Rules

Violating any item below can invalidate the research conclusion. Stop and ask
before continuing.

### 4.1 Chronology and Leakage

1. Train, validation, and holdout/test boundaries must be chronological. Random
   splits, shuffled validation, and `train_test_split` style time-series splits
   are forbidden.
2. Preprocessing that learns statistics, including scaling, imputation, or
   normalization, must fit on train rows only. For pooled multi-stock runs,
   split each ticker chronologically first, fit the shared scaler on pooled
   train rows only, then transform train and validation.
3. The closed holdout/test interval may be used only as a boundary marker for
   invalidating labels near the validation edge. It must not be transformed,
   windowed, scored, summarized, or used for selection.
4. Label horizons must not cross train/validation or validation/closed-holdout
   boundaries. Samples whose future label horizon reaches into the next split
   must be marked invalid and skipped.
5. Input windows and label horizons must not cross trading-day boundaries.
6. Multi-stock windows must be generated per ticker. No window may span tickers.
7. Features may use only the current completed bar and earlier completed bars.
   Forward-looking rolling means, returns, fills, or future-aware features are
   forbidden.
8. Invalid labels are markers, not missing-data cleanup targets. Do not fill
   them, and do not globally drop them before split-boundary, trading-day, and
   window validity have been enforced.

### 4.2 Evaluation Honesty

1. Model choice, thresholds, feature changes, and hyperparameters use train plus
   validation only.
2. The final holdout/test has already been opened once for this research line
   and is now closed. Reopening it requires a separate pre-registered note with
   the exact model, metric, decision rule, and allowed wording before looking.
3. After holdout/test has been viewed, do not change features, labels,
   thresholds, model architecture, or evaluation wording based on that result.
4. Main metrics are macro F1 and balanced accuracy. Accuracy is auxiliary.
5. Every model comparison must include a stratified dummy baseline on the same
   target rows and `delta_macro_f1_vs_dummy`.
5a. The word "improvement" may be used in validation-only wording ONLY when
    `delta_macro_f1_vs_dummy_lcb_95 >= 0.005` AND `positive_ticker_count >= 4`.
    Below these thresholds, use "weak", "mixed", or "no detected signal".
6. Report pooled results, per-ticker results, sample counts, and result scope.
   Do not cherry-pick one strong ticker, seed, or chart as the conclusion.
7. Every result must be labeled with scope:

```text
exploratory
diagnostic
validation_only
evidence_ready
```

Most notebook work is `exploratory`, `diagnostic`, or `validation_only`.

### 4.3 Failure Behavior

- Do not fabricate metrics, file paths, model behavior, or experiment outcomes.
- Do not silently work around bugs.
- Do not catch and ignore exceptions.
- Do not change data, labels, thresholds, or metrics to make a result look
  better.
- If required data or code is missing, report the exact missing path.
- The cross-notebook validation-budget ledger
  (`notebook07_validation_budget_ledger.csv` is the project-level source of
  truth) is **append-only across N07, N08, and any thesis chapter that cites
  official-validation metrics**. Any downstream read of an official-validation
  metric MUST append a row recording the intent BEFORE the read; pre-existing
  rows MUST NOT be modified, dropped, or reordered. A read without prior
  append is a contract violation detected by the static gate
  (`tests/test_notebook08_artifact_contract.py::validate_08o_ledger_append_precedes_read`).

---

## 5. Active Source Artifacts

Treat these as active research artifacts:

- `docs/RESEARCH_WORKFLOW.md`
- `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`
- `notebooks/01_research_direction_colab.ipynb`
- `notebooks/02_config_screening_colab.ipynb`
- `intraday_research/`
- active tests under `tests/`

The Colab notebook generators intentionally duplicate active feature, label,
split, and window logic so the notebooks can run in Colab without importing a
local helper package. If the locked split, label, feature, or window rules
change, review and update the active notebook generators and generated
notebooks before rerunning them.

Treat these as outside the active project surface unless the user says
otherwise:

- prior route files or freeze notes outside the active `2026-06-04` freeze

---

## 6. Cleanup Policy

Cleanup is allowed when it makes the active research route clearer, but every
cleanup must preserve raw data and the current active notebook route.

- Do not delete raw stock data.
- Do not delete active freeze documents, the active 02 notebook, source code, or
  current result artifacts that are cited by active freezes.
- Direct deletion is acceptable for generated caches, build artifacts, and
  obsolete route artifacts when the user asks to remove old project surfaces.
- Before deleting or archiving, verify the target is not raw data, not the
  active notebook, not an active freeze, and not cited as current source
  evidence.
- Do not rename stale docs/notebooks as active examples.

---

## 7. Notebook Style

- Keep notebooks linear and readable.
- Put the research question, frozen protocol, and result scope near the top.
- Keep configuration values in one early cell.
- Prefer tables and a small number of useful plots over long print logs.
- Use `RUN_* = False` or equivalent guards for heavy training cells.
- Do not read, display, summarize, transform, window, or score holdout/test
  metrics in validation-only notebooks.
- Keep large outputs, raw tensor dumps, and long training logs out of shared
  notebooks.

## 7.1 Colab Creation Workflow

For active Colab notebooks, default to a package-backed, raw-data-first
workflow:

- Import `intraday_research` only after installing the package from an exact git
  commit or from an explicitly versioned archive whose commit is recorded in the
  run manifest. Do not import prior notebooks as the active path.
- Do not mount MyDrive in the default setup cell. Colab mountpoints may already
  contain files and fail before the research code starts.
- Use an explicit Google Drive raw-data manifest with file IDs for the five
  source ticker files, then download those files through the Drive API into a
  local runtime directory such as `/content/stage0_raw_stock_data`.
- Treat mounted Drive folders, shortcuts, `Dow_30_1min`, and copied Colab
  project folders as non-authoritative unless the user explicitly approves a
  cleanup task.
- Write run outputs first to a local runtime directory such as
  `/content/stage0_config_screening_results`. Copying results back to Drive is
  optional and should be a separate explicit cell or user action.
- Notebook generators must contain or generate the active notebook logic
  directly. They must not silently source active logic from stale notebooks,
  prior fixed-lane notebooks.
- Use `nbformat` for notebook creation or structural edits. After generation,
  validate: notebook parses, all code cells AST-parse, outputs are empty,
  execution counts are `None`, and forbidden active-code strings such as
  `from intraday_research` and `baseline_helpers` are absent unless explicitly
  approved. Drive-mount calls are also forbidden in default Colab setup cells.

## 7.2 Package-First Colab Boundary

The long-term repository target is package-first:

```text
src/intraday_research/ = canonical research logic
configs/               = stage parameters and ordered pipeline registry
notebooks/             = generated thin Colab execution/reporting interfaces
tests/                 = package, stage, notebook, and artifact contracts
results/               = reproducible stage outputs
```

- Every research stage that can be run locally or from Colab must expose a
  tested `run_stage(config)` entry point from `src/intraday_research/stages/`.
- Generated Colab notebooks may import `intraday_research`, but only after
  installing the package from an exact git commit. Installing from a floating
  branch such as `main` is not an acceptable research path.
- Every package-backed notebook run must write a run manifest that records:
  repo URL, git commit, package version if present, stage name, config SHA-256,
  notebook SHA-256, input artifacts, output artifacts, validation scope, and
  `holdout_test_contact=false` unless a separately authorized holdout/test
  protocol says otherwise.
- Generators remain responsible for notebook structure. A generated notebook
  should contain configuration, dependency pinning, stage invocation, and
  result display; it should not hide mutable local paths or unstated imports.
- Self-contained notebooks are allowed only as optional archival snapshots or
  review packets. They are not the default canonical logic once the package-first
  migration begins.
- Extracted package code must be stable, pure where possible, explicitly tested,
  and behavior-preserving. Good extraction targets include schema validators,
  artifact contracts, data manifests, feature/label/split/window helpers,
  train-only preprocessing helpers, dummy baselines, metric aggregation, ledger
  checks, and stage orchestration.
- Do not extract exploratory, result-dependent, one-off plotting, thesis prose,
  run-copy switch cells, or post-validation wording decisions into shared stage
  code unless a design note first defines the contract.
- When package behavior used by a generated notebook changes, update the stage
  config, generator, generated notebook, static gate, artifact contract, and
  protocol/design note if the research contract changed. Do not leave package
  behavior and notebook behavior to drift apart.
- Static gates for package-backed notebooks must verify the exact-commit install
  or equivalent commit record, the expected stage entry point, manifest writing,
  empty committed outputs, and the absence of forbidden holdout/test access.
- Package code must not weaken the hard research rules: no random time-series
  splits, no train/validation/holdout leakage, no holdout/test contact, no
  post-validation threshold/model/wording changes, no validation-budget ledger
  read before append, and no silent exception swallowing.
- Package-backed local and Colab pipelines must remain reproducible from
  documented environment files and tests. Hidden installs, unstated dependencies,
  and machine-local paths are not acceptable as the active research path.

---

## 8. PM And Agents

PM+agents may be used to coordinate cleanup, audits, and review. Their job is
to create clarity and catch risk, not to create new process documents by
default.

- Use agents for bounded read-only inventories or disjoint edits.
- Main-thread integration must verify agent output against live files before
  acting on it.
- Do not let agent plans override the hard research rules in this file.
- Do not create new specs, plans, handoffs, or closeout docs unless the user
  explicitly asks.

---

## 9. Environment And Git

- Use `E:\codex_workspace\_envs\py311_shared\python.exe` for local Python work.
- Do not use bare `python` or bare `pytest` for verification.
- Do not install dependencies implicitly.
- Do not run heavy training unless explicitly instructed.
- Do not delete raw data.
- Do not commit, push, or create branches unless the user explicitly asks.
- Start and end scoped work by reporting `git status --short` and
  `git diff --stat` when file edits are involved.

On this Windows workspace, PowerShell file commands can occasionally fail with
`CreateProcessAsUserW failed: 5`. Treat that as a tool-runner issue, not proof
that files are missing. Prefer lightweight fallbacks such as `rg --files`,
`rg -n`, and project-Python subprocess checks.

---

## 10. One-Line Summary

Rebuild the project around one raw-data-first notebook route, protect
chronology and holdout/test boundaries, compare against dummy baselines, and
remove stale process artifacts that obscure the active research path.
