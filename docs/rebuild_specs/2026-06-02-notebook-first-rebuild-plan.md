# Notebook-First Rebuild Implementation Plan

> For agentic workers: implement this plan task-by-task. Keep the active
> project notebook-first. Do not execute notebooks, train models, mutate raw
> data, mutate checkpoints, open holdout/test, restore archived runner workflows,
> or use `git add .`.

**Goal:** Create a guarded, readable, validation-only notebook skeleton for
`notebooks/04_ian_research_memo.ipynb`, with PM+agents decisions recorded in
project-local rebuild specs.

**Architecture:** The first implementation pass changes only the active
notebook structure. Any reusable helper module waits until notebook logic proves
the helper boundary. Archived model/runner assets are references for later
migration audits, not active imports.

**Tech Stack:** `nbformat`, Python executable
`E:\codex_workspace\_envs\py311_shared\python.exe`, Markdown project docs,
static notebook validation.

**Current status, 2026-06-02:** Tasks 1-3 are complete. Task 4 Steps 1-2 are
complete after both smoke reports were explicitly labeled non-decision
diagnostics and a validation-only preregistration template was added. Task 4
Step 3 remains pending and requires a separate future approval before any full
validation command.

---

## File Structure

Create or modify:

- `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md`
  - Records PM+agents design decisions and work-package boundaries.
- `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md`
  - Defines the task sequence for implementation.
- `notebooks/04_ian_research_memo.ipynb`
  - Future implementation target for the 17-section skeleton.

Do not modify:

- `data/`
- `sources/`
- `checkpoints/`
- `artifacts/`
- `archive/legacy_model_runner_reference/`
- `notebooks/archive/`

## Task 1: Record PM+Agents Design Artifacts

**Files:**

- Create: `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md`
- Create: `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md`

- [x] Step 1: Verify local rules and current state.

Run:

```powershell
Get-Content -LiteralPath AGENTS.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short --branch
```

Expected:

- `AGENTS.md` confirms notebook-first research rules.
- Git status shows current tracked edits before new work.

- [x] Step 2: Ensure the rebuild spec directory exists.

Run:

```powershell
Test-Path -LiteralPath docs\rebuild_specs
```

Expected:

- `True` after directory creation.

- [x] Step 3: Add the design and plan markdown files.

Implementation:

- Write the PM+agents design to
  `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md`.
- Write this implementation plan to
  `docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md`.

- [x] Step 4: Validate docs are discoverable.

Run:

```powershell
rg --files docs\rebuild_specs
```

Expected:

```text
docs\rebuild_specs\2026-06-02-notebook-first-rebuild-design.md
docs\rebuild_specs\2026-06-02-notebook-first-rebuild-plan.md
```

## Task 2: Rewrite Active Notebook Skeleton With nbformat

**Files:**

- Modify: `notebooks/04_ian_research_memo.ipynb`

- [x] Step 1: Inspect the current notebook without execution.

Run:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -c "import nbformat; p=r'notebooks\04_ian_research_memo.ipynb'; nb=nbformat.read(p, as_version=4); print(len(nb.cells)); print(sum(len(getattr(c,'outputs',[])) for c in nb.cells if c.cell_type=='code')); print([c.get('execution_count') for c in nb.cells if c.cell_type=='code'])"
```

Expected:

- The current notebook is read successfully.
- No code cell outputs or execution counts are introduced by inspection.

- [x] Step 2: Replace notebook cells with the 17-section skeleton.

Use `nbformat` with project Python. Preserve notebook metadata when possible.
All code cells must have `execution_count = None` and `outputs = []`.

Required cell headings:

```text
# Notebook 04 - Ian Research Memo
## Setup And Run Guards
## Baseline Configuration
## Data Contract And Paths
## Load And Coverage Diagnostics
## Causal Feature Construction
## Label Construction And Invalid Markers
## Chronological Split And Split-Boundary Invalidation
## Train-Only Preprocessing
## Per-Ticker Per-Split Window Construction
## Target And Class Balance Diagnostics
## Stratified Dummy Baselines
## Model Availability And Validation Plan
## Validation-Only Model Cells
## Comparison Table
## Validation Plots
## Ian-Facing Interpretation
```

Required guard defaults in code cells:

```python
RUN_DATA_LOAD = False
RUN_FEATURE_BUILD = False
RUN_MODEL_VALIDATION = False
RUN_TRAINING = False
```

Required metadata in code cells:

```python
RESULT_SCOPE = "validation_only"
FEATURE_SET_ID = "baseline_v1"
LABEL_POLICY = "no_trade_band"
THRESHOLD_SOURCE = "fixed_pre_registered_5bps"
THRESHOLD_BPS = 5.0
LABEL_HORIZON_K = 12
WINDOW_SIZE = 12
DECISION_TIME_POLICY = "post_bar_close_completed_bar"
SCALER_ID = "standard_pooled_train_only_v1"
SEED = 42
TICKERS = ["CSCO", "JPM", "KO", "MSFT", "WMT"]
```

- [x] Step 3: Keep model cells inert.

The validation/model cell should contain a visible guard pattern similar to:

```python
if not RUN_MODEL_VALIDATION:
    print("Model validation is disabled. Set RUN_MODEL_VALIDATION=True only in an approved validation-only task.")
else:
    raise NotImplementedError("Active model adapter migration is required before validation.")
```

Expected:

- No training can start by default.
- No archived runner is imported.

- [x] Step 4: Static validate notebook structure.

Run:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -c "import nbformat; p=r'notebooks\04_ian_research_memo.ipynb'; nb=nbformat.read(p, as_version=4); print('cells', len(nb.cells)); print('outputs', sum(len(getattr(c,'outputs',[])) for c in nb.cells if c.cell_type=='code')); print('exec_counts', [c.get('execution_count') for c in nb.cells if c.cell_type=='code']); print('titles', [c.source.splitlines()[0] for c in nb.cells if c.cell_type=='markdown' and c.source.splitlines()])"
```

Expected:

- `outputs 0`
- Every execution count is `None`
- Required titles are present in order.

- [x] Step 5: Static scan for forbidden active references.

Run:

```powershell
rg -n "archive|legacy_model_runner|ml_utils|train_test_split|holdout|test_metrics|RUN_TRAINING = True|RUN_MODEL_VALIDATION = True" notebooks\04_ian_research_memo.ipynb
```

Expected:

- Any hits for holdout/test appear only in markdown guard language.
- No imports or active paths from `archive`, `legacy_model_runner`, or
  `ml_utils`.
- No default `RUN_TRAINING = True` or `RUN_MODEL_VALIDATION = True`.

## Task 3: Review Diff And Prepare Exact Commit Candidates

**Files:**

- Review: all changed files

- [x] Step 1: Review unstaged diff summary.

Run:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --stat
```

Expected:

- Only expected files are changed.

- [x] Step 2: Check whitespace.

Run:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --check
```

Expected:

- No whitespace errors. Git may warn that LF will be replaced by CRLF.

- [x] Step 3: Report exact staged candidates without staging.

Candidate files:

```text
AGENTS.md
docs/BASELINE_REFERENCE.md
docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md
docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md
docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md
docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md
docs/rebuild_specs/2026-06-02-validation-only-preregistration-template.md
docs/rebuild_specs/2026-06-02-helper-extraction-readiness.md
docs/rebuild_specs/2026-06-02-p1-helper-test-plan.md
notebooks/04_ian_research_memo.ipynb
```

P1 helper status:

- Tests-first helper extraction has started for the minimal safety-critical
  slice.
- Synthetic tests in `tests/test_baseline_v1_helpers.py` were created first and
  initially failed for missing implementation.
- `intraday_research/baseline_v1.py` now contains only the tested label, split,
  scaler, window, and stratified dummy helpers.
- P1 post-review fixes added single-ticker guards, pooled-ticker rejection
  tests, a stronger dummy train-only proof, and an AST import guard. A later
  critical-review repair pass fixed the cumulative-return label contract and
  feature-construction defects. Current P1 helper/static validation is
  `36 passed`.
- `notebooks/04_ian_research_memo.ipynb` now imports
  `intraday_research.baseline_v1` for the tested safety-critical helpers instead
  of carrying duplicate label, split, scaler, window, and dummy implementations.
- `tests/test_notebook_static_gate.py` now locks notebook integration invariants:
  no saved outputs, default `RUN_* = False`, no forbidden imports, no duplicate
  safety-critical helper definitions, and no raw-feature fallback implementation.
  Current combined P1 validation is `36 passed`.
- Do not widen P1 into model adapters, archived runners, raw-data reads, or
  checkpoint/artifact readers.

Additional P1 candidate files:

```text
intraday_research/__init__.py
intraday_research/baseline_v1.py
tests/test_baseline_v1_helpers.py
tests/test_notebook_static_gate.py
```

Do not stage or commit unless explicitly asked.

## Task 4: Quarantine Smoke Results Before Future Validation

**Files:**

- Review: `docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md`
- Review: `docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md`
- Create:
  `docs/rebuild_specs/2026-06-02-validation-only-preregistration-template.md`

- [x] Step 1: Confirm smoke reports are labeled non-decision.

Run:

```powershell
rg -n "Non-Decision Policy|validation_only_smoke_not_evidence|not evidence-ready|must not be used" docs\rebuild_specs
```

Expected:

- Both smoke reports contain non-decision language.
- The reports do not authorize feature, threshold, model, or claim changes.

- [x] Step 2: Add a future full-validation preregistration template.

Before any future full validation command, fill out a new pre-registration note
from `docs/rebuild_specs/2026-06-02-validation-only-preregistration-template.md`
that states:

```text
tickers
split scope
model family
feature set
label policy
threshold policy
scaler policy
output path
metrics
claim language
stop rules
```

Expected:

- The template itself is not an approval to run.
- Smoke metrics are not copied into the pre-registration as decision rules.
- The closed holdout/test remains closed unless separately authorized.

- [ ] Step 3: Only then run the approved validation command.

Expected:

- The run uses the pre-registered scope.
- The output path is new and non-overwriting.
- The final report separates pipeline diagnostics from evidence claims.

## Self-Review Checklist

- [x] The plan keeps the project notebook-first.
- [x] The plan does not restore old PM/route-control workflow.
- [x] The plan does not execute notebook cells.
- [x] The plan does not train models.
- [x] The plan does not read or write holdout/test metrics.
- [x] The plan treats LightGBM and MS-DLinear+TCN as migration-audit-required,
  not active by default.
- [x] The plan preserves exact-path staging discipline.
- [x] The plan quarantines smoke results as diagnostics, not hard rules.
