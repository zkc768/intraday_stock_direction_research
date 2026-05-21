# Current Next Task Prompt Draft: P1B.20b

This is a prompt draft only. Do not execute it during PromptOps.1.

---

你现在执行 P1B.20b — guarded entrypoint narrow smoke.

## Current State

The project is on `master` after the Notebook 03 guarded entrypoint patch.
Notebook 03 has a `run_model_comparison()` entrypoint available. Prior smoke
work validated helper-path wiring only; helper-path success does not prove full
entrypoint readiness.

This task is a narrow guarded entrypoint smoke. It is not a full run. It is not
a helper-path smoke. It must call `run_model_comparison()`.

## Goal

Run a tiny real-data Notebook 03 entrypoint smoke that calls
`run_model_comparison()` with one candidate, one model, one ticker, one seed,
one epoch, capped raw rows, and no artifact writes.

The goal is to verify that the guarded full entrypoint can complete the narrow
scope. Do not claim model signal or full experiment validity from this smoke.

## Hard Boundaries

Task type: `Colab execution`

Allowed files to inspect or execute:

```text
notebooks/03_model_comparison.ipynb
AGENTS.md
docs/PHASE_1B_BASELINE_RERUN_PLAN.md
docs/PROJECT_OVERVIEW.md
```

Forbidden files to modify:

```text
ml_utils/
notebooks/
tests/
docs/
prompts/
reference_excerpts/
requirements.txt
data/
checkpoints/
artifacts/
```

Forbidden actions:

- Do not edit repo files.
- Do not execute a full run.
- Do not run helper-path smoke code instead of `run_model_comparison()`.
- Do not use TCN.
- Do not use DLinear.
- Do not write artifacts.
- Do not overwrite existing outputs.
- Do not commit.
- Do not push.
- Do not continue into broader candidates, models, tickers, or seeds.
- Do not claim LSTM beats dummy from this smoke.

## Pre-Flight

Run from PowerShell locally before any Colab execution:

```powershell
Set-Location -LiteralPath "E:\codex_workspace\projects\hf_stock_clf"
Get-Location
git branch --show-current
git status --short --untracked-files=all
git status -sb
git log --oneline --decorate -5
```

Expected:

- Branch is `master`.
- Working tree is clean.
- Latest commit is the approved Notebook 03 guarded entrypoint state.

Stop if the local working tree is dirty.

In Colab, verify:

- Current repo commit hash.
- `notebooks/03_model_comparison.ipynb` is from the approved commit.
- The call path uses `run_model_comparison()`.
- Execution flags exactly match this prompt.

Stop if any guard differs.

## Exact Task Instructions

Run `run_model_comparison()` with exactly:

```python
FULL_RUN = True
RUN_TRAINING = True
WRITE_ARTIFACTS = False
ALLOW_OVERWRITE = False
SELECTED_CANDIDATES = ["A"]
SELECTED_MODELS = ["lstm"]
SELECTED_TICKERS = ["CSCO"]
SELECTED_SEEDS = [42]
MAX_RAW_ROWS_PER_TICKER = 20000
MAX_EPOCHS = 1
```

Requirements:

1. Confirm this is not a full A-D run.
2. Confirm this is not helper-path smoke.
3. Confirm `run_model_comparison()` is called.
4. Confirm only Candidate A is selected.
5. Confirm only LSTM is selected.
6. Confirm only CSCO is selected.
7. Confirm only seed 42 is selected.
8. Confirm max raw rows per ticker is 20000.
9. Confirm max epochs is 1.
10. Confirm artifacts are disabled.
11. Confirm overwrite is disabled.
12. Stop after this narrow smoke, even if it succeeds.

## Validation

Collect and report:

- Exact execution configuration.
- `run_model_comparison()` call evidence.
- Result table or summarized result rows.
- Data and label diagnostics.
- Window counts.
- Baseline metrics and model metrics.
- Confusion matrix with explicit `labels=[0, 1]`.
- Manifest or manifest-like run metadata.
- Final git status.
- Artifact absence evidence from the expected output location.

Artifact absence evidence must be independently checked. Do not claim no
artifacts solely because `WRITE_ARTIFACTS=False` was set.

## Expected Final State

- Narrow entrypoint smoke completed or stopped with a clear blocker.
- No repo files changed.
- No artifacts written.
- No commit.
- No push.
- No TCN/DLinear validation claimed.
- No full-run validity claimed.

## Final Report Format

Use:

```text
A. Overall verdict
B. Execution configuration
C. Entry point evidence
D. Results
E. Diagnostics
F. Manifest
G. Artifact absence evidence
H. Final git status
I. Remaining warnings
J. Explicit non-actions
```

Verdict choices:

- `PASS — guarded entrypoint narrow smoke completed`
- `PASS WITH WARNINGS — smoke completed but needs review`
- `FAIL — blocker or guard failure`

## Explicit Non-Actions

- No full run.
- No helper-path smoke substitution.
- No TCN.
- No DLinear.
- No artifact writes.
- No overwrite.
- No git staging.
- No commit.
- No push.
- No repo file edits.
- No follow-on execution.
