# Notebook Patch Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `notebook-patch`

## Goal

Patch only the approved notebook structure or cells. Do not run the notebook.
Use `nbformat` for structural edits.

## Hard Boundaries

Allowed files:

```text
{ALLOWED_FILES}
```

Forbidden files:

```text
{FORBIDDEN_FILES}
```

Forbidden actions:

- No `ml_utils/` changes.
- No `tests/` changes.
- No data changes.
- No artifact creation.
- No notebook execution.
- No training.
- No Colab.
- No `git add`.
- No commit.
- No push.
- No TCN/DLinear expansion unless explicitly listed in the goal.
- No full run unless explicitly requested and reviewed.

## Pre-Flight

Run from PowerShell:

```powershell
Set-Location -LiteralPath "{PROJECT_ROOT}"
Get-Location
git branch --show-current
git status --short --untracked-files=all
git status -sb
```

If status is not clean before editing, stop and report.

Read:

```text
AGENTS.md
docs/PHASE_1B_BASELINE_RERUN_PLAN.md
{ALLOWED_FILES}
```

## Exact Task Instructions

1. Inspect the target notebook with `nbformat`.
2. Apply only the requested patch.
3. Preserve default safety flags unless the prompt explicitly changes them.
4. Keep notebook outputs cleared unless the prompt explicitly permits retained
   outputs.
5. Do not duplicate `ml_utils` core logic inside the notebook.
6. Do not run training or call the patched entrypoint.

## Validation

Run:

```text
git status --short --untracked-files=all
git diff --stat
git diff --check
{VALIDATION_COMMANDS}
```

If notebook structural validation is requested, run it without executing cells.

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Include notebook cell counts or structural checks if inspected.

## Explicit Non-Actions

- No git staging.
- No commit.
- No push.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
- No `ml_utils` changes.
- No tests changes.
