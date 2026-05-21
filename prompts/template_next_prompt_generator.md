# Next Prompt Generator Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `prompt-generator`

## Goal

Generate the next Codex prompt draft for this project. The output is a prompt
draft only. Do not execute the generated prompt.

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

- No execution of the generated prompt.
- No `ml_utils/` changes unless this generator task explicitly allows editing a
  prompt draft under `prompts/`.
- No notebook execution.
- No training.
- No artifacts.
- No `git add`.
- No commit.
- No push.
- No Colab.
- No scope expansion.

## Pre-Flight

Run from PowerShell:

```powershell
Set-Location -LiteralPath "{PROJECT_ROOT}"
Get-Location
git branch --show-current
git status --short --untracked-files=all
git status -sb
```

If status is not clean and this prompt assumes a clean tree, stop and report.

Read:

```text
AGENTS.md
docs/PROMPT_OPS.md
docs/PHASE_1B_BASELINE_RERUN_PLAN.md
docs/PROJECT_OVERVIEW.md
{ALLOWED_FILES}
```

## Exact Task Instructions

1. Identify the exact next task and task type.
2. Choose the smallest safe scope.
3. Fill all required PromptOps fields.
4. Include allowed files, forbidden files, forbidden actions, pre-flight,
   validation, stop conditions, expected final state, final report format, and
   explicit non-actions.
5. Preserve project-specific defaults from `docs/PROMPT_OPS.md`.
6. Include anti-overclaiming rules relevant to the task.
7. Save only the approved prompt draft path if file editing is allowed.
8. Do not execute the prompt.

## Validation

Run:

```text
git status --short --untracked-files=all
git diff --stat
git diff --check
{VALIDATION_COMMANDS}
```

Expected changed paths must be limited to:

```text
{ALLOWED_FILES}
```

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Include the generated prompt path and state clearly that it was not executed.

## Explicit Non-Actions

- No generated prompt execution.
- No git staging.
- No commit.
- No push.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
