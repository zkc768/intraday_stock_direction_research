# Review-Only Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `review-only`

## Goal

Review the specified docs, diffs, notebook structure, or run report. Produce
findings only. Do not edit files.

## Hard Boundaries

Allowed files to inspect:

```text
{ALLOWED_FILES}
```

Forbidden files to modify:

```text
{FORBIDDEN_FILES}
```

Forbidden actions:

- No file edits.
- No new files or directories.
- No notebook execution.
- No training.
- No artifacts.
- No `git add`.
- No commit.
- No push.
- No Colab.

## Pre-Flight

Run from PowerShell:

```powershell
Set-Location -LiteralPath "{PROJECT_ROOT}"
Get-Location
git branch --show-current
git status --short --untracked-files=all
git status -sb
git diff --stat
```

Read:

```text
AGENTS.md
{ALLOWED_FILES}
```

## Stop Conditions

Stop and report if:

- Required files are missing.
- Requested validation exceeds read-only scope.
- The prompt asks for fixes instead of review.
- The working tree state makes the requested review ambiguous.
- Any instruction conflicts with `AGENTS.md` or `docs/PROMPT_OPS.md`.

## Exact Task Instructions

1. Review only the requested scope.
2. Lead with findings ordered by severity.
3. Include exact file paths and line references when possible.
4. Flag overclaiming, missing validation, dirty scope, and forbidden actions.
5. Do not propose execution as already approved.

## Validation

Run only:

```text
{VALIDATION_COMMANDS}
```

If validation is not requested, state that the review was inspection-only.

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Recommended default:

```text
A. Overall verdict
B. Findings
C. Files inspected
D. Commands run
E. Validation results
F. Remaining warnings
G. Recommended next step
H. Explicit non-actions
```

## Explicit Non-Actions

- No files changed.
- No git staging.
- No commit.
- No push.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
