# Commit-Only Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `commit-only`

## Goal

Create one atomic commit for the already reviewed files. Do not edit files. Do
not push.

## Hard Boundaries

Allowed files to stage:

```text
{ALLOWED_FILES}
```

Forbidden files:

```text
{FORBIDDEN_FILES}
```

Forbidden actions:

- No file edits.
- No `git add -A`.
- No staging outside `{ALLOWED_FILES}`.
- No push.
- No notebook execution.
- No training.
- No artifacts.
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
git diff --check
```

Stop if:

- Any changed file is outside `{ALLOWED_FILES}`.
- `git diff --check` fails.
- The branch is not the expected branch in `{EXPECTED_STATUS}`.
- The commit message is not provided or approved.

## Exact Task Instructions

1. Reconfirm changed files match `{ALLOWED_FILES}` exactly.
2. Stage only the approved files by explicit path.
3. Run `git status --short --untracked-files=all`.
4. Create one commit with the approved message.
5. Run `git status -sb`.
6. Do not push.

Use explicit path staging only, for example:

```powershell
git add -- path/to/file1 path/to/file2
```

Never use:

```powershell
git add -A
```

## Validation

Run:

```text
git status --short --untracked-files=all
git status -sb
git log --oneline --decorate -3
{VALIDATION_COMMANDS}
```

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Include the commit hash and exact files staged.

## Explicit Non-Actions

- No file edits.
- No `git add -A`.
- No push.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
