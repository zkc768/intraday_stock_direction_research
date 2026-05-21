# Push-Only Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `push-only`

## Goal

Push the already reviewed local commit or branch. Do not edit files. Do not
create commits.

## Hard Boundaries

Approved branch:

```text
{APPROVED_BRANCH}
```

Approved remote:

```text
{APPROVED_REMOTE}
```

Approved commit or range:

```text
{APPROVED_COMMIT_OR_RANGE}
```

Allowed files to modify:

```text
none
```

Forbidden files to modify:

```text
{FORBIDDEN_FILES}
```

Forbidden actions:

- No file edits.
- No staging.
- No commit.
- No notebook execution.
- No training.
- No artifacts.
- No Colab.
- No branch change unless explicitly approved.

## Pre-Flight

Run from PowerShell:

```powershell
Set-Location -LiteralPath "{PROJECT_ROOT}"
Get-Location
git branch --show-current
git status --short --untracked-files=all
git status -sb
git log --oneline --decorate -5
```

Stop if:

- Working tree is not clean.
- Current branch is not the approved branch.
- The commit to push is not the approved commit.

## Exact Task Instructions

1. Verify the branch and clean status.
2. Verify the local commit range to push.
3. Push only the approved branch.
4. Re-run status and recent log.
5. Do not create new commits.

## Validation

Run:

```text
git status -sb
git log --oneline --decorate -5
{VALIDATION_COMMANDS}
```

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Include pushed branch, pushed commit hash, and final status.

## Explicit Non-Actions

- No files changed.
- No git staging.
- No commit.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
