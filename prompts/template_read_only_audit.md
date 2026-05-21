# Read-Only Audit Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `read-only`

## Goal

Audit the requested project state and produce a report only. Do not edit files.
Do not create files. Do not run training. Do not execute notebooks.

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
```

If the prompt requires a clean tree and status is not clean, stop and report.

## Exact Task Instructions

1. Read `AGENTS.md`.
2. Read the source-of-truth docs listed in `{ALLOWED_FILES}`.
3. Inspect only the files or diffs required by this audit.
4. Report findings with file paths and line references where possible.
5. Separate observed facts from assumptions and unresolved questions.

## Validation

Run only these validation commands:

```text
{VALIDATION_COMMANDS}
```

If no validation command is required, state that this was a read-only audit.

## Expected Final State

{EXPECTED_STATUS}

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Include explicit counts for any commands run and any failures found.

## Explicit Non-Actions

- No files changed.
- No git staging.
- No commit.
- No push.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
