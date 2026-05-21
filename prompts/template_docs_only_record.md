# Docs-Only Record Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `docs-only`

## Goal

Create or update the specified documentation record only. Preserve existing
project docs and do not rewrite unrelated sections.

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
- No `notebooks/` changes.
- No `tests/` changes.
- No `reference_excerpts/` changes.
- No `requirements.txt` changes.
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

If status is not clean before editing, stop and report.

Read:

```text
AGENTS.md
{ALLOWED_FILES}
```

## Exact Task Instructions

1. Make only the documentation edits described by this prompt.
2. Keep the record evidence-based.
3. Do not claim results that are not present in the source material.
4. Preserve anti-overclaiming language for smoke tests, helper paths, model
   signal, TCN/DLinear validation, artifacts, and full-run readiness.
5. Do not create scripts or executable helpers.

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

Report inspected files, changed files, commands run, validation results, and
unresolved issues.

## Explicit Non-Actions

- No git staging.
- No commit.
- No push.
- No notebook changes.
- No `ml_utils` changes.
- No tests changes.
- No notebook execution.
- No training.
- No Colab.
- No artifacts.
