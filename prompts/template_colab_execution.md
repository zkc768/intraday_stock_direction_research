# Colab Execution Prompt Template

你现在执行 {TASK_ID} — {TASK_NAME}.

## Current State

{CURRENT_STATE}

Project root:

```text
{PROJECT_ROOT}
```

Task type: `Colab execution`

## Goal

Run only the approved Colab or notebook execution scope and report evidence.
Do not modify repo files unless this prompt explicitly permits a notebook
patch before execution.

## Hard Boundaries

Allowed files or notebooks:

```text
{ALLOWED_FILES}
```

Forbidden files:

```text
{FORBIDDEN_FILES}
```

Forbidden actions:

- No repo code changes unless explicitly listed.
- No TCN/DLinear unless explicitly listed.
- No full run unless explicitly requested and reviewed.
- No artifacts unless explicitly enabled.
- No git staging.
- No `git add`.
- No `git add -A`.
- No commit.
- No push.
- No hidden scope expansion.

## Pre-Flight

Run or verify:

```text
{VALIDATION_COMMANDS}
```

Required checks:

- Confirm the exact repo commit in Colab.
- Confirm the intended notebook or entrypoint.
- Confirm execution flags.
- Confirm artifact policy.
- Confirm selected candidates, models, tickers, seeds, epoch caps, and row caps.

Stop if any guard differs from this prompt.

## Exact Task Instructions

1. Execute only the approved entrypoint and configuration.
2. Capture result tables, diagnostics, manifest, and runtime guard output.
3. Capture artifact evidence according to the artifact policy.
4. Do not reinterpret smoke output as model signal.
5. Do not proceed to broader scope after success.

## Validation

Required evidence:

```text
{EXPECTED_STATUS}
```

Include exact stdout snippets or summarized tables sufficient for review.

## Expected Final State

The approved execution scope is complete or stopped with a clear blocker. No
unapproved repo changes, commits, pushes, or artifacts exist.

## Final Report Format

Use these sections:

```text
{FINAL_REPORT_SECTIONS}
```

Recommended default:

```text
A. Overall verdict
B. Execution configuration
C. Results
D. Diagnostics
E. Manifest
F. Artifact evidence
G. Final git status
H. Remaining warnings
I. Explicit non-actions
```

## Explicit Non-Actions

- No git staging.
- No `git add`.
- No `git add -A`.
- No commit.
- No push.
- No unapproved files changed.
- No unapproved models.
- No unapproved candidates.
- No unapproved seeds.
- No unapproved artifacts.
- No follow-on full run.
