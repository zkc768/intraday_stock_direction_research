# Staging Candidates And Closeout

Date: 2026-06-02
Status: staging plan only; no files staged.
Scope: exact-path closeout for the current notebook-first rebuild worktree.

## Boundary

This document does not authorize `git add .`, commit, push, notebook execution,
training, raw-data edits, checkpoint/artifact edits, or holdout/test access.

Use exact paths only. Re-run the listed validation commands immediately before
staging or committing.

## Current Worktree Shape

Tracked modified files:

```text
AGENTS.md
docs/BASELINE_REFERENCE.md
notebooks/04_ian_research_memo.ipynb
```

Untracked candidate files:

```text
docs/rebuild_specs/2026-06-02-helper-extraction-readiness.md
docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md
docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md
docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md
docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md
docs/rebuild_specs/2026-06-02-p1-helper-test-plan.md
docs/rebuild_specs/2026-06-02-staging-candidates-and-closeout.md
docs/rebuild_specs/2026-06-02-validation-only-preregistration-template.md
intraday_research/__init__.py
intraday_research/baseline_v1.py
tests/test_baseline_v1_helpers.py
tests/test_notebook_static_gate.py
```

No raw data, sources, checkpoints, artifacts, archive files, or notebook archive
files should be staged for this closeout.

Validation commands may create ignored cache files such as `__pycache__/` and
`.pytest_cache/`. They are covered by `.gitignore` and are not staging
candidates.

## Recommended Commit Strategy

Recommendation: two exact-path commits, plus one optional meta closeout document
that is not staged by default.

Reason: helper/notebook runtime code and smoke/preregistration provenance should
stay separate. At the same time, do not stage the design docs before the helper
and notebook files they reference exist in the committed tree.

The optional closeout file:

```text
docs/rebuild_specs/2026-06-02-staging-candidates-and-closeout.md
```

is a workspace planning aid. It should not be staged by default because the
project rules discourage reviving closeout/PM document machinery. If the user
explicitly wants to keep this provenance, stage it as a separate docs-only commit.

## Commit 1 Candidate

Active helper, notebook integration, safety tests, and the minimum boundary docs
that explain P1 helper extraction:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- AGENTS.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/BASELINE_REFERENCE.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- notebooks/04_ian_research_memo.ipynb
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- intraday_research/__init__.py
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- intraday_research/baseline_v1.py
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- tests/test_baseline_v1_helpers.py
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- tests/test_notebook_static_gate.py
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-helper-extraction-readiness.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-p1-helper-test-plan.md
```

Suggested commit message:

```text
research: add notebook baseline helper tests
```

## Commit 2 Candidate

Docs-only smoke quarantine, preregistration template, and rebuild design
provenance:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-notebook-first-rebuild-design.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-notebook-first-rebuild-plan.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md
git -C E:\codex_workspace\projects\intraday_stock_direction_research add -- docs/rebuild_specs/2026-06-02-validation-only-preregistration-template.md
```

Do not stage with `git add .`.

Suggested commit message:

```text
docs: record rebuild smoke boundaries
```

## Pre-Staging Validation

Run before staging:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_baseline_v1_helpers.py tests\test_notebook_static_gate.py -q
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile intraday_research\baseline_v1.py tests\test_baseline_v1_helpers.py tests\test_notebook_static_gate.py
E:\codex_workspace\_envs\py311_shared\python.exe -c "import ast, nbformat; nb=nbformat.read(r'notebooks\04_ian_research_memo.ipynb', as_version=4); code=[c for c in nb.cells if c.cell_type=='code']; [ast.parse(c.source) for c in code]; print('cells', len(nb.cells)); print('code_cells', len(code)); print('outputs', sum(len(getattr(c,'outputs',[])) for c in code)); print('exec_counts', [c.get('execution_count') for c in code])"
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --check
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short --branch
```

Expected:

- helper and notebook static gate tests pass;
- py_compile passes;
- notebook has 30 cells, 13 code cells, 0 outputs, all execution counts `None`;
- `git diff --check` reports no whitespace errors, aside from possible LF/CRLF
  warnings on existing text files;
- status contains only the exact candidate files above.

## Post-Staging Validation

Run after exact-path staging and before commit:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --cached --name-only
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --cached --check
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short --branch
```

Expected cached paths should match the candidate list exactly.

## Explicit Non-Claims

This closeout does not claim:

- full validation performance;
- model selection;
- threshold selection;
- feature selection;
- evidence-ready results;
- reopened holdout/test access;
- active LightGBM or MS-DLinear+TCN adapter approval.

Smoke results remain diagnostic only. Future full validation still requires a
separate completed preregistration.
