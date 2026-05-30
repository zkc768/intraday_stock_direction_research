# Automated Agent Workflow - 2026-05-24

## Purpose

This document defines the operating loop for `hf_stock_clf` after the canonical
Phase 1 full-binary result. The goal is to make the project process automatic
without expanding the scientific scope too early.

The workflow automates:

- preflight checks;
- bounded local runs;
- result summarization;
- skeptical review;
- dirty-tree control;
- handoff updates.

It does not automatically add new models or broaden the research claim.

## Fixed Project Goal

```text
Build and maintain a local-first, leakage-safe evaluation harness for
high-frequency stock direction classification.
```

Current scientific status:

```text
The canonical Phase 1 full-binary table is complete.
LSTM, TCN, and DLinear did not beat dummy_stratified.
The next work should be reporting, protocol analysis, and simpler baselines
before larger neural architectures.
```

## Agent Roles

### Manager Agent

Owns:

- project objective;
- experiment scope;
- stop rules;
- interpretation;
- next-step selection;
- final user-facing report.

Manager rule:

```text
Do not approve new model work unless the reviewer accepts the protocol and the
existing baselines show robust positive evidence.
```

### Runner Agent

Owns:

- local command execution;
- smoke runs;
- full runs only when explicitly approved;
- exact command capture;
- output directory and result-file checks.

Runner rule:

```text
Run smoke before full-run. Report commands, row counts, suspicious rows, and
artifact paths. Do not interpret beyond the run evidence.
```

### Reviewer Agent

Owns:

- leakage skepticism;
- baseline fairness;
- label semantics;
- coverage disclosure;
- seed/ticker robustness;
- overclaim prevention.

Reviewer question:

```text
Would this result survive if dummy, coverage, ticker, seed, and label semantics
are all reported?
```

### Code-Control Agent

Owns:

- `git status --short`;
- `git diff --stat`;
- changed-file scope;
- generated artifacts;
- pycache/cache cleanup recommendations;
- handoff accuracy.

Code-Control rule:

```text
Never revert user/project dirty changes. Record them and work around them.
```

## Standard Loop

Use this sequence for each project step:

```text
1. Preflight
2. Plan the smallest bounded action
3. Assign runner/reviewer/code-control roles
4. Execute smoke or read-only audit
5. Summarize artifacts
6. Reviewer gate
7. Code-control gate
8. Update docs or handoff
9. Decide next action
```

## Preflight Checklist

Always read:

```text
AGENTS.md
NEXT_WINDOW_HANDOFF.md
docs/PROJECT_DIRECTION_DECISION_2026-05-24.md
docs/PHASE_1B_FULL_RUN_ANALYSIS_2026-05-24.md
```

Always use:

```text
E:\codex_workspace\_envs\py311_shared\python.exe
```

Always check:

```text
git -C E:\codex_workspace\projects\hf_stock_clf status --short
git -C E:\codex_workspace\projects\hf_stock_clf diff --stat
```

## Run Types

### Read-Only Audit

Use for:

- checking files;
- summarizing existing runs;
- reviewing docs;
- generating a patch-scope plan.

No code changes.

### Smoke Run

Use before any heavier run:

```text
one or few tickers
one seed unless multi-seed behavior is the target
one epoch or capped rows
explicit output directory under checkpoints/
```

### Full Run

Only run when explicitly approved or already covered by the active user request.

Full-run output must include:

```text
metadata.json
manifest.csv
results.csv
summary_pooled_by_model.csv
summary_by_model_ticker.csv
```

## Stop Rules

Stop model expansion when:

- pooled delta vs `dummy_stratified` is below `+0.01`;
- positive evidence is isolated to one ticker or one seed;
- label semantics are diagnostic rather than canonical;
- retained coverage is too narrow for a broad claim;
- reviewer finds leakage or baseline ambiguity.

Current status:

```text
PatchTST/new-model gate: blocked
```

## Automation Backlog

Build these before another model family:

1. Result summarizer for existing run directories.
2. Report table generator from `results.csv` and `manifest.csv`.
3. Run registry that records:
   - run directory;
   - label mode;
   - feature set;
   - tickers;
   - seeds;
   - model list;
   - primary conclusion.
4. Reviewer checklist script or markdown template.
5. Final handoff updater.

## Near-Term Recommended Task

Next implementation target:

```text
Create a small local report/summarizer tool under scripts/phase1b_local/ that
loads one or more completed run directories and writes clean markdown/CSV
summary tables for canonical and diagnostic runs.
```

This should use existing output files rather than re-running training.

Status on 2026-05-25:

```text
completed
```

Implemented:

```text
scripts/phase1b_local/summarize_runs.py
tests/test_phase1b_report_summarizer.py
```

Validated output:

```text
checkpoints/phase1b_local_reports/table_records_20260525
```

The summarizer writes:

```text
run_summary.csv
pooled_by_model.csv
by_model_ticker.csv
coverage_by_ticker.csv
report.md
```

It keeps canonical and diagnostic protocol fields explicit in every summary
table and does not modify existing run directories.

Patch audit follow-up:

```text
Window-end label alignment is now the official Dataset contract.
The local runner default label mode is now legacy_binary/canonical.
No-trade-band diagnostics require explicit --label-mode no_trade_band.
Runner baseline rows now include balanced accuracy and confusion-matrix fields.
trim_labels_at_split_boundary now fails fast on out-of-order per-ticker
timestamps instead of sorting before validation.
```

Next workflow target:

```text
Use consolidated report tables for protocol analysis and simpler baseline
planning. Keep PatchTST/new-model gate blocked until a protocol review finds
robust positive evidence above the +0.01 dummy-delta threshold.
```

## New-Window Contract

The next Codex window should act as project manager first, not as model builder.

Default order:

```text
1. Read the handoff and docs.
2. Inspect dirty tree.
3. Review current patch scope.
4. Automate reporting.
5. Run tests.
6. Update handoff.
```
