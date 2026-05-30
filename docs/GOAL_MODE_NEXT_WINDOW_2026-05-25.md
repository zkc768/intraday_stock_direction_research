# Goal Mode Next Window Plan - 2026-05-25

## Purpose

This file is a new-window execution plan for `hf_stock_clf`.

The next window should run in **Goal Mode**:

```text
Turn the current protocol-safe experiment package into a paper-ready claim and
innovation package, without opening the new-model gate or running heavy
training.
```

The new window should not treat this as a generic brainstorming session. It
should produce concrete files that make the thesis/paper easier to write.

## Current Project Facts

The project is currently best framed as:

```text
Protocol-safe weak-signal evaluation for intraday stock direction
classification.
```

Locked empirical facts:

```text
canonical full-binary best pooled model:
  model = tcn
  delta_macro_f1_vs_dummy = -0.002317962437516166

0bps no-trade-band diagnostic:
  best model = tcn
  delta_macro_f1_vs_dummy = +0.004889691925515733

5bps no-trade-band diagnostic:
  best model = lstm
  delta_macro_f1_vs_dummy = +0.0018927356051448342

model expansion gate:
  blocked_delta_lt_0.01 for all three regimes
```

The important interpretation:

```text
The evidence supports protocol analysis, weak-signal diagnostics, and paper
framing. It does not support adding PatchTST, attention, NLP, sentiment, RL,
backtesting, or copied external repositories.
```

## Existing Evidence Package

The new window should read these before making decisions:

```text
AGENTS.md
NEXT_WINDOW_HANDOFF.md
docs/PAPER_INNOVATION_ROADMAP_2026-05-25.md
docs/EXTERNAL_METHOD_REVIEW_BACKLOG_2026-05-25.md
docs/AUTOMATED_AGENT_WORKFLOW_2026-05-24.md
docs/PROJECT_DIRECTION_DECISION_2026-05-24.md
docs/PHASE_1B_FULL_RUN_ANALYSIS_2026-05-24.md
```

Current consolidated report inputs:

```text
checkpoints/phase1b_local_reports/table_records_20260525/run_summary.csv
checkpoints/phase1b_local_reports/table_records_20260525/pooled_by_model.csv
checkpoints/phase1b_local_reports/table_records_20260525/by_model_ticker.csv
checkpoints/phase1b_local_reports/table_records_20260525/coverage_by_ticker.csv
```

Current paper table outputs:

```text
checkpoints/phase1b_paper_tables_20260525/paper_tables.md
checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_3_canonical_ticker_delta.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_7_regime_shift_by_ticker.csv
checkpoints/phase1b_paper_tables_20260525/paper_table_8_coverage_fragility_flags.csv
checkpoints/phase1b_paper_tables_20260525/figure_delta_vs_coverage.csv
checkpoints/phase1b_paper_tables_20260525/figure_ticker_delta_heatmap.csv
checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv
```

## Goal Mode Opening Prompt

Use this prompt in the new Codex window:

```text
Continue the hf_stock_clf project in Goal Mode.

First read:
AGENTS.md
NEXT_WINDOW_HANDOFF.md
docs/GOAL_MODE_NEXT_WINDOW_2026-05-25.md
docs/PAPER_INNOVATION_ROADMAP_2026-05-25.md
docs/EXTERNAL_METHOD_REVIEW_BACKLOG_2026-05-25.md

Use the existing evidence package under:
checkpoints/phase1b_local_reports/table_records_20260525
checkpoints/phase1b_paper_tables_20260525

Goal:
Create a paper-ready claim and innovation package for the current project.

Do not train models. Do not add PatchTST, attention, NLP, sentiment, RL,
backtesting, or external repository code. Do not modify ml_utils. Do not touch
completed run directories. Treat diagnostic results as descriptive and
hypothesis-generating only.

Recommended write scope:
docs/PAPER_GOAL_CLAIM_MAP_2026-05-25.md
docs/PAPER_INNOVATION_CARDS_2026-05-25.md
docs/PAPER_RESULTS_NARRATIVE_DRAFT_2026-05-25.md

Use separate agents if available:
1. evidence auditor: verify every numeric claim against the table files;
2. innovation reviewer: map paper/code sources to project-owned innovations;
3. claim-control reviewer: block overclaims, alpha/profit language, and
   diagnostic-to-canonical drift.

End with files inspected, files changed, commands run, validation results, and
unresolved issues.
```

## Goal Deliverables

The new window should produce three docs-only artifacts.

### Deliverable 1 - Paper Goal Claim Map

Target file:

```text
docs/PAPER_GOAL_CLAIM_MAP_2026-05-25.md
```

Purpose:

```text
Map every allowed paper claim to concrete project evidence, and list the
claims that remain blocked.
```

Required sections:

```text
1. Thesis statement
2. Evidence table
3. Allowed claims
4. Blocked claims
5. Diagnostic-only claims
6. Required caveats
7. Missing evidence
```

Minimum evidence rows:

```text
canonical pooled result
canonical per-ticker result
ticker delta count result
seed ticker stability result
regime diagnostic comparison result
coverage fragility result
threshold-retention proxy result
model-expansion gate result
```

Strict rules:

```text
Do not write "alpha".
Do not write "profitable strategy".
Do not write "market regime shift".
Do not write "stocks are unpredictable".
Do not write "Transformers fail".
Do not treat 0bps or 5bps diagnostics as replacement tasks.
Do not call threshold-retention proxy a confidence curve.
```

### Deliverable 2 - Paper Innovation Cards

Target file:

```text
docs/PAPER_INNOVATION_CARDS_2026-05-25.md
```

Purpose:

```text
Convert the brainstormed innovation directions into falsifiable, project-owned
research cards.
```

Recommended cards:

```text
1. Leakage-safe weak-signal evaluation harness
2. Dummy-first model expansion gate
3. Diagnostic risk/coverage framing for stock direction labels
4. Ticker-local weak-signal heterogeneity
5. Seed sensitivity and robustness disclosure
6. Label/window/horizon/coverage stability map
7. External-method adapter firewall
8. Simple TSC baseline stress test
9. Linear-family ablation for stock direction
```

Each card must include:

```text
Source inspiration:
Project problem addressed:
Existing evidence:
Next artifact needed:
Falsifiable experiment:
Allowed claim if supported:
Blocked claim even if supported:
Implementation gate status:
```

Innovation ranking for the current paper:

```text
P0 - can be written now:
  leakage-safe weak-signal evaluation harness
  dummy-first model expansion gate
  diagnostic risk/coverage framing
  ticker-local heterogeneity
  seed/coverage robustness disclosure

P1 - can be future-work/spec now:
  label/window/horizon/coverage stability map
  adapter firewall

P2 - do not implement now:
  MiniROCKET/ROCKET baseline
  linear-family ablation
  foundation-model adapter
```

### Deliverable 3 - Paper Results Narrative Draft

Target file:

```text
docs/PAPER_RESULTS_NARRATIVE_DRAFT_2026-05-25.md
```

Purpose:

```text
Draft the Results and Limitations story using the current 11 paper tables,
without inventing metrics or outcomes.
```

Required sections:

```text
1. Main result: canonical sequence models do not beat dummy
2. Per-ticker heterogeneity: local positives/negatives are not broad evidence
3. Seed stability: 3-seed dispersion and sensitivity
4. Diagnostic regimes: 0bps/5bps are descriptive only
5. Coverage fragility: retained_pct and n_test_windows change interpretation
6. Threshold-retention proxy: not confidence-based selective classification
7. Model expansion gate: why it remains blocked
8. Limitations and next approved evidence
```

Writing rules:

```text
Every numeric result must cite one table path.
Every diagnostic sentence must include "diagnostic" or "descriptive".
Every local positive result must be paired with pooled/canonical caution.
Every future-method paragraph must say the method is gated, not approved.
```

## Brainstormed Research Position

The strongest version of this project is not:

```text
We built a better stock predictor.
```

It is:

```text
We built a protocol firewall for intraday stock direction prediction and used
it to show that apparent model gains are fragile under dummy baselines,
ticker-level disclosure, seed reporting, and coverage accounting.
```

Two-sentence paper pitch:

```text
High-frequency stock direction classification is vulnerable to leakage,
weak baselines, and task-definition drift, which can make ordinary sequence
models look more useful than they are. We present a leakage-safe evaluation
harness and show that, on the current 5-stock 5-minute setup, LSTM, TCN, and
DLinear do not robustly outperform stratified dummy baselines once ticker,
seed, coverage, and diagnostic label semantics are disclosed.
```

Core tension:

```text
Architecture ambition vs. protocol credibility.
```

Resolution:

```text
Do not add larger models until simple, protocol-safe evidence establishes a
stable signal.
```

Current P0 innovation candidates:

```text
1. Leakage-safe weak-signal evaluation harness:
   protocol control is the contribution, not a bigger architecture.

2. Dummy-first model expansion gate:
   architecture expansion is blocked until model-vs-dummy evidence clears the
   documented threshold.

3. Coverage and label-semantics disentanglement:
   diagnostic no-trade-band results must be interpreted through retained_pct,
   n_test_windows, and class balance.

4. Ticker-local weak-signal heterogeneity:
   local positive diagnostics are hypothesis-generating and must be paired with
   pooled/canonical caution.

5. Seed stability and robustness disclosure:
   3-seed dispersion is reported as sensitivity evidence, not confirmatory
   inference.
```

## Agent Workflow For The New Window

Use agents only for bounded sidecar work.

Recommended split:

```text
Main agent:
  read docs, create the three target files, run final checks

Evidence auditor:
  verify table paths, columns, and numeric claims

Innovation reviewer:
  check each innovation card against roadmap/backlog and paper requirements

Claim-control reviewer:
  find overclaims, diagnostic drift, and blocked wording
```

Do not let agents modify files unless the main agent has assigned a disjoint
write set. For this goal, read-only agents are preferred.

## Validation Commands

This is a docs-first goal, so validation is mostly consistency checking.

Required:

```powershell
git -C E:\codex_workspace\projects\hf_stock_clf diff --check
git -C E:\codex_workspace\projects\hf_stock_clf status --short
git -C E:\codex_workspace\projects\hf_stock_clf diff --stat
```

If the new window touches any report scripts or tests, also run:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\phase1b_local\build_paper_tables.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_phase1b_paper_tables.py -q
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\ -q -m "not integration"
```

Do not run heavy training unless the user explicitly asks.

## Stop Rules

Stop and ask if the new window wants to:

```text
change label semantics
change split/scaler/window behavior
change the canonical task
open PatchTST or Transformer implementation
add NLP/sentiment/RL/backtesting
copy code from external repositories or reference_excerpts
modify ml_utils while doing paper-claim work
write claims not supported by current table files
turn diagnostic findings into main-task conclusions
```

## Done Criteria

The goal is complete when:

```text
1. The three docs target files exist.
2. Every numeric claim points to an existing table or report file.
3. Blocked claims are listed explicitly.
4. Innovation cards are ranked and gate-labeled.
5. No new model or heavy training was introduced.
6. Claim-control review has no unresolved blocker.
7. Final report lists files inspected, files changed, commands run,
   validation results, and unresolved issues.
```
