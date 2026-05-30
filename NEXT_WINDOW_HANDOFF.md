# NEXT WINDOW HANDOFF - hf_stock_clf - 2026-05-24

## Short Answer

Start a new Codex window for the next phase.

The current thread has enough context to continue, but the project state is now
large enough that a fresh window is safer. The agent workflow should continue:
Manager / Runner / Reviewer / Code-Control.

## Project Decision

Continue `hf_stock_clf`, but keep the goal narrow:

```text
Local-first, leakage-safe evaluation harness for high-frequency stock direction
classification, with external models integrated only after the protocol and
baseline table survive review.
```

Do not continue as a model-stacking project yet. Do not add PatchTST, attention,
NLP, sentiment, RL, or copied external repositories based on the current
evidence.

## Current Evidence

The canonical Phase 1 full-binary run completed:

```text
run_dir = checkpoints/phase1_canonical_binary_full/phase1b_local_legacy_binary_full_20260524_230605
label_mode = legacy_binary
label_semantics = canonical_phase1_full_binary
label_formula = label = 1 if future_avg_r > 0 else 0
zero_return_policy = class_0_non_up
no_trade_band_enabled = false
feature_set_id = technical_v1
tickers = CSCO, JPM, KO, MSFT, WMT
models = lstm, tcn, dlinear
seeds = 42, 43, 44
max_epochs = 3
batch_size = 512
rows = 54
suspicious rows = 0
pooled retained_pct = 0.847315
pooled test windows = 235333
pooled zero_return rows = 13
```

Pooled canonical result:

```text
tcn      delta_macro_f1_vs_dummy = -0.002318
lstm     delta_macro_f1_vs_dummy = -0.015578
dlinear  delta_macro_f1_vs_dummy = -0.023092
```

All model/ticker mean deltas versus the per-ticker dummy baseline are negative.
The current model-expansion gate is therefore closed.

## Gate Status

```text
dataset/test gate: ready
runner gate: supports diagnostics and canonical full-binary
5bps Phase 1B diagnostic table: complete
0bps strict-sign diagnostic analog: complete
canonical Phase 1 full-binary table: complete
PatchTST/new-model gate: blocked
```

## Files To Read First In The New Window

Read these before doing anything else:

```text
AGENTS.md
NEXT_WINDOW_HANDOFF.md
docs/PROJECT_DIRECTION_DECISION_2026-05-24.md
docs/PHASE_1B_FULL_RUN_ANALYSIS_2026-05-24.md
docs/AUTOMATED_AGENT_WORKFLOW_2026-05-24.md
```

## Dirty Tree At Handoff

Latest observed `git status --short`:

```text
 M ml_utils/dataset.py
 M tests/test_window_boundaries.py
?? .codegraph/
?? docs/PHASE_1B_FULL_RUN_ANALYSIS_2026-05-24.md
?? docs/PROJECT_DIRECTION_DECISION_2026-05-24.md
?? notebooks/04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb
?? "notebooks/Binary classification comparison_Zhang (2).ipynb"
?? notebooks/P1B.21d_notebook03_ticker_axis_narrow_smoke.ipynb
?? scripts/phase1b_local/
?? tests/test_phase1b_local_runner.py
```

Latest observed `git diff --stat` for tracked changes:

```text
ml_utils/dataset.py             | 200 ++++++++++++++++++++++++++++++----------
tests/test_window_boundaries.py | 180 ++++++++++++++++++++++++++++++++++++
2 files changed, 332 insertions(+), 48 deletions(-)
```

Treat all pre-existing dirty changes as user/project state. Do not revert.

## Validation Already Reported

Use the project Python:

```text
E:\codex_workspace\_envs\py311_shared\python.exe
```

Validation reported in the current session:

```text
runner py_compile: passed
runner semantic tests + label/config/window tests: 74 passed
full non-integration tests: 154 passed, 1 existing checkpoint warning
5-ticker canonical smoke: completed, 18 rows, suspicious rows = 0
canonical full-run: completed, 54 rows, suspicious rows = 0
```

The existing warning is from checkpoint-related tests and was treated as
pre-existing.

## Next Window Opening Prompt

Use this prompt in the new Codex window:

```text
Continue the hf_stock_clf project. First read AGENTS.md,
NEXT_WINDOW_HANDOFF.md, docs/PROJECT_DIRECTION_DECISION_2026-05-24.md,
docs/PHASE_1B_FULL_RUN_ANALYSIS_2026-05-24.md, and
docs/AUTOMATED_AGENT_WORKFLOW_2026-05-24.md.

Continue using the Manager / Runner / Reviewer / Code-Control agent workflow.
Do not add PatchTST, attention, NLP, sentiment, RL, or copied external
repositories. The canonical Phase 1 full-binary run is complete and did not beat
dummy. Keep the PatchTST/new-model gate blocked.

Next task: automate the project workflow around the existing local runner:
standardize report generation from result CSVs, produce clean summary tables,
audit the current dirty tree as a coherent patch set, and design the next
protocol-analysis step. Use local data only and do not run heavy training unless
the task explicitly approves it.
```

## Recommended Next Phase

Do not run another full model experiment first. Do this instead:

1. Build a small report generator for existing run directories.
2. Produce canonical/diagnostic summary tables from CSVs without notebook-only
   logic.
3. Review the current patch set as a coherent change:
   - `ml_utils/dataset.py`
   - `tests/test_window_boundaries.py`
   - `scripts/phase1b_local/local_baseline_matrix.py`
   - `tests/test_phase1b_local_runner.py`
   - docs added on 2026-05-24
4. Decide whether the next scientific question is:
   - simpler non-sequence baselines;
   - feature/label stability analysis;
   - ticker-specific weak-signal analysis;
   - report/paper framing around strict evaluation and negative evidence.

## Automation Direction

Automate the workflow, not the modeling ambition:

```text
preflight -> smoke -> full-run only when approved -> result summary -> reviewer
gate -> code-control gate -> docs/handoff update
```

Keep run outputs structured under `checkpoints/` and keep reusable reporting code
inside an approved project location such as `scripts/phase1b_local/`.

## Report Automation Update - 2026-05-25

The reusable local report summarizer now exists:

```text
scripts/phase1b_local/summarize_runs.py
tests/test_phase1b_report_summarizer.py
```

It reads completed run directories only. Required run inputs are:

```text
metadata.json
manifest.csv
results.csv
```

It writes normalized report-layer artifacts without modifying original run
outputs:

```text
run_summary.csv
pooled_by_model.csv
by_model_ticker.csv
coverage_by_ticker.csv
report.md
```

Validated consolidated report:

```text
checkpoints/phase1b_local_reports/table_records_20260525
```

The report covers:

```text
canonical Phase 1 full-binary:
  checkpoints/phase1_canonical_binary_full/phase1b_local_legacy_binary_full_20260524_230605

5bps Phase 1B high-magnitude diagnostic:
  checkpoints/phase1b_local_table_record_5bps/phase1b_local_full_20260524_215040

0bps strict-sign diagnostic analog:
  checkpoints/phase1b_local_table_record_0bps/phase1b_local_full_20260524_220040
```

Current consolidated gate:

```text
canonical best pooled model = tcn, delta_macro_f1_vs_dummy = -0.002318
0bps diagnostic best pooled model = tcn, delta_macro_f1_vs_dummy = +0.004890
5bps diagnostic best pooled model = lstm, delta_macro_f1_vs_dummy = +0.001893
model expansion gate = blocked_delta_lt_0.01 for all three runs
suspicious rows = 0 for all three runs
```

Important compatibility detail:

```text
Older 0bps/5bps diagnostic CSVs do not contain the newer protocol-semantic
columns or zero-return count columns. The summarizer backfills protocol labels
from metadata/label_mode, but keeps missing zero-return counts blank rather than
fabricating them.
```

Validation:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\phase1b_local\summarize_runs.py
passed

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_phase1b_report_summarizer.py -q
4 passed

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_window_boundaries.py tests\test_phase1b_local_runner.py tests\test_phase1b_report_summarizer.py -q
24 passed

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\ -q -m "not integration"
158 passed, 1 existing checkpoint scheduler warning
```

Recommended next task:

```text
Use the consolidated report to design a protocol-analysis step, not a model
expansion step. Preferred candidates:

1. simpler non-sequence baselines;
2. feature/label stability analysis;
3. ticker-specific weak-signal diagnostics;
4. report framing around strict evaluation and negative evidence.
```

## Patch Audit Resolution - 2026-05-25

A read-only Reviewer / Runner / Code-Control audit was run after report
automation. The audit findings were handled in the same window.

### 1. Window label alignment contract

Current implementation and tests use the label at the end of the input window:

```text
target_idx = local_start_idx + window_size - 1
label = labels[target_idx]
```

Relevant files:

```text
ml_utils/dataset.py
tests/test_window_boundaries.py
tests/test_window_label_alignment.py
```

But `AGENTS.md` and `docs/ml_utils_construction_plan_v2.md` still state that
window construction skips `label==NaN` starting points, and plan v2 says the
label at starting point `t` must be non-NaN.

Resolution:

```text
Window-end label alignment is now the official project contract.
AGENTS.md and docs/ml_utils_construction_plan_v2.md were updated to state that
WindowedClassificationDataset skips windows whose window-end label is NaN.
```

### 2. Runner default label mode

The local runner defaults to:

```text
label_mode = no_trade_band
```

That is useful for Phase 1B diagnostics, but it is not the AGENTS-defined
canonical full-binary task. The runner now supports canonical mode via:

```text
--label-mode legacy_binary
```

Resolution:

```text
Runner default changed to legacy_binary so accidental runs are canonical.
Phase 1B diagnostic subsets now require explicit --label-mode no_trade_band.
```

### 3. Runner baseline detail

The runner emits baseline macro-F1 fields, but does not yet emit baseline
balanced accuracy or baseline confusion matrices. This weakens review of class
imbalance behavior.

Resolution:

```text
local_baseline_matrix.py now emits baseline balanced accuracy and confusion
matrix fields for pooled-train baselines and per-ticker baselines.
summarize_runs.py tolerates these fields as optional for older CSV artifacts.
```

### 4. Trim-time timestamp order

`trim_labels_at_split_boundary` sorts each ticker group before validating
timestamp order. `WindowedClassificationDataset` rejects out-of-order timestamps,
but trim itself could still accept and reorder disordered split input.

Resolution:

```text
trim_labels_at_split_boundary now validates timestamp order in the incoming
per-ticker order and fails fast instead of sorting before validation.
```

Current validation around the dirty patch:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\phase1b_local\local_baseline_matrix.py scripts\phase1b_local\summarize_runs.py
passed

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_window_boundaries.py tests\test_phase1b_local_runner.py tests\test_phase1b_report_summarizer.py -q
superseded by broader targeted validation

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_config.py tests\test_dataset_leakage.py tests\test_label_generation.py tests\test_no_trade_band_labels.py tests\test_window_boundaries.py tests\test_window_label_alignment.py tests\test_phase1b_local_runner.py tests\test_phase1b_report_summarizer.py -q
superseded by broader targeted validation including tests/test_metrics.py

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_config.py tests\test_metrics.py tests\test_label_generation.py tests\test_no_trade_band_labels.py tests\test_dataset_leakage.py tests\test_window_boundaries.py tests\test_window_label_alignment.py tests\test_phase1b_local_runner.py tests\test_phase1b_report_summarizer.py -q
105 passed

E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\ -q -m "not integration"
163 passed, 1 existing checkpoint scheduler warning

E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\summarize_runs.py --run-dir checkpoints\phase1_canonical_binary_full\phase1b_local_legacy_binary_full_20260524_230605 --run-dir checkpoints\phase1b_local_table_record_0bps\phase1b_local_full_20260524_220040 --run-dir checkpoints\phase1b_local_table_record_5bps\phase1b_local_full_20260524_215040 --output-dir checkpoints\phase1b_local_reports\table_records_20260525
completed
```
