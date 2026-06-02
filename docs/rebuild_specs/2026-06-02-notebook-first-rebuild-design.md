# Notebook-First Rebuild Design

Date: 2026-06-02
Status: active PM design
Scope: planning and structure only; no notebook execution, training, raw-data
mutation, checkpoint mutation, or holdout/test access.

## Goal

Rebuild `intraday_stock_direction_research` around one readable
validation-only research notebook first, then extract only the small helpers
that the notebook proves are necessary. The project should remain a research
notebook project, not a restored backend runner or PM route-control system.

## Current Evidence

- Active notebook: `notebooks/04_ian_research_memo.ipynb`
- Current notebook state from read-only nbformat audit: 7 cells, 1 code cell,
  6 markdown cells, no outputs, no execution counts.
- Active docs: `AGENTS.md`, `README.md`, `docs/RESEARCH_WORKFLOW.md`,
  `docs/BASELINE_REFERENCE.md`, `docs/ENVIRONMENT.md`.
- Archive: `archive/legacy_model_runner_reference/` contains old runner,
  helper, model, test, and reference material.
- Data files observed by parallel audit: `data/CSCO.csv`, `data/JPM.csv`,
  `data/KO.csv`, `data/MSFT.csv`, `data/WMT.csv`.
- Current git state at planning time: `master...origin/master [ahead 8]` with
  tracked edits in `AGENTS.md` and `docs/BASELINE_REFERENCE.md`.

## PM And Agent Inputs

### Background Thread: Archive Migration And Risk Audit

Thread `019e868a-6964-7400-95f1-201bb4319998` concluded:

- P0 should be notebook-first `baseline_v1` reconstruction.
- `archive/legacy_model_runner_reference/` is richer than the active docs imply.
- Archive contains references for `baseline_v1`, LightGBM validation-only logic,
  MS-DLinear+TCN, metrics, dataset/window logic, and tests.
- These assets are reference material only. They require migration/spec audit
  before active use.

### Explorer B: Notebook Skeleton Audit

Sub-agent `Averroes` concluded:

- The active notebook is safe but too thin.
- It should become a 17-section notebook skeleton covering setup, config, data,
  features, labels, splits, train-only scaling, windows, dummy baselines, model
  availability, comparison tables, plots, and Ian-facing interpretation.
- All heavy/model cells should default off.
- The notebook must not import archive helpers, old runner utilities, or
  `ml_utils`.

## Design Decision

Use an A-first rebuild:

1. Build the active research notebook skeleton first.
2. Keep baseline logic inline until notebook evidence proves helper extraction
   is useful and safety-critical.
3. Treat archived LightGBM and MS-DLinear+TCN code as migration references, not
   active implementations.
4. Use active model adapters only after separate migration/spec review.

This sequence is deliberately notebook-first. It is still PM+agents driven, but
the PM artifacts stay minimal and do not recreate `PM_NNN`, closeout, handoff,
or route-control documents.

## Work Packages

### WP-A: Notebook-First Baseline Reconstruction

Primary file:

- `notebooks/04_ian_research_memo.ipynb`

Purpose:

- Convert the current thin memo skeleton into a linear, readable,
  validation-only research notebook skeleton.
- Make the notebook ready for later guarded implementation of `baseline_v1`
  data loading, features, labels, splits, train-only scaling, windowing, dummy
  baselines, and comparison reporting.

Allowed:

- Structural notebook edits with `nbformat`.
- Markdown and code cells that define guards, metadata, schemas, and placeholder
  logic.
- Static validation that the notebook has no outputs, no execution counts, and
  required headings.

Forbidden:

- Executing notebook cells.
- Running training.
- Reading or reporting holdout/test metrics.
- Importing `archive/legacy_model_runner_reference`, old runner utilities, or
  `ml_utils`.
- Writing checkpoints or artifacts.

### WP-B: Leakage-Safe Helper Extraction

Primary future files:

- A small active helper module only if the notebook proves repeated,
  safety-critical logic.
- Focused tests only for extracted helper behavior.

Reference-only assets:

- `archive/legacy_model_runner_reference/runner_utils/dataset.py`
- `archive/legacy_model_runner_reference/runner_utils/metrics.py`
- Related archived tests for split boundaries, windows, labels, and metrics.

Purpose:

- Extract label invalidation, split logic, train-only scaling, per-ticker window
  building, and dummy baseline logic only after notebook logic stabilizes.

Forbidden:

- Restoring the old helper library wholesale.
- Recreating a backend framework.
- Adding generic registries, hooks, callback systems, or route-control layers.

### WP-C: Model Adapter And Evaluation Harness Migration

Primary future files:

- Active LightGBM adapter or validation cell only after a migration audit.
- Active MS-DLinear+TCN adapter only after shape/spec audit.
- Small evaluation schema for validation-only reporting.

Reference-only assets:

- `archive/legacy_model_runner_reference/scripts/local_runner_reference/local_baseline_matrix.py`
- `archive/legacy_model_runner_reference/runner_utils/models/ms_dlinear_tcn_classifier.py`
- Archived model and runner tests.

Correct status wording:

- LightGBM: archive has validation-only references; active migration audit
  required before use.
- MS-DLinear+TCN: archive has model/tests; active migration/spec audit required
  before use.

Forbidden:

- Running the archived CLI as the active workflow.
- Reporting old checkpoint results as current notebook evidence.
- Calling `sklearn_logreg` LightGBM.
- Opening or selecting from holdout/test.

## Notebook Skeleton Target

The active notebook should be restructured into these sections:

1. `# Notebook 04 - Ian Research Memo`
2. `## Setup And Run Guards`
3. `## Baseline Configuration`
4. `## Data Contract And Paths`
5. `## Load And Coverage Diagnostics`
6. `## Causal Feature Construction`
7. `## Label Construction And Invalid Markers`
8. `## Chronological Split And Split-Boundary Invalidation`
9. `## Train-Only Preprocessing`
10. `## Per-Ticker Per-Split Window Construction`
11. `## Target And Class Balance Diagnostics`
12. `## Stratified Dummy Baselines`
13. `## Model Availability And Validation Plan`
14. `## Validation-Only Model Cells`
15. `## Comparison Table`
16. `## Validation Plots`
17. `## Ian-Facing Interpretation`

Required static metadata:

- `RESULT_SCOPE = "validation_only"`
- `RUN_DATA_LOAD = False`
- `RUN_FEATURE_BUILD = False`
- `RUN_MODEL_VALIDATION = False`
- `RUN_TRAINING = False`
- `FEATURE_SET_ID = "baseline_v1"`
- `LABEL_POLICY = "no_trade_band"`
- `THRESHOLD_SOURCE = "fixed_pre_registered_5bps"`
- `THRESHOLD_BPS = 5.0`
- `LABEL_HORIZON_K = 12`
- `WINDOW_SIZE = 12`
- `DECISION_TIME_POLICY = "post_bar_close_completed_bar"`
- `SCALER_ID = "standard_pooled_train_only_v1"`
- `SEED = 42`
- `TICKERS = ["CSCO", "JPM", "KO", "MSFT", "WMT"]`

Required comparison schema:

- `model`
- `ticker_or_pooled`
- `macro_f1`
- `balanced_accuracy`
- `dummy_macro_f1`
- `delta_macro_f1_vs_dummy`
- `n`
- `scope`

## Validation Strategy

Static notebook validation only:

- All code cell `execution_count` values are `None`.
- All code cell `outputs` lists are empty.
- Required section headings appear in order.
- Required metadata symbols appear.
- Default guards are false.
- No active code imports or path references `archive`, `legacy_model_runner`,
  or `ml_utils`.
- No active code calls `train_test_split`.
- No active code reads or reports holdout/test metrics.

Smoke validation policy:

- Smoke runs may execute the notebook in memory and use train/validation rows
  only after explicit approval.
- Smoke results are pipeline diagnostics, not decision evidence.
- Smoke metrics must not select features, thresholds, models, hyperparameters,
  claim wording, or holdout/test access.
- Smoke reports must state `validation_only_smoke_not_evidence` or equivalent
  non-decision scope language.
- Current smoke reports:
  - `docs/rebuild_specs/2026-06-02-p0-validation-smoke-report.md`
  - `docs/rebuild_specs/2026-06-02-p0-multiticker-validation-smoke-report.md`

If a future tiny validation is approved, it must be a separate task with:

- Explicit approval.
- New non-overwriting output path.
- Validation-only scope.
- No holdout/test readout.
- Complete baseline metadata.
- Dummy baseline comparison.

If a future full validation is approved, it must be pre-registered separately
from the smoke reports. The pre-registration must define tickers, split scope,
model family, feature set, threshold policy, sample/output path, metrics, and
claim language before running. Smoke values cannot be copied into that decision.

## Open Decisions

These decisions are not blockers for writing the skeleton, but they must be
settled before any run:

- Confirm whether the desired decision time remains post-bar-close or moves to
  stricter pre-close prediction.
- Confirm whether data coverage and column-schema audit may read `data/*.csv`.
- Confirm whether any validation-only smoke may write a new output directory.
- Confirm when helper extraction is allowed.
