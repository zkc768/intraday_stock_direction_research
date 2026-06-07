# Controlled Follow-Up Protocol - 2026-06-04

Status: frozen before Notebook 04 implementation

Frozen: 2026-06-04

Scope: `validation_only`

Target notebook: `notebooks/04_controlled_followup_colab.ipynb`

Predecessors:

- `notebooks/02_config_screening_colab.ipynb`
- `notebooks/02_diagnostic_h0_tabular_sweep_colab.ipynb`
- `notebooks/03_model_family_screening_colab.ipynb`

Version log:

- 1.0 (2026-06-04): Initial controlled follow-up protocol after Notebook 03
  model-family screening and the diagnostic-only H0 window appendix.
- 1.1 (2026-06-04): Add pre-04 provenance requirement for the
  MS-DLinear+TCN design-review exit and add a cumulative validation-trial
  budget tracker for future holdout authorization review.

This is a technical research protocol, not a PM handoff, not a holdout/test
authorization, and not a final thesis result claim. It freezes what Notebook 04
is allowed to do before any new validation result is used.

## 1. Evidence Refresh Summary

The local literature library and current online academic sources support one
change in emphasis, not a new model-family expansion.

New design implications:

1. Selective/no-trade analysis should be a controlled post-hoc validation
   diagnostic over saved prediction probabilities. It should report the full
   retained-coverage profile, not a cherry-picked high-confidence subset.
2. Confidence values should be interpreted within the same model only.
   Calibration work shows that raw model confidence is not guaranteed to be
   comparable across model families.
3. Selective curves are not proof of profitability. They are validation-only
   evidence about whether a model ranks its own predictions by confidence in a
   useful way.
4. Calibration layers, threshold learning, SelectiveNet-style training, and
   trading backtests are separate future methodology branches. They are not
   added to Notebook 04.
5. The reviewed model literature does not justify adding PatchTST,
   InceptionTime, DeepLOB, NLP/news features, sentiment features, or external
   market features to the immediate Notebook 04 panel.

Design anchors used for this protocol:

- Selective classification and risk-coverage framing:
  <https://arxiv.org/abs/1705.08500>
- SelectiveNet as a future method, not a Notebook 04 method:
  <https://arxiv.org/abs/1901.09192>
- Trading via selective classification and abstention/no-position framing:
  <https://arxiv.org/abs/2110.14914>
- Calibrated selective classification and uncertainty-quality caveats:
  <https://arxiv.org/abs/2208.12084>
- Common selective-evaluation flaws and multi-threshold reporting:
  <https://arxiv.org/abs/2407.01032>
- Neural-network calibration caveat:
  <https://arxiv.org/abs/1706.04599>
- TCN baseline motivation:
  <https://arxiv.org/abs/1803.01271>
- DLinear baseline motivation:
  <https://arxiv.org/abs/2205.13504>
- LightGBM tabular baseline:
  <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>

## 2. Purpose

Notebook 04 answers one narrow question:

```text
Given the official Stage 0 candidate and the Notebook 03 selected branch
context, does a fixed, small follow-up panel still show validation-only signal
under fresh seeds, and does any model show useful within-model selective
confidence structure?
```

Notebook 04 does not answer:

- final model quality;
- profitability;
- transaction-cost performance;
- final holdout/test performance;
- whether `window_size=32` should replace the official Stage 0 candidate;
- whether a new architecture family should be introduced after seeing
  validation results.

The immediate output of Notebook 04 is a controlled decision record for the
next notebook. The next notebook may be a bounded tuning notebook only if
Notebook 04 justifies that direction under the gates below.

## 3. Entry Conditions

Notebook 04 may start only if all conditions hold:

1. Notebook 02 has fixed the official Stage 0 candidate:

   ```text
   label_config = h03_bps1p5
   feature_set = price_volume_time
   window_size = 20
   ```

2. The H0 `window_size=32` result, if present, is labeled
   `diagnostic-only` and `non-selecting`.
3. Notebook 03 has produced `notebook03_validation_selection.json`.
4. `notebook03_validation_selection.json` contains:

   ```json
   {
     "scope": "validation_only",
     "holdout_test_authorized": false
   }
   ```

5. The final holdout/test interval remains closed.
6. The Notebook 04 model panel, seed set, metrics, gates, and output schema in
   this document are frozen before Notebook 04 runs.

If any condition is not satisfied, Notebook 04 must stop before fitting any
model.

## 4. Route Boundary

The active route is:

```text
02_config_screening_colab
  -> diagnostic H0 appendix, non-selecting
  -> 03_model_family_screening_colab
  -> 04_controlled_followup_colab
  -> conditional 05 tuning/design notebook
  -> eventual holdout/test only under a separate pre-registered note
```

Notebook 04 must not retroactively change Notebook 02, H0, or Notebook 03
decisions. It carries their context forward and adds fresh validation-only
evidence.

## 5. Data And Leakage Boundary

Notebook 04 inherits the active raw-data-first Colab boundary:

1. Do not import `intraday_research`, prior notebooks, `baseline_helpers`, or
   archived helper packages as the active path.
2. Download the five approved raw ticker files into a local Colab runtime
   directory.
3. Do not read, transform, window, summarize, score, display, or otherwise use
   holdout/test rows.
4. Split each ticker chronologically before pooled preprocessing.
5. Fit preprocessing only on pooled train rows.
6. Transform validation rows using train-fitted preprocessing only.
7. Generate windows per ticker and per trading day.
8. Drop samples whose input window or label horizon crosses a ticker boundary,
   trading-day boundary, train/validation boundary, or validation/closed-holdout
   boundary.
9. Features may use only the current completed bar and earlier completed bars.
10. Invalid labels are exclusion markers, not fill targets.

Any deviation from the Notebook 02 feature, label, split, or window logic is a
protocol failure unless a separate pre-run note freezes that deviation before
execution.

## 6. Fixed Model Panel

Notebook 04 uses a smaller panel than Notebook 03. This is intentional.

| model | role in Notebook 04 |
|---|---|
| `stratified_dummy` | mandatory dummy baseline |
| `always_up_dummy` | mandatory directional sanity baseline |
| `logreg` | simple linear/tabular reference |
| `lightgbm` | Ian-aligned tabular candidate and strongest simple branch |
| `standalone_tcn` | controlled sequence comparator |
| `ms_dlinear_tcn` | Ian-aligned multi-scale DLinear plus residual TCN candidate |

Excluded from Notebook 04:

- `vanilla_lstm`
- `simple_gru`
- `standard_dlinear`
- PatchTST
- InceptionTime
- DeepLOB
- NLP/news/sentiment models
- external market features

The excluded Notebook 03 sequence models are not being declared invalid. They
are excluded because Notebook 04 is a controlled follow-up around the simple
tabular branch, the standalone TCN comparator, and the Ian-aligned
MS-DLinear+TCN branch. Adding more families after Notebook 03 would increase
validation search width without a pre-registered reason.

## 7. Fixed Hyperparameters

Notebook 04 does not tune hyperparameters.

Tabular defaults:

```text
logreg:
  sklearn LogisticRegression
  solver = liblinear
  class_weight = balanced
  max_iter = 2000
  C = 1.0
  input_view = flattened window

lightgbm:
  n_estimators = 200
  learning_rate = 0.03
  max_depth = 6
  num_leaves = 31
  subsample = 0.9
  subsample_freq = 1
  colsample_bytree = 0.9
  class_weight = balanced
  random_state = seed
  verbosity = -1
  input_view = flattened window
```

Torch shared defaults:

```text
epochs = 8
batch_size = 1024
learning_rate = 0.001
optimizer = AdamW
weight_decay = 0.0001
dropout = 0.10
loss = class-weighted CrossEntropyLoss using train labels
early_stopping = disabled
```

Sequence defaults:

```text
standalone_tcn:
  causal TCN
  channels = (32, 32)
  kernel_size = 3
  dropout = 0.10
  head = final timestep -> 2 logits

ms_dlinear_tcn:
  moving_avg_kernels = (3, 5, 9, 15)
  residual TCN channels = (32, 32)
  TCN kernel_size = 3
  dropout = 0.10
  head = DLinear branch + residual TCN branch -> 2 logits
```

Notebook 04 must not add early stopping, class-threshold tuning, probability
calibration, architecture changes, feature changes, or additional window sizes.

## 8. Fresh-Seed Budget

Notebook 04 uses fresh seeds that were not used by Notebook 02 or Notebook 03:

```text
seeds = 606, 707, 808, 909, 1010
```

Expected Notebook 04 fitted rows:

```text
6 models x 5 seeds = 30 pooled result rows
4 fitted real models x 5 seeds = 20 fitted model rows
2 dummy models x 5 seeds = 10 baseline rows
```

04C, 04D, and optional 04E must not fit models. They read only Notebook 04
result and prediction artifacts.

Multiple-comparison caveat:

Notebook 04 is still validation-only search. It may confirm or weaken a branch,
but it cannot erase the cumulative selection cost from Notebook 02, H0, and
Notebook 03. Any positive result must be described as selected validation
evidence, not as expected holdout/test performance.

Numeric validation-trial budget tracker:

| source | validation model-seed rows | count status | basis |
|---|---:|---|---|
| Stage 0S schema smoke | 1 | exact planned | `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` lists one LightGBM seed run. |
| Stage 0A1 label-feature screen | 90 | exact planned | 3 labels x 3 feature sets x 1 window x 2 models x 5 seeds. |
| Stage 0A2 window sensitivity | 15-30 | planned range | 15 rows if `mean_candidate` and `lcb_candidate` share one label-feature pair; 30 if they differ. |
| Stage 0B deep-model second-view | 40 | planned maximum | Freeze lists 2 candidates x 4 models x 5 seeds; actual completed candidate count should be checked from Stage 0 artifacts before any holdout note. |
| Diagnostic H0 tabular sweep | 180 | exact canonical planned before optional Round 2 | `docs/DIAGNOSTIC_H0_TABULAR_SWEEP_PROTOCOL_2026-06-04.md` Section 13. |
| Diagnostic H0 optional Round 2 | 45 | conditional planned | Count only if Round 2 trigger ran; Section 13 canonical grid. |
| Notebook 03 model-family screen | 45 | exact planned in current generator | `scripts/create_model_family_screening_colab_notebook.py` fixes 1 candidate x 9 models x 5 seeds. |
| Notebook 04 controlled follow-up | 30 | exact planned | 6 models x 5 fresh seeds, including 20 fitted real-model rows and 10 dummy rows. |

Current visible cumulative validation search width before Notebook 04 is
approximately 372-387 model-seed rows without H0 optional Round 2, or
417-432 if H0 optional Round 2 ran. Including Notebook 04, the planned visible
width becomes approximately 402-417 without H0 optional Round 2, or 447-462 if
H0 optional Round 2 ran. These totals are budgeting context, not a new gate;
before any future holdout/test authorization, replace planned ranges with
artifact-verified actual counts and explain any de-duplicated or failed rows.

## 9. Run Switches

All expensive switches are `False` by default in the shared notebook:

```python
INSTALL_LIGHTGBM_IF_MISSING = False
RUN_04S_SCHEMA_SMOKE = False
RUN_04A_READ_CONTEXT = False
RUN_04B_FRESH_SEED_PANEL = False
RUN_04C_SELECTIVE_COVERAGE = False
RUN_04D_GATE_DECISION = False
RUN_04E_BOOTSTRAP_CI = False
BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE = False
```

The recommended run order is:

1. `RUN_04S_SCHEMA_SMOKE = True`
2. `RUN_04A_READ_CONTEXT = True`
3. `RUN_04B_FRESH_SEED_PANEL = True`
4. `RUN_04C_SELECTIVE_COVERAGE = True`
5. `RUN_04D_GATE_DECISION = True`
6. Optional: `RUN_04E_BOOTSTRAP_CI = True`

The first real fitted seed in 04B counts as part of the official 04B budget.
Do not run hidden exploratory seed retries before the official panel.

Dependency handling:

- The notebook may check whether `lightgbm` is importable.
- It must not silently install packages.
- If `lightgbm` is missing, stop with an explicit message unless
  `INSTALL_LIGHTGBM_IF_MISSING = True`.
- Torch should use the Colab runtime-provided installation. Do not install a
  different torch build inside Notebook 04.

Colab resource boundary:

- 04S and 04A can run on CPU.
- 04B should be run on a T4 GPU runtime because it includes
  `standalone_tcn` and `ms_dlinear_tcn`.
- If only the tabular rows complete, the output is a partial diagnostic, not a
  completed Notebook 04 gate.
- 04C and 04D may run on CPU after 04B artifacts exist.
- If the runtime disconnects before all 04B rows finish, rerun from setup and
  context cells, then resume only from complete saved artifacts. Do not merge
  duplicate or shape-mismatched prediction files.

## 10. 04S - Schema Smoke

04S is a no-selection smoke. It may use tiny synthetic arrays to verify:

- tabular model interface shape;
- torch model forward-pass shape;
- prediction artifact writer schema;
- selective coverage function shape;
- result table columns.

04S must not load real validation results, fit real data, or write selection
decisions.

## 11. 04A - Read-Only Context Check

04A reads the Notebook 03 selection JSON and optional diagnostic H0 files.

Required assertions:

1. `scope == "validation_only"`.
2. `holdout_test_authorized is False`.
3. Official Stage 0 candidate equals:

   ```text
   h03_bps1p5 + price_volume_time + window_size=20
   ```

4. Any H0 window appendix is marked diagnostic-only and non-selecting.
5. The Notebook 04 model panel is exactly the panel in Section 6.
6. The Notebook 04 seeds are exactly the seeds in Section 8.

04A writes:

```text
notebook04_context_checks.json
```

04A does not fit models.

## 12. 04B - Fresh-Seed Confirmation Panel

04B fits the fixed model panel on the fixed official candidate using fresh
seeds.

For every `(model, seed)` row, report:

- `stage`
- `candidate_id`
- `model`
- `seed`
- `label_config`
- `feature_set`
- `window_size`
- `scope`
- `macro_f1`
- `balanced_accuracy`
- `accuracy`
- `delta_macro_f1_vs_stratified_dummy`
- `delta_macro_f1_vs_always_up_dummy`
- `train_n`
- `validation_n`
- `run_failed`
- `failure_reason`
- `fit_status`
- `fit_seconds`
- `predict_seconds`

Per-ticker rows must report the same core metrics by ticker.

04B writes:

```text
notebook04_pooled.csv
notebook04_per_ticker.csv
notebook04_summary.csv
notebook04_prediction_manifest.csv
notebook04_run_manifest.json
predictions/*.npz
```

## 13. Fresh-Seed Confirmation Gates

Notebook 04 does not choose a model only by mean macro F1. A real-model branch
must satisfy all basic gates:

```text
run_failed = False for all five seeds
macro_f1_lcb_95 > stratified_dummy_macro_f1_mean
delta_macro_f1_vs_stratified_dummy_mean > 0
positive_ticker_count >= 3
top_ticker_gain_share <= 0.50
```

Fresh-seed stability is interpreted against the Notebook 03 same-model summary:

```text
fresh_minus_03 = notebook04_macro_f1_mean - notebook03_macro_f1_mean
```

Stability tags:

```text
fresh_minus_03 >= -0.001:
  confirmed_or_improved

-0.003 < fresh_minus_03 < -0.001:
  marginal_drop_note_only

fresh_minus_03 <= -0.003:
  failed_fresh_seed_confirmation
```

Positive shifts are not automatically stronger evidence. If
`fresh_minus_03 > 0.003`, the notebook should add
`positive_shift_review = True` and remind the reader that validation-only
search variance remains.

## 14. Prediction Artifact Schema

04B must persist per-sample validation prediction artifacts for every
`(model, seed)` row.

Path convention:

```text
/content/notebook04_controlled_followup_results/predictions/{model}__seed{seed}.npz
```

Required arrays:

```text
validation_sample_id
ticker
timestamp
y_true
y_pred
prob_up
confidence
```

Requirements:

1. `validation_sample_id` must be stable across models and seeds for the same
   candidate.
2. `prob_up` is `predict_proba[:, 1]` for tabular probability models.
3. `prob_up` is `softmax(logits)[:, 1]` for torch models.
4. `confidence = max(prob_up, 1 - prob_up)`.
5. `confidence` must be within `[0.5, 1.0]`.
6. `prob_up` must be within `[0.0, 1.0]`.
7. Dummy baselines should save `y_pred` on the same sample ids. If dummy
   `prob_up` is not meaningful, the manifest must set
   `selective_eligible = False`.
8. Artifacts must not save raw OHLCV rows, train features, validation feature
   tensors, or holdout/test data.

If a saved prediction artifact is missing, misaligned, contains NaNs, or has an
unexpected row count, Notebook 04 must raise an error before 04C.

## 15. 04C - Selective Coverage Diagnostic

04C reads only 04B prediction artifacts. It must not fit models.

Selective coverage is within-model only. Do not rank LightGBM confidence
against torch confidence as if the scales were calibrated and comparable.

Eligible models:

```text
logreg
lightgbm
standalone_tcn
ms_dlinear_tcn
```

Coverage grid:

```text
1.00, 0.80, 0.60, 0.40, 0.20, 0.10
```

For each `(model, seed, coverage)`:

1. Sort validation samples by `confidence` descending.
2. Break ties by `validation_sample_id` ascending.
3. Retain:

   ```text
   retained_n = max(1, ceil(validation_n * coverage))
   ```

4. Evaluate only retained rows.

Metrics:

- `retained_n`
- `retained_pct`
- `class0_n`
- `class1_n`
- `pred0_n`
- `pred1_n`
- `macro_f1`
- `balanced_accuracy`
- `accuracy`
- `selective_error = 1 - accuracy`
- `precision_down`
- `precision_up`
- `auc` only when both true classes remain and `prob_up` is available
- `delta_macro_f1_vs_stratified_dummy_same_rows`
- `delta_macro_f1_vs_always_up_dummy_same_rows`
- `max_ticker_retained_share`
- `min_ticker_retained_n`

Dummy baselines for retained subsets must be evaluated on the exact retained
sample ids. Do not compare a retained model subset against a full-coverage
dummy score.

Warnings:

```text
coverage <= 0.20:
  low_coverage_exploratory

min_ticker_retained_n < 500:
  per_ticker_retained_n_low

max_ticker_retained_share > 0.40:
  ticker_concentration_warning

only one true class remains:
  auc_not_defined
```

04C writes:

```text
notebook04_selective_coverage.csv
```

## 16. 04D - Manual Gate Decision

04D reads:

```text
notebook04_context_checks.json
notebook04_summary.csv
notebook04_selective_coverage.csv
```

It creates a decision matrix, but it must not auto-authorize the next notebook.
The operator must read and accept the decision matrix.

Allowed next-step exits:

### Exit A - Proceed To 05 LightGBM Tuning

Use only if:

1. `lightgbm` passes all 04B basic gates.
2. `lightgbm` is tagged `confirmed_or_improved` or `marginal_drop_note_only`.
3. 04C shows no severe ticker concentration at 40% and 20% coverage.
4. The selective profile is not worse than full coverage by more than
   `0.005 macro_f1` at both 40% and 20% coverage.

Notebook 05 would then be a bounded LightGBM tuning notebook.

### Exit B - Proceed To 05 MS-DLinear+TCN Design Review

Use only if:

1. `ms_dlinear_tcn` passes all 04B basic gates.
2. `ms_dlinear_tcn` is not tagged `failed_fresh_seed_confirmation`.
3. It is not clearly dominated by `standalone_tcn` at full coverage.
4. The design-review question is frozen before any architecture edit.
5. The design-review question and hypothesized failure mode cite a pre-04
   source, such as Ian email date, a KB note path, or a prior protocol section.
   They must not be invented post-hoc from fresh-seed 04B results.

Notebook 05 would then be a design-review or ablation notebook, not an
unrestricted tuning notebook.

### Exit C - Stop Modeling And Write Weak-Signal Result

Use if:

1. No real model passes the 04B basic gates, or
2. fresh-seed confirmation fails for all candidate branches, or
3. selective gains appear only at low coverage with ticker concentration or
   too few retained samples.

This exit treats a weak or unstable result as a valid research outcome.

### Exit D - Inconclusive, Pre-Register One New Diagnostic

Use if the result is mixed and cannot justify Exit A, B, or C.

The new diagnostic must be a separate pre-registered notebook or protocol. It
must name the exact question, model, metric, and stop rule before running.

04D writes:

```text
notebook04_decision_matrix.csv
```

## 17. Optional 04E - Bootstrap CI Appendix

04E is optional and off by default.

If enabled, it reads only 04B prediction artifacts and computes bootstrap
confidence intervals for full-coverage macro F1. It must not fit models,
change selection gates, or become the sole reason for a next-step exit.

Recommended defaults:

```text
bootstrap_resamples = 1000
bootstrap_seed = 260604
unit = validation sample row
reported_models = logreg, lightgbm, standalone_tcn, ms_dlinear_tcn
```

Because adjacent sliding-window samples are autocorrelated, row-level bootstrap
intervals are diagnostic only. They should be used as a variance warning, not
as formal independent-sample inference.

04E writes:

```text
notebook04_bootstrap_ci.csv
```

## 18. Output Directory

Notebook 04 writes local runtime outputs first:

```text
/content/notebook04_controlled_followup_results
```

Expected files:

```text
notebook04_context_checks.json
notebook04_pooled.csv
notebook04_per_ticker.csv
notebook04_summary.csv
notebook04_prediction_manifest.csv
notebook04_selective_coverage.csv
notebook04_decision_matrix.csv
notebook04_run_manifest.json
predictions/*.npz
```

Optional files:

```text
notebook04_bootstrap_ci.csv
```

Google Drive backup is allowed only when:

```python
BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE = True
```

Local runtime output remains the authoritative immediate output. Drive backup
is persistence, not a second source of truth.

## 19. Failure Behavior

Notebook 04 must fail fast if:

- required Notebook 03 context files are missing;
- the official candidate is not the fixed Stage 0 candidate;
- `holdout_test_authorized` is not explicitly false;
- any H0 appendix participates in selection;
- prediction artifacts are missing or misaligned;
- probability or confidence values are outside valid bounds;
- 04C or 04D attempts to refit a model;
- any code path materializes holdout/test rows.

Do not catch and ignore these failures. Do not silently substitute a different
candidate, seed, model, or artifact.

## 20. Interpretation Template

Allowed wording:

```text
Notebook 04 provides validation-only fresh-seed confirmation and within-model
selective-coverage diagnostics for the fixed official Stage 0 candidate. The
result does not authorize holdout/test evaluation.
```

If Exit A is selected:

```text
The LightGBM branch is eligible for a bounded Notebook 05 tuning protocol under
validation-only constraints.
```

If Exit B is selected:

```text
The MS-DLinear+TCN branch is eligible for a bounded design-review protocol, not
an unrestricted architecture search.
```

If Exit C is selected:

```text
Under the current fixed candidate and model panel, the validation-only evidence
does not justify further modeling expansion before writing the weak-signal
result.
```

Forbidden wording:

```text
The best model is ready for holdout.
The model is profitable.
The high-confidence subset is tradable.
Window 32 replaces the official Stage 0 candidate.
Notebook 04 tuned the final model.
The selective curve proves one model is globally better than another.
```

## 21. Notebook 04 Debug Checklist

Before running real 04B:

- 04S passes on synthetic data.
- 04A confirms the official candidate and `holdout_test_authorized = false`.
- all `RUN_*` switches except the intended section are false.
- output directory is empty or intentionally versioned.

After 04B:

- every model has five seed rows or an explicit failure row;
- dummy baselines exist on the same target rows;
- per-ticker rows exist for all five tickers;
- prediction manifest row count matches generated `.npz` files;
- every `.npz` row count matches the pooled validation target count;
- no probability, confidence, or label array contains NaNs;
- no holdout/test path appears in output manifests.

After 04C:

- every eligible model has the full coverage grid;
- retained rows are subsets of the saved 04B validation sample ids;
- retained dummy deltas use the same retained sample ids;
- low-coverage and ticker-concentration warnings are visible.

After 04D:

- the decision matrix lists all four exits;
- exactly one exit is manually marked as the proposed next step;
- the matrix states that holdout/test remains closed.

## 22. Acceptance Criteria For The Notebook Builder

When Notebook 04 is created, the generator must validate:

1. The notebook parses with `nbformat`.
2. All code cells AST-parse.
3. All outputs are empty.
4. All execution counts are `None`.
5. Default heavy `RUN_*` switches are false.
6. Forbidden active-code strings are absent unless explicitly approved:

   ```text
   from intraday_research
   baseline_helpers
   train_test_split
   drive.mount(
   ```

7. The notebook contains no code that reads, scores, or displays holdout/test
   rows.

## 23. Bottom Line

Notebook 04 is a confirmation and diagnostic notebook. It is not the tuning
notebook yet.

The main technical additions over Notebook 03 are:

- fresh unused seeds;
- a smaller fixed follow-up panel;
- per-sample prediction persistence;
- within-model selective coverage diagnostics;
- a manual gate that decides whether Notebook 05 should tune LightGBM, review
  MS-DLinear+TCN design, stop modeling, or pre-register one more diagnostic.
