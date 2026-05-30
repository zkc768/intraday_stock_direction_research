# Phase 1B Full-Run Analysis — 2026-05-24

## Scope

This analysis summarizes the completed local Phase 1B diagnostic matrix. It is
not the canonical Phase 1 baseline.

```text
run_dir = checkpoints/phase1b_local_baseline_full/phase1b_local_full_20260524_202110
feature_set_id = technical_v1
candidate_id = A
window_size = 12
label_horizon_k = 12
threshold_bps = 5
models = lstm, tcn, dlinear
seeds = 42, 43, 44
training_scope = pooled
evaluation_scope = pooled + per ticker
max_epochs = 3
batch_size = 512
```

Result files:

```text
manifest.csv
metadata.json
results.csv
summary_pooled_by_model.csv
summary_by_model_ticker.csv
```

## Data And Coverage

The Phase 1B diagnostic no-trade-band subset retains only a minority of rows.
This is expected and important: these results describe a retained high-magnitude
subset, not the canonical full-market direction prediction task.

| ticker | retained_pct | train_windows | test_windows | train_up_pct | test_up_pct |
|---|---:|---:|---:|---:|---:|
| CSCO | 0.196988 | 58085 | 4692 | 0.478247 | 0.447357 |
| JPM | 0.182963 | 53085 | 4560 | 0.497561 | 0.450658 |
| KO | 0.093391 | 26575 | 2239 | 0.510593 | 0.449308 |
| MSFT | 0.153963 | 42126 | 4854 | 0.483454 | 0.439431 |
| WMT | 0.118102 | 33394 | 2837 | 0.504162 | 0.484314 |
| pooled | n/a | 213265 | 19182 | 0.492172 | 0.451830 |

The low retained coverage is not a defect, but it must be disclosed in any
paper or presentation. KO is the narrowest retained subset, so KO-only gains
should be treated especially cautiously.

## Pooled Results

Primary comparison is `delta_macro_f1_vs_dummy`, where dummy is the 10-run
stratified baseline fitted on training labels and evaluated on test labels.

| model | macro_f1_mean | delta_mean | delta_std | 95% CI for delta | balanced_accuracy_mean |
|---|---:|---:|---:|---:|---:|
| dlinear | 0.483400 | -0.014646 | 0.045739 | [-0.128268, 0.098976] | 0.518245 |
| lstm | 0.499939 | 0.001893 | 0.028273 | [-0.068342, 0.072127] | 0.522107 |
| tcn | 0.494158 | -0.003888 | 0.033868 | [-0.088021, 0.080245] | 0.526212 |

Interpretation:

- No pooled model meets the current continuation threshold:

```text
mean delta_macro_f1_vs_dummy >= +0.01
```

- All 95% intervals are wide and include zero. With only three seeds, these
  intervals are descriptive rather than confirmatory.
- Balanced accuracy is slightly above 0.50 for all models, but macro F1 does
  not robustly beat the stratified dummy baseline.

## Ticker Pattern

Mean `delta_macro_f1_vs_dummy` by ticker:

| model | CSCO | JPM | KO | MSFT | WMT |
|---|---:|---:|---:|---:|---:|
| dlinear | -0.004296 | -0.017273 | 0.013207 | -0.028524 | -0.054037 |
| lstm | 0.025522 | -0.014103 | 0.043336 | -0.046033 | -0.065756 |
| tcn | 0.028544 | -0.021911 | 0.046269 | -0.044705 | -0.080091 |

Positive seed-level deltas across the 15 per-ticker rows:

```text
dlinear: 7 / 15
lstm:    8 / 15
tcn:     8 / 15
```

Interpretation:

- CSCO and KO carry most of the positive evidence.
- JPM, MSFT, and WMT are negative on average across the three model families.
- The result is ticker-local rather than broad market-direction evidence.

## Confusion Matrix Diagnosis

Average pooled confusion matrices across the three seeds:

```text
dlinear: [[5739.7, 4775.3],
          [4414.7, 4252.3]]

lstm:    [[6963.3, 3551.7],
          [5356.3, 3310.7]]

tcn:     [[8426.7, 2088.3],
          [6491.3, 2175.7]]
```

Predicted-up rate compared with true-up rate:

```text
true_up_pct = 0.4518
dlinear_pred_up_pct = 0.4706
lstm_pred_up_pct    = 0.3577
tcn_pred_up_pct     = 0.2223
```

Interpretation:

- TCN is strongly biased toward class 0 on pooled test.
- LSTM is also conservative, but less extreme.
- DLinear predicts class balance closest to the test distribution, but its
  macro F1 remains unstable and negative on average.

## Baselines

Pooled baselines were constant across model rows:

```text
dummy_stratified_macro_f1_mean = 0.498046
dummy_prior_macro_f1           = 0.354076
always_up_macro_f1             = 0.311214
always_down_macro_f1           = 0.354076
```

The stratified dummy baseline is the hard baseline here. Beating only
`always_up`, `always_down`, or `dummy_prior` is not enough.

## Decision

Do not add PatchTST yet.

The local harness is now useful, but the current result does not justify a
larger model. Because this run used the Phase 1B diagnostic no-trade-band subset
and a dirty working tree, it should not be treated as the canonical Phase 1
baseline. The immediate research framing should be:

```text
Under the current leakage-controlled local harness, high-frequency stock
direction prediction behaves like a weak-signal problem; model complexity is
less important than protocol, coverage disclosure, and robust baselines.
```

## Recommended Next Step

Run protocol diagnostics before adding models:

1. Check whether the positive CSCO/KO pattern survives longer training.
2. Add a shuffled-label sanity run to Notebook 02 or an explicit local
   diagnostic notebook.
3. Compare no-trade-band coverage and class balance across thresholds before
   choosing another candidate.
4. Keep PatchTST blocked until the existing three baselines show a robust
   pooled or multi-ticker signal.

The project should continue as an evaluation harness and weak-signal study, not
as a model-stacking project.

## Post-Run Gate Updates

After the initial full-run analysis, the project control workflow split the next
work into runner, review, and code-control agents.

### CSCO/KO Longer-Training Diagnostic

A post-hoc diagnostic was run on the two tickers that looked most positive in
the full-run:

```text
run_dir = checkpoints/phase1b_local_diagnostics_csco_ko_epoch10/phase1b_local_full_20260524_203610
tickers = CSCO, KO
models = lstm, tcn, dlinear
seeds = 42, 43, 44
max_epochs = 10
```

Pooled CSCO/KO summary:

| model | delta_mean | delta_std | macro_f1_mean |
|---|---:|---:|---:|
| dlinear | -0.029398 | 0.048260 | 0.465141 |
| lstm | 0.029570 | 0.009614 | 0.524108 |
| tcn | -0.005217 | 0.032947 | 0.489322 |

Interpretation:

- LSTM remains positive on the post-hoc CSCO/KO subset.
- This is a hypothesis-generating diagnostic only, because the tickers were
  selected after seeing the five-ticker full-run.
- It does not justify a claim of stable or generalizable alpha.

### Runner Baseline Patch

The local runner now reports both pooled-train and per-ticker-train baseline
comparisons:

```text
delta_macro_f1_vs_dummy          = comparison against pooled-train dummy
delta_macro_f1_vs_ticker_dummy   = comparison against per-ticker-train dummy
```

The runner also has a `--shuffle-train-labels` diagnostic flag. Default behavior
remains unchanged.

### Shuffled-Label Sanity Smoke

A small shuffled-label sanity run completed:

```text
run_dir = checkpoints/phase1b_local_shuffle_sanity_smoke/phase1b_local_smoke_20260524_205841
tickers = CSCO, KO
model = lstm
seeds = 42, 43, 44
max_rows_per_ticker = 5000
max_epochs = 3
shuffle_train_labels = true
```

Summary:

```text
pooled model_macro_f1_mean = 0.4600
pooled delta vs dummy mean = -0.0227
suspicious_status rows     = 0
```

Interpretation: the diagnostic entry point behaves as expected. It does not
show a stable positive signal after shuffling training labels.

### Dataset Gate

The dataset implementation and window-boundary tests were strengthened after
review:

- invalid labels now raise with context/ticker/label/row/value;
- duplicate timestamps raise with offending rows;
- out-of-order timestamps raise with offending row/index and previous/current
  timestamps;
- tests cover multi-ticker trim grouping, cross-day label horizons, cross-day
  input windows, invalid label values, duplicate timestamps, and out-of-order
  timestamps.

Final dataset gate review verdict:

```text
ready
```

Validation:

```text
target dataset/window tests: 44 passed
full non-integration tests: 148 passed, 1 existing warning
```

The existing warning is from `tests/test_checkpoint.py` and is unrelated to the
dataset changes.

### Threshold Sensitivity Manifest Sweep

Before training any threshold grid, a manifest-only sweep was run for the
Phase 1B diagnostic no-trade-band subset:

```text
threshold_bps = 0, 5, 10
tickers = CSCO, JPM, KO, MSFT, WMT
feature_set_id = technical_v1
output = checkpoints/phase1b_local_threshold_sensitivity_manifest/threshold_sensitivity_summary.csv
```

Retained coverage and test-window counts by threshold:

| threshold_bps | CSCO retained | JPM retained | KO retained | MSFT retained | WMT retained |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8475 | 0.8473 | 0.8471 | 0.8475 | 0.8471 |
| 5 | 0.1970 | 0.1830 | 0.0934 | 0.1540 | 0.1181 |
| 10 | 0.0599 | 0.0563 | 0.0190 | 0.0381 | 0.0273 |

| threshold_bps | CSCO test windows | JPM test windows | KO test windows | MSFT test windows | WMT test windows |
|---:|---:|---:|---:|---:|---:|
| 0 | 47135 | 47085 | 46955 | 47167 | 46991 |
| 5 | 4692 | 4560 | 2239 | 4854 | 2837 |
| 10 | 985 | 1041 | 543 | 1230 | 515 |

Interpretation:

- `0 bps` preserves broad coverage and is the closest no-trade-band diagnostic
  analog to the canonical full binary task, but it is still emitted through the
  Phase 1B diagnostic runner.
- `5 bps` creates a high-magnitude subset with substantial coverage loss. It is
  usable for diagnostic experiments only if coverage is disclosed.
- `10 bps` is too narrow for the current five-stock setup. KO and WMT test
  windows fall near 500, which makes seed/ticker conclusions fragile.

Manager decision:

```text
Do not train a 0/5/10 threshold grid yet.
Use this sweep to choose diagnostic regimes and to document coverage tradeoffs.
If a table-of-record rerun is needed next, prioritize 0 bps and 5 bps; treat
10 bps as low-coverage exploratory only.
```

### Table-Of-Record Diagnostic Reruns

After the runner provenance patch and review pass, two bounded reruns completed.
Both runs are diagnostics, not canonical Phase 1 baselines.

```text
5bps run_dir = checkpoints/phase1b_local_table_record_5bps/phase1b_local_full_20260524_215040
0bps run_dir = checkpoints/phase1b_local_table_record_0bps/phase1b_local_full_20260524_220040
rows per run = 54
manifest rows per run = 6
feature_set_id = technical_v1
models = lstm, tcn, dlinear
seeds = 42, 43, 44
max_epochs = 3
batch_size = 512
suspicious rows = 0
```

The `5bps` run is the current Phase 1B high-magnitude no-trade-band
table-of-record. It reproduces the earlier full-run values while adding
row-level provenance and per-ticker dummy comparisons.

| model | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | delta_vs_pooled_dummy_mean | delta_vs_ticker_dummy_mean |
|---|---:|---:|---:|---:|---:|
| lstm | 0.499939 | 0.028273 | 0.522107 | 0.001893 | 0.001893 |
| tcn | 0.494158 | 0.033868 | 0.526212 | -0.003888 | -0.003888 |
| dlinear | 0.483400 | 0.045739 | 0.518245 | -0.014646 | -0.014646 |

Per-ticker `delta_macro_f1_vs_ticker_dummy` for the 5bps run:

| model | CSCO | JPM | KO | MSFT | WMT |
|---|---:|---:|---:|---:|---:|
| dlinear | -0.004641 | -0.017399 | 0.008065 | -0.029013 | -0.052495 |
| lstm | 0.025177 | -0.014229 | 0.038194 | -0.046522 | -0.064214 |
| tcn | 0.028199 | -0.022037 | 0.041127 | -0.045194 | -0.078549 |

The `0bps` run is a strict-sign no-trade-band diagnostic analog, not the
canonical Phase 1 full binary label. It drops exact-zero `future_avg_r` rows as
neutral/NaN, while canonical Phase 1 keeps `future_avg_r == 0` as class 0
`non_up`. The exact-zero count is small here (`13 / 2,217,742` rows), but the
semantic difference must be disclosed.

Coverage for the 0bps diagnostic analog:

| ticker | retained_pct | retained | neutral_exact_zero | cross_day | tail | train_windows | test_windows | train_up_pct | test_up_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CSCO | 0.847482 | 376365 | 5 | 67716 | 12 | 219993 | 47135 | 0.511680 | 0.538750 |
| JPM | 0.847301 | 375746 | 0 | 67704 | 12 | 219527 | 47085 | 0.523735 | 0.509207 |
| KO | 0.847073 | 375189 | 7 | 67716 | 12 | 219193 | 46955 | 0.531390 | 0.524481 |
| MSFT | 0.847539 | 376442 | 1 | 67704 | 12 | 220014 | 47167 | 0.516531 | 0.523947 |
| WMT | 0.847149 | 375371 | 0 | 67716 | 12 | 219342 | 46991 | 0.524008 | 0.515886 |
| pooled | 0.847309 | 1879113 | 13 | 338556 | 60 | 1098069 | 235333 | 0.521459 | 0.522460 |

Pooled 0bps diagnostic results:

| model | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | delta_vs_pooled_dummy_mean | delta_vs_ticker_dummy_mean |
|---|---:|---:|---:|---:|---:|
| dlinear | 0.480098 | 0.025195 | 0.509847 | -0.019760 | -0.019760 |
| lstm | 0.482813 | 0.027825 | 0.507818 | -0.017044 | -0.017044 |
| tcn | 0.504747 | 0.007611 | 0.511862 | 0.004890 | 0.004890 |

Per-ticker `delta_macro_f1_vs_ticker_dummy` for the 0bps run:

| model | CSCO | JPM | KO | MSFT | WMT |
|---|---:|---:|---:|---:|---:|
| dlinear | -0.005461 | -0.072179 | -0.009955 | -0.067879 | -0.067856 |
| lstm | -0.012242 | -0.032529 | 0.000667 | -0.029304 | -0.034254 |
| tcn | -0.006726 | -0.063177 | -0.013451 | -0.046467 | -0.036164 |

Average pooled 0bps confusion matrices across the three seeds:

```text
dlinear: [[61704.3, 50676.7],
          [65087.0, 57865.0]]

lstm:    [[76891.7, 35489.3],
          [82202.0, 40750.0]]

tcn:     [[65442.7, 46938.3],
          [68681.7, 54270.3]]
```

Interpretation:

- The 5bps table-of-record still does not meet the continuation threshold of
  `mean delta_macro_f1_vs_dummy >= +0.01`.
- The 0bps diagnostic analog gives TCN a small pooled positive delta
  (`+0.004890`), but it is below the continuation threshold and does not survive
  per-ticker dummy comparison.
- DLinear and LSTM are negative on the 0bps pooled diagnostic analog.
- These results strengthen the decision to keep PatchTST/new-model work
  blocked. They motivated the canonical binary run reported in the next
  section.

## Canonical Phase 1 Full-Binary Baseline

The canonical Phase 1 full-binary runner now exists and completed a five-ticker
full-run. This is the first table in this document that uses the AGENTS-defined
canonical label:

```text
label_mode = legacy_binary
label_semantics = canonical_phase1_full_binary
label_formula = label = 1 if future_avg_r > 0 else 0
zero_return_policy = class_0_non_up
no_trade_band_enabled = false
run_dir = checkpoints/phase1_canonical_binary_full/phase1b_local_legacy_binary_full_20260524_230605
feature_set_id = technical_v1
models = lstm, tcn, dlinear
seeds = 42, 43, 44
max_epochs = 3
batch_size = 512
rows = 54
suspicious rows = 0
```

Coverage:

| ticker | retained_pct | retained | neutral | cross_day | tail | zero_return | train_windows | test_windows | train_up_pct | test_up_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CSCO | 0.847493 | 376370 | 0 | 67716 | 12 | 5 | 219995 | 47135 | 0.511675 | 0.538750 |
| JPM | 0.847301 | 375746 | 0 | 67704 | 12 | 0 | 219527 | 47085 | 0.523735 | 0.509207 |
| KO | 0.847089 | 375196 | 0 | 67716 | 12 | 7 | 219196 | 46955 | 0.531383 | 0.524481 |
| MSFT | 0.847541 | 376443 | 0 | 67704 | 12 | 1 | 220015 | 47167 | 0.516528 | 0.523947 |
| WMT | 0.847149 | 375371 | 0 | 67716 | 12 | 0 | 219342 | 46991 | 0.524008 | 0.515886 |
| pooled | 0.847315 | 1879126 | 0 | 338556 | 60 | 13 | 1098075 | 235333 | 0.521456 | 0.522460 |

The `zero_return` rows are retained as class 0 in this canonical run. This is
the semantic difference from the 0bps no-trade-band diagnostic analog, which
drops exact-zero rows as neutral/NaN.

Pooled multi-seed results:

| model | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | delta_vs_dummy_mean | delta_vs_dummy_std |
|---|---:|---:|---:|---:|---:|
| dlinear | 0.476766 | 0.002797 | 0.508643 | -0.023092 | 0.002797 |
| lstm | 0.484279 | 0.044964 | 0.508385 | -0.015578 | 0.044964 |
| tcn | 0.497540 | 0.017373 | 0.509389 | -0.002318 | 0.017373 |

Per-ticker `delta_macro_f1_vs_ticker_dummy`:

| model | CSCO | JPM | KO | MSFT | WMT |
|---|---:|---:|---:|---:|---:|
| dlinear | -0.057389 | -0.011735 | -0.074298 | -0.002258 | -0.015578 |
| lstm | -0.034409 | -0.043262 | -0.054667 | -0.042457 | -0.043492 |
| tcn | -0.032842 | -0.016494 | -0.046671 | -0.007508 | -0.007952 |

Average pooled confusion matrices across the three seeds:

```text
dlinear: [[27214.7, 85166.3],
          [27649.3, 95302.7]]

lstm:    [[40556.0, 71825.0],
          [42309.0, 80643.0]]

tcn:     [[42487.0, 69894.0],
          [44174.7, 78777.3]]
```

Interpretation:

- None of LSTM, TCN, or DLinear beats the stratified dummy baseline on the
  canonical pooled table.
- TCN is closest to dummy (`delta = -0.002318`), but still negative.
- Every model/ticker mean delta versus per-ticker dummy is negative.
- The result closes the current model-expansion gate: do not add PatchTST,
  attention, NLP, or a copied external repository based on this evidence.
- The project remains worth continuing as a leakage-safe evaluation harness and
  weak-signal study. The next research work should be protocol analysis,
  reporting, and possibly simpler/non-sequence baselines before larger models.

Validation completed around this run:

```text
runner py_compile: passed
runner semantic tests + label/config/window tests: 74 passed
full non-integration tests: 154 passed, 1 existing checkpoint warning
5-ticker canonical smoke: completed, 18 rows, suspicious rows = 0
canonical full-run: completed, 54 rows, suspicious rows = 0
```
