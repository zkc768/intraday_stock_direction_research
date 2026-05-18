# Phase 1B Plan — Library-First Approach

Status: Steps 1-2 complete; Step 3 capacity profiling next
Date: 2026-05-17
Depends on: ml_utils_construction_plan_v2.md (§5.7, §5.8, §6 steps 10-12)
Continues from: MVP complete plus Phase 1B label/config commits

---

## 0. Core Principle: Build the Library, Then Run Experiments

Phase 1B follows the same workflow as the MVP: test-first development in ml_utils (PyTorch, local), then experiment in notebooks. The Colab TF/Keras notebooks (binary_clf_comparison, Untitled2) were exploratory prototypes. They are reference material, not the production path.

The ml_utils construction plan v2 §6 already defines steps 10-12:
- Step 10: models/tcn_classifier.py
- Step 11: models/dlinear_classifier.py
- Step 12: Notebook 03 — three-model comparison

Phase 1B extends this with Ian's new requirements: no-trade-band labels, multi-scale DLinear, and residual TCN branch. These get built into the library first, tested, then used in notebooks.

Current implementation status:
- Step 1 no-trade-band labels: complete in `c7162c2 feat(dataset): add no-trade-band labels`
- Step 2 config extension: complete in `d9352fc feat(config): add Phase 1B label fields`
- Latest validation: 107 passed, 1 known pre-existing scheduler-order warning
- Next step: Step 3 capacity profiling
- TCN, DLinear, improved model, and Notebook 03 are still absent

---

## 1. What Ian Asked For

1. Improve label definition: use a real no-trade band, remove small-return samples, classify remaining as Up/Down. Try a few threshold levels, report results.
2. Keep LSTM, TCN, and standard DLinear as baselines.
3. Build a stock-aware multi-scale DLinear model with a residual TCN branch.

## 2. Existing Settled Decisions (carry forward)

From ml_utils_construction_plan_v2.md and MVP:
- Framework: PyTorch, ml_utils library
- Tickers: CSCO, JPM, KO, MSFT, WMT
- Bar frequency: 5-minute
- Features: parameterized via config, not hardcoded
- Label basis: future k-bar average return
- Split: chronological per ticker, then pooled
- Scaler: StandardScaler, fit on pooled train only
- Windowing: WindowedClassificationDataset with stride, same-day + consecutive checks
- Test-first: write tests, get approval, then implement
- One module per agent session, self-review in fresh session

From hf_stock_clf_after_w7_memory.md:
- MVP modules complete: config, seed, metrics, dataset, checkpoint, lstm_classifier, trainer
- Current non-integration suite: 107 tests passing, with 1 known pre-existing scheduler-order warning
- Notebooks 01/02 smoke tests passed
- Phase 1B Steps 1-2 complete; TCN/DLinear files and Notebook 03 absent

## 3. Data Location

Data source: raw 1-min txt files on Google Drive at /content/drive/MyDrive/stockdata/.

For local ml_utils development and profiling, one of two options must be chosen:
- Option A: Download processed 5-min CSV data to local data/ directory. Allows local pytest and profiling scripts.
- Option B: Keep data on Drive only. Profiling runs in a Colab notebook that imports ml_utils (synced via Drive or pip install from local).

This must be resolved in Step 0 (preflight).

## 4. Label Upgrade: No-Trade Band Binary

### 4.1 New Label Semantics

MVP legacy (unchanged, kept for backward compatibility):
- class 0 = non_up (return <= 0), class 1 = up (return > 0)
- exact-zero `future_avg_r` maps to label 0.0
- last k rows are NaN
- legacy function does not handle cross-day internally

Phase 1B no-trade band binary:
- class 0 = down (return < -threshold)
- class 1 = up (return > +threshold)
- neutral = |future_avg_r| <= threshold → NaN, window skipped
- exact zero is neutral when threshold_bps=0

The old function make_binary_labels_from_future_avg_return is NOT modified.

### 4.2 New Function in dataset.py

```python
def make_no_trade_band_labels(
    df: pd.DataFrame,
    price_col: str,
    k: int,
    threshold_bps: float,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]
```

- Uses the same `future_avg_r` computation as the existing function
- threshold = threshold_bps / 10_000
- return > +threshold → label 1 (up)
- return < -threshold → label 0 (down)
- |return| <= threshold → label NaN (neutral, skipped by WindowedDataset)
- optional timestamp_col performs cross-day invalidation using `.dt.date`
- timestamp_col=None disables cross-day filtering
- last k tail rows remain NaN
- Returns df with new columns: `future_avg_r`, `label`
- Returns diagnostics dict: `n_total`, `n_tail`, `n_cross_day`, `n_neutral`, `n_up`, `n_down`

Sanity check: at threshold_bps=0, Up+Down counts match existing function on nonzero same-day returns; exact-zero handling differs because no-trade-band treats exact zero as neutral.

### 4.3 Config Extension

Implemented on `DataConfig`:

```python
label_mode: str = "legacy_binary"
threshold_bps: float = 0.0
```

Allowed label_mode values are exactly `"legacy_binary"` and `"no_trade_band"`. No unused fields for future modes.

### 4.4 Selection Bias Disclosure

This experiment estimates P(sign(r) | X, |r| > τ), not P(sign(r) | X). The classifier covers non-trivial future movements only. Retained-subset metrics are not full-market deployment performance. Must report coverage alongside performance.

Three justifiable framings: noise reduction, no-action semantics, conditional directional classification.

## 5. Window Size and Horizon

### 5.1 Physical Constraints

78 five-minute bars per standard US trading day (theoretical max). Actual bars may be fewer on early-close days.

Hard rule: if window_size + k > P5(actual bars_per_day), mark INFEASIBLE.

Known infeasible: window=60+k=24=84>78, window=78+k=12=90>78.

### 5.2 Must Profile Before Locking

| window | k  | w+k | MA kernels if chosen |
|--------|----|-----|---------------------|
| 12     | 12 | 24  | single odd (5)      |
| 12     | 24 | 36  | single odd (5)      |
| 24     | 12 | 36  | [3, 5, 7]           |
| 24     | 24 | 48  | [3, 5, 7]           |
| 60     | 12 | 72  | see §5.3            |

### 5.3 Even-Kernel Implementation Constraint

LTSF-Linear, Autoformer, and TSLib moving_avg use symmetric (k-1)//2 endpoint padding. Only odd kernels preserve output length. Even kernels 6/12/24 cause L-1 shape mismatch.

FEDformer uses asymmetric padding that preserves length for even kernels.

For the standard DLinear baseline: MUST use a single odd kernel.
For multi-scale improved model with window=60: either implement FEDformer-style padding (to use [3,6,12,24]) or use all-odd kernels [3,7,13,25].

TSLib series_decomp_multi (simple averaging) and FEDformer series_decomp_multi (learnable softmax weights) are different algorithms despite sharing the same class name. The improved model should use FEDformer-style per-scale scalar softmax weights.

### 5.4 Feasibility Gates

Pooled PASS: train retained >= 5,000; val >= 1,000; test >= 1,000; train minority >= 30%.
Per-ticker WARNING: any ticker train < 800 or minority < 25%.
Per-ticker FAIL: any ticker train < 500 or minority < 20%.

## 6. Model Architectures

### 6.1 Baselines (rerun under new labels)

All baselines recomputed under no-trade-band labels. MVP results not comparable.

1. Stratified dummy, always-up, always-down (from ml_utils/metrics.py)
2. LSTM binary (existing ml_utils/models/lstm_classifier.py, retrained)
3. TCN binary (ml_utils/models/tcn_classifier.py — build in Phase 1B)
4. Standard DLinear binary (ml_utils/models/dlinear_classifier.py — build in Phase 1B)

### 6.2 Standard DLinear Classifier

Following construction plan v2 §5.8, updated with research report 3:
- Single odd kernel (default: 5, parameterized)
- Model-internal decomposition (in forward pass)
- individual=False by default (shared linear layers, original DLinear CLI default)
- Classification head: flatten [B, L*C] → Linear(L*C, num_classes)
- No multi-scale, no TCN branch, no stock-aware
- Writing: "TSLib-style DLinear adapted for classification"

### 6.3 Improved Model: Multi-Scale DLinear + Residual TCN

New file: ml_utils/models/ms_dlinear_tcn_classifier.py

- Multi-scale decomposition (model-internal) with FEDformer-style padding or all-odd kernels
- Per-scale scalar softmax fusion (FEDformer style, not simple averaging)
- Trend branch: DLinear-style linear projections per scale
- Residual branch: lightweight TCN (2 blocks, 32 channels, dilations [1,2], kernel=3, dropout=0.1)
- Fusion: concat(trend_features, residual_features) → Linear → num_classes
- Residual TCN must be smaller than standalone TCN baseline

### 6.4 Optional: Stock-Aware Component

Deferred until base improved model works.

## 7. Evaluation Protocol

Metrics: macro F1 (primary), balanced accuracy, delta vs dummy_stratified, confusion matrix.
Per threshold: retained_pct, Up/Down ratio, dummy baselines.

Profiling output tables (schemas in PHASE_1B_RESEARCH_HANDOFF.md):
- Table A: bars-per-day distribution
- Table B: threshold profile (pooled + per-ticker + effective independence)
- Table C: scaler diagnostic by feature group

## 8. Execution Steps — Library-First

### Step 0: Preflight (you, manually, ~10 min)

```
cd E:\codex_workspace\projects\hf_stock_clf
git status
git checkout -b phase-1b
dir data\
```

Confirm git clean, create branch, check if data is local.

### Step 1: No-trade-band label function (complete)

Following construction plan v2 §6.1 workflow:

Commit: `c7162c2 feat(dataset): add no-trade-band labels`

Session A1 — tests:
- File: tests/test_no_trade_band_labels.py
- threshold=0 matches legacy function
- threshold=10bps drops neutral correctly
- cross-day handling matches existing
- tail-row handling matches
- diagnostics dict correct

Session A2 — implement:
- Add to ml_utils/dataset.py (or new ml_utils/labels.py if dataset.py too large)
- Do not modify existing make_binary_labels_from_future_avg_return

Session A3 — self-review.

### Step 2: Config extension (complete)

Commit: `d9352fc feat(config): add Phase 1B label fields`

Add label_mode and threshold_bps to DataConfig.
Tests: config validates with new fields; old configs still work.

### Step 3: Capacity profiling (next, ~1 afternoon)

Create: scripts/phase1b_capacity_profile.py or notebooks/phase1b_capacity_profile.ipynb

Imports ml_utils. Computes:
- Bars-per-day distribution per ticker (min, P5, P10, median, max)
- Pre-check: window+k > P5 → INFEASIBLE
- For window=[12,24,60], k=[12,24], threshold=[0,5,10,15,20,30]:
  - Retained windows per ticker per split
  - Up/Down ratio, minority class %
  - Neutral drop %
  - Effective independent targets ≈ retained_bars / k
- Scaler diagnostic: retained-vs-full feature distribution by group
  - price: Open, High, Low, Close
  - volume: Volume
  - bounded: RSI_14, BB_pctB
  - return-like: MACD, MACD_signal, MACD_hist, rolling_std_20, OBV_roc

Runs where data is (local or Colab). Output: CSV + printed verdict.

### Step 4: Review results, lock parameters (~30min)

Lock: window_size, k, threshold candidates, kernel strategy.
Update this plan with locked values.
If window=60 infeasible → fall back, communicate to Ian.

### Step 5: Download reference code for TCN + DLinear

```
reference_excerpts/pytorch_tcn_core.py         (already present from MVP)
reference_excerpts/ltsf_dlinear_model.py        (already present from MVP)
reference_excerpts/tslib_dlinear_model.py       (NEW — TSLib DLinear with classification)
reference_excerpts/tslib_autoformer_encdec.py   (NEW — moving_avg, series_decomp)
```

Add source/license headers per construction plan v2 §2.1 rules.

### Step 6: TCN classifier (construction plan step 10, test-first, ~2h)

File: ml_utils/models/tcn_classifier.py
Spec: construction plan v2 §5.7
Reference: reference_excerpts/pytorch_tcn_core.py
Tests: test_models_shape.py additions

### Step 7: Standard DLinear classifier (construction plan step 11, test-first, ~2h)

File: ml_utils/models/dlinear_classifier.py
Spec: construction plan v2 §5.8 + research report 3 constraints (odd kernel, individual=False)
Reference: reference_excerpts/ltsf_dlinear_model.py + tslib_dlinear_model.py + tslib_autoformer_encdec.py
Tests: test_models_shape.py additions

### Step 8: Notebook 03 — baseline comparison under new labels

File: notebooks/03_baseline_comparison.ipynb
Uses ml_utils pipeline end-to-end.
Runs selected thresholds from Step 4.
Models: dummy baselines + LSTM + TCN + standard DLinear.
Reports: macro F1, balanced accuracy, confusion matrix, coverage per threshold.

### Step 9: Download additional reference code for improved model

```
reference_excerpts/fedformer_autoformer_encdec.py  (NEW — asymmetric padding, multi-scale softmax)
reference_excerpts/tslib_exp_classification.py      (NEW — classification training loop)
```

### Step 10: Improved multi-scale DLinear + residual TCN (test-first, ~3h)

File: ml_utils/models/ms_dlinear_tcn_classifier.py
Tests: tests/test_ms_dlinear_tcn.py
Spec: Section 6.3 of this plan.

### Step 11: Full experiment notebook

Extend Notebook 03 or create Notebook 04.
Full comparison: baselines + LSTM + TCN + DLinear + improved model.

### Step 12: Results to Ian

Email with profiling summary, baseline results, improved model results.

## 9. Deferred Items

| Item | Trigger |
|------|---------|
| Volatility-scaled threshold | After fixed-bps results reported |
| Three-class up/flat/down | After binary results, Ian confirms |
| Stock-aware embedding | After improved model works |

## 10. Version Control

All Phase 1B work on branch phase-1b.
Each step gets atomic commits.
Merge to main after Notebook 03 produces validated results.
