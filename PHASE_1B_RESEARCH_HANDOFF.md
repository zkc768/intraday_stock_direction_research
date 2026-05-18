# Phase 1B Research Handoff — Library-First

Status: Steps 1-2 complete; Step 3 capacity profiling next
Date: 2026-05-17
For: The AI agent managing hf_stock_clf / ml_utils development

---

## 0. Context

Research A is complete. Three research reports exist:
- deep-research-report (1).md — early exploration (superseded)
- deep-research-report (2).md — Phase 1B literature review
- deep-research-report (3).md — Phase 1B engineering reference (most precise)

The project builds a PyTorch library (ml_utils) using test-first development. Phase 1B extends the MVP library with new label functions, TCN, DLinear, and an improved model. All code goes into ml_utils first, then experiments run in notebooks.

Do NOT create Colab TF/Keras experiment notebooks. The library IS the product.

Current verified status:
- Step 1 no-trade-band labels complete: `c7162c2 feat(dataset): add no-trade-band labels`
- Step 2 config extension complete: `d9352fc feat(config): add Phase 1B label fields`
- Latest validation: 107 passed, 1 known pre-existing scheduler-order warning
- Next step: Step 3 capacity profiling
- TCN, DLinear, improved model, and Notebook 03 are still absent

---

## 1. What Research Report 3 Changes

Six findings require updates to how Phase 1B code is built.

### 1.1 Even kernels break standard moving_avg implementations

LTSF-Linear, Autoformer, and TSLib use symmetric (k-1)//2 endpoint padding with AvgPool1d. Only odd kernels preserve output length. Even kernels (6, 12, 24) produce L-1 output, which is a shape mismatch.

Only FEDformer uses asymmetric padding that preserves length for even kernels.

Action for agent:
- Standard DLinear classifier (ml_utils/models/dlinear_classifier.py): MUST use odd kernel only. Default moving_avg_kernel=5.
- Improved model (ml_utils/models/ms_dlinear_tcn_classifier.py): if using [3,6,12,24], must implement FEDformer-style padding. Alternative: use all-odd [3,7,13,25].
- Add this as a code comment in the moving_avg implementation.

### 1.2 Standard DLinear baseline definition is strict

Original DLinear: single-scale decomposition, individual=False (shared linear layers), no regularization.
TSLib classification adaptation: flatten full sequence [B, L*C] → Linear(L*C, num_classes).

Action for agent:
- dlinear_classifier.py must default to individual=False
- Classification head must be flatten + single linear layer, not pooling or last-step
- Do not add multi-scale, TCN, or stock-aware to this file
- Write as "TSLib-style DLinear adapted for classification" in docstring

### 1.3 TSLib multi-scale != FEDformer multi-scale

TSLib series_decomp_multi: averages multi-scale outputs (simple mean).
FEDformer series_decomp_multi: learnable per-scale scalar softmax weights.

Action for agent:
- In the improved model, use FEDformer-style per-scale scalar softmax weights
- Do not use simple averaging
- Add a comment documenting this distinction

### 1.4 individual=False is the default

Original DLinear CLI: --individual defaults to False (store_true).

Action for agent:
- dlinear_classifier.py __init__ parameter: individual: bool = False
- Test both individual=True and individual=False in shape tests

### 1.5 Selection bias has precise formulation

The no-trade-band classifier estimates P(sign(r) | X, |r| > τ), not P(sign(r) | X).

Action for agent:
- Add this to the docstring of make_no_trade_band_labels
- Notebook 03 must print a disclosure when reporting metrics

### 1.6 Reporting must include coverage

Every threshold sweep result must report retained_pct alongside metrics.

Action for agent:
- Capacity profiling output tables must include retained_pct column
- Notebook 03 must include retained_pct in all results tables

---

## 2. Reference Code — Download Schedule

Download reference files to reference_excerpts/ with source headers. Do not download until the step that needs them.

### Before Step 6 (TCN classifier):

Already present from MVP:
- reference_excerpts/pytorch_tcn_core.py

### Before Step 7 (standard DLinear classifier):

Download:
- reference_excerpts/tslib_dlinear_model.py — from thuml/Time-Series-Library models/DLinear.py
- reference_excerpts/tslib_autoformer_encdec.py — from thuml/Time-Series-Library layers/Autoformer_EncDec.py

Already present:
- reference_excerpts/ltsf_dlinear_model.py — from cure-lab/LTSF-Linear models/DLinear.py

### Before Step 10 (improved model):

Download:
- reference_excerpts/fedformer_autoformer_encdec.py — from MAZiqing/FEDformer layers/Autoformer_EncDec.py
- reference_excerpts/tslib_exp_classification.py — from thuml/Time-Series-Library exp/exp_classification.py

### Source header format:

```python
# Source: https://github.com/{org}/{repo}/blob/main/{path}
# Commit: {hash}
# Downloaded: {date}
# License: {license}
# Purpose: Reference only — not imported by ml_utils production code
```

---

## 3. Step-by-Step Agent Instructions

### Step 1: No-trade-band label function — complete

Module: ml_utils/dataset.py (add function) or ml_utils/labels.py (new file)
Test file: tests/test_no_trade_band_labels.py
Commit: `c7162c2 feat(dataset): add no-trade-band labels`

Implemented API:

```python
def make_no_trade_band_labels(
    df: pd.DataFrame,
    price_col: str,
    k: int,
    threshold_bps: float,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]
```

Actual outputs:
- DataFrame columns: `future_avg_r`, `label`
- Diagnostics dict: `n_total`, `n_tail`, `n_cross_day`, `n_neutral`, `n_up`, `n_down`
- `timestamp_col=None` disables cross-day filtering
- providing `timestamp_col` enables cross-day invalidation by date
- legacy `make_binary_labels_from_future_avg_return` remains unchanged; exact-zero `future_avg_r` maps to label 0.0, last k rows are NaN, and cross-day handling is not internal to the legacy function

Tests to write:
1. threshold_bps=0: Up/Down counts match existing make_binary_labels_from_future_avg_return (except exact-zero handling — document difference)
2. threshold_bps=10: neutral samples correctly become NaN
3. Symmetric threshold: samples at exactly +threshold and -threshold become NaN
4. Cross-day handling: matches existing function
5. Tail-row handling: last k rows are NaN
6. Diagnostics dict: n_total, n_tail, n_cross_day, n_neutral, n_up, n_down sum correctly

Implementation rules:
- Reuse existing `future_avg_r` computation logic
- Do NOT modify make_binary_labels_from_future_avg_return
- threshold = threshold_bps / 10_000
- return > +threshold → 1 (up)
- return < -threshold → 0 (down)
- |return| <= threshold → NaN
- Add docstring noting this estimates P(sign(r) | X, |r| > τ)

### Step 2: Config extension — complete

Module: ml_utils/config.py
Test file: tests/test_config.py (extend existing)
Commit: `d9352fc feat(config): add Phase 1B label fields`

Add fields:
```python
label_mode: str = "legacy_binary"
threshold_bps: float = 0.0
```

Allowed label_mode values:
- `"legacy_binary"`
- `"no_trade_band"`

Deferred and not configured yet:
- volatility-scaled threshold
- three-class flat/up/down
- threshold lists

Tests:
- Config with new fields validates
- Config without new fields uses defaults
- Existing tests still pass

### Step 3: Capacity profiling — next

Tests first: tests/test_phase1b_capacity_profile.py
Core implementation: ml_utils/profiling.py
Optional later wrapper: scripts/phase1b_capacity_profile.py

The core helpers live in ml_utils/profiling.py. The optional wrapper imports ml_utils.profiling helpers and only handles data loading, CSV output, and printed verdicts; it must not duplicate core profiling logic.

Input: data directory path (local or Drive)
Output: CSV files + printed verdict

Parameters to sweep:
- window_size: [12, 24, 60]
- label_horizon_k: [12, 24]
- threshold_bps: [0, 5, 10, 15, 20, 30]

Must compute:
- Bars-per-day distribution (min, P5, P10, median, max per ticker)
- Pre-check: window+k > P5 → INFEASIBLE
- Per combo: retained windows, Up/Down ratio, minority %, neutral drop %, effective target independence (retained_bars/k)
- Scaler diagnostic: per feature group mean shift and std ratio (full train vs retained train)

Feature groups for scaler diagnostic:
- price: Open, High, Low, Close
- volume: Volume
- bounded: RSI_14, BB_pctB
- return-like: MACD, MACD_signal, MACD_hist, rolling_std_20, OBV_roc

Feasibility gates:
- Pooled PASS: train >= 5000, val >= 1000, test >= 1000, minority >= 30%
- Per-ticker WARNING: train < 800 or minority < 25%
- Per-ticker FAIL: train < 500 or minority < 20%
- Scaler WARNING: |mean_shift| > 0.25 * full_std or std_ratio outside [0.75, 1.25]

### Steps 6-7: TCN + DLinear classifiers

Follow construction plan v2 §5.7 and §5.8 exactly.
Agent must read reference files before writing.
Key update for DLinear: individual=False default, odd kernel only.

### Step 10: Improved model

File: ml_utils/models/ms_dlinear_tcn_classifier.py

Agent must read:
- reference_excerpts/fedformer_autoformer_encdec.py (asymmetric padding, softmax fusion)
- reference_excerpts/tslib_autoformer_encdec.py (standard decomp for contrast)
- reference_excerpts/tslib_dlinear_model.py (DLinear backbone)

Architecture:
- Multi-scale moving_avg with length-preserving padding (FEDformer-style if using even kernels)
- Per-scale scalar softmax fusion weights (learnable)
- Trend: per-scale DLinear linear projection → concatenate
- Residual: X - fused_trend → lightweight TCN (2 blocks, 32 channels, dilation [1,2], kernel=3)
- Fusion: concat(trend_out, residual_out) → Linear(combined_dim, num_classes)

---

## 4. What NOT to Do

- Do NOT write Colab TF/Keras notebooks for Phase 1B experiments
- Do NOT skip test-first workflow
- Do NOT modify existing make_binary_labels_from_future_avg_return
- Do NOT use even kernels in standard DLinear baseline
- Do NOT use TSLib-style simple averaging for multi-scale fusion
- Do NOT call the standard DLinear a "published classification model"
- Do NOT download reference code before the step that needs it
- Do NOT update memory file with plan details (only verified facts)
- Do NOT email Ian before profiling and baseline results exist

## 5. Immediate Next Action

Step 3: capacity profiling.

Everything else follows sequentially after Step 3.
