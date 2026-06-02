# 🔴 CRITICAL REVIEW — ROUND 3: Full-Stack Adversarial Audit

**Date:** 2026-06-02
**Review Type:** 4 parallel adversarial agents + direct quantitative verification
**Scope:** All active code post-Round-2 fixes, with emphasis on new `validation_pipeline.py` (589 lines)
**Total Findings:** 42 distinct issues (13 CRITICAL, 11 HIGH, 10 MEDIUM, 8 LOW)
**Test Status:** 40 passed

---

## EXECUTIVE SUMMARY

**You added ~950 lines of runnable pipeline code. That is real progress.** Three structural problems remain that individually invalidate any research conclusion drawn from the current pipeline:

1. **Feature engineering has three independently-fatal flaws:**
   - RSI still computed with wrong alpha (1.87× too fast) — flagged in Round 2, not fixed
   - MACD resets daily → 43.6% of each trading day destroyed → **model is afternoon-only**
   - `time_of_day` sin/cos wraps around: 9:30 AM and 4:00 PM → identical (0, 1) encoding

2. **`cap_rows_chronologically` is a lie.** `np.linspace(0, N-1, 20000)` does uniform-stride subsampling across concatenated ticker blocks — neither chronological, nor stratified, nor random. The function name actively deceives readers.

3. **Documentation debt remains severe.** ~5.5:1 docs-to-code ratio, stale review documents describing deleted code, notebook claiming dependencies that don't exist in `requirements.txt`, and `BASELINE_REFERENCE.md` referencing an archived CLI runner that is not the active pipeline.

---

## PART I: CRITICAL ISSUES (13 findings)

### 1.1 [CRITICAL] `time_of_day_sin/cos` maps market open and market close to the EXACT same point

**File:** [baseline_v1.py:111-118](intraday_research/baseline_v1.py#L111-L118)

```python
minutes_since_open = minute_of_day - MARKET_OPEN_MINUTE   # 0..390
np.sin(2.0 * np.pi * minutes_since_open / TRADING_DAY_MINUTES)
np.cos(2.0 * np.pi * minutes_since_open / TRADING_DAY_MINUTES)
```

At open (`minutes_since_open=0`): `(sin, cos) = (0, 1)`.
At close (`minutes_since_open=390`): `2π` → `(sin, cos) = (0, 1)`.

**The first bar of the day and the last bar of the day are numerically indistinguishable.** Any linear model gets identical cyclical input for 9:30 AM and 4:00 PM despite radically different market microstructures. Additionally, on half-days (Thanksgiving eve, Christmas Eve: close at 1:00 PM = 210 minutes), the encoding silently shifts — `cos(2π*210/390) = -0.97`, not 1.0.

**Fix:** Map to slightly less than 2π. Use `2.0 * np.pi * minutes_since_open / (TRADING_DAY_MINUTES - 1)` or a non-wrapping encoding like `minutes_since_open / TRADING_DAY_MINUTES` as a linear feature.

---

### 1.2 [CRITICAL] Wilder's RSI alpha still wrong — Round 2 flagged, NOT fixed

**File:** [baseline_v1.py:58-62](intraday_research/baseline_v1.py#L58-L62), [90-91](intraday_research/baseline_v1.py#L90-L91)

```python
# grouped_ewm(span=14) → alpha = 2/(14+1) = 2/15 = 0.1333
# True Wilder's RSI: alpha = 1/14 = 0.0714
```

Quantitative verification confirmed:
- `span=14` alpha = 0.1333 (nearly double the correct value)
- True Wilder's alpha = 0.0714
- Ratio: 1.87× too fast
- Correlation between span=14 and true Wilder's `avg_gain`: 0.942
- Mean absolute difference: 0.00033, max: 0.00122

The feature labeled `rsi_14` is not RSI-14 — it is equivalent to a ~7-period Wilder's RSI. Any model coefficient learned for this feature is optimized for the wrong indicator.

**Fix:** Use `ewm(alpha=1/14, adjust=False, min_periods=14)` or equivalently `ewm(span=27, adjust=False, min_periods=14)`.

---

### 1.3 [CRITICAL] MACD EMA cascade with daily group reset destroys ~44% of each trading day — model is afternoon-only

**File:** [baseline_v1.py:105-108](intraday_research/baseline_v1.py#L105-L108)

```python
ema_12 = grouped_ewm(close, day, 12)    # resets every day
ema_26 = grouped_ewm(close, day, 26)    # resets every day
signal = grouped_ewm(macd, day, 9)       # resets every day
```

Warmup cascade:
| Component | `min_periods` | First non-NaN bar (0-indexed) |
|---|---|---|
| EMA_12 | 12 | bar 11 |
| EMA_26 | 26 | bar 25 |
| Signal (EMA_9 of MACD) | 9 from bar 25 | bar 33 |
| `normalized_macd_hist` | — | **bar 33 (~12:15 PM)** |

With `window_size=12`, the first valid window endpoint is bar 44 (**~1:10 PM**). Only ~33 windows survive per day, all from the afternoon. **The entire morning session (9:30 AM–1:10 PM) is structurally invisible to the model.** Opening reversals, gap-fades, and the first-hour volume surge cannot be learned.

**Fix:** Remove `groupby(day)` from EMA computations. Run EMA continuously across days. The `invalid_cross_day` check already handles label boundary invalidation; features do not need the same guard.

---

### 1.4 [CRITICAL] `cap_rows_chronologically` uses `np.linspace` uniform subsampling — not chronological

**File:** [validation_pipeline.py:183-190](intraday_research/validation_pipeline.py#L183-L190)

```python
selected = np.linspace(0, len(y_values) - 1, num=max_rows, dtype=int)
return x_values[selected], y_values[selected]
```

The function name claims chronological capping. The implementation does uniform-stride subsampling across a concatenated array structured as `[CSCO 1998-2013, JPM 1998-2013, KO 1998-2013, MSFT 1998-2013, WMT 1998-2013]` — not a single time stream. `dtype=int` truncation produces off-by-one at the tail. The result:
- Disproportionately represents tickers with more valid windows
- Destroys chronological continuity
- Silently destroys class balance if temporal class imbalance exists
- Applied to BOTH train and validation data (in `compute_mutual_information_diagnostic`)

**Fix:** Rename to `subsample_rows_uniformly`, document the sampling strategy in output metadata, or use stratified sampling, or use truncation (keep last N rows to simulate deployment).

---

### 1.5 [CRITICAL] Labels use positional `.shift(-k)` not time-aware shifting

**File:** [baseline_v1.py:128-129](intraday_research/baseline_v1.py#L128-L129)

```python
current["future_cumulative_return"] = close.shift(-horizon_k) / close - 1.0
```

`pd.DataFrame.shift(-12)` shifts by positional index, not by timestamp. If intraday data has missing bars (data feed gaps, illiquid stocks in early years), the 12th positional bar forward may represent 60 minutes (as intended), 72 minutes (one bar missing), or span a weekend. The `invalid_cross_day` check catches cross-day cases, but within the same day, missing bars create inconsistent look-ahead horizons across dates.

**Fix:** At minimum, add missing-bar detection to the audit. Ideally, reindex to a complete time grid to verify bar completeness before shifting.

---

### 1.6 [CRITICAL] `json_default` silently corrupts `np.bool_` into `1`/`0`

**File:** [run_validation_only_pipeline_smoke.py:36-41](scripts/run_validation_only_pipeline_smoke.py#L36-L41)

```python
if isinstance(value, (np.integer, np.floating)):
    return value.item()
```

`np.bool_` is a subclass of `np.integer`. `isinstance(np.bool_(True), np.integer)` returns `True`. `np.bool_(True).item()` returns `1`, not `true`. Every boolean field in the report JSON (`exists`, `chronological`, `available`, `current_verification`, `current_file_exists`) is serialized as integer `1`/`0` instead of `true`/`false`.

**Fix:** Check for `isinstance(value, (np.bool_, bool))` BEFORE `np.integer`.

---

### 1.7 [CRITICAL] Notebook claims `lightgbm` is in `requirements.txt` — it's not

**File:** [04_ian_research_memo.ipynb](notebooks/04_ian_research_memo.ipynb), cell `d5ee1d70`

The notebook's "Model Availability And Validation Plan" section states: "`requirements.txt` includes LightGBM." `requirements.txt` contains zero mention of `lightgbm`. Yet `validation_pipeline.py` imports and uses `LGBMClassifier`. If a reader follows the notebook's instructions, the LightGBM adapter silently reports "unavailable." If they install `lightgbm` separately (the env has it), the adapter runs — but the dependency isn't declared.

**Fix:** Add `lightgbm==4.6.0` to `requirements.txt`, or correct the notebook text.

---

### 1.8 [CRITICAL] `AGENTS.md` Section 4 lists MS-DLinear+TCN as a "default" model — zero active implementation

**File:** [AGENTS.md:145-153](AGENTS.md#L145-L153)

The table says default models are "LightGBM and MS-DLinear+TCN." Active code contains: sklearn LogisticRegression (working), LightGBM adapter (gated behind precheck), zero MS-DLinear+TCN path. MS-DLinear+TCN exists only in `archive/` — which AGENTS.md Section 1 forbids importing without explicit user request. This is a direct contradiction.

**Fix:** Remove MS-DLinear+TCN from the default table or annotate it as "archived only, no active adapter." Add `sklearn_logreg` to the table since it's the only working model.

---

### 1.9 [CRITICAL] `BASELINE_REFERENCE.md` describes an archived runner as if it's the active pipeline

**File:** [BASELINE_REFERENCE.md:45-48](docs/BASELINE_REFERENCE.md#L45-L48), [218-270](docs/BASELINE_REFERENCE.md#L218-L270)

References `resolve_feature_set(...)` (archived), torch LSTM/TCN/dlinear support (archived), `sklearn_logreg` via CLI flags (archived). The "Tiny Validation Prompt" (lines 218-270) tells the operator to inspect argparse in an archived file. The active pipeline (`build_validation_only_report` / `run_validation_only_pipeline_smoke.py`) has a completely different interface.

**Fix:** Rewrite BASELINE_REFERENCE.md to reference the actual active pipeline.

---

### 1.10 [CRITICAL] 9 code artifacts duplicated verbatim across 3 locations

`find_timestamp_column`, `load_ticker_csv`, `audit_ticker_frame`, `pooled_train_validation_labels`, `summarize_window_class_balance`, `CALENDAR_SPLITS`, `FEATURE_COLUMNS`, `DEFAULT_TICKERS`, and metric-computation-with-warning-suppression all exist in two or more of: `validation_pipeline.py`, `baseline_v1.py`, and the notebook. The pipeline versions differ from the notebook versions (e.g., `load_ticker_csv` takes `data_dir` vs `data_root`, the pipeline version validates columns, the notebook version doesn't). A fix in one location silently diverges from the other.

**Fix:** Make the notebook import from `intraday_research.validation_pipeline`. The pipeline is the single source of truth.

---

### 1.11 [CRITICAL] Feature ablation uses leave-one-out on linear model — confounded by correlated features

**File:** [validation_pipeline.py:342-379](intraday_research/validation_pipeline.py#L342-L379)

Removing one feature at a time from logistic regression measures delta-F1. When features are correlated (and at least three pairs are: `log_return`/`close_to_open_return`, `rolling_volatility_20`/`bollinger_pctb`, `rsi_14`/`bollinger_pctb`), removing one shifts its contribution to the correlated partner. `delta ≈ 0` for every feature even though the set collectively carries signal. This produces false negatives for feature importance.

**Fix:** Use permutation importance or SHAP on a non-linear model, or at minimum document this limitation in the ablation output.

---

### 1.12 [CRITICAL] Documentation-to-code ratio is ~5.5:1

Active production code: 965 lines (`baseline_v1.py` 317 + `validation_pipeline.py` 589 + smoke script 59). Documentation: ~1,922 lines (AGENTS.md 207 + BASELINE_REFERENCE.md 277 + RESEARCH_WORKFLOW.md 275 + CRITICAL_REVIEW 1120 + ENVIRONMENT.md 24 + README.md 19). Plus 208 lines in AGENTS.md restating rules the pipeline already enforces programmatically. The "PM document machine" that AGENTS.md claims to reject is still here — just renamed files.

---

### 1.13 [CRITICAL] `CRITICAL_REVIEW_2026-06-02.md` is partially stale — audits code that no longer exists

**File:** [CRITICAL_REVIEW_2026-06-02.md:72-80](docs/CRITICAL_REVIEW_2026-06-02.md#L72-L80)

Section 1.2 audits `future_avg_return` computed from arithmetic mean of one-bar returns. Current code uses `future_cumulative_return = close[t+k]/close[t] - 1`. The review's most damning finding describes code that was fixed — but the document still presents it as current. A reader trusting this review believes the current label is mathematically wrong when it has been corrected.

**Fix:** Move to `docs/archive/` or prefix with a clear "RESOLVED IN CURRENT CODEBASE — see Round 3 review" header.

---

## PART II: HIGH SEVERITY (11 findings)

### 2.1 [HIGH] `[:, -1, :]` discards 11/12 of every window — temporal modeling claim is false

**File:** [validation_pipeline.py:173](intraday_research/validation_pipeline.py#L173)

```python
x_parts.append(bundle["X"][:, -1, :])  # last step only
```

The pipeline builds 3D arrays with 12 time steps (expensive O(n) computation in `build_windows_for_segment`), then every downstream consumer takes only `[:, -1, :]`. The model is a **point-in-time classifier** using only contemporaneous features. Any claim about 12-bar lookback windows is false — the lookback is computed, stored, and silently discarded.

---

### 2.2 [HIGH] `ConvergenceWarning` promoted to hard exception — no fallback

**File:** [validation_pipeline.py:325-327](intraday_research/validation_pipeline.py#L325-L327)

```python
warnings.filterwarnings("error", category=ConvergenceWarning)
model.fit(x_train, y_train)
```

If `LogisticRegression(max_iter=200)` fails to converge, the entire `build_validation_only_report` crashes. No try/except, no `max_iter` increase, no fallback to report non-convergence as a diagnostic field.

**Fix:** Wrap in try/except, catch convergence failure, and return a `"converged": False` field in the diagnostic output.

---

### 2.3 [HIGH] `build_validation_only_report` is monolithic — 10+ interlocked sub-steps

**File:** [validation_pipeline.py:491-589](intraday_research/validation_pipeline.py#L491-L589)

The function loads data → builds features → fits scaler → builds windows → runs 6+ diagnostics → assembles metadata → returns a single dict. One diagnostic failure kills the entire report. No intermediate step can be independently tested, reused, or extended.

**Fix:** Split into individually-callable sub-builders, each wrapped in try/except that returns an error entry rather than crashing.

---

### 2.4 [HIGH] LogisticRegression has no `class_weight` — degenerates on imbalanced labels

**File:** [validation_pipeline.py:320-323](intraday_research/validation_pipeline.py#L320-L323)

Intraday no-trade-band labels are inherently imbalanced. Without `class_weight='balanced'`, logistic regression learns to always predict the majority class to minimize cross-entropy loss. High accuracy, near-zero macro-F1 — not because features are weak, but because the model was instructed to ignore the minority class.

**Fix:** Add `class_weight='balanced'` to the LogisticRegression constructor.

---

### 2.5 [HIGH] Walk-forward fold specs never populated with model scores

**File:** [validation_pipeline.py:436-488](intraday_research/validation_pipeline.py#L436-L488), [585-588](intraday_research/validation_pipeline.py#L585-L588)

The pipeline computes 3-fold walk-forward date ranges and row counts, but never trains or evaluates any model on these folds — not even the dummy baseline. The report JSON includes a `"walk_forward_contract"` section with fold specs alongside actual model metrics (logreg, LightGBM), creating the appearance of a walk-forward study. A skimming reviewer would be misled.

---

### 2.6 [HIGH] Window class balance rows stamped with wrong scope — train rows labeled as `validation_only`

**File:** [validation_pipeline.py:129-148](intraday_research/validation_pipeline.py#L129-L148)

The function iterates over `("train", "validation")` but stamps every row for every split with `"scope": "validation_only"`. Training-set class distributions are injected into output labeled as validation-only.

---

### 2.7 [HIGH] `rolling_volatility_20` warmup is 22 bars, not 20 — undocumented

**File:** [baseline_v1.py:79-81](intraday_research/baseline_v1.py#L79-L81)

Trace chain: `log_return` → bar 0 is NaN → `grouped_shift(log_return, 1)` → bars 0-1 are NaN → `rolling(20, min_periods=20)` → first valid at bar 21. **22 bars lost** (0-indexed bars 0-21). The name `rolling_volatility_20` misleads the reader. `BASELINE_REFERENCE.md` mentions "first valid at bar 21" but doesn't state the 22-bar warmup count.

---

### 2.8 [HIGH] `normalized_macd_hist = (macd - signal) / ema_26` — non-standard, no theoretical basis

**File:** [baseline_v1.py:109](intraday_research/baseline_v1.py#L109)

This is `(ema_12 - ema_26 - ema_9(ema_12 - ema_26)) / ema_26`. Three distinct scales are mixed: a price-difference momentum signal, a smooth of that difference (second derivative), and division by a price level. No published reference defines this formula. The normalization is an ad-hoc engineering choice with undefined statistical distribution.

---

### 2.9 [HIGH] 9 artifacts duplicated between pipeline and notebook

| Artifact | Pipeline | Notebook |
|---|---|---|
| `find_timestamp_column` | L48-53 | cell d1c1392e |
| `load_ticker_csv` | L56-69 | cell d1c1392e |
| `audit_ticker_frame` | L73-84 | cell d1c1392e |
| `pooled_train_validation_labels` | L151-159 | cell d30ffc4d |
| `summarize_window_class_balance` | L129-148 | cell a55bc0f7 |
| `CALENDAR_SPLITS` | L39-43 | cell 73a4eee6 |
| `FEATURE_COLUMNS` | L27-38 | cell 73a4eee6 |
| `DEFAULT_TICKERS` | L26 | cell 73a4eee6 |
| Metric calc + warning suppression | L275-299 | baseline_v1.py L268-280 |

---

### 2.10 [HIGH] `AGENTS.md` rule 3.2.3 potentially violated by label formula change

**File:** [AGENTS.md:109-113](AGENTS.md#L109-L113) states: "After holdout/test is viewed, do not change features, labels, thresholds..." The label changed from `future_avg_return` (arithmetic mean of 12 one-bar returns) to `future_cumulative_return` (`close[t+k]/close[t] - 1`). If this change occurred after holdout was viewed, it violates AGENTS.md's own hard rule. The change should be documented as a legitimate mathematical correction, not a silent fix.

---

### 2.11 [HIGH] `pytest.ini` `filterwarnings = error` is too aggressive

**File:** [pytest.ini:4](pytest.ini#L4)

Converts ALL warnings to errors, including future deprecation warnings from numpy, pandas, and sklearn. A dependency upgrade will produce cascading test failures on unrelated code. Production code already suppresses known sklearn warnings in `_balanced_accuracy` and `evaluate_predictions`.

**Fix:** Use `filterwarnings = error::pytest.PytestWarning` instead, or add specific `ignore` entries for known dependency deprecations.

---

## PART III: MEDIUM SEVERITY (10 findings)

### 3.1 [MEDIUM] `build_walk_forward_fold_specs` silently skips empty chunks

**File:** [validation_pipeline.py:451-456](intraday_research/validation_pipeline.py#L451-L456)

`np.array_split` can produce empty chunks. The `continue` silently skips them. Caller receives fewer folds than `n_folds` with no indication.

### 3.2 [MEDIUM] `evaluate_lightgbm_last_step_adapter` returns inconsistent schema

**File:** [validation_pipeline.py:398-433](intraday_research/validation_pipeline.py#L398-L433)

When LightGBM is unavailable: returns `{"adapter", "available", "blocker", "scope"}`. When available: returns `{"adapter", "available", "model", ..., "macro_f1", ...}`. No consumer can statically determine which shape they got.

### 3.3 [MEDIUM] Missing return type annotations and docstrings on 17/18 functions

**File:** [validation_pipeline.py](intraday_research/validation_pipeline.py)

Only `precheck_lightgbm_dependency` has a return type hint (`-> dict`). Every other function returns untyped dicts. Every intermediate data structure is opaque.

### 3.4 [MEDIUM] Tests cover only 4 of 19 functions in validation_pipeline.py

**File:** [test_validation_pipeline.py](tests/test_validation_pipeline.py)

Only `load_ticker_csv`, `build_validation_only_report`, `build_walk_forward_fold_specs`, and `precheck_lightgbm_dependency` are tested. Fifteen functions have zero direct coverage. The sole integration test is a smoke test checking shapes and key presence — it does not verify that logistic regression learned anything, that dummy scores are correct, or that holdout was truly excluded.

### 3.5 [MEDIUM] Mutual information estimates are not comparable across features, no confidence intervals

**File:** [validation_pipeline.py:245-272](intraday_research/validation_pipeline.py#L245-L272)

`mutual_info_classif` with k-NN entropy estimation (`n_neighbors=3` default) produces bias that depends on each feature's marginal distribution. Features with wider dynamic range or heavier tails show higher MI regardless of true predictive value. No bootstrap or permutation-derived confidence intervals are reported.

### 3.6 [MEDIUM] `summarize_feature_signal` computes means on scaled features — uninterpretable under distribution shift

**File:** [validation_pipeline.py:208-242](intraday_research/validation_pipeline.py#L208-L242)

`class_1_mean - class_0_mean` is computed in units of the training set's standard deviation. For the validation split, features are transformed using training mean/std, so any "signal" metric is confounded by the scaling, not just true signal change.

### 3.7 [MEDIUM] Logistic regression uses 1 seed, dummy baseline uses 5 seeds — statistically invalid comparison

**File:** [validation_pipeline.py:320](intraday_research/validation_pipeline.py#L320), [530](intraday_research/validation_pipeline.py#L530)

`delta_macro_f1_vs_dummy` compares a single point estimate against a distribution's mean with no variance estimate. The reader cannot assess whether the delta is statistically meaningful.

### 3.8 [MEDIUM] Dummy baseline uses `ddof=0` (population std) with `n=5` — underestimates ~11%

**File:** [validation_pipeline.py:199-200](intraday_research/validation_pipeline.py#L199-L200)

With 5 seeds, `ddof=1` (sample std) is the unbiased estimator. Population std biases reported error bars downward by `sqrt(4/5) = 0.894`.

### 3.9 [MEDIUM] `high_low_range = log(high / low)` — no guard for `high < low` or zero/negative prices

**File:** [baseline_v1.py:78](intraday_research/baseline_v1.py#L78)

If `high < low` (data feed error), produces negative "range" values that are physically impossible. If `low == 0` (flash crash bad print), produces `inf`. `_finite_rows` catches `inf`/`NaN` but not negative finite values.

### 3.10 [MEDIUM] LightGBM adapter has no early stopping — stops at arbitrary 25 iterations

**File:** [validation_pipeline.py:412-419](intraday_research/validation_pipeline.py#L412-L419)

No `eval_set`, no `early_stopping_rounds`. Always trains exactly 25 iterations. The model result is conditional on an arbitrary `n_estimators=25`. For weak signal, this may overfit; for strong signal, underfit.

---

## PART IV: LOW SEVERITY (8 findings)

### 4.1 [LOW] Missing `class_weight` in pipeline also affects ablation conclusions

Ablation compares each leave-one-out model against the full model, but both suffer from the same imbalance vulnerability. Removing a minority-class-critical feature may be masked because the model already ignores the minority class.

### 4.2 [LOW] Metric computation duplicated between validation_pipeline.py and baseline_v1.py

Both files inline the same `f1_score` + `balanced_accuracy_score` + `accuracy_score` pattern with identical warning suppression. Extract into a shared function.

### 4.3 [LOW] Train/validation labels extracted through two different code paths

`pooled_train_validation_labels` (L151) and `collect_last_step_xy` (L162) both concatenate labels, relying on undocumented dictionary insertion order for consistency. If window build order changes, dummy baseline could be evaluated on different labels than logistic regression.

### 4.4 [LOW] `make_daily_rows` duplicated between test files

Two diverging synthetic-data factories exist in `test_validation_pipeline.py` and `test_baseline_v1_helpers.py`. Extract to `conftest.py`.

### 4.5 [LOW] RSI edge-case mask logic is order-fragile but currently correct

`rs = avg_gain / avg_loss.replace(0.0, np.nan)` → `rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100)` → `rsi.mask(avg_loss.eq(0) & avg_gain.eq(0), 50)`. Depends on checking the original `avg_loss` (not the NaN-replaced version) and strictly ordered masks. Future editors could easily break this.

### 4.6 [LOW] `assign_calendar_split` doesn't report `outside_defined_calendar` row count

Any timestamp outside the defined calendar is silently excluded from all downstream processing with no diagnostic count.

### 4.7 [LOW] `TRADING_DAY_MINUTES = 390` assumes no early closes

On half-days (Thanksgiving eve, Christmas Eve: 210-minute session), time features produce misleading encodings past 1:00 PM.

### 4.8 [LOW] `pytest.ini` missing `norecursedirs = archive`

Archived tests reference modules not on `sys.path`. A bare `pytest` from project root without `--ignore` would encounter import errors.

---

## PART V: UNRESOLVED FROM PREVIOUS REVIEWS

| Issue | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| Label math (avg return → cumulative return) | CRITICAL | ✅ Fixed | ✅ Verified |
| Volume self-reference (prior-only shift) | CRITICAL | ✅ Fixed | ✅ Verified |
| Time encoding (24hr → 390-min session) | CRITICAL | ✅ Fixed | ❌ Now wraps around |
| RSI SMA→EWM | HIGH | ✅ Fixed (EWM) | ❌ Wrong alpha |
| MACD daily reset → 44% data loss | CRITICAL | Not fixed | ❌ Still not fixed |
| Bollinger div-by-zero | HIGH | ✅ Fixed | ✅ Verified |
| Scaler ignores non-finite rows | CRITICAL | ✅ Fixed | ✅ Verified |
| Dummy single-class crash | CRITICAL | ✅ Fixed | ✅ Verified |
| Dead deps (torch, lightgbm in req.txt) | HIGH | ✅ Removed | ⚠️ lightgbm now used but missing from req.txt |
| Notebook hardcoded coverage | MEDIUM | Not fixed | ❌ Still hardcoded |
| No end-to-end pipeline | CRITICAL | Not fixed | ✅ Fixed (validation_pipeline.py) |
| No LightGBM adapter | MEDIUM | Not fixed | ✅ Fixed |
| No walk-forward spec | MEDIUM | Not fixed | ✅ Fixed (specs only, no scores) |
| Documentation-to-code ratio | 12.8:1 | ~8:1 | ~5.5:1 (improving) |

---

## PART VI: FIX PRIORITY ROADMAP

### Now (~30 minutes):
1. Rename `cap_rows_chronologically` → `subsample_rows_uniformly`; document in output metadata
2. Add `lightgbm` to `requirements.txt`, or wrap `LGBMClassifier` import in try/except
3. Fix `json_default` `np.bool_` handling before `np.integer` check
4. Delete archived-runner references from BASELINE_REFERENCE.md
5. Remove MS-DLinear+TCN from AGENTS.md default model table

### This week (2-4 hours):
6. Fix Wilder's RSI alpha: `ewm(alpha=1/14, adjust=False)`
7. Fix time_of_day wrap-around: map to slightly less than 2π
8. Remove `groupby(day)` from MACD EMA — run continuously across days
9. Add try/except around ConvergenceWarning with `"converged": false` graceful return
10. Add `class_weight='balanced'` to LogisticRegression
11. Deduplicate 9 shared artifacts between pipeline and notebook

### Next phase:
12. Make notebook import from `validation_pipeline` instead of redefining functions
13. Add tests for 15 uncovered functions in validation_pipeline.py
14. Move `CRITICAL_REVIEW_2026-06-02.md` to `docs/archive/` or add "RESOLVED" header
15. Run logistic regression with multiple seeds to match dummy baseline statistical rigor
16. Add `scipy` to `requirements.txt` (transitive dependency of sklearn used by `mutual_info_classif`)

---

## REFERENCES

- Settings file: `.claude/settings.local.json`
- Core helper library: [intraday_research/baseline_v1.py](intraday_research/baseline_v1.py)
- Validation pipeline: [intraday_research/validation_pipeline.py](intraday_research/validation_pipeline.py)
- CLI entry point: [scripts/run_validation_only_pipeline_smoke.py](scripts/run_validation_only_pipeline_smoke.py)
- Tests: [tests/test_baseline_v1_helpers.py](tests/test_baseline_v1_helpers.py), [tests/test_validation_pipeline.py](tests/test_validation_pipeline.py), [tests/test_notebook_static_gate.py](tests/test_notebook_static_gate.py)
- Research rules: [AGENTS.md](AGENTS.md)
- Baseline reference: [docs/BASELINE_REFERENCE.md](docs/BASELINE_REFERENCE.md)
- Research workflow: [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)
- Round 1 review: [docs/CRITICAL_REVIEW_2026-06-02.md](docs/CRITICAL_REVIEW_2026-06-02.md)
- Round 3 review (this document): [docs/CRITICAL_REVIEW_ROUND3_2026-06-02.md](docs/CRITICAL_REVIEW_ROUND3_2026-06-02.md)

---

*42 findings. 40 tests pass. The pipeline is now runnable — that matters. But it runs with RSI alpha nearly double what it should be, MACD resetting every morning and blinding the model to the first 3.5 hours of trading, and time encoding that cannot tell 9:30 AM from 4:00 PM. Fix those three before trusting any model coefficient or metric this pipeline produces.*
