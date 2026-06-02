# 🔴 CRITICAL REVIEW: `intraday_stock_direction_research`

**Date:** 2026-06-02
**Review Type:** Multi-agent adversarial audit (4 parallel reviews)
**Review Dimensions:** Research Validity & Leakage | Code Architecture | ML Feature Engineering | Project Structure & Process
**Total Findings:** 65+ distinct problems identified

---

## EXECUTIVE SUMMARY

**Your rebuild is a documentation project that masquerades as a research project.** The four reviews collectively found 65+ distinct problems across all dimensions of the project. The most damning findings:

1. **Nothing runs.** A fresh clone produces zero research output. Not a single code path goes from `data/CSCO.csv` to a model evaluation.
2. **The features carry no signal.** Your own smoke report shows LightGBM Macro F1 = 0.396 vs Stratified Dummy = 0.496. Your model is *worse than random guessing* by 10 percentage points.
3. **The label is mathematically wrong.** `future_avg_return` is the arithmetic mean of 12 one-bar returns, not cumulative return. No trading strategy uses this quantity as its signal.
4. **The rebuild is an 85:1 regression.** You replaced 16,612 lines of working runner code with 195 lines of disconnected helpers.
5. **Your documentation-to-code ratio is 12.8:1.** ~2,500 lines of specs/plans/rules govern 195 lines of code that cannot produce any result.

---

## REVIEW DIMENSION 1: RESEARCH VALIDITY & DATA LEAKAGE

*Auditor: Research validity specialist. Examined shift logic, look-ahead bias, split-boundary handling, scaler contamination, window construction, label validity, statistical methodology.*

### 1.1 Pipeline Leakage Assessment (THE GOOD NEWS)

The pipeline architecture is fundamentally clean from a look-ahead perspective. Every component was verified:

**Shift logic in label construction — CORRECT**
```python
one_bar_return = current["close"].pct_change()
future_returns = [one_bar_return.shift(-1 - offset) for offset in range(int(horizon_k))]
```

Traced with a manual example (closes = [100, 101, 102, 103, 104, 105], k=2):
- `one_bar_return` = [NaN, 0.01, 0.0099, 0.0098, 0.0097, 0.0096]
- offset=0: `shift(-1)` = [0.01, 0.0099, 0.0098, 0.0097, 0.0096, NaN]
- offset=1: `shift(-2)` = [0.0099, 0.0098, 0.0097, 0.0096, NaN, NaN]
- `future_avg_return[0]` = mean(0.01, 0.0099) = 0.00995

**Cross-day check — CORRECT**
```python
for offset in range(1, int(horizon_k) + 1):
    same_day &= current_day.shift(-offset).eq(current_day)
```
For k=12, offset ranges 1-12, checking all future bars. Covers exactly the horizon.

**Cross-split boundary invalidation — CORRECT**
```python
horizon_split = current["split"].shift(-int(horizon_k))
current["invalid_cross_split"] = current["future_avg_return"].notna() & (
    current["split"] != horizon_split
)
```
Checks whether split at t+12 differs from split at t. Since splits are contiguous calendar blocks, checking the farthest bar suffices.

**Scaler fit on train only — CORRECT AND TEST-VERIFIED**
`fit_train_only_scaler` selects only `split == "train"` rows, confirmed by unit test `test_scaler_fit_ignores_validation_and_closed_holdout_values`.

**Window construction loop — CORRECT**
```python
for end_pos in range(window_size - 1, len(day_frame)):
    window = day_frame.iloc[end_pos - window_size + 1 : end_pos + 1]
    target = day_frame.iloc[end_pos]
```
For window_size=12, `end_pos` starts at 11 (0-indexed). First window: bars 0-11, target: bar 11. Bar 11's label looks forward to bars 12-23. This is correct — the prediction target is genuinely in the future.

**Window day constraint — CORRECT**
The outer loop groups by trading date via `segment.groupby(segment["timestamp"].dt.date)`. Windows are `iloc`-sliced within single-day DataFrames, so they cannot cross trading days.

### 1.2 CRITICAL: Label construction is mathematically incorrect for any trading application

```python
# baseline_v1.py, lines 22-28
one_bar_return = current["close"].pct_change()
future_returns = [
    one_bar_return.shift(-1 - offset) for offset in range(int(horizon_k))
]
current["future_avg_return"] = pd.concat(future_returns, axis=1).mean(
    axis=1, skipna=False
)
```

**The problem:** `future_avg_return` is the **arithmetic mean** of k=12 individual one-bar percentage returns. This is NOT the same as the cumulative return over the horizon.

- **Cumulative return (what a trader actually experiences):** `(close[t+k] / close[t]) - 1`
- **Arithmetic mean of one-bar returns (what your label computes):** `(1/k) * sum(r_{t+1} + r_{t+2} + ... + r_{t+k})`

For small returns these approximate each other (log-additivity), but for any significant move the error grows quadratically due to compounding. More importantly:

**No trading strategy uses the arithmetic mean of 12 one-bar returns as its signal.** A trader entering at bar t's close and exiting at bar t+12's close experiences the cumulative return. Your label does not correspond to any executable trade. The 5bps classification threshold is being applied to a quantity with no direct economic interpretation.

Furthermore, the following scenario is possible: bars t+1 through t+6 have large positive returns (+20 bps each), bars t+7 through t+12 have large negative returns (-20 bps each). The arithmetic mean would be near zero (labeled as "no-trade"), even though the stock was extremely volatile and a trader could have profited from the directional swing. The mean-based label conflates volatility with direction.

**The `RESEARCH_WORKFLOW.md` (line 151) replicates this same error**, embedding it in your methodology documentation.

### 1.3 CRITICAL: MACD EMA cascade destroys 43.6% of every trading day

The MACD computation is a cascade of three exponential moving averages, each resetting at day boundaries:

```
EMA_12: span=12, min_periods=12 → first non-NaN at bar 12 (10:25 AM)
EMA_26: span=26, min_periods=26 → first non-NaN at bar 26 (11:35 AM)
Signal (EMA_9 on MACD): span=9 → first non-NaN at bar 26 + 8 = bar 34 (12:15 PM)
normalized_macd_hist: first non-NaN at bar 34
```

A standard 6.5-hour trading day has 78 five-minute bars (09:30 to 16:00).

**Bars lost to MACD warmup: 34 out of 78 = 43.6% of the trading day.**

But the problem is worse than raw bar count suggests:

| Feature | First non-NaN bar | Bars lost |
|---|---|---|
| `log_return` | bar 1 | 1 |
| `rolling_volatility_20` | bar 20 | 19 |
| `rsi_14` | bar 14 | 13 |
| `bollinger_pctb` | bar 19 | 18 |
| `normalized_volume_20` | bar 19 | 18 |
| `normalized_macd_hist` | **bar 34** | **33** |

**`normalized_macd_hist` is the limiting feature.** Because `build_windows_for_segment` checks `window[scaled_columns].isna().any().any()` (all features in all window bars must be non-NaN), the entire window is rejected if any feature is NaN. This means:

- First valid window end position: bar 34 + window_size - 1 = 45 (~1:10 PM)
- Last valid window end position: bar 77 (4:00 PM)
- **Total: ~33 windows per trading day** out of a theoretical maximum of 67
- **51% of potential windows are lost**

All 33 windows come from the afternoon session (1:10 PM to 4:00 PM). The morning session — including the market open, the first hour of concentrated trading, and overnight gap adjustments — is **structurally invisible** to your model.

**The EMA reset at day boundaries is the root cause.** A continuous EMA carrying state across trading days would eliminate most of this loss, though cross-day label checks would still need validation.

**Impact:** Your model is an "afternoon-only" classifier. Morning-specific patterns (opening reversals, gap-fades, news reactions at 9:30-10:30 AM) cannot be learned. This is not a sample-size problem — it is a **systematic regime bias.**

### 1.4 CRITICAL: No cross-validation. Single train/validation split over 22 years of data.

The calendar split is:
- **Train:** 1998-01-02 to 2013-09-16 = **15.7 years** (~3,955 trading days)
- **Validation:** 2013-09-16 to 2017-01-25 = **3.35 years** (~844 trading days)
- **Holdout:** 2017-01-25 to 2020-06-06 = **3.36 years** (~847 trading days)

One contiguous validation window of 3.3 years tests **exactly one market regime.** There is no walk-forward validation, no rolling-origin retraining, and no expanding-window evaluation. A single validation score on one time period tells you nothing about stability, generalization, or robustness.

For a Northeastern thesis project, this is unacceptable. Marcos Lopez de Prado's *"Advances in Financial Machine Learning"* (2018) dedicates multiple chapters to why single-split evaluation in financial time series is indefensible. Any committee reviewer with ML/finance background will flag this immediately.

### 1.5 CRITICAL: Regime mismatch between train, validation, and holdout

The three calendar periods have fundamentally different market structures:

| Period | Market Character | VIX Range | Split |
|---|---|---|---|
| 1998-2000 | Dot-com bubble, extreme tech volatility | 20-45 | Train |
| 2000-2002 | Dot-com crash, high volatility | 20-45 | Train |
| 2003-2007 | Low volatility, steady bull market | 10-20 | Train |
| 2008-2009 | Financial crisis, extreme volatility | 30-80 | Train |
| 2010-2013 | Post-crisis recovery, ZIRP | 15-30 | Train |
| 2013-2017 | QE bull market, lowest vol in dataset | 10-20 | **Validation** |
| 2017-2020 | Tax cuts, late-cycle, COVID crash | 12-65 | **Holdout** |

**The validation period (2013-2017) is the most benign, lowest-volatility, easiest-to-predict regime in the entire dataset.** It is a steady quantitative-easing-driven bull market with minimal drawdowns. A model that performs well on validation may be doing nothing more than correctly identifying a persistent uptrend — which is not a directional classification skill, it's a regime-detection skill.

**The training set (1998-2013) spans multiple crises** (dot-com bubble/crash, 2008 financial crisis) that the validation set never experiences. The model may overfit to crisis-recovery patterns that do not repeat in the validation period.

**The holdout set (2017-2020) includes the COVID crash** — a regime unlike anything in the training or validation sets. Testing on COVID without training on anything comparable is a setup for catastrophic failure that provides zero information about model robustness.

### 1.6 CRITICAL: Pooled StandardScaler across 5 heterogeneous stocks and 15 years of data

`fit_train_only_scaler` concatenates training features from all 5 tickers and fits one `StandardScaler`:

```python
train_parts = []
for frame in split_frames_by_ticker.values():
    train = frame.loc[frame["split"] == "train", list(feature_columns)]
    train_parts.append(train.dropna())
train_matrix = pd.concat(train_parts, axis=0)
scaler.fit(train_matrix)
```

The 5 stocks have vastly different price levels, volatility profiles, and sector characteristics:

| Ticker | Approx Price Range (1998-2013) | Sector | Volatility Character |
|---|---|---|---|
| CSCO | $7 - $60 | Technology | High, tech-bubble affected |
| JPM | $15 - $55 | Financial | Crisis-affected, sector-specific vol |
| KO | $20 - $45 | Consumer Staples | Low volatility, defensive |
| MSFT | $20 - $200+ | Technology | High growth, split-adjusted |
| WMT | $30 - $80 | Consumer Retail | Moderate, defensive |

Features like `log_return` and `high_low_range` have fundamentally different distributions for CSCO (tech, high vol) vs KO (consumer staple, low vol). Features like `normalized_macd_hist / close` are price-level-dependent, making them 3-5x larger for a $7 stock than a $200 stock (see Finding 2.4).

A single mean/std per feature is a meaningless weighted average across incompatible distributions. Even within a single stock, market microstructure changed dramatically over 15 years:
- **2001:** Decimalization (tick size from 1/16 to $0.01)
- **2007:** Regulation NMS (fragmented liquidity)
- **2005-2010:** HFT proliferation transformed intraday dynamics

**The scaler is computing `(feature - mu_1998_2013_all_stocks) / sigma_1998_2013_all_stocks`**, which for any given stock in any given regime is a nonsense normalization.

### 1.7 CRITICAL: `normalized_volume_20` is self-referencing — bug documented but unfixed

```python
# notebook cell add_baseline_v1_features
log_volume = np.log1p(volume)
volume_mean_20 = grouped_rolling(log_volume, day, 20, "mean")
frame["normalized_volume_20"] = log_volume - volume_mean_20
```

At bar t, `volume_mean_20[t]` includes `log_volume[t]` in the 20-bar rolling mean. The current bar's volume appears on **both sides of the subtraction**:

```
normalized_volume_20[t] = log1p(volume[t]) - mean(log1p(volume[t-19:t+1]))
```

This is a form of **statistical self-reference** — bar t's volume "knows" its own deviation before the deviation should be observable. Standard practice uses `volume[t-20:t]` (exclusive of current bar):

```python
volume_mean_20 = grouped_rolling(log_volume.shift(1), day, 19, "mean")  # one-line fix
# or equivalently:
volume_mean_20 = grouped_rolling(log_volume, day, 20, "mean").shift(1)
```

**The `BASELINE_REFERENCE.md` explicitly acknowledges this bug on line 102:**

> *"if strict prior-only normalization is required, shift the rolling mean in a future approved patch."*

**This is a one-line fix.** Deferring it to "a future approved patch" while proceeding with experiments is indefensible. Every model trained with this feature is using a contaminated signal where the current bar's own volume partially determines its "normalized" volume feature.

### 1.8 MEDIUM: Overlapping windows create dependent observations and inflated effective sample size

With stride=1 (one-bar shifts), consecutive windows overlap by 11 out of 12 bars. Labels at positions t and t+1 share 11 of 12 future bars:

```
Window at t:    features[0:12],  label from bars 12-23
Window at t+1:  features[1:13],  label from bars 13-24
```

The overlap ratio is (window_size - 1) / window_size = 11/12 = 91.7%. This creates strong autocorrelation across training samples. **Standard errors are dramatically underestimated, and effective sample size is far smaller than the reported window count suggests.** This is a known issue in sliding-window time series ML but is rarely acknowledged in accuracy or F1 reporting. Your evaluation metrics will appear more precise than they actually are.

### 1.9 MEDIUM: Stratified dummy baseline is regime-dependent and potentially misleading

`evaluate_stratified_dummy` trains `DummyClassifier(strategy="stratified")` on training labels (learning the training class distribution) and evaluates on validation labels. The test `test_stratified_dummy_does_not_use_validation_distribution` correctly confirms no validation leakage.

**However:** The stratified dummy samples from the training class distribution. If the training period has 55% "up" labels (crisis-heavy, volatile) and the validation period has 52% "up" labels (steady bull market), the dummy predicts "up" ~55% of the time but sees "up" only ~52% of the time, introducing a negative bias. This **inflates the model's apparent edge over the baseline.**

The baseline is also too weak — it only tests "can we beat random guessing from class priors?" A more informative baseline would include simple heuristics like:
- "Always predict trend continuation" (predict same direction as last bar)
- "Always predict mean reversion" (predict opposite of last bar)
- Simple momentum: "predict up if 5-bar return > 0"

### 1.10 MEDIUM: 5bps no-trade band threshold is very tight relative to noise

5 bps = 0.0005. For 5-minute bars on large-cap stocks, typical one-bar return standard deviation is 8-15 bps. The standard deviation of the 12-bar average return is approximately `sigma / sqrt(12)` ≈ 2.3-4.3 bps. A threshold of 5 bps is approximately **1.2-2.2 standard deviations** away from zero.

Under a normal distribution, roughly 12-23% of bars would be no-trade based on statistical noise alone. But your smoke report shows only 5.7% of bars getting directional labels — most bars are lost to missing data and boundary invalidation, not the threshold itself.

**The more concerning issue:** The bars that survive the no-trade filter are the most extreme current-bar returns. The model is being asked to predict future direction for bars that just experienced an unusual move. The features at decision time (which include the current bar's return and volatility) already encode this extremeness. The model may simply be learning "this bar had a large positive return → predict up" (trend continuation) or the opposite (mean reversion), rather than any genuine predictive signal.

---

## REVIEW DIMENSION 2: FEATURE ENGINEERING & ML DESIGN

*Auditor: Quantitative finance ML specialist. Examined every feature for mathematical correctness, causal validity, redundancy, information content, and statistical properties.*

### 2.1 CRITICAL: Empirical evidence — your features are proven to fail

From the multiticker smoke report (`2026-06-02-p0-multiticker-validation-smoke-report.md`):

| Model | Macro F1 | Balanced Accuracy |
|---|---|---|
| LightGBM (baseline_v1 features) | **0.3961** | 0.5093 |
| Stratified Dummy (random from class priors) | **0.4961** | 0.4966 |
| Delta | **-0.1000** | +0.0127 |

**Your model is 10 percentage points worse than random guessing on your primary metric.** The balanced accuracy is only 1.27 percentage points above 0.50 (coin-flip territory). This is not a "needs tuning" result — this is a **"features contain effectively zero directional signal"** result.

This is the single most important finding in this entire review. Before any architectural discussion, any code quality concerns, any process critique — the core empirical question ("do these features work?") has already been answered, and the answer is **no.**

### 2.2 CRITICAL: Rolling volatility includes current bar's return in its own standard deviation

```python
frame["rolling_volatility_20"] = grouped_rolling(frame["log_return"], day, 20, "std")
```

`log_return[t]` is used in the computation of `rolling_volatility_20[t]`. The current bar's volatility is partially defined by itself. This is statistically circular:

```
std(returns[t-19:t+1]) = sqrt(mean((r_i - mean(r))^2 for i in t-19..t+1))
```

The return at time t contributes to both the mean and the squared deviation. The correct causal implementation would exclude the current bar:

```python
frame["rolling_volatility_20"] = grouped_rolling(
    frame["log_return"].shift(1), day, 19, "std"
)
```

The `BASELINE_REFERENCE.md` notes this (line 101: *"specify whether current row inclusion is desired before production"*) but does not take a position. For a thesis project, both versions must be tested and the impact on signal quantified. Neither has been done.

### 2.3 CRITICAL: `time_of_day_sin/cos` encodes only 27.1% of the unit circle

```python
minute_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
frame["time_of_day_sin"] = np.sin(2.0 * np.pi * minute_of_day / (24 * 60))
frame["time_of_day_cos"] = np.cos(2.0 * np.pi * minute_of_day / (24 * 60))
```

The trading day runs from 09:30 (minute 570) to 16:00 (minute 960). On the full 24-hour unit circle (1440 minutes):

- Start angle: 2π × 570/1440 = 2.487 rad (142.5°)
- End angle: 2π × 960/1440 = 4.189 rad (240.0°)
- Arc used: 4.189 - 2.487 = 1.702 rad (**97.5° out of 360° = 27.1%**)

The sin/cos values are compressed into a narrow range:
- sin: [sin(2.487), sin(4.189)] = [0.607, -0.866]
- cos: [cos(4.189), cos(2.487)] = [-0.500, -0.795]

The two features are **nearly collinear** over the trading window because the arc is less than a quarter circle and does not cross axes. Any model learning a linear combination of these two features effectively learns **one weak feature.**

**The correct encoding** maps market hours to the full 2π range:

```python
minutes_since_open = minute_of_day - 570              # 0 to 390
sin_time = np.sin(2 * np.pi * minutes_since_open / 390)  # full cycle
cos_time = np.cos(2 * np.pi * minutes_since_open / 390)
```

### 2.4 CRITICAL: `normalized_macd_hist / close` is price-level-dependent, defeating normalization

```python
frame["normalized_macd_hist"] = (macd - signal) / close
```

The MACD histogram `(macd - signal)` is in dollars (or dollar-equivalent units). Dividing by `close` converts it to a percentage-like quantity, but:

- A stock at $200 with a MACD hist of $0.50 → `normalized_macd_hist = 0.0025`
- A stock at $7 with a MACD hist of $0.50 → `normalized_macd_hist = 0.0714`

**The same dollar-level signal is 28x larger for CSCO than for MSFT.** This is the opposite of normalization — it **amplifies** cross-stock differences. Even within a single stock, as CSCO trends from $7 to $60 over 22 years, the same MACD oscillation produces systematically smaller `normalized_macd_hist` values at higher prices.

The correct normalization divides by `ema_26` (the baseline from which MACD measures deviation):

```python
frame["normalized_macd_hist"] = (macd - signal) / ema_26
```

This produces a true percentage deviation: "MACD histogram as a fraction of the reference price level."

### 2.5 CRITICAL: `rsi_14` is not RSI-14 — uses SMA, not Wilder's smoothing

```python
avg_gain = grouped_rolling(gain, day, 14, "mean")  # THIS IS SMA
avg_loss = grouped_rolling(loss, day, 14, "mean")  # THIS IS SMA
rs = avg_gain / avg_loss.replace(0.0, np.nan)
frame["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
```

**Wilder's RSI uses an exponential moving average** (Wilder's smoothing), not a simple moving average:

```
Wilder: avg_gain[t] = (13 * avg_gain[t-1] + gain[t]) / 14
Your code: avg_gain[t] = (gain[t-13] + gain[t-12] + ... + gain[t]) / 14
```

These produce different numerical values with different stationarity properties. The SMA version gives equal weight to all 14 observations; Wilder's gives exponentially more weight to recent bars. The feature is named `rsi_14` but it is not RSI-14. This is either a naming error (if SMA was intentional) or a methodological error (if the research motivation was "standard technical indicator" which implies Wilder's).

Additionally, `avg_loss.replace(0.0, np.nan)` means that any 14-bar period where the average loss is exactly zero (e.g., 14 consecutive up bars) produces NaN RSI. In the standard Wilder formulation, this corresponds to RSI=100 (since RS → ∞). Your implementation drops these rows instead.

### 2.6 CRITICAL: `bollinger_pctb` — division by zero when volatility collapses

```python
frame["bollinger_pctb"] = (close - lower_band) / (upper_band - lower_band)
```

Where `upper_band - lower_band = (MA20 + 2*std) - (MA20 - 2*std) = 4 * rolling_std_20`.

When `rolling_std_20 == 0` (20 consecutive bars at the same price — which happens intraday for low-volume stocks, especially during lunch hours for consumer staples like KO), the denominator is zero. Pandas produces `inf` for float division by zero. This then propagates through `StandardScaler.transform()`, producing extreme values. The NaN check in `build_windows_for_segment` catches `inf` values (they're not NaN) but doesn't catch them — they pass through as valid features. **A model trained on windows containing `inf` values will produce garbage predictions.**

For KO (Coca-Cola), 20-bar flat periods during lunch hours over 22 years are a near-certainty. This is a deterministic crash, not a theoretical edge case.

### 2.7 CRITICAL: Feature redundancy — at most 3-4 independent dimensions out of 10 features

| Group | Features | Shared Basis | Unique Signal? |
|---|---|---|---|
| Returns | `log_return`, `close_to_open_return` | Both close[t] vs prior | ~1.5 dims (r ≈ 0.6-0.8) |
| Volatility | `rolling_volatility_20`, `bollinger_pctb` | Same `rolling_std_20(close)` | ~1 dim (near-perfect correlation) |
| Overbought | `rsi_14`, `bollinger_pctb` | Both close position vs range | ~0 dim (already counted above) |
| Volume | `normalized_volume_20` | Standalone (but self-ref, Finding 2.2) | Degraded |
| Time | `time_of_day_sin`, `time_of_day_cos` | Collinear (Finding 2.3) | ~0.5 dim |
| Momentum | `normalized_macd_hist` | Standalone (but 44% lost, Finding 1.3) | Degraded |
| Range | `high_low_range` | Standalone (but no causal link, Finding 2.8) | Questionable |

**Analysis of each pair:**

- `rolling_volatility_20` and `bollinger_pctb` share the exact same `rolling_std_20` as their core component. `bollinger_pctb` = `(close - MA20) / (2 * rolling_std_20)`, which is a z-score using the same std as `rolling_volatility_20`. The conditional mutual information between these two given either one is near zero.

- `log_return` and `close_to_open_return`: `log_return` = log(close[t]/close[t-1]), `close_to_open_return` = close[t]/open[t] - 1. For intraday bars, the close-open spread is typically 2-5 bps, so `close_to_open_return` ≈ log(close[t]/open[t]). Both measure the same directional movement with different baselines. Expected correlation: 0.6-0.8 on 5-min bars.

- `rsi_14` measures (via gain/loss ratios) the same "price position relative to recent range" that `bollinger_pctb` measures more directly. Both are overbought/oversold indicators that will be highly correlated.

**Effective dimensionality: ~3-4 independent features.** The rest are redundant, degenerate (MACD warmup), incorrectly constructed (time encoding, volume self-reference), or causally unconnected to the label (high_low_range).

### 2.8 MEDIUM: `high_low_range` has zero theoretical or empirical link to 60-minute future direction

`high_low_range = log(high[t] / low[t])` measures intra-bar volatility within a 5-minute window. There is no financial theory — behavioral, microstructure, or risk-based — that predicts future 60-minute direction from the previous 5-minute bar's range.

This feature was likely included as part of a grab-bag of common technical indicators with no stated causal mechanism. It exemplifies the broader problem: **the feature set was assembled by convention, not by hypothesis.** Each feature should have a documented reason for why it might predict the specific label at the specific horizon.

### 2.9 MEDIUM: 5bps no-trade band — 94% of bars discarded, selection bias introduced

From the smoke report: 18,988 valid validation labels out of 332,414 validation rows = **5.7% labeled**. The other 94.3% are either neutral (within ±5bps band, ~90%) or invalid (missing data, cross-day, cross-split, ~4.3%).

The bars that survive the label filter are the most extreme current-bar returns. This creates a **selection bias:** the model is trained only on instances where the bar just completed had an unusually large directional move. The features at decision time (which include the current bar's return, volatility, volume) already encode this extremeness. The model may simply be learning to recognize "this bar was already extreme" rather than any predictive signal about the next 12 bars.

**A 3-class formulation (up/neutral/down) would use all data** and avoid this selection bias. A regression target on `future_avg_return` (once the label is fixed per Finding 1.2) would be even better.

### 2.10 MEDIUM: `close_to_open_return` mixes overnight gap with intraday drift

For the 09:35 bar (first completed 5-min bar): `close_to_open_return = close[09:35] / open[09:35] - 1`. The open[09:35] is the opening auction price, reflecting overnight information and pre-market activity. This is fundamentally different from the open of any subsequent intraday bar. The feature has a **regime shift between the first bar and all subsequent bars each day** that is not documented or handled.

### 2.11 MEDIUM: No feature importance, no ablation study, no signal detection was ever run

Ten features were defined and frozen without any evidence that each contributes independent signal. No analysis was conducted:
- No mutual information computation between each feature and label
- No SHAP/LIME/permutation importance
- No recursive feature elimination
- No ablation study removing one feature at a time

The LightGBM smoke that produced F1=0.396 could have immediately output feature importance. This would have revealed which (if any) features carry signal and at what magnitude. Instead, the project moved on to writing helper extraction specs and notebook skeleton sections.

**This is the root process failure:** the project built infrastructure before testing whether the signal exists.

### 2.12 MEDIUM: Macro F1 is a poor primary metric for a trading classification problem

For binary classification with near-balanced classes (smoke report shows ~51/49), Macro F1 ≈ standard binary F1. F1 ignores true negatives entirely. A model with F1=0.50 could lose money; a model with F1=0.45 could be profitable if the misclassifications occur on small moves.

For a trading problem, economically meaningful metrics include:
- **Matthews Correlation Coefficient (MCC):** only metric that produces high score when all four confusion matrix entries are good
- **Cohen's kappa:** directly answers "can we beat chance?"
- **Expected return per trade / Sharpe ratio of signals**
- **Precision at top-k confidence:** if only the most confident predictions are traded

None of these are implemented or even mentioned.

### 2.13 MEDIUM: No transaction cost model, slippage, or market impact

The 5bps no-trade band is a gross-return threshold. Actual profitability depends on:
- **Bid-ask spread:** 1-5bps for liquid stocks, but historically up to 20bps for CSCO in the 1990s
- **Slippage:** delay between signal generation and fill
- **Market impact:** for position sizes large enough to matter
- **Commissions:** negligible now but nonzero historically (especially pre-2000s)

None of these are modeled. The project classifies based on gross return direction, which is a necessary but far from sufficient condition for a trading strategy.

### 2.14 MEDIUM: `window_size = horizon_k = 12` — no theoretical justification for symmetric lookback/horizon

There is no theoretical reason that the optimal lookback window for predicting 60-minute direction should be exactly 60 minutes. The efficient market hypothesis in its weakest form predicts that whatever signal exists should decay exponentially with horizon. Microstructure theory suggests lookback should be either longer than horizon (to capture regime context) or shorter (to capture very recent order flow dynamics). Setting them equal is an arbitrary symmetry assumption.

---

## REVIEW DIMENSION 3: CODE ARCHITECTURE & QUALITY

*Auditor: Senior Python code reviewer. Examined architecture, error handling, coupling, memory management, API design, dependency management, and code quality patterns.*

### 3.1 CRITICAL: Silent failures that produce empty/incorrect results with zero indication

These five failure modes are all silent — they produce plausible-looking empty or wrong results without any warning, log message, or error:

**Failure 1: `transform_train_and_validation` silently produces all-NaN scaled columns**

```python
# baseline_v1.py, lines 81-98
def transform_train_and_validation(split_frames_by_ticker, scaler, feature_columns):
    for ticker, frame in split_frames_by_ticker.items():
        ...
        rows = eligible & complete
        if rows.any():
            current.loc[rows, scaled_columns] = scaler.transform(...)
        # else: scaled_columns remain NaN for ALL rows of this ticker
```

If a ticker has zero rows where `split ∈ {"train", "validation"}` AND all features are complete, the `if rows.any():` branch is skipped. The scaled columns remain NaN for the entire ticker. `build_windows_for_segment` checks that scaled columns *exist* (line 106) but not whether they are *all NaN*. The `window[scaled_columns].isna().any().any()` check rejects every window, producing empty arrays. No warning is emitted.

**Failure 2: `build_windows_for_segment` returns empty arrays with no warning**

When every window in a segment is invalid (all NaN labels, all NaN features, or cross-day violations), the function returns `{"X": np.empty((0, ...)), "y": empty_array, "metadata": empty_df}`. The caller `build_windows_by_ticker_and_split` stores it. Downstream code like `pooled_train_validation_labels` concatenates these into empty results. Zero indication of data loss.

**Failure 3: `evaluate_stratified_dummy` crashes on single-class labels**

If any ticker's labels collapse to a single class (possible with thin data + 5bps threshold), `f1_score(..., average="macro")` raises `ValueError` because F1 for the missing class is undefined. The notebook's `pooled_train_validation_labels` concatenates across tickers, masking per-ticker failures — but the helper itself is a landmine for per-ticker callers.

**Failure 4: `fit_train_only_scaler` has no defense against constant features**

If a feature is constant across all train data (e.g., `time_of_day_sin` in a narrow window), `StandardScaler` produces zero variance → `scale_` = 0 → division by zero during transform → `inf` values. `inf` passes through NaN checks. The model receives `inf` as a valid feature value.

**Failure 5: `bollinger_pctb` division by zero (Finding 2.6)**

`upper_band - lower_band = 4 * rolling_std_20`. When std = 0 for 20 consecutive bars, denominator is zero → `inf`. This is guaranteed to occur for low-volatility stocks (e.g., KO at lunch). The `inf` passes through NaN checks.

### 3.2 CRITICAL: Functions are coupled by undocumented execution order, not by code contracts

`add_split_and_invalidate_boundaries` directly accesses `current["future_avg_return"]` on line 61:

```python
current["invalid_cross_split"] = current["future_avg_return"].notna() & (...)
```

The column `future_avg_return` is created by `make_no_trade_band_labels`. There is:
- No docstring documenting this dependency
- No precondition assertion (`assert "future_avg_return" in frame.columns`)
- No explicit coupling contract between the two functions

If anyone calls `add_split_and_invalidate_boundaries` on a frame that hasn't been through `make_no_trade_band_labels` first, it crashes with `KeyError`. The pipeline is held together by **notebook cell execution order convention**, not by code. This is the kind of implicit coupling that causes silent bugs when the pipeline is reordered, refactored, or scripted outside the notebook.

### 3.3 CRITICAL: `evaluate_stratified_dummy` drops NaN labels that never exist in the current pipeline

```python
def evaluate_stratified_dummy(y_train, y_validation, seeds=(41, 42, 43, 44, 45)):
    y_train = y_train[~pd.isna(y_train)]         # dead code
    y_validation = y_validation[~pd.isna(y_validation)]  # dead code
```

`build_windows_for_segment` line 120 already skips windows where `pd.isna(target["label"])`. These NaN labels never reach `evaluate_stratified_dummy` in the current pipeline. The NaN-dropping is either:
- **Dead code** (if the pipeline is the only caller) — misleading maintenance burden
- **An undocumented assumption** that external callers might pass dirty labels — dangerous ambiguity

### 3.4 CRITICAL: `make_no_trade_band_labels` has a mutant contract — both mutates AND returns the frame

```python
def make_no_trade_band_labels(frame, horizon_k, threshold_bps):
    _require_single_ticker_frame(frame)
    current = frame.sort_values("timestamp").copy()  # creates copy
    ...
    return current
```

The function calls `.copy()` (creating a new object), mutates that copy, and returns it. The caller cannot know from the signature whether the input frame was modified or not. This is an **ambiguous contract**. In practice:

- `make_no_trade_band_labels` makes one copy
- `add_split_and_invalidate_boundaries` calls `.copy()` again on the returned frame — second copy
- `transform_train_and_validation` calls `.copy()` — third copy
- `build_windows_for_segment` calls `.copy()` — fourth copy

For 5 tickers × 440K rows × 10+ columns, this is **meaningful memory pressure from redundant defensive copies.** If someone removes `.copy()` from one function thinking another handles it, they get silent in-place mutation bugs.

### 3.5 HIGH: `_require_single_ticker_frame` is fragile and inconsistently placed

```python
def _require_single_ticker_frame(frame):
    if "ticker" in frame.columns and frame["ticker"].nunique(dropna=False) > 1:
        raise ValueError("Expected a single ticker frame.")
```

**Fragility 1:** A frame where `ticker` column exists but all values are `None`/`NaN` passes this check. `nunique(dropna=False)` counts `NaN` as one unique value, so a 440K-row frame with all-NaN ticker passes.

**Fragility 2:** The guard is called manually in 3 of 5 functions that need it (`make_no_trade_band_labels`, `add_split_and_invalidate_boundaries`, `build_windows_for_segment`). If a new function is added to the pipeline, someone must remember to add the guard. A decorator pattern would be self-documenting:

```python
@require_single_ticker
def make_no_trade_band_labels(frame, ...):
    ...
```

### 3.6 HIGH: `make_no_trade_band_labels` unnecessarily materializes a wide intermediate DataFrame

```python
future_returns = [one_bar_return.shift(-1 - offset) for offset in range(int(horizon_k))]
current["future_avg_return"] = pd.concat(future_returns, axis=1).mean(axis=1, skipna=False)
```

For horizon_k=12 on a 440K-row frame, this creates a 440K × 12 intermediate DataFrame (~42 million float elements as temporary garbage) just to compute a row-wise mean. For 5 tickers, that's ~210 million temporary elements. Equivalent operation without materialization:

```python
future_avg = sum(one_bar_return.shift(-1 - i) for i in range(k)) / k
```

### 3.7 HIGH: `requirements.txt` lists dead dependencies; missing test dependency

**Listed but never imported in active code:**
- `torch==2.12.0+cpu` — ~800MB on disk, never used
- `lightgbm==4.6.0` — never used in active module or notebook cells

**Used in tests but not listed:**
- `nbformat` — imported in `test_notebook_static_gate.py`, not in requirements.txt

A fresh `pip install -r requirements.txt` **cannot run the test suite.** This is a basic environment reproducibility failure. Additionally, no transitive dependencies are pinned — `scikit-learn` pulls `scipy`, `joblib`, `threadpoolctl` at whatever versions pip resolves. Environment is not reproducible.

### 3.8 HIGH: `test_notebook_forbids_raw_feature_fallback_without_implementing_it` tests documentation, not behavior

```python
def test_notebook_forbids_raw_feature_fallback_without_implementing_it():
    notebook_text = NOTEBOOK_PATH.read_text()
    assert "raw feature fallback for model windows" in notebook_text
    assert "available_columns" not in notebook_text
```

This test asserts:
1. A specific English phrase appears in a markdown cell
2. A specific variable name (`available_columns`) is not used anywhere in the notebook

If someone rewords the markdown, the test breaks. If someone uses a different variable name (e.g., `allowed_cols`), the test passes but the forbidden behavior is present. **This is documentation testing masquerading as a code gate.** It tests nothing about actual code execution or behavior.

### 3.9 HIGH: Zero integration tests — all tests use synthetic 3-8 row dataframes

Every single test constructs tiny synthetic dataframes:

```python
def make_one_ticker_frame():
    return pd.DataFrame({
        "ticker": ["AAA"] * 8,
        "timestamp": pd.to_datetime([...8 timestamps...]),
        ...
    })
```

No test:
- Loads actual CSV data from `data/`
- Exercises the full pipeline end-to-end (features → labels → splits → scale → windows → model)
- Validates against known expected outputs on real ticker data
- Tests edge cases that only emerge at scale (memory pressure, numerical stability on real prices with actual gaps, actual volume patterns)

The archived test suite (`archive/legacy_model_runner_reference/tests/`) had comprehensive integration tests. The rebuild has none.

### 3.10 HIGH: No test for edge cases that will certainly occur in production

The following scenarios are untested:
- Empty frames (0 rows)
- Single-row frames
- All-NaN feature columns
- Single-class labels (all zeros or all ones)
- Extreme numerical values (prices during 2008 crisis, COVID crash)
- Ticker with all NaN values in key columns
- Duplicate timestamps
- Out-of-order timestamps
- The `if rows.any():` False branch in `transform_train_and_validation`
- The empty-return path from `build_windows_for_segment`
- Missing `future_avg_return` column when calling `add_split_and_invalidate_boundaries` out of order

### 3.11 MEDIUM: `__init__.py` defines zero public API

```python
"""Small active helpers for intraday stock direction research."""
```

No imports, no `__all__`, no re-exports. Users must import from `intraday_research.baseline_v1` directly, which exposes the private `_require_single_ticker_frame` alongside the 8 public functions. There is no distinction between public and private API surface.

### 3.12 MEDIUM: No type annotations on function parameters

```python
def transform_train_and_validation(split_frames_by_ticker, scaler, feature_columns):
    # What type is split_frames_by_ticker? Dict[str, pd.DataFrame]?
    # What type is scaler? StandardScaler? Any sklearn transformer?
    # What type is feature_columns? List[str]? Tuple[str, ...]?
```

The dict `split_frames_by_ticker` has no documented structure. Callers must read the implementation to understand the expected key/value types. The `splits` dict in `assign_calendar_split` and `add_split_and_invalidate_boundaries` has an ad-hoc structure with three fixed keys that is enforced only by `pd.Timestamp()` parsing — if someone passes dates in the wrong format, it fails at runtime with an unhelpful error.

### 3.13 MEDIUM: Magic number `10000.0` with no explanation

```python
threshold = threshold_bps / 10000.0
```

Basis points to decimal conversion is `bps / 10000`. There's no named constant (`BPS_TO_DECIMAL = 10000.0`), no comment, and no documentation. If someone passes a threshold already in decimal form (e.g., 0.0005), the function silently produces a threshold 10,000× too small, classifying everything as directional.

### 3.14 MEDIUM: `int(horizon_k)` called inline 3 times instead of once at function entry

```python
for offset in range(int(horizon_k)):          # line 25
for offset in range(1, int(horizon_k) + 1):   # line 33
horizon_split = current["split"].shift(-int(horizon_k))  # line 60
```

If `horizon_k` is passed as a float (e.g., `12.0`), the casts work but signal that the parameter was never validated. A single `horizon_k = int(horizon_k)` at function entry would be cleaner and more defensive.

### 3.15 MEDIUM: `.gitignore` has gaps for a data-science project

Missing patterns:
- `*.h5`, `*.parquet` — processed data files
- `*.pt`, `*.pth` — PyTorch models (though torch is unused — inconsistent)
- `*.log` — run logs
- `notebooks/**/*.executed.*` — executed notebook outputs
- `results/`, `output/` — experiment output directories
- `wandb/`, `mlruns/` — experiment tracking artifacts

---

## REVIEW DIMENSION 4: PROJECT STRUCTURE & PROCESS

*Auditor: Project management and process specialist. Examined documentation-to-code ratio, process quality, archive management, checkpoint hygiene, and project identity coherence.*

### 4.1 CRITICAL: Nothing is runnable end-to-end

**A fresh clone of this repository produces zero research output.** The only active Python code consists of 8 helper functions in a single file. There is:

- **No entry point:** No `main()`, no CLI, no `__main__.py`, no script
- **No data loading:** The notebook has `load_ticker_csv` and `audit_all_tickers` defined, but `RUN_DATA_LOAD = False` and the functions are never called
- **No training loop:** No model training code exists in the active codebase
- **No evaluation harness:** The notebook's model cell raises `NotImplementedError`
- **No pipeline orchestration:** Nothing connects `data/CSCO.csv` → features → labels → splits → scaler → windows → model → metrics

The only code that ever ran end-to-end (`archive/legacy_model_runner_reference/`) is explicitly forbidden. The ratio of "words about what to do" to "code that produces results" is **effectively infinite** — zero lines of new code produce an end-to-end result.

### 4.2 CRITICAL: Documentation-to-code ratio of 12.8:1

| Artifact | Lines | Can produce a result? |
|---|---|---|
| `intraday_research/baseline_v1.py` | 195 | No (helpers only, no orchestrator) |
| `intraday_research/__init__.py` | 1 | No |
| `tests/test_baseline_v1_helpers.py` | 327 | No (tests pass, but test synthetic data only) |
| `tests/test_notebook_static_gate.py` | 94 | No (tests notebook structure, not behavior) |
| `AGENTS.md` | 207 | No |
| `README.md` | 20 | No |
| `docs/RESEARCH_WORKFLOW.md` | 280 | No |
| `docs/BASELINE_REFERENCE.md` | 277 | No |
| `docs/ENVIRONMENT.md` | 25 | No |
| `docs/rebuild_specs/*.md` (9 files) | 1,659 | No |
| **Total functional code** | **195** | |
| **Total documentation** | **~2,500** | |
| **Ratio** | **12.8:1** | |

**The rebuild design doc (256 lines) is longer than the module it governs (194 lines).** The rebuild plan (367 lines) is nearly double the functional code. The P1 helper test plan (451 lines) is more than double the helper module itself.

### 4.3 CRITICAL: The rebuild is an 85:1 regression from the archive

The `archive/legacy_model_runner_reference/` contains **16,612 lines of Python code** that:

| Component | Lines | Status in rebuild |
|---|---|---|
| CLI runner (`local_baseline_matrix.py`) | 2,965 | **Deleted** |
| 4 model architectures (LSTM, TCN, DLinear, MS-DLinear+TCN) | 713 | **Deleted** |
| Dataset pipeline with leakage guards | 500 | **Replaced by 195 lines** |
| Trainer with checkpoint management | 267 | **Deleted** |
| Metrics evaluation | 132 | **Replaced by 30 lines** |
| Paper-table builders | 784 | **Deleted** |
| Colab runner | 1,254 | **Deleted** |
| Comprehensive tests | ~3,000 | **Replaced by 421 lines** |
| Active notebook (executed) | 1 | **Replaced by skeleton** |
| Checkpoint management | 66 | **Deleted** |
| Profiling | 215 | **Deleted** |

**The "clean rebuild" replaced all of this with 195 lines of disconnected helpers.** The archive code was working — it had a CLI, ran models, produced 519 checkpoint directories of output, and had comprehensive tests. The rebuild has none of these capabilities.

The AGENTS.md and README.md insist the archive is "historical reference only." But the archive is **objectively better code** — it ran, it was tested, it had multiple models, and it proved the pipeline could produce results. **The rebuild is a strictly worse version of the same project.**

### 4.4 CRITICAL: The rebuild recreated the PM document machine under a new naming convention

AGENTS.md Section 6 explicitly forbids:

> *"Do not create new PM_NNN_*, handoff, readiness, session-context, or closeout documents for ordinary research progress."*

Yet the rebuild created **9 documents under `docs/rebuild_specs/`** that are functionally identical to the forbidden PM documents:

| Document | Lines | PM Equivalent |
|---|---|---|
| `2026-06-02-notebook-first-rebuild-design.md` | 256 | PM Design Document |
| `2026-06-02-notebook-first-rebuild-plan.md` | 367 | PM Implementation Plan |
| `2026-06-02-helper-extraction-readiness.md` | 134 | PM Readiness Assessment |
| `2026-06-02-p0-validation-smoke-report.md` | 90 | PM Smoke Report |
| `2026-06-02-p0-multiticker-validation-smoke-report.md` | 122 | PM Smoke Report |
| `2026-06-02-p1-helper-test-plan.md` | 451 | PM Test Plan |
| `2026-06-02-staging-candidates-and-closeout.md` | 158 | PM Closeout Document |
| `2026-06-02-validation-only-preregistration-template.md` | 81 | PM Gate Template |

**The only difference is the naming convention: `2026-06-02-*` instead of `PM_###_*`.** The PM document machine was not dismantled — it was rebranded.

### 4.5 CRITICAL: The "notebook-first" claim is false

The AGENTS.md, README.md, and all rebuild specs describe the project as "notebook-first." A notebook-first workflow means:

1. Open a notebook
2. Write code cells
3. Run them
4. See results
5. Iterate
6. Extract reusable helpers only after patterns emerge

**What actually happened:**
1. The notebook was assembled programmatically via `nbformat` by an agent
2. It has 17 sections with markdown headers and empty code cells
3. Every `RUN_*` flag defaults to `False`
4. Every `execution_count` is `None`
5. The model cell raises `NotImplementedError`
6. Zero cells have been executed — by a human or by code

**This is a static document, not a research artifact.** The notebook was constructed to *look like* a research notebook, not created through the iterative process of doing research.

### 4.6 HIGH: AGENTS.md rules have zero enforcement — nothing runs

AGENTS.md Section 3 specifies 14 hard research rules across 6 sub-sections:

- 3.1 Chronology and Leakage (7 rules)
- 3.2 Evaluation Honesty (7 rules)
- 3.3 Failure Behavior (5 rules)

These rules are well-written and correct in principle. They are also **completely unenforceable** because:
- Nothing loads data, so split chronology can't be checked against actual timestamps
- Nothing trains models, so evaluation honesty rules can't be violated or verified
- No holdout/test is ever accessed, so the closed-holdout rule can't be broken

**The rules are aspirational text with zero runtime teeth.** They exist to make the project look rigorous without the project actually being rigorous (since no rigor is needed for code that never runs).

### 4.7 HIGH: BASELINE_REFERENCE.md documents a dead baseline

The 277-line baseline reference describes in meticulous detail:
- Feature availability table with decision-time conventions
- Scaler and threshold policy with explicit CLI flags
- Model availability assessment for LightGBM and MS-DLinear+TCN
- A 3-step workflow
- A "Tiny Validation Prompt" with explicit CLI flags

The CLI flags reference `archive/legacy_model_runner_reference/scripts/local_runner_reference/local_baseline_matrix.py` — **which is in the archive and forbidden to execute.** The baseline cannot be reproduced with the new code because the new code has no runner, no CLI, and no training harness.

**The baseline reference is a 277-line ghost — it documents something that does not exist in the active project.**

### 4.8 HIGH: 519 checkpoint directories of dead artifact pollution

The `checkpoints/` directory contains **519 subdirectories, 845 files, ~15MB** of metadata.json files from the old runner. Naming patterns (`phase1b_local_legacy_binary_smoke`, `fixed_calendar_*`, `pm_lgbm_*`, `pm_ms_dlinear_tcn_*`) confirm they are old-runner output.

| Stat | Value |
|---|---|
| Total subdirectories | 519 |
| Total files | 845 |
| Disk usage | ~15 MB |
| Earliest date | 2026-05-26 |
| Latest date | 2026-05-31 |
| From new code? | Zero. New code cannot produce checkpoints |

These checkpoints are:
- **Irreproducible:** the runner that created them is forbidden
- **Useless:** policy forbids using old-runner results as evidence
- **Polluting:** every `git status` scans 519 directories

The rebuild plan explicitly lists `checkpoints/` under "Do not modify" but never addressed whether they should be archived, deleted, or left as noise.

### 4.9 HIGH: The "rebuild" produced zero research-relevant output

The rebuild plan (Task 4, Step 3: "Only then run the approved validation command") is **UNCHECKED.** Tasks 1-3 are marked complete. What did three complete tasks produce?

| Task | Output | Research value |
|---|---|---|
| Task 1 | Created 2 more process documents (design + plan) | Zero |
| Task 2 | Rewrote notebook skeleton via nbformat (17 sections) | Zero (static document) |
| Task 3 | Reviewed git diff, identified staging candidates | Zero |

**Three complete tasks, zero research output.** No data was loaded end-to-end. No features were computed on real CSVs. No labels were constructed. No model was trained or evaluated. No baseline comparison was run. No interpretation was written.

### 4.10 HIGH: Premature helper extraction — helpers extracted from a notebook that never ran

The P1 helper extraction extracted 8 functions from the notebook into `baseline_v1.py`. But the notebook had **never been executed.** The helpers were extracted from static code, not from proven patterns.

The correct sequence is:
1. Notebook cells prove the logic works on real data
2. The same logic appears in 2+ places or proves safety-critical
3. Extract to a helper module with tests

What actually happened:
1. Notebook was assembled via nbformat (never run)
2. Logic was extracted before any execution proved it correct
3. Tests were written for the extracted logic (all pass on synthetic data)

This is backwards. The helpers are tested in isolation but were never validated on real data. **The project has 21 passing tests for functions that have never processed a single real CSV row.**

### 4.11 MEDIUM: Project identity crisis — is this research or governance?

The project describes itself as:
- "A small research project" (AGENTS.md)
- "Lightweight" (user's stated goal)
- "Notebook-first" (README.md, AGENTS.md, all specs)
- "Default to one clear notebook" (AGENTS.md)

The project actually is:
- A 207-line governance document with 14 hard rules across 6 sub-sections
- A 277-line baseline reference documenting a dead runner
- A 280-line research workflow prescribing exactly how notebooks should look
- 9 rebuild spec documents totaling 1,659 lines
- A single 195-line helper module
- A notebook skeleton that has never been executed

**The project's self-description and actual behavior are in direct contradiction.** It claims to be lightweight and notebook-first, but it behaves as a process-governance documentation machine with trace amounts of untested code.

### 4.12 MEDIUM: README.md is stale and misleading

The README claims:
- "Default entry points: `notebooks/04_ian_research_memo.ipynb` — current active research notebook"
- But the notebook has never been executed, has all guards off, and raises NotImplementedError
- "Active notebook work should not depend on archived helper library or old runner scripts"
- But the active code *cannot produce any result* without the archived runner or a new orchestrator

### 4.13 MEDIUM: Environment documented but never validated end-to-end

`ENVIRONMENT.md` lists pinned package versions and a validated Python interpreter. But whether these packages actually work together to produce an end-to-end result with real data was never tested. The packages can import; whether they can run a full pipeline is unknown.

---

## SUMMARY OF ALL FINDINGS BY SEVERITY

### CRITICAL (34 findings — project-correctness threatening)

| # | Dimension | Finding |
|---|---|---|
| 1 | Validity | Label uses arithmetic mean of returns, not cumulative return — no economic interpretation |
| 2 | Validity | MACD EMA cascade destroys 43.6% of each trading day; model is afternoon-only |
| 3 | Validity | No cross-validation; single train/val split over 22 years |
| 4 | Validity | Regime mismatch: validation is easiest market in dataset, train is crisis-heavy |
| 5 | Validity | Pooled StandardScaler across 5 heterogeneous stocks and 15 years |
| 6 | Validity | `normalized_volume_20` is self-referencing — bug documented but unfixed (one-line fix) |
| 7 | Features | Empirical proof: LightGBM F1=0.396 vs Dummy F1=0.496 — features are noise |
| 8 | Features | `time_of_day_sin/cos` encodes only 27.1% of unit circle — nearly collinear |
| 9 | Features | `normalized_macd_hist / close` is price-level-dependent — amplifies cross-stock differences |
| 10 | Features | `rsi_14` uses SMA, not Wilder's smoothing — not actually RSI-14 |
| 11 | Features | `bollinger_pctb` division by zero when 20-bar volatility is zero — deterministic crash |
| 12 | Features | Effective feature dimensionality: 3-4 out of 10; rest are redundant or degenerate |
| 13 | Features | No feature importance, signal detection, or ablation study was ever run |
| 14 | Architecture | 5 silent failure modes: all-NaN columns, empty arrays, single-class crash, constant features, div-by-zero |
| 15 | Architecture | Functions coupled by undocumented execution order; `KeyError` if called out of sequence |
| 16 | Architecture | `evaluate_stratified_dummy` has dead NaN-dropping code |
| 17 | Architecture | `make_no_trade_band_labels` has ambiguous mutate+return contract |
| 18 | Architecture | `_require_single_ticker_frame` is fragile (NaN ticker passes) and inconsistently placed |
| 19 | Architecture | `requirements.txt` lists dead deps (torch 800MB, lightgbm); missing `nbformat` for tests |
| 20 | Architecture | Test `test_notebook_forbids_raw_feature_fallback` tests documentation strings, not behavior |
| 21 | Architecture | Zero integration tests; all tests use 3-8 row synthetic data |
| 22 | Architecture | No edge case tests: empty frames, single-class, extreme values, duplicate timestamps |
| 23 | Architecture | `__init__.py` defines zero public API; private helper is exposed |
| 24 | Architecture | No type annotations on function parameters |
| 25 | Architecture | Magic number `10000.0` with no explanation |
| 26 | Architecture | `int(horizon_k)` called inline 3 times instead of validated once |
| 27 | Architecture | `.gitignore` missing patterns for processed data, model files, logs, experiment tracking |
| 28 | Process | **Nothing is runnable end-to-end.** Zero new-code paths from CSV to model evaluation |
| 29 | Process | Documentation-to-code ratio = 12.8:1 (~2,500 lines docs / 195 lines code) |
| 30 | Process | Rebuild is 85:1 regression: 16,612 lines of working code → 195 lines of disconnected helpers |
| 31 | Process | 9 rebuild spec documents are PM documents rebranded under `2026-06-02-*` naming |
| 32 | Process | "Notebook-first" claim is false — notebook constructed via nbformat, never executed |
| 33 | Process | BASELINE_REFERENCE.md (277 lines) documents a dead runner that is forbidden to run |
| 34 | Process | Premature helper extraction: helpers extracted from a notebook that never ran |

### HIGH (18 findings — significant quality/design degradation)

| # | Dimension | Finding |
|---|---|---|
| 35 | Validity | `rolling_volatility_20` includes current bar in own std — statistically circular |
| 36 | Validity | Overlapping windows (91.7% overlap) inflate effective sample size |
| 37 | Validity | 5bps no-trade band: 94% of bars discarded, selection bias toward extreme bars |
| 38 | Validity | `close_to_open_return` mixes overnight gap with intraday drift on first bar |
| 39 | Features | 5bps threshold too tight for 5-min bar noise level |
| 40 | Features | `window_size = horizon_k = 12` — no theoretical justification |
| 41 | Features | Macro F1 is a poor primary metric for trading; MCC or economic metrics would be appropriate |
| 42 | Features | No transaction cost model, slippage, or market impact |
| 43 | Features | Stratified dummy is weakest possible baseline; heuristic baselines would be more informative |
| 44 | Features | Binary classification discards 94% of bars; 3-class or regression would use all data |
| 45 | Features | No calibration analysis, prediction intervals, or selective prediction |
| 46 | Process | AGENTS.md rules have zero enforcement — nothing runs to trigger violations |
| 47 | Process | 519 dead checkpoint directories (15MB) from old runner polluting working tree |
| 48 | Process | Three "complete" rebuild tasks produced zero research output |
| 49 | Process | Project identity crisis: claims "lightweight" but has more governance than code |
| 50 | Process | README.md stale: claims "active research notebook" that has never executed |
| 51 | Process | Environment documented but never validated end-to-end with real data |
| 52 | Architecture | Unnecessary wide intermediate DataFrame in `make_no_trade_band_labels` |

### MEDIUM (11 findings — methodology and design concerns)

| # | Dimension | Finding |
|---|---|---|
| 53 | Validity | Stratified dummy distribution mismatch due to regime change inflates model delta |
| 54 | Features | `high_low_range` has no causal link to 60-minute direction — included by convention |
| 55 | Features | No hyperparameter tuning protocol |
| 56 | Features | Label is binarized after no-trade filtering; soft labels would preserve information |
| 57 | Features | `bollinger_pctb` uses 2σ bands — standard but arbitrary; sensitivity not tested |
| 58 | Features | Time encoding uses minute-of-day at bar start — should it use bar midpoint? |
| 59 | Features | No analysis of label autocorrelation structure |
| 60 | Architecture | Frame copies at every pipeline stage cause redundant memory pressure |
| 61 | Architecture | `float("nan")` used inconsistently vs `np.nan` |
| 62 | Architecture | `future_avg_return` computation is wasteful: computes full average then invalidates |
| 63 | Process | Checkpoint directory structure deeply nested with redundant metadata |

### LOW (6 findings — cleanup and polish)

| # | Dimension | Finding |
|---|---|---|
| 64 | Features | `log_return` (log) in features vs `pct_change` (simple) in labels — negligible but inconsistent |
| 65 | Features | No `conftest.py` for shared test fixtures; `make_one_ticker_frame` is duplicated |
| 66 | Architecture | `requirements.txt` has no dev/test extras separation |
| 67 | Architecture | No `pyproject.toml` or build configuration |
| 68 | Process | `ENVIRONMENT.md` duplicates information that should be in a single config file |
| 69 | Process | `data/` and `checkpoints/` in `.gitignore` but `checkpoints/` contains `.py` code files |

---

## ROOT CAUSE ANALYSIS

The 65+ findings are not independent. They trace back to **five root causes:**

### Root Cause 1: The rebuild prioritized documentation over execution

The rebuild plan (367 lines) and design doc (256 lines) together took more effort than the 195-line helper module. The notebook skeleton was constructed to *look correct* rather than *be correct through execution.* The result is a project that passes static validation (no outputs, no forbidden imports, correct headings) but produces zero research results.

**Fix:** Stop writing documents. Make the pipeline run end-to-end with real data. Only write docs to record results, decisions made, and evidence gathered.

### Root Cause 2: No signal detection was performed before building infrastructure

Four notebook iterations, 9 spec documents, 21 tests, a helper module extraction, and a notebook skeleton — all built before anyone asked: "Is there actually any signal in these features at this horizon?" The one empirical test (LightGBM smoke) answered this question (answer: no), but the project treated it as a "diagnostic smoke" rather than the signal it was, and continued building infrastructure.

**Fix:** Compute mutual information between each candidate feature and the label at multiple horizons, per stock. If no MI > 0.01 exists at horizon 12, kill the 60-minute direction problem and pivot.

### Root Cause 3: The archive was locked away instead of migrated

16,612 lines of working code were declared "historical reference only" and replaced with 195 lines of disconnected helpers. The archive had bugs, but it also had a CLI, training loop, model implementations, comprehensive tests, and a history of producing results. A migration strategy would have incrementally replaced components while keeping the system runnable. Instead, the entire system was discarded and a new one was built from scratch — a classic second-system effect where the new system never catches up to the old one.

**Fix:** Either (a) use the archived runner as the starting point and incrementally improve it, or (b) commit to the rebuild by writing a working end-to-end pipeline before extracting any more helpers or writing any more docs.

### Root Cause 4: The PM document habit was rebranded, not broken

9 spec documents under `docs/rebuild_specs/2026-06-02-*` are the same document types that AGENTS.md forbids: design docs, plans, readiness assessments, smoke reports, closeout docs. The naming convention changed from `PM_NNN_*` to `YYYY-MM-DD-*` but the behavior — producing process documents instead of research results — did not.

**Fix:** Delete all 9 rebuild spec documents. The AGENTS.md, README.md, RESEARCH_WORKFLOW.md, and BASELINE_REFERENCE.md are sufficient governance. No new process documents until code produces results.

### Root Cause 5: The feature set was assembled by convention, not by hypothesis

Each of the 10 features should have a documented causal mechanism linking it to the 60-minute directional label. Instead, the features are a grab-bag of common technical indicators (RSI, MACD, Bollinger, volume normalization) with no individual justification. As a result:

- Three features share the same underlying computation (rolling_std_20)
- One feature has zero theoretical link to the label (high_low_range)
- One feature is self-referencing by design (normalized_volume_20)
- Two features are nearly collinear (time_of_day_sin/cos)

**Fix:** For each feature, write down: (1) what market phenomenon it measures, (2) why that phenomenon should predict 60-minute direction, and (3) empirical evidence (mutual information). Drop features that fail step 3.

---

## IF YOU WANT TO SALVAGE THIS PROJECT

Stop writing documentation. The next actions should be code-only, in this order:

### Phase 1: Fix the known bugs (1-2 hours)

These are all one-line or few-line fixes to errors this review identified:

1. **Fix the label:** Replace `future_avg_return` (mean of k returns) with cumulative return:
   ```python
   current["future_cumulative_return"] = current["close"].shift(-int(horizon_k)) / current["close"] - 1
   ```

2. **Fix volume self-reference:**
   ```python
   volume_mean_20 = grouped_rolling(log_volume.shift(1), day, 19, "mean")
   ```

3. **Fix time encoding:**
   ```python
   minutes_since_open = minute_of_day - 570
   sin_time = np.sin(2 * np.pi * minutes_since_open / 390)
   ```

4. **Fix MACD normalization:**
   ```python
   frame["normalized_macd_hist"] = (macd - signal) / ema_26
   ```

5. **Fix bollinger_pctb division by zero:**
   ```python
   denom = upper_band - lower_band
   denom = denom.replace(0.0, np.nan)  # or use a small epsilon
   frame["bollinger_pctb"] = (close - lower_band) / denom
   ```

### Phase 2: Run signal detection before building anything else (2-4 hours)

Before extracting another helper, writing another spec, or tuning another model:

1. Fix the bugs from Phase 1
2. Load real data for all 5 tickers
3. Compute features and the corrected label
4. For each feature and the label, compute mutual information at horizon_k = 12
5. Plot MI by feature, by ticker
6. Also compute MI at horizons 3, 6, 12, 24 to see if signal exists at ANY horizon

**If no MI > 0.01 exists at any horizon:** The 5-min bar directional classification problem is dead at this frequency. Pivot to a shorter horizon, a different label type (volatility prediction, volume forecasting, spread analysis), or a different asset class.

**If MI > 0.01 exists for some features at some horizon:** Congratulations, you've found signal. Now build the minimum viable model to exploit it.

### Phase 3: Build a runnable end-to-end pipeline (4-8 hours)

Write a single Python script (not a notebook, not a spec doc) that:
1. Loads data from `data/*.csv`
2. Computes features
3. Computes labels
4. Applies chronological splits
5. Fits scaler on train only
6. Builds windows
7. Trains ONE model (LightGBM, simplest option)
8. Evaluates on validation
9. Prints a comparison table with dummy baseline

Target: `python run_validation.py` should produce a complete result.

### Phase 4: Delete process documents

Remove the 9 rebuild spec documents. They are PM documents. You said you'd stop creating them. Keep AGENTS.md, README.md, RESEARCH_WORKFLOW.md, BASELINE_REFERENCE.md, and ENVIRONMENT.md. That's sufficient governance for a project this size.

### Phase 5: Clean up the repository

1. Delete or archive the 519 old-runner checkpoint directories
2. Remove `torch` and `lightgbm` from `requirements.txt` (add back only when actually used)
3. Add `nbformat` to `requirements.txt` (it's used in tests)
4. Add `.gitignore` patterns for `.h5`, `.parquet`, `.pt`, `.log`, `results/`, `output/`
5. Create a `conftest.py` with shared test fixtures

### What NOT to do:

- Do NOT write another spec document
- Do NOT extract more helpers until the pipeline runs end-to-end
- Do NOT add more notebook sections
- Do NOT create a preregistration template
- Do NOT write a "response to review" document — fix the code instead
- Do NOT rename anything — renaming is not fixing

---

*Review conducted 2026-06-02 via four parallel adversarial agents. This document records findings without sugar-coating. The goal is to help the project become a real research project that produces honest, reproducible results — not to criticize for its own sake.*
