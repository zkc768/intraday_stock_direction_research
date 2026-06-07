# Notebook 06/07 Intraday Concentration Guardrails Materials - 2026-06-05

Scope: KB-ready research note for planned Notebook 06/07 selective/no-trade
diagnostics. This note supports concentration guardrails for high-confidence
retained subsets. It does not authorize training, notebook execution,
holdout/test access, feature changes, threshold search, or coverage selection.

## Short Plan

1. Treat Notebook 06 selective/no-trade outputs as validation-only diagnostic
   artifacts, not a new model-selection surface.
2. Add concentration checks for ticker, date, time-of-day bucket, open/close,
   regime proxy, and overlapping-window dependence.
3. Use guardrails only to downgrade or qualify wording when retained rows or
   apparent gains concentrate in narrow slices. Do not use them to pick the best
   coverage level.
4. Carry the same concentration summary into Notebook 07 so the final
   validation-only synthesis distinguishes broad signal from localized behavior.

## Project Fit

Existing Notebook 06/07 materials already require fixed coverage grids,
same-row dummy baselines, `delta_macro_f1_vs_dummy`, no holdout/test contact,
and cautious calibration wording. The missing piece is a stronger diagnostic
layer for cases where a high-confidence subset looks better only because it is
mostly one ticker, one date cluster, one intraday time bucket, or one
high-volatility/open-close regime.

The guardrail should answer:

> Is selective/no-trade performance spread across the validation surface, or is
> it concentrated in a small slice that should lower the strength of the claim?

It should not answer:

> Which coverage level should we choose after seeing validation results?

## Must Sources

| Source | Link / DOI | Why it matters here | Use in Notebook 06/07 | Cannot support |
|---|---|---|---|---|
| Wood, McInish, and Ord, "An Investigation of Transactions Data for NYSE Stocks" (Journal of Finance, 1985) | https://digitalcommons.memphis.edu/facpubs/11516/ ; DOI: https://doi.org/10.1111/j.1540-6261.1985.tb04996.x | Minute-level NYSE evidence that beginning/end-of-day behavior differs from the rest of the day, with high returns and standard deviations at open/close and reduced autocorrelation after omitting those effects. | Add `open_close_concentration`, `time_of_day_bucket_share`, and separate open/close wording in retained-subset diagnostics. | Does not prove this project's five-stock 5-minute labels are predictive or tradable. |
| Andersen and Bollerslev, "Intraday periodicity and volatility persistence in financial markets" (Journal of Empirical Finance, 1997) | DOI: https://doi.org/10.1016/S0927-5398(97)00004-2 ; metadata: https://scholars.duke.edu/publication/761297 | Shows strong intraday periodicity in high-frequency return volatility and that ignoring it affects inferred dynamics. | Justifies reporting retained rows by time-of-day bucket and comparing retained share to eligible validation share by bucket. | Does not validate a classifier, no-trade threshold, or profitability claim. |
| Admati and Pfleiderer, "A Theory of Intraday Patterns: Volume and Price Variability" (RFS, 1988) | https://academic.oup.com/rfs/article/1/1/3/1601212 ; DOI: https://doi.org/10.1093/rfs/1.1.3 | Provides market-microstructure rationale for endogenous concentrated trading patterns. | Use as conceptual support that time-of-day concentration is not an incidental nuisance; it can reflect liquidity/information timing. | Does not say a high-confidence subset concentrated at open/close is robust alpha. |
| Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure" (Ecography, 2017) | DOI: https://doi.org/10.1111/ecog.02881 ; open metadata: https://colab.ws/articles/10.1111%2Fecog.02881 | Structured dependence can make random validation underestimate error; block/structured validation is often more appropriate when dependencies exist. | Use for blocked/day/ticker caution and for avoiding iid row-level interpretation of retained windows. | Ecology examples are not finance evidence; it does not provide a ready-made intraday split for this project. |
| Politis and Romano, "The Stationary Bootstrap" (JASA, 1994) | DOI: https://doi.org/10.1080/01621459.1994.10476870 | Block bootstrap family for weakly dependent stationary observations. | If Notebook 07 reports uncertainty intervals, prefer block/stationary or day-level resampling over iid row bootstrap. | Does not make overlapping 5-minute windows independent or provide a confirmatory test by itself. |
| Kuensch, "The Jackknife and the Bootstrap for General Stationary Observations" (Annals of Statistics, 1989) | DOI: https://doi.org/10.1214/aos/1176347265 ; metadata: https://cir.nii.ac.jp/crid/1362544419497266176 | Classic block jackknife/bootstrap treatment for stationary dependent observations. | Support the warning that overlapping windows reduce independent information and iid intervals are too optimistic. | Does not give a plug-in threshold for valid retained subset size in this project. |
| Hansen and Hodrick, "Forward Exchange Rates as Optimal Predictors of Future Spot Rates" (JPE, 1980) | DOI: https://doi.org/10.1086/260910 ; metadata: https://econpapers.repec.org/article/ucpjpolec/v_3a88_3ay_3a1980_3ai_3a5_3ap_3a829-53.htm | Econometric anchor for overlapping observations and adjusted inference. | Cite as a finance/econometrics warning that overlap in labels/windows changes uncertainty accounting. | The exchange-rate setting and standard-error method do not directly solve classification F1 inference. |
| U.S. DOJ/FTC Herfindahl-Hirschman Index explanation | https://www.justice.gov/atr/herfindahl-hirschman-index | Official explanation of HHI as sum of squared shares, a simple concentration measure. | Use HHI-style concentration for retained ticker/date/time shares and convert to effective number via `1 / HHI`. | Antitrust thresholds are not scientific thresholds for this research; do not import DOJ cutoff values. |

## Useful Sources

| Source | Link / DOI | Why it matters here | Use in Notebook 06/07 | Cannot support |
|---|---|---|---|---|
| Harris, "A transaction data study of weekly and intradaily patterns in stock returns" (JFE, 1986) | DOI: https://doi.org/10.1016/0304-405X(86)90044-9 ; publisher: https://www.sciencedirect.com/science/article/pii/0304405X86900449 | Documents intraday and weekday effects, including first-45-minute behavior and last-trade effects. | Support first-30/45-minute and close-bucket diagnostics. | Does not validate this project's exact date range, five tickers, or 5-minute label rule. |
| Heston, Korajczyk, and Sadka, "Intraday Patterns in the Cross-section of Stock Returns" | arXiv: https://arxiv.org/abs/1005.3535 | Shows recurring half-hour intraday patterns and related volume/order-imbalance/volatility/spread patterns. | Use for 30-minute time buckets and "same clock-time tomorrow" style periodicity caution. | Do not use its return-continuation results as evidence that this project's classifier has alpha. |
| Lo, "The Statistics of Sharpe Ratios" (Financial Analysts Journal, 2002) | DOI: https://doi.org/10.2469/faj.v58.n4.2453 ; metadata: https://ideas.repec.org/a/taf/ufajxx/v58y2002i4p36-52.html | Serial correlation can materially change performance inference. | Use as finance-facing language for why dependence reduces confidence in apparent performance. | Sharpe-ratio math is not the main metric here; do not report Sharpe or PnL. |
| scikit-learn `TimeSeriesSplit` docs | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html | Official time-ordered split object with a `gap` parameter. | Use as implementation inspiration for train-inner diagnostics only, while still enforcing ticker/day/window boundaries. | Default pooled `TimeSeriesSplit` is not sufficient for this project if it mixes tickers or label horizons. |
| scikit-learn `DummyClassifier` docs | https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html | Official support for stratified dummy predictions. | Keep same-row stratified dummy baselines for every retained subset and coverage point. | Beating dummy does not prove broad robustness or tradability. |
| SciPy `bootstrap` docs | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html | Practical reference for bootstrap intervals and paired resampling API. | Useful only for descriptive intervals if resampling units are changed to days/tickers/blocks. | Default iid row bootstrap is unsafe for overlapping intraday windows. |
| Pyper and Peterman, "Comparison of methods to account for autocorrelation in correlation analyses of fish data" (1998) | DOI: https://doi.org/10.1139/f98-104 ; metadata: https://cir.nii.ac.jp/crid/1363107370954264704 | Shows the general problem of adjusting effective degrees of freedom under autocorrelation. | Use as a general autocorrelation/effective-sample-size caution if ESS is reported. | Non-finance correlation analysis; do not import its exact adjustment mechanically. |

## Optional Sources

| Source | Link / DOI | Why it matters here | Use in Notebook 06/07 | Cannot support |
|---|---|---|---|---|
| Hill, "Diversity and Evenness: A Unifying Notation and Its Consequences" (1973) | DOI: https://doi.org/10.2307/1934352 ; metadata: https://cir.nii.ac.jp/crid/1362544420008325248 | Effective number of categories via entropy/diversity measures. | Use `exp(entropy)` and `1 / HHI` as readable effective ticker/day/bucket counts. | Ecology diversity thresholds are not finance thresholds. |
| xskillscore effective sample size docs | https://xskillscore.readthedocs.io/en/stable/api/xskillscore.effective_sample_size.html | Clear API-level explanation that autocorrelation lowers independent information. | Useful for explaining an approximate ESS diagnostic in plain language. | Do not add dependency or treat its correlation-specific ESS as exact for macro F1. |
| "Intraday Seasonalities and Nonstationarity of Trading Volume in Financial Markets" | arXiv: https://arxiv.org/abs/1810.12099 | Discusses U-shaped intraday volume/volatility profiles and changes over regimes. | Optional support for time-of-day and regime concentration discussion. | Do not use as a direct model template or as evidence for this project's labels. |
| MAPIE conformal classification docs | https://mapie.readthedocs.io/en/latest/api.html | Prior 06/07 materials already list MAPIE for conformal/risk-control ideas. | Optional future extension if conformal methods are separately pre-registered. | Exchangeability assumptions are weakened in chronological intraday data; do not claim conformal guarantees casually. |

## Risky Or Use-With-Caution Sources

| Source / Pattern | Link | Why risky | Safe use | Do not use |
|---|---|---|---|---|
| Trading blogs about open/close volatility or volume | Example search category only; avoid as primary evidence | Often strategy-marketing, not peer-reviewed, and may include profit claims. | At most as intuition for UI wording, not KB evidence. | Do not cite for academic support or thresholds. |
| `TunedThresholdClassifierCV` / generic threshold-tuning examples | https://scikit-learn.org/stable/modules/classification_threshold.html | Encourages supervised threshold optimization; easy to turn Notebook 06 into validation threshold scraping. | Cite only as a caution that thresholds need a separate design and time-aware split. | Do not tune arbitrary probability thresholds on official validation and present the best one. |
| Iid row bootstrap / random CV examples | SciPy/sklearn generic docs | Rows are overlapping, autocorrelated, and grouped by ticker/date. | Use only after replacing row units with day/ticker/block units and labeling intervals diagnostic. | Do not present iid row CIs as confirmatory. |
| Finance PnL/backtest papers | Existing 06/07 note has transaction-cost and backtest-overfitting sources | This project is classifier validation, not a trading execution notebook. | Use for caveats about not making profitability claims. | Do not add PnL, Sharpe, transaction-cost tuning, or trading thresholds in Notebook 06. |

## Recommended Guardrail Metrics

All metrics should be computed for every pre-registered coverage level, model,
seed, and pooled summary. They should be reported alongside `coverage_actual`,
selected-set macro F1, balanced accuracy, same-row dummy metrics, and
`delta_macro_f1_vs_dummy`. They are diagnostic downgrade flags, not selection
rules.

### Ticker Concentration

- `retained_share_by_ticker[t] = retained_n_t / retained_n`.
- `eligible_share_by_ticker[t] = eligible_validation_n_t / eligible_validation_n`.
- `ticker_lift[t] = retained_share_by_ticker[t] / eligible_share_by_ticker[t]`.
- `top_ticker_retained_share = max_t retained_share_by_ticker[t]`.
- `ticker_entropy = -sum_t retained_share_by_ticker[t] * log(retained_share_by_ticker[t])`.
- `ticker_entropy_norm = ticker_entropy / log(number_of_eligible_tickers)`.
- `ticker_hhi = sum_t retained_share_by_ticker[t]^2`.
- `ticker_effective_n_hhi = 1 / ticker_hhi`.
- `positive_ticker_count = count_t(delta_macro_f1_vs_dummy_selected_by_ticker[t] > 0)`.
- `top_ticker_gain_share`: diagnostic only. Recommended definition:
  `max_t positive_delta_t / sum_t positive_delta_t`, where
  `positive_delta_t = max(delta_macro_f1_vs_dummy_selected_by_ticker[t], 0)`.
  If the denominator is zero, set to `NaN` and state there is no positive gain
  to attribute.

Suggested interpretation:

- Fixed as diagnostic fields before reading results.
- Warning if `top_ticker_retained_share > 0.50`, `ticker_entropy_norm < 0.70`,
  or `positive_ticker_count < 3`.
- Severe downgrade if `top_ticker_retained_share > 0.65`,
  `ticker_entropy_norm < 0.50`, or `top_ticker_gain_share > 0.70`.
- These are wording guardrails only. Do not choose a lower/higher coverage level
  because it passes these flags after looking.

### Date Concentration

- `retained_share_by_date[d] = retained_n_d / retained_n`.
- `eligible_share_by_date[d] = eligible_validation_n_d / eligible_validation_n`.
- `date_lift[d] = retained_share_by_date[d] / eligible_share_by_date[d]`.
- `top_day_selected_share = max_d retained_share_by_date[d]`.
- `top_5_day_selected_share = sum of five largest retained_share_by_date`.
- `selected_day_count = count_d(retained_n_d > 0)`.
- `date_entropy = -sum_d retained_share_by_date[d] * log(retained_share_by_date[d])`.
- `date_entropy_norm = date_entropy / log(number_of_eligible_dates)`.
- `date_effective_n_entropy = exp(date_entropy)`.
- `date_hhi = sum_d retained_share_by_date[d]^2`.
- `date_effective_n_hhi = 1 / date_hhi`.
- `selected_day_span = max(selected_date) - min(selected_date)`.

Suggested interpretation:

- Use adaptive diagnostics because validation date count may vary.
- Warning if `top_day_selected_share > max(5 / eligible_day_count, 0.10)`,
  `top_5_day_selected_share > 0.40`, or `date_entropy_norm < 0.60`.
- Severe downgrade if `top_day_selected_share > max(10 / eligible_day_count, 0.20)`,
  `top_5_day_selected_share > 0.60`, or `date_effective_n_entropy < 0.25 * eligible_day_count`.
- If selected rows come mostly from a short date span, Notebook 07 should call
  the result localized validation behavior, not broad validation evidence.

### Time-Of-Day And Open/Close Concentration

Use both coarse market buckets and 30-minute clock buckets. For 5-minute bars,
predefine examples such as:

- `open_30`: first 30 minutes after regular-session open.
- `open_60`: first 60 minutes after regular-session open.
- `midday`: neither open nor close bucket.
- `close_30`: last 30 minutes before regular-session close.
- `close_60`: last 60 minutes before regular-session close.
- `clock_30min_bucket`: 13 half-hour buckets for a 6.5-hour regular session.

Metrics:

- `time_of_day_bucket_share[b] = retained_n_b / retained_n`.
- `eligible_time_bucket_share[b] = eligible_validation_n_b / eligible_validation_n`.
- `time_bucket_lift[b] = time_of_day_bucket_share[b] / eligible_time_bucket_share[b]`.
- `top_time_bucket_share = max_b time_of_day_bucket_share[b]`.
- `open_close_concentration_30 = share(open_30) + share(close_30)`.
- `open_close_concentration_60 = share(open_60) + share(close_60)`.
- `open_close_lift_30 = open_close_concentration_30 / eligible_open_close_share_30`.
- `time_bucket_entropy_norm = entropy(time_of_day_bucket_share) / log(number_of_nonempty_buckets)`.

Suggested interpretation:

- Prefer lift relative to eligible validation rows because open/close activity
  is naturally high.
- Warning if `top_time_bucket_share > 0.25`, any `time_bucket_lift > 2.0` with
  retained share above 0.15, or `open_close_lift_30 > 1.5`.
- Severe downgrade if `open_close_concentration_30 > 0.50` and
  `open_close_lift_30 > 2.0`, or if one 30-minute bucket explains most positive
  gain.
- Do not remove open/close rows after seeing that they drive results. Report the
  localization and downgrade the claim.

### Regime Concentration

Regime diagnostics should be pre-defined from allowed validation metadata or
current/past-bar features only. Do not use future returns, holdout/test rows, or
new labels.

Possible buckets:

- `realized_vol_regime`: low/medium/high based on current-and-past bar realized
  volatility or an already saved validation feature.
- `volume_regime`: low/medium/high based on current-and-past normalized volume.
- `spread_or_liquidity_regime`: only if already present in saved artifacts.
- `calendar_regime`: month, earnings-adjacent flag, or market-wide event flag
  only if pre-registered and available without new data snooping.

Metrics:

- `regime_bucket_share`, `regime_bucket_lift`, `top_regime_retained_share`,
  `regime_entropy_norm`, and `top_regime_gain_share`.

Suggested interpretation:

- Diagnostic only. Regime buckets are especially easy to turn into a post-hoc
  story, so Notebook 06 should list the bucket definitions before loading
  selective results.
- Warning if a single regime bucket has retained share above 0.50 or lift above
  2.0.
- Severe downgrade if positive delta exists only in one regime bucket.

### Overlapping-Window Dependence And Effective Sample Size

These metrics should be labeled approximate and descriptive:

- `retained_n`: nominal selected rows.
- `unique_ticker_day_count`: number of ticker-date groups with retained rows.
- `unique_date_count`: number of dates with retained rows.
- `unique_input_bar_count`: number of distinct input bars touched by retained
  windows, if sample artifacts include window start/end ids.
- `window_reuse_factor = retained_n * window_size / unique_input_bar_count`.
- `mean_adjacent_overlap = mean overlap ratio between adjacent retained windows
  within ticker-day`, if window ids are available.
- `acf_ess_by_ticker_day`: approximate effective sample size from selected-row
  correctness or loss autocorrelation within ticker-day:
  `n_eff = n / (1 + 2 * sum_k rho_k)`, using a small pre-specified lag cutoff.
- `ess_ratio = pooled_n_eff / retained_n`.
- `block_unit_n = unique_ticker_day_count`; this is often a more honest unit
  count than retained rows.

Suggested interpretation:

- Warning if `ess_ratio < 0.25`, `window_reuse_factor > window_size / 2`, or
  `unique_ticker_day_count` is too small for the intended claim.
- Severe downgrade if `ess_ratio < 0.10` or retained rows come from fewer than
  10 ticker-day blocks.
- Treat ESS as a caveat, not a corrected score. It can lower confidence in the
  claim but should not be used to select coverage.

## Reporting Template

Use this wording pattern in Notebook 06/07:

> At fixed coverage levels, selective/no-trade diagnostics are reported on
> validation-only rows with same-row dummy baselines. Concentration diagnostics
> show whether retained rows and apparent gains are spread across tickers,
> dates, and intraday buckets. When concentration warnings are triggered, the
> result is interpreted as localized validation behavior and is not treated as
> broad evidence or holdout-ready signal.

Allowed conclusions:

- "The high-confidence subset retained positive delta vs same-row dummy, but
  gains were concentrated in one ticker/date/time bucket; the result is
  validation-localized and should be downgraded."
- "Selective/no-trade behavior is broad enough for descriptive follow-up because
  positive deltas appear across at least three tickers and no dominant
  ticker/date/time concentration flag is triggered."
- "Coverage below 0.30 is visualization-only unless separately pre-registered."

Forbidden conclusions:

- "Coverage X is optimal" if X was chosen after inspecting concentration or
  selective metrics.
- "The no-trade threshold is final" or "holdout-ready."
- "The subset is tradable" without a separate execution-cost/PnL protocol.
- "Confidence intervals are confirmatory" when they used iid retained rows.

## Notebook 06/07 Implementation Checklist

- Predefine coverage grid before loading probability artifacts.
- Compute all concentration metrics at every coverage level, not only the best
  looking level.
- Always compare selected rows to same-row stratified dummy and always-up dummy.
- Report retained counts by ticker, date, time bucket, and ticker-day block.
- Compare retained shares against eligible validation shares, not against a
  naive uniform expectation alone.
- Include `top_ticker_retained_share`, `top_ticker_gain_share`,
  `top_day_selected_share`, `top_5_day_selected_share`,
  `time_of_day_bucket_share`, `open_close_concentration_30`,
  `date_entropy_norm`, `ticker_entropy_norm`, `ticker_effective_n_hhi`,
  `date_effective_n_entropy`, `window_reuse_factor`, and `ess_ratio`.
- Attach downgrade flags to wording, not to model/coverage selection.
- Keep holdout/test closed.

## Open Questions

- Does Notebook 05 save enough sample metadata for window start/end ids and
  unique input bar counts? If not, Notebook 06 can still compute ticker/date/time
  concentration from `ticker` and `timestamp`, but cannot compute
  `window_reuse_factor` exactly.
- Are market-session timestamps already normalized to regular-session New York
  time? If not, open/close bucket definitions must be frozen before using them.
- Should Notebook 07 report block/day bootstrap intervals? If yes, the resampling
  unit should be ticker-day or date block, not iid retained rows.

