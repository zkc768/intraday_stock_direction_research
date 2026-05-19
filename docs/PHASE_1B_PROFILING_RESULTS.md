# Phase 1B Profiling Results — P1B.7 / P1B.7d

Status: recorded after real-data label-level profiling (P1B.7) and window-aware profiling (P1B.7d).

## 1. Run Environment

| Field | Value |
|---|---|
| Runtime | Colab runtime |
| Repo branch | `master` |
| Repo commit | `76bdf47` |
| Data path | `/content/drive/MyDrive/stockdata/Dow_30_1min` |
| Output path | `/content/drive/MyDrive/stockdata/phase1b_capacity_profile_outputs/` |

## 2. Data Overview

Tickers profiled: `CSCO`, `JPM`, `KO`, `MSFT`, `WMT`.

| Ticker | Rows | P5 bars/day | Median bars/day |
|---|---:|---:|---:|
| CSCO | 444305 | 79 | 79 |
| JPM | 443589 | 78 | 79 |
| KO | 443273 | 78 | 79 |
| MSFT | 444322 | 79 | 79 |
| WMT | 443278 | 78 | 79 |

## 3. Split Ratios

| Split | Ratio |
|---|---:|
| Train | 0.70 |
| Validation | 0.15 |
| Test | 0.15 |

These ratios use the current `DataConfig` defaults. They do not use the old TensorFlow notebook's 80/10/10 split.

## 4. Infeasible Combinations

The following combinations were marked `INFEASIBLE_INTRADAY_CAPACITY` in full, train, validation, and test splits:

| Window size | k |
|---:|---:|
| 60 | 24 |
| 78 | 12 |
| 78 | 24 |

Do not use these combinations in the first Phase 1B baseline rerun.

## 5. Recommended First Rerun Candidates

### Main Candidate

`window=12`, `k=12`, `threshold_bps=5`

| Split | n_valid_windows | minority_pct |
|---|---:|---:|
| Train | 213116 | 0.4921 |
| Validation | 11903 | 0.4990 |
| Test | 19129 | 0.4531 |

### Secondary Candidate

`window=24`, `k=12`, `threshold_bps=5`

| Split | n_valid_windows | minority_pct |
|---|---:|---:|
| Train | 161397 | 0.4927 |
| Validation | 8118 | 0.4963 |
| Test | 13395 | 0.4465 |

### Longer-Horizon Candidate

`window=12`, `k=24`, `threshold_bps=5`

| Split | n_valid_windows | minority_pct |
|---|---:|---:|
| Train | 88783 | 0.4880 |
| Validation | 2960 | 0.4811 |
| Test | 6644 | 0.3975 |

### Long-Context Stress Test

`window=60`, `k=12`, `threshold_bps=5`

| Split | n_valid_windows | minority_pct |
|---|---:|---:|
| Train | 36780 | 0.4965 |
| Validation | 1695 | 0.4956 |
| Test | 2940 | 0.4650 |

## 6. Sensitivity Candidates

`window=12`, `k=12`, `threshold_bps=10` may be used as a sensitivity-only candidate.

| Split | n_valid_windows |
|---|---:|
| Train | 56900 |
| Validation | 1018 |
| Test | 4269 |

Warning: the validation sample is near the lower bound, so this candidate is not recommended as the main rerun setting.

## 7. Rejected Or Deferred Options

- `threshold_bps >= 15` is generally too sparse for the first rerun.
- `threshold_bps=0` is only a sanity baseline, not a real no-trade band.
- Volatility-scaled thresholding is deferred.
- Three-class flat/up/down labeling is deferred.
- TCN and DLinear implementation is deferred until this profiling decision is reviewed.

## 8. Interpretation

This profiling supports a conditional classification task:

```text
P(sign(r) | X, |r| > tau)
```

It does not estimate full-market direction:

```text
P(sign(r) | X)
```

Because the no-trade-band label drops neutral samples, every model report using these settings must disclose retained/window coverage alongside model metrics. The selection-bias and coverage tradeoff must be disclosed in reports.

## 9. Saved CSV Files

The Colab profiling run saved the following CSV files under `/content/drive/MyDrive/stockdata/phase1b_capacity_profile_outputs/`:

- `bar_count_summary.csv`
- `capacity_by_split.csv`
- `capacity_with_feasibility.csv`
- `per_ticker_capacity_with_verdicts.csv`
- `pooled_capacity_summary_with_verdicts.csv`
- `pooled_window_capacity_summary.csv`
- `scaler_diagnostic_train_ohlcv_only.csv`
- `window_capacity_by_ticker.csv`

## 10. Next Action

- P1B.8b ChatGPT review of this document.
- Then docs-only atomic commit.
- Then prepare the baseline rerun plan using the main and secondary candidates.
- Do not start TCN or DLinear yet.
