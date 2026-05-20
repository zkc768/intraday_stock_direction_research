# Phase 1B LSTM Rerun Results

## 1. Status

PASS WITH WARNINGS

P1B.10 full A-D LSTM no-trade-band rerun after label-alignment fix completed. No suspicious 0.95+ F1 appeared, and `suspicious_status=False` for all candidates.

These results are not a final research conclusion. They are a single-seed rerun only and should be used to select the next verification step, not to claim robust predictive performance.

## 2. Scope

- LSTM only.
- No-trade-band binary labels.
- Seed `42` only.
- No TCN.
- No DLinear.
- No Notebook 03.
- No seed aggregation yet.

## 3. Git and artifact basis

Run label:

```text
P1B.10 full A-D LSTM no-trade-band rerun after label-alignment fix
```

Git basis:

- `e2e2869` fix(dataset): align window labels to prediction point
- `fc7b863` docs(phase1b): record label-alignment fixed smoke

Output directory:

```text
/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed_full_ad/
```

Saved artifacts:

- `results_A.csv`
- `results_B.csv`
- `results_C.csv`
- `results_D.csv`
- `results_all_candidates.csv`
- `summary_per_ticker.csv`
- `summary_by_candidate.csv`
- `run_manifest.json`

## 4. Task framing disclosure

No-trade-band binary classification estimates P(sign(r) | X, |r| > tau), not P(sign(r) | X). Retained-subset metrics must be interpreted together with coverage/window retention.

These metrics apply only to retained non-neutral samples. They should not be interpreted as full-market directional prediction performance. Coverage must be reported together with F1 because a higher F1 on a small retained subset can still leave most market intervals outside the evaluated task.

## 5. Candidate grid

| candidate_id | candidate_name | window_size | label_horizon_k | threshold_bps | seed |
|---|---|---:|---:|---:|---:|
| A | main | 12 | 12 | 5 | 42 |
| B | secondary_window24 | 24 | 12 | 5 | 42 |
| C | longer_horizon24 | 12 | 24 | 5 | 42 |
| D | long_context60 | 60 | 12 | 5 | 42 |

## 6. Summary by candidate

| candidate_id | candidate_name | model_macro_f1_mean | model_macro_f1_std | model_balanced_accuracy_mean | dummy_stratified_macro_f1_mean | delta_macro_f1_vs_dummy_mean | total_train_windows | total_val_windows | total_test_windows | suspicious_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | main | 0.5203 | 0.0154 | 0.5227 | 0.4990 | +0.0213 | 213384 | 11903 | 19190 | False |
| B | secondary_window24 | 0.4548 | 0.0842 | 0.5229 | 0.4981 | -0.0434 | 161897 | 8118 | 13507 | False |
| C | longer_horizon24 | 0.4468 | 0.0464 | 0.5083 | 0.4969 | -0.0502 | 88895 | 2960 | 6688 | False |
| D | long_context60 | 0.4228 | 0.0802 | 0.5163 | 0.5008 | -0.0779 | 37003 | 1695 | 2981 | False |

Training summary:

| candidate_id | best_epoch | best_val_macro_f1 | training_time_seconds |
|---|---:|---:|---:|
| A | 18 | 0.5026 | 122.9 |
| B | 3 | 0.5023 | 44.7 |
| C | 11 | 0.4714 | 42.2 |
| D | 18 | 0.4670 | 27.9 |

## 7. Candidate A per-ticker breakdown

| ticker | model_macro_f1 | model_balanced_accuracy | dummy_stratified_macro_f1_mean | delta_macro_f1_vs_dummy | n_train_windows | n_val_windows | n_test_windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| CSCO | 0.5256 | 0.5284 | 0.5002 | +0.0254 | 58096 | 2918 | 4697 |
| JPM | 0.5244 | 0.5250 | 0.5023 | +0.0221 | 53102 | 2921 | 4561 |
| KO | 0.5409 | 0.5408 | 0.4986 | +0.0422 | 26602 | 967 | 2239 |
| MSFT | 0.5010 | 0.5078 | 0.4991 | +0.0020 | 42142 | 3713 | 4856 |
| WMT | 0.5097 | 0.5114 | 0.4949 | +0.0148 | 33442 | 1384 | 2837 |

## 8. Coverage / retained sample warning

| ticker | retained_pct |
|---|---:|
| CSCO | 19.6979% |
| JPM | 18.2968% |
| KO | 9.3378% |
| MSFT | 15.3936% |
| WMT | 11.8155% |

Candidate A retains only about 9% to 20% depending on ticker. KO and WMT have especially low retained coverage. This limits interpretation because the model is evaluated only on a narrow non-neutral subset, not on all 5-minute market bars.

## 9. Interpretation

Candidate A is the only candidate with positive mean delta vs dummy. Candidate A is also the only candidate where all five tickers have positive delta. The average gain is small: +0.0213 macro F1. This is encouraging but weak. It should be treated as a signal candidate, not as proof of robust prediction.

For the other candidates:

- Candidate B has positive results for CSCO and KO, but large negative deltas for MSFT, WMT, and JPM.
- Candidate C has mostly weak or negative deltas.
- Candidate D is dominated by CSCO and weak or negative elsewhere.

B/C/D are not selected for immediate continuation.

## 10. Decision

Selected for next verification:

- Candidate A only.

Rejected / defer for now:

- Candidate B.
- Candidate C.
- Candidate D.

Next step:

```text
P1B.11 Candidate A multi-seed verification
```

Recommended seeds:

- 42
- 43
- 44

Do not enter:

- TCN.
- DLinear.
- Notebook 03.
- Improved model.

## 11. Known limitations

- Single seed only.
- Retained subset only.
- No transaction cost / trading strategy evaluation.
- No hyperparameter sweep.
- No statistical significance test.
- Possible sensitivity to training stochasticity.
- Low coverage especially for KO and WMT.

## 12. Follow-up checklist

- [ ] Review saved Colab artifacts.
- [ ] Verify `run_manifest.json` contains git hash and config.
- [ ] Run Candidate A multi-seed verification.
- [ ] Compare mean/std across seeds.
- [ ] Only after multi-seed review decide whether to enter TCN/DLinear baselines.
