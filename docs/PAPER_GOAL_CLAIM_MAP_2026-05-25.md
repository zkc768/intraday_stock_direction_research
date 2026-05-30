# Paper Goal Claim Map - 2026-05-25

## 1. Thesis Statement

The current paper should be framed as a protocol-safe weak-signal evaluation
study for intraday stock direction classification.

The central claim is:

```text
hf_stock_clf provides a leakage-safe evaluation harness and shows that, in the
current canonical full-binary setup, LSTM, TCN, and DLinear do not robustly
outperform a stratified dummy classifier once ticker, seed, label semantics, and
coverage are disclosed.
```

This is a protocol and evidence-control claim. It is not a trading deployment
claim, not a broad market claim, and not a claim about untested external model
families.

## 2. Evidence Table

| Evidence row | Source path | Verified numeric anchor | Allowed use | Claim-control note |
|---|---|---|---|---|
| Canonical pooled result | `checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`; `checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv` | Canonical retained_pct = `0.8473149717144736`, n_test_windows = `235333`, best_model = `tcn`, best_delta_macro_f1_vs_dummy = `-0.0023179624375161`; DLinear/LSTM/TCN deltas = `-0.0230916827204172`, `-0.0155783212992666`, `-0.0023179624375161`. | Main result: none of the three sequence baselines beats `dummy_stratified` on the canonical pooled comparison. | The claim is limited to the current local setup and current baselines. |
| Canonical per-ticker result | `checkpoints/phase1b_paper_tables_20260525/paper_table_3_canonical_ticker_delta.csv`; `checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv` | Canonical n_positive_delta = `0` and n_non_positive_delta = `5` for each model; canonical mean per-ticker deltas are DLinear `-0.03225153890759262`, LSTM `-0.04365741281236411`, TCN `-0.02229340815331688`. | Supports ticker-level disclosure and blocks a broad positive model claim. | Per-ticker rows are mean deltas over seeds, not independent market evidence. |
| Ticker delta count result | `checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv` | Canonical all_non_positive_delta = `True` for DLinear, LSTM, and TCN; canonical max per-ticker deltas are DLinear `-0.0022577097643043`, LSTM `-0.0344091585986996`, TCN `-0.0075083894202687`. | Supports the statement that canonical per-ticker positives are absent at the model-mean level. | Do not convert diagnostic local positives into canonical positives. |
| Seed ticker stability result | `checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv` | n_seeds = `3` throughout; canonical examples with positive seed rows despite negative mean include DLinear/MSFT positive_seed_rate `0.6666666666666666`, LSTM/WMT `0.3333333333333333`, and TCN/JPM, TCN/MSFT, TCN/WMT each `0.3333333333333333`. | Supports seed-sensitivity disclosure. | Three seeds provide descriptive sensitivity evidence, not confirmatory inference. |
| Regime diagnostic comparison result | `checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`; `checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv` | The 0bps diagnostic best delta is TCN `+0.0048896919255157`; the 5bps diagnostic best delta is LSTM `+0.0018927356051448`; both diagnostic rows have model_expansion_gate = `blocked_delta_lt_0.01`. | Supports descriptive comparison across diagnostic label settings. | The diagnostic settings do not replace the canonical full-binary task. |
| Coverage fragility result | `checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`; `checkpoints/phase1b_paper_tables_20260525/paper_table_8_coverage_fragility_flags.csv` | The 5bps diagnostic pooled retained_pct is `0.1491151811166493`, pooled n_test_windows is `19182`, and all 5bps diagnostic ticker rows have low_coverage_flag = `True`; KO has retained_pct `0.0933907397205841` and n_test_windows `2239`. | Supports coverage disclosure and low-coverage caution. | Coverage changes are part of the result, not an implementation detail to hide. |
| Threshold-retention proxy result | `checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv` | proxy_kind = `threshold_retention_not_confidence`; best deltas are canonical `-0.0023179624375161`, 0bps diagnostic `+0.0048896919255157`, and 5bps diagnostic `+0.0018927356051448`; all gates are `blocked_delta_lt_0.01`. | Supports a threshold-retention proxy figure only. | No per-sample probability export exists, so this is not probability-ranked selective classification. |
| Model-expansion gate result | `checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv` | All three rows have n_suspicious_rows = `0` and model_expansion_gate = `blocked_delta_lt_0.01`. | Supports keeping future model work gated. | A closed gate means "not justified by current evidence," not "larger methods are impossible." |

## 3. Allowed Claims

- Under the current canonical full-binary run, LSTM, TCN, and DLinear do not
  beat the stratified dummy baseline on pooled macro F1; the best canonical
  delta is TCN at `-0.0023179624375161`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- The canonical per-ticker table contains no positive model/ticker mean delta:
  each canonical model has n_positive_delta = `0` and n_non_positive_delta =
  `5`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).
- The 0bps diagnostic result is descriptive: TCN has a diagnostic pooled delta
  of `+0.0048896919255157`, but the diagnostic gate remains
  `blocked_delta_lt_0.01`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- The 5bps diagnostic result is descriptive: LSTM has a diagnostic pooled delta
  of `+0.0018927356051448`, but the diagnostic retained_pct is only
  `0.1491151811166493`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- The threshold-retention plot is a proxy over retained subsets, with
  proxy_kind = `threshold_retention_not_confidence`
  (`checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv`).
- Current evidence supports protocol analysis, robust reporting, and gated
  future-method design before any larger model implementation.

## 4. Blocked Claims

- Do not claim that the current project discovered a deployable trading rule.
- Do not claim broad stock-market predictability or broad stock-market
  non-predictability.
- Do not claim that untested external model families failed.
- Do not claim that complex models are generally useless.
- Do not claim that CSCO/KO diagnostic positives generalize beyond their
  post-hoc, descriptive context.
- Do not claim that the 0bps diagnostic or 5bps diagnostic task replaces the
  canonical full-binary task.
- Do not claim that retained-threshold behavior is probability calibration or
  probability-ranked selective classification.
- Do not claim that the adapter firewall improves model scores; its current
  value is protocol ownership and contamination control.

## 5. Diagnostic-Only Claims

- The 0bps diagnostic is descriptive because it uses
  `phase1b_no_trade_band_diagnostic` label semantics and `neutral_nan`
  zero-return policy
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- The 0bps diagnostic is descriptive because its best pooled delta
  `+0.0048896919255157` stays below the documented gate string
  `blocked_delta_lt_0.01`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- The 5bps diagnostic is descriptive because it retains only
  `0.1491151811166493` of pooled rows and evaluates `19182` pooled test windows
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`).
- The 5bps diagnostic is descriptive because local positives occur next to
  low-coverage flags, including KO retained_pct `0.0933907397205841` and
  n_test_windows `2239`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_8_coverage_fragility_flags.csv`).
- The diagnostic threshold-retention proxy is descriptive because it is labeled
  `threshold_retention_not_confidence`
  (`checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv`).

## 6. Required Caveats

- The current canonical evidence covers 5 tickers, visible as CSCO, JPM, KO,
  MSFT, WMT rows in
  `checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`.
- The seed evidence covers 3 seeds, listed as `42,43,44` in
  `checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`.
- The model evidence covers 3 baseline model families, listed as DLinear, LSTM,
  and TCN in
  `checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`.
- The canonical zero-return policy keeps `13` pooled zero-return rows as class
  0 non-up
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`).
- The 0bps diagnostic exact-zero rows are handled under `neutral_nan`, not
  canonical `class_0_non_up`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`).
- The 5bps diagnostic has a much lower retained_pct than the canonical task:
  `0.1491151811166493` versus `0.8473149717144736`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
- Balanced accuracy is secondary to macro F1 for the gate; for example,
  canonical TCN balanced_accuracy_mean is `0.5093891440878746` while its
  delta_macro_f1_vs_dummy_mean is `-0.0023179624375161`
  (`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).

## 7. Missing Evidence

- No per-sample logits or probabilities are exported yet, so probability-ranked
  selective classification remains missing evidence.
- No approved simple TSC baseline result exists yet; ROCKET/MiniROCKET-style
  work remains gated and not approved.
- No approved linear-family ablation result exists beyond the current DLinear
  baseline rows in
  `checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`.
- No external adapter dry-run has been approved; external-method integration
  remains a gated specification task.
- No larger ticker universe has been evaluated in the current paper tables; the
  current evidence is limited to the 5 ticker rows in
  `checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`.
