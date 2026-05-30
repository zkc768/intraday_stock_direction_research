# Paper Results Narrative Draft - 2026-05-25

## 1. Main Result: Canonical Sequence Models Do Not Beat Dummy

The canonical full-binary run is the primary result because it uses
`canonical_phase1_full_binary` label semantics and `class_0_non_up`
zero-return policy. In this run, the pooled retained_pct is
`0.8473149717144736`, pooled n_test_windows is `235333`, and label_n_zero_return
is `13`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

Against `dummy_stratified`, the canonical pooled deltas are DLinear
`-0.0230916827204172`, LSTM `-0.0155783212992666`, and TCN
`-0.0023179624375161`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).
TCN is closest to dummy, with macro_f1_mean `0.4975395135407587`,
dummy_stratified_macro_f1_mean `0.4998574759782748`, and
balanced_accuracy_mean `0.5093891440878746`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).

The result supports a narrow conclusion: under the current canonical harness,
the three implemented sequence baselines do not robustly outperform the
stratified dummy baseline on macro F1. The result does not support expanding
the model family before stronger protocol evidence exists.

## 2. Per-Ticker Heterogeneity: Local Positives/Negatives Are Not Broad Evidence

The canonical per-ticker table strengthens the pooled caution. Each canonical
model has n_positive_delta = `0` and n_non_positive_delta = `5`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).

Canonical mean per-ticker deltas are negative for every model: DLinear
`-0.03225153890759262`, LSTM `-0.04365741281236411`, and TCN
`-0.02229340815331688`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).
The least-negative canonical per-ticker mean values are DLinear/MSFT
`-0.0022577097643043`, LSTM/CSCO `-0.0344091585986996`, and TCN/MSFT
`-0.0075083894202687`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).

The 5bps diagnostic local positives are descriptive and must be paired with the
canonical pooled caution: the 5bps diagnostic has LSTM/CSCO
`+0.0251769280904116`, LSTM/KO `+0.0381944596913546`, TCN/CSCO
`+0.0281990237201111`, and TCN/KO `+0.0411269703853817`, while the canonical
best pooled delta remains TCN `-0.0023179624375161`
(`checkpoints/phase1b_paper_tables_20260525/figure_ticker_delta_heatmap.csv`;
`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 5bps diagnostic local positives are descriptive because the 5bps diagnostic
pooled retained_pct is only `0.1491151811166493`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

The paper should therefore present ticker-local variation as a reason for
disclosure and follow-up design, not as broad evidence.

## 3. Seed Stability: 3-Seed Dispersion And Sensitivity

The seed table reports n_seeds = `3` with seeds `42,43,44`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`).
The canonical model/ticker means are negative, but several canonical
model/ticker rows still contain positive seed-level deltas: DLinear/MSFT has
positive_seed_rate `0.6666666666666666`, LSTM/WMT has
`0.3333333333333333`, and TCN/JPM, TCN/MSFT, and TCN/WMT each have
`0.3333333333333333`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`).

This pattern should be written as sensitivity evidence. A positive seed row is
not enough to override a negative model/ticker mean, and a 3-seed table is not
enough for confirmatory inference.

## 4. Diagnostic Regimes: 0bps/5bps Are Descriptive Only

The 0bps diagnostic is descriptive because it uses
`phase1b_no_trade_band_diagnostic` semantics and `neutral_nan` policy
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 0bps diagnostic best pooled model is TCN with diagnostic delta
`+0.0048896919255157`, diagnostic retained_pct `0.8473091098964622`, and
diagnostic n_test_windows `235333`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 0bps diagnostic remains descriptive because its diagnostic gate is
`blocked_delta_lt_0.01`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 0bps diagnostic local evidence is descriptive because TCN has
n_positive_delta = `0` and n_non_positive_delta = `5` in the diagnostic
per-ticker count table
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).

The 5bps diagnostic is descriptive because it changes retained coverage and
label semantics relative to the canonical task
(`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`).
The 5bps diagnostic best pooled model is LSTM with diagnostic delta
`+0.0018927356051448`, diagnostic retained_pct `0.1491151811166493`, and
diagnostic n_test_windows `19182`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 5bps diagnostic remains descriptive because its diagnostic gate is
`blocked_delta_lt_0.01`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

The diagnostic story is therefore not "choose the best diagnostic setting."
The diagnostic story is descriptive: label semantics and retained coverage can
change apparent model deltas, so diagnostic findings must stay separate from
the canonical claim.

## 5. Coverage Fragility: Retained Pct And Test Windows Change Interpretation

Coverage is part of the result. The canonical pooled retained_pct is
`0.8473149717144736` with `235333` pooled test windows, while the 5bps
diagnostic pooled retained_pct is `0.1491151811166493` with `19182` pooled test
windows
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

The 5bps diagnostic coverage is descriptive and fragile at the ticker level:
CSCO has retained_pct `0.1969880521866795` and n_test_windows `4692`, JPM has
`0.1829626890240877` and `4560`, KO has `0.0933907397205841` and `2239`, MSFT
has `0.1539628826613892` and `4854`, and WMT has `0.1181022751123338` and
`2837`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_8_coverage_fragility_flags.csv`).
The 5bps diagnostic is descriptive because all 5bps diagnostic ticker rows have
low_coverage_flag = `True`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_8_coverage_fragility_flags.csv`).

The paper should show retained_pct and n_test_windows near any model score,
especially for descriptive diagnostic rows.

## 6. Threshold-Retention Proxy: Not Confidence-Based Selective Classification

The threshold-retention proxy is sourced from
`checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv`
and is explicitly labeled proxy_kind = `threshold_retention_not_confidence`.
The proxy rows show canonical best delta `-0.0023179624375161`, 0bps diagnostic
best delta `+0.0048896919255157`, and 5bps diagnostic best delta
`+0.0018927356051448`
(`checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv`).

The threshold-retention proxy should be described as retained-subset analysis.
It should not be described as probability-ranked selective classification,
because no per-sample logits or probabilities are exported in the current
paper tables.

## 7. Model Expansion Gate: Why It Remains Blocked

The current model-expansion gate remains blocked because all rows in the run
gate table have model_expansion_gate = `blocked_delta_lt_0.01`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The best canonical delta is `-0.0023179624375161`, the best 0bps diagnostic
delta is `+0.0048896919255157`, and the best 5bps diagnostic delta is
`+0.0018927356051448`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

Future methods are gated, not approved. A future method would need a separate
spec, unchanged project-owned labels and splits, train-only fitting, dummy
baselines, ticker disclosure, seed disclosure, coverage disclosure, and a
positive result that clears the documented gate without relying on a diagnostic
task.

## 8. Limitations And Next Approved Evidence

The current evidence covers 5 tickers, visible as CSCO, JPM, KO, MSFT, and WMT
rows in
`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`.
The current evidence covers 3 seeds, listed as `42,43,44` in
`checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`.
The current evidence covers 3 implemented model families, DLinear, LSTM, and
TCN
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).

The next approved evidence should be protocol-analysis evidence, not an
unreviewed model expansion. Suitable next artifacts are a manifest-first
label/window/horizon/coverage stability map, a prediction-export spec for
probability-ranked future analysis, and a gated simple-baseline spec. Each
future method is gated, not approved, until the project records target files,
tests, dependency status, unchanged protocol ownership, and an explicit smoke
plan.

The limitations do not invalidate the current paper story. They define the
story: strict protocol, dummy baselines, ticker disclosure, seed disclosure, and
coverage accounting are necessary before interpreting weak model deltas.
