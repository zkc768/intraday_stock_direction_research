# Paper Innovation Cards - 2026-05-25

## Ranking Summary

| Rank | Card | Current use |
|---|---|---|
| P0 | Leakage-safe weak-signal evaluation harness | Can be written now |
| P0 | Dummy-first model expansion gate | Can be written now |
| P0 | Diagnostic risk/coverage framing for stock direction labels | Can be written now |
| P0 | Ticker-local weak-signal heterogeneity | Can be written now |
| P0 | Seed sensitivity and robustness disclosure | Can be written now |
| P1 | Label/window/horizon/coverage stability map | Future-work/spec now |
| P1 | External-method adapter firewall | Future-work/spec now |
| P2 | Simple TSC baseline stress test | Do not implement now |
| P2 | Linear-family ablation for stock direction | Do not implement now |

## Card 1 - Leakage-Safe Weak-Signal Evaluation Harness

Implementation gate status: P0 - can be written now.

Source inspiration:
Financial ML evaluation discipline, classic time-series-classification baseline
discipline, and the project-owned protocol in `AGENTS.md`.

Project problem addressed:
High-frequency direction experiments are easy to overstate when labels,
splits, scaler fitting, dummy baselines, coverage, ticker rows, and seed rows
are not disclosed together.

Existing evidence:
The canonical full-binary pooled table has retained_pct
`0.8473149717144736`, n_test_windows `235333`, and best canonical delta
`-0.0023179624375161`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The canonical pooled table reports DLinear/LSTM/TCN deltas
`-0.0230916827204172`, `-0.0155783212992666`, and
`-0.0023179624375161`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).

Next artifact needed:
A paper Figure 1 protocol diagram showing labels, chronological splits,
train-only scaling, no-cross-day windows/horizons, dummy baselines, coverage
disclosure, ticker rows, seed rows, and external-method gate ownership.

Falsifiable experiment:
Run a pre-approved new setting through the same harness and show whether the
best pooled delta clears the gate while also surviving ticker and seed
disclosure.

Allowed claim if supported:
The harness changes interpretation by making weak or fragile model gains
visible under dummy, ticker, seed, label-semantics, and coverage reporting.

Blocked claim even if supported:
Do not claim a deployable trading rule from a harness result.

## Card 2 - Dummy-First Model Expansion Gate

Implementation gate status: P0 - can be written now.

Source inspiration:
Baseline-first evaluation discipline and the project decision rule documented
in `docs/PROJECT_DIRECTION_DECISION_2026-05-24.md`.

Project problem addressed:
Model expansion can become architecture-driven unless a simple, auditable
gate controls when new model families are allowed.

Existing evidence:
All three run rows have model_expansion_gate = `blocked_delta_lt_0.01`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The best canonical row is TCN at `-0.0023179624375161`, the best 0bps
diagnostic row is TCN at `+0.0048896919255157`, and the best 5bps diagnostic
row is LSTM at `+0.0018927356051448`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

Next artifact needed:
A short methods subsection defining the gate as a paper protocol decision, plus
a compact table that reproduces the gate string from the table source.

Falsifiable experiment:
An approved future baseline or adapter must beat `dummy_stratified` by the
documented threshold and must not rely on one ticker, one seed, or a diagnostic
label setting.

Allowed claim if supported:
Model expansion is governed by evidence against dummy baselines rather than by
architecture ambition.

Blocked claim even if supported:
Do not claim that a closed gate proves larger methods are generally useless.

## Card 3 - Diagnostic Risk/Coverage Framing For Stock Direction Labels

Implementation gate status: P0 - can be written now.

Source inspiration:
Selective prediction and abstention literature as a reporting lens, not as an
implemented confidence method.

Project problem addressed:
No-trade-band diagnostic settings can change retained coverage, class balance,
and apparent deltas, so the diagnostic results need coverage accounting before
interpretation.

Existing evidence:
The 0bps diagnostic best delta is `+0.0048896919255157` and the 0bps
diagnostic retained_pct is `0.8473091098964622`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 5bps diagnostic best delta is `+0.0018927356051448` and the 5bps
diagnostic retained_pct is `0.1491151811166493`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).
The 5bps diagnostic pooled n_test_windows is `19182`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_4_coverage_label_semantics.csv`).

Next artifact needed:
A threshold-retention proxy figure using
`checkpoints/phase1b_paper_tables_20260525/figure_threshold_retention_proxy.csv`.

Falsifiable experiment:
Add a prediction export with logits and probabilities, then test whether
probability-ranked retained subsets improve macro F1 against dummy at matched
coverage. This method is gated, not approved.

Allowed claim if supported:
Diagnostic retained-subset analysis can separate label/coverage effects from
model evidence.

Blocked claim even if supported:
Do not claim that threshold-retention proves calibrated uncertainty or a
deployable decision rule.

## Card 4 - Ticker-Local Weak-Signal Heterogeneity

Implementation gate status: P0 - can be written now.

Source inspiration:
Per-asset disclosure in financial ML and per-dataset/per-class reporting in
time-series classification.

Project problem addressed:
Pooled means can hide local positives and local negatives; local positives can
also distract from the canonical pooled caution.

Existing evidence:
The canonical table has n_positive_delta = `0` and n_non_positive_delta = `5`
for DLinear, LSTM, and TCN
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).
The 5bps diagnostic has descriptive local positives: DLinear has `1` positive
ticker, LSTM has `2`, and TCN has `2`, while the 5bps diagnostic pooled best
delta remains `+0.0018927356051448`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`;
`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

Next artifact needed:
A model-by-ticker heatmap and a paragraph that places every local positive next
to the canonical pooled result.

Falsifiable experiment:
Pre-register a ticker-local follow-up and test whether local positives remain
positive under canonical labels, held-out time, multiple seeds, and matched
dummy baselines.

Allowed claim if supported:
Apparent gains are ticker-local and need disclosure before broad interpretation.

Blocked claim even if supported:
Do not claim that a small set of local positives establishes broad
generalization.

## Card 5 - Seed Sensitivity And Robustness Disclosure

Implementation gate status: P0 - can be written now.

Source inspiration:
Multi-seed reporting norms and robustness disclosure in ML experiments.

Project problem addressed:
Weak-signal results can flip at the seed level even when model/ticker means are
non-positive.

Existing evidence:
The seed table uses n_seeds = `3` with seeds `42,43,44`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`).
Canonical examples with positive seed rows despite non-positive means include
DLinear/MSFT positive_seed_rate `0.6666666666666666`, LSTM/WMT
`0.3333333333333333`, and TCN/JPM, TCN/MSFT, TCN/WMT each
`0.3333333333333333`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_6_seed_ticker_stability.csv`).

Next artifact needed:
A compact seed-stability table or appendix table that reports mean, standard
deviation, min, max, positive_seed_rate, and suspicious-row count.

Falsifiable experiment:
Increase seed count under an approved budget and test whether positive seed
rates stabilize without changing labels, splits, scaler rules, or metrics.

Allowed claim if supported:
Seed disclosure changes the interpretation of weak model effects.

Blocked claim even if supported:
Do not claim statistical certainty from the current 3-seed evidence.

## Card 6 - Label/Window/Horizon/Coverage Stability Map

Implementation gate status: P1 - future-work/spec now.

Source inspiration:
Project threshold-sensitivity manifest work and weak-signal reporting
discipline.

Project problem addressed:
The current results depend on label semantics, window length, horizon, retained
coverage, ticker, and seed; these axes need a manifest-first map before any
training grid.

Existing evidence:
The current paper tables already show canonical retained_pct
`0.8473149717144736`, 0bps diagnostic retained_pct
`0.8473091098964622`, and 5bps diagnostic retained_pct
`0.1491151811166493`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_1_run_gate_summary.csv`).

Next artifact needed:
A manifest-only stability map that reports coverage, class balance, and test
window counts for proposed label/window/horizon settings before any model run.

Falsifiable experiment:
For each approved manifest setting, train only if coverage and class balance
pass the pre-specified feasibility checks. This method is gated, not approved.

Allowed claim if supported:
The project can distinguish stable evidence from protocol-induced artifacts
across label/window/horizon/coverage settings.

Blocked claim even if supported:
Do not claim that the best retained setting becomes the canonical task.

## Card 7 - External-Method Adapter Firewall

Implementation gate status: P1 - future-work/spec now.

Source inspiration:
`docs/EXTERNAL_METHOD_REVIEW_BACKLOG_2026-05-25.md` and the adapter contract in
`docs/PROJECT_DIRECTION_DECISION_2026-05-24.md`.

Project problem addressed:
External papers and repositories can be useful sources of hypotheses, but they
must not take ownership of labels, splits, scalers, metrics, data loading, or
claims.

Existing evidence:
The external-method backlog defines an 18-item gate and keeps external methods
as source-review or future-spec candidates until the gate passes
(`docs/EXTERNAL_METHOD_REVIEW_BACKLOG_2026-05-25.md`).

Next artifact needed:
An adapter dry-run spec that proves an external model can accept project-owned
NLC tensors and return raw logits without touching data construction.

Falsifiable experiment:
Run a tiny adapter smoke after approval and verify that the adapter cannot
change labels, splits, scaler fitting, or metrics. This method is gated, not
approved.

Allowed claim if supported:
The adapter firewall is a reproducibility and contamination-control mechanism.

Blocked claim even if supported:
Do not claim that the firewall itself improves predictive scores.

## Card 8 - Simple TSC Baseline Stress Test

Implementation gate status: P2 - do not implement now.

Source inspiration:
ROCKET/MiniROCKET-style time-series classification baselines and classic TSC
benchmark discipline.

Project problem addressed:
If simple strong TSC baselines explain apparent sequence-model gains, larger
neural model work should remain gated.

Existing evidence:
No approved ROCKET/MiniROCKET result exists in the current paper tables. The
current evidence only covers DLinear, LSTM, and TCN rows
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`).

Next artifact needed:
A dependency/license review, a train-only fit/transform spec, and tests proving
the baseline remains inside the project harness.

Falsifiable experiment:
Run the simple TSC baseline under the exact same canonical label, split, scaler,
dummy baseline, ticker, seed, and coverage reporting rules. This method is
gated, not approved.

Allowed claim if supported:
A simple TSC baseline can be used to stress-test whether sequence-model gains
survive stronger non-neural comparison.

Blocked claim even if supported:
Do not claim that external benchmark strength transfers automatically to this
stock-direction task.

## Card 9 - Linear-Family Ablation For Stock Direction

Implementation gate status: P2 - do not implement now.

Source inspiration:
DLinear/LTSF-Linear simplicity results and stock-specific linear-family ideas
recorded in the external-method backlog.

Project problem addressed:
The current DLinear result is negative in the canonical pooled table, but it
does not isolate which linear design choices matter for direction
classification.

Existing evidence:
The canonical DLinear pooled delta is `-0.0230916827204172`
(`checkpoints/phase1b_paper_tables_20260525/paper_table_2_pooled_model_vs_dummy.csv`),
and canonical DLinear has n_positive_delta = `0` across `5` tickers
(`checkpoints/phase1b_paper_tables_20260525/paper_table_5_ticker_delta_counts.csv`).

Next artifact needed:
A first-principles local ablation spec with exact target file, tests, line
budget, unchanged labels, unchanged splits, unchanged scaler rules, and
unchanged metrics.

Falsifiable experiment:
Compare linear-family ablations against the current DLinear row and
dummy_stratified under the same canonical harness. This method is gated, not
approved.

Allowed claim if supported:
Linear design choices can be evaluated under the same project-owned protocol.

Blocked claim even if supported:
Do not claim that forecasting-task improvements transfer directly to binary
stock direction classification.
