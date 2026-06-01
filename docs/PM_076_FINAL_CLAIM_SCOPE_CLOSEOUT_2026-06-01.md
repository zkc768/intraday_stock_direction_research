# PM-076 Final Claim-Scope Closeout

Date: 2026-06-01

Status: PASS / final result accepted as weak-mixed bounded evidence for one
frozen route

## Inputs

PM-076 uses the PM-074 final holdout-test artifact reviewed in PM-075:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_074_final_holdout_ms_dlinear_tcn_20260601\phase1b_local_no_trade_band_full_20260531_201555
```

PM-076 does not run training, rerun the script, execute notebooks, tune models,
select routes, or modify code.

## Final Result Boundary

Allowed result statement:

> One frozen MS-DLinear+TCN route using `mentor_clean_v1`, `no_trade_band`,
> fixed 5 bps threshold, train-only pooled scaling, seed 42, one epoch, five
> tickers, and the predeclared calendar holdout interval was evaluated once on
> the final held-out/test split. The pooled result is weakly positive versus the
> dummy baseline, while ticker-level results are mixed.

Exact pooled metrics:

- macro F1: `0.5090765910439399`
- balanced accuracy: `0.5199419698841798`
- precision macro: `0.5233149809566453`
- recall macro: `0.5199419698841798`
- dummy-stratified macro F1 mean: `0.5031400963934844`
- delta macro F1 vs dummy: `0.005936494650455426`
- test windows: `7388`

Ticker-level macro-F1 deltas versus dummy:

- CSCO: `0.05249391250349544`
- JPM: `-0.005339713730280471`
- KO: `-0.0020172258275721333`
- MSFT: `0.006805483062818807`
- WMT: `-0.01331217888324332`

## Allowed Claims

The project may claim:

- PM-074 opened the final holdout-test gate exactly once for the frozen route.
- The artifact contains six final `split=test` rows: pooled plus the five
  predeclared tickers.
- The pooled held-out macro F1 is slightly above the dummy-stratified baseline.
- Ticker-level results are mixed, with CSCO positive, MSFT slightly positive,
  and JPM, KO, and WMT negative versus dummy on macro-F1 delta.
- The artifact confirms the frozen route's final held-out/test behavior under
  the exact PM-073 route locks.

## Forbidden Claims

The project must not claim:

- model-family superiority;
- paper-ready performance;
- trading profitability;
- deployability;
- robustness across seeds, thresholds, tickers beyond this set, time periods,
  feature sets, labels, or model families;
- that MS-DLinear+TCN generally outperforms baselines;
- that the validation metrics justify, rescue, or reinterpret the held-out
  result;
- that the route should be retuned, rerun, or reselected after seeing the final
  holdout-test metrics;
- that this result promotes anything into `evidence_matrix.csv` or claim maps;
- that historical project work never produced any test metrics.

## Caveats

- The final result is weak/mixed, not strong.
- `best_val_macro_f1=0.4803618327815362` and
  `val_delta_macro_f1_vs_dummy=-0.019245201434752113` are frozen training
  metadata only. They are not a selection basis after holdout opening.
- The runtime metadata records known untracked files at execution time:
  `.codegraph/` and three notebooks. These were not staged or modified by
  PM-074/075/076.
- Checkpoint artifacts are preserved but are not staged for git.
- PM-076 does not promote this artifact into the literature evidence matrix or
  any claim map.

## No-Retune and No-Selection Closeout

PM-076 closes the route as evaluated.

No additional PM-074 rerun, seed search, threshold adjustment, route fallback,
feature-set change, label-policy change, scaler-policy change, model-capacity
change, validation-metric selection, or result-driven repair is authorized from
this result. Any future experiment must be a separately named and preregistered
task that does not overwrite or reinterpret this final held-out result.

## PM-076 Verdict

PM-076 verdict: PASS.

The final claim is limited to one frozen-route held-out evaluation with weak
pooled lift and mixed ticker results. The result is recorded honestly, with no
retune, reselection, rerun, checkpoint staging, notebook action, code edit, or
evidence-matrix/claim-map promotion.
