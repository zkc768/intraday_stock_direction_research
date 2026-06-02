# P0 Multi-Ticker Validation Smoke Report

Date: 2026-06-02
Scope: validation-only smoke, not evidence-ready.

## Command Shape

The smoke was run through the project Python with a temporary script:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -c "exec(open(...temporary multi-ticker smoke script...).read())"
```

The temporary script executed `notebooks/04_ian_research_memo.ipynb` in memory,
then used the notebook-defined functions on train/validation data only.

## Boundary

Allowed in this smoke:

- Read raw OHLCV CSVs for CSCO, JPM, KO, MSFT, and WMT.
- Build `baseline_v1` features, no-trade labels, chronological splits,
  train-only scaler, per-ticker/per-split windows, dummy baselines, and one
  small LightGBM last-step diagnostic.
- Use train and validation rows only.

Not allowed and not done:

- No closed holdout/test scoring.
- No notebook output saved.
- No checkpoint, artifact, model file, or raw data mutation.
- No threshold selection, feature selection, model selection, or claim update.

## Sample Size

| Item | Value |
|---|---:|
| Tickers | CSCO, JPM, KO, MSFT, WMT |
| Total rows | 1885348 |
| Train rows | 1552934 |
| Validation rows | 332414 |
| Valid train labels | 284185 |
| Valid validation labels | 18988 |
| Train windows | 94450 |
| Validation windows | 4652 |

## Window Class Balance

| Ticker | Split | Windows | Class 0 | Class 1 |
|---|---|---:|---:|---:|
| CSCO | train | 25645 | 13335 | 12310 |
| CSCO | validation | 1100 | 540 | 560 |
| JPM | train | 23192 | 11890 | 11302 |
| JPM | validation | 1154 | 594 | 560 |
| KO | train | 11647 | 5651 | 5996 |
| KO | validation | 471 | 238 | 233 |
| MSFT | train | 19280 | 10045 | 9235 |
| MSFT | validation | 1370 | 640 | 730 |
| WMT | train | 14686 | 7527 | 7159 |
| WMT | validation | 557 | 295 | 262 |

## Dummy Baseline Smoke

| Seed | Macro F1 | Balanced Accuracy | Validation n |
|---:|---:|---:|---:|
| 41 | 0.492220 | 0.492368 | 4652 |
| 42 | 0.491588 | 0.491711 | 4652 |
| 43 | 0.504514 | 0.504543 | 4652 |

Mean dummy macro F1: `0.496107`.

Mean dummy balanced accuracy: `0.496207`.

## LightGBM Last-Step Smoke

| Item | Value |
|---|---:|
| Train sample | 10000 |
| Validation sample | 4652 |
| Macro F1 | 0.396066 |
| Balanced accuracy | 0.509320 |

This only proves that the rebuilt notebook functions can feed a small
validation-only LightGBM diagnostic. It is not a tuned LightGBM result.

## Non-Decision Policy

This smoke result must not become a hard rule for later work.

It may be used for:

- confirming that the P0 notebook pipeline is executable in memory;
- detecting implementation bugs in feature, label, split, scaler, window, or
  metric logic;
- deciding whether another validation-only smoke should be broader or better
  instrumented.

It must not be used for:

- selecting features;
- selecting the no-trade threshold;
- selecting LightGBM over another model or rejecting LightGBM;
- tuning hyperparameters;
- changing the final interpretation;
- opening, justifying, or interpreting holdout/test;
- claiming model performance.

Any future full validation must be treated as a separate pre-registered
validation task with its own approved scope, command, output path, and claim
language. This smoke is a pipeline diagnostic, not evidence.

## Interpretation

The multi-ticker P0 pipeline ran end-to-end on train/validation data only.
The LightGBM diagnostic is weak versus dummy on macro F1 and only slightly above
dummy on balanced accuracy. That is useful as a caution signal, but not as a
decision signal.

The correct response is to keep the result quarantined as a smoke diagnostic
artifact:
fix implementation bugs if found, improve instrumentation if needed, and design
the next validation task independently.
