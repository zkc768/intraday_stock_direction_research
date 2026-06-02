# Baseline Reference

Date: 2026-05-30
Status: baseline reference only

## Current Baseline

This file records the current baseline setup and the research-safety boundaries
around it. Some implementation-readiness notes below are historical because the
old model runner has since been archived.

Current baseline state:

- `baseline_v1`, `no_trade_band`, explicit fixed `threshold_bps=5.0`,
  `post_bar_close_completed_bar`, and pooled train-only scaling after
  per-ticker chronological split remain the default baseline.
- The prior LightGBM record is diagnostic/setup observability only, not
  model-performance evidence and not permission to tune or open test/holdout.
- `sklearn_logreg` is the dependency-free active diagnostic model path.
- LightGBM is declared as an optional active tiny adapter dependency and remains
  validation-only diagnostic, not model-performance evidence.
- Combined MS-DLinear+TCN exists only as archived helper-library reference until
  a separate active adapter/spec task is approved.
- "Ready" in the feature table below means post-bar-close feature availability
  under this baseline only; it is not experiment readiness, model
  evidence, or test/holdout authorization.
- Papers may motivate design constraints only. They cannot select thresholds,
  tune capacity, authorize test/holdout access, or serve as local evidence.

This reference reconciles Ian's 2026-05-29 mentor direction with the current
`intraday_stock_direction_research` historical runner state and the `stock_ml_knowledge_base`
research-support artifacts. It is intentionally docs-only: no helper-library, runner,
notebook, training, Zotero, wiki, or evidence-matrix changes are part of this
reference.

## Source Alignment

Ian's latest direction says the current 5-minute technical indicators look
weak and the next work should clean features and check selective prediction:
remove non-stationary raw OHLCV/raw volume/raw MACD, use normalized and
decision-time-safe features, keep the no-trade band, and rerun LightGBM plus
MS-DLinear+TCN only when those implementations are real and fair.

Active code state:

- `intraday_research/baseline_v1.py` defines the active `baseline_v1` feature,
  label, split-boundary, scaler, window, and dummy-baseline helper contracts.
- `intraday_research/validation_pipeline.py` defines the active validation-only
  report builder, dependency-free `sklearn_logreg` diagnostic, LightGBM tiny
  adapter precheck/result, feature diagnostics, and walk-forward contract specs.
- `scripts/run_validation_only_pipeline_smoke.py` is the active CLI smoke entry
  point. It prints a validation-only report and must not be treated as a
  research-result claim without artifact review.
- Archived references contain old runner and model material, but archive paths
  are not active execution paths unless the user explicitly asks for historical
  reconstruction.
- The current scaler is fit on the concatenated training frames after
  chronological per-ticker splitting, then applied to train/validation/test.

The knowledge-base mentor specs remain useful as baseline requirements, but
their statement that `baseline_v1` still needs implementation is stale
relative to the current runner.

## Scope

Allowed for this reference:

- Read repository and knowledge-base evidence.
- Create or review this single baseline reference.
- Draft the next tiny validation prompt.
- Run syntax or docs-only checks that do not train and do not mutate model
  artifacts.

Forbidden for this reference:

- Editing the archived helper library, the runner, notebooks, data, evidence matrix, wiki, or
  Zotero.
- Running training, notebooks, full baselines, Colab jobs, or paper-table
  regeneration.
- Re-labeling `sklearn_logreg` as LightGBM.
- Treating a notebook prototype as the canonical MS-DLinear+TCN implementation.
- Selecting thresholds or features from test performance.

## Decision-Time Convention

`baseline_v1` is a post-bar-close feature set: the model
scores only after the current 5-minute bar is complete and before the next
actionable interval.

If the desired production decision is before the current bar closes, then all
features using current-bar `close`, `high`, `low`, or `volume` must be lagged
or rebuilt before validation. Under that stricter pre-close convention, this
reference is not sufficient.

## Feature Availability Table

| Feature | Current implementation basis | Decision-time availability | Warmup / grouping | Baseline decision |
| --- | --- | --- | --- | --- |
| `log_return` | `log(close[t]) - log(close[t-1])` | Available only after `close[t]` is known. | Grouped by ticker and trading day when ticker exists. First bar per group drops. | Ready under post-bar-close convention; lag if pre-close. |
| `close_to_open_return` | `close[t] / open[t] - 1` | Uses current bar close; unavailable before bar close. | No rolling warmup. | Ready only under post-bar-close convention. |
| `high_low_range` | `log(high[t] / low[t])` after raw OHLCV sanity checks. | Uses current bar high/low; unavailable before bar close. | No rolling warmup. | Ready only under post-bar-close convention; raw rows fail if prices are non-positive, `high < low`, `open`/`close` outside high-low, or volume is negative. |
| `rolling_volatility_20` | Prior 20-bar std of `log_return`, grouped by ticker/day. | Available after the current bar return is known, but the statistic excludes the current row. | Prior-only shift means first valid value is at 0-based bar 21 in a full trading day. | Revised after critical review to avoid current-row self-reference; this intentionally costs one extra warmup bar. |
| `normalized_volume_20` | `log1p(volume[t]) - prior_rolling_mean_20(log1p(volume))`. | Uses current completed volume against a same-day prior-only baseline. | Prior-only shift means first valid value is at 0-based bar 20 in a full trading day. | Revised after critical review to avoid current-row self-reference and previous-day carryover; this intentionally costs one extra warmup bar. |
| `rsi_14` | Rolling 14-bar average gain/loss from close differences. | Uses close history through current completed bar. | Grouped by ticker/day; warmup rows drop. | Ready under post-bar-close convention. |
| `bollinger_pctb` | Current close relative to rolling Bollinger mean/std. | Uses current completed close and trailing window. | Grouped by ticker/day; warmup and zero-width-band rows are invalid. | Revised after critical review to avoid infinite values. |
| `normalized_macd_hist` | MACD histogram from continuous per-ticker close EMA history, normalized by EMA-26. | Uses close history through current completed bar, including prior-session EMA state for the same ticker. | Continuous per ticker, not reset by trading day. | Revised after critical review to avoid daily EMA reset and afternoon-only survivorship; the cross-day EMA carry is an explicit post-bar-close feature semantic, not label-horizon permission. |
| `time_of_day_sin` | Sine transform of minutes since regular-market open over a 395-minute endpoint-safe period. | Known before and after the bar. | No warmup. | Revised after critical review to avoid mapping 9:30 and 16:00 to the same point; not a 390-minute session wrap. |
| `time_of_day_cos` | Cosine transform of minutes since regular-market open over a 395-minute endpoint-safe period. | Known before and after the bar. | No warmup. | Revised after critical review to avoid mapping 9:30 and 16:00 to the same point; not a 390-minute session wrap. |

Feature set:

- `feature_set_id = baseline_v1`
- `feature_view = sequence` for torch baselines unless the next prompt
  explicitly uses the validation-only `sklearn_logreg` tabular path.
- `technical_v1` is allowed only as a historical control, not as the mentor
  clean baseline.

## Scaler And Threshold Policy

Scaler policy:

- `scaler_id = standard_pooled_train_only_v1`
- Fit scope: concatenate training frames after chronological per-ticker splits,
  then fit `StandardScaler` only on that training pool.
- Transform scope: apply the fitted scaler to train, validation, and test.
- Forbidden: fitting on validation/test, fitting before chronological splits,
  per-full-sample normalization, or changing scaler type without writing down
  the change first.

Future result rows should carry explicit metadata:

- `feature_set_id`
- `decision_time_policy`
- `scaler_id`
- `scaler_fit_scope`
- `threshold_source`
- `threshold_bps`

Threshold policy:

- For the first approved tiny validation, use
  `threshold_source = fixed_pre_registered_5bps`.
- Pass `--label-mode no_trade_band --threshold-bps 5.0` explicitly even though
  the current runner defaults no-trade-band threshold to 5.0 bps.
- Validation-selected thresholds are allowed only in a separate approved task,
  must use train/validation only, and must be written down before any test
  readout.
- Forbidden: choosing the no-trade threshold from test metrics, paper tables,
  or old notebook outcomes.

## Model Availability

The notes in this section explain active diagnostic paths and archived model
references. They do not authorize runtime, model selection, evidence promotion,
or test/holdout access.

sklearn_logreg: active dependency-free validation-only diagnostic.

Reason: it is cheap, available through scikit-learn, and useful as a sanity
check. It is not a sequence model and must not be described as using the full
12-step window.

LightGBM: active tiny adapter dependency, validation-only diagnostic.

Reason: `requirements.txt` declares LightGBM and the active pipeline prechecks
the dependency before running the tiny adapter. That does not make LightGBM a
selected model or evidence-ready path. Smoke values are pipeline diagnostics
only.

Minimum next task:

- If execution is approved, run the active validation-only smoke script with
  explicit metadata and review the resulting report before making any research
  statement.

MS-DLinear+TCN: archived model/tests reference only; active shape/spec audit
required before validation use.

Reason: archive material contains combined-model references and related tests,
while the active notebook path still needs an explicit shape/spec and migration
audit. No archived or prototype implementation becomes canonical until
separately reviewed, cleaned, and promoted.

Minimum next task:

- Approve a model-spec task that defines the combined architecture, shape
  contract, sequence length, loss/metric parity, and tests before any training.

## Three-Step Workflow

Step 1: Baseline check.

- Confirm this baseline reference is the active baseline reference.
- Confirm the decision-time convention is post-bar-close.
- Confirm `threshold_bps = 5.0` is fixed and pre-registered.
- Stop if the user wants pre-close prediction, a different threshold policy, or
  a different feature set.

Step 2: Tiny validation-only smoke.

- Use only existing approved notebook or runner paths.
- Prefer the cheapest validation-only smoke that checks leakage and split
  mechanics. `sklearn_logreg` remains an optional sanity baseline; LightGBM or
  MS-DLinear+TCN smoke requires the relevant migration/audit boundary first.
- Do not run torch training unless separately approved.
- Stop on missing raw data, unsupported CLI flags, class collapse, split/scaler
  policy mismatch, or any attempt to inspect test metrics for selection.

Step 3: Implementation review.

- If tiny validation passes, choose one implementation path to prepare:
  LightGBM adapter review or combined MS-DLinear+TCN model spec.
- Add tests before training.
- Keep paper/evidence-matrix updates paused until real validation artifacts
  exist and are reviewed.

## Tiny Validation Prompt

Copyable prompt for the next approved task:

```text
BASELINE-TINY-VAL -- baseline_v1 validation-only smoke

Task type: tiny validation / no notebook / no full training.
Goal: verify that baseline_v1 can produce a validation-only sanity report
under the post-bar-close, train-only-scaler, fixed-5bps no-trade-band
baseline.

Allowed files/actions:
- Read intraday_stock_direction_research/AGENTS.md and docs/BASELINE_REFERENCE.md.
- Run at most one active validation-only smoke through
  `scripts/run_validation_only_pipeline_smoke.py`.
- Write only to a new, clearly named validation output directory under
  intraday_stock_direction_research/checkpoints/baseline_v1_tiny_validation_2026-05-30/.

Forbidden:
- No archived helper-library edits.
- No archived runner execution.
- No notebook execution.
- No torch training.
- No LightGBM evidence claim from a smoke value.
- No combined MS-DLinear+TCN claim.
- No evidence_matrix/wiki/Zotero update.
- No threshold selection from test metrics.

Required baseline metadata:
- feature_set_id=baseline_v1
- decision_time_policy=post_bar_close_completed_bar
- scaler_id=standard_pooled_train_only_v1
- label_mode=no_trade_band
- threshold_source=fixed_pre_registered_5bps
- threshold_bps=5.0
- report_scope=validation_only

Stop rules:
- Stop if required raw data path is missing.
- Stop if the active smoke script cannot produce validation-only metadata.
- Stop if any command would train torch models or execute notebooks.
- Stop if scaler fit is not train-only after chronological split.
- Stop if output would overwrite an existing artifact directory.
- Stop if the result cannot report macro F1, balanced accuracy, class counts,
  and dummy-baseline comparison.

Final report:
- Exact command run.
- Files read and files written.
- Validation artifact path.
- Confirmed scaler/threshold metadata, or exact missing fields.
- No paper/result claim beyond "validation-only smoke completed" unless the
  artifact is inspected and supports it.
```

## Minimal Next Approval Needed

Approve only `BASELINE-TINY-VAL` if the next move should execute anything.
The approval should state whether the validation may write a new
checkpoint/output directory. Without that approval, the baseline remains
read-only and reference-only.
