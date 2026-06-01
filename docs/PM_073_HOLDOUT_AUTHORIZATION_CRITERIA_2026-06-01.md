# PM-073 Holdout Authorization Criteria

Date: 2026-06-01

Status: PASS_WITH_CAVEAT / criteria-ready only / parent PM approval required
before any final holdout/test execution

## Scope

PM-073 freezes the criteria for a possible future PM-074 final holdout/test
execution gate. It does not run final holdout/test, authorize final holdout/test
inside this window, expose test/holdout metrics, select from validation metrics,
retune, rerun, or promote evidence.

PM-073 may be used only because PM-072 returned `PASS_WITH_CAVEAT`.

## Required Lineage For PM-074

PM-074 may open only if all of the following are true at the start of that
future gate:

| Requirement | Required value |
|---|---|
| Base docs commit | PM-072/073 docs commit is present and pushed to `origin/master`. |
| Prior validation lineage | PM-059/060, PM-065B/066, and PM-068/069/070 remain accepted as validation-only/non-claim lineage. |
| Current readiness verdict | PM-072 remains `PASS_WITH_CAVEAT` or better, with no later blocker. |
| Current criteria verdict | PM-073 remains `PASS_WITH_CAVEAT` or better, with no later blocker. |
| Required artifact root | `checkpoints/pm_069_controlled_validation_only_model_testing_20260601`. |
| Required child run | `phase1b_local_no_trade_band_smoke_20260531_192407`. |
| Required accepted docs | `docs/PM_068_VALIDATION_ONLY_MODEL_TESTING_SPEC_2026-06-01.md`, `docs/PM_069_CONTROLLED_VALIDATION_ONLY_RUNTIME_CLOSEOUT_2026-06-01.md`, and `docs/PM_069_CONTROLLED_VALIDATION_ONLY_ARTIFACT_REVIEW_070_2026-06-01.md`. |
| Repo state | `HEAD == origin/master`; ahead/behind `0 0`; tracked and cached diffs empty before the PM-074 command is reviewed. |
| KB state | KB handoff/log/protocol CSV record the PM-072/073 package and no intervening blocker. |

If any required lineage item is missing, stale, contradicted, or not pushed,
PM-074 must stop before any execution.

## Frozen Candidate

The only candidate eligible for PM-074 is:

| Field | Frozen value |
|---|---|
| `model_family` | `torch` |
| `models` | `[ms_dlinear_tcn]` |
| `feature_set_id` | `mentor_clean_v1` |
| `label_mode` | `no_trade_band` |
| `threshold_bps` | `5.0` |
| `threshold_source` | `fixed_pre_registered_5bps` |
| `decision_time_policy` | `post_bar_close_completed_bar` |
| `scaler_id` | `standard_pooled_train_only_v1` |
| `scaler_fit_scope` | `pooled_train_after_per_ticker_chronological_split` |
| `tickers` | `[CSCO, JPM, KO, MSFT, WMT]` |
| `seeds` | `[42]` |
| `max_epochs` | `1` |
| `batch_size` | `256` |
| `window_size` | `12` |
| `feature_view` | `last_step` |
| split mode | `calendar` |
| train interval | `[1998-01-02, 2013-09-16)` |
| validation interval | `[2013-09-16, 2017-01-25)` |
| holdout/test interval | `[2017-01-25, 2020-06-06)` |

No alternate model family, model list, seed, feature set, threshold, scaler,
decision-time policy, label mode, split, row cap, epoch budget, or ticker list
is authorized by PM-073.

## Pre-Holdout Claim Scope

Before PM-074 is executed and reviewed, the only allowed claim is:

The project has a frozen route-control candidate that passed validation-only
protocol-observability review and is eligible for parent-PM consideration of a
single final holdout/test execution gate.

Forbidden pre-holdout claims:

- model-quality, profitability, robustness, publishability, or deployment
  claims;
- claims that MS-DLinear+TCN is better than another model family;
- claims based on validation metric ranking, comparison, or fallback;
- claims that the route is already final-test proven;
- claims that Ian guidance or papers are local experiment results.

After PM-074, any claim must be written only after post-artifact review confirms
the exact command, output schema, route locks, and one-time rule. Even then, the
maximum claim scope is one frozen-route held-out evaluation, not broad trading
or model-family superiority.

## One-Time Holdout/Test Rule

PM-074 may execute final holdout/test at most once for this frozen candidate.

After final holdout/test output is produced:

- no retuning;
- no threshold change;
- no feature change;
- no scaler change;
- no decision-time change;
- no label change;
- no model-capacity change;
- no seed search;
- no model-family search;
- no rerun because the result is poor, surprising, or inconvenient;
- no fallback to validation metrics to override, soften, or replace the
  holdout/test result.

If PM-074 fails before creating artifacts for an environmental reason, the
parent PM must decide whether a repaired command attempt is still the same
one-time gate. The executing window must not silently retry or broaden.

## Future Execution Feasibility

Runner source inspection shows that `scripts/phase1b_local/local_baseline_matrix.py`
supports calendar split boundaries including `--holdout-start-ts` and
`--holdout-end-ts`. In validation-only mode it excludes test data and emits
embargo fields. Outside validation-only mode, the torch path evaluates the test
split, which corresponds to the frozen holdout interval for a calendar split.

Therefore PM-074 can be specified without code changes, subject to parent-PM
approval and fresh preflight.

Allowed future command family:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py `
  --data-dir data `
  --output-dir checkpoints\pm_074_final_holdout_ms_dlinear_tcn_20260601 `
  --candidate A `
  --feature-set mentor_clean_v1 `
  --label-mode no_trade_band `
  --threshold-bps 5.0 `
  --model-family torch `
  --models ms_dlinear_tcn `
  --feature-view last_step `
  --window-size 12 `
  --tickers CSCO JPM KO MSFT WMT `
  --seeds 42 `
  --max-epochs 1 `
  --batch-size 256 `
  --split-mode calendar `
  --train-start-ts 1998-01-02 `
  --train-end-ts 2013-09-16 `
  --val-start-ts 2013-09-16 `
  --val-end-ts 2017-01-25 `
  --holdout-start-ts 2017-01-25 `
  --holdout-end-ts 2020-06-06 `
  --full-run
```

The future PM-074 preflight must confirm whether `--full-run` is the intended
claim-scope/run-mode token for final holdout/test execution. If the parent PM
prefers to keep the output non-claim until post-review, PM-074 must explicitly
resolve that wording before execution. Do not substitute `--smoke`,
`--validation-only-report`, `--validation-only-per-ticker`, row caps, unsupported
metadata-lock flags, or any route variant.

The command above is criteria text only. It was not run in PM-073.

## Expected PM-074 Outputs

PM-074 must produce a new dedicated output root with exactly one child run unless
the command fails before artifact creation for an environmental reason.

Expected output files:

- `metadata.json`;
- `manifest.csv`;
- `results.csv`;
- model checkpoint files produced by the existing torch path.

Expected post-artifact review checks:

- command text matches the approved PM-074 command exactly;
- route locks match PM-073;
- one and only one final holdout/test artifact set exists under the approved
  PM-074 root;
- result rows represent the final test/holdout split and are not validation
  rows;
- concrete test/holdout metric columns are present only because PM-074 explicitly
  opened the one-time final test/holdout gate;
- validation metrics were not used to retune, reselect, rerun, filter, or soften
  interpretation;
- no evidence/claim promotion occurs before post-artifact review;
- final wording stays inside the predeclared one-frozen-route held-out
  evaluation scope.

## PM-074 Stop Rules

Stop before execution if:

- parent PM has not explicitly authorized PM-074;
- `HEAD != origin/master` or ahead/behind is not `0 0`;
- tracked or cached diff is non-empty;
- PM-072/073 docs are not committed and pushed;
- required KB sync is stale or missing;
- PM-069 artifact root, child run, or review docs are missing;
- source docs disagree on any route lock;
- the approved command would need code or runner changes;
- the command includes validation-only flags, row caps, search knobs, route
  variants, unsupported metadata-lock flags, notebooks, or broad runtime;
- the planned output root already exists;
- any human or agent attempts to choose, defer, or revise the candidate based on
  validation metric values.

Stop after execution if:

- more than one child run appears under the PM-074 output root;
- artifacts are missing required files;
- route locks drift;
- outputs indicate validation-only mode rather than the final holdout/test split;
- any rerun, tuning, reselection, or fallback interpretation is requested after
  seeing holdout/test output.

## PM+Agent Audit Results

| Role | Result |
|---|---|
| Lineage Auditor | PASS: PM-074 lineage requirements are exact and tied to PM-072/073 plus PM-068/069/070. |
| Route-Freeze Auditor | PASS: one candidate and one route are frozen. |
| Validation-Leakage Auditor | PASS: no validation metric ranking or fallback may choose or revise the final route. |
| Test-Embargo Auditor | PASS_WITH_CAVEAT: criteria define the one-time opening rule, but no opening occurs until separate parent PM approval. |
| Claim-Scope Auditor | PASS: pre-holdout and post-review claim scopes are bounded. |
| CLI/Execution Feasibility Auditor | PASS_WITH_CAVEAT: current runner appears capable without code changes, but PM-074 must fresh-check `--full-run` run-mode/claim-scope intent before execution. |
| Exact-Path Git Auditor | PASS: PM-073 authorizes no staging except the PM-072/073 docs in this window. |
| KB Sync Auditor | PASS pending commit/push: KB sync should record criteria-ready status and no holdout/test execution. |
| Final Adversarial Reviewer | PASS_WITH_CAVEAT: no retuning loophole remains, but the PM-074 command must be parent-approved and reviewed before use. |

## PM-073 Verdict

PM-073 verdict: PASS_WITH_CAVEAT.

Criteria are ready for parent-PM consideration of a separate PM-074 final
holdout/test execution gate. PM-073 does not authorize PM-074, does not run final
holdout/test, and does not permit any final test/holdout access inside this
window.
