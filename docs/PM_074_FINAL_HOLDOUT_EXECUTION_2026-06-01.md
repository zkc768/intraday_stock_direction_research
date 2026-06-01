# PM-074 Final Holdout Execution

Date: 2026-06-01

Status: PASS / one authorized final holdout-test run completed / no retry

## Hard-Rule Prompt Used

Role: PM/reconciler for PM-074 through PM-077. Execute only the parent-PM
authorized final holdout-test gate for the frozen route, then review artifacts,
close claim scope, commit exact docs, push, and sync KB.

Context: PM-072/073 were accepted and pushed at
`ac48d2b79405832c2af80c54c9bfa55b2568c485`. PM-073 criteria require a
separate PM-074 parent gate before any final holdout-test opening; the current
delegation is that explicit parent-PM authorization, subject to fresh stop
rules.

Allowed actions:

- Read project rules, environment, runner help/source, route docs, git state,
  KB handoff/log/protocol CSV, and PM-074 output root state.
- If preflight clears, run exactly one final holdout-test command for the
  frozen route.
- On success, inspect artifacts, metrics, route locks, claim scope, and
  no-retune evidence.
- Write only the PM-074, PM-075, and PM-076 docs in `docs/`.
- Commit exactly those three docs, push normally, then sync only the approved KB
  handoff/log/protocol CSV.

Forbidden actions:

- No code edits, notebook edits/execution, `.codegraph` staging, notebook
  staging, checkpoint/artifact staging, route substitutions, threshold changes,
  feature/label/scaler/decision/model-capacity changes, validation-only flags,
  row caps, fallback model, retuning, reselection, rerun for result reasons,
  evidence-matrix edits, claim-map edits, Zotero/PDF/MinerU/paper work, or
  broad evidence promotion.

Stop rules:

- Stop before execution if AGENTS or ENVIRONMENT is unreadable; HEAD differs
  from `origin/master`; ahead/behind differs from `0 0`; tracked/cached diff is
  nonempty; `git diff --check` fails; PM-072/073 docs or KB records are
  missing; PM-074 root exists; runner does not support the exact CLI; `--full-run`
  final holdout-test meaning is ambiguous; the command needs unsupported flags
  or substitutions; data is missing; or route docs disagree.
- If the PM-074 command returns nonzero, stop immediately, preserve partial
  artifacts, and do not retry, tune, or substitute.

Exact command:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe scripts\phase1b_local\local_baseline_matrix.py --data-dir data --output-dir checkpoints\pm_074_final_holdout_ms_dlinear_tcn_20260601 --candidate A --feature-set mentor_clean_v1 --label-mode no_trade_band --threshold-bps 5.0 --model-family torch --models ms_dlinear_tcn --feature-view last_step --window-size 12 --tickers CSCO JPM KO MSFT WMT --seeds 42 --max-epochs 1 --batch-size 256 --split-mode calendar --train-start-ts 1998-01-02 --train-end-ts 2013-09-16 --val-start-ts 2013-09-16 --val-end-ts 2017-01-25 --holdout-start-ts 2017-01-25 --holdout-end-ts 2020-06-06 --full-run
```

Output schema:

- Goal status, optimized prompt used, preflight results, lane verdicts, command
  result, artifact root and child, final metrics if allowed, PM-075 verdict,
  PM-076 claim-scope conclusion, PM-077 commit/push/KB status, files inspected
  and changed, commands run, validation results, unresolved issues, and explicit
  forbidden-action caveat.

Drift checklist:

- Re-check live git/root state before execution, after runtime, before staging,
  after commit, after push, and after KB sync.
- Keep committed docs, working-tree state, checkpoint artifacts, and KB state
  separate.

## Fresh Preflight Results

| Gate | Result |
|---|---|
| `hf_stock_clf/AGENTS.md` readable | PASS |
| `docs/ENVIRONMENT.md` readable | PASS; interpreter is `E:\codex_workspace\_envs\py311_shared\python.exe` |
| `HEAD == origin/master` | PASS; both were `ac48d2b79405832c2af80c54c9bfa55b2568c485` |
| ahead/behind | PASS; `0 0` |
| tracked diff | PASS; empty |
| cached diff | PASS; empty |
| `git diff --check` | PASS; no output |
| known untracked only | PASS_WITH_CAVEAT; `.codegraph/` and three known notebooks only |
| PM-072/073 docs | PASS; readiness and criteria docs present |
| KB PM-072/073 records | PASS; handoff, log, and protocol CSV record PM-072/073 and no prior final access |
| PM-074 output root before execution | PASS; absent |
| data files | PASS; `CSCO.csv`, `JPM.csv`, `KO.csv`, `MSFT.csv`, and `WMT.csv` exist |
| runner CLI support | PASS_WITH_CAVEAT; exact flags supported, route locks enforced by command review and artifact review |
| `--full-run` meaning | PASS; full torch path materializes the calendar holdout interval as internal `test` and writes concrete `split=test` rows |

## Auditor Lane Verdicts

| Lane | Verdict | Notes |
|---|---|---|
| Preflight | PASS | No hard stop after live git/root/data/doc checks. |
| CLI contract | WARN | Exact CLI and `--full-run` semantics are supported; non-validation torch route-lock enforcement is external, so exact command and artifact review are required. |
| One-time holdout-test unsealing | PASS | PM-073 required parent authorization; this delegation provided it. No prior PM-074 frozen-route final artifact was present. |
| Artifact integrity | PASS | Exactly one child run, expected files present, route locks match, six `split=test` rows, concrete metrics present. |
| Claim scope | PASS | Final wording must be weak/mixed and bounded to one frozen-route held-out evaluation. |
| No-retune | PASS | No artifact evidence of retune, reselection, rerun, or fallback. |
| Exact-path git | PASS pending PM-077 | Only the three PM docs are authorized for staging. |
| KB sync | PASS pending push | KB sync is allowed only after verified PM-077 push. |
| Adversarial review | WARN | Preserve dirty-untracked reproducibility caveat and do not claim no test metrics ever existed historically. |

## PM-074 Execution

The final command above was run exactly once from
`E:\codex_workspace\projects\hf_stock_clf`.

Command result:

- exit code: `0`
- wall time: `104.8s`
- stdout: `wrote result rows: 6`
- retry/tune/substitution count: `0`

Output root:

```text
E:\codex_workspace\projects\hf_stock_clf\checkpoints\pm_074_final_holdout_ms_dlinear_tcn_20260601
```

Child run:

```text
phase1b_local_no_trade_band_full_20260531_201555
```

Artifacts:

```text
manifest.csv
metadata.json
results.csv
ms_dlinear_tcn_seed_42\best.pt
ms_dlinear_tcn_seed_42\last.pt
```

## Route Locks Observed

`metadata.json` records:

- `run_mode=full`
- `model_family=torch`
- `models=[ms_dlinear_tcn]`
- `feature_set_id=mentor_clean_v1`
- `label_mode=no_trade_band`
- `threshold_bps=5.0`
- `threshold_source=fixed_pre_registered_5bps`
- `decision_time_policy=post_bar_close_completed_bar`
- `scaler_id=standard_pooled_train_only_v1`
- `scaler_fit_scope=pooled_train_after_per_ticker_chronological_split`
- `tickers=[CSCO, JPM, KO, MSFT, WMT]`
- `seeds=[42]`
- `max_epochs=1`
- `batch_size=256`
- `window_size=12`
- `feature_view=last_step`
- `split_mode=calendar`
- train `[1998-01-02T00:00:00, 2013-09-16T00:00:00)`
- validation `[2013-09-16T00:00:00, 2017-01-25T00:00:00)`
- holdout-test `[2017-01-25T00:00:00, 2020-06-06T00:00:00)`

## Final Holdout-Test Metrics

All rows in `results.csv` are `split=test`.

| ticker | macro F1 | balanced accuracy | precision macro | recall macro | delta macro F1 vs dummy | n test windows |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 0.5090765910439399 | 0.5199419698841798 | 0.5233149809566453 | 0.5199419698841798 | 0.005936494650455426 | 7388 |
| CSCO | 0.552588176595744 | 0.5614058163886396 | 0.570849048883731 | 0.5614058163886396 | 0.05249391250349544 | 1766 |
| JPM | 0.488508942637383 | 0.4971468909935879 | 0.49672827846735024 | 0.4971468909935879 | -0.005339713730280471 | 1743 |
| KO | 0.48839508574021845 | 0.4972965854867695 | 0.49692142126248195 | 0.4972965854867695 | -0.0020172258275721333 | 885 |
| MSFT | 0.5093173440837355 | 0.522188461909108 | 0.5263417877582225 | 0.522188461909108 | 0.006805483062818807 | 1884 |
| WMT | 0.48693956235978486 | 0.5045219638242894 | 0.5055559786900558 | 0.5045219638242894 | -0.01331217888324332 | 1110 |

Every row records `suspicious_status=ok`.

## PM-074 Verdict

PM-074 verdict: PASS.

Exactly one authorized final holdout-test execution completed for the frozen
route. The result is carried forward to PM-075 artifact review and PM-076
claim-scope closeout. No code edits, notebook execution, route changes,
retuning, reselection, rerun, checkpoint staging, `.codegraph` staging, or KB
evidence promotion occurred in PM-074.
