# PM-LGBM-RUNTIME-007 Review Closure

Date: 2026-05-30
Status: REVIEW HOLD
Mode: review-only / no new smoke / no training / no notebook

This artifact reconciles the PM-LGBM-RUNTIME-007 read-only multi-agent review.
It does not change runner code, tests, notebooks, data, paper artifacts,
evidence matrix, wiki, or Zotero.

## Optimized Prompt

```text
PM-LGBM-RUNTIME-007-REVIEW - LightGBM runtime/smoke closure

Task type: review-only / PM audit / no execution.
Goal: review the existing LightGBM adapter, environment state, and tiny smoke
artifact without making a model-quality, full-run, paper, or performance claim.

Allowed:
- Read hf_stock_clf/AGENTS.md.
- Read docs/PM_LGBM_SPEC_005A_2026-05-30.md.
- Read docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md.
- Read current LightGBM code/test diff.
- Read checkpoints/pm_lgbm_runtime_007/** metadata/results/manifest.
- Use read-only multi-agent review:
  1. Artifact/metadata reviewer.
  2. Code/test scope reviewer.
  3. Synthesis adversary.

Forbidden:
- No new smoke run.
- No training.
- No notebook execution.
- No code edits.
- No dependency install.
- No evidence_matrix/wiki/Zotero updates.
- No paper-to-performance claim.
- No full-run or model-signal claim.
```

## PM Verdict

`PM-LGBM-RUNTIME-007` is not cleared as a full PM route completion.

Accepted narrow statement:

> LightGBM validation-only runtime path executed on a bounded CSCO smoke
> artifact, with test performance metrics embargoed.

Forbidden statements:

- LightGBM has signal.
- LightGBM beats dummy.
- LightGBM full-run passed.
- LightGBM performance evidence exists.
- Paper-only shards support LightGBM performance.

## Review Findings

| area | verdict | PM finding |
|---|---|---|
| Runtime artifact metadata | PASS | One runtime artifact exists under `checkpoints/pm_lgbm_runtime_007/`, with `metadata.json`, `manifest.csv`, and `results.csv`. |
| Model identity in artifact | PASS | `model_family=lightgbm`; `model_name=lightgbm_lgbm_classifier`; no `sklearn_logreg` relabeling in artifact. |
| Protocol metadata in artifact | PASS | `feature_set_id=mentor_clean_v1`, `label_mode=no_trade_band`, `threshold_source=fixed_pre_registered_5bps`, `threshold_bps=5.0`, `decision_time_policy=post_bar_close_completed_bar`, `scaler_id=standard_pooled_train_only_v1`. |
| Validation-only embargo in artifact | PASS | `report_scope=validation_only`, `test_metrics_embargoed=True`, `test_metrics_used=False`. `results.csv` contains no test performance metric columns. |
| Smoke scope | WARNING | Artifact has two rows, `pooled` and `CSCO`, and is a bounded smoke. Validation metrics must not be interpreted as signal. |
| Manifest test distribution fields | WARNING | `manifest.csv` includes holdout/test distribution observability fields such as test counts/class balance. They are not model metrics, but must not be used for threshold, model, or feature selection. |
| Environment mutation | WARNING | `lightgbm==4.6.0` was installed in the shared project interpreter and added to `requirements.txt`. This must be reported as an environment mutation. |
| Protocol-lock staleness | WARNING | `MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md` still records LightGBM as blocked/no path found. After PM-LGBM-SPEC-005A, that statement is historical for the original lock, not proof of current full readiness. |
| CLI route lock | BLOCKER | Current LightGBM CLI path is not hard-locked to `mentor_clean_v1`, `no_trade_band`, and `fixed_pre_registered_5bps`; non-PM feature/label/threshold combinations remain possible. |
| Full-run claim guard | BLOCKER | `--model-family lightgbm --validation-only-report --full-run` remains possible and can produce full-run-style claim metadata, violating the PM adapter scope. |
| Worktree isolation | BLOCKER before commit/package | Dirty worktree includes modified LightGBM files plus unrelated/untracked `.codegraph/`, docs, and notebooks. Broad staging or clean-delivery claims are blocked. |

## Agent Reconciliation

| role | accepted finding | PM action |
|---|---|---|
| Artifact/Metadata Reviewer | Runtime artifact metadata and results rows match validation-only LightGBM protocol; no test performance metric columns in `results.csv`. | Accepted as artifact-level PASS. |
| Code/Test Scope Reviewer | Adapter/tests verify several safeguards, but CLI still allows non-PM route args and full-run claim metadata for LightGBM. | Accepted as BLOCKER. |
| Synthesis Adversary | Tiny smoke is overreadable; dirty worktree and environment mutation must be recorded; paper-only shard is not performance evidence. | Accepted as BLOCKER/WARNING. |

## Required Follow-Up Before Clearance

The next LightGBM work should be a narrow code/test hardening prompt, not a new
smoke:

```text
PM-LGBM-HARDEN-008 - Lock LightGBM adapter to PM validation route

Task type: code/test hardening, no training, no notebook.
Goal: prevent LightGBM validation-only adapter from being run or reported
outside the approved PM route.

Allowed files:
- scripts/phase1b_local/local_baseline_matrix.py
- tests/test_phase1b_local_runner.py

Required behavior:
- Reject `--model-family lightgbm --full-run`.
- Reject or explicitly require LightGBM route args:
  feature_set_id=mentor_clean_v1
  label_mode=no_trade_band
  threshold_bps=5.0
  threshold_source=fixed_pre_registered_5bps
- Preserve validation-only test-metric embargo.
- Add tests for the two rejected paths.

Forbidden:
- No notebook execution.
- No training.
- No new smoke.
- No evidence_matrix/wiki/Zotero updates.
- No broad refactor.
- No full-run or model-signal claim.
```

## Stop Rules

1. Stop if anyone writes "LightGBM has signal", "beats dummy", "full-run passed", or "performance evidence".
2. Stop if any staged set includes `.codegraph/` or notebooks, or uses broad staging.
3. Stop if `manifest.csv` test distribution fields are used for threshold/model/feature selection.
4. Stop if environment mutation is not reported alongside `requirements.txt` and interpreter `lightgbm` version.
5. Stop if paper-only shards are cited as local result evidence.
6. Stop if LightGBM can emit full-run claim metadata.
7. Stop if LightGBM can run outside `mentor_clean_v1` + `no_trade_band` + fixed 5 bps protocol.

## Non-Claims

This review does not claim:

- LightGBM has predictive signal.
- LightGBM beats dummy.
- LightGBM is full-run ready.
- LightGBM is paper-evidence-backed.
- MS-DLinear+TCN is unblocked.
- The current dirty worktree is ready for commit or packaging.
