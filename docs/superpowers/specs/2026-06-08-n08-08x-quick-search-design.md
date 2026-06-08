# N08 #5F-6 — 08X Quick-Search Loop Design (RUN_08X_QUICK_SEARCH)

> Status: design 2026-06-08. Slice ④b of Block 1 (08X) — the loop that turns the
> #5F-5 single-trial atom into the first train-inner scientific evidence
> (`08x_trial_ledger.csv` + `08x_seed_summary.csv` + `08x_failure_ledger.csv`).
> Train-inner only; no official-validation / holdout contact (AGENTS §4.1).
> Tooling: inline design + `humanize:ask-codex` design review
> (`.humanize/skill/2026-06-08_02-17-21-248-0497f9f5/`). No P0; 5 P1s folded in
> below; Q1–Q9 adjudicated by the contract owner.

## 1. Goal & Scope

`run_single_trial` (#5F-5), the fold builders (#5F-2/#5E-1), the windowed-index
chain (#5F-4), and `compute_trial_metrics` (#5E-2) all exist. This slice wires
them into a finite, pre-registered `(candidate × fold × seed)` loop for the
QUICK budget tier (§11: 4–8 configs, 1–2 folds, 1–2 seeds), assembles the §8.3
ledger + §8.4 failure map + the §14.1 seed summary, applies the §14.4
class-collapse guard, and dispatches on `RUN_08X_QUICK_SEARCH`.

**In scope**
- `models/deep_sequence/registry.py`: extend to the 5 SEARCH_ELIGIBLE families.
- `stages/deep_sequence_trial.py`: getattr-guard the non-epoch model reads (P1/Q1).
- `stages/deep_sequence_fold_build.py`: expose `resolve_train_inner_arrays`
  (masked `X, y, ticker_ids, timestamps`) + a `fold_assignment_sha256` helper.
- new `stages/deep_sequence_quick_search.py::run_quick_search`.
- `stages/deep_sequence_exploration.py`: dispatch `RUN_08X_QUICK_SEARCH`.
- synthetic tests.

**Out of scope (deferred)**: tier escalation (§11.1) + medium/aggressive tiers;
concentration gates (ticker/date/time-of-day — no hard threshold constant; date/
time need per-row metadata not on the trial output) (Q5); late-fusion wrapper
variants (`late_average` / `logits_sum` / `small_fusion_mlp`) (Q2); candidate
compression table population (08F).

## 2. Research-safety invariants
- Fit ONLY on `X[train_idx]`; score ONLY `X[val_idx]` (purge/embargo already in
  the fold indices). No official-validation / holdout read.
- Every trial-ledger + run-manifest row: `scope="exploratory"`,
  `official_validation_used=False`, `holdout_test_authorized=False`
  (live validators enforce; Q9).
- Search space (families, configs, config hashes, HPO method, seed list,
  budgets, `search_space_sha256`) is materialized, validated, and persisted
  BEFORE trial 0 (P1/Q3 — preregistration).
- Folds are derived once from a single source (`build_fold_plan`) over the
  masked train arrays; the exact membership is hashed into the manifest so the
  evidence is provably scored on the recorded folds (P1/Q6).

## 3. Codex design-review outcomes folded in
- **P1-1 / Q6** fold consistency by **assignment hash**, not counts. Add
  `fold_assignment_sha256` over the ordered `(fold_id, train_idx, val_idx)`
  arrays; if `08x_fold_results.csv` exists, compare the FULL recomputed frame and
  fail loud on any drift.
- **P1-2 / Q8** enforce the QUICK envelope upper bounds (`n_folds ≤ 2`,
  `n_seeds ≤ 2`, `n_candidates ≤ 8`) AND `grid ≤ scientific_budget_cap` (250).
  Smaller-than-quick grids are allowed but recorded as `quick_evidence_complete=False`
  (test/smoke, not complete quick evidence).
- **P1-3 / Q3** default-per-family candidates are written + hash-stamped into
  `08x_search_space.json` and validated BEFORE any fit.
- **P1-4 / Q9** missing optional deps (LightGBM) are a dependency preflight
  failure (fail loud before trial 0), NOT a `training_divergence` row.
- **P1-5 / Q9** fold class-balance preflight: a fold whose train OR val slice is
  single-class yields `skipped` rows (no fit), keeping the failure map clean.
- **Q1** getattr-guard `max_epochs` / `actual_epochs_` / `early_stop_reason_` in
  `run_single_trial` (runner, not fake attrs on models).
- **Q2** `ms_dlinear_tcn → DLinearTrendPlusTCNResidualFusion`.
- **Q4** seed LCB: `z=1.96`, `seed_mean − 1.96·seed_std/√n`, normal descriptive,
  `n=1 → lcb=mean`; grain = per-seed mean over completed folds, then mean/std
  over seeds; metrics `{macro_f1, delta_macro_f1_vs_dummy, balanced_accuracy}`.
- **Q5** class-collapse → keep metrics, `fit_status=failed`,
  `failure_type=class_collapse`, exclude from seed summary, include in failure
  ledger.
- **Q7** OVERWRITE (never append) the three CSVs; manifest carries full
  provenance (shas + counts + tier + timestamp).

## 4. Module changes

### 4.1 `registry.py` — 5 SEARCH_ELIGIBLE families
```python
SEQUENCE_CLASSIFIER_REGISTRY: dict[str, type] = {
    "dlinear_only": DLinearClassifier,
    "tcn_only": TCNClassifier,
    "ms_dlinear_tcn": DLinearTrendPlusTCNResidualFusion,  # Q2: jointly-trained
    "last_step_mlp_sequence_ablation": LastStepMLPSequenceAblation,
    "last_step_lightgbm_control": LastStepLightGBMControl,  # no max_epochs
}
```
`shallow_gru` / `shallow_lstm` stay unregistered (not search-eligible). Lazy
imports keep `folds`/`base` torch-free; the registry imports the model classes.

### 4.2 `run_single_trial` — non-epoch guard (Q1)
Replace the direct `model.max_epochs` / `actual_epochs_` / `early_stop_reason_`
reads with:
```python
me = getattr(model, "max_epochs", None)
row["max_epochs"] = int(me) if type(me) is int else _NAN
# ... after fit:
ae = getattr(model, "actual_epochs_", None)
row["actual_epochs"] = int(ae) if type(ae) is int else _NAN
row["early_stop_reason"] = getattr(model, "early_stop_reason_", None) or ""
```
LightGBM control (and the late-fusion wrappers) thus record honest `completed`
rows with `max_epochs=NaN` instead of a spurious `training_divergence`.

### 4.3 `deep_sequence_fold_build.py` — shared resolver + fold hash
- Extract `resolve_train_inner_arrays(config, injected) -> (X, y, ticker_ids,
  timestamps)` ALL masked to `PARTITION_TRAIN` using the SAME partition-domain
  guard `resolve_train_inner_index` applies; `resolve_train_inner_index`
  delegates to it and returns `(timestamps, ticker_ids)` (no behavior change to
  the fold-build path; its tests stay green).
- `fold_assignment_sha256(folds: list[tuple[str, np.ndarray, np.ndarray]]) -> str`:
  deterministic sha256 over `fold_id` + the int64 bytes of each `train_idx` /
  `val_idx` (sorted arrays from the builders → stable).

### 4.4 new `stages/deep_sequence_quick_search.py::run_quick_search(config, out)`
Ordered, preregistration-correct:
1. `_refuse_rebuild_over_existing_trials(out)` is NOT called here — QUICK_SEARCH
   OWNS the trial ledger; it overwrites (Q7). (BUILD_FOLDS still refuses to
   rebuild folds under existing trials — correct.)
2. `candidate = _frozen_candidate(config)`; `resolve_train_inner_arrays(config)`.
3. `fold_plan = build_fold_plan(config)`; `_assert_label_horizon_provenance`.
   Build folds in-memory via `_build_for_scheme`; collect
   `folds = [(f"{scheme}__{i}", train_idx, val_idx), ...]`; compute
   `fold_assignment_sha256`. Recompute the fold-results frame; if
   `08x_fold_results.csv` exists, full-frame compare (fail loud on drift) else
   write it.
4. Resolve candidate list (§4.5) → `[(family, candidate_id, model_config,
   config_hash)]`; assert candidate_ids unique (trial-id collision guard, Q9).
5. Resolve seeds from `config['search_space']['seed_list']` (default `[0]`).
6. **Budget gates** (§4.6): scientific cap + QUICK envelope upper bounds.
7. Write + validate `08x_search_space.json` (with `search_space_sha256`) — BEFORE
   any fit (P1-3).
8. **Dependency preflight** (§4.7): import each declared family's heavy dep;
   missing → raise (P1-4).
9. **Fold class-balance preflight** (§4.8): mark unusable folds → `skipped` rows.
10. Loop `candidate × usable-fold × seed` → `run_single_trial(...)`; trial_id =
    `f"{candidate_id}__{fold_id}__seed{seed}"`; budget_tier="quick".
11. **Class-collapse guard** (§4.9) per completed row.
12. Assemble + `validate_trial_ledger_frame`; OVERWRITE `08x_trial_ledger.csv`.
13. `08x_seed_summary.csv` (§4.10) + `08x_failure_ledger.csv` (project failed
    rows). OVERWRITE both.
14. `08x_run_manifest.json` with provenance (§4.11).

### 4.5 Candidate resolution (Q3 / P1-3)
```text
families = config['search_space']['architecture_families']
           or list(SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES)   # default = all 5
candidates = config['search_space']['candidates']           # pre-registered, optional
  each: {family ∈ SEARCH_ELIGIBLE, candidate_id, model_config}
else  → one default per family: candidate_id=f"{family}__default", model_config={}
config_hash = sha256(json.dumps({"family":f, "model_config":mc}, sort_keys=True))
```
The materialized list (families, candidates, per_family_trial_budget, hpo_method,
seed_list, eligibility_thresholds, `search_space_sha256`) is written to
`08x_search_space.json` and passed through `validate_08x_search_space` before any
trial.

### 4.6 Budget gates (P1-2 / Q8)
```text
grid = n_candidates * n_usable_folds * n_seeds   # requested = all folds
fail loud if n_folds > 2 or n_seeds > 2 or n_candidates > 8     (QUICK upper bound)
fail loud if grid > min(quick_cap, scientific_budget_cap_total_trials)
quick_evidence_complete = (4 <= n_candidates <= 8) and 1 <= n_folds <= 2
                          and 1 <= n_seeds <= 2     # recorded in the manifest
```
`scientific_budget_cap_total_trials` is read from the search space (≤ 250).

### 4.7 Dependency preflight (P1-4)
For each declared family, import its heavy dependency (lightgbm for the control;
torch is already imported). Missing → `RuntimeError`/`ImportError` with an
actionable message BEFORE trial 0 — a dependency gap is an environment failure,
not a per-trial `training_divergence`.

### 4.8 Fold class-balance preflight (P1-5)
A fold is usable iff `set(y[train_idx]) == {0,1}` AND `set(y[val_idx]) == {0,1}`
(both classes present — `compute_trial_metrics` requires both in val; a 1-class
train cannot yield a 2-class model). Unusable fold → every `(candidate, seed)`
on it is recorded as a `skipped` row (no fit), `failure_type=""`,
`failure_message="single-class train/val fold; skipped before fit"`,
metrics=NaN. Skipped rows count toward `trial_count_skipped` (§11 accounting)
and are NOT projected into the failure ledger (a data property, not a failure).

### 4.9 Class-collapse guard (§14.4 / Q5)
For a `completed` row: if `min(class0_pred_rate, class1_pred_rate) <
CLASS_COLLAPSE_PRED_RATE_MIN (0.05)` → set `fit_status="failed"`,
`failure_type="class_collapse"`, append a message; KEEP the computed metrics
(auditable). Excluded from the seed summary, included in the failure ledger.

### 4.10 Seed summary (Q4)
For each `candidate_id` × `metric ∈ {macro_f1, delta_macro_f1_vs_dummy,
balanced_accuracy}` over COMPLETED rows only:
```text
per_seed_value(s) = mean over that seed's completed folds of <metric>
seed_mean = mean(per_seed_values)
seed_std  = std(per_seed_values, ddof=1)            # NaN if n_seeds < 2
seed_lcb_95 = seed_mean - 1.96 * seed_std / sqrt(n_seeds)   # n=1 → lcb = mean
```
Columns exactly `SEED_SUMMARY_COLUMNS`. Candidates with zero completed trials are
omitted (their failures live in the failure ledger). `n_seeds=1` rows are
low-evidence by construction (lcb = mean).

### 4.11 Run manifest provenance (Q7)
Reuse `write_run_manifest` with `REQUIRED_08X_RUN_MANIFEST_FIELDS`. Set
`search_budget_tier="quick"`, the fold-policy fields (as `write_fold_run_manifest`),
`source_stage0_candidate` = candidate provenance id, the trial counts
(`requested/completed/failed/skipped`), and a `provenance` block:
`search_space_sha256`, `fold_results_sha256`, `fold_assignment_sha256`,
`frozen_candidate_provenance_id`, `data_txt_manifest_present`,
`quick_evidence_complete`, `generated_at_utc` (the only non-deterministic field;
tests assert schema/shas, not its value). git/python/deps stay in the
environment manifest.

## 5. Dispatcher wiring (`deep_sequence_exploration.py`)
Add `QUICK_SEARCH_SWITCH="RUN_08X_QUICK_SEARCH"` to `handled_switches`, remove it
from `OTHER_SWITCHES`, and route `active == QUICK_SEARCH_SWITCH` →
`run_quick_search(config, out)`. The single-handled-switch + unknown-switch
guards are unchanged.

## 6. Test plan (synthetic only)
- registry: each of the 5 families builds; unknown raises.
- runner non-epoch guard: a fake classifier with no `max_epochs` → `completed`
  row, `max_epochs=NaN` (not `training_divergence`).
- happy loop: tiny windowed index (2 tickers, both classes), 1 fold, 2 seeds,
  `dlinear_only` + a monkeypatched trivial family → ledger has
  `n_candidates*n_folds*n_seeds` rows, all schema-valid; seed_summary present.
- search space written + validated before trials (assert file + sha before any
  fit via a monkeypatched `run_single_trial` recorder).
- budget gates: `n_seeds=3` or `grid>cap` → raises before any fit.
- dependency preflight: monkeypatch lightgbm import to fail → raises (not a row).
- fold class-balance: single-class val fold → `skipped` rows, no fit, no failure
  ledger entry.
- class-collapse: family predicting one class → `failed/class_collapse`, in
  failure ledger, excluded from seed_summary; metrics retained.
- seed_summary: deterministic metric stub → mean/std/lcb match hand-calc;
  `n=1 → lcb=mean`.
- fold-assignment drift: pre-existing `08x_fold_results.csv` with altered
  membership → fail loud.
- dispatcher: `RUN_08X_QUICK_SEARCH` routes to `run_quick_search`; >1 handled
  switch still `ValueError`; quick-search no longer in the NotImplementedError set.

## 7. Files
- M `src/intraday_research/models/deep_sequence/registry.py`
- M `src/intraday_research/stages/deep_sequence_trial.py`
- M `src/intraday_research/stages/deep_sequence_fold_build.py`
- A `src/intraday_research/stages/deep_sequence_quick_search.py`
- M `src/intraday_research/stages/deep_sequence_exploration.py`
- A `tests/stages/test_deep_sequence_quick_search.py` (+ registry/runner test adds)
