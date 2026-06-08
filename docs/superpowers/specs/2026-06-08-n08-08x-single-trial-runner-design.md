# N08 #5F-5 — 08X Single-Trial Runner Design

> Status: design 2026-06-08. First atom of ④ QUICK_SEARCH (the first REAL model
> training). `run_single_trial(...)` fits ONE model on ONE fold/seed and returns
> ONE 29-column trial-ledger row (with fail-loud failure handling). #5F-6 (separate
> slice) owns the loop over configs×folds×seeds + ledger/seed-summary/failure
> artifacts + the `RUN_08X_QUICK_SEARCH` dispatch. Tooling: inline design +
> `humanize:ask-codex` review (run `.humanize/skill/2026-06-08_01-45-54-1678-66f57081/`).

## 1. Goal & Scope

The model bodies, folds, windowed index, and `compute_trial_metrics` all exist.
This slice is the atom that ties them into one scored trial: fit a deep-sequence
classifier on the train-inner-fit rows of a fold, score the train-inner-validation
rows, and emit the §8.3 trial-ledger row.

**In scope**: new `models/deep_sequence/registry.py` (family→class, `dlinear_only`
only); new `stages/deep_sequence_trial.py::run_single_trial`; synthetic tests
(tiny real DLinear fit).

**Out of scope**: the trial loop / search-space iteration / ledger I/O (#5F-6);
families beyond `dlinear_only`; budget-tier escalation; class-collapse /
concentration gates (those are #5F-6 over the assembled ledger).

## 2. Research-safety invariants (Codex Q8)
- Fit ONLY on `X[train_idx]` (train-inner-fit); score ONLY `X[val_idx]`
  (train-inner-validation). No other data. (Purge/embargo already encoded in the
  fold indices by #5F-2/#5F-4.)
- Every row: `scope="exploratory"`, `official_validation_used=False`,
  `holdout_test_authorized=False`, `compute_tier="full_compute"`.
- Invalid indices (out of bounds / empty split / **train∩val overlap**) → raise
  loud (caller contract), NOT a failed row.

## 3. New module — `models/deep_sequence/registry.py`

```python
SEQUENCE_CLASSIFIER_REGISTRY: dict[str, type] = {
    "dlinear_only": DLinearClassifier,
}

def build_classifier(family: str, *, random_state: int, **config):
    if family not in SEQUENCE_CLASSIFIER_REGISTRY:
        raise ValueError(f"unknown family {family!r}; known {sorted(REGISTRY)}")
    return SEQUENCE_CLASSIFIER_REGISTRY[family](random_state=random_state, **config)
```
(#5F-6 extends the registry to the other SEARCH_ELIGIBLE families incl.
`last_step_lightgbm_control`.)

## 4. New module — `stages/deep_sequence_trial.py`

```python
def run_single_trial(
    X: np.ndarray, y: np.ndarray, ticker_ids: np.ndarray,
    *,
    train_idx: np.ndarray, val_idx: np.ndarray,
    trial_id: str, candidate_family: str, candidate_id: str,
    config_hash: str, fold_id: str, seed: int, budget_tier: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one model on a fold/seed and return one 29-col trial-ledger row."""
```

Algorithm:
1. **Validate (pre-fit, fail loud)**: X 3-D, y/ticker_ids aligned with X;
   train_idx/val_idx int arrays, non-empty, in `[0, len(X))`, disjoint.
2. Seed bookkeeping defaults into the row (all 29 cols; `failure_type=""`,
   `failure_message=""`, metrics/actual_epochs/peak_memory = NaN until filled).
3. **Snapshot RNG** (Codex Q3): Python `random`, `np.random`, `torch` CPU RNG,
   and CUDA RNG (if available).
4. `tracemalloc.start()`; `t0 = perf_counter()`.
5. `try`:
   - `model = build_classifier(candidate_family, random_state=seed, **model_config)`
   - `model.fit(X[train_idx], y[train_idx])`
   - `proba = model.predict_proba(X[val_idx])`; `y_pred = proba.argmax(axis=1)`
   - `metrics = compute_trial_metrics(y[val_idx], y_pred, ticker_ids[val_idx])`
   - fill 8 metric cols; `fit_status="completed"`;
     `max_epochs=model.max_epochs`; `actual_epochs=model.actual_epochs_`;
     `early_stop_reason=model.early_stop_reason_ or ""`.
6. `except` → `fit_status="failed"`; `failure_type=_classify_failure(exc)`;
   `failure_message=str(exc)[:500]`; metric cols stay NaN; `actual_epochs=NaN`.
7. `finally`: `peak_memory_mb = tracemalloc peak / 1e6`; `tracemalloc.stop()`;
   restore RNG; `actual_wall_clock_seconds = perf_counter()-t0`;
   `gpu_seconds_or_null=None`.
8. `train_inner_fit_n=len(train_idx)`, `train_inner_validation_n=len(val_idx)`.
9. return row.

`_classify_failure(exc)` (Codex Q4):
- `MemoryError` → `"memory_error"`
- `TimeoutError` → `"timeout"`
- `NotImplementedError` → `"not_implemented"`
- `ValueError | TypeError` → `"artifact_schema_failure"` (bad config / bad shape)
- else → `"training_divergence"`

`max_epochs` comes from the instantiated `model.max_epochs` (resolved default;
Codex Q7); on a construction failure (model never built) it falls back to
`model_config.get("max_epochs", NaN)`.

`peak_memory_mb` is best-effort Python-heap (tracemalloc tracks Python allocs,
not torch C++ tensors); documented as such. `NaN` on failure paths where it
cannot be measured.

## 5. Tests — `tests/stages/test_deep_sequence_trial.py` (synthetic, tiny real fit)

Synthetic `X (N, T, F)` float32, `y (N,) {0,1}`, `ticker_ids (N,)`,
`train_idx`/`val_idx` disjoint. DLinear with `max_epochs=1, batch_size=8` (a real
but tiny fit — fast).
1. happy path: returns a dict with EXACTLY `REQUIRED_TRIAL_LEDGER_COLUMNS`;
   `fit_status="completed"`; 8 metric cols finite; `train_inner_fit_n==len(train_idx)`;
   `scope=="exploratory"`, official/holdout False; `gpu_seconds_or_null is None`.
2. a single-row trial frame passes `validate_trial_ledger_frame`
   (`pd.DataFrame([row])`).
3. determinism: same seed + same data → identical metric values (two calls).
4. RNG isolation: a call does not change the global `np.random` draw sequence
   (snapshot/restore works).
5. failure: a model_config that makes fit raise (e.g. bad axis value) → row with
   `fit_status="failed"`, mapped `failure_type`, NaN metrics; does NOT raise.
6. invalid indices: empty val_idx / out-of-bounds / train∩val overlap → ValueError
   (fail loud, not a failed row).
7. registry: `build_classifier("dlinear_only", random_state=0)` builds a
   DLinearClassifier; unknown family → ValueError.

## 6. Files touched
- NEW `src/intraday_research/models/deep_sequence/registry.py`
- NEW `src/intraday_research/stages/deep_sequence_trial.py`
- NEW `tests/stages/test_deep_sequence_trial.py`
- NEW this spec

**Do NOT touch**: the model bodies, metrics, contract module, fold_build,
windowed_index.

## 7. Acceptance criteria
1. `bash scripts/check_n08_resume_gate.sh` exits 0.
2. `pytest tests/stages tests/stages/models -q` green (existing + new trial tests).
3. A completed trial row passes `validate_trial_ledger_frame`.
4. Deterministic given seed; RNG-isolated (no global-state leak across calls).
5. Model-fit failures produce a `failed` row (no crash); invalid indices raise.
6. No official-validation / holdout data touched.

## 8. Next (#5F-6, the loop)
Iterate the search space (configs) × folds (from the fold builders) × seeds per
budget tier, call `run_single_trial`, assemble `08x_trial_ledger.csv` +
`08x_seed_summary.csv` + `08x_failure_ledger.csv`, apply the class-collapse /
concentration gates, and wire `RUN_08X_QUICK_SEARCH`. Extend the registry to the
other SEARCH_ELIGIBLE families (incl. `last_step_lightgbm_control`).
