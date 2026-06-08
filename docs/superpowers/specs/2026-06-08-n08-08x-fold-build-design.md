# N08 #5F-2 — 08X `BUILD_TRAIN_INNER_FOLDS` Fold-Build Slice Design

> Status: design 2026-06-08. First sub-slice of the 08X trial harness (Slice 1 in
> the 2026-06-08 continuation handoff). Wires the EXISTING
> `models/deep_sequence/folds.py` into the package stage behind
> `RUN_08X_BUILD_TRAIN_INNER_FOLDS`, producing real `08x_fold_results.csv` rows.
> NO model fit, NO official-validation read. Tooling: inline design +
> `humanize:ask-codex` review (run `.humanize/skill/2026-06-07_23-51-18-53-08e6b005/`).
> Codex (contract owner) sanctioned the one contract-module addition (Q6).

## 1. Goal & Scope

Add the `RUN_08X_BUILD_TRAIN_INNER_FOLDS` dispatch path to
`stages/deep_sequence_exploration.py::run_stage`. When enabled, the stage builds
train-inner folds from the windowed **official-train-partition** sample index and
writes real rows to `08x_fold_results.csv`, plus updates the fold-policy fields
of `08x_run_manifest.json`. The heavy fold logic already exists in
`models/deep_sequence/folds.py`; this slice is WIRING + a thin pure executor +
one contract validator.

**In scope**: pure fold-results builder; train-partition resolution seam;
run_stage dispatch; `08x_run_manifest.json` policy update;
`validate_08x_fold_results_frame` contract validator (Codex-sanctioned);
synthetic-array tests.

**Out of scope**: model fit / predict / trial loop / metrics (later slices);
08F / 08O; real-data execution run (this slice ships wiring + synthetic tests;
the real-data path is gated/manual); search-space candidate selection.

## 2. Research-safety invariants (acceptance gates)

1. **08X folds are built ONLY on `PARTITION_TRAIN` rows** (AGENTS §4.1 / tech
   design §8.1). The official-validation and closed-holdout partitions are never
   folded or read.
2. **No model fit, no official-validation read** in this path.
3. **`label_horizon_k` provenance**: the fold plan's `label_horizon_k` MUST equal
   the frozen Stage-0 candidate `horizon_k` (`source_stage0_candidate.horizon_k`
   when present in config). Mismatch fails loud (Codex Q5 — keeps the purge
   pre-registered, not inferred).
4. **Scheme selection is explicit** (Codex Q2): `validation_design.fold_modes` in
   the YAML is an ALLOW-LIST, not the plan. BUILD_FOLDS runs only the explicitly
   selected scheme(s); default = `["rolling_origin_folds"]` (forward-chaining).
   `purged_time_series_folds` / `embargoed_train_inner_folds` run only when the
   config/search-space explicitly selects them.
5. **Fail-loud** on fold construction failure (Codex Q7): a fold builder
   `ValueError` (e.g. too few samples) propagates; it is an invalid fold
   policy/index before trial 0, not a per-trial model failure. Failure-ledger
   rows are a later (fit) slice concern.

## 3. New module — `stages/deep_sequence_fold_build.py`

Mirrors the `deep_sequence_schema_smoke.py` / `deep_sequence_official_readout.py`
module pattern.

```python
@dataclass(frozen=True)
class FoldSpec:
    scheme: str            # one of FOLD_SCHEMES
    n_folds: int
    label_horizon_k: int
    inner_validation_size: int   # used by rolling_origin only
    embargo_size: int            # used by embargoed only; 0 otherwise

FOLD_SCHEMES = (
    "rolling_origin_folds",
    "purged_time_series_folds",
    "embargoed_train_inner_folds",
)

def build_fold_plan(config: Mapping[str, Any]) -> list[FoldSpec]: ...
def build_fold_results(
    timestamps: np.ndarray,
    ticker_ids: np.ndarray,
    fold_plan: Sequence[FoldSpec],
) -> pd.DataFrame: ...                     # PURE — synthetic-testable
def resolve_train_inner_index(
    config: Mapping[str, Any],
    injected_window_index: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...    # seam; filters to PARTITION_TRAIN
def write_fold_results(out: Path, df: pd.DataFrame) -> None: ...
```

### 3.1 `build_fold_plan(config)`
Read `config["fold_plan"]` (preferred) or fall back to selecting from
`config` defaults:
- `selected_fold_modes`: list, default `["rolling_origin_folds"]`. Each must be in
  `FOLD_SCHEMES`. (NOT the YAML allow-list — explicit selection only, Codex Q2.)
- `n_folds`: materialized int from the budget tier (`quick`→2, `medium`→3,
  `aggressive`→3) or an explicit `n_folds` field.
- `label_horizon_k`: explicit int; asserted == `source_stage0_candidate.horizon_k`
  in the dispatch (§4 step 3).
- `inner_validation_size`: explicit int (rolling_origin).
- `embargo_size`: explicit int (embargoed; default 0).

### 3.2 `build_fold_results(timestamps, ticker_ids, fold_plan)` — PURE
For each `FoldSpec`, call the matching `folds.py` builder, enumerate folds, emit
one row per `(scheme, fold_i)`:

| column | value |
|---|---|
| `fold_id` | `f"{scheme}__{i}"` (string, unique) |
| `fold_scheme` | `scheme` |
| `split_index` | `i` (int, 0-based) |
| `train_inner_fit_n` | `len(train_idx)` |
| `train_inner_validation_n` | `len(val_idx)` |
| `purge_gap_k` | `label_horizon_k` |
| `embargo_gap_k` | `embargo_size` (0 for non-embargoed) |

Builder argument mapping:
- `rolling_origin_folds(ts, tk, n_folds, inner_validation_size, label_horizon_k)`
- `purged_time_series_folds(ts, tk, n_folds, label_horizon_k)`
- `embargoed_train_inner_folds(ts, tk, n_folds, label_horizon_k, embargo_size)`

Returns a `pd.DataFrame` with exactly `FOLD_RESULTS_COLUMNS` (imported from
`deep_sequence_schema_smoke`). Validated via `validate_08x_fold_results_frame`
before return.

### 3.3 `resolve_train_inner_index(config, injected_window_index=None)` — SEAM
- If `injected_window_index` is given (tests): use its
  `target_partition` / `target_timestamps` / `target_ticker_ids`.
- Else (production): call the data layer
  (`load_ticker_bars`→`build_features`→`build_no_trade_band_labels`→
  `apply_chronological_split`→`build_windows`) from `config["inputs"]`. If the
  data config is absent/incomplete, raise a clear error ("real-data fold build
  requires inputs.*; this run had none") — fail-loud, never silently empty.
- In BOTH paths: filter to `target_partition == PARTITION_TRAIN`, return
  `(target_timestamps[train], target_ticker_ids[train])`. Assert the result is
  non-empty and train-only.

## 4. run_stage dispatch — `_run_08x_build_folds(config, out)`

1. `ts, tk = resolve_train_inner_index(config, injected_window_index=...)`
   (the dispatch reads an optional injected index from config for tests).
2. `plan = build_fold_plan(config)`.
3. Provenance gate (§2.3): if `source_stage0_candidate.horizon_k` present in
   config, assert every `FoldSpec.label_horizon_k == horizon_k`; else fail.
4. `df = build_fold_results(ts, tk, plan)`; `write_fold_results(out, df)`
   (overwrite `08x_fold_results.csv` with real rows).
5. Update `08x_run_manifest.json` policy fields (Codex Q4):
   - `train_inner_fold_policy = "+".join(selected_modes)`
   - `purge_policy = f"horizon_bar_purge_k={label_horizon_k}"`
   - `embargo_policy = "none"` or `f"symmetric_embargo_k={embargo_size}"`
   - Create the manifest skeleton if missing; do NOT clobber non-fold artifacts.
   Overwrite ONLY `08x_fold_results.csv` + `08x_run_manifest.json` for this switch.
6. `validate_08x_fold_results_frame(df, require_non_empty=True)`.

Switch wiring in `run_stage`: add `RUN_08X_BUILD_TRAIN_INNER_FOLDS` to
`handled_switches`; remove it from `OTHER_SWITCHES`. SCHEMA_SMOKE + BUILD_FOLDS
remain mutually exclusive per invocation (same pattern as the
smoke/readout guard).

## 5. Contract addition (Codex-sanctioned, Q6)

In `contracts/deep_sequence_exploration.py`:

```python
FOLD_RESULTS_REQUIRED_COLUMNS = {
    "fold_id", "fold_scheme", "split_index",
    "train_inner_fit_n", "train_inner_validation_n",
    "purge_gap_k", "embargo_gap_k",
}
FOLD_SCHEMES = (... three schemes ...)

def validate_08x_fold_results_frame(df, *, require_non_empty=False) -> None:
    # required columns present
    # if df.empty: raise iff require_non_empty else return
    # fold_id: unique, non-empty strings
    # fold_scheme in FOLD_SCHEMES
    # split_index: int >= 0
    # train_inner_fit_n, train_inner_validation_n: int > 0
    # purge_gap_k, embargo_gap_k: int >= 0
```

Schema-smoke's header-only writer calls it with `require_non_empty=False`
(empty branch returns); BUILD_FOLDS calls `require_non_empty=True`.

## 6. Tests — `tests/stages/test_deep_sequence_fold_build.py` (synthetic only)

1. `test_build_fold_results_rolling_origin` — synthetic `(ts, tk)`, one scheme;
   assert row count = n_folds, columns == FOLD_RESULTS_COLUMNS, counts/gaps correct.
2. `test_build_fold_results_multi_scheme` — selected = all 3; row blocks per scheme.
3. `test_train_partition_filter` — injected window index with mixed
   PARTITION_TRAIN/PARTITION_VALIDATION; assert only train rows reach the builder
   (fold counts match the train-only subset).
4. `test_label_horizon_provenance_mismatch_fails` — plan k != candidate horizon_k → raises.
5. `test_dispatch_writes_real_fold_results` — run_stage with
   `RUN_08X_BUILD_TRAIN_INNER_FOLDS=True` + injected synthetic index →
   `08x_fold_results.csv` non-empty + passes validator; manifest policies set.
6. `test_fold_build_fails_loud_on_too_few_samples` — tiny index → ValueError propagates.
7. `test_default_scheme_is_rolling_origin_only` — no explicit selection → plan = [rolling_origin].
8. Contract tests (in `tests/contracts/`): `validate_08x_fold_results_frame`
   happy + each rejection (bad scheme, dup fold_id, negative split_index,
   zero counts, empty when require_non_empty).
9. Existing schema-smoke test still green (header-only fold_results +
   `require_non_empty=False`).

## 7. Open item (Codex flagged)
The explicit selected fold plan lives in `config["fold_plan"]` for this slice
(stage config / search-space payload). A future slice may move it into
`08x_search_space.json` as the pre-registered selection; this slice does not infer
selection from the 3-mode YAML allow-list.

## 8. Files touched
- NEW `src/intraday_research/stages/deep_sequence_fold_build.py`
- MODIFY `src/intraday_research/stages/deep_sequence_exploration.py` (dispatch +
  switch wiring)
- MODIFY `src/intraday_research/contracts/deep_sequence_exploration.py` (add
  `validate_08x_fold_results_frame` + constants) — Codex-sanctioned
- NEW `tests/stages/test_deep_sequence_fold_build.py`
- MODIFY `tests/contracts/test_deep_sequence_exploration_contract.py` (validator tests)
- NEW this spec

**Do NOT touch**: folds.py, the data layer, other stage modules, frozen design docs.

## 9. Acceptance criteria
1. `bash scripts/check_n08_resume_gate.sh` exits 0.
2. `pytest tests/stages/models -q` unchanged (419 + 2 skip).
3. `pytest tests/stages tests/contracts tests/notebooks tests/data -q` green
   (existing + new fold-build + validator tests).
4. BUILD_FOLDS dispatch writes non-empty `08x_fold_results.csv` passing
   `validate_08x_fold_results_frame(require_non_empty=True)` on synthetic index.
5. Train-partition-only folding proven by test 3.
6. label_horizon_k provenance gate proven by test 4.
7. No real ticker CSV touched by any test.
