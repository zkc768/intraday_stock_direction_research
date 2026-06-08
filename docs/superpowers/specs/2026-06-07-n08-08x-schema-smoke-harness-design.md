# N08 #5F-1 — 08X Schema-Smoke Harness (`stages/deep_sequence_exploration.py`) Design

> Status: design 2026-06-07. **First slice of the 08X harness body** (post-migration
> §3 gate PASSED at 7e7af05). Tooling: inline design + `humanize:ask-codex` review
> (P0/P1 answers absorbed from Codex run
> `.humanize/skill/2026-06-07_20-04-41-891-d296d492/`).
> Coexistence: Codex owns the 12-validator contract surface
> (`contracts/deep_sequence_exploration.py`); this stage body PRODUCES the artifacts
> those validators check. **No contract edit in this slice** — Codex will add
> `validate_08x_environment_manifest` in a separate commit (Q5 deferred).
> Check `git status` clean + `git log` for parallel Codex work before editing.

## 1. Goal & Scope

Migrate `src/intraday_research/stages/deep_sequence_exploration.py` from
`NotImplementedError` skeleton to a **schema-smoke** body gated on
`RUN_08X_SCHEMA_SMOKE=true`. The smoke path emits all 8 §13.1 08X artifacts
in **minimal-valid mode** (header-only CSVs, minimal JSON), each passing its
corresponding contract validator. **No trial loop, no fold construction, no
model fit, no official-validation read.** Schema-smoke is the first migrated
stage body in the package-first architecture and sets the precedent for the
other 6 stages.

**In scope**:
- `run_stage(config, *, output_dir=None)` signature + body for schema-smoke
- 8 08X artifact writers (4 JSON + 4 CSV header-only + 1 env_manifest)
- 4 stage tests (`tests/stages/`)
- Skeleton-test rename + targeted deletion
- Governance note recording §6.1 supersession

**Out of scope**: trial loop, fold construction (already in `models/deep_sequence/folds.py`),
model fit (already in 10 §7.1 families), `compute_trial_metrics` (already in
`models/deep_sequence/metrics.py`), 08F / 08O artifacts, contract validator
additions, thin-notebook regeneration, search-space loader, HPO library
integration, legacy generator migration (Phase 5).

## 2. Codex P0/P1 Decisions Absorbed

**Q1 — governance**: Package-first canonical. Tech design §6.1 "no active import
from intraday_research" is **stale frozen notebook-posture text** superseded by
AGENTS.md / CODE_ORGANIZATION.md / RESUME_GATES.md / pipeline.yaml. Preserve §6.1
safety intent (no stale helpers; exact-commit Colab install). Do NOT regenerate
the thin notebook in this slice.

**Q2 — signature**:
```python
def run_stage(
    config: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> None: ...
```
Caller loads YAML → dict. No `load_stage_config()` introduced in this slice.
No per-stage dataclass.

**Q3 — output dir resolution**:
1. `output_dir` kwarg if provided
2. else `Path(config["outputs"]["results_dir"])`
Tests pass `tmp_path`. No env-var.

**Q4 — schema-smoke CSV semantics**: **Header-only** (no synthetic rows; rows
would be fake trial evidence).

**Q5 — env_manifest**: emit minimal payload; **no contract validator added in
this slice** (Codex defers to its own follow-up commit). Stage tests assert
key presence directly.

**Q6 — git_dirty**: **Allowed `true`** for schema-smoke (clean-tree forbidding
only applies to 08F freeze).

**Q7 — search_space minimal payload**: adds Codex-required fields
(`search_space_version`, `stage`, `scope`, `low_compute_mode`, `low_compute_submode`,
`seed_list`, `deferred_07g_gaps`).

**Q8 — trial_count_***: **all zeros** (schema-smoke = no trial events).

## 3. `run_stage` Body — Pseudocode

```python
def run_stage(
    config: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
) -> None:
    """Schema-smoke writer for 08X artifacts.

    Gated on `config["run_switches"]["RUN_08X_SCHEMA_SMOKE"]` (default False:
    no-op, log "no work ran"). When True, emits 8 §13.1 08X artifacts in
    minimal-valid mode through their respective contract validators.

    Other RUN_08X_* switches (BUILD_TRAIN_INNER_FOLDS / SEARCH_SPACE_DRY_RUN /
    QUICK/MEDIUM/AGGRESSIVE_SEARCH / AGGREGATE_FAILURE_MAP) raise
    NotImplementedError in this slice (next slices will migrate them).
    """
    switches = config.get("run_switches", {})
    smoke = bool(switches.get("RUN_08X_SCHEMA_SMOKE", False))

    if any switches[RUN_08X_*] true except SCHEMA_SMOKE:
        raise NotImplementedError("only RUN_08X_SCHEMA_SMOKE is implemented in this slice")

    if not smoke:
        logger.info("deep_sequence_exploration: no run-switch enabled, exiting no-op")
        return

    out = output_dir if output_dir is not None else Path(config["outputs"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # Order: search_space first (validates fastest, exposes schema drift before I/O)
    _write_search_space(out, config)
    _write_trial_ledger_header(out)
    _write_fold_results_header(out)
    _write_seed_summary_header(out)
    _write_failure_ledger_header(out)
    _write_candidate_compression_header(out)
    _write_run_manifest(out, config)
    _write_environment_manifest(out, config)
```

## 4. Artifact Payloads

### 4.1 `08x_search_space.json` (Q7)
```json
{
  "search_space_version": "08x_schema_smoke_v1",
  "stage": "08X",
  "scope": "exploratory",
  "architecture_families": ["dlinear_only"],
  "hpo_method": "random_search",
  "eligibility_thresholds": {
    "min_train_inner_lcb_delta_macro_f1": 0.005
  },
  "scientific_budget_cap_total_trials": 1,
  "per_family_trial_budget": {"dlinear_only": 1},
  "low_compute_mode": false,
  "low_compute_submode": "",
  "seed_list": [],
  "deferred_07g_gaps": {},
  "official_validation_used": false,
  "holdout_test_authorized": false
}
```
Passes `validate_08x_search_space` (architecture_families ⊆
SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES; hpo_method ∈ HPO_METHODS; thresholds
numeric; budget cap ≤ 250; per-family complete; official=False; holdout=False;
low_compute_mode=False → submode B nested-fold check skipped).

### 4.2 Five CSV ledgers — header-only

| File | Header columns source |
|---|---|
| `08x_trial_ledger.csv` | `sorted(REQUIRED_TRIAL_LEDGER_COLUMNS)` (29 cols) |
| `08x_fold_results.csv` | `["fold_id", "fold_scheme", "split_index", "train_inner_fit_n", "train_inner_validation_n", "purge_gap_k", "embargo_gap_k"]` (no contract validator; mirrors §8.2 fold-spec) |
| `08x_seed_summary.csv` | `["candidate_id", "metric", "seed_mean", "seed_std", "seed_lcb_95"]` (mirrors §13.3 08O `08o_seed_summary.csv` schema) |
| `08x_failure_ledger.csv` | `["trial_id", "failure_type", "failure_message", "fold_id", "seed", "candidate_family", "candidate_id"]` (no validator; mirrors §8.4 failure_type enum) |
| `08x_candidate_compression_table.csv` | `["candidate_id", "candidate_family", "paper_safe_score", "z_lcb_delta", "z_mean_delta", "z_seed_stability", "z_fold_consistency", "z_per_ticker", "complexity_penalty", "compute_penalty", "compute_tier"]` (mirrors §9.2 score components) |

All written via `pd.DataFrame(columns=...).to_csv(path, index=False)`.

`08x_trial_ledger.csv` round-tripped through `validate_trial_ledger_frame`
(empty-df branch passes — `if df.empty: return` at validator line 683).

### 4.3 `08x_run_manifest.json` (Q8)
```json
{
  "notebook08_version": "08x_schema_smoke_v1",
  "stage": "08X",
  "scope": "exploratory",
  "source_stage0_candidate": "schema_smoke_no_candidate",
  "official_validation_used": false,
  "holdout_test_authorized": false,
  "train_inner_fold_policy": "none_smoke",
  "purge_policy": "none_smoke",
  "embargo_policy": "none_smoke",
  "search_budget_tier": "schema_smoke",
  "trial_count_requested": 0,
  "trial_count_completed": 0,
  "trial_count_failed": 0,
  "trial_count_skipped": 0
}
```
Passes `validate_08x_run_manifest` (all 14 REQUIRED fields present; stage=08X;
scope=exploratory; official=False; holdout=False).

### 4.4 `08x_environment_manifest.json` (Q5)
```json
{
  "manifest_mode": "schema_smoke",
  "python_version": "3.11.x",
  "python_executable_sha256": "<sha256 of sys.executable bytes>",
  "pip_freeze_sha256": "<sha256 of sorted pip freeze output>",
  "dependency_versions": {
    "torch": "<version>",
    "scikit-learn": "<version>",
    "numpy": "<version>",
    "pandas": "<version>",
    "lightgbm": "<version-or-absent-marker>"
  },
  "platform": "<sys.platform>",
  "git_commit": "<repo HEAD sha>",
  "git_dirty": true_or_false
}
```
No contract validator (Q5). Stage test asserts all 8 keys present.

`git_commit` resolution: `subprocess.check_output(["git", "rev-parse", "HEAD"])`
from repo root; on failure (no git / not in repo), record `"unknown"`.

`git_dirty` resolution: `subprocess.run(["git", "diff", "--quiet"])` returncode
inversion; on git-unavailable, record `null`.

`pip_freeze_sha256`: `hashlib.sha256("\n".join(sorted(pip_freeze_lines)).encode("utf-8")).hexdigest()`.

`python_executable_sha256`: `hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()`.

## 5. Test Surface — `tests/stages/test_deep_sequence_exploration.py`

**Rename** `tests/stages/test_deep_sequence_exploration_skeleton.py` →
`tests/stages/test_deep_sequence_exploration.py` (Codex audit #6: drop
`_skeleton` since it no longer reflects state).

**Delete** `test_run_stage_not_implemented`. **Keep** `test_stage_constants`.

**Add four positive tests**:

1. `test_default_run_no_op(tmp_path)` — config with all RUN_08X_* False → no
   files written in `tmp_path`; function returns None.

2. `test_schema_smoke_emits_all_8_artifacts(tmp_path)` — config with
   `RUN_08X_SCHEMA_SMOKE=True` + `output_dir=tmp_path` → all 8 files exist
   at expected paths (filenames from `OUTPUT_FILES_08X` in contract module).

3. `test_schema_smoke_passes_contract_validators(tmp_path)` — after smoke run,
   reload each artifact and re-run its validator:
   - `validate_08x_search_space(json.load("08x_search_space.json"))` → no raise
   - `validate_trial_ledger_frame(pd.read_csv("08x_trial_ledger.csv"))` → no raise (empty branch)
   - `validate_08x_run_manifest(json.load("08x_run_manifest.json"))` → no raise

4. `test_other_run_switches_raise_not_implemented(tmp_path)` — config with
   `RUN_08X_BUILD_TRAIN_INNER_FOLDS=True` (or any non-smoke switch True) →
   `NotImplementedError` raised; no partial artifacts written.

**Test placement** (Codex audit #4): all in `tests/stages/`; no
`tests/contracts/` touch (no validator added).

**Resume gate**: `bash scripts/check_n08_resume_gate.sh` must continue to
exit 0 after rename.

## 6. Skeleton-Test Migration Notes

- File rename via `git mv` (preserves history)
- `test_run_stage_not_implemented` deletion is intentional (the
  `NotImplementedError` it asserts no longer holds for the smoke path)
- `test_stage_constants` retained verbatim (STAGE_NAME + REQUIRED_ARTIFACTS
  still correct, with REQUIRED_ARTIFACTS reflecting smoke-input expectations
  — currently `("notebook07_validation_budget_ledger.csv",)` per skeleton
  line 11)

**Codex audit #5** clarification: schema-smoke does NOT require
`notebook07_validation_budget_ledger.csv` to exist at runtime. Hard
ledger-existence enforcement begins at 08O. The skeleton's
`REQUIRED_ARTIFACTS` tuple is currently informational; this slice keeps it
as-is (later slices may reconcile).

## 7. Governance Supersession Note (Codex flag)

The spec doc records, for future agents, that:

> Tech design §6.1's "Preferred 08X implementation posture: ... no active
> import from `intraday_research`" is **stale frozen text** referring to
> the pre-migration Colab posture. Per
> `AGENTS.md` / `docs/CODE_ORGANIZATION.md` /
> `docs/NOTEBOOK08_RESUME_GATES.md` / `configs/pipeline.yaml` —
> all written or amended AFTER §6.1 was frozen — package-first is
> canonical: substantive 08X work lives in
> `src/intraday_research/stages/deep_sequence_exploration.py::run_stage`.
> §6.1's safety intent (no stale helpers, exact-commit Colab install, no
> floating-branch installs) is preserved and not re-litigated by this
> slice.

This note must appear in §2 (or equivalent) of the committed spec so a
future agent re-reading §6.1 doesn't try to "fix" the apparent contradiction.

## 8. File Touch Manifest

**Modify**:
- `src/intraday_research/stages/deep_sequence_exploration.py` (skeleton →
  schema-smoke body)

**Rename + edit**:
- `tests/stages/test_deep_sequence_exploration_skeleton.py` →
  `tests/stages/test_deep_sequence_exploration.py` (delete one test, add four)

**Create**:
- `docs/superpowers/specs/2026-06-07-n08-08x-schema-smoke-harness-design.md`
  (this spec)

**Do NOT touch**:
- `src/intraday_research/contracts/deep_sequence_exploration.py` (Codex domain)
- `configs/stages/deep_sequence_exploration.yaml` (governance frozen)
- `docs/NOTEBOOK08_*.md` (frozen design docs)
- `docs/NOTEBOOK08_RESUME_GATES.md` (governance)
- Any other `stages/*.py` skeleton (out of scope)

## 9. Acceptance Criteria

1. `bash scripts/check_n08_resume_gate.sh` exits 0
2. `pytest tests/stages/models -q` = 419 passed (+2 skipped) — unchanged
3. `pytest tests/stages tests/contracts tests/notebooks tests/data -q` ≥ 815
   passed (185 contract + others unchanged + 4 new stage tests = 819)
4. New stage tests pass on schema-smoke happy path
5. `validate_08x_search_space` / `validate_trial_ledger_frame` /
   `validate_08x_run_manifest` all clean over written artifacts
6. Default-switches config writes zero files (no-op contract)
7. Non-smoke switches raise `NotImplementedError` cleanly (no partial output)

## 10. Risks & Open Items

- **R1**: `08x_environment_manifest` has no contract validator until Codex
  follows up. Stage test enforces key presence; downstream consumers must
  not assume Codex's eventual validator will accept exactly this shape.
- **R2**: `manifest_mode: "schema_smoke"` is a new field not in any tech
  design §. If Codex's eventual env_manifest validator rejects unknown
  fields, this needs reconciliation.
- **R3**: `08x_fold_results.csv` / `08x_failure_ledger.csv` /
  `08x_candidate_compression_table.csv` headers are inferred from tech
  design §s, not from a contract enum. Header drift between this slice
  and later contract validators is possible — slice B (folds) and slice
  C (quick_search) must re-confirm.
- **R4**: `git_dirty=True` schema-smoke runs are recorded honestly but
  carry no audit gating. Per Codex Q6, that's correct for 08X; future
  08F slice must add the clean-tree gate.
