# Code Simplification And Generator Boundary Design

> Status: proposed design, docs-only.
> Date: 2026-06-07.
> Scope: repository organization, generator simplification, and research-safety
> guard classification for `intraday_stock_direction_research`.
> Non-scope: notebook execution, training, holdout/test access, metric changes,
> threshold changes, or thesis-result wording changes.

## 1. Goal

Make the codebase easier to review in small blocks without weakening the
research contract.

The current problem is not just "too much defensive code". The project has a
migration overlap:

- numbered notebook route: `02_...`, `07_...`, `08_...`;
- semantic package-first route: `validation_synthesis_gap_audit`,
  `deep_sequence_exploration`;
- legacy script paths: `scripts/create_*_colab_notebook.py`;
- target generator paths: `scripts/notebooks/generate_*_colab.py`;
- compatibility contract shims in `scripts/notebook0{6,7,8}_contract.py`;
- canonical contracts in `src/intraday_research/contracts/`.

This overlap makes scripts look defensive, verbose, and confusing even when
some guards are scientifically necessary.

## 2. Skills And Review Lenses Used

- `brainstorming`: separated the problem into naming, boundary, guard, and
  implementation phases before proposing changes.
- `notebook-code-reviewer`: treated generated notebooks and generators as
  reproducibility artifacts, with leakage, hidden state, and stale-output risks.
- `Time Series Analysis`: kept chronology, split boundaries, and causal feature
  timing as first-class constraints.
- `python-expert-best-practices-code-review`: classified Python error handling
  into fail-fast required-key checks, broad-except risks, unnecessary comments,
  and over-defensive branches.
- `writing-plans`: used only as a structure reference for future implementation
  planning; no implementation plan is included here.

## 3. Current Evidence From Live Files

### 3.1 Naming state

`configs/pipeline.yaml` already defines semantic target names:

| Stage | Target notebook | Legacy notebook |
|---|---|---|
| `validation_synthesis_gap_audit` | `notebooks/validation_synthesis_gap_audit_colab.ipynb` | `notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb` |
| `deep_sequence_exploration` | `notebooks/deep_sequence_exploration_colab.ipynb` | `notebooks/08_deep_sequence_exploration_colab.ipynb` |

`docs/LEGACY_NAME_MAPPING.md` says the generator target is
`scripts/notebooks/generate_<stage>_colab.py`, but the actual active files are
still flat `scripts/create_*_colab_notebook.py`.

Conclusion: the semantic names are the intended target. The confusing part is
that only part of the migration has landed.

### 3.2 Generator size and responsibility

Current script sizes from live file scans:

| File | Approx. lines | Issue |
|---|---:|---|
| `scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py` | 2219 | Generator, runtime helpers, ledger IO, artifact checks, Drive backup, and static validation are mixed. |
| `scripts/create_deep_sequence_exploration_colab_notebook.py` | 1828 | Generator, schema stubs, trial-loop logic, freeze/readout gates, Drive backup, and static validation are mixed. |
| `src/intraday_research/stages/validation_synthesis_gap_audit.py` | 20 | Stage entrypoint exists but is still a `NotImplementedError` skeleton. |
| `src/intraday_research/stages/deep_sequence_exploration.py` | 20 | Stage entrypoint exists but is still a `NotImplementedError` skeleton. |

Conclusion: N07/N08 generator files are carrying stage runtime responsibilities
because package-first stage entrypoints are not yet implemented.

### 3.3 Contract state

Canonical contract code is already in:

- `src/intraday_research/contracts/validation_synthesis_gap_audit.py`
- `src/intraday_research/contracts/deep_sequence_exploration.py`

The old paths:

- `scripts/notebook07_contract.py`
- `scripts/notebook08_contract.py`

are compatibility shims. Some generated notebook markdown and generator text
still says "inline copy of scripts/notebook07_contract.py" or
"scripts/notebook08_contract.py", which is now misleading.

## 4. Guard Classification

The simplification must not remove scientific safety guards. The right question
is: is this guard protecting research validity, or is it compensating for code
layout confusion?

### 4.1 Keep: research-safety guards

These are not optional defensive code. They protect validity.

| Guard | Keep because |
|---|---|
| chronological split checks | Random or shuffled time-series validation invalidates conclusions. |
| label horizon boundary invalidation | Prevents train/validation and validation/closed-holdout leakage. |
| per-ticker window construction | Prevents windows crossing tickers. |
| train-only scaler/preprocessor checks | Prevents validation leakage. |
| `holdout_test_contact=false` manifest checks | Final holdout/test is closed for this route. |
| `RUN_* = False` defaults for heavy or official-read cells | Prevents accidental notebook execution and validation-budget misuse. |
| append-before-read validation-budget ledger checks | Required by AGENTS.md section 4.3 for N07/N08. |
| artifact schema checks and exact missing-path failures | Prevents silent use of stale or incomplete artifacts. |
| generated-notebook empty-output and AST-parse checks | Prevents hidden state and stale committed outputs. |

### 4.2 Simplify: engineering-defense overhead

These can be reduced after tests pin behavior.

| Pattern | Better shape |
|---|---|
| Repeating the same path-existence checks inside every phase block | One typed preflight per stage or phase, returning a small artifact bundle. |
| Duplicating output filename dictionaries between generator and contract | Put canonical artifact names in `src/intraday_research/contracts/` or stage config; generator reads them. |
| Markdown/text saying the contract lives in `scripts/notebook0*_contract.py` | Rewrite to the canonical `src/intraday_research/contracts/...` path. |
| Broad `except Exception` around artifact hashing or IO | Catch expected exceptions only, or let fail-fast exceptions surface with exact path context. |
| Long explanatory comments that restate code | Replace with function names and one short comment only for research invariants. |
| Stage runtime code inside notebook generator cells | Move to `src/intraday_research/stages/<stage>.py`; notebook calls a tested entrypoint. |
| Compatibility shim imports used as active source references | Keep shims for old callers, but stop pointing new code and docs at them. |

### 4.3 Do not simplify yet

These look verbose but should not be touched until their replacement is tested.

- `EXPECTED_DESIGN_DOC_SHA256` and readout authorization checks.
- N07/N08 ledger monotonicity and prefix-invariance checks.
- N08 `schema_only_stub` wording protections until real readout completeness is
  proven beyond header/non-empty checks.
- Static gates that reject forbidden imports, `drive.mount`, or holdout/test
  strings.

## 5. Brainstormed Approaches

### Approach A: Documentation-only alignment

Only update docs and generator markdown to stop referring to legacy shims as
canonical.

Pros:

- lowest risk;
- no behavior change;
- useful before implementation.

Cons:

- generator files stay too large;
- old `scripts/create_*` names remain visible;
- code still feels defensive because the structure is unchanged.

### Approach B: Path and shim cleanup first

Move active generator bodies into `scripts/notebooks/generate_*_colab.py` and
leave old `scripts/create_*_colab_notebook.py` files as thin wrappers.

Pros:

- fixes the visible naming problem;
- follows `docs/LEGACY_NAME_MAPPING.md`;
- low behavior risk if wrappers call the moved `main()`.

Cons:

- still leaves runtime logic inside generator bodies;
- requires updating static gates and pipeline path tests.

### Approach C: Package-first extraction first

Move reusable runtime helpers from N07/N08 generators into
`src/intraday_research/stages/` and keep generators as thin notebook shells.

Pros:

- addresses the real complexity source;
- makes code review smaller and testable;
- aligns with AGENTS.md section 7.2.

Cons:

- higher risk;
- must be done in very small chunks;
- requires behavior-equivalence tests for artifact schemas and static gates.

### Recommended sequence

Use A -> B -> C, but start with only N07/N08.

Do not rename the entire repo or all notebooks in one pass. Do not move 02-06
until N07/N08 prove the pattern.

## 6. Target Architecture

### 6.1 Authoritative registry

`configs/pipeline.yaml` is the ordered stage registry:

```text
stage name -> config -> stage module -> contract -> generator -> notebook
```

Notebook order should be read from the registry, not inferred from filename
prefixes.

### 6.2 Contracts

Canonical:

```text
src/intraday_research/contracts/validation_synthesis_gap_audit.py
src/intraday_research/contracts/deep_sequence_exploration.py
```

Compatibility only:

```text
scripts/notebook07_contract.py
scripts/notebook08_contract.py
```

Rule: new generator text, tests, and docs should refer to canonical contract
modules. Shims remain only so historical notebooks and imports do not break.

### 6.3 Stage runtime

Target:

```text
src/intraday_research/stages/validation_synthesis_gap_audit.py
src/intraday_research/stages/deep_sequence_exploration.py
```

Each stage should expose:

```python
def run_stage(config):
    ...
```

For the first extraction pass, the stage entrypoints can remain small. Good
first targets are pure helpers:

- canonical JSON/CSV hashing;
- ledger append and prefix-invariance helpers;
- artifact preflight bundles;
- run manifest writing;
- canonical output filename maps.

Do not move thesis prose generation, plotting, Drive backup, or exploratory
search logic until the stage boundary is stable.

### 6.4 Generators

Target:

```text
scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py
scripts/notebooks/generate_deep_sequence_exploration_colab.py
```

Legacy wrappers:

```text
scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py
scripts/create_deep_sequence_exploration_colab_notebook.py
```

Wrapper behavior:

```python
from scripts.notebooks.generate_validation_synthesis_gap_audit_colab import main

if __name__ == "__main__":
    main()
```

Generators should own notebook structure only:

- markdown sequence;
- setup/config cells;
- exact-commit package install cell once package-backed path is enabled;
- stage invocation cell;
- result display cells;
- notebook static self-check.

## 7. Phased Change Plan

### Phase 0: Freeze the current review surface

Purpose: confirm baseline before moving names or helpers.

Actions:

- record `git status --short` and `git diff --stat`;
- run targeted static gates only, not notebooks;
- list current N07/N08 generator target notebook paths;
- confirm no holdout/test path is touched.

Acceptance:

- no generated outputs added;
- no notebook execution counts changed;
- ambient user changes remain untouched.

### Phase 1: Correct misleading references

Purpose: reduce cognitive overhead without moving code.

Actions:

- update generator markdown strings that say the active contract is
  `scripts/notebook07_contract.py` or `scripts/notebook08_contract.py`;
- change wording to "inline copy of canonical contract module";
- update comments that point future edits to legacy shims;
- keep old import shims untouched.

Acceptance:

- static gates still pass;
- generated notebook still inlines canonical contract code;
- no artifact schema changes.

### Phase 2: Relocate N07/N08 generators

Purpose: fix the visible naming mismatch.

Actions:

- create `scripts/notebooks/`;
- move generator bodies to `generate_validation_synthesis_gap_audit_colab.py`
  and `generate_deep_sequence_exploration_colab.py`;
- replace old `scripts/create_*` files with thin wrappers;
- update `configs/pipeline.yaml` and tests only where they still target old
  generator bodies as canonical.

Acceptance:

- old commands still work through wrappers;
- target paths in `configs/pipeline.yaml` match actual files;
- no notebook body changes except expected generator metadata if any.

### Phase 3: Extract shared stage helpers

Purpose: reduce generator size without changing research behavior.

Candidate N07 extraction:

- `sha256_file`, `canonical_csv_bytes`, `canonical_json_bytes`;
- ledger append/read/flush helpers;
- N05/N06 artifact preflight helpers;
- run manifest writer.

Candidate N08 extraction:

- `operator_readout_authorization_sha_runtime` if it can delegate to contract
  code directly;
- ledger append/write helpers;
- trial ledger frame helpers;
- output artifact completeness helpers.

Acceptance:

- extracted helpers have direct unit tests or contract tests;
- generated notebook inlined code still matches the canonical helper source
  where static gates require it;
- CSV column order and required manifest fields are unchanged.

### Phase 4: Make generated notebooks thinner

Purpose: transition from self-contained runtime notebooks to package-backed
execution/reporting notebooks.

Actions:

- add exact-commit install cell;
- call `intraday_research.stages.<stage>.run_stage(config)`;
- write manifest with repo URL, commit, config hash, notebook hash,
  input/output artifacts, validation scope, and `holdout_test_contact=false`;
- retain result display cells.

Acceptance:

- package import is pinned to an exact commit, not a branch;
- static gate verifies stage entrypoint and manifest fields;
- no `drive.mount` in default setup;
- no holdout/test read, transform, summary, or score.

### Phase 5: Reduce remaining defensive code

Purpose: remove defensive noise once stage boundaries are clear.

Rules:

- replace repeated existence checks with one preflight object per phase;
- use required dictionary keys (`payload["field"]`) for required schema fields;
- avoid broad `except Exception` unless re-raised with exact context;
- remove comments that restate obvious assignments;
- keep scientific guard comments short and explicit.

Acceptance:

- smaller files;
- no loss of exact missing-path errors;
- no swallowed exceptions;
- targeted static and contract tests pass.

## 8. Naming Recommendation

The repository name `intraday_stock_direction_research` is acceptable. It says
what the thesis project studies and is clearer than old PM-style names.

Use this naming policy:

| Entity | Policy |
|---|---|
| Stage names | semantic snake_case from `configs/pipeline.yaml`. |
| Long-term notebooks | semantic `_colab.ipynb`, no numeric prefix. |
| Legacy notebooks | keep numbered paths until static gates and external links migrate. |
| Generators | `scripts/notebooks/generate_<stage>_colab.py`. |
| Old generator paths | thin compatibility wrappers only. |
| Contracts | canonical in `src/intraday_research/contracts/`. |
| Old contract paths | compatibility shims only. |

Do not do a repo-wide rename while N07/N08 stage bodies are still skeletons.
The immediate value comes from making canonical vs legacy paths explicit.

## 9. Review Blocks For Future Code Review

Use these blocks instead of reviewing the whole codebase at once.

### Block 1: Registry and names

Files:

- `configs/pipeline.yaml`
- `docs/LEGACY_NAME_MAPPING.md`
- `docs/CODE_ORGANIZATION.md`
- static-gate path constants

Question:

- Does every stage have one canonical name and one clearly labeled legacy path?

### Block 2: Contract source of truth

Files:

- `src/intraday_research/contracts/*.py`
- `scripts/notebook0{6,7,8}_contract.py`
- contract tests

Question:

- Are tests and generators reading canonical contract modules, not shims?

### Block 3: Generator responsibility

Files:

- `scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py`
- `scripts/create_deep_sequence_exploration_colab_notebook.py`
- future `scripts/notebooks/generate_*.py`

Question:

- Is this file building notebook structure, or is it secretly running the
  stage?

### Block 4: Research-safety guards

Files:

- notebook static gates;
- artifact contract tests;
- stage configs;
- generated notebooks.

Question:

- Do chronology, no-holdout, ledger-before-read, train-only preprocessing, and
  dummy-baseline constraints still fail closed?

### Block 5: Python simplification

Files:

- extracted helpers in `src/intraday_research/stages/`;
- generator wrappers;
- shared IO helpers.

Question:

- Is the code fail-fast and readable without repeated local defensive branches?

## 10. Verification Budget

This is a docs/design cleanup. Verification should stay static unless a later
implementation phase explicitly authorizes runtime work.

Recommended checks per phase:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile <changed .py files>
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest <targeted static/contract tests> -q
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --stat
```

Do not run notebooks. Do not train models. Do not inspect holdout/test.

## 11. First Implementation Slice

The first safe implementation slice should be Phase 1 only:

1. Change misleading generator markdown/comments from `scripts/notebook07_contract.py`
   and `scripts/notebook08_contract.py` to canonical contract-module wording.
2. Add or update a static assertion that generated notebooks do not describe the
   shim as canonical.
3. Regenerate only if the static gate expects notebook text to change.
4. Run targeted N07/N08 static gates and contract tests.

This gives an immediate clarity win without moving code or changing behavior.

## 12. Open Questions Before Implementation

1. Should N07 stay numbered until its stage body exists, or should it follow N08
   into semantic notebook naming immediately?
2. Should generator relocation happen before or after extracting N07/N08 helper
   functions into `src/intraday_research/stages/`?
3. Should run-copy notebooks such as
   `07_validation_synthesis_and_gap_audit_colab_drive_core_full_run.ipynb`
   stay ignored in place, move to `artifacts/review_packets/`, or remain
   untouched until after package-first conversion?

