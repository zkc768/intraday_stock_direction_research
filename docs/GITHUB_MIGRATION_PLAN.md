# GitHub Migration Plan

> **For agentic workers:** Execute this plan task-by-task. Do not batch rename
> the whole repository. After each phase, run the listed verification before
> continuing. Do not commit, push, branch, delete raw data, run heavy training,
> or touch holdout/test data unless the user explicitly authorizes it.

## Objective

Migrate the project to a package-first GitHub research layout:

```text
src/intraday_research/ = canonical research logic
configs/               = ordered stage configs and pipeline registry
notebooks/             = thin generated Colab execution/reporting interfaces
tests/                 = package, notebook, stage, and artifact contracts
results/               = reproducible stage outputs
artifacts/             = run manifests, inventories, review packets
reports/ or paper/     = thesis-facing tables, figures, and manuscript assets
```

The migration must preserve the research contract:

- chronological splits only;
- train-only preprocessing;
- no holdout/test contact in validation-only stages;
- no post-validation threshold/model/wording changes;
- validation-budget ledger append-before-read;
- generated notebooks with empty committed outputs;
- Colab package installs pinned to exact git commits.

## Current Inputs

Use these documents as active constraints:

- `AGENTS.md`
- `docs/CODE_ORGANIZATION.md`
- `docs/RESEARCH_WORKFLOW.md`
- `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`
- `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md`
- `docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md`
- `docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md`

## Optimized Agent Prompt

Use this prompt when starting an implementation session:

```text
You are migrating E:\codex_workspace\projects\intraday_stock_direction_research
to the package-first GitHub layout defined in docs/CODE_ORGANIZATION.md and
docs/GITHUB_MIGRATION_PLAN.md.

Read AGENTS.md first. Preserve all research hard rules: chronological splits,
train-only preprocessing, no holdout/test contact, no post-validation selection
or wording changes, and validation-budget ledger append-before-read.

Work phase-by-phase. Do not batch rename the entire repo. Do not delete raw data.
Do not run heavy training. Do not commit, push, or create a branch unless the
user explicitly asks.

For every code move:
1. identify all callers;
2. add or update targeted tests;
3. keep compatibility shims where needed;
4. run the phase verification commands;
5. report files changed, commands run, validation results, and unresolved issues.

The target architecture is:
- src/intraday_research/ for canonical logic;
- configs/ for stage parameters and pipeline order;
- scripts/notebooks/ for notebook generators;
- notebooks/ for thin generated Colab notebooks;
- tests/contracts, tests/notebooks, tests/stages for verification;
- results/ and artifacts/ for outputs and manifests.

Colab notebooks may import intraday_research only after installing from an exact
git commit, and each stage run must write a manifest containing repo URL, commit,
config hash, notebook hash, input artifacts, output artifacts, validation scope,
and holdout_test_contact.
```

## Phase 0: Migration Inventory And Safety Baseline

Goal: know what exists before moving anything.

- [ ] Run status and diff inventory:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --stat
```

- [ ] Build a file inventory for active source surfaces:

```powershell
rg --files docs scripts notebooks tests
```

- [ ] Create or update `artifacts/code_management/migration_inventory.csv` with
  these columns:

```text
path,current_role,target_role,stage,tracked_state,commit_policy,next_action,notes
```

- [ ] Classify files into:

```text
canonical_source
generated_notebook
run_copy
contract_test
static_gate
design_doc
runtime_output
legacy_or_backup
```

- [ ] Confirm no planned move deletes raw data or Drive-only artifacts.

Verification:

```powershell
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short
```

Acceptance criteria:

- inventory exists;
- no code moved;
- raw data untouched;
- current dirty/untracked state is documented.

## Phase 1: Package Scaffold

Goal: introduce `src/intraday_research/` without changing behavior.

- [ ] Update `pyproject.toml` for `src` layout if needed:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "intraday-stock-direction-research"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] Add package files:

```text
src/intraday_research/__init__.py
src/intraday_research/config.py
src/intraday_research/contracts/__init__.py
src/intraday_research/stages/__init__.py
```

- [ ] Add a package import smoke test:

```text
tests/stages/test_package_import.py
```

Required assertions:

```python
import intraday_research

def test_package_imports():
    assert hasattr(intraday_research, "__version__")
```

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile src\intraday_research\__init__.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\stages\test_package_import.py -q
```

Acceptance criteria:

- package imports locally;
- no notebook behavior changes;
- no stage logic moved yet.

## Phase 2: Contract Helper Migration

Goal: move stable validators first because they are low-risk and testable.

- [ ] Move current contract helpers:

```text
scripts/notebook06_contract.py -> src/intraday_research/contracts/selective_no_trade_calibration.py
scripts/notebook07_contract.py -> src/intraday_research/contracts/validation_synthesis_gap_audit.py
scripts/notebook08_contract.py -> src/intraday_research/contracts/deep_sequence_exploration.py
```

- [ ] Keep temporary compatibility shims in `scripts/`:

```python
from intraday_research.contracts.validation_synthesis_gap_audit import *
```

- [ ] Update tests to import the package path directly:

```python
from intraday_research.contracts import validation_synthesis_gap_audit as c
```

- [ ] Move or add contract tests under `tests/contracts/`.

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\contracts -q -rs
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_notebook06_artifact_contract.py tests\test_notebook07_artifact_contract.py tests\test_notebook08_artifact_contract.py -q -rs
```

Acceptance criteria:

- contract behavior unchanged;
- old imports still work through shims during migration;
- no generator or notebook logic changed.

## Phase 3: Config And Pipeline Registry

Goal: move stage order and fixed parameters out of file names.

- [ ] Add:

```text
configs/base.yaml
configs/data.yaml
configs/pipeline.yaml
configs/validation_rules.yaml
configs/stages/config_screening.yaml
configs/stages/model_family_screening.yaml
configs/stages/controlled_followup.yaml
configs/stages/lightgbm_tuning.yaml
configs/stages/selective_no_trade_calibration.yaml
configs/stages/validation_synthesis_gap_audit.yaml
configs/stages/deep_sequence_exploration.yaml
```

- [ ] Add `src/intraday_research/config.py` helpers:

```python
def load_yaml(path): ...
def load_stage_config(stage_name): ...
def load_pipeline_registry(): ...
def sha256_file(path): ...
```

- [ ] Add `tests/stages/test_pipeline_registry.py`:

Required checks:

```text
pipeline has seven active stages
stage names are unique
stage configs exist
notebook paths are semantic, not numbered
result dirs match stage names
```

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\stages\test_pipeline_registry.py -q
```

Acceptance criteria:

- stage order is represented in `configs/pipeline.yaml`;
- numbered filenames are no longer the only ordering source;
- no files renamed yet.

## Phase 4: Stage Entrypoints

Goal: expose tested `run_stage(config)` functions before thinning notebooks.

Migrate one stage at a time in dependency order:

```text
config_screening
model_family_screening
controlled_followup
lightgbm_tuning
selective_no_trade_calibration
validation_synthesis_gap_audit
deep_sequence_exploration
```

For each stage:

- [ ] Add `src/intraday_research/stages/<stage_name>.py`.
- [ ] Expose:

```python
def run_stage(config):
    ...
```

- [ ] Move only stable behavior first:

```text
manifest handling
artifact schema validation
path/config resolution
metric aggregation
ledger append/check helpers
```

- [ ] Do not move heavy training loops until a targeted test exists.
- [ ] Add `tests/stages/test_<stage_name>.py`.
- [ ] Confirm no holdout/test path is introduced.

Verification for each stage:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\stages\test_<stage_name>.py -q -rs
```

Acceptance criteria:

- each migrated stage has a callable `run_stage(config)`;
- targeted tests cover the moved behavior;
- generator/notebook behavior remains unchanged until Phase 6.

## Phase 5: Notebook Generator Relocation

Goal: move generators under a clearer namespace before changing notebook bodies.

- [ ] Move generator files:

```text
scripts/create_config_screening_colab_notebook.py -> scripts/notebooks/generate_config_screening_colab.py
scripts/create_model_family_screening_colab_notebook.py -> scripts/notebooks/generate_model_family_screening_colab.py
scripts/create_controlled_followup_colab_notebook.py -> scripts/notebooks/generate_controlled_followup_colab.py
scripts/create_lightgbm_tuning_colab_notebook.py -> scripts/notebooks/generate_lightgbm_tuning_colab.py
scripts/create_selective_no_trade_calibration_colab_notebook.py -> scripts/notebooks/generate_selective_no_trade_calibration_colab.py
scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py -> scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py
scripts/create_deep_sequence_exploration_colab_notebook.py -> scripts/notebooks/generate_deep_sequence_exploration_colab.py
```

- [ ] Keep temporary compatibility shims at old paths that call the new
  generator modules.
- [ ] Update tests/static gates to resolve generators through
  `configs/pipeline.yaml`.

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\notebooks -q -rs
```

Acceptance criteria:

- old generator commands still work through shims;
- new generator paths work;
- generated notebook structure has not changed yet.

## Phase 6: Thin Notebook Conversion

Goal: convert generated notebooks from code-heavy to package-backed.

For each notebook:

- [ ] Keep the research question, protocol summary, config display, and result
  display cells.
- [ ] Replace embedded reusable logic with:

```python
%pip install -q "git+https://github.com/<user>/intraday_stock_direction_research.git@<commit_sha>"

from intraday_research.config import load_stage_config
from intraday_research.stages.<stage_name> import run_stage

config = load_stage_config("<stage_name>")
result = run_stage(config)
```

- [ ] During local pre-GitHub migration, allow a local development mode only in
  generated run copies, not in canonical GitHub notebooks.
- [ ] Add manifest-writing checks to every stage invocation.
- [ ] Keep committed notebook outputs empty.

Static gate requirements:

```text
exact commit install is present for GitHub/Colab release notebooks
stage import path matches configs/pipeline.yaml
run_stage(config) is called
run manifest path is defined
outputs are empty
execution_count is null
no holdout/test reads
```

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\notebooks -q -rs
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\contracts -q -rs
```

Acceptance criteria:

- notebook bodies are thin;
- stage logic lives in `src/intraday_research/`;
- static gates verify package pinning and manifest contract.

## Phase 7: Semantic Rename

Goal: remove numbered stage names after registry-based lookup works.

- [ ] Rename canonical notebooks:

```text
notebooks/02_config_screening_colab.ipynb -> notebooks/config_screening_colab.ipynb
notebooks/03_model_family_screening_colab.ipynb -> notebooks/model_family_screening_colab.ipynb
notebooks/04_controlled_followup_colab.ipynb -> notebooks/controlled_followup_colab.ipynb
notebooks/05_lightgbm_tuning_colab.ipynb -> notebooks/lightgbm_tuning_colab.ipynb
notebooks/06_selective_no_trade_calibration_colab.ipynb -> notebooks/selective_no_trade_calibration_colab.ipynb
notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb -> notebooks/validation_synthesis_gap_audit_colab.ipynb
notebooks/08_deep_sequence_exploration_colab.ipynb -> notebooks/deep_sequence_exploration_colab.ipynb
```

- [ ] Move tests into:

```text
tests/contracts/
tests/notebooks/
tests/stages/
```

- [ ] Add `docs/LEGACY_NAME_MAPPING.md` with old-to-new paths.

Verification:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\contracts tests\notebooks tests\stages -q -rs
```

Acceptance criteria:

- no test relies on numeric filename order;
- old names are documented;
- Drive/Colab references can be mapped to new names.

## Phase 8: GitHub Hygiene

Goal: make the repository safe to publish.

- [ ] Update `.gitignore` to exclude:

```text
data/raw/
data/interim/
data/processed/
results/**/*
models/**/*
*.pt
*.pkl
*.joblib
*.parquet
*.feather
*.npy
*.npz
notebooks/*_run_all.ipynb
notebooks/*_full_run.ipynb
notebooks/*.bak*
.ipynb_checkpoints/
```

- [ ] Add `data/README.md` explaining where raw data lives and how to provide it.
- [ ] Add `docs/REPRODUCIBILITY.md` covering:

```text
local install
local test commands
Colab exact-commit install
data manifest requirements
stage execution order
expected outputs
known non-determinism
```

- [ ] Update `README.md` to show:

```text
research question
project structure
quick local install
Colab execution path
no-holdout/test policy
```

Verification:

```powershell
rg -n "E:\\\\|C:\\\\|holdout=True|train_test_split|from intraday_research" README.md docs notebooks scripts src tests
git -C E:\codex_workspace\projects\intraday_stock_direction_research status --short
```

Acceptance criteria:

- raw data and large outputs are not staged;
- README describes package-first workflow;
- reproducibility docs describe Colab exact-commit path.

## Phase 9: Colab Exact-Commit Smoke

Goal: prove GitHub package code can run from Colab.

This phase requires a GitHub remote and an actual commit SHA.

- [ ] After user authorizes commit/push, install from exact commit in a fresh
  environment:

```python
%pip install -q "git+https://github.com/<user>/intraday_stock_direction_research.git@<commit_sha>"
import intraday_research
```

- [ ] Run one low-cost stage smoke that does not train and does not touch
  holdout/test.
- [ ] Verify the run manifest contains the expected commit and
  `holdout_test_contact=false`.

Acceptance criteria:

- Colab install succeeds from exact commit;
- package imports;
- smoke run writes manifest;
- no heavy training required.

## Phase 10: Final Verification

Run the broad local suite:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\contracts tests\notebooks tests\stages -q -rs
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests -q --ignore=tests\test_validation_pipeline.py -rs
```

Known pre-existing blocker to resolve separately:

```text
tests/test_validation_pipeline.py may fail collection if
scripts.run_validation_only_pipeline_smoke is still missing.
```

Final acceptance criteria:

- package imports locally;
- stage registry tests pass;
- contract tests pass;
- notebook static gates pass;
- artifact contracts pass;
- generated notebooks have empty outputs;
- no raw data or large runtime outputs are staged;
- Colab exact-commit install path is documented;
- migration mapping exists.

## Stop Conditions

Stop and report before continuing if any of these occur:

- any code path reads, transforms, scores, or summarizes holdout/test data;
- a stage output changes without an explicit protocol/design update;
- a notebook can run only through a machine-local path;
- package-backed Colab install cannot be pinned to an exact commit;
- tests require heavy training to prove basic import or contract behavior;
- raw data or large output artifacts appear staged for GitHub.

