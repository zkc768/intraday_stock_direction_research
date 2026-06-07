# Code Organization

This note defines a simpler, GitHub-facing structure for the intraday stock
direction research project. It does not move files by itself. It is the naming
and directory target for a later, staged cleanup.

## Goal

The project should look like a reproducible research codebase, not a temporary
sequence of numbered Colab notebooks.

The selected target is **package-first**:

```text
protocol and design -> src package -> stage configs -> generated thin notebooks
-> tests -> results and thesis artifacts
```

Notebook order should be documented in `configs/pipeline.yaml`, not encoded in
file names such as `02_`, `03_`, or `notebook07_contract.py`.

`src/intraday_research/` is the only long-term source of research logic.
Generated Colab notebooks are execution/reporting interfaces: they install the
package from an exact git commit, load a stage config, call a tested stage entry
point, write a run manifest, and display results.

Self-contained notebooks are allowed only as optional archival snapshots or
review packets. They are not the default source of truth after the package-first
migration begins.

## References

The structure below combines patterns from these public repositories:

| Reference | Link | Pattern to borrow | Do not copy |
|---|---|---|---|
| Cookiecutter Data Science | https://github.com/drivendataorg/cookiecutter-data-science | Standard data science folders: `data`, `docs`, `models`, `notebooks`, `reports`, source module | Do not blindly run the template over this existing repo |
| Cookiecutter Data Science docs | https://cookiecutter-data-science.drivendata.org/ | Clear role comments for `raw`, `interim`, `processed`, `reports/figures`, source module | Its notebook naming includes personal initials; that is not useful here |
| Controlled financial forecasting example | https://github.com/NabeelAhmad9/compare_forecasting_models | Research-paper structure: `configs`, `data`, `models`, `results`, `notebooks`, `paper`, `scripts`, `src`, `Makefile`, reproducibility guide | Low-star example; use as shape reference, not authority |
| Microsoft Qlib | https://github.com/microsoft/qlib | Mature quant-research separation: `docs`, `examples`, core package, `scripts`, `tests` | Too large and platform-oriented for this thesis project |
| Microsoft Recommenders | https://github.com/recommenders-team/recommenders | Notebook examples plus reusable utilities, evaluation helpers, and documentation | Recommendation-system task does not match this project |
| THUML Time-Series-Library | https://github.com/thuml/Time-Series-Library | Time-series research structure: data provider, experiment layer, models, scripts, utilities, tutorials | Benchmark-library architecture is heavier than needed |
| Kedro | https://github.com/kedro-org/kedro | Reproducible, modular data science pipeline principles and data catalog thinking | Do not migrate to Kedro unless the project later needs a full pipeline framework |

## Recommended Repository Shape

Use semantic names. Keep the active research stage order in
`configs/pipeline.yaml`.

```text
intraday_stock_direction_research/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── Makefile                         # Optional local convenience commands
├── configs/
│   ├── base.yaml                    # Common paths, environment flags, scope
│   ├── data.yaml                    # Raw-data manifest and ticker universe
│   ├── pipeline.yaml                # Ordered stage registry
│   ├── validation_rules.yaml        # Pre-registered wording and audit rules
│   └── stages/
│       ├── config_screening.yaml
│       ├── model_family_screening.yaml
│       ├── controlled_followup.yaml
│       ├── lightgbm_tuning.yaml
│       ├── selective_no_trade_calibration.yaml
│       ├── validation_synthesis_gap_audit.yaml
│       └── deep_sequence_exploration.yaml
├── data/
│   ├── README.md                    # Explains where raw data lives
│   ├── raw/                         # Local-only, gitignored
│   ├── interim/                     # Local-only intermediate outputs
│   └── processed/                   # Local-only generated datasets
├── docs/
│   ├── CODE_ORGANIZATION.md
│   ├── REPRODUCIBILITY.md
│   ├── research_workflow.md
│   ├── configuration_screening_freeze.md
│   ├── protocols/
│   └── technical_designs/
├── references/
│   ├── literature/
│   └── external_repos/
├── src/
│   └── intraday_research/
│       ├── data/
│       ├── features/
│       ├── labels/
│       ├── splits/
│       ├── windows/
│       ├── evaluation/
│       ├── contracts/
│       └── stages/
├── scripts/
│   ├── notebooks/
│   ├── run_stage.py
│   ├── export_artifacts.py
│   └── build_reproducibility_manifest.py
├── notebooks/
│   ├── config_screening_colab.ipynb
│   ├── model_family_screening_colab.ipynb
│   ├── controlled_followup_colab.ipynb
│   ├── lightgbm_tuning_colab.ipynb
│   ├── selective_no_trade_calibration_colab.ipynb
│   ├── validation_synthesis_gap_audit_colab.ipynb
│   └── deep_sequence_exploration_colab.ipynb
├── tests/
│   ├── contracts/
│   ├── notebooks/
│   └── stages/
├── results/
│   ├── config_screening/
│   ├── model_family_screening/
│   ├── controlled_followup/
│   ├── lightgbm_tuning/
│   ├── selective_no_trade_calibration/
│   ├── validation_synthesis_gap_audit/
│   └── deep_sequence_exploration/
├── artifacts/
│   ├── run_manifests/
│   ├── code_management/
│   └── review_packets/
├── reports/
│   ├── figures/
│   └── thesis_tables/
└── paper/
    ├── figures/
    ├── tables/
    └── sections/
```

## Package-First Execution Contract

Every stage should expose a stable local entry point:

```python
from intraday_research.stages.validation_synthesis_gap_audit import run_stage

result = run_stage(config)
```

Every generated Colab notebook should use the same pattern:

```python
%pip install -q "git+https://github.com/<user>/intraday_stock_direction_research.git@<commit_sha>"

from intraday_research.config import load_config
from intraday_research.stages.validation_synthesis_gap_audit import run_stage

EXPECTED_REPO_COMMIT = "<commit_sha>"
EXPECTED_STAGE = "validation_synthesis_gap_audit"

config = load_config("configs/stages/validation_synthesis_gap_audit.yaml")
result = run_stage(config)
```

Installing from a floating branch such as `main` is not an acceptable research
path for Colab runs. The commit must be exact and recorded in the run manifest.

Each stage run must write a machine-readable manifest:

```json
{
  "repo_url": "https://github.com/<user>/intraday_stock_direction_research",
  "git_commit": "<commit_sha>",
  "package_version": null,
  "stage": "validation_synthesis_gap_audit",
  "config_sha256": "<sha256>",
  "notebook_sha256": "<sha256>",
  "input_artifacts": [],
  "output_artifacts": [],
  "validation_scope": "validation_only",
  "holdout_test_contact": false
}
```

## Naming Rules

Prefer short semantic names:

| Current style | Target style | Reason |
|---|---|---|
| `02_config_screening_colab.ipynb` | `config_screening_colab.ipynb` | Stage order belongs in `pipeline.yaml` |
| `07_validation_synthesis_and_gap_audit_colab.ipynb` | `validation_synthesis_gap_audit_colab.ipynb` | Remove filler words such as `and` |
| `scripts/notebook07_contract.py` | `src/intraday_research/contracts/validation_synthesis_gap_audit.py` | Contract is package logic, not a loose script |
| `scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py` | `scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py` | Verb is shorter and generator location is explicit |
| `NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md` | `docs/technical_designs/deep_sequence_exploration.md` | Date can move to document metadata or changelog |

Use these conventions:

- Directories: lowercase snake_case.
- Python files: lowercase snake_case.
- Notebooks: semantic name plus `_colab.ipynb` when they are Colab-facing.
- Config files: one stage per YAML under `configs/stages/`.
- Results directories: same stage name as `configs/pipeline.yaml`.
- Avoid notebook numbers in filenames unless a document is explicitly a teaching
  tutorial where visual ordering matters.

## Stage Registry

Use `configs/pipeline.yaml` as the stable ordering source:

```yaml
stages:
  - name: config_screening
    config: configs/stages/config_screening.yaml
    generator: scripts/notebooks/generate_config_screening_colab.py
    notebook: notebooks/config_screening_colab.ipynb
    results: results/config_screening/

  - name: model_family_screening
    config: configs/stages/model_family_screening.yaml
    generator: scripts/notebooks/generate_model_family_screening_colab.py
    notebook: notebooks/model_family_screening_colab.ipynb
    results: results/model_family_screening/

  - name: controlled_followup
    config: configs/stages/controlled_followup.yaml
    generator: scripts/notebooks/generate_controlled_followup_colab.py
    notebook: notebooks/controlled_followup_colab.ipynb
    results: results/controlled_followup/

  - name: lightgbm_tuning
    config: configs/stages/lightgbm_tuning.yaml
    generator: scripts/notebooks/generate_lightgbm_tuning_colab.py
    notebook: notebooks/lightgbm_tuning_colab.ipynb
    results: results/lightgbm_tuning/

  - name: selective_no_trade_calibration
    config: configs/stages/selective_no_trade_calibration.yaml
    generator: scripts/notebooks/generate_selective_no_trade_calibration_colab.py
    notebook: notebooks/selective_no_trade_calibration_colab.ipynb
    results: results/selective_no_trade_calibration/

  - name: validation_synthesis_gap_audit
    config: configs/stages/validation_synthesis_gap_audit.yaml
    generator: scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py
    notebook: notebooks/validation_synthesis_gap_audit_colab.ipynb
    results: results/validation_synthesis_gap_audit/

  - name: deep_sequence_exploration
    config: configs/stages/deep_sequence_exploration.yaml
    generator: scripts/notebooks/generate_deep_sequence_exploration_colab.py
    notebook: notebooks/deep_sequence_exploration_colab.ipynb
    results: results/deep_sequence_exploration/
```

## Source Code Layout

Use `src/intraday_research/` for local reusable logic:

```text
src/intraday_research/
├── data/
│   ├── manifest.py
│   └── loaders.py
├── features/
│   └── stationarity.py
├── labels/
│   └── no_trade_band.py
├── splits/
│   └── chronological.py
├── windows/
│   └── rolling_windows.py
├── evaluation/
│   ├── metrics.py
│   ├── dummy_baselines.py
│   └── aggregation.py
├── contracts/
│   ├── selective_no_trade_calibration.py
│   ├── validation_synthesis_gap_audit.py
│   └── deep_sequence_exploration.py
└── stages/
    ├── config_screening.py
    ├── model_family_screening.py
    ├── controlled_followup.py
    ├── lightgbm_tuning.py
    ├── selective_no_trade_calibration.py
    ├── validation_synthesis_gap_audit.py
    └── deep_sequence_exploration.py
```

This package is the canonical code source. Local tests, local runners, and
generated Colab notebooks all call it. Colab usage must follow `AGENTS.md`
section 7.2: install from an exact git commit, call a tested stage entry point,
and write a run manifest.

## Code Extraction Rules

Extraction is allowed only when it improves clarity, testability, or reuse
without changing the research behavior.

### Allowed Extraction Targets

These are good candidates for `src/intraday_research/`:

- schema validators and artifact contracts;
- data manifests, path resolution, and artifact inventory helpers;
- feature construction pure functions;
- label construction and boundary invalidation helpers;
- chronological split helpers;
- per-ticker window construction helpers;
- train-only preprocessing helpers;
- dummy baselines and metric calculations;
- aggregation, confidence interval, and diagnostic summary helpers;
- validation-budget ledger helpers;
- stage orchestration that has an explicit `run_stage(config)` interface.

### Avoid Extracting

Do not extract these into shared package logic unless a design note first defines
the contract:

- exploratory plotting cells;
- one-off debug or inspection code;
- Colab UI / drive backup switch cells;
- current-run result wording or thesis prose;
- run-copy-only switches;
- post-validation threshold, model, or wording decisions;
- heavy training orchestration before tests and config contracts exist;
- any code that touches holdout/test.

### Extraction Gate

Before extracting code:

1. Identify every caller: generator, notebook, test, local runner, and artifact
   contract.
2. Classify the code as pure helper, side-effecting IO, contract validator, or
   stage orchestration.
3. Confirm the change does not alter research behavior.
4. Confirm no holdout/test path, metric, transformation, or summary is added.

During extraction:

1. Use explicit inputs and outputs.
2. Avoid hidden global state.
3. Avoid machine-local paths.
4. Do not catch and ignore exceptions.
5. Preserve column names, artifact schemas, and decision timing.

After extraction:

1. Add or update targeted unit/contract tests.
2. Update the affected stage config if parameters moved out of notebooks.
3. Update the generator and regenerate the notebook if notebook behavior changed.
4. Run the affected notebook static gate and artifact contract tests.
5. Update protocol/design docs if the research contract changed.

### Behavior Equivalence

A refactor is acceptable only if it is behavior-preserving or intentionally
changes behavior under an explicit protocol/design update.

For behavior-preserving extraction, verify at least one of:

- exact schema equality for produced artifacts;
- exact column set and column order equality for CSV outputs;
- exact validator pass/fail behavior for contract helpers;
- identical run manifest fields except expected code-location metadata;
- targeted tests proving chronology, train-only preprocessing, label-boundary,
  no-holdout, and ledger-before-read rules are unchanged.

## Colab Compatibility Rules

Package-first does not mean Colab can rely on local state. A Colab notebook is
acceptable only when it can run from GitHub plus documented external data
artifacts.

Colab notebooks must:

- install `intraday_research` from an exact git commit or record the same exact
  commit when running from an uploaded package archive;
- avoid `E:\`, `C:\`, `/Users/...`, or other machine-local paths;
- load data through config, Drive file IDs, manifests, or documented local
  upload paths;
- fail clearly when required data artifacts are missing;
- write a run manifest;
- keep committed outputs empty unless the file is explicitly a run-copy artifact.

Colab notebooks must not:

- install from floating `main`;
- silently install unstated dependencies;
- mount or scan arbitrary Drive folders as the authoritative data source;
- read, transform, summarize, or score holdout/test data in validation-only
  stages.

## Notebook Generator Layout

Use `scripts/notebooks/` for notebook generation:

```text
scripts/notebooks/
├── generate_config_screening_colab.py
├── generate_model_family_screening_colab.py
├── generate_controlled_followup_colab.py
├── generate_lightgbm_tuning_colab.py
├── generate_selective_no_trade_calibration_colab.py
├── generate_validation_synthesis_gap_audit_colab.py
└── generate_deep_sequence_exploration_colab.py
```

Generators remain the source of notebook structure. Do not hand-edit generated
notebooks as the only copy of active logic.

## Tests

Split tests by purpose:

```text
tests/
├── contracts/
│   ├── test_selective_no_trade_calibration_contract.py
│   ├── test_validation_synthesis_gap_audit_contract.py
│   └── test_deep_sequence_exploration_contract.py
├── notebooks/
│   ├── test_config_screening_notebook.py
│   ├── test_model_family_screening_notebook.py
│   ├── test_controlled_followup_notebook.py
│   ├── test_lightgbm_tuning_notebook.py
│   ├── test_selective_no_trade_calibration_notebook.py
│   ├── test_validation_synthesis_gap_audit_notebook.py
│   └── test_deep_sequence_exploration_notebook.py
└── stages/
    ├── test_splits.py
    ├── test_labels.py
    ├── test_metrics.py
    └── test_stage_registry.py
```

Static gates should verify:

- notebook parses and code cells AST-parse;
- generated notebooks have empty outputs and no execution counts;
- heavy `RUN_*` gates default to false unless a file is explicitly a run copy;
- holdout/test is not read, scored, summarized, or transformed;
- chronological split and train-only preprocessing rules are preserved;
- package-backed notebooks pin an exact git commit;
- expected stage entry points are imported and called;
- run manifests are written and include commit/config/notebook hashes;
- package code used by notebooks has targeted tests.

## Results And Artifacts

Use `results/` for reproducible stage outputs and `artifacts/` for supporting
project-management or review artifacts.

```text
results/<stage_name>/
├── tables/
├── figures/
├── manifests/
└── diagnostics/

artifacts/
├── run_manifests/
├── code_management/
└── review_packets/
```

Use `reports/` and `paper/` for thesis-facing materials:

```text
reports/
├── figures/
└── thesis_tables/

paper/
├── figures/
├── tables/
└── sections/
```

## Migration Strategy

Do not rename everything in one change. Use small, reversible steps:

1. Add `configs/pipeline.yaml` and this naming document.
2. Create `src/intraday_research/` with package metadata in `pyproject.toml`.
3. Move contract helpers into `src/intraday_research/contracts/`.
4. Update tests to import contracts from the new package path.
5. Move pure shared helpers into `src/intraday_research/` behind targeted tests.
6. Add stage entry points under `src/intraday_research/stages/`.
7. Move notebook generators into `scripts/notebooks/`.
8. Convert generated notebooks into thin package-backed Colab interfaces.
9. Rename canonical notebooks after tests can find them through
   `configs/pipeline.yaml`.
10. Move results into `results/<stage_name>/` and keep Drive IDs in run manifests.
11. Add `docs/REPRODUCIBILITY.md` with exact local and Colab reproduction steps.
12. Keep a temporary `docs/LEGACY_NAME_MAPPING.md` until old Drive / Colab /
   commit references are no longer needed.

At each step, run targeted static gates before broad tests.

## Non-Goals

- Do not reopen holdout/test or change research claims as part of cleanup.
- Do not run heavy training just to validate file movement.
- Do not delete raw data.
- Do not adopt Kedro, MLflow, DVC, or nbdev unless a separate decision says the
  added dependency is worth it.
- Do not turn a naming cleanup into a new experiment.
- Do not keep notebook bodies as the canonical long-term source of research
  logic once package-first migration starts.
