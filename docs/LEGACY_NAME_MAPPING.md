# Legacy Name Mapping

This document preserves the mapping from the current numbered / notebook-first
layout to the planned package-first GitHub layout. It is a reference document
only. It does not authorize moving files by itself.

Use this mapping when updating tests, generators, Colab links, Drive manifests,
README examples, run manifests, and historical notes.

## Naming Principles

- Stage order moves to `configs/pipeline.yaml`.
- File names become semantic and stage-oriented.
- Canonical logic moves to `src/intraday_research/`.
- Generated notebooks become thin Colab execution/reporting interfaces.
- Old numbered names remain valid historical references until all active links
  are migrated.

## Stage Name Registry

| Legacy stage label | New stage name | Scope |
|---|---|---|
| `02` | `config_screening` | Stage 0 configuration screening |
| `03` | `model_family_screening` | Model-family validation screening |
| `04` | `controlled_followup` | Controlled validation-only follow-up |
| `05` | `lightgbm_tuning` | Train-inner LightGBM tuning and validation confirmation |
| `06` | `selective_no_trade_calibration` | Selective / no-trade calibration diagnostics |
| `07` | `validation_synthesis_gap_audit` | Validation synthesis and gap audit |
| `08` | `deep_sequence_exploration` | Deep sequence exploration and freeze/readout |

## Canonical Notebook Mapping

| Current path | Target path | Policy |
|---|---|---|
| `notebooks/02_config_screening_colab.ipynb` | `notebooks/config_screening_colab.ipynb` | Rename after `configs/pipeline.yaml` is active |
| `notebooks/03_model_family_screening_colab.ipynb` | `notebooks/model_family_screening_colab.ipynb` | Rename after static gates use registry |
| `notebooks/04_controlled_followup_colab.ipynb` | `notebooks/controlled_followup_colab.ipynb` | Rename after tests use registry |
| `notebooks/05_lightgbm_tuning_colab.ipynb` | `notebooks/lightgbm_tuning_colab.ipynb` | Rename after generator relocation |
| `notebooks/06_selective_no_trade_calibration_colab.ipynb` | `notebooks/selective_no_trade_calibration_colab.ipynb` | Rename after contract migration |
| `notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb` | `notebooks/validation_synthesis_gap_audit_colab.ipynb` | Rename after N07 package entrypoint exists |
| `notebooks/08_deep_sequence_exploration_colab.ipynb` | `notebooks/deep_sequence_exploration_colab.ipynb` | RENAMED 2026-06-06 (Phase 7); substantive work goes into src/intraday_research/stages/ + models/deep_sequence/ |

## Non-Canonical Notebook Files

These files should not become canonical GitHub notebooks without explicit review.

| Current path | Target handling | Reason |
|---|---|---|
| `notebooks/02_diagnostic_h0_tabular_sweep_colab.ipynb` | `notebooks/diagnostic_h0_tabular_sweep_colab.ipynb` or archive | Diagnostic lane, not part of final 02-08 canonical route unless re-approved |
| `notebooks/04_ian_research_memo.ipynb` | `artifacts/review_packets/` or archive | Research memo, not an executable stage notebook |
| `notebooks/05_lightgbm_tuning_colab_run_all.ipynb` | `artifacts/run_manifests/` or ignored run copy | Run-copy artifact, not canonical source |
| `notebooks/05_06_chained_validation_colab.ipynb` | archive or delete only with user approval | Chained run helper, not package-first stage source |
| `notebooks/05_06_chained_validation_colab_resume06.ipynb` | archive or delete only with user approval | Resume helper, not canonical source |
| `notebooks/07_validation_synthesis_and_gap_audit_colab_drive_core_full_run.ipynb` | ignored run copy or `artifacts/review_packets/` | Drive run-copy artifact |
| `notebooks/07_validation_synthesis_and_gap_audit_colab_drive_core_full_run.ipynb.bak_keyerror_ticker` | archive or remove only with user approval | Debug backup from a fixed Colab issue |

## Generator Mapping

| Current path | Target path | Policy |
|---|---|---|
| `scripts/create_config_screening_colab_notebook.py` | `scripts/notebooks/generate_config_screening_colab.py` | Move with temporary compatibility shim |
| `scripts/create_model_family_screening_colab_notebook.py` | `scripts/notebooks/generate_model_family_screening_colab.py` | Move with temporary compatibility shim |
| `scripts/create_controlled_followup_colab_notebook.py` | `scripts/notebooks/generate_controlled_followup_colab.py` | Move with temporary compatibility shim |
| `scripts/create_lightgbm_tuning_colab_notebook.py` | `scripts/notebooks/generate_lightgbm_tuning_colab.py` | Move with temporary compatibility shim |
| `scripts/create_selective_no_trade_calibration_colab_notebook.py` | `scripts/notebooks/generate_selective_no_trade_calibration_colab.py` | Move with temporary compatibility shim |
| `scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py` | `scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py` | Move with temporary compatibility shim |
| `scripts/create_deep_sequence_exploration_colab_notebook.py` | `scripts/notebooks/generate_deep_sequence_exploration_colab.py` | Move with temporary compatibility shim |

## Contract Helper Mapping

| Current path | Target path | Policy |
|---|---|---|
| `scripts/notebook06_contract.py` | `src/intraday_research/contracts/selective_no_trade_calibration.py` | Move first; keep shim during migration |
| `scripts/notebook07_contract.py` | `src/intraday_research/contracts/validation_synthesis_gap_audit.py` | Move first; keep shim during migration |
| `scripts/notebook08_contract.py` | `src/intraday_research/contracts/deep_sequence_exploration.py` | Move first; keep shim during migration |

## Stage Entrypoint Mapping

These files do not all exist yet. They are the package-first targets.

| Stage name | Target source file | Public entrypoint |
|---|---|---|
| `config_screening` | `src/intraday_research/stages/config_screening.py` | `run_stage(config)` |
| `model_family_screening` | `src/intraday_research/stages/model_family_screening.py` | `run_stage(config)` |
| `controlled_followup` | `src/intraday_research/stages/controlled_followup.py` | `run_stage(config)` |
| `lightgbm_tuning` | `src/intraday_research/stages/lightgbm_tuning.py` | `run_stage(config)` |
| `selective_no_trade_calibration` | `src/intraday_research/stages/selective_no_trade_calibration.py` | `run_stage(config)` |
| `validation_synthesis_gap_audit` | `src/intraday_research/stages/validation_synthesis_gap_audit.py` | `run_stage(config)` |
| `deep_sequence_exploration` | `src/intraday_research/stages/deep_sequence_exploration.py` | `run_stage(config)` |

## Test Mapping

| Current path | Target path | Policy |
|---|---|---|
| `tests/test_notebook03_static_gate.py` | `tests/notebooks/test_model_family_screening_notebook.py` | Rename after registry lookup works |
| `tests/test_notebook04_static_gate.py` | `tests/notebooks/test_controlled_followup_notebook.py` | Rename after registry lookup works |
| `tests/test_notebook05_static_gate.py` | `tests/notebooks/test_lightgbm_tuning_notebook.py` | Rename after registry lookup works |
| `tests/test_notebook06_static_gate.py` | `tests/notebooks/test_selective_no_trade_calibration_notebook.py` | Rename after registry lookup works |
| `tests/test_notebook06_artifact_contract.py` | `tests/contracts/test_selective_no_trade_calibration_contract.py` | Move after package contract import works |
| `tests/test_notebook07_static_gate.py` | `tests/notebooks/test_validation_synthesis_gap_audit_notebook.py` | Rename after N07 stage entrypoint exists |
| `tests/test_notebook07_artifact_contract.py` | `tests/contracts/test_validation_synthesis_gap_audit_contract.py` | Move after package contract import works |
| `tests/test_notebook08_static_gate.py` | `tests/notebooks/test_deep_sequence_exploration_notebook.py` | Rename after N08 stage entrypoint exists |
| `tests/test_notebook08_artifact_contract.py` | `tests/contracts/test_deep_sequence_exploration_contract.py` | Move after package contract import works |

## Documentation Mapping

| Current path | Target path | Policy |
|---|---|---|
| `docs/RESEARCH_WORKFLOW.md` | `docs/research_workflow.md` | Rename only after README links are updated |
| `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` | `docs/configuration_screening_freeze.md` | Preserve date in document metadata |
| `docs/MODEL_FAMILY_SCREENING_PROTOCOL_2026-06-04.md` | `docs/protocols/model_family_screening.md` | Preserve date in document metadata |
| `docs/CONTROLLED_FOLLOWUP_PROTOCOL_2026-06-04.md` | `docs/protocols/controlled_followup.md` | Preserve date in document metadata |
| `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md` | `docs/protocols/lightgbm_tuning.md` | Preserve date in document metadata |
| `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_CALIBRATION_PROTOCOL_2026-06-05.md` | `docs/protocols/selective_no_trade_calibration.md` | Preserve date in document metadata |
| `docs/NOTEBOOK06_SELECTIVE_NO_TRADE_TECHNICAL_DESIGN_2026-06-05.md` | `docs/technical_designs/selective_no_trade_calibration.md` | Preserve date in document metadata |
| `docs/NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md` | `docs/technical_designs/validation_synthesis_gap_audit.md` | Preserve date in document metadata |
| `docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md` | `docs/technical_designs/deep_sequence_exploration.md` | Preserve freeze/readout notes in metadata |

## Config And Results Mapping

| Stage name | Config target | Results target |
|---|---|---|
| `config_screening` | `configs/stages/config_screening.yaml` | `results/config_screening/` |
| `model_family_screening` | `configs/stages/model_family_screening.yaml` | `results/model_family_screening/` |
| `controlled_followup` | `configs/stages/controlled_followup.yaml` | `results/controlled_followup/` |
| `lightgbm_tuning` | `configs/stages/lightgbm_tuning.yaml` | `results/lightgbm_tuning/` |
| `selective_no_trade_calibration` | `configs/stages/selective_no_trade_calibration.yaml` | `results/selective_no_trade_calibration/` |
| `validation_synthesis_gap_audit` | `configs/stages/validation_synthesis_gap_audit.yaml` | `results/validation_synthesis_gap_audit/` |
| `deep_sequence_exploration` | `configs/stages/deep_sequence_exploration.yaml` | `results/deep_sequence_exploration/` |

## Historical Drive And Colab References

Do not rename or delete existing Drive artifacts during local repository
migration. Instead, record old names and Drive IDs in run manifests.

Known N07 Drive folder:

```text
https://drive.google.com/drive/folders/1ymtg3gncPJahaNNWlLShH6_SPXgdEYPL
```

Representative N07 artifact IDs from the completed Colab run:

| Legacy artifact name | Drive file ID | New stage |
|---|---|---|
| `notebook07_gap_audit_for_08x.csv` | `111G8HFHxVrRe78-4AYH1J5GF0m7PxKbt` | `validation_synthesis_gap_audit` |
| `notebook07_lockfile_scope_gate.json` | `1bFeaHXTabe7nser2ka0oGGmldDyQTNg8` | `validation_synthesis_gap_audit` |
| `notebook07_validation_budget_ledger.csv` | `1CPEwW2sMlwFFuQsXwCq6gwWR5UD5DOju` | `validation_synthesis_gap_audit` |
| `notebook07_final_validation_comparison.csv` | `19SVp7eQYZSpM8z_0W1dGqQTv6TpLIM6c` | `validation_synthesis_gap_audit` |

Future run manifests should include both legacy and semantic identifiers:

```json
{
  "legacy_stage_label": "07",
  "stage": "validation_synthesis_gap_audit",
  "legacy_notebook": "notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb",
  "notebook": "notebooks/validation_synthesis_gap_audit_colab.ipynb"
}
```

## Migration Guardrails

- Do not rename files before `configs/pipeline.yaml` can resolve stage order.
- Do not remove compatibility shims until all tests import the new paths.
- Do not delete run-copy notebooks, backups, or Drive artifacts without explicit
  user approval.
- Do not move raw data into GitHub.
- Do not use semantic renaming as a reason to alter model behavior, thresholds,
  wording rules, or validation outputs.
- Do not touch holdout/test data as part of migration.

## Alignment Status

`configs/pipeline.yaml` is the authoritative stage registry; this document is
the legacy <-> target path reference, kept in sync with it. Cross-checks below
hold for the current workspace state (this file and `configs/pipeline.yaml`
are both still untracked; once committed, restate as "as of commit <sha>"):

- Stage names in `configs/pipeline.yaml` and the Stage Name Registry above
  match 1:1 (7 stages).
- Every `legacy_paths.notebook` / `legacy_paths.generator` /
  `legacy_paths.contract` from `configs/pipeline.yaml` is referenced in the
  tables above.
- Phase 2B (flat -> `src/` layout) completed: `pyproject.toml`
  `[tool.setuptools.packages.find]` uses `where = ["src"]`; `pytest.ini` uses
  `pythonpath = src .`; existing helpers (`baseline_v1.py`,
  `validation_pipeline.py`) moved into `src/intraday_research/`.
- Phase 3 (contract migration) completed: canonical contract code now lives at
  `src/intraday_research/contracts/<stage>.py`; the old
  `scripts/notebook0{6,7,8}_contract.py` paths are explicit re-export shims
  (no bare `import *`) so existing notebooks, tests, and generators continue
  to resolve `c.<name>` and `c._<name>` without modification.
- Run copies, backups (`*.bak_*`), and N07 Drive artifact IDs above remain
  preserved verbatim; no historical entries were removed or reordered in this
  alignment pass.

## Static-Gate Path Anchor (Phase 10-fix-2)

Both the generators and the matching static-gate test must read the contract
module from its canonical src/ location, NOT from the legacy `scripts/`
shim:

| Stage | Generator `CONTRACT_MODULE` (and test `CONTRACT_PATH` if any) | Old shim (compat only) |
|---|---|---|
| selective_no_trade_calibration (N06) | `src/intraday_research/contracts/selective_no_trade_calibration.py` | `scripts/notebook06_contract.py` |
| validation_synthesis_gap_audit (N07) | `src/intraday_research/contracts/validation_synthesis_gap_audit.py` | `scripts/notebook07_contract.py` |
| deep_sequence_exploration (N08) | `src/intraday_research/contracts/deep_sequence_exploration.py` | `scripts/notebook08_contract.py` |

The `scripts/notebook0{6,7,8}_contract.py` paths are explicit re-export
shims only; reading them would inline ~21 lines into the regenerated
notebook and silently break downstream cells. Do NOT revert the generators
or `tests/test_notebook07_static_gate.py::CONTRACT_PATH` back to the shim
paths during a future cleanup pass.

