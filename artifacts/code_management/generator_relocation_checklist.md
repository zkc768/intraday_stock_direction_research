# Generator relocation checklist (Phase 8)

This checklist enumerates the planned move of `scripts/create_*_colab_notebook.py`
generators into `scripts/notebooks/generate_*_colab.py`. The authoritative file
inventory is `artifacts/code_management/migration_inventory.csv`; no file is moved here.

| Old | New | Shim policy | Static-gate update order |
|---|---|---|---|
| `scripts/create_config_screening_colab_notebook.py` | `scripts/notebooks/generate_config_screening_colab.py` | Keep old path as shim re-exporting `main()` | After move; N02 has no numbered static gate (covered by umbrella `tests/test_notebook_static_gate.py`) |
| `scripts/create_model_family_screening_colab_notebook.py` | `scripts/notebooks/generate_model_family_screening_colab.py` | Keep old path as shim re-exporting `main()` | Update `tests/test_notebook03_static_gate.py` after move |
| `scripts/create_controlled_followup_colab_notebook.py` | `scripts/notebooks/generate_controlled_followup_colab.py` | Keep old path as shim re-exporting `main()` | Update `tests/test_notebook04_static_gate.py` after move |
| `scripts/create_lightgbm_tuning_colab_notebook.py` | `scripts/notebooks/generate_lightgbm_tuning_colab.py` | Keep old path as shim re-exporting `main()` | Update `tests/test_notebook05_static_gate.py` after move |
| `scripts/create_selective_no_trade_calibration_colab_notebook.py` | `scripts/notebooks/generate_selective_no_trade_calibration_colab.py` | Keep old path as shim re-exporting `main()` | Update `tests/test_notebook06_static_gate.py` after move |
| `scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py` | `scripts/notebooks/generate_validation_synthesis_gap_audit_colab.py` | Keep old path as shim re-exporting `main()`; new name drops the filler word "and" | Update `tests/test_notebook07_static_gate.py` after move |
| `scripts/create_deep_sequence_exploration_colab_notebook.py` | `scripts/notebooks/generate_deep_sequence_exploration_colab.py` | Keep old path as shim re-exporting `main()` | Update `tests/test_notebook08_static_gate.py` after move |
| `scripts/create_diagnostic_h0_tabular_sweep_colab_notebook.py` | `scripts/notebooks/generate_diagnostic_h0_tabular_sweep_colab.py` (OPTIONAL — NOT in canonical 02-08 pipeline; diagnostic lane only) | Shim or archive per user decision | No canonical static gate; diagnostic lane outside the umbrella test |
