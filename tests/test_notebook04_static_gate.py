import ast
from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/04_controlled_followup_colab.ipynb")
PROTOCOL_PATH = Path("docs/CONTROLLED_FOLLOWUP_PROTOCOL_2026-06-04.md")


def load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def code_cells():
    return [cell for cell in load_notebook().cells if cell.cell_type == "code"]


def assignment_value(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment for {name}")


def joined_code():
    return "\n".join(cell.source for cell in code_cells())


def test_notebook04_has_no_saved_outputs_or_execution_counts():
    cells = code_cells()

    assert sum(len(getattr(cell, "outputs", [])) for cell in cells) == 0
    assert [cell.get("execution_count") for cell in cells] == [None] * len(cells)


def test_notebook04_code_cells_parse_as_python():
    for index, cell in enumerate(code_cells(), start=1):
        ast.parse(cell.source, filename=f"notebook04_cell_{index}")


def test_notebook04_run_guards_default_false_and_fixed_scope():
    setup_source = code_cells()[1].source

    for name in (
        "INSTALL_LIGHTGBM_IF_MISSING",
        "INSTALL_TORCH_IF_MISSING",
        "RUN_04S_SCHEMA_SMOKE",
        "RUN_04A_READ_CONTEXT",
        "RUN_04B_FRESH_SEED_PANEL",
        "RUN_04C_SELECTIVE_COVERAGE",
        "RUN_04D_GATE_DECISION",
        "RUN_04E_BOOTSTRAP_CI",
        "BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE",
    ):
        assert assignment_value(setup_source, name) is False
    assert assignment_value(setup_source, "RESULT_SCOPE") == "validation_only"


def test_notebook04_fixed_candidate_panel_and_fresh_seeds():
    setup_source = code_cells()[1].source

    assert assignment_value(setup_source, "FRESH_SEEDS") == (606, 707, 808, 909, 1010)
    assert assignment_value(setup_source, "NOTEBOOK04_CANDIDATE") == {
        "candidate_id": "stage0_official",
        "label_config": "h03_bps1p5",
        "feature_set": "price_volume_time",
        "window_size": 20,
        "source": "official_stage0_candidate_from_notebook02",
    }
    assert assignment_value(setup_source, "BASELINE_MODELS") == ("stratified_dummy", "always_up_dummy")
    assert assignment_value(setup_source, "TABULAR_MODELS") == ("logreg", "lightgbm")
    assert assignment_value(setup_source, "SEQUENCE_MODELS") == ("standalone_tcn", "ms_dlinear_tcn")


def test_notebook04_forbidden_active_code_strings_absent():
    source = joined_code()

    forbidden = {
        "from intraday_research",
        "baseline_helpers",
        "train_test_split",
        "drive.mount(",
        '"holdout_test_authorized": True',
        "holdout_test_authorized = True",
    }
    assert not [text for text in forbidden if text in source]


def test_notebook04_prediction_artifact_schema_and_manifest_paths():
    source = joined_code()

    assert 'PREDICTION_DIR / f"{model_name}__seed{int(seed)}.npz"' in source
    for array_name in (
        "validation_sample_id",
        "ticker",
        "timestamp",
        "y_true",
        "y_pred",
        "prob_up",
        "confidence",
    ):
        assert f'"{array_name}"' in source
    assert "np.savez_compressed(artifact_path, **payload)" in source
    assert 'OUTPUT_FILES["prediction_manifest"]' in source
    assert 'prob_up_source": "predict_proba[:, 1]"' in source
    assert 'softmax(logits)[:, 1]' in source


def test_notebook04_downloads_latest_timestamped_notebook03_context_from_drive():
    source = joined_code()

    assert 'NOTEBOOK03_DRIVE_RESULTS_FOLDER_ID = "1qQbkwV07X6L_D_WtRYrHDmZ3KXjsju9r"' in source
    assert "ensure_latest_notebook03_context_from_drive" in source
    assert "find_latest_drive_file_by_suffix" in source
    assert '"notebook03_validation_selection.json"' in source
    assert '"notebook03_summary.csv"' in source
    assert 'Path("/content/notebook03_model_family_screening_results/notebook03_validation_selection.json")' in source
    assert 'Path("/content/notebook03_model_family_screening_results/notebook03_summary.csv")' in source
    assert "Downloaded Notebook 03 selection:" in source
    assert "Downloaded Notebook 03 summary:" in source


def test_notebook04_drive_backup_helpers_and_stage_hooks_present():
    source = joined_code()

    assert "NOTEBOOK04_DRIVE_BACKUP_FOLDER_NAME" in source
    assert "find_or_create_drive_folder" in source
    assert "upload_local_file_to_drive" in source
    assert "backup_notebook04_outputs" in source
    assert 'PREDICTION_DIR.glob("*.npz")' in source
    assert 'PREDICTION_DIR.glob(f"{prediction_model_name}__seed*.npz")' in source
    assert 'backup_notebook04_outputs("completed_04A_context_check")' in source
    assert 'f"checkpoint_04B_after_{model_name}"' in source
    assert "prediction_model_name=model_name" in source
    assert 'backup_notebook04_outputs("completed_04B_fresh_seed_panel")' in source
    assert 'backup_notebook04_outputs("completed_04C_selective_coverage")' in source
    assert 'backup_notebook04_outputs("completed_04D_gate_decision")' in source
    assert 'backup_notebook04_outputs("completed_04E_bootstrap_ci")' in source


def test_notebook04_context_only_does_not_trigger_raw_data_loading():
    source = joined_code()

    assert "RUN_ANY_STAGE = bool(RUN_04B_FRESH_SEED_PANEL)" in source
    assert "RUN_ANY_STAGE = bool(RUN_04A_READ_CONTEXT or RUN_04B_FRESH_SEED_PANEL)" not in source


def test_notebook04_selective_coverage_same_row_dummy_deltas():
    source = joined_code()

    assert 'load_prediction_artifact("stratified_dummy", seed)' in source
    assert 'load_prediction_artifact("always_up_dummy", seed)' in source
    assert 'np.array_equal(model_payload["validation_sample_id"], stratified_payload["validation_sample_id"])' in source
    assert 'np.array_equal(model_payload["validation_sample_id"], always_up_payload["validation_sample_id"])' in source
    assert "SELECTIVE_COVERAGE_GRID = (1.00, 0.80, 0.60, 0.40, 0.20, 0.10)" in source
    assert "delta_macro_f1_vs_stratified_dummy_same_rows" in source
    assert "delta_macro_f1_vs_always_up_dummy_same_rows" in source


def test_notebook04_manual_gate_does_not_auto_authorize_next_step():
    source = joined_code()

    assert "operator_selected" in source
    assert '"operator_selected": False' in source
    assert '"manual_operator_decision_required": True' in source
    assert '"holdout_test_authorized": False' in source
    assert "pre04_design_review_source_required_for_exit_b" in source
    assert "auto-authorize" in load_notebook().cells[19].source


def test_controlled_followup_protocol_v11_review_suggestions_present():
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    assert "1.1 (2026-06-04)" in protocol
    assert "pre-04 provenance requirement" in protocol
    assert "must not be invented post-hoc from fresh-seed 04B results" in protocol
    assert "Numeric validation-trial budget tracker" in protocol
    assert "Notebook 04 controlled follow-up | 30 | exact planned" in protocol
