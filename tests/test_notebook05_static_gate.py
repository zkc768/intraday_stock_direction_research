import ast
from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/05_lightgbm_tuning_colab.ipynb")


def load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def code_cells():
    return [cell for cell in load_notebook().cells if cell.cell_type == "code"]


def joined_code():
    return "\n".join(cell.source for cell in code_cells())


def joined_notebook_text():
    return "\n".join(cell.source for cell in load_notebook().cells)


def assignment_value(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment for {name}")


def assignment_values(name):
    values = []
    for cell in code_cells():
        tree = ast.parse(cell.source)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                values.append(ast.literal_eval(node.value))
    return values


def function_def(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def test_notebook05_has_no_saved_outputs_or_execution_counts():
    cells = code_cells()
    assert cells
    assert all(not cell.get("outputs") for cell in cells)
    assert [cell.get("execution_count") for cell in cells] == [None] * len(cells)


def test_notebook05_code_cells_parse_as_python():
    for index, cell in enumerate(code_cells(), start=1):
        ast.parse(cell.source, filename=f"notebook05_cell_{index}")


def test_notebook05_run_guards_default_false_and_scope_fixed():
    setup_source = code_cells()[1].source

    for name in (
        "INSTALL_LIGHTGBM_IF_MISSING",
        "INSTALL_TORCH_IF_MISSING",
        "RUN_05A_TO_05E_FULL_PIPELINE",
        "RUN_05S_SCHEMA_SMOKE",
        "RUN_05A_04D_ENTRY_GATE",
        "RUN_05B_TRAIN_INNER_HPO",
        "RUN_05C_SELECT_FINALISTS",
        "RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION",
        "RUN_05E_DECISION_RECORD",
        "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE",
    ):
        assert assignment_value(setup_source, name) is False

    assert assignment_value(setup_source, "RESULT_SCOPE") == "validation_only"
    assert assignment_value(setup_source, "OPERATOR_SELECTED_EXIT") == ""
    assert assignment_value(setup_source, "OPERATOR_ACCEPTS_EXIT_A") is False
    assert assignment_values("INSTALL_LIGHTGBM_IF_MISSING")
    assert assignment_values("INSTALL_TORCH_IF_MISSING")
    assert all(value is False for value in assignment_values("INSTALL_LIGHTGBM_IF_MISSING"))
    assert all(value is False for value in assignment_values("INSTALL_TORCH_IF_MISSING"))


def test_notebook05_fixed_candidate_and_hpo_budget():
    setup_source = code_cells()[1].source

    assert assignment_value(setup_source, "NOTEBOOK05_CANDIDATE") == {
        "candidate_id": "stage0_official",
        "label_config": "h03_bps1p5",
        "horizon_k": 3,
        "threshold_bps": 1.5,
        "feature_set": "price_volume_time",
        "window_size": 20,
        "source": "official_stage0_candidate_from_notebook02_and_notebook04d_exit_a",
    }
    assert assignment_value(setup_source, "OFFICIAL_VALIDATION_SEEDS") == (606, 707, 808, 909, 1010)
    assert assignment_value(setup_source, "HPO_BUDGET") == 100
    assert assignment_value(setup_source, "INNER_FOLD_COUNT") == 3
    assert assignment_value(setup_source, "N_FINALISTS") == 5
    assert assignment_value(setup_source, "PRIMARY_SELECTION") == "train_inner_winner"
    assert assignment_value(setup_source, "INNER_DUMMY_SEED") == 260605
    assert assignment_value(setup_source, "MIN_FINALIST_MEDIAN_BEST_ITERATION") == 20
    assert assignment_value(setup_source, "PROMOTION_MIN_DELTA_MACRO_F1_VS_DEFAULT") == 0.001
    assert -1 not in assignment_value(setup_source, "HPO_MAX_DEPTH_CHOICES")


def test_notebook05_forbidden_active_code_strings_absent():
    source = joined_code()

    forbidden = {
        "from intraday_research",
        "baseline_helpers",
        "train_test_split",
        "drive.mount",
        "runpy",
        '"holdout_test_authorized": True',
        "holdout_test_authorized = True",
        "/content/drive/MyDrive",
        'selected_profile_source = "official_validation_best"',
    }
    present = sorted(text for text in forbidden if text in source)
    assert present == []
    compact_source = source.lower().replace(" ", "")
    assert "holdout_test_authorized=true" not in compact_source
    assert '"holdout_test_authorized":true' not in compact_source
    assert "'holdout_test_authorized':true" not in compact_source


def test_notebook05_04d_entry_gate_is_required_before_fit_stages():
    source = joined_code()

    assert "NOTEBOOK04_DRIVE_RESULTS_FOLDER_ID" in source
    assert "notebook04_context_checks.json" in source
    assert "notebook04_summary.csv" in source
    assert "notebook04_selective_coverage.csv" in source
    assert "notebook04_decision_matrix.csv" in source
    assert "notebook04_run_manifest.json" in source
    assert "def assert_notebook05_entry_gate" in source
    assert "OPERATOR_SELECTED_EXIT != REQUIRED_OPERATOR_EXIT_A" in source
    assert "OPERATOR_ACCEPTS_EXIT_A is not True" in source
    assert "Notebook 04 decision matrix does not include Exit A." in source
    assert "NOTEBOOK05_STATE[\"entry_decision\"] = entry" in source
    assert "assert_notebook05_entry_gate(download_if_missing=True)" in source
    assert "validate_context_official_candidate(context)" in source
    assert "official_candidate" in source
    assert "entry = NOTEBOOK05_STATE.get(\"entry_decision\")" in source
    assert "hpo_authorized" in source
    assert "authorized_candidate" in source
    assert "modifiedTime" in source
    assert "nextPageToken" in source
    assert source.index("def build_drive_service") < source.index("if RUN_05A_04D_ENTRY_GATE:")


def test_notebook05_context_gate_does_not_trigger_raw_data_loading():
    source = joined_code()

    assert "RUN_ANY_STAGE = bool(RUN_05B_TRAIN_INNER_HPO or RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION)" in source
    assert "RUN_ANY_STAGE = bool(RUN_05A_04D_ENTRY_GATE" not in source
    assert "Enable RUN_05B_TRAIN_INNER_HPO or RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION and rerun data loading first." in source
    assert source.index("if RUN_05A_04D_ENTRY_GATE:") < source.index(
        "RUN_ANY_STAGE = bool(RUN_05B_TRAIN_INNER_HPO or RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION)"
    )


def test_notebook05_train_inner_hpo_not_official_validation_selection():
    source = joined_code()

    assert "make_train_inner_folds_05" in source
    assert '"train_timestamp": train_timestamp' in source
    assert '"validation_timestamp": validation_timestamp' in source
    assert '"train_timestamp_seq": train_timestamp_seq' in source
    assert '"validation_timestamp_seq": validation_timestamp_seq' in source
    assert "inner_lcb_macro_f1" in source
    assert "MIN_FINALIST_MEDIAN_BEST_ITERATION" in source
    assert 'record["median_best_iteration"] >= MIN_FINALIST_MEDIAN_BEST_ITERATION' in source
    assert "selected_profile_source" in source
    assert "PRIMARY_SELECTION" in source
    assert "official_validation_used_for_selection" in source
    assert "official_validation_rank_by_macro_f1" in source
    assert "official_validation_diagnostic_rank_by_macro_f1" in source
    assert "selected_by_official_validation" in source
    assert "official_validation_diagnostic_best" in source
    assert "official_validation_ranking_disagrees_with_train_inner" in source
    assert "official_validation_status" in source
    assert "promotion_checks" in source
    assert "downstream_primary_profile_id" in source
    assert "class_count_fields_05" in source
    assert 'f"{prefix}class0_n": class0_n' in source
    assert 'f"{prefix}class1_n": class1_n' in source
    assert 'f"{prefix}positive_rate": positive_rate' in source
    assert "Notebook 05 finalists artifact is missing. Run 05C before 05D." in source
    assert "use_inner_early_stopping=True" in source
    assert "use_inner_early_stopping=False" in source
    assert "dummy_seed = int(INNER_DUMMY_SEED) + int(fold[\"inner_fold_id\"])" in source
    assert "stratified_dummy_seed" in source
    assert "n_finalists_found" in source
    assert "finalist_count_below_target" in source
    assert "official validation" in source.lower()
    assert "official validation selected the final profile" not in source.lower()
    assert 'selected_profile_source = "official_validation_best"' not in source


def test_notebook05_entry_gate_is_first_statement_in_fit_stages():
    source = joined_code()

    for name in ("run_train_inner_hpo_05b", "run_official_validation_confirmation_05d"):
        node = function_def(source, name)
        first = node.body[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "assert_notebook05_entry_gate"


def test_notebook05_dummy_and_selective_boundaries_present():
    source = joined_code()
    notebook_text = joined_notebook_text()

    assert "stratified_dummy_macro_f1" in source
    assert "always_up_dummy_macro_f1" in source
    assert "delta_macro_f1_vs_stratified_dummy_same_rows" in source
    assert "delta_macro_f1_vs_always_up_dummy_same_rows" in source
    assert "sample_id_hash" in source
    assert "make_notebook05_sample_ids" in source
    assert '"validation_sample_id": make_notebook05_sample_ids' in source
    assert '"train_sample_id": make_notebook05_sample_ids' in source
    assert 'pooled_train_class_counts = class_count_fields_05(y_train, "train_")' in source
    assert 'ticker_train_class_counts = class_count_fields_05(y_train[train_mask], "train_")' in source
    assert "**pooled_train_class_counts" in source
    assert "**ticker_train_class_counts" in source
    assert "selective_threshold_selected" in source
    assert '"selective_threshold_selected": False' in source
    assert "Notebook 05 does not select" in notebook_text
    assert "selective/no-trade" in notebook_text


def test_notebook05_drive_backup_is_explicit_and_default_off():
    setup_source = code_cells()[1].source
    source = joined_code()

    assert assignment_value(setup_source, "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE") is False
    assert "backup_notebook05_outputs" in source
    assert 'backup_notebook05_outputs("completed_05B_train_inner_hpo")' in source
    assert 'backup_notebook05_outputs("completed_05C_select_finalists")' in source
    assert 'backup_notebook05_outputs("completed_05D_official_validation_confirmation", include_predictions=True)' in source
    assert 'backup_notebook05_outputs("completed_05E_decision_record")' in source
