import ast
from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("notebooks/04_ian_research_memo.ipynb")


def load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def code_cells():
    return [cell for cell in load_notebook().cells if cell.cell_type == "code"]


def test_notebook_has_no_saved_outputs_or_execution_counts():
    cells = code_cells()

    assert sum(len(getattr(cell, "outputs", [])) for cell in cells) == 0
    assert [cell.get("execution_count") for cell in cells] == [None] * len(cells)


def test_notebook_run_guards_default_false():
    setup_source = code_cells()[0].source
    tree = ast.parse(setup_source)
    expected_names = {
        "RUN_DATA_LOAD",
        "RUN_FEATURE_BUILD",
        "RUN_MODEL_VALIDATION",
        "RUN_TRAINING",
        "RESULT_SCOPE",
    }
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in expected_names:
                assignments[target.id] = ast.literal_eval(node.value)

    assert assignments["RUN_DATA_LOAD"] is False
    assert assignments["RUN_FEATURE_BUILD"] is False
    assert assignments["RUN_MODEL_VALIDATION"] is False
    assert assignments["RUN_TRAINING"] is False
    assert assignments["RESULT_SCOPE"] == "validation_only"


def test_notebook_uses_active_helper_and_no_forbidden_imports():
    imported = set()
    for cell in code_cells():
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

    assert "intraday_research" in imported
    forbidden = {
        "archive",
        "legacy_model_runner",
        "ml_utils",
        "runner_utils",
        "train_test_split",
    }
    assert not (imported & forbidden)


def test_notebook_has_no_duplicate_safety_critical_helper_defs():
    duplicate_defs = {
        "add_baseline_v1_features",
        "make_no_trade_band_labels",
        "assign_calendar_split",
        "add_split_and_invalidate_boundaries",
        "fit_train_only_scaler",
        "transform_train_and_validation",
        "build_windows_for_segment",
        "build_windows_by_ticker_and_split",
        "evaluate_stratified_dummy",
    }
    found = set()
    for cell in code_cells():
        tree = ast.parse(cell.source)
        found.update(
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        )

    assert not (found & duplicate_defs)


def test_notebook_does_not_reference_raw_feature_fallback_symbols():
    names = set()
    for cell in code_cells():
        tree = ast.parse(cell.source)
        names.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))

    assert "available_columns" not in names


def test_notebook_uses_cumulative_return_label_contract():
    label_columns = None
    for cell in code_cells():
        tree = ast.parse(cell.source)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "label_columns":
                label_columns = ast.literal_eval(node.value)

    assert label_columns == [
        "future_cumulative_return",
        "label",
        "invalid_cross_day",
        "invalid_missing_future",
    ]


def test_notebook_static_coverage_includes_live_file_existence_check():
    coverage_columns = None
    for cell in code_cells():
        tree = ast.parse(cell.source)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "coverage_columns":
                coverage_columns = ast.literal_eval(node.value)

    assert "current_file_exists" in coverage_columns
