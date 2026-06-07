import ast
import json
import math
from pathlib import Path
import time

import nbformat
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


NOTEBOOK_PATH = Path("notebooks/03_model_family_screening_colab.ipynb")


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


def test_notebook03_has_no_saved_outputs_or_execution_counts():
    cells = code_cells()

    assert sum(len(getattr(cell, "outputs", [])) for cell in cells) == 0
    assert [cell.get("execution_count") for cell in cells] == [None] * len(cells)


def test_notebook03_code_cells_parse_as_python():
    for index, cell in enumerate(code_cells(), start=1):
        ast.parse(cell.source, filename=f"notebook03_cell_{index}")


def test_notebook03_run_guards_default_false():
    setup_source = code_cells()[1].source

    assert assignment_value(setup_source, "RUN_03S_SCHEMA_SMOKE") is False
    assert assignment_value(setup_source, "RUN_03A_TABULAR_PANEL") is False
    assert assignment_value(setup_source, "RUN_03B_SEQUENCE_PANEL") is False
    assert assignment_value(setup_source, "RUN_03C_BOOTSTRAP_CI") is False
    assert assignment_value(setup_source, "RUN_H0_DIAGNOSTIC_NOTE") is False
    assert assignment_value(setup_source, "BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE") is False
    assert assignment_value(setup_source, "RUN_OVERNIGHT_03A_03B_PROFILE") is False
    assert assignment_value(setup_source, "RESULT_SCOPE") == "validation_only"


def test_notebook03_official_candidate_is_stage0_window20_only():
    candidates = assignment_value(code_cells()[1].source, "NOTEBOOK03_CANDIDATES")

    assert candidates == (
        {
            "candidate_id": "stage0_official",
            "label_config": "h03_bps1p5",
            "feature_set": "price_volume_time",
            "window_size": 20,
            "source": "completed_stage0_desktop_review_2026-06-04",
        },
    )


def test_notebook03_h0_appendix_is_non_selecting():
    appendix_source = code_cells()[-1].source
    setup_source = code_cells()[1].source
    h0_candidates_block = setup_source.split("H0_OUTPUT_CANDIDATES = (", 1)[1].split(
        "H0_DIAGNOSTIC_WINDOW", 1
    )[0]

    assert "RUN_H0_DIAGNOSTIC_NOTE" in appendix_source
    assert "update_notebook03_outputs" not in appendix_source
    assert "OUTPUT_FILES" not in appendix_source
    assert "NOTEBOOK03_STATE[\"h0_cross_window_appendix\"]" in appendix_source
    assert "non-selecting" in appendix_source
    assert "/content/drive/MyDrive" not in h0_candidates_block


def test_notebook03_records_h0_window32_appendix_result():
    source = "\n".join(cell.source for cell in load_notebook().cells)

    assert "Recorded appendix result" in source
    assert "part1_window_sweep" in source
    assert "lightgbm/profile_B" in source
    assert "0.002265" in source
    assert "noise_level_positive_no_action" in source
    assert "not_selected_for_confirmation" in source
    assert "does not replace the official Stage 0" in source
    assert "does not alter Notebook 03 `selected_branches`" in source


def test_notebook03_uses_fixed_model_panel_and_no_forbidden_active_imports():
    source = "\n".join(cell.source for cell in code_cells())

    for model_name in (
        "stratified_dummy",
        "always_up_dummy",
        "logreg",
        "lightgbm",
        "vanilla_lstm",
        "simple_gru",
        "standalone_tcn",
        "standard_dlinear",
        "ms_dlinear_tcn",
    ):
        assert model_name in source

    forbidden = {
        "from intraday_research",
        "baseline_helpers",
        "train_test_split",
        "RUN_H0_DIAGNOSTIC_NOTE = True",
    }
    assert not [text for text in forbidden if text in source]


def test_notebook03_documents_get_dataset_override_order():
    helper_source = "\n".join(cell.source for cell in code_cells())

    assert "This get_dataset intentionally overrides the 02-copied helper above" in helper_source
    assert "assert_sample_alignment" in helper_source


def test_notebook03_drive_backup_is_explicit_and_default_off():
    source = "\n".join(cell.source for cell in code_cells())

    assert "BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE = False" in source
    assert "RUN_OVERNIGHT_03A_03B_PROFILE = False" in source
    assert "if RUN_OVERNIGHT_03A_03B_PROFILE:" in source
    assert "build_drive_service_for_backup" in source
    assert "find_or_create_drive_folder" in source
    assert "MediaFileUpload" in source
    assert "DRIVE_BACKUP_PROJECT_FOLDER_ID" in source
    assert "notebook03_model_family_screening_results" in source
    assert "backup_notebook03_outputs(reason_from_stages" in source
    assert "drive.mount(" not in source


def test_notebook03_fit_predict_return_order_matches_stage0_helpers():
    source = "\n".join(cell.source for cell in code_cells())

    assert (
        "predictions, fit_seconds, predict_seconds, train_n, fit_status = "
        "fit_predict_model_03(dataset, model_name, seed)"
    ) in source
    assert (
        'return predictions, 0.0, time.perf_counter() - start, len(dataset["y_train"]), '
        '"baseline_stratified"'
    ) in source
    assert (
        'return predictions, 0.0, time.perf_counter() - start, len(dataset["y_train"]), '
        '"baseline_always_up"'
    ) in source
    assert (
        'return predictions, fit_seconds, predict_seconds, train_n, f"fit_ok_device_{device}"'
    ) in source


def test_notebook03_bootstrap_is_only_written_by_03c_cell():
    cells = code_cells()
    helper_source = cells[5].source
    stage03c_source = cells[-2].source

    assert "summary = bootstrap_candidate_rows(summary)" not in helper_source
    assert 'pooled = pd.read_csv(OUTPUT_FILES["pooled"])' in stage03c_source
    assert 'per_ticker = pd.read_csv(OUTPUT_FILES["per_ticker"])' in stage03c_source
    assert "selection = build_selection_record(summary)" in stage03c_source
    assert 'summary.to_csv(OUTPUT_FILES["summary"], index=False)' in stage03c_source
    assert 'OUTPUT_FILES["selection"].open("w", encoding="utf-8")' in stage03c_source
    assert "write_run_manifest(pooled, per_ticker, summary)" in stage03c_source
    assert 'backup_notebook03_outputs("completed_03C_bootstrap_ci")' in stage03c_source


def test_notebook03_03c_synthetic_output_smoke(tmp_path):
    cells = code_cells()
    namespace = {
        "Path": Path,
        "json": json,
        "math": math,
        "time": time,
        "np": np,
        "pd": pd,
        "DummyClassifier": DummyClassifier,
        "f1_score": f1_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "RANDOM_SUBSAMPLE_SEED": 101,
        "MAX_TRAIN_ROWS": None,
        "LABEL_CONFIGS": {"h03_bps1p5": {"horizon_k": 3, "threshold_bps": 1.5}},
        "MODEL_SEEDS": (101, 202, 303, 404, 505),
        "BASELINE_MODELS": ("stratified_dummy", "always_up_dummy"),
        "SEQUENCE_MODELS": ("vanilla_lstm", "simple_gru", "standalone_tcn", "standard_dlinear", "ms_dlinear_tcn"),
        "NOTEBOOK03_CANDIDATES": (
            {
                "candidate_id": "stage0_official",
                "label_config": "h03_bps1p5",
                "feature_set": "price_volume_time",
                "window_size": 20,
                "source": "synthetic_test",
            },
        ),
        "RESULT_SCOPE": "validation_only",
        "MIN_PRACTICAL_DELTA_MACRO_F1": 0.005,
        "LGBM_PARAMS": {},
        "TORCH_EPOCHS": 8,
        "TORCH_BATCH_SIZE": 1024,
        "TORCH_LEARNING_RATE": 1e-3,
        "TORCH_WEIGHT_DECAY": 1e-4,
        "TORCH_DROPOUT": 0.10,
        "BOOTSTRAP_CI_FULL_PANEL": False,
        "BOOTSTRAP_CI_FOR_CANDIDATES": True,
        "BOOTSTRAP_RESAMPLES": 8,
        "RUN_03S_SCHEMA_SMOKE": False,
        "RUN_03A_TABULAR_PANEL": True,
        "RUN_03B_SEQUENCE_PANEL": True,
        "RUN_03C_BOOTSTRAP_CI": True,
        "RUN_H0_DIAGNOSTIC_NOTE": False,
        "BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE": False,
        "BACKUP_FAILURE_IS_FATAL": True,
        "RUN_OVERNIGHT_03A_03B_PROFILE": True,
        "DRIVE_BACKUP_PROJECT_FOLDER_ID": "synthetic_project_folder",
        "DRIVE_BACKUP_FOLDER_NAME": "notebook03_model_family_screening_results",
        "NOTEBOOK03_STATE": {"pooled_predictions": {}, "h0_cross_window_appendix": "not_loaded"},
        "display": lambda value: None,
    }
    namespace["OUTPUT_DIR"] = tmp_path
    namespace["OUTPUT_FILES"] = {
        "pooled": tmp_path / "notebook03_pooled.csv",
        "per_ticker": tmp_path / "notebook03_per_ticker.csv",
        "summary": tmp_path / "notebook03_summary.csv",
        "selection": tmp_path / "notebook03_validation_selection.json",
        "manifest": tmp_path / "notebook03_run_manifest.json",
    }

    exec(cells[4].source, namespace)
    exec(cells[5].source, namespace)

    pooled_rows = []
    for index, seed in enumerate(namespace["MODEL_SEEDS"]):
        pooled_rows.append({
            "stage": "03A_tabular_panel",
            "model": "logreg",
            "candidate_id": "stage0_official",
            "model_role": "tabular_model",
            "label_config": "h03_bps1p5",
            "horizon_k": 3,
            "threshold_bps": 1.5,
            "feature_set": "price_volume_time",
            "window_size": 20,
            "seed": seed,
            "scope": "validation_only",
            "ticker_or_pooled": "pooled",
            "n": 20,
            "macro_f1": 0.620 + index * 0.0001,
            "balanced_accuracy": 0.620 + index * 0.0001,
            "accuracy": 0.620 + index * 0.0001,
            "stratified_dummy_macro_f1": 0.500,
            "stratified_dummy_balanced_accuracy": 0.500,
            "delta_macro_f1_vs_stratified_dummy": 0.120 + index * 0.0001,
            "delta_balanced_accuracy_vs_stratified_dummy": 0.120 + index * 0.0001,
            "always_up_dummy_macro_f1": 0.333,
            "always_up_dummy_balanced_accuracy": 0.500,
            "delta_macro_f1_vs_always_up_dummy": 0.287 + index * 0.0001,
            "delta_balanced_accuracy_vs_always_up_dummy": 0.120 + index * 0.0001,
            "pred_up_pct": 0.50,
            "pred_down_pct": 0.50,
            "one_class_collapse": False,
            "cm_tn": 10,
            "cm_fp": 0,
            "cm_fn": 0,
            "cm_tp": 10,
            "prep_seconds": 0.0,
            "fit_seconds": 0.01,
            "predict_seconds": 0.01,
            "total_seconds": 0.02,
            "fit_status": "converged",
            "run_failed": False,
            "failure_reason": "",
            "train_n": 100,
            "positive_ticker_count": 5,
            "top_ticker_gain_share": 0.20,
        })
        namespace["NOTEBOOK03_STATE"]["pooled_predictions"][("stage0_official", "logreg", seed)] = {
            "y_true": np.array([0, 1] * 10),
            "predictions": np.array([0, 1] * 10),
        }

    pooled = pd.DataFrame(pooled_rows)
    per_ticker = pooled.assign(ticker_or_pooled="CSCO")
    _, _, summary, selection = namespace["update_notebook03_outputs"](pooled, per_ticker)

    assert selection["selection_status"] == "candidate_signal_found"
    assert summary["signal_strength_tag"].eq("candidate_signal").any()
    assert summary["macro_f1_bootstrap_ci_lower"].isna().all()

    exec(cells[-2].source, namespace)

    updated_summary = pd.read_csv(namespace["OUTPUT_FILES"]["summary"])
    assert updated_summary["macro_f1_bootstrap_ci_lower"].notna().any()
    assert updated_summary["macro_f1_bootstrap_ci_upper"].notna().any()
    assert namespace["OUTPUT_FILES"]["selection"].exists()
    assert namespace["OUTPUT_FILES"]["manifest"].exists()
