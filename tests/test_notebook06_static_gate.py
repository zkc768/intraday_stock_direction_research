import ast
from pathlib import Path

import nbformat

from scripts import notebook06_contract as contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "06_selective_no_trade_calibration_colab.ipynb"
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "create_selective_no_trade_calibration_colab_notebook.py"


def load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def joined_code_source():
    nb = load_notebook()
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


def test_notebook06_exists_and_parses():
    assert NOTEBOOK_PATH.exists()
    nb = load_notebook()
    nbformat.validate(nb)
    assert len(nb.cells) >= 12


def test_notebook06_has_no_saved_outputs_or_execution_counts():
    nb = load_notebook()
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(not cell.get("outputs") for cell in code_cells)
    assert [cell.get("execution_count") for cell in code_cells] == [None] * len(code_cells)


def test_notebook06_code_cells_ast_parse():
    for index, cell in enumerate(load_notebook().cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook06_cell_{index}")


def test_notebook06_default_run_switches_are_inert():
    source = joined_code_source()
    for switch in (
        "RUN_06A_ARTIFACT_GATE",
        "RUN_06B_PREDICTION_FRAME_AND_PROBABILITY_DIAGNOSTICS",
        "RUN_06C_FIXED_COVERAGE_GRID",
        "RUN_06D_AGGREGATE_AND_RISK_COVERAGE",
        "RUN_06E_CONCENTRATION_GUARDRAILS",
        "RUN_06F_DECISION_RECORD",
        "RUN_06G_BACKUP_TO_GOOGLE_DRIVE",
    ):
        assert f"{switch} = False" in source
        assert f"{switch} = True" not in source
    assert "BACKUP_NOTEBOOK06_TO_GOOGLE_DRIVE = False" in source
    assert "OPERATOR_ACKNOWLEDGES_VALIDATION_ONLY_SCOPE = False" in source
    assert "OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST = False" in source
    assert "OPERATOR_ACKNOWLEDGES_NO_SELECTIVE_THRESHOLD = False" in source


def test_notebook06_forbidden_active_code_strings_absent():
    source = joined_code_source()
    forbidden = (
        "from intraday_research",
        "baseline_helpers",
        "train_test_split",
        "drive.mount",
        "runpy",
        "select_threshold",
        "best_threshold",
        "optimal_threshold",
        "optimal_coverage",
        "confidence_threshold_implied",
    )
    for needle in forbidden:
        assert needle not in source


def test_notebook06_contract_is_inlined_not_project_imported():
    source = joined_code_source()
    assert "def assert_notebook06_artifact_contract(" in source
    assert "def load_notebook06_prediction_artifact(" in source
    assert "def same_row_stratified_dummy_predict(" in source
    assert "def ticker_stratified_random_abstention(" in source
    assert "from scripts import notebook06_contract" not in source


def test_notebook06_required_contract_functions_are_present():
    source = joined_code_source()
    for name in (
        "assert_notebook06_artifact_contract",
        "resolve_notebook06_primary_profile",
        "load_notebook06_prediction_artifact",
        "build_canonical_prediction_frame",
        "calibration_bins",
        "ece_from_bins",
        "risk_coverage_curve",
        "aurc_from_curve",
        "selective_retained_indices",
        "same_row_stratified_dummy_predict",
        "ticker_stratified_random_abstention",
        "concentration_metrics",
        "aggregate_across_seeds",
        "evaluate_decision_outcome",
    ):
        assert f"def {name}(" in source


def test_official_pooled_contract_uses_train_positive_rate_not_holdout_flag():
    assert "train_positive_rate" in contract.REQUIRED_OFFICIAL_POOLED_FIELDS
    assert "holdout_test_authorized" not in contract.REQUIRED_OFFICIAL_POOLED_FIELDS
    assert "holdout_test_authorized" in contract.HARD_REQUIRED_DECISION_FIELDS


def test_notebook06_probability_artifact_contract_uses_prob_up_and_confidence_check():
    source = joined_code_source()
    assert '"prob_up"' in source
    assert '"y_prob_up"' in source
    assert "np.maximum(prob_up, 1.0 - prob_up)" in source
    assert "confidence differs from max(prob_up, 1 - prob_up)" in source


def test_notebook06_writes_required_artifacts():
    source = joined_code_source()
    for artifact in (
        "notebook06_artifact_contract_check.json",
        "notebook06_prediction_frame_manifest.csv",
        "notebook06_probability_diagnostics.csv",
        "notebook06_reliability_bins.csv",
        "notebook06_coverage_grid.csv",
        "notebook06_same_row_baselines.csv",
        "notebook06_random_abstention_baselines.csv",
        "notebook06_risk_coverage_summary.csv",
        "notebook06_concentration_guardrails.csv",
        "notebook06_per_ticker_coverage.csv",
        "notebook06_decision_record.json",
        "notebook06_run_manifest.json",
        "notebook06_drive_backup_manifest.json",
    ):
        assert artifact in source


def test_notebook06_coverage_grid_has_explicit_key_columns():
    source = joined_code_source()
    coverage_section = source[source.index("coverage_rows.append(") : source.index("pd.DataFrame(coverage_rows)")]
    for key in ("profile_id", "profile_role", "seed", "coverage_target"):
        assert key in coverage_section


def test_notebook06_contains_same_row_dummy_and_random_abstention_logic():
    source = joined_code_source()
    assert "same_row_stratified_dummy_predict(" in source
    assert "train_class0_n" in source
    assert "train_class1_n" in source
    assert "train_positive_rate" in source
    assert "ticker_stratified_random_abstention(" in source
    assert "if float(coverage_target) == 1.0:" in source
    assert "random_abstention_repeat_count" in source


def test_notebook06_contains_calibration_risk_and_concentration_metrics():
    source = joined_code_source()
    for needle in (
        "brier_score_binary",
        "log_loss_binary",
        "ece_from_bins",
        "risk_coverage_curve",
        "aurc_from_curve",
        "e_aurc",
        "concentration_metrics",
        "top_ticker_retained_share",
        "ticker_entropy_normalized",
    ):
        assert needle in source


def test_notebook06_decision_record_keeps_holdout_closed_and_no_threshold_selected():
    source = joined_code_source()
    assert '"selective_threshold_selected": False' in source
    assert '"holdout_test_authorized": False' in source
    assert "reported_coverage_grid" in source
    assert "decision_coverage_grid" in source
    assert "secondary_profiles_diagnostic_only" in source
    assert "holdout/test remains closed" in source


def test_notebook06_drive_backup_creates_timestamped_files_without_overwrite():
    source = joined_code_source()
    assert ".create(body=metadata, media_body=media" in source
    assert "service.files().update" not in source
    assert "DRIVE_BACKUP_PREFIX" in source
    assert "timestamp" in source


def test_generator_uses_nbformat_and_validates_forbidden_strings():
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    assert "nbformat.write" in source
    assert "nbformat.validate" in source
    assert "ast.parse" in source
    assert "drive.mount" in source
    assert "optimal_threshold" in source
