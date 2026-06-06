"""Static gate for Notebook 07 design + (when generated) notebook.

Phase 1 / Task P1-T01..T04 of phased_implementation_plan.md.

Two layers:
  A. design-doc-only checks (always run): confirm the 8 patches landed in the
     2026-06-06 N07 technical design file (Pre-registration Constants Table,
     thesis_paragraph_kit.json output, ledger cumulative field, scope tag).
  B. notebook static gate (skipped until 07 colab notebook is generated):
     default RUN_07*/OPERATOR_ACKNOWLEDGES_* must be False, AST parses, no
     forbidden imports, etc.

Style is intentionally aligned with tests/test_notebook06_static_gate.py so a
later Codex pass can fold this into the canonical 06-style harness.
"""

import ast
from pathlib import Path

import nbformat
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "07_validation_synthesis_and_gap_audit_colab.ipynb"
DESIGN_PATH = (
    PROJECT_ROOT
    / "docs"
    / "NOTEBOOK07_VALIDATION_SYNTHESIS_AND_GAP_AUDIT_TECHNICAL_DESIGN_2026-06-06.md"
)
GENERATOR_PATH = (
    PROJECT_ROOT / "scripts" / "create_validation_synthesis_and_gap_audit_colab_notebook.py"
)


# ---------- Helpers ---------------------------------------------------------


def design_text() -> str:
    assert DESIGN_PATH.exists(), f"missing design doc: {DESIGN_PATH}"
    return DESIGN_PATH.read_text(encoding="utf-8")


def load_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def joined_code_source() -> str:
    nb = load_notebook()
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


notebook_required = pytest.mark.skipif(
    not NOTEBOOK_PATH.exists(),
    reason=(
        "07 colab notebook not generated yet (design-only stage). "
        "These checks will activate once the notebook lands."
    ),
)


# ---------- A. Design-doc checks (always run) -------------------------------


def test_design_doc_exists_and_nonempty():
    text = design_text()
    assert len(text) > 1000, "design doc unexpectedly small"


def test_design_has_pre_registration_constants_table():
    """P1-T02: Pre-registration Constants Table section + 8 constant names."""
    text = design_text()
    assert "## Pre-registration Constants Table" in text, "section header missing"
    required_constants = [
        "improvement_threshold_delta_macro_f1_lcb_95",
        "improvement_threshold_positive_ticker_count_min",
        "weak_signal_band_upper",
        "weak_signal_band_lower",
        "concentration_warning_top_ticker_share_max",
        "concentration_warning_positive_ticker_count_min",
        "weak_seed_evidence_count_threshold",
        "null_control_alpha_total",
    ]
    missing = [c for c in required_constants if c not in text]
    assert not missing, f"constants missing from N07 design: {missing}"


def test_design_constants_reference_agents_md_4_2_5a():
    """The improvement-threshold constants must point at AGENTS.md §4.2.5a."""
    text = design_text()
    # Both constants should appear on lines referencing AGENTS.md §4.2.5a.
    assert "AGENTS.md §4.2.5a" in text, "AGENTS.md §4.2.5a back-reference missing"


def test_design_includes_thesis_paragraph_kit_output():
    """P1-T03: thesis_paragraph_kit.json appears in Required outputs."""
    text = design_text()
    assert "notebook07_thesis_paragraph_kit.json" in text
    # And the schema block describing it must be present.
    assert "thesis_paragraph_kit.json` schema" in text


def test_design_ledger_has_cumulative_across_notebooks_field():
    """P1-T04: §07C Ledger columns must include the cross-notebook field."""
    text = design_text()
    assert "cumulative_official_validation_inspections_across_notebooks" in text
    assert "Cross-notebook append-only rule" in text


def test_design_states_validation_only_scope():
    text = design_text()
    # The design doc explicitly tags scope: validation_only at multiple points.
    assert "validation_only" in text


def test_design_forbids_holdout_test_access():
    text = design_text()
    # These must remain visible as explicit prohibitions even after Edits.
    assert "no holdout/test access" in text.lower() or "holdout/test" in text


def test_design_forbids_diagnostics_as_selection_gates():
    text = design_text()
    assert "Do not use SHAP, permutation importance, ECE, Brier, AURC, or null-control results as selection gates." in text


# ---------- B. Notebook static gate (active once notebook is generated) -----


@notebook_required
def test_notebook07_exists_and_parses():
    nb = load_notebook()
    nbformat.validate(nb)
    assert len(nb.cells) >= 8


@notebook_required
def test_notebook07_has_no_saved_outputs_or_execution_counts():
    nb = load_notebook()
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(not cell.get("outputs") for cell in code_cells)
    assert [cell.get("execution_count") for cell in code_cells] == [None] * len(code_cells)


@notebook_required
def test_notebook07_code_cells_ast_parse():
    for index, cell in enumerate(load_notebook().cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook07_cell_{index}")


@notebook_required
def test_notebook07_default_run_switches_are_inert():
    source = joined_code_source()
    for switch in (
        "RUN_07A_LOCKFILE_SCOPE_GATE",
        "RUN_07B_FINAL_VALIDATION_COMPARISON",
        "RUN_07C_VALIDATION_BUDGET_LEDGER",
        "RUN_07D_ROBUSTNESS_AND_CONCENTRATION",
        "RUN_07E_EXPLAINABILITY_APPENDIX",
        "RUN_07F_PERMUTATION_NULL_CONTROL_APPENDIX",
        "RUN_07G_GAP_AUDIT_FOR_08X",
        "RUN_07H_PAPER_READY_SYNTHESIS",
        "RUN_07I_BACKUP_TO_GOOGLE_DRIVE",
    ):
        assert f"{switch} = False" in source, f"{switch} not defaulted to False"
        assert f"{switch} = True" not in source
    assert "BACKUP_NOTEBOOK07_TO_GOOGLE_DRIVE = False" in source
    for ack in (
        "OPERATOR_ACKNOWLEDGES_07_IS_NOT_SEARCH",
        "OPERATOR_ACKNOWLEDGES_NO_HOLDOUT_TEST",
        "OPERATOR_ACKNOWLEDGES_NO_SELECTION_FROM_EXPLANATIONS",
        "OPERATOR_ACKNOWLEDGES_NO_THRESHOLD_SEARCH",
    ):
        assert f"{ack} = False" in source, f"{ack} not defaulted to False"


@notebook_required
def test_notebook07_declares_validation_only_scope():
    source = joined_code_source()
    assert 'NOTEBOOK07_SCOPE = "validation_only"' in source


@notebook_required
def test_notebook07_rejects_forbidden_imports_and_calls():
    source = joined_code_source()
    forbidden = [
        "from intraday_research",
        "baseline_helpers",
        "drive.mount(",
        "train_test_split",
        "holdout_test_authorized = True",
    ]
    found = [f for f in forbidden if f in source]
    assert not found, f"forbidden tokens present in N07 notebook: {found}"


@notebook_required
def test_notebook07_has_no_selection_function_names():
    """Defensive: do not allow obvious selector function/variable names."""
    source = joined_code_source()
    banned = (
        "select_threshold",
        "best_threshold",
        "optimal_threshold",
        "optimal_coverage",
        "select_feature_subset",
        "run_hpo",
        "train_new_model",
    )
    for token in banned:
        assert token not in source, f"banned selection token in N07 notebook: {token}"
