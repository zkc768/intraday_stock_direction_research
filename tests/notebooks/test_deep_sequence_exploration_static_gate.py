"""Static gate for Notebook 08 design + (when generated) notebook.

Phase 1 / Task P1-T07..T09 of phased_implementation_plan.md.

Two layers (same shape as test_notebook07_static_gate.py):
  A. design-doc-only checks (always run): confirm the 2026-06-06 N08 patches
     are present (Pre-registration Constants Table §5.5 with 13 constants;
     DMC entry gate text in §9.1; ledger append step in §10.2;
     §11.1 Tier Escalation Rule).
  B. notebook static gate (skipped until 08 colab notebook is generated):
     default RUN_08X/F/O_* and operator acknowledgements must be False,
     AST parses, no forbidden imports, etc.
"""

import ast
from pathlib import Path

import nbformat
import pytest


# Relocated 2026-06-06 (Phase 7 test relocation): tests/test_notebook08_static_gate.py
# -> tests/notebooks/test_deep_sequence_exploration_static_gate.py. parents[2]
# (file -> notebooks -> tests -> repo root) preserves the project-root anchor.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Renamed 2026-06-06 (Phase 7 semantic rename); legacy path was
# `notebooks/08_deep_sequence_exploration_colab.ipynb`.
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "deep_sequence_exploration_colab.ipynb"
DESIGN_PATH = (
    PROJECT_ROOT
    / "docs"
    / "NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md"
)
GENERATOR_PATH = (
    PROJECT_ROOT / "scripts" / "create_deep_sequence_exploration_colab_notebook.py"
)
CONTRACT_PATH = (
    PROJECT_ROOT
    / "src"
    / "intraday_research"
    / "contracts"
    / "deep_sequence_exploration.py"
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


def joined_notebook_source() -> str:
    return "\n".join(cell.source for cell in load_notebook().cells)


def notebook_code_cells():
    return [cell for cell in load_notebook().cells if cell.cell_type == "code"]


notebook_required = pytest.mark.skipif(
    not NOTEBOOK_PATH.exists(),
    reason=(
        "08 colab notebook not generated yet (design-only stage). "
        "These checks will activate once the notebook lands."
    ),
)


# ---------- A. Design-doc checks (always run) -------------------------------


def test_design_doc_exists_and_nonempty():
    text = design_text()
    assert len(text) > 1000, "design doc unexpectedly small"


def test_design_has_pre_registration_constants_table():
    """P1-T08: §5.5 Pre-registration Constants Table + 13 constant names."""
    text = design_text()
    assert "## 5.5. Pre-registration Constants Table" in text, "section header missing"
    required_constants = [
        "improvement_threshold_delta_macro_f1_lcb_95",
        "improvement_threshold_positive_ticker_count_min",
        "fusion_min_lcb_advantage_over_components",
        "candidate_eligibility_min_train_inner_lcb_delta",
        "paper_safe_score_weight_lcb_delta",
        "paper_safe_score_weight_mean_delta",
        "paper_safe_score_weight_seed_stability",
        "paper_safe_score_weight_fold_consistency",
        "paper_safe_score_weight_per_ticker",
        "paper_safe_score_penalty_complexity",
        "paper_safe_score_penalty_compute",
        "class_collapse_pred_rate_min",
        "total_trial_budget_cap_across_all_families",
    ]
    missing = [c for c in required_constants if c not in text]
    assert not missing, f"constants missing from N08 design: {missing}"


def test_design_constants_back_reference_agents_md_4_2_5a():
    text = design_text()
    assert "AGENTS.md §4.2.5a" in text, "AGENTS.md §4.2.5a back-reference missing"


def test_design_has_dmc_entry_gate():
    """P1-T10 partial: §9.1 must mention dmc_attestation.json gate."""
    text = design_text()
    assert "dmc_attestation.json" in text
    assert "separate Colab session by a non-08X-author" in text


def test_design_has_ledger_step_zero():
    """§10.2 step 0 reads project-level ledger BEFORE official validation read."""
    text = design_text()
    assert "0. Load `notebook07_validation_budget_ledger.csv`" in text
    assert "BEFORE reading any official-validation metric" in text


def test_design_has_tier_escalation_rule():
    """P1-T09: §11.1 Tier Escalation Rule section + gate values."""
    text = design_text()
    assert "### 11.1 Tier Escalation Rule" in text
    # Quick→medium gate threshold must be 0.003 (per top-5 #5).
    assert ">= 0.003 AND positive on >= 4 tickers" in text
    assert "08x_tier_escalation_blocked.json" in text


def test_design_states_validation_only_for_08o():
    text = design_text()
    # 08O scope must remain validation_only.
    assert "scope = validation_only" in text


def test_design_keeps_holdout_test_closed():
    text = design_text()
    assert "touch holdout/test" in text  # appears in §16 "must not"


# ---------- B. Notebook static gate (active once notebook is generated) -----


@notebook_required
def test_notebook08_exists_and_parses():
    nb = load_notebook()
    nbformat.validate(nb)
    assert len(nb.cells) >= 8


@notebook_required
def test_notebook08_has_no_saved_outputs_or_execution_counts():
    nb = load_notebook()
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(not cell.get("outputs") for cell in code_cells)
    assert [cell.get("execution_count") for cell in code_cells] == [None] * len(code_cells)


@notebook_required
def test_notebook08_code_cells_ast_parse():
    for index, cell in enumerate(load_notebook().cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"notebook08_cell_{index}")


@notebook_required
def test_notebook08_inlined_contract_helper_matches_source_module():
    contract_source = CONTRACT_PATH.read_text(encoding="utf-8").strip()
    first_code_cell = notebook_code_cells()[0].source.strip()
    assert first_code_cell == contract_source, (
        "N08 notebook inlined contract helper is stale. Regenerate "
        "notebooks/deep_sequence_exploration_colab.ipynb from "
        "scripts/create_deep_sequence_exploration_colab_notebook.py "
        "(its CONTRACT_MODULE must point at the canonical "
        "src/intraday_research/contracts/deep_sequence_exploration.py, "
        "NOT at the legacy scripts/notebook08_contract.py shim)."
    )


@notebook_required
def test_notebook08_contract_text_does_not_describe_legacy_shim_as_source():
    source = joined_notebook_source()
    assert "Inline copy of `scripts/notebook08_contract.py`" not in source
    assert "sourced from notebook08_contract" not in source
    assert "see ``validate_08f_entry`` in scripts/notebook08_contract.py" not in source
    assert "families subset of ARCHITECTURE_FAMILIES, single" not in source
    assert "src/intraday_research/contracts/deep_sequence_exploration.py" in source


@notebook_required
def test_notebook08_default_run_switches_are_inert():
    source = joined_code_source()
    for switch in (
        "RUN_08X_SCHEMA_SMOKE",
        "RUN_08X_BUILD_TRAIN_INNER_FOLDS",
        "RUN_08X_SEARCH_SPACE_DRY_RUN",
        "RUN_08X_QUICK_SEARCH",
        "RUN_08X_MEDIUM_SEARCH",
        "RUN_08X_AGGRESSIVE_SEARCH",
        "RUN_08X_AGGREGATE_FAILURE_MAP",
        "RUN_08F_CONTRACT_GATE",
        "RUN_08F_CANDIDATE_COMPRESSION",
        "RUN_08F_WRITE_FREEZE_RECORD",
        "RUN_08O_ENTRY_GATE",
        "RUN_08O_OFFICIAL_VALIDATION_READOUT",
        "RUN_08O_AGGREGATE_AND_WRITE_MANIFEST",
    ):
        assert f"{switch} = False" in source, f"{switch} not defaulted to False"
        assert f"{switch} = True" not in source
    assert "BACKUP_NOTEBOOK08_TO_GOOGLE_DRIVE = False" in source


@notebook_required
def test_notebook08_rejects_forbidden_imports_and_calls():
    source = joined_code_source()
    forbidden = [
        "from intraday_research",
        "baseline_helpers",
        "drive.mount(",
        "train_test_split",
        "holdout_test_authorized = True",
    ]
    found = [f for f in forbidden if f in source]
    assert not found, f"forbidden tokens present in N08 notebook: {found}"


@notebook_required
def test_notebook08_has_no_official_validation_selection_strings_in_08x():
    """08X must not select on official validation; defensive token grep."""
    source = joined_code_source()
    banned = (
        "official_validation_used_for_selection = True",
        "select_on_official_validation",
        "official_val_best_picked",
    )
    for token in banned:
        assert token not in source, f"banned 08X selection token: {token}"
