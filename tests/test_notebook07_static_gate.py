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

from scripts import notebook07_contract as c


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
CONTRACT_PATH = (
    PROJECT_ROOT
    / "src"
    / "intraday_research"
    / "contracts"
    / "validation_synthesis_gap_audit.py"
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
        "RUN_07J_WRITE_MONITORING_PLAN",
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


# ---------- Design §07 item 17: AST-level forbidden call check -------------


_FORBIDDEN_AST_CALL_NAMES = {"exec", "eval", "compile", "__import__"}
_FORBIDDEN_AST_ATTR_PREFIXES = ("select_", "best_", "optimal_", "run_hpo")


@notebook_required
def test_notebook07_ast_rejects_dynamic_dispatch_to_forbidden_calls():
    """Design §07 item 17 defense: even if the literal `select_threshold`
    string is absent, ban dynamic dispatch (`exec`, `eval`, `compile`,
    `__import__`) and attribute access whose name starts with
    `select_/best_/optimal_/run_hpo` that could bypass the item-16 list via
    `getattr(obj, "select_threshold")` or `module.run_hpo()`.
    """
    nb = load_notebook()
    violations = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source, filename=f"notebook07_cell_{index}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_AST_CALL_NAMES:
                violations.append(f"cell {index}: forbidden call {func.id}()")
            if isinstance(func, ast.Attribute) and func.attr.startswith(
                _FORBIDDEN_AST_ATTR_PREFIXES
            ):
                violations.append(f"cell {index}: forbidden attribute call .{func.attr}()")
    assert not violations, "N07 notebook contains forbidden AST patterns: " + "; ".join(violations)


@notebook_required
def test_notebook07_design_doc_sha_pin_is_hard_gate():
    """07A must refuse to sign the lockfile when EXPECTED_DESIGN_DOC_SHA256
    is empty. Verifies the gate is *active* in the generated source, not
    just declared.
    """
    source = joined_code_source()
    assert 'EXPECTED_DESIGN_DOC_SHA256 = ""' in source, (
        "EXPECTED_DESIGN_DOC_SHA256 default must remain empty in committed source"
    )
    assert "if not EXPECTED_DESIGN_DOC_SHA256:" in source, (
        "07A must hard-raise when EXPECTED_DESIGN_DOC_SHA256 is empty"
    )
    assert "Pin the freeze-time sha" in source or "pin the freeze-time sha" in source.lower(), (
        "07A SHA hard-gate error message must instruct the operator to pin a freeze-time sha"
    )


@notebook_required
def test_notebook07_07F_does_not_synthesize_chronology_aware_null():
    """07F must NOT permute macro_f1 in-cell and call the output
    `chronology-aware`. It must read a pre-registered artifact via
    PRE_REGISTERED_NULL_CONTROL_PATH instead.
    """
    source = joined_code_source()
    assert "PRE_REGISTERED_NULL_CONTROL_PATH" in source, (
        "07F must reference PRE_REGISTERED_NULL_CONTROL_PATH (design 07F option 1)"
    )
    # Prior implementation looped `rng.permutation(...)` over macro_f1 means
    # and labelled the output chronology-aware. That pattern must not return.
    assert "rng.permutation(y_true)" not in source, (
        "07F must not permute macro_f1 in-cell; null must be pre-registered"
    )


@notebook_required
def test_notebook07_07D_emits_diagnostic_scope_not_validation_only():
    """Design §07D: per-ticker / seed / concentration outputs must carry
    ``scope = diagnostic``, not ``scope = validation_only``. Mixing them
    pollutes the official-validation surface.
    """
    nb = load_notebook()
    # Find the 07D cell (heuristic: contains RUN_07D_ROBUSTNESS_AND_CONCENTRATION)
    found = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "RUN_07D_ROBUSTNESS_AND_CONCENTRATION" not in cell.source:
            continue
        if "if RUN_07D_ROBUSTNESS_AND_CONCENTRATION:" not in cell.source:
            continue
        found = True
        # 07D writes per_ticker / seed / concentration records. Each record
        # dict must carry "scope": "diagnostic" (string literal).
        diagnostic_count = cell.source.count('"scope": "diagnostic"')
        # 3 record builders (per_ticker, seed, concentration) emit scope.
        assert diagnostic_count >= 3, (
            f"07D code cell has {diagnostic_count} 'scope: diagnostic' assignments; "
            "expected >= 3 (per_ticker / seed / concentration records)"
        )
        validation_only_count = cell.source.count('"scope": NOTEBOOK07_SCOPE')
        assert validation_only_count == 0, (
            f"07D code cell still emits scope = NOTEBOOK07_SCOPE in {validation_only_count} record(s); "
            "diagnostic outputs must not be labelled validation_only"
        )
    assert found, "Could not locate the 07D code cell in the generated notebook"


@notebook_required
def test_notebook07_07D_accepts_n05_ticker_or_pooled_schema():
    """N05 official per-ticker artifacts use ``ticker_or_pooled`` as the
    ticker identity column. 07D may normalize that to ``ticker`` locally, but
    must not hard-require a non-existent source ``ticker`` column.
    """
    nb = load_notebook()
    found = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "if RUN_07D_ROBUSTNESS_AND_CONCENTRATION:" not in cell.source:
            continue
        found = True
        assert '"ticker" not in per_ticker_rows.columns' in cell.source
        assert '"ticker_or_pooled" not in per_ticker_rows.columns' in cell.source
        assert 'per_ticker_rows["ticker"] = per_ticker_rows["ticker_or_pooled"].astype(str)' in cell.source
        assert 'per_ticker_rows.groupby("ticker", dropna=False, sort=True)' in cell.source
        assert 'per_ticker_rows.groupby(["ticker"], dropna=False, sort=True)' not in cell.source
    assert found, "Could not locate the 07D code cell in the generated notebook"


@notebook_required
def test_notebook07_ledger_helper_is_flush_on_append():
    """Runtime ledger helper must auto-flush to disk under prefix invariance
    on every append, so each phase's intent row is durable before any
    official-validation read.
    """
    source = joined_code_source()
    assert "def flush_ledger_to_disk" in source, (
        "Runtime must define flush_ledger_to_disk()"
    )
    assert "validate_ledger_prefix_invariance" in source, (
        "Runtime must call validate_ledger_prefix_invariance before overwriting the ledger"
    )
    assert "_hydrate_ledger_from_disk_if_needed" in source, (
        "Runtime must hydrate the ledger from disk on first append so kernel restarts are safe"
    )


@notebook_required
def test_notebook07_inlined_contract_helper_matches_source_module():
    """The generated notebook inlines the canonical contract module at
    ``src/intraday_research/contracts/validation_synthesis_gap_audit.py`` for
    Colab portability. If the canonical helper is patched, the notebook must
    be regenerated; otherwise Colab runs a stale contract implementation.

    Post-Phase-3: the old ``scripts/notebook07_contract.py`` path is a thin
    re-export shim for legacy callers and is NOT the byte-for-byte source
    that the notebook inlines. Generators read CONTRACT_MODULE from the
    canonical src/ path; the test's CONTRACT_PATH points there too.
    """
    contract_source = CONTRACT_PATH.read_text(encoding="utf-8").strip()
    first_code_cell = notebook_code_cells()[0].source.strip()
    assert first_code_cell == contract_source, (
        "N07 notebook inlined contract helper is stale. Regenerate "
        "notebooks/07_validation_synthesis_and_gap_audit_colab.ipynb from "
        "scripts/create_validation_synthesis_and_gap_audit_colab_notebook.py "
        "(its CONTRACT_MODULE must point at the canonical "
        "src/intraday_research/contracts/validation_synthesis_gap_audit.py, "
        "NOT at the legacy scripts/notebook07_contract.py shim)."
    )


@notebook_required
def test_notebook07_contract_text_does_not_describe_legacy_shim_as_source():
    source = joined_notebook_source()
    assert "Inline copy of `scripts/notebook07_contract.py`" not in source
    assert "sourced from notebook07_contract" not in source
    assert "follow `scripts/notebook07_contract.py`" not in source
    assert "src/intraday_research/contracts/validation_synthesis_gap_audit.py" in source


@notebook_required
def test_notebook07_inlined_prefix_invariance_uses_strict_column_contract():
    """Round 6 P2 regression lock for the actual Colab runtime copy, not
    only the imported pytest helper module.
    """
    source = joined_code_source()
    assert "shared_cols = sorted(set(existing_reset.columns) & set(new_prefix.columns))" not in source, (
        "N07 notebook still carries the old shared-column intersection prefix check"
    )
    assert "validation_budget_ledger column set changed" in source
    assert "validation_budget_ledger column order changed" in source


@notebook_required
def test_notebook07_ledger_reads_preserve_na_like_literals():
    """Ledger prefix invariance compares literal CSV values. Pandas defaults
    parse strings such as ``n/a`` as NaN and infer strings such as ``07`` as
    integers, which can create false prefix violations after a write/read
    roundtrip.
    """
    source = joined_code_source()
    assert "pd.read_csv(target_path, dtype=str, keep_default_na=False)" in source, (
        "ledger reads must preserve literal strings such as 'n/a' and '07'"
    )
    assert "pd.read_csv(target_path)" not in source, (
        "ledger reads must not use pandas default NA/type inference"
    )


# ---------- Round 4 (Phase B review) regression locks ----------------------


@notebook_required
def test_notebook07_require_lockfile_re_validates_artifact_hashes():
    """Design §07A demands every downstream RUN_07* phase revalidate 07A by
    recomputing artifact hashes and refusing to proceed if any drift since
    the lockfile signing. require_artifact_lockfile() must call the
    revalidator on every invocation, not just check contract_passed.
    """
    source = joined_code_source()
    assert "def revalidate_artifact_hashes" in source, (
        "Runtime must define revalidate_artifact_hashes(lockfile)"
    )
    assert "revalidate_artifact_hashes(lockfile)" in source, (
        "require_artifact_lockfile() must call revalidate_artifact_hashes(lockfile)"
    )
    assert "sha256 drifted" in source or "hashes drifted" in source, (
        "revalidator must report drift explicitly so the operator can act"
    )


@notebook_required
def test_notebook07_07B_downgrades_wording_on_weak_seed_evidence():
    """Design §07 + AGENTS.md §4.2.5a: seed_count < WEAK_SEED_EVIDENCE_COUNT_THRESHOLD
    forces allowed_wording_tag = "weak" and disqualifies improvement wording,
    regardless of which band the LCB would otherwise reach.
    """
    source = joined_code_source()
    assert "WEAK_SEED_EVIDENCE_COUNT_THRESHOLD" in source, (
        "07B must reference the WEAK_SEED_EVIDENCE_COUNT_THRESHOLD constant"
    )
    assert "weak_seed_evidence_flag" in source, (
        "07B must compute a weak_seed_evidence_flag to drive wording downgrade"
    )
    assert 'allowed_wording_tag = "weak"' in source, (
        "07B must downgrade allowed_wording_tag to 'weak' under weak-seed evidence"
    )


@notebook_required
def test_notebook07_07B_hard_stops_on_missing_n05_dummy_columns():
    """Silent NaN at the same-row dummy column would let a row reach thesis
    wording without a baseline. 07B must raise with the exact missing
    column name when N05 official_pooled lacks stratified_dummy_macro_f1
    or delta_macro_f1_vs_stratified_dummy.
    """
    source = joined_code_source()
    assert "stratified_dummy_macro_f1" in source, "07B must reference stratified_dummy_macro_f1"
    assert "delta_macro_f1_vs_stratified_dummy" in source, (
        "07B must reference delta_macro_f1_vs_stratified_dummy"
    )
    assert "is missing required same-row dummy column" in source, (
        "07B must hard-stop with an explicit 'missing required same-row dummy column' message"
    )


# ---------- Round 5 (before-read ledger + 07C audit-only) -----------------


def _find_cell_starting_with(nb, switch_name: str):
    """Return the first code cell whose source contains the literal
    ``if {switch_name}:`` block opener, or None.
    """
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if f"if {switch_name}:" in cell.source:
            return cell
    return None


def _assert_append_precedes_read(cell_source: str, *, read_marker: str, phase_label: str):
    """Helper: in the given cell source, the FIRST ``append_ledger_row(`` call
    must appear at an earlier byte offset than the FIRST occurrence of the
    ``read_marker`` substring. This catches "read-before-append" patterns
    that violate AGENTS.md §4.3.
    """
    append_pos = cell_source.find("append_ledger_row(")
    read_pos = cell_source.find(read_marker)
    assert append_pos != -1, f"{phase_label}: cell must call append_ledger_row(...)"
    assert read_pos != -1, (
        f"{phase_label}: expected read marker '{read_marker}' in cell source"
    )
    assert append_pos < read_pos, (
        f"{phase_label}: append_ledger_row(...) must come BEFORE the first "
        f"'{read_marker}' read (AGENTS.md §4.3). "
        f"append at byte {append_pos}, read at byte {read_pos}."
    )


@notebook_required
def test_notebook07_07E_appends_intent_row_before_reading_n05():
    """07E reads N05 official_pooled and per-seed prediction artifacts.
    Per AGENTS.md §4.3, the intent ledger row MUST be appended before any
    such read.
    """
    nb = load_notebook()
    cell = _find_cell_starting_with(nb, "RUN_07E_EXPLAINABILITY_APPENDIX")
    assert cell is not None, "Could not locate the 07E code cell"
    _assert_append_precedes_read(
        cell.source,
        read_marker='read_csv_required(n05_paths["official_pooled"])',
        phase_label="07E",
    )


@notebook_required
def test_notebook07_07G_appends_intent_row_before_reading_comparison():
    """07G reads the final comparison frame whose rows are derived from
    N05/N06 official-validation metrics. The intent ledger row MUST be
    appended before that read.
    """
    nb = load_notebook()
    cell = _find_cell_starting_with(nb, "RUN_07G_GAP_AUDIT_FOR_08X")
    assert cell is not None, "Could not locate the 07G code cell"
    _assert_append_precedes_read(
        cell.source,
        read_marker="pd.read_csv(comparison_path)",
        phase_label="07G",
    )


@notebook_required
def test_notebook07_07H_appends_intent_row_before_reading_comparison():
    """07H reads the final comparison frame whose rows are derived from
    N05/N06 official-validation metrics. The intent ledger row MUST be
    appended before that read.
    """
    nb = load_notebook()
    cell = _find_cell_starting_with(nb, "RUN_07H_PAPER_READY_SYNTHESIS")
    assert cell is not None, "Could not locate the 07H code cell"
    _assert_append_precedes_read(
        cell.source,
        read_marker="pd.read_csv(comparison_path)",
        phase_label="07H",
    )


@notebook_required
def test_notebook07_07H_paragraph_literals_do_not_leak_forbidden_phrases():
    """07H validates the emitted thesis kit at runtime, but this static gate
    catches forbidden wording in the generated paragraph literals before the
    notebook reaches Colab.
    """
    nb = load_notebook()
    cell = _find_cell_starting_with(nb, "RUN_07H_PAPER_READY_SYNTHESIS")
    assert cell is not None, "Could not locate the 07H code cell"
    tree = ast.parse(cell.source, filename="notebook07_07H_cell")
    paragraph_literals = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "results_paragraph",
                "robustness_paragraph",
                "limitation_paragraph",
            }:
                paragraph_literals[target.id] = node.value.value

    assert set(paragraph_literals) == {
        "results_paragraph",
        "robustness_paragraph",
        "limitation_paragraph",
    }
    leaked = {
        name: sorted({match.lower() for match in c.FORBIDDEN_PHRASE_REGEX_PATTERN.findall(text)})
        for name, text in paragraph_literals.items()
        if c.FORBIDDEN_PHRASE_REGEX_PATTERN.findall(text)
    }
    assert not leaked, f"07H paragraph literals leak forbidden phrases: {leaked}"


@notebook_required
def test_notebook07_07C_is_audit_only_not_direct_to_csv():
    """07C must NOT direct-write the ledger via ``ledger.to_csv(...)`` because
    that bypasses prefix-invariance and could overwrite project-level rows
    appended by N08/thesis downstream. 07C must use ``flush_ledger_to_disk()``
    (which runs prefix invariance) and then validate the disk frame.
    """
    nb = load_notebook()
    cell = _find_cell_starting_with(nb, "RUN_07C_VALIDATION_BUDGET_LEDGER")
    assert cell is not None, "Could not locate the 07C code cell"
    src = cell.source
    assert "ledger.to_csv" not in src, (
        "07C must not direct-write the ledger via ledger.to_csv(...); use flush_ledger_to_disk()"
    )
    assert "flush_ledger_to_disk()" in src, (
        "07C must call flush_ledger_to_disk() (which runs prefix invariance)"
    )
    assert "validate_ledger_frame(" in src, (
        "07C must still call validate_ledger_frame() on the on-disk frame"
    )
