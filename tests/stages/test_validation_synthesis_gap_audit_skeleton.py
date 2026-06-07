"""Skeleton test for stage validation_synthesis_gap_audit (N07 ledger writer)."""
import pytest

from intraday_research.stages import validation_synthesis_gap_audit as m


def test_stage_constants():
    assert m.STAGE_NAME == "validation_synthesis_gap_audit"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert "notebook07_validation_budget_ledger.csv" in m.REQUIRED_ARTIFACTS


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    msg = str(exc.value)
    assert "validation_synthesis_gap_audit" in msg
    # N07/N08 skeleton must surface the ledger artifact in the error path so
    # operators see the required dependency before any body migration.
    assert "notebook07_validation_budget_ledger.csv" in msg
