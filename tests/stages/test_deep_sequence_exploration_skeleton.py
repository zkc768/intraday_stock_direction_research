"""Skeleton test for stage deep_sequence_exploration (N08 ledger writer)."""
import pytest

from intraday_research.stages import deep_sequence_exploration as m


def test_stage_constants():
    assert m.STAGE_NAME == "deep_sequence_exploration"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert "notebook07_validation_budget_ledger.csv" in m.REQUIRED_ARTIFACTS


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    msg = str(exc.value)
    assert "deep_sequence_exploration" in msg
    assert "notebook07_validation_budget_ledger.csv" in msg
