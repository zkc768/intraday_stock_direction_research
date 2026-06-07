"""Skeleton test for stage controlled_followup."""
import pytest

from intraday_research.stages import controlled_followup as m


def test_stage_constants():
    assert m.STAGE_NAME == "controlled_followup"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert m.REQUIRED_ARTIFACTS == ()


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    assert "controlled_followup" in str(exc.value)
