"""Skeleton test for stage config_screening."""
import pytest

from intraday_research.stages import config_screening as m


def test_stage_constants():
    assert m.STAGE_NAME == "config_screening"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert m.REQUIRED_ARTIFACTS == ()


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    assert "config_screening" in str(exc.value)
