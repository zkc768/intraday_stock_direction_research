"""Skeleton test for stage lightgbm_tuning."""
import pytest

from intraday_research.stages import lightgbm_tuning as m


def test_stage_constants():
    assert m.STAGE_NAME == "lightgbm_tuning"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert m.REQUIRED_ARTIFACTS == ()


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    assert "lightgbm_tuning" in str(exc.value)
