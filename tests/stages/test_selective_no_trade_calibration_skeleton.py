"""Skeleton test for stage selective_no_trade_calibration."""
import pytest

from intraday_research.stages import selective_no_trade_calibration as m


def test_stage_constants():
    assert m.STAGE_NAME == "selective_no_trade_calibration"
    assert isinstance(m.REQUIRED_ARTIFACTS, tuple)
    assert m.REQUIRED_ARTIFACTS == ()


def test_run_stage_not_implemented():
    with pytest.raises(NotImplementedError) as exc:
        m.run_stage(None)
    assert "selective_no_trade_calibration" in str(exc.value)
