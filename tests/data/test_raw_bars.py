"""Behavioral tests for ``intraday_research.data.raw_bars`` (N08 #5C-3).

Synthetic-CSV tests only. No raw bar I/O against real ``data/*.csv``
(those are gitignored), no fixture files committed to the repo, no
official validation, no holdout. Verifies the section 4 contract
documented in
``docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intraday_research.data.raw_bars import VAL_END, load_ticker_bars


def _synthetic_bar_frame(
    n: int = 10,
    start: str = "2010-01-04 09:30:00",
    drift_bps_per_bar: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """5-min OHLCV bars that pass baseline_v1._validated_ohlcv."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq="5min")
    per_bar_return = drift_bps_per_bar / 10_000.0
    noise = rng.standard_normal(n) * 1e-5
    close = 100.0 * np.cumprod(1.0 + per_bar_return + noise)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close * 0.9995,
        "high": close * 1.0005,
        "low": close * 0.9990,
        "close": close,
        "volume": rng.integers(1000, 10_000, n),
    })


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_single_ticker_happy_path_returns_canonical_columns(tmp_path: Path):
    """1 ticker, 10 rows → pooled DataFrame with canonical column order."""
    path = _write_csv(_synthetic_bar_frame(n=10), tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert list(pooled.columns) == [
        "ticker", "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert len(pooled) == 10
    assert (pooled["ticker"] == "CSCO").all()
    assert str(pooled["timestamp"].dtype) == "datetime64[ns]"
    assert pooled["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Task 2: multi-ticker pooling
# ---------------------------------------------------------------------------


def test_three_ticker_pooling_returns_pooled_frame(tmp_path: Path):
    manifest = {}
    for ticker in ("CSCO", "JPM", "KO"):
        path = _write_csv(_synthetic_bar_frame(n=5), tmp_path / f"{ticker}.csv")
        manifest[ticker] = path
    pooled = load_ticker_bars(manifest)
    assert len(pooled) == 15
    assert pooled.groupby("ticker").size().to_dict() == {
        "CSCO": 5, "JPM": 5, "KO": 5,
    }


def test_per_ticker_timestamps_remain_ascending_after_pooling(tmp_path: Path):
    manifest = {
        "CSCO": _write_csv(_synthetic_bar_frame(n=8), tmp_path / "CSCO.csv"),
        "JPM":  _write_csv(_synthetic_bar_frame(n=8), tmp_path / "JPM.csv"),
    }
    pooled = load_ticker_bars(manifest)
    for ticker, group in pooled.groupby("ticker"):
        assert group["timestamp"].is_monotonic_increasing, (
            f"timestamps for {ticker} not ascending"
        )


def test_pooled_sort_is_by_ticker_then_timestamp(tmp_path: Path):
    """Even if manifest order is reversed, output sort is (ticker, timestamp)."""
    manifest = {
        "WMT":  _write_csv(_synthetic_bar_frame(n=3), tmp_path / "WMT.csv"),
        "CSCO": _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv"),
    }
    pooled = load_ticker_bars(manifest)
    assert (pooled["ticker"].iloc[:3] == "CSCO").all()
    assert (pooled["ticker"].iloc[3:] == "WMT").all()


# ---------------------------------------------------------------------------
# Task 3: holdout closure (priority gate)
# ---------------------------------------------------------------------------


def test_row_at_exact_val_end_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=3, start="2017-01-24 15:50:00")
    frame.loc[2, "timestamp"] = pd.Timestamp("2017-01-25 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="holdout closure violated"):
        load_ticker_bars({"CSCO": path})


def test_holdout_error_message_carries_contamination_count(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5, start="2017-01-25 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "rows=5/5" in msg
    assert "first contaminated timestamp" in msg


def test_csv_entirely_before_val_end_passes(tmp_path: Path):
    frame = _synthetic_bar_frame(n=10, start="2016-12-01 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert len(pooled) == 10


def test_custom_val_end_string_overrides_default(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5, start="2018-06-01 09:30:00")
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path}, val_end="2020-01-02")
    assert len(pooled) == 5


def test_val_end_int_42_raises_type_error_before_file_io(tmp_path: Path):
    """Locks the guard against pd.Timestamp(42) silently parsing to epoch ns.

    val_end=42 must raise TypeError BEFORE any file I/O happens; the
    manifest path is intentionally not even created so a TypeError is
    the only way the test can pass.
    """
    nonexistent = tmp_path / "does_not_exist.csv"
    with pytest.raises(TypeError, match="val_end must be str or pd.Timestamp"):
        load_ticker_bars({"CSCO": nonexistent}, val_end=42)


# ---------------------------------------------------------------------------
# Task 4: OHLCV validation delegation
# ---------------------------------------------------------------------------


def test_high_less_than_low_raises_with_ticker_prefix(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "high"] = 50.0
    frame.loc[2, "low"] = 200.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "high must be >= low" in msg


def test_close_outside_high_low_range_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "close"] = frame.loc[2, "high"] + 10.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_non_positive_price_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "open"] = -1.0
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_negative_volume_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "volume"] = -100
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


# ---------------------------------------------------------------------------
# Task 5: schema + column normalization
# ---------------------------------------------------------------------------


def test_missing_volume_column_raises_with_missing_list(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5).drop(columns=["volume"])
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "missing columns" in msg
    assert "'volume'" in msg


def test_missing_timestamp_column_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5).drop(columns=["timestamp"])
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="'timestamp'"):
        load_ticker_bars({"CSCO": path})


def test_unparseable_timestamp_raises_nat_or_parse_error(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame["timestamp"] = "not a date"
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="ticker=CSCO"):
        load_ticker_bars({"CSCO": path})


def test_case_and_whitespace_mixed_headers_succeed(tmp_path: Path):
    """Headers like 'Timestamp', 'OPEN ', 'High' normalize via strip().lower()."""
    frame = _synthetic_bar_frame(n=5)
    frame.columns = ["Timestamp", "OPEN ", " High", "Low", "Close", "VOLUME"]
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert list(pooled.columns) == [
        "ticker", "timestamp", "open", "high", "low", "close", "volume",
    ]
    assert len(pooled) == 5


def test_duplicate_columns_after_normalization_raise(tmp_path: Path):
    """If 'Open' and 'open' both appear, they collide on .lower()."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2010-01-04 09:30", periods=3, freq="5min"),
        "Open": [100.0, 100.5, 101.0],
        "open": [99.0, 99.5, 100.0],
        "high": [101.0, 101.5, 102.0],
        "low":  [98.0, 98.5, 99.0],
        "close": [100.0, 100.5, 101.0],
        "volume": [1000, 2000, 3000],
    })
    path = tmp_path / "CSCO.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate column names after normalization"):
        load_ticker_bars({"CSCO": path})


# ---------------------------------------------------------------------------
# Task 6: file / IO + ParserError wrapping
# ---------------------------------------------------------------------------


def test_nonexistent_path_raises_file_not_found_with_ticker_context(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_ticker_bars({"CSCO": missing})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert str(missing) in msg


def test_zero_data_row_csv_raises(tmp_path: Path):
    path = tmp_path / "CSCO.csv"
    path.write_text("timestamp,open,high,low,close,volume\n")
    with pytest.raises(ValueError, match="zero data rows"):
        load_ticker_bars({"CSCO": path})


def test_garbled_csv_wraps_as_value_error_with_chained_cause(tmp_path: Path):
    """ParserError / UnicodeDecodeError → ValueError with __cause__ set."""
    path = tmp_path / "CSCO.csv"
    path.write_bytes(
        b'timestamp,open,high,low,close,volume\n'
        b'"unterminated quote,1,2,3,4,5\n'
        b'\x80\x81\x82,more,garbage,1,2,3\n'
    )
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "CSV parse failed" in msg
    assert excinfo.value.__cause__ is not None
    assert isinstance(
        excinfo.value.__cause__,
        (pd.errors.ParserError, UnicodeDecodeError),
    )


# ---------------------------------------------------------------------------
# Task 7: manifest guards + intra-ticker dup-timestamp
# ---------------------------------------------------------------------------


def test_empty_manifest_raises():
    with pytest.raises(ValueError, match="manifest is empty"):
        load_ticker_bars({})


def test_empty_ticker_key_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "x.csv")
    with pytest.raises(ValueError, match="manifest ticker key empty"):
        load_ticker_bars({"": path})


def test_duplicate_ticker_after_strip_raises(tmp_path: Path):
    p1 = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "a.csv")
    p2 = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "b.csv")
    with pytest.raises(ValueError, match="duplicate ticker after normalization"):
        load_ticker_bars({"CSCO": p1, " CSCO ": p2})


def test_non_str_or_path_manifest_value_raises_type_error():
    with pytest.raises(TypeError, match="must be str or Path"):
        load_ticker_bars({"CSCO": 42})


def test_duplicate_timestamp_within_ticker_raises(tmp_path: Path):
    frame = _synthetic_bar_frame(n=5)
    frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
    path = _write_csv(frame, tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="duplicate timestamp within ticker"):
        load_ticker_bars({"CSCO": path})


# ---------------------------------------------------------------------------
# Task 8: timezone semantics (P2 #4)
# ---------------------------------------------------------------------------


def test_tz_naive_csv_passes(tmp_path: Path):
    """All timestamps are tz-naive (the normal case)."""
    path = _write_csv(_synthetic_bar_frame(n=5), tmp_path / "CSCO.csv")
    pooled = load_ticker_bars({"CSCO": path})
    assert pooled["timestamp"].dt.tz is None


def test_tz_aware_csv_timestamp_raises(tmp_path: Path):
    """Timestamps with embedded offset (e.g. -05:00) must be rejected."""
    df = pd.DataFrame({
        "timestamp": [
            "2010-06-01 09:30:00-05:00",
            "2010-06-01 09:35:00-05:00",
            "2010-06-01 09:40:00-05:00",
        ],
        "open":   [100.0, 100.5, 101.0],
        "high":   [101.0, 101.5, 102.0],
        "low":    [99.0,  99.5,  100.0],
        "close":  [100.5, 101.0, 101.5],
        "volume": [1000, 2000, 3000],
    })
    path = tmp_path / "CSCO.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="timestamp is tz-aware"):
        load_ticker_bars({"CSCO": path})


def test_tz_aware_val_end_string_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="val_end must be timezone-naive"):
        load_ticker_bars(
            {"CSCO": path},
            val_end="2017-01-25 00:00:00-05:00",
        )


def test_tz_aware_val_end_timestamp_raises(tmp_path: Path):
    path = _write_csv(_synthetic_bar_frame(n=3), tmp_path / "CSCO.csv")
    with pytest.raises(ValueError, match="val_end must be timezone-naive"):
        load_ticker_bars(
            {"CSCO": path},
            val_end=pd.Timestamp("2017-01-25", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# Codex #5C-3 review fixes: val_end string parse-failure + zero-byte CSV
# ---------------------------------------------------------------------------


def test_val_end_unparseable_string_raises_with_context_before_file_io(
    tmp_path: Path,
):
    """val_end='not a date' must raise a wrapped ValueError BEFORE any
    file I/O. Locked via a nonexistent manifest path — a FileNotFoundError
    would mean the val_end guard ran too late."""
    nonexistent = tmp_path / "does_not_exist.csv"
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": nonexistent}, val_end="not a date")
    msg = str(excinfo.value)
    assert "val_end must parse as pd.Timestamp" in msg
    assert "'not a date'" in msg
    # The underlying pandas/dateutil parse error must be chained.
    assert excinfo.value.__cause__ is not None


def test_zero_byte_csv_raises_with_ticker_and_path_context(tmp_path: Path):
    """A truly empty file raises pd.errors.EmptyDataError (NOT ParserError).
    Loader must catch it explicitly and wrap with ticker + path context."""
    path = tmp_path / "CSCO.csv"
    path.write_text("")  # zero bytes, not even a header
    with pytest.raises(ValueError) as excinfo:
        load_ticker_bars({"CSCO": path})
    msg = str(excinfo.value)
    assert "ticker=CSCO" in msg
    assert "CSV parse failed" in msg
    assert str(path) in msg
    assert isinstance(excinfo.value.__cause__, pd.errors.EmptyDataError)
