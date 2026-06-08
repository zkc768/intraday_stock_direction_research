"""Tests for the #5F-3 1-minute .txt -> 5-minute loader in data/raw_bars.py.

All synthetic: tiny .txt files written to tmp_path; no real ticker file, no Drive.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from intraday_research.data import raw_bars
from intraday_research.data.raw_bars import (
    VAL_END,
    load_one_minute_txt,
    load_ticker_bars,
    load_ticker_bars_txt,
    resample_to_five_minutes,
)


# --------------------------------------------------------------------------
# synthetic fixtures
# --------------------------------------------------------------------------

def _bar(base: float):
    """Return (open, high, low, close, volume) for a valid OHLCV bar."""
    return (base, base + 0.5, base - 0.5, base + 0.2, 1000.0)


def _day_rows(date: str, n_minutes: int, *, start="09:30", base=100.0):
    """n_minutes consecutive 1-min rows from `start` on `date`."""
    h0, m0 = (int(x) for x in start.split(":"))
    rows = []
    for i in range(n_minutes):
        total = h0 * 60 + m0 + i
        hh, mm = divmod(total, 60)
        o, hi, lo, c, v = _bar(base + i * 0.1)
        rows.append((date, f"{hh:02d}:{mm:02d}", o, hi, lo, c, v))
    return rows


def _write_txt(path, rows, *, header=False):
    lines = ["Date,Time,Open,High,Low,Close,Volume"] if header else []
    for (date, time, o, hi, lo, c, v) in rows:
        lines.append(f"{date},{time},{o},{hi},{lo},{c},{v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _one_min_frame(rows):
    """Build a [timestamp, ohlcv] 1-min frame directly (for resample tests)."""
    recs = []
    for (date, time, o, hi, lo, c, v) in rows:
        ts = pd.to_datetime(f"{date} {time}", format="%m/%d/%Y %H:%M")
        recs.append({"timestamp": ts, "open": o, "high": hi, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(recs)


def _reference_resample(frame):
    """Independent re-encoding of the Stage-0 5-min recipe (the parity lock)."""
    r = (
        frame.set_index("timestamp")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .reset_index()
    )
    t = r["timestamp"].dt.time
    return r.loc[
        (t >= dt.time(9, 30)) & (t <= dt.time(16, 0)),
        ["timestamp", "open", "high", "low", "close", "volume"],
    ].reset_index(drop=True)


# --------------------------------------------------------------------------
# load_one_minute_txt
# --------------------------------------------------------------------------

def test_load_one_minute_txt_happy(tmp_path):
    p = _write_txt(tmp_path / "X.txt", _day_rows("01/02/2015", 10))
    frame = load_one_minute_txt(p)
    assert list(frame.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(frame) == 10
    assert frame["timestamp"].is_monotonic_increasing


def test_load_one_minute_txt_header_tolerated(tmp_path):
    p = _write_txt(tmp_path / "X.txt", _day_rows("01/02/2015", 6), header=True)
    frame = load_one_minute_txt(p)
    assert len(frame) == 6


def test_load_one_minute_txt_rth_filter(tmp_path):
    # rows at 09:29 (pre-open) and 16:01 (post-close) must be dropped; edges kept.
    rows = (
        [("01/02/2015", "09:29", *_bar(100.0))]
        + _day_rows("01/02/2015", 1, start="09:30")
        + _day_rows("01/02/2015", 1, start="16:00")
        + [("01/02/2015", "16:01", *_bar(101.0))]
    )
    p = _write_txt(tmp_path / "X.txt", rows)
    frame = load_one_minute_txt(p)
    times = frame["timestamp"].dt.time
    assert times.min() == dt.time(9, 30)
    assert times.max() == dt.time(16, 0)
    assert len(frame) == 2


def test_load_one_minute_txt_post_val_end_capped(tmp_path):
    # rows spanning VAL_END: only pre-VAL_END survive; no raise on later rows.
    rows = _day_rows("01/23/2017", 5) + _day_rows("01/26/2017", 5)  # VAL_END = 2017-01-25
    p = _write_txt(tmp_path / "X.txt", rows)
    frame = load_one_minute_txt(p)
    assert (frame["timestamp"] < VAL_END).all()
    assert len(frame) == 5


def test_load_one_minute_txt_all_post_val_end_raises(tmp_path):
    p = _write_txt(tmp_path / "X.txt", _day_rows("01/26/2017", 5))
    with pytest.raises(ValueError, match="no pre-holdout"):
        load_one_minute_txt(p)


def test_load_one_minute_txt_bad_timestamp_raises(tmp_path):
    p = _write_txt(tmp_path / "X.txt", [("not-a-date", "09:30", *_bar(100.0))])
    with pytest.raises(ValueError, match="timestamp parse failed"):
        load_one_minute_txt(p)


def test_load_one_minute_txt_duplicate_timestamp_raises(tmp_path):
    # Codex impl review P1-2: duplicate 1-min timestamps must fail BEFORE resample
    # (resample would silently merge them).
    rows = _day_rows("01/02/2015", 1) + _day_rows("01/02/2015", 1)  # 09:30 twice
    p = _write_txt(tmp_path / "X.txt", rows)
    with pytest.raises(ValueError, match="duplicate one-minute timestamp"):
        load_one_minute_txt(p)


def test_load_one_minute_txt_nan_volume_raises(tmp_path):
    # Codex impl review P1-1: a missing 1-min volume (NaN) must fail loud, not be
    # silently undercounted by the 5-min sum.
    p = tmp_path / "X.txt"
    p.write_text("01/02/2015,09:30,100,100.5,99.5,100.2,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_one_minute_txt(p)


# --------------------------------------------------------------------------
# resample_to_five_minutes
# --------------------------------------------------------------------------

def test_resample_aggregation_exact():
    rows = _day_rows("01/02/2015", 10, start="09:30")  # two 5-min blocks
    one_min = _one_min_frame(rows)
    five = resample_to_five_minutes(one_min)
    assert len(five) == 2
    first = five.iloc[0]
    block0 = one_min.iloc[0:5]
    assert first["open"] == block0["open"].iloc[0]
    assert first["high"] == block0["high"].max()
    assert first["low"] == block0["low"].min()
    assert first["close"] == block0["close"].iloc[-1]
    assert first["volume"] == block0["volume"].sum()
    assert first["timestamp"] == pd.Timestamp("2015-01-02 09:30:00")


def test_resample_recipe_parity_lock():
    rows = _day_rows("01/02/2015", 23)
    one_min = _one_min_frame(rows)
    pd.testing.assert_frame_equal(
        resample_to_five_minutes(one_min), _reference_resample(one_min)
    )


def test_resample_missing_column_raises():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-02 09:30"]), "open": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        resample_to_five_minutes(df)


# --------------------------------------------------------------------------
# load_ticker_bars_txt (pooled)
# --------------------------------------------------------------------------

def test_load_ticker_bars_txt_schema_matches_csv(tmp_path):
    a = _write_txt(tmp_path / "AAA.txt", _day_rows("01/02/2015", 10, base=100.0))
    b = _write_txt(tmp_path / "BBB.txt", _day_rows("01/02/2015", 10, base=50.0))
    pooled = load_ticker_bars_txt({"AAA": a, "BBB": b})
    assert list(pooled.columns) == list(raw_bars._CANONICAL_COLUMN_ORDER)
    assert set(pooled["ticker"]) == {"AAA", "BBB"}
    # sorted by (ticker, timestamp)
    assert pooled.equals(
        pooled.sort_values(["ticker", "timestamp"], kind="stable").reset_index(drop=True)
    )
    # no holdout rows
    assert (pooled["timestamp"] < VAL_END).all()


def test_load_ticker_bars_txt_holdout_postcondition(tmp_path):
    a = _write_txt(tmp_path / "AAA.txt", _day_rows("01/23/2017", 10) + _day_rows("01/26/2017", 10))
    pooled = load_ticker_bars_txt({"AAA": a})
    assert (pooled["timestamp"] < VAL_END).all()


def test_load_ticker_bars_txt_empty_manifest_raises():
    with pytest.raises(ValueError, match="manifest is empty"):
        load_ticker_bars_txt({})


def test_load_ticker_bars_txt_same_columns_as_csv_loader(tmp_path):
    # Build a 5-min CSV the CSV loader accepts, and a .txt the txt loader accepts;
    # both must yield identical canonical column order/dtypes.
    txt = _write_txt(tmp_path / "AAA.txt", _day_rows("01/02/2015", 10))
    txt_frame = load_ticker_bars_txt({"AAA": txt})
    csv_path = tmp_path / "AAA.csv"
    txt_frame.drop(columns=["ticker"]).to_csv(csv_path, index=False)
    csv_frame = load_ticker_bars({"AAA": csv_path})
    assert list(txt_frame.columns) == list(csv_frame.columns)
    assert [str(d) for d in txt_frame.dtypes] == [str(d) for d in csv_frame.dtypes]
