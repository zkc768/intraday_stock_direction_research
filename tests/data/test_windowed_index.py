"""Tests for the #5F-4 train-inner windowed-index chain (synthetic 5-min frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intraday_research.data.features import FEATURE_SETS
from intraday_research.data.splits import PARTITION_TRAIN, PARTITION_VALIDATION
from intraday_research.data.windowed_index import build_train_inner_windowed_index


def _five_min_frame(
    tickers=("AAA", "BBB"),
    days=("2013-09-12", "2013-09-13", "2013-09-16", "2013-09-17"),
    bars_per_day=10,
):
    """Multi-ticker 5-min frame straddling VALIDATION_START (2013-09-16):
    2013-09-12/13 are train (< boundary), 2013-09-16/17 are validation."""
    recs = []
    for ticker in tickers:
        base = 100.0 if ticker == "AAA" else 50.0
        for day in days:
            for i in range(bars_per_day):
                total = 9 * 60 + 30 + i * 5
                hh, mm = divmod(total, 60)
                ts = pd.Timestamp(f"{day} {hh:02d}:{mm:02d}:00")
                o = base + i * 0.1
                recs.append(
                    {
                        "ticker": ticker,
                        "timestamp": ts,
                        "open": o,
                        "high": o + 0.5,
                        "low": o - 0.5,
                        "close": o + 0.2,
                        "volume": 1000.0 + i,
                    }
                )
    frame = pd.DataFrame(recs).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    return frame


def _build(frame, *, feature_set="price_action_core", horizon_k=3, threshold_bps=1.5, window_size=3):
    return build_train_inner_windowed_index(
        frame,
        feature_set=feature_set,
        horizon_k=horizon_k,
        threshold_bps=threshold_bps,
        window_size=window_size,
    )


def test_build_windowed_index_shapes_and_partitions():
    wi = _build(_five_min_frame())
    assert set(wi.keys()) >= {
        "X", "y", "target_partition", "target_timestamps", "target_ticker_ids",
    }
    assert wi["X"].ndim == 3
    assert wi["X"].shape[1] == 3  # window_size
    assert wi["X"].shape[2] == len(FEATURE_SETS["price_action_core"])  # F == 3
    parts = {int(p) for p in np.unique(wi["target_partition"])}
    assert int(PARTITION_TRAIN) in parts
    assert int(PARTITION_VALIDATION) in parts


def test_build_windowed_index_only_valid_labels():
    # invalid-label rows (-1) must never become window targets
    wi = _build(_five_min_frame())
    assert {int(v) for v in np.unique(wi["y"])}.issubset({0, 1})


def test_build_windowed_index_train_rows_present():
    wi = _build(_five_min_frame())
    assert int((wi["target_partition"] == PARTITION_TRAIN).sum()) > 0


def test_build_windowed_index_window_size_respected():
    wi = _build(_five_min_frame(), window_size=4)
    assert wi["X"].shape[1] == 4


def test_build_windowed_index_feature_count_matches_set():
    wi = _build(_five_min_frame(), feature_set="price_volume_time")
    assert wi["X"].shape[2] == len(FEATURE_SETS["price_volume_time"])


def test_build_windowed_index_excludes_cross_day_label_horizon():
    # Codex impl review P0: the last horizon_k bars of a trading day must NOT be
    # targets -- their forward label horizon would reach into the next day.
    frame = _five_min_frame(
        tickers=("AAA",), days=("2013-09-12", "2013-09-13"), bars_per_day=6
    )
    wi = _build(frame, horizon_k=2, window_size=2)
    target_ts = {pd.Timestamp(t) for t in wi["target_timestamps"]}
    # 2013-09-12 bars span 09:30..09:55; the last 2 (09:50, 09:55) would cross day
    assert pd.Timestamp("2013-09-12 09:50:00") not in target_ts
    assert pd.Timestamp("2013-09-12 09:55:00") not in target_ts
    # sanity: an earlier same-day target is still present (mask not over-excluding)
    assert pd.Timestamp("2013-09-12 09:45:00") in target_ts


def test_build_windowed_index_empty_frame_raises():
    with pytest.raises(ValueError, match="empty"):
        _build(_five_min_frame().iloc[0:0])


def test_build_windowed_index_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        _build(_five_min_frame().drop(columns=["volume"]))
