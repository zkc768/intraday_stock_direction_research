"""Train-inner windowed-index builder for the 08X harness (#5F-4).

Chains the existing #5C data primitives into the windowed sample index the 08X
fold builder consumes:

    raw 5-min frame
      -> per ticker: build_features / build_no_trade_band_labels /
         apply_stage0_chronological_split
      -> pool (row-aligned concat)
      -> build_windows

Pure over a 5-minute frame: no config, no I/O, no Drive. The chronological
split + label-horizon invalidation + per-ticker / per-day window construction are
owned by the upstream primitives; this module only wires them and composes the
validity masks (target_valid = label_valid AND split_valid -- Codex #5F-4 Q4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intraday_research.data.features import build_features
from intraday_research.data.labels import build_no_trade_band_labels
from intraday_research.data.splits import apply_stage0_chronological_split
from intraday_research.data.windows import build_windows


_REQUIRED_FRAME_COLUMNS: frozenset[str] = frozenset(
    {"ticker", "timestamp", "open", "high", "low", "close", "volume"}
)


def _same_day_horizon_mask(timestamps: np.ndarray, horizon_k: int) -> np.ndarray:
    """True at row ``i`` iff ``i + horizon_k`` exists AND is the same trading day.

    Forbids a forward label horizon crossing a trading-day boundary (AGENTS.md
    4.1.5). ``build_windows`` guards the feature-window side but not the label
    side, and the chronological split only invalidates cross-PARTITION horizons,
    so the same-day horizon mask is applied here (per ticker, before pooling).
    """
    n = len(timestamps)
    mask = np.zeros(n, dtype=np.bool_)
    if n > horizon_k:
        dates = np.asarray(timestamps).astype("datetime64[D]")
        mask[: n - horizon_k] = dates[: n - horizon_k] == dates[horizon_k:]
    return mask


def build_train_inner_windowed_index(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    horizon_k: int,
    threshold_bps: float,
    window_size: int,
) -> dict[str, np.ndarray]:
    """Build the pooled windowed sample index from a 5-minute frame.

    Args:
        frame: pooled 5-minute bars with columns ``{ticker, timestamp, open,
            high, low, close, volume}``, ``timestamp`` tz-naive ``datetime64[ns]``,
            sorted by ``(ticker, timestamp)`` (as returned by
            ``raw_bars.load_ticker_bars`` / ``load_ticker_bars_txt``).
        feature_set: one of ``data.features.FEATURE_SETS``.
        horizon_k: frozen label horizon (must equal the fold-purge horizon).
        threshold_bps: frozen no-trade-band half-width.
        window_size: frozen sliding-window length.

    Returns:
        The ``build_windows`` dict: ``X, y, target_partition, target_timestamps,
        target_row_positions, target_ticker_ids``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be pd.DataFrame; got {type(frame).__name__}")
    missing = sorted(_REQUIRED_FRAME_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
    if len(frame) == 0:
        raise ValueError("frame is empty; cannot build a windowed index")

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    partition_blocks: list[np.ndarray] = []
    feature_valid_blocks: list[np.ndarray] = []
    target_valid_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    ticker_blocks: list[np.ndarray] = []

    # frame is pre-sorted by (ticker, timestamp); group in first-appearance order
    # and re-sort each group defensively so per-ticker arrays are chronological.
    for _ticker, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("timestamp", kind="stable")
        timestamps = group["timestamp"].to_numpy()
        close = group["close"].to_numpy(dtype=float)

        features, feature_valid = build_features(group, feature_set=feature_set)
        labels, label_valid = build_no_trade_band_labels(
            close, timestamps, horizon_k=horizon_k, threshold_bps=threshold_bps
        )
        partition, split_valid = apply_stage0_chronological_split(
            timestamps, horizon_k=horizon_k
        )
        same_day_horizon_valid = _same_day_horizon_mask(timestamps, horizon_k)
        # Codex #5F-4 Q4 + impl-review P0: a target row is valid only when its
        # label is computable (label_valid), its label horizon stays in the same
        # partition (split_valid), AND its label horizon stays within the same
        # trading day (same_day_horizon_valid -- AGENTS 4.1.5; build_windows only
        # guards the feature-window side, not the forward label horizon). Feature
        # validity stays a separate mask.
        target_valid = label_valid & split_valid & same_day_horizon_valid

        feature_blocks.append(features)
        label_blocks.append(labels)
        partition_blocks.append(partition)
        feature_valid_blocks.append(feature_valid)
        target_valid_blocks.append(target_valid)
        timestamp_blocks.append(timestamps)
        ticker_blocks.append(group["ticker"].to_numpy())

    features = np.concatenate(feature_blocks, axis=0)
    labels = np.concatenate(label_blocks)
    partition = np.concatenate(partition_blocks)
    feature_valid_mask = np.concatenate(feature_valid_blocks)
    target_valid_mask = np.concatenate(target_valid_blocks)
    timestamps = np.concatenate(timestamp_blocks)
    ticker_ids = np.concatenate(ticker_blocks)

    return build_windows(
        features,
        labels,
        timestamps,
        ticker_ids,
        partition=partition,
        feature_valid_mask=feature_valid_mask,
        target_valid_mask=target_valid_mask,
        window_size=window_size,
    )
