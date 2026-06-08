"""5-minute pre-aggregated raw-bar CSV loader for the N08 #5C pipeline.

See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from intraday_research.baseline_v1 import _validated_ohlcv


VAL_END: pd.Timestamp = pd.Timestamp("2017-01-25")
# Bars at or after VAL_END belong to the closed holdout/test partition
# (docs/CONFIG_SCREENING_FREEZE_2026-06-04.md). The loader REFUSES to
# silently cap or drop; it raises ValueError so any contamination is
# loud, not hidden in dropped rows.

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)
_CANONICAL_COLUMN_ORDER: tuple[str, ...] = (
    "ticker", "timestamp", "open", "high", "low", "close", "volume",
)

# ---- #5F-3: 1-minute .txt source -> 5-minute frame (mirrors Stage 0) ----------
# These replicate the config_screening (Stage 0) loader recipe verbatim so the
# raw->5min rules do not drift between stages (AGENTS.md section 5).
MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()
RAW_TXT_COLUMNS: tuple[str, ...] = (
    "Date", "Time", "Open", "High", "Low", "Close", "Volume",
)
_FIVE_MIN_COLUMNS: tuple[str, ...] = (
    "timestamp", "open", "high", "low", "close", "volume",
)
_TXT_RENAME: tuple[tuple[str, str], ...] = (
    ("Open", "open"), ("High", "high"), ("Low", "low"),
    ("Close", "close"), ("Volume", "volume"),
)


def load_ticker_bars(
    manifest: Mapping[str, str | Path],
    *,
    val_end: str | pd.Timestamp = VAL_END,
) -> pd.DataFrame:
    """Return pooled pre-holdout 5-minute bars across all manifest tickers.

    Output columns (exactly, in this order):
        ["ticker", "timestamp", "open", "high", "low", "close", "volume"]

    Output sort: ascending by ``(ticker, timestamp)``.

    Type discipline:
        - ``val_end`` MUST be ``str`` or ``pd.Timestamp``; any other
          type (including ``int``, ``float``, ``datetime.date``) raises
          ``TypeError`` before any file I/O. This closes the
          ``pd.Timestamp(42)`` epoch-ns silent-parse hole.
        - All timestamps (CSV column and ``val_end``) MUST be
          timezone-naive; tz-aware timestamps raise ``ValueError``
          rather than being implicitly converted.

    Raises:
        See docs/superpowers/specs/2026-06-07-n08-data-raw-bars-csv-loader-design.md
        section "Error handling" for the full enumeration of 14 fail-loud modes.
    """
    # ---- Pre-loop A: manifest guards ----
    if not manifest:
        raise ValueError("manifest is empty; cannot load 0 tickers")
    normalized_manifest: list[tuple[str, Path]] = []
    seen_normalized: set[str] = set()
    for raw_ticker, raw_path in manifest.items():
        ticker = str(raw_ticker).strip()
        if not ticker:
            raise ValueError("manifest ticker key empty")
        if ticker in seen_normalized:
            raise ValueError(
                f"duplicate ticker after normalization: {ticker!r}"
            )
        seen_normalized.add(ticker)
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(
                f"manifest[{raw_ticker!r}] must be str or Path; "
                f"got {type(raw_path).__name__}"
            )
        normalized_manifest.append((ticker, Path(raw_path)))

    # ---- Pre-loop B: val_end normalization ----
    if isinstance(val_end, str):
        try:
            val_end_ts = pd.Timestamp(val_end)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"val_end must parse as pd.Timestamp; got {val_end!r}"
            ) from exc
    elif isinstance(val_end, pd.Timestamp):
        val_end_ts = val_end
    else:
        raise TypeError(
            f"val_end must be str or pd.Timestamp; "
            f"got {type(val_end).__name__}"
        )
    if val_end_ts.tzinfo is not None:
        raise ValueError(
            f"val_end must be timezone-naive; got tz={val_end_ts.tzinfo}"
        )

    # ---- Per-ticker loop ----
    frames: list[pd.DataFrame] = []
    for ticker, path in normalized_manifest:
        # Pre-step: file existence check
        if not path.exists():
            raise FileNotFoundError(f"ticker={ticker} path={path}")

        # Step 1: read CSV with ParserError / EmptyDataError / decode wrap.
        # EmptyDataError is NOT a subclass of ParserError — a zero-byte file
        # raises it separately, so it must be caught explicitly to keep the
        # ticker + path context on the wrapped ValueError.
        try:
            df = pd.read_csv(path)
        except (
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
        ) as exc:
            raise ValueError(
                f"ticker={ticker}: CSV parse failed path={path}"
            ) from exc

        # Step 2: normalize column names + dedup check
        normalized_cols = [str(c).strip().lower() for c in df.columns]
        col_counts: dict[str, int] = {}
        for c in normalized_cols:
            col_counts[c] = col_counts.get(c, 0) + 1
        dup_cols = sorted(c for c, count in col_counts.items() if count > 1)
        if dup_cols:
            raise ValueError(
                f"ticker={ticker}: duplicate column names "
                f"after normalization: {dup_cols}"
            )
        df.columns = normalized_cols

        # Empty CSV check (header-only)
        if len(df) == 0:
            raise ValueError(f"ticker={ticker}: CSV has zero data rows")

        # Step 3: assert required columns present
        missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(
                f"ticker={ticker}: CSV missing columns: {missing}"
            )

        # Step 4: inject ticker column
        df["ticker"] = ticker

        # Step 5: parse timestamp + NaT + tz-aware checks.
        # Suppress pandas's "Could not infer format" UserWarning so it does
        # not get promoted to an error by pytest.ini's
        # `filterwarnings = error::Warning:intraday_research\..*` (pandas
        # uses stacklevel to attribute the warning to the calling module,
        # which is intraday_research.data.raw_bars). The real parse
        # failures still raise ValueError below.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="raise"
                )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"ticker={ticker}: timestamp parse failed: {exc}"
            ) from exc
        if df["timestamp"].isna().any():
            raise ValueError(
                f"ticker={ticker}: timestamp contains NaT"
            )
        if df["timestamp"].dt.tz is not None:
            raise ValueError(
                f"ticker={ticker}: timestamp is tz-aware "
                f"(tz={df['timestamp'].dt.tz}); strip timezone before loading"
            )

        # Step 6: sort within ticker by timestamp
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

        # Step 7: HOLDOUT CLOSURE CHECK (highest priority data check)
        contam_mask = df["timestamp"] >= val_end_ts
        if contam_mask.any():
            n_contam = int(contam_mask.sum())
            n_total = len(df)
            first_bad = df.loc[contam_mask, "timestamp"].iloc[0]
            raise ValueError(
                f"ticker={ticker}: holdout closure violated; "
                f"first contaminated timestamp={first_bad}; "
                f"rows={n_contam}/{n_total}"
            )

        # Step 8: intra-ticker timestamp uniqueness
        dup_ts_mask = df["timestamp"].duplicated()
        if dup_ts_mask.any():
            first_dup = df.loc[dup_ts_mask, "timestamp"].iloc[0]
            raise ValueError(
                f"ticker={ticker}: duplicate timestamp within ticker; "
                f"first duplicate={first_dup}"
            )

        # Step 9: OHLCV sanity + canonical column selection
        try:
            _validated_ohlcv(df)
        except ValueError as exc:
            raise ValueError(f"ticker={ticker}: {exc}") from exc
        df = df[list(_CANONICAL_COLUMN_ORDER)]
        frames.append(df)

    # ---- Post-loop: concat + sort + return ----
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(
        ["ticker", "timestamp"], kind="stable",
    ).reset_index(drop=True)
    return pooled


def _normalize_val_end(val_end: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize ``val_end`` to a tz-naive ``pd.Timestamp`` (fail-loud).

    Mirrors the inline normalization in ``load_ticker_bars`` (kept separate so the
    CSV loader stays byte-for-byte unchanged, #5F-3).
    """
    if isinstance(val_end, str):
        try:
            val_end_ts = pd.Timestamp(val_end)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"val_end must parse as pd.Timestamp; got {val_end!r}"
            ) from exc
    elif isinstance(val_end, pd.Timestamp):
        val_end_ts = val_end
    else:
        raise TypeError(
            f"val_end must be str or pd.Timestamp; got {type(val_end).__name__}"
        )
    if val_end_ts.tzinfo is not None:
        raise ValueError(f"val_end must be timezone-naive; got tz={val_end_ts.tzinfo}")
    return val_end_ts


def load_one_minute_txt(
    path: str | Path, *, val_end: str | pd.Timestamp = VAL_END
) -> pd.DataFrame:
    """Read ONE 1-minute ``.txt`` and return its pre-holdout RTH 1-minute frame.

    Output columns (exactly, in order): ``_FIVE_MIN_COLUMNS`` =
    ``[timestamp, open, high, low, close, volume]`` (single ticker; no ticker
    column). The ``.txt`` schema is ``RAW_TXT_COLUMNS`` (``Date,Time,O,H,L,C,V``,
    ``Date`` as ``%m/%d/%Y``, ``Time`` as ``%H:%M``); a literal ``Date`` header
    row is tolerated.

    Rows are CAPPED to ``timestamp < val_end`` and to regular trading hours
    (``MARKET_OPEN <= time <= MARKET_CLOSE``, inclusive) -- a raw provider file
    MAY contain later rows; they are dropped here, not treated as an error
    (#5F-3 / Codex Q2). Fail-loud on parse / NaT / tz-aware / missing columns /
    zero surviving rows.
    """
    val_end_ts = _normalize_val_end(val_end)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"txt path={path}")
    try:
        raw = pd.read_csv(
            path, header=None, names=list(RAW_TXT_COLUMNS), low_memory=False
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"CSV parse failed path={path}") from exc
    # Drop a literal header row if present.
    raw = raw.loc[
        raw["Date"].astype(str).str.strip().str.lower() != "date"
    ].reset_index(drop=True)
    if len(raw) == 0:
        raise ValueError(f"txt has zero data rows path={path}")
    try:
        timestamp = pd.to_datetime(
            raw["Date"].astype(str) + " " + raw["Time"].astype(str),
            format="%m/%d/%Y %H:%M",
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"timestamp parse failed path={path}: {exc}") from exc
    if timestamp.isna().any():
        raise ValueError(f"timestamp contains NaT path={path}")
    if timestamp.dt.tz is not None:
        raise ValueError(
            f"timestamp is tz-aware (tz={timestamp.dt.tz}) path={path}; strip tz"
        )
    frame = pd.DataFrame({"timestamp": timestamp})
    for src, dst in _TXT_RENAME:
        frame[dst] = pd.to_numeric(raw[src], errors="raise")
    frame = frame[list(_FIVE_MIN_COLUMNS)]
    times = frame["timestamp"].dt.time
    frame = frame.loc[
        (frame["timestamp"] < val_end_ts)
        & (times >= MARKET_OPEN)
        & (times <= MARKET_CLOSE),
        list(_FIVE_MIN_COLUMNS),
    ]
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(frame) == 0:
        raise ValueError(
            f"no pre-holdout RTH one-minute rows before {val_end_ts} in path={path}"
        )
    # Codex impl review P1-2: reject duplicate 1-minute timestamps BEFORE
    # resample. .resample("5min") would silently merge them, collapsing the
    # duplicate evidence into one 5-minute bar with no warning.
    if frame["timestamp"].duplicated().any():
        first_dup = frame.loc[frame["timestamp"].duplicated(), "timestamp"].iloc[0]
        raise ValueError(
            f"duplicate one-minute timestamp path={path}; first={first_dup}"
        )
    # Codex impl review P1-1: validate the surviving 1-minute OHLCV BEFORE
    # resample. Otherwise a NaN/short-row volume is silently undercounted by the
    # 5-minute sum and invalid 1-minute OHLC is masked by aggregation.
    try:
        _validated_ohlcv(frame)
    except ValueError as exc:
        raise ValueError(f"path={path}: {exc}") from exc
    return frame


def resample_to_five_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a 1-minute frame to 5-minute bars (mirrors Stage 0 exactly).

    ``frame`` must carry ``_FIVE_MIN_COLUMNS`` with a tz-naive ``datetime64``
    ``timestamp``. Aggregation: ``open=first, high=max, low=min, close=last,
    volume=sum``; bars with any NaN component are dropped; the result is filtered
    to RTH ``MARKET_OPEN..MARKET_CLOSE`` inclusive. Pure; no I/O.
    """
    missing = [c for c in _FIVE_MIN_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"resample_to_five_minutes frame missing columns: {missing}")
    ts = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        raise ValueError(f"frame['timestamp'] must be datetime64; got {ts.dtype}")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise ValueError("frame['timestamp'] must be timezone-naive")
    resampled = (
        frame.set_index("timestamp")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .reset_index()
    )
    times = resampled["timestamp"].dt.time
    return resampled.loc[
        (times >= MARKET_OPEN) & (times <= MARKET_CLOSE),
        list(_FIVE_MIN_COLUMNS),
    ].reset_index(drop=True)


def load_ticker_bars_txt(
    manifest: Mapping[str, str | Path],
    *,
    val_end: str | pd.Timestamp = VAL_END,
) -> pd.DataFrame:
    """Return pooled pre-holdout 5-minute bars from per-ticker 1-minute ``.txt``.

    Per ticker: ``load_one_minute_txt`` -> ``resample_to_five_minutes`` -> inject
    the ticker column. The pooled frame is sorted ascending by
    ``(ticker, timestamp)`` and returned with EXACTLY the same columns/order as
    ``load_ticker_bars`` (the CSV loader): ``_CANONICAL_COLUMN_ORDER``. Runs the
    shared OHLCV sanity check and a holdout-closure postcondition (no bar
    ``>= val_end`` survives) as defense-in-depth.
    """
    if not manifest:
        raise ValueError("manifest is empty; cannot load 0 tickers")
    val_end_ts = _normalize_val_end(val_end)

    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_ticker, raw_path in manifest.items():
        ticker = str(raw_ticker).strip()
        if not ticker:
            raise ValueError("manifest ticker key empty")
        if ticker in seen:
            raise ValueError(f"duplicate ticker after normalization: {ticker!r}")
        seen.add(ticker)
        if not isinstance(raw_path, (str, Path)):
            raise TypeError(
                f"manifest[{raw_ticker!r}] must be str or Path; "
                f"got {type(raw_path).__name__}"
            )
        normalized.append((ticker, Path(raw_path)))

    frames: list[pd.DataFrame] = []
    for ticker, path in normalized:
        one_minute = load_one_minute_txt(path, val_end=val_end_ts)
        five_minute = resample_to_five_minutes(one_minute)
        if len(five_minute) == 0:
            raise ValueError(f"ticker={ticker}: no 5-minute bars after resample path={path}")
        five_minute = five_minute.copy()
        five_minute["ticker"] = ticker

        contam_mask = five_minute["timestamp"] >= val_end_ts
        if contam_mask.any():
            first_bad = five_minute.loc[contam_mask, "timestamp"].iloc[0]
            raise ValueError(
                f"ticker={ticker}: holdout closure violated post-resample; "
                f"first contaminated timestamp={first_bad}"
            )
        if five_minute["timestamp"].duplicated().any():
            first_dup = five_minute.loc[
                five_minute["timestamp"].duplicated(), "timestamp"
            ].iloc[0]
            raise ValueError(
                f"ticker={ticker}: duplicate 5-minute timestamp; first={first_dup}"
            )
        try:
            _validated_ohlcv(five_minute)
        except ValueError as exc:
            raise ValueError(f"ticker={ticker}: {exc}") from exc
        frames.append(five_minute[list(_CANONICAL_COLUMN_ORDER)])

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(
        ["ticker", "timestamp"], kind="stable",
    ).reset_index(drop=True)
    return pooled
