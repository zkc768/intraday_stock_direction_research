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
