"""Pure DataFrame profiling helpers for Phase 1B capacity checks."""

import numpy as np
import pandas as pd

from runner_utils.dataset import make_no_trade_band_labels


BAR_SUMMARY_COLUMNS = [
    "ticker",
    "n_days",
    "min_bars_per_day",
    "p5_bars_per_day",
    "p10_bars_per_day",
    "median_bars_per_day",
    "max_bars_per_day",
]

CAPACITY_COLUMNS = [
    "ticker",
    "k",
    "threshold_bps",
    "n_total",
    "n_tail",
    "n_cross_day",
    "n_neutral",
    "n_up",
    "n_down",
    "n_retained",
    "retained_pct",
    "neutral_dropped_pct",
    "up_pct_retained",
    "down_pct_retained",
    "minority_pct_retained",
    "eff_target_independent",
]

SCALER_DIAGNOSTIC_COLUMNS = [
    "feature_group",
    "full_train_mean",
    "retained_train_mean",
    "mean_shift",
    "full_train_std",
    "retained_train_std",
    "std_ratio",
    "warning",
]


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")


def _require_datetime_column(df: pd.DataFrame, column: str, context: str) -> None:
    _require_columns(df, [column], context)
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise ValueError(f"{context} column {column!r} must be datetime dtype")


def _safe_total_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def _safe_retained_ratio(numerator: int, n_retained: int) -> float:
    if n_retained == 0:
        return np.nan
    return numerator / n_retained


def compute_bar_count_summary(
    frames: dict[str, pd.DataFrame],
    timestamp_col: str,
) -> pd.DataFrame:
    """Return per-ticker daily bar-count distribution statistics."""

    rows = []
    for ticker, frame in frames.items():
        _require_datetime_column(frame, timestamp_col, f"compute_bar_count_summary ticker {ticker!r}")
        counts = frame[timestamp_col].dt.date.value_counts(sort=False).sort_index().to_numpy(dtype=float)
        rows.append(
            {
                "ticker": ticker,
                "n_days": int(len(counts)),
                "min_bars_per_day": np.min(counts) if len(counts) else np.nan,
                "p5_bars_per_day": np.percentile(counts, 5) if len(counts) else np.nan,
                "p10_bars_per_day": np.percentile(counts, 10) if len(counts) else np.nan,
                "median_bars_per_day": np.median(counts) if len(counts) else np.nan,
                "max_bars_per_day": np.max(counts) if len(counts) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=BAR_SUMMARY_COLUMNS)


def profile_label_capacity(
    frames: dict[str, pd.DataFrame],
    price_col: str,
    timestamp_col: str,
    k_values: list[int],
    threshold_bps_values: list[float],
) -> pd.DataFrame:
    """Return label-retention diagnostics for each ticker, horizon, and threshold."""

    rows = []
    for ticker, frame in frames.items():
        for k in k_values:
            for threshold_bps in threshold_bps_values:
                _, diagnostics = make_no_trade_band_labels(
                    frame,
                    price_col=price_col,
                    k=k,
                    threshold_bps=threshold_bps,
                    timestamp_col=timestamp_col,
                )
                n_total = diagnostics["n_total"]
                n_up = diagnostics["n_up"]
                n_down = diagnostics["n_down"]
                n_neutral = diagnostics["n_neutral"]
                n_retained = n_up + n_down
                rows.append(
                    {
                        "ticker": ticker,
                        "k": k,
                        "threshold_bps": threshold_bps,
                        "n_total": n_total,
                        "n_tail": diagnostics["n_tail"],
                        "n_cross_day": diagnostics["n_cross_day"],
                        "n_neutral": n_neutral,
                        "n_up": n_up,
                        "n_down": n_down,
                        "n_retained": n_retained,
                        "retained_pct": _safe_total_ratio(n_retained, n_total),
                        "neutral_dropped_pct": _safe_total_ratio(n_neutral, n_total),
                        "up_pct_retained": _safe_retained_ratio(n_up, n_retained),
                        "down_pct_retained": _safe_retained_ratio(n_down, n_retained),
                        "minority_pct_retained": _safe_retained_ratio(min(n_up, n_down), n_retained),
                        "eff_target_independent": n_retained / k,
                    }
                )
    return pd.DataFrame(rows, columns=CAPACITY_COLUMNS)


def add_intraday_feasibility_flags(
    capacity_df: pd.DataFrame,
    bar_summary_df: pd.DataFrame,
    window_sizes: list[int],
) -> pd.DataFrame:
    """Return capacity rows expanded with p5 intraday feasibility flags."""

    _require_columns(capacity_df, ["ticker", "k"], "add_intraday_feasibility_flags capacity_df")
    _require_columns(bar_summary_df, ["ticker", "p5_bars_per_day"], "add_intraday_feasibility_flags bar_summary_df")
    rows = []
    for _, capacity_row in capacity_df.iterrows():
        base = capacity_row.to_dict()
        for window_size in window_sizes:
            expanded = dict(base)
            expanded["window_size"] = window_size
            rows.append(expanded)

    expanded_df = pd.DataFrame(rows)
    if expanded_df.empty:
        for column in ["window_size", "required_bars", "p5_bars_per_day", "feasible_by_p5", "feasibility_status"]:
            expanded_df[column] = pd.Series(dtype=object)
        return expanded_df

    p5_lookup = bar_summary_df[["ticker", "p5_bars_per_day"]].copy(deep=True)
    expanded_df = expanded_df.merge(p5_lookup, on="ticker", how="left")
    expanded_df["required_bars"] = expanded_df["window_size"] + expanded_df["k"]
    expanded_df["feasible_by_p5"] = expanded_df["required_bars"] <= expanded_df["p5_bars_per_day"]
    expanded_df["feasibility_status"] = np.where(
        expanded_df["feasible_by_p5"],
        "PASS",
        "INFEASIBLE_INTRADAY_CAPACITY",
    )
    return expanded_df


def compute_scaler_diagnostic(
    full_train_df: pd.DataFrame,
    retained_train_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """Return coarse group-level feature distribution diagnostics."""

    rows = []
    for feature_group, columns in feature_groups.items():
        context = f"compute_scaler_diagnostic feature group {feature_group!r}"
        _require_columns(full_train_df, columns, f"{context} full_train_df")
        _require_columns(retained_train_df, columns, f"{context} retained_train_df")
        full_values = full_train_df[columns].to_numpy(dtype=float).reshape(-1)
        retained_values = retained_train_df[columns].to_numpy(dtype=float).reshape(-1)

        full_train_mean = float(np.mean(full_values))
        retained_train_mean = float(np.mean(retained_values))
        mean_shift = retained_train_mean - full_train_mean
        full_train_std = float(np.std(full_values))
        retained_train_std = float(np.std(retained_values))
        std_ratio = retained_train_std / full_train_std if full_train_std > 0 else np.nan
        warning = abs(mean_shift) > 0.25 * full_train_std or std_ratio < 0.75 or std_ratio > 1.25
        rows.append(
            {
                "feature_group": feature_group,
                "full_train_mean": full_train_mean,
                "retained_train_mean": retained_train_mean,
                "mean_shift": mean_shift,
                "full_train_std": full_train_std,
                "retained_train_std": retained_train_std,
                "std_ratio": std_ratio,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=SCALER_DIAGNOSTIC_COLUMNS)
