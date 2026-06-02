import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_price_df():
    """AGENTS.md section 3.4 Stage 1 - Raw input.
    Provide a small single-ticker OHLCV and technical-feature frame before label generation.
    """
    close = [
        100.0,
        101.0,
        102.0,
        101.0,
        103.0,
        104.0,
        104.0,
        103.0,
        105.0,
        106.0,
        105.0,
        107.0,
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30",
                    "2024-01-02 09:35",
                    "2024-01-02 09:40",
                    "2024-01-02 09:45",
                    "2024-01-02 09:50",
                    "2024-01-02 09:55",
                    "2024-01-03 09:30",
                    "2024-01-03 09:35",
                    "2024-01-03 09:40",
                    "2024-01-03 09:45",
                    "2024-01-03 09:50",
                    "2024-01-03 09:55",
                ]
            ),
            "open": [value - 0.2 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1180, 1400, 1350, 1280, 1500],
            "macd": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.25, 0.15, 0.35, 0.45, 0.4, 0.5],
            "macd_signal": [-0.15, -0.1, -0.05, 0.0, 0.1, 0.2, 0.22, 0.18, 0.25, 0.3, 0.32, 0.38],
            "macd_hist": [-0.05, 0.0, 0.05, 0.1, 0.1, 0.1, 0.03, -0.03, 0.1, 0.15, 0.08, 0.12],
            "rsi_14": [45.0, 48.0, 52.0, 49.0, 55.0, 58.0, 57.0, 53.0, 60.0, 62.0, 59.0, 65.0],
            "bb_pctb": [0.40, 0.45, 0.50, 0.47, 0.55, 0.60, 0.58, 0.52, 0.63, 0.66, 0.61, 0.70],
            "rolling_std_20": [1.0, 1.1, 1.2, 1.15, 1.25, 1.3, 1.28, 1.22, 1.35, 1.4, 1.32, 1.45],
            "obv_roc": [0.00, 0.02, 0.01, 0.03, 0.025, 0.04, 0.035, 0.015, 0.05, 0.045, 0.02, 0.055],
        }
    )


@pytest.fixture
def raw_multi_ticker_dict(raw_price_df):
    """AGENTS.md section 3.4 Stage 1 - Raw input.
    Provide independent per-ticker DataFrame copies for multi-ticker split and merge tests.
    """
    aaa = raw_price_df.copy(deep=True)
    bbb = raw_price_df.copy(deep=True)
    for column in ["open", "high", "low", "close"]:
        bbb[column] = bbb[column] + 10.0
    return {"AAA": aaa, "BBB": bbb}


@pytest.fixture
def labeled_df_with_tail_nan(raw_price_df):
    """AGENTS.md section 3.4 Stage 2 - Labeled frame.
    Provide a labeled frame before split trimming with only natural tail label invalid markers.
    """
    frame = raw_price_df.copy(deep=True)
    frame["label"] = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, np.nan, np.nan]
    return frame


@pytest.fixture
def split_df_after_trim(labeled_df_with_tail_nan):
    """AGENTS.md section 3.4 Stage 3 - Post-trim frame.
    Provide a post-trim frame with label invalid markers for Dataset window boundary tests.
    """
    frame = labeled_df_with_tail_nan.copy(deep=True)
    # Mark split boundary invalid.
    frame.loc[3, "label"] = np.nan
    # Mark cross-day horizon invalid.
    frame.loc[5, "label"] = np.nan
    # Keep natural tail invalid markers.
    frame.loc[[10, 11], "label"] = np.nan
    return frame
