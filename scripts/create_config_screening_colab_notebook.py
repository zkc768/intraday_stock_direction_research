from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_config_screening_colab.ipynb"
SOURCE_NOTEBOOK_PATH = NOTEBOOK_PATH


def code_cell(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def source_cell(source_nb, *required: str) -> str:
    for cell in source_nb.cells:
        if cell.cell_type == "code" and all(text in cell.source for text in required):
            return cell.source.strip()
    raise ValueError(f"Missing source notebook cell containing: {required}")


def raw_drive_data_source() -> str:
    return """
RAW_DRIVE_FOLDER_ID = "154SlcH3nViUcvPXFBM-E4NPg_ybljBTG"
RAW_DRIVE_FOLDER_NAME = "s&p 100 adjusted 1 min data"
RAW_DATA_DIR = Path("/content/stage0_raw_stock_data")
DOWNLOAD_RAW_DATA_FROM_DRIVE = True

RAW_DRIVE_FILES = {
    "CSCO": {"name": "CSCO.txt", "file_id": "17A49kUiMELuQqdkOhw1KrpudjP5i5xIN"},
    "JPM": {"name": "JPM.txt", "file_id": "11UQUJKVXTrBb8XFWY5Z8JDQ8_4i_DE-q"},
    "KO": {"name": "KO.txt", "file_id": "1XmtwuZ2dTP20NsU27w5dMyRdSvdnNTSn"},
    "MSFT": {"name": "MSFT.txt", "file_id": "1Ud1SQpQbaiRKemFf9dgu1o_raUPnFvGs"},
    "WMT": {"name": "WMT.txt", "file_id": "1NNfsoUJrrsj2ae5EnC-PTPcZs_QGR_7c"},
}


def is_real_raw_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.lower().endswith(".gshortcut"):
        return False
    if path.suffix.lower() not in DATA_FILE_SUFFIXES:
        return False
    try:
        return path.stat().st_size > 50_000
    except OSError:
        return False


def build_drive_service():
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Drive API is unavailable. Open this notebook in Google Colab and "
            "authenticate when prompted; do not use drive.mount for Stage 0 data."
        ) from exc
    auth.authenticate_user()
    return build("drive", "v3")


def download_raw_drive_files():
    if not DOWNLOAD_RAW_DATA_FROM_DRIVE:
        return
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError("googleapiclient is unavailable in this Colab runtime.") from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    service = build_drive_service()
    for ticker, item in RAW_DRIVE_FILES.items():
        target = RAW_DATA_DIR / item["name"]
        if is_real_raw_file(target):
            print(f"{ticker}: using cached raw file {target}")
            continue
        print(f"{ticker}: downloading raw Drive file {item['name']} -> {target}")
        request = service.files().get_media(fileId=item["file_id"])
        with target.open("wb") as output:
            downloader = MediaIoBaseDownload(output, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if not is_real_raw_file(target):
            raise ValueError(f"Downloaded file is not a real raw ticker file: {target}")


def resolve_data_files():
    if DOWNLOAD_RAW_DATA_FROM_DRIVE:
        download_raw_drive_files()
    files = {}
    missing = []
    for ticker, item in RAW_DRIVE_FILES.items():
        path = RAW_DATA_DIR / item["name"]
        if is_real_raw_file(path):
            files[ticker] = path
        else:
            missing.append(f"{ticker}: {path}")
    if missing:
        raise FileNotFoundError(
            "Missing required raw ticker files after Drive API resolution: "
            + "; ".join(missing)
        )
    print("resolved raw Drive data files:")
    for ticker, path in files.items():
        print(f"  {ticker}: {path}")
    return files


def find_timestamp_column(columns):
    for candidate in ("timestamp", "datetime", "date", "time"):
        for column in columns:
            if str(column).lower() == candidate:
                return column
    raise ValueError(f"No timestamp-like column found in columns: {list(columns)}")


def normalize_ohlcv_columns(frame, source_name):
    lower_map = {str(column).lower(): column for column in frame.columns}
    rename = {}
    for required in EXPECTED_COLUMNS:
        if required == "timestamp":
            continue
        if required not in lower_map:
            raise ValueError(f"{source_name} missing required column: {required}")
        rename[lower_map[required]] = required
    return frame.rename(columns=rename)


def resample_to_five_minutes(frame):
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
        list(EXPECTED_COLUMNS),
    ].reset_index(drop=True)


def txt_date_key(date_text):
    parts = str(date_text).strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"Unexpected Date field in raw txt file: {date_text!r}")
    month, day, year = parts
    return int(year), int(month), int(day)


def count_txt_rows_before_validation_end(path):
    validation_end_key = txt_date_key(pd.Timestamp(VAL_END).strftime("%m/%d/%Y"))
    safe_rows = 0
    has_header = False
    reached_boundary = False
    with Path(path).open("rt", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            first_field = stripped.split(",", 1)[0].strip()
            if first_field.lower() == "date":
                has_header = True
                continue
            if txt_date_key(first_field) >= validation_end_key:
                reached_boundary = True
                break
            safe_rows += 1
    if safe_rows == 0:
        raise ValueError(f"No train/validation rows found before {VAL_END} in: {path}")
    if not reached_boundary:
        print(f"{Path(path).name}: no row at or after {VAL_END} found; read capped to file end.")
    return safe_rows, has_header


def read_one_minute_txt(path):
    safe_rows, has_header = count_txt_rows_before_validation_end(path)
    print(f"{Path(path).name}: loading {safe_rows:,} raw one-minute rows before {VAL_END}.")
    frame = pd.read_csv(
        path,
        header=None,
        names=RAW_TXT_COLUMNS,
        nrows=safe_rows,
        skiprows=1 if has_header else None,
        low_memory=False,
    )
    frame = frame.loc[frame["Date"].astype(str).str.lower() != "date"].reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="%m/%d/%Y %H:%M",
        errors="raise",
    )
    frame = frame.drop(columns=["Date", "Time"]).rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    validation_end = pd.Timestamp(VAL_END)
    times = frame["timestamp"].dt.time
    frame = frame.loc[
        (frame["timestamp"] < validation_end)
        & (times >= MARKET_OPEN)
        & (times <= MARKET_CLOSE),
        list(EXPECTED_COLUMNS),
    ]
    return resample_to_five_minutes(frame)


def read_five_minute_csv(path):
    validation_end = pd.Timestamp(VAL_END)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        timestamp_column = find_timestamp_column(chunk.columns)
        chunk = chunk.rename(columns={timestamp_column: "timestamp"}).copy()
        chunk = normalize_ohlcv_columns(chunk, path.name)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="raise")
        raw_chunk_max_timestamp = chunk["timestamp"].max()
        chunk = chunk.loc[chunk["timestamp"] < validation_end, list(EXPECTED_COLUMNS)]
        if not chunk.empty:
            chunks.append(chunk)
        if raw_chunk_max_timestamp >= validation_end:
            break
    if not chunks:
        raise ValueError(f"No train/validation rows found in: {path}")
    return pd.concat(chunks, ignore_index=True)


def load_ticker(ticker, path):
    path = Path(path)
    frame = read_one_minute_txt(path) if path.suffix.lower() == ".txt" else read_five_minute_csv(path)
    frame["ticker"] = ticker
    return frame.sort_values("timestamp").reset_index(drop=True)


RUN_ANY_STAGE = bool(RUN_STAGE0S or RUN_STAGE0A1 or RUN_STAGE0A2 or RUN_STAGE0B)

if RUN_ANY_STAGE:
    DATA_FILES = resolve_data_files()
    raw_data = {ticker: load_ticker(ticker, DATA_FILES[ticker]) for ticker in TICKERS}

    display(pd.DataFrame([
        {
            "ticker": ticker,
            "rows": len(frame),
            "start": frame["timestamp"].min(),
            "end": frame["timestamp"].max(),
            "source": DATA_FILES[ticker].name,
            "path": str(DATA_FILES[ticker]),
        }
        for ticker, frame in raw_data.items()
    ]))
else:
    DATA_FILES = {}
    raw_data = {}
    print("All RUN_STAGE0* switches are False; data loading skipped.")
"""


def build_notebook():
    if not SOURCE_NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Missing required source notebook: {SOURCE_NOTEBOOK_PATH}")

    source_nb = nbf.read(SOURCE_NOTEBOOK_PATH, as_version=4)
    data_source = raw_drive_data_source()
    feature_source = source_cell(
        source_nb,
        "def add_features",
        "def prepare_split_frames",
        "def build_last_step_windows",
    )

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}

    cells = []
    cells.append(
        markdown_cell(
            """
# Stage 0 Configuration Screening - Validation Only

Research question: can five-minute stock bars support an honest directional
configuration candidate under chronological train/validation screening?

Scope: `validation_only`.

Active freeze: `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md`.

This notebook starts from the raw ticker files and the candidate space in the
active freeze. Earlier local screening notes and route files are not part of
the active project surface.

Naming note: every `label_config`, `feature_set`, and `window_size` in this
notebook is an untested candidate before validation rows are produced.

Frozen route:

```text
Stage 0S = runtime/schema smoke, no selection
Stage 0A1 = label-feature screen with LogReg + LightGBM at window_size=10
Stage 0A2 = LightGBM window sensitivity on Stage 0A1 short list
Stage 0B = LogReg + LightGBM + simple GRU + MS-DLinear+TCN second-view
scope = validation_only
```

Hard boundaries:

- raw input comes only from the five Drive `.txt` files listed in this notebook
- the notebook does not mount MyDrive and does not depend on a mounted
  MyDrive path
- no holdout/test rows are loaded, transformed, windowed, scored, summarized, or
  used for wording decisions
- train and validation are chronological
- preprocessing is fit on pooled train rows only, after per-ticker splits
- labels are invalidated at train/validation and validation/closed-holdout
  boundaries
- every model-seed row is compared with a stratified dummy comparator on the same
  target rows
- Stage 0B is a second-view diagnostic only; it must not rerank or retract
  Stage 0A candidates

This notebook performs a fresh Stage 0 configuration screen. The stratified
dummy is only a lower-bound comparator for reporting deltas.

The output directory is:

```text
notebooks/results/02_config_screening/
```

In Colab this maps to the local runtime directory
`/content/stage0_config_screening_results`. Download or copy those results only
after the run finishes.

Default runtime switches run only Stage 0S. Flip only the stage you intend to
run before starting Stage 0A1, Stage 0A2, or Stage 0B.
"""
        )
    )

    cells.append(
        code_cell(
            """
from pathlib import Path
from collections import Counter
import importlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 160)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

INSTALL_LIGHTGBM_IF_MISSING = True
INSTALL_TORCH_IF_MISSING = False
PYTHON_DEPS_DIR = Path("/content/stage0_python_deps")


def install_package_to_local_target(package_name):
    target_dir = PYTHON_DEPS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = package_name.split("[", 1)[0].replace("-", "_")
    for pattern in (base_name, f"{base_name}-*.dist-info"):
        for path in target_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--upgrade",
        "--target",
        str(target_dir),
        package_name,
    ]
    print("Running dependency install:", " ".join(cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pip install failed for {package_name} with exit code {proc.returncode}. "
            "Read the pip output above. If Colab reports a package or filesystem "
            "error, restart the runtime and rerun setup cells."
        )
    target_text = str(target_dir)
    if target_text not in sys.path:
        sys.path.insert(0, target_text)
    importlib.invalidate_caches()


def ensure_lightgbm():
    try:
        import lightgbm as lgb
        return lgb
    except (ImportError, OSError) as exc:
        if INSTALL_LIGHTGBM_IF_MISSING:
            print(f"LightGBM import failed before install: {type(exc).__name__}: {exc}")
            install_package_to_local_target("lightgbm")
            sys.modules.pop("lightgbm", None)
            try:
                import lightgbm as lgb
                return lgb
            except (ImportError, OSError) as retry_exc:
                raise RuntimeError(
                    "LightGBM still cannot import after installing into "
                    f"{PYTHON_DEPS_DIR}. Restart the Colab runtime and rerun "
                    "setup cells before Stage 0S."
                ) from retry_exc
        raise RuntimeError(
            "LightGBM import failed. Set INSTALL_LIGHTGBM_IF_MISSING=True and "
            "rerun setup cells. The notebook will install LightGBM into "
            f"{PYTHON_DEPS_DIR} and place that directory first on sys.path."
        ) from exc


def ensure_torch():
    try:
        import torch
        return torch
    except (ImportError, OSError) as exc:
        if INSTALL_TORCH_IF_MISSING:
            print(f"PyTorch import failed before install: {type(exc).__name__}: {exc}")
            install_package_to_local_target("torch")
            sys.modules.pop("torch", None)
            try:
                import torch
                return torch
            except (ImportError, OSError) as retry_exc:
                raise RuntimeError(
                    "PyTorch still cannot import after installing into "
                    f"{PYTHON_DEPS_DIR}. Restart the Colab runtime and rerun "
                    "setup cells before Stage 0B."
                ) from retry_exc
        raise RuntimeError(
            "PyTorch import failed. Set INSTALL_TORCH_IF_MISSING=True and rerun "
            "setup cells before enabling Stage 0B deep models."
        ) from exc
"""
        )
    )

    cells.append(
        code_cell(
            """
TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
MODEL_SEEDS = (101, 202, 303, 404, 505)
RESULT_SCOPE = "validation_only"

RUN_STAGE0S = True
RUN_STAGE0A1 = False
RUN_STAGE0A2 = False
RUN_STAGE0B = False
RUN_STAGE0B_DEEP_MODELS = True

ALL_FEATURES = (
    "log_return",
    "close_to_open_return",
    "high_low_range",
    "rolling_volatility_20",
    "normalized_volume_20",
    "rsi_14",
    "bollinger_pctb",
    "normalized_macd_hist",
    "time_of_day_sin",
    "time_of_day_cos",
)

FEATURE_SETS = {
    "price_action_core": (
        "log_return",
        "close_to_open_return",
        "high_low_range",
    ),
    "technical_price": (
        "log_return",
        "high_low_range",
        "rsi_14",
        "bollinger_pctb",
        "normalized_macd_hist",
    ),
    "price_volume_time": ALL_FEATURES,
}

LABEL_CONFIGS = {
    "h03_bps1p5": {"horizon_k": 3, "threshold_bps": 1.5},
    "h09_bps3p0": {"horizon_k": 9, "threshold_bps": 3.0},
    "h24_bps7p5": {"horizon_k": 24, "threshold_bps": 7.5},
}

STAGE0S_SPEC = {
    "stage": "Stage 0S",
    "model": "lightgbm",
    "label_config": "h09_bps3p0",
    "feature_set": "price_volume_time",
    "window_size": 10,
    "seed": 101,
}

STAGE0A1_LABEL_CONFIGS = ("h03_bps1p5", "h09_bps3p0", "h24_bps7p5")
STAGE0A1_FEATURE_SETS = ("price_action_core", "technical_price", "price_volume_time")
STAGE0A1_MODELS = ("logreg", "lightgbm")
STAGE0A1_WINDOW_SIZE = 10

STAGE0A2_MODELS = ("lightgbm",)
STAGE0A2_WINDOW_SIZES = (5, 10, 20)

STAGE0B_MODELS = ("logreg", "lightgbm", "simple_gru", "ms_dlinear_tcn")

MAX_TRAIN_ROWS = None
RANDOM_SUBSAMPLE_SEED = 42

LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    # max_depth and num_leaves must be consistent: with max_depth=6,
    # num_leaves=31 is realisable instead of being capped by max_depth=3.
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
}

TORCH_EPOCHS = 8
TORCH_BATCH_SIZE = 1024
TORCH_LEARNING_RATE = 1e-3
TORCH_WEIGHT_DECAY = 1e-4
TORCH_TCN_CHANNELS = (32, 32)
TORCH_TCN_KERNEL_SIZE = 3
TORCH_MOVING_AVG_KERNELS = (3, 5, 9, 15)
TORCH_DROPOUT = 0.10

TRAIN_START, TRAIN_END = "1998-01-02", "2013-09-16"
VAL_START, VAL_END = "2013-09-16", "2017-01-25"
CALENDAR_SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "validation": (VAL_START, VAL_END),
}

BPS_TO_DECIMAL = 10000.0
BAR_INTERVAL_MINUTES = 5
MARKET_OPEN_MINUTE = 9 * 60 + 30
TRADING_DAY_MINUTES = 390
TIME_OF_DAY_ENCODING_PERIOD_MINUTES = TRADING_DAY_MINUTES + BAR_INTERVAL_MINUTES
MARKET_OPEN = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()
EXPECTED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DATA_FILE_SUFFIXES = (".csv", ".txt")
RAW_TXT_COLUMNS = ("Date", "Time", "Open", "High", "Low", "Close", "Volume")

OUTPUT_DIR = Path("/content/stage0_config_screening_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILES = {
    "stage0s_pooled": OUTPUT_DIR / "stage0s_pooled.csv",
    "stage0s_per_ticker": OUTPUT_DIR / "stage0s_per_ticker.csv",
    "stage0a1_pooled": OUTPUT_DIR / "stage0a1_pooled.csv",
    "stage0a1_per_ticker": OUTPUT_DIR / "stage0a1_per_ticker.csv",
    "stage0a1_summary": OUTPUT_DIR / "stage0a1_summary.csv",
    "stage0a2_pooled": OUTPUT_DIR / "stage0a2_pooled.csv",
    "stage0a2_per_ticker": OUTPUT_DIR / "stage0a2_per_ticker.csv",
    "stage0a2_summary": OUTPUT_DIR / "stage0a2_summary.csv",
    "stage0b_pooled": OUTPUT_DIR / "stage0b_pooled.csv",
    "stage0b_per_ticker": OUTPUT_DIR / "stage0b_per_ticker.csv",
    "stage0b_summary": OUTPUT_DIR / "stage0b_summary.csv",
    "stage0_candidates": OUTPUT_DIR / "stage0_candidates.json",
}

display(pd.DataFrame([
    {"feature_set": name, "n_features": len(features), "features": ", ".join(features)}
    for name, features in FEATURE_SETS.items()
]))
print("Output directory:", OUTPUT_DIR)
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Data Loading

This cell resolves the five real ticker files and downloads them through the
Google Drive API when needed. It does not mount MyDrive. For raw `.txt` files it
first scans to the validation boundary and then reads only rows before
`VAL_END`, so closed holdout/test rows are not materialized into the notebook
dataframes. Outputs stay local under `/content/stage0_config_screening_results`.
"""
        )
    )
    cells.append(code_cell(data_source))

    cells.append(
        markdown_cell(
            """
## Feature, Label, Split, Scale, Window

These functions implement the active chronology-safe Stage 0 contracts inside a
standalone Colab notebook. They do not import a local helper package or a prior
route. Keep the causal post-bar-close feature rules, cumulative
horizon-return labels, split-boundary invalidation, train-only scaler fitting,
and per-ticker/per-day windows aligned with the active freeze documents before
rerunning Stage 0.

Tabular models use flattened windows: each LogReg/LightGBM sample is
`window_size * n_features`, built from the same per-ticker/per-day rows used by
the sequence models. This makes Stage 0A2 a true window-length sensitivity
check rather than a last-bar sample-count check.

Feature timing boundary: prediction is after the current five-minute bar has
completed, so `close[t]`, `high[t]`, `low[t]`, `volume[t]`, and same-row
timestamp encodings are available. Same-day trailing Bollinger windows may
include `close[t]`; RSI and MACD EWM states are causal but intentionally
continuous across trading days.
"""
        )
    )
    cells.append(code_cell(feature_source))

    cells.append(
        markdown_cell(
            """
## Stage 0 Model And Metrics Helpers

The tabular models use last-step views after valid-window construction. The
sequence models use full windows and are available only in Stage 0B. All rows
include stratified dummy-comparator metrics, dummy deltas, timing columns, and
per-ticker concentration diagnostics.
"""
        )
    )
    cells.append(
        code_cell(
            """
DATASET_CACHE = {}


def label_spec(label_config):
    spec = LABEL_CONFIGS[label_config]
    return int(spec["horizon_k"]), float(spec["threshold_bps"])


def subsample_rows_uniformly(x_values, y_values, max_rows, seed=RANDOM_SUBSAMPLE_SEED):
    if max_rows is None or len(y_values) <= max_rows:
        return x_values, y_values
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(y_values), size=int(max_rows), replace=False))
    return x_values[selected], y_values[selected]


def subsample_rows_with_owner(x_values, y_values, owner_values, max_rows, seed=RANDOM_SUBSAMPLE_SEED):
    if max_rows is None or len(y_values) <= max_rows:
        return x_values, y_values, owner_values
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(y_values), size=int(max_rows), replace=False))
    return x_values[selected], y_values[selected], owner_values[selected]


def evaluate_predictions(y_true, predictions):
    return {
        "macro_f1": float(f1_score(y_true, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def dummy_metrics(y_train, y_validation, seed):
    if len(y_train) == 0 or len(y_validation) == 0:
        return {"dummy_macro_f1": np.nan, "dummy_balanced_accuracy": np.nan}
    x_train = np.zeros((len(y_train), 1))
    x_validation = np.zeros((len(y_validation), 1))
    dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(x_train, y_train)
    pred = dummy.predict(x_validation)
    return {
        "dummy_macro_f1": float(f1_score(y_validation, pred, labels=[0, 1], average="macro", zero_division=0)),
        "dummy_balanced_accuracy": float(balanced_accuracy_score(y_validation, pred)),
    }


def sample_std(values):
    values = pd.Series(values).dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


T_CRITICAL_ONE_SIDED_95 = {
    2: 6.314,
    3: 2.920,
    4: 2.353,
    5: 2.132,
    6: 2.015,
    7: 1.943,
    8: 1.895,
    9: 1.860,
    10: 1.833,
    11: 1.812,
    12: 1.796,
}


def t_critical_one_sided_95(seed_count):
    if seed_count <= 1:
        return 0.0
    return T_CRITICAL_ONE_SIDED_95.get(int(seed_count), 1.645)


def build_sequence_windows(frames_by_ticker, feature_columns, split_name, window_size):
    scaled_columns = [f"{name}_scaled" for name in feature_columns]
    x_parts, y_parts, owner_parts, timestamp_parts = [], [], [], []
    for ticker, frame in frames_by_ticker.items():
        segment = frame.loc[frame["split"] == split_name].sort_values("timestamp")
        for _, day_frame in segment.groupby(segment["timestamp"].dt.date, sort=True):
            day_frame = day_frame.sort_values("timestamp")
            values = day_frame[scaled_columns].to_numpy(dtype=float)
            labels = day_frame["label"].to_numpy()
            timestamps = day_frame["timestamp"].to_numpy()
            complete_rows = np.isfinite(values).all(axis=1)
            for end_idx in range(window_size - 1, len(day_frame)):
                start_idx = end_idx - window_size + 1
                if not complete_rows[start_idx : end_idx + 1].all():
                    continue
                if pd.isna(labels[end_idx]):
                    continue
                x_parts.append(values[start_idx : end_idx + 1])
                y_parts.append(int(labels[end_idx]))
                owner_parts.append(ticker)
                timestamp_parts.append(timestamps[end_idx])
    if not x_parts:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=float),
            np.asarray([], dtype=int),
            np.asarray([], dtype=object),
            np.asarray([], dtype="datetime64[ns]"),
        )
    return (
        np.stack(x_parts).astype(np.float32),
        np.asarray(y_parts, dtype=int),
        np.asarray(owner_parts, dtype=object),
        np.asarray(timestamp_parts, dtype="datetime64[ns]"),
    )


def get_dataset(label_config, feature_set, window_size):
    key = (label_config, feature_set, int(window_size))
    if key in DATASET_CACHE:
        dataset = DATASET_CACHE[key].copy()
        dataset["prep_seconds"] = 0.0
        return dataset
    if not raw_data:
        raise RuntimeError("raw_data is empty. Enable a RUN_STAGE0* switch and rerun data loading first.")
    horizon_k, threshold_bps = label_spec(label_config)
    feature_columns = FEATURE_SETS[feature_set]
    start = time.perf_counter()
    split_frames = prepare_split_frames(raw_data, horizon_k=horizon_k, threshold_bps=threshold_bps)
    scaled_frames = fit_transform_train_validation(split_frames, feature_columns)
    x_train, y_train, train_owner, train_timestamp = build_last_step_windows(
        scaled_frames, feature_columns, "train", window_size
    )
    x_validation, y_validation, validation_owner, validation_timestamp = build_last_step_windows(
        scaled_frames, feature_columns, "validation", window_size
    )
    x_train_seq, y_train_seq, train_owner_seq, train_timestamp_seq = build_sequence_windows(
        scaled_frames, feature_columns, "train", window_size
    )
    x_validation_seq, y_validation_seq, validation_owner_seq, validation_timestamp_seq = build_sequence_windows(
        scaled_frames, feature_columns, "validation", window_size
    )
    if len(y_train) == 0 or len(y_validation) == 0:
        raise ValueError(f"No tabular windows available for {label_config} / {feature_set} / window={window_size}")
    if len(y_train_seq) != len(y_train) or len(y_validation_seq) != len(y_validation):
        raise ValueError("Tabular and sequence window counts disagree; inspect window construction.")
    dataset = {
        "label_config": label_config,
        "horizon_k": horizon_k,
        "threshold_bps": threshold_bps,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "window_size": int(window_size),
        "x_train": x_train,
        "y_train": y_train,
        "train_owner": train_owner,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "validation_owner": validation_owner,
        "x_train_seq": x_train_seq,
        "y_train_seq": y_train_seq,
        "train_owner_seq": train_owner_seq,
        "x_validation_seq": x_validation_seq,
        "y_validation_seq": y_validation_seq,
        "validation_owner_seq": validation_owner_seq,
        "prep_seconds": time.perf_counter() - start,
    }
    DATASET_CACHE[key] = dataset.copy()
    return dataset


def fit_predict_logreg(dataset, seed):
    x_train, y_train = subsample_rows_uniformly(dataset["x_train"], dataset["y_train"], MAX_TRAIN_ROWS, seed=seed)
    # Tabular features are flattened windows, so allow more iterations for the
    # higher-dimensional design matrix.
    max_iter = 2000
    model = LogisticRegression(
        solver="liblinear",
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
    )
    start_fit = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start_fit
    start_predict = time.perf_counter()
    pred = model.predict(dataset["x_validation"])
    predict_seconds = time.perf_counter() - start_predict
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    max_iter_reached = bool((model.n_iter_ >= max_iter).any())
    fit_status = "converged" if not convergence_warnings and not max_iter_reached else "convergence_warning"
    return pred, fit_seconds, predict_seconds, int(len(y_train)), fit_status


def fit_predict_lightgbm(dataset, seed):
    lgb = ensure_lightgbm()
    x_train, y_train = subsample_rows_uniformly(dataset["x_train"], dataset["y_train"], MAX_TRAIN_ROWS, seed=seed)
    model = lgb.LGBMClassifier(
        **LGBM_PARAMS,
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
    )
    start_fit = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start_fit
    start_predict = time.perf_counter()
    pred = model.predict(dataset["x_validation"])
    predict_seconds = time.perf_counter() - start_predict
    return pred, fit_seconds, predict_seconds, int(len(y_train)), "not_applicable"


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch = ensure_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def make_simple_gru(input_dim, seed):
    torch = set_global_seed(seed)
    nn = torch.nn

    class SimpleGRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_dim, 32, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(TORCH_DROPOUT)
            self.head = nn.Linear(32, 2)

        def forward(self, x):
            output, _ = self.gru(x)
            return self.head(self.dropout(output[:, -1, :]))

    return SimpleGRUClassifier()


def make_ms_dlinear_tcn(input_dim, window_size, seed):
    torch = set_global_seed(seed)
    nn = torch.nn
    functional = torch.nn.functional

    class CausalConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dropout):
            super().__init__()
            self.pad = kernel_size - 1
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
            self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            residual = self.proj(x)
            padded = functional.pad(x, (self.pad, 0))
            out = self.conv(padded)
            out = self.dropout(torch.relu(self.norm(out)))
            return out + residual

    class MultiScaleDLinearTCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.tcn = nn.Sequential(
                CausalConvBlock(input_dim, TORCH_TCN_CHANNELS[0], TORCH_TCN_KERNEL_SIZE, TORCH_DROPOUT),
                CausalConvBlock(TORCH_TCN_CHANNELS[0], TORCH_TCN_CHANNELS[1], TORCH_TCN_KERNEL_SIZE, TORCH_DROPOUT),
            )
            self.scale_head = nn.Linear(input_dim * len(TORCH_MOVING_AVG_KERNELS), 16)
            self.head = nn.Linear(TORCH_TCN_CHANNELS[-1] + 16, 2)

        def moving_average_last(self, x, kernel):
            pad = kernel - 1
            padded = functional.pad(x.transpose(1, 2), (pad, 0), mode="replicate")
            avg = functional.avg_pool1d(padded, kernel_size=kernel, stride=1)
            return avg[:, :, -1]

        def forward(self, x):
            tcn_last = self.tcn(x.transpose(1, 2))[:, :, -1]
            scale_parts = [self.moving_average_last(x, kernel) for kernel in TORCH_MOVING_AVG_KERNELS]
            scale = torch.relu(self.scale_head(torch.cat(scale_parts, dim=1)))
            return self.head(torch.cat([tcn_last, scale], dim=1))

    return MultiScaleDLinearTCNClassifier()


def run_torch_shape_smoke(input_dim, window_size):
    torch = ensure_torch()
    for model_name, factory in (
        ("simple_gru", lambda: make_simple_gru(input_dim, 41)),
        ("ms_dlinear_tcn", lambda: make_ms_dlinear_tcn(input_dim, window_size, 41)),
    ):
        model = factory()
        x = torch.zeros((2, window_size, input_dim), dtype=torch.float32)
        y = torch.tensor([0, 1], dtype=torch.long)
        logits = model(x)
        if tuple(logits.shape) != (2, 2):
            raise ValueError(f"{model_name} shape smoke failed: logits shape {tuple(logits.shape)}")
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
    print("Deep adapter shape smoke passed.")


def fit_predict_torch_sequence(dataset, seed, model_name):
    torch = set_global_seed(seed)
    x_train, y_train, train_owner = subsample_rows_with_owner(
        dataset["x_train_seq"],
        dataset["y_train_seq"],
        dataset["train_owner_seq"],
        MAX_TRAIN_ROWS,
        seed=seed,
    )
    x_validation = dataset["x_validation_seq"]
    input_dim = x_train.shape[-1]
    window_size = x_train.shape[1]
    if model_name == "simple_gru":
        model = make_simple_gru(input_dim, seed)
    elif model_name == "ms_dlinear_tcn":
        model = make_ms_dlinear_tcn(input_dim, window_size, seed)
    else:
        raise ValueError(f"Unknown torch model: {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    counts = np.bincount(y_train, minlength=2).astype(float)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TORCH_LEARNING_RATE, weight_decay=TORCH_WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor),
        batch_size=TORCH_BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    start_fit = time.perf_counter()
    model.train()
    for _ in range(TORCH_EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    fit_seconds = time.perf_counter() - start_fit

    start_predict = time.perf_counter()
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_validation), TORCH_BATCH_SIZE):
            batch = torch.tensor(x_validation[start : start + TORCH_BATCH_SIZE], dtype=torch.float32, device=device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    predict_seconds = time.perf_counter() - start_predict
    return np.concatenate(preds), fit_seconds, predict_seconds, int(len(y_train)), "fixed_epochs_no_early_stopping"


def fit_predict_model(dataset, model_name, seed):
    if model_name == "logreg":
        return fit_predict_logreg(dataset, seed)
    if model_name == "lightgbm":
        return fit_predict_lightgbm(dataset, seed)
    if model_name in {"simple_gru", "ms_dlinear_tcn"}:
        return fit_predict_torch_sequence(dataset, seed, model_name)
    raise ValueError(f"Unknown model: {model_name}")


def concentration_from_per_ticker(per_ticker_rows):
    deltas = [row["per_ticker_delta_macro_f1_vs_dummy"] for row in per_ticker_rows]
    positive = [float(delta) for delta in deltas if pd.notna(delta) and delta > 0]
    positive_ticker_count = int(len(positive))
    top_ticker_gain_share = float(max(positive) / sum(positive)) if positive else 0.0
    return positive_ticker_count, top_ticker_gain_share


def run_one_model_seed(stage, model_name, label_config, feature_set, window_size, seed):
    dataset = get_dataset(label_config, feature_set, window_size)
    prep_seconds = float(dataset["prep_seconds"])
    pred, fit_seconds, predict_seconds, train_n, fit_status = fit_predict_model(dataset, model_name, seed)
    pooled_metrics = evaluate_predictions(dataset["y_validation"], pred)
    pooled_dummy = dummy_metrics(dataset["y_train"], dataset["y_validation"], seed)
    per_ticker_rows = []
    for ticker in TICKERS:
        val_mask = dataset["validation_owner"] == ticker
        train_mask = dataset["train_owner"] == ticker
        if not val_mask.any():
            continue
        ticker_metrics = evaluate_predictions(dataset["y_validation"][val_mask], pred[val_mask])
        ticker_dummy = dummy_metrics(dataset["y_train"][train_mask], dataset["y_validation"][val_mask], seed)
        per_ticker_rows.append({
            "stage": stage,
            "model": model_name,
            "label_config": label_config,
            "horizon_k": dataset["horizon_k"],
            "threshold_bps": dataset["threshold_bps"],
            "feature_set": feature_set,
            "window_size": int(window_size),
            "seed": int(seed),
            "scope": RESULT_SCOPE,
            "macro_f1": ticker_metrics["macro_f1"],
            "balanced_accuracy": ticker_metrics["balanced_accuracy"],
            "accuracy": ticker_metrics["accuracy"],
            "dummy_macro_f1": ticker_dummy["dummy_macro_f1"],
            "dummy_balanced_accuracy": ticker_dummy["dummy_balanced_accuracy"],
            "delta_macro_f1_vs_dummy": ticker_metrics["macro_f1"] - ticker_dummy["dummy_macro_f1"],
            "delta_balanced_accuracy_vs_dummy": (
                ticker_metrics["balanced_accuracy"] - ticker_dummy["dummy_balanced_accuracy"]
            ),
            "n": int(val_mask.sum()),
            "ticker_or_pooled": ticker,
            "prep_seconds": prep_seconds,
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "total_seconds": prep_seconds + float(fit_seconds) + float(predict_seconds),
            "per_ticker_delta_macro_f1_vs_dummy": ticker_metrics["macro_f1"] - ticker_dummy["dummy_macro_f1"],
            "positive_ticker_count": np.nan,
            "top_ticker_gain_share": np.nan,
            "train_n": int(train_mask.sum()),
            "fit_status": fit_status,
        })
    positive_ticker_count, top_ticker_gain_share = concentration_from_per_ticker(per_ticker_rows)
    for row in per_ticker_rows:
        row["positive_ticker_count"] = positive_ticker_count
        row["top_ticker_gain_share"] = top_ticker_gain_share
    pooled_row = {
        "stage": stage,
        "model": model_name,
        "label_config": label_config,
        "horizon_k": dataset["horizon_k"],
        "threshold_bps": dataset["threshold_bps"],
        "feature_set": feature_set,
        "window_size": int(window_size),
        "seed": int(seed),
        "scope": RESULT_SCOPE,
        "macro_f1": pooled_metrics["macro_f1"],
        "balanced_accuracy": pooled_metrics["balanced_accuracy"],
        "accuracy": pooled_metrics["accuracy"],
        "dummy_macro_f1": pooled_dummy["dummy_macro_f1"],
        "dummy_balanced_accuracy": pooled_dummy["dummy_balanced_accuracy"],
        "delta_macro_f1_vs_dummy": pooled_metrics["macro_f1"] - pooled_dummy["dummy_macro_f1"],
        "delta_balanced_accuracy_vs_dummy": pooled_metrics["balanced_accuracy"] - pooled_dummy["dummy_balanced_accuracy"],
        "n": int(len(dataset["y_validation"])),
        "ticker_or_pooled": "pooled",
        "prep_seconds": prep_seconds,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": prep_seconds + float(fit_seconds) + float(predict_seconds),
        "per_ticker_delta_macro_f1_vs_dummy": np.nan,
        "positive_ticker_count": positive_ticker_count,
        "top_ticker_gain_share": top_ticker_gain_share,
        "train_n": int(train_n),
        "fit_status": fit_status,
    }
    return pooled_row, per_ticker_rows


def run_stage_grid(stage, specs):
    pooled_rows = []
    per_ticker_rows = []
    for spec in specs:
        print(
            stage,
            spec["model"],
            spec["label_config"],
            spec["feature_set"],
            "window",
            spec["window_size"],
            "seed",
            spec["seed"],
        )
        pooled, per_ticker = run_one_model_seed(
            stage=stage,
            model_name=spec["model"],
            label_config=spec["label_config"],
            feature_set=spec["feature_set"],
            window_size=spec["window_size"],
            seed=spec["seed"],
        )
        pooled_rows.append(pooled)
        per_ticker_rows.extend(per_ticker)
    return pd.DataFrame(pooled_rows), pd.DataFrame(per_ticker_rows)


def summarize_pooled(pooled):
    if pooled.empty:
        return pd.DataFrame()
    rows = []
    keys = ["stage", "model", "label_config", "horizon_k", "threshold_bps", "feature_set", "window_size", "scope"]
    for key_values, group in pooled.groupby(keys, sort=False):
        record = dict(zip(keys, key_values))
        seed_count = int(group["seed"].nunique())
        macro_std = sample_std(group["macro_f1"])
        bal_std = sample_std(group["balanced_accuracy"])
        record.update({
            "seed_count": seed_count,
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": macro_std,
            "macro_f1_lcb_95": float(
                group["macro_f1"].mean()
                - t_critical_one_sided_95(seed_count) * macro_std / math.sqrt(max(seed_count, 1))
            ),
            "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
            "balanced_accuracy_std": bal_std,
            "dummy_macro_f1_mean": float(group["dummy_macro_f1"].mean()),
            "dummy_balanced_accuracy_mean": float(group["dummy_balanced_accuracy"].mean()),
            "delta_macro_f1_vs_dummy_mean": float(group["delta_macro_f1_vs_dummy"].mean()),
            "delta_balanced_accuracy_vs_dummy_mean": float(group["delta_balanced_accuracy_vs_dummy"].mean()),
            "n_mean": float(group["n"].mean()),
            "positive_ticker_count": int(round(group["positive_ticker_count"].mean())),
            "top_ticker_gain_share": float(group["top_ticker_gain_share"].mean()),
            "prep_seconds_mean": float(group["prep_seconds"].mean()),
            "fit_seconds_mean": float(group["fit_seconds"].mean()),
            "predict_seconds_mean": float(group["predict_seconds"].mean()),
            "total_seconds_mean": float(group["total_seconds"].mean()),
        })
        record["basic_gate"] = bool(
            record["delta_macro_f1_vs_dummy_mean"] > 0
            and record["macro_f1_lcb_95"] > record["dummy_macro_f1_mean"]
        )
        record["lcb_eligible"] = bool(
            record["basic_gate"]
            and record["delta_balanced_accuracy_vs_dummy_mean"] > 0
            and record["top_ticker_gain_share"] < 0.50
            and record["positive_ticker_count"] >= 3
        )
        rows.append(record)
    return pd.DataFrame(rows)


def tuple_from_row(row, include_window):
    if row is None:
        return None
    if include_window:
        return {
            "label_config": row["label_config"],
            "feature_set": row["feature_set"],
            "window_size": int(row["window_size"]),
        }
    return {"label_config": row["label_config"], "feature_set": row["feature_set"]}


def select_candidates(summary, include_window):
    if summary.empty or not summary["basic_gate"].any():
        return {
            "stage0_result": "do_not_decide_config",
            "mean_candidate": None,
            "lcb_candidate": None,
            "candidate_count": 0,
        }
    basic = summary.loc[summary["basic_gate"]].sort_values("macro_f1_mean", ascending=False)
    lcb = summary.loc[summary["lcb_eligible"]].sort_values("macro_f1_lcb_95", ascending=False)
    mean_candidate = tuple_from_row(basic.iloc[0], include_window=include_window)
    lcb_candidate = tuple_from_row(lcb.iloc[0], include_window=include_window) if not lcb.empty else None
    unique_candidates = []
    for candidate in (mean_candidate, lcb_candidate):
        if candidate is not None and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return {
        "stage0_result": "candidate_config_selected",
        "mean_candidate": mean_candidate,
        "lcb_candidate": lcb_candidate,
        "candidate_count": len(unique_candidates),
        "candidates": unique_candidates,
    }


def write_outputs(pooled, per_ticker, summary, file_keys):
    pooled.to_csv(OUTPUT_FILES[file_keys[0]], index=False)
    per_ticker.to_csv(OUTPUT_FILES[file_keys[1]], index=False)
    if summary is not None:
        summary.to_csv(OUTPUT_FILES[file_keys[2]], index=False)
    print("wrote", [str(OUTPUT_FILES[key]) for key in file_keys])
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Stage 0S - Runtime And Schema Smoke

Fixed smoke:

```text
model=LightGBM
label_config=h09_bps3p0
feature_set=price_volume_time
window_size=10
seed=101
```

Stage 0S is for data/schema/runtime only and does not participate in
selection.
"""
        )
    )
    cells.append(
        code_cell(
            """
if RUN_STAGE0S:
    stage0s_pooled, stage0s_per_ticker = run_stage_grid("Stage 0S", [STAGE0S_SPEC])
    required_columns = [
        "stage", "model", "label_config", "horizon_k", "threshold_bps", "feature_set",
        "window_size", "seed", "scope", "macro_f1", "balanced_accuracy", "accuracy",
        "dummy_macro_f1", "dummy_balanced_accuracy", "delta_macro_f1_vs_dummy",
        "delta_balanced_accuracy_vs_dummy", "n", "ticker_or_pooled", "prep_seconds",
        "fit_seconds", "predict_seconds", "total_seconds",
        "per_ticker_delta_macro_f1_vs_dummy", "positive_ticker_count", "top_ticker_gain_share",
    ]
    missing_columns = [col for col in required_columns if col not in stage0s_pooled.columns]
    if missing_columns:
        raise ValueError(f"Stage 0S missing required columns: {missing_columns}")
    stage0s_pooled.to_csv(OUTPUT_FILES["stage0s_pooled"], index=False)
    stage0s_per_ticker.to_csv(OUTPUT_FILES["stage0s_per_ticker"], index=False)
    display(stage0s_pooled)
else:
    print("RUN_STAGE0S is False; Stage 0S smoke not run.")
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Stage 0A1 - Label-Feature Screen

Frozen grid:

```text
models       = LogReg, LightGBM
label_config = h03_bps1p5, h09_bps3p0, h24_bps7p5
feature_set  = price_action_core, technical_price, price_volume_time
window_size  = 10
seeds        = 101, 202, 303, 404, 505
```

Expected pooled model-seed rows: `90`.

Before this cell runs, these label and feature entries are search candidates
only. Names are neutral protocol labels, not performance claims.
"""
        )
    )
    cells.append(
        code_cell(
            """
stage0a1_decision = None

if RUN_STAGE0A1:
    stage0a1_specs = [
        {
            "stage": "Stage 0A1",
            "model": model,
            "label_config": label_config,
            "feature_set": feature_set,
            "window_size": STAGE0A1_WINDOW_SIZE,
            "seed": seed,
        }
        for label_config in STAGE0A1_LABEL_CONFIGS
        for feature_set in STAGE0A1_FEATURE_SETS
        for model in STAGE0A1_MODELS
        for seed in MODEL_SEEDS
    ]
    stage0a1_pooled, stage0a1_per_ticker = run_stage_grid("Stage 0A1", stage0a1_specs)
    if len(stage0a1_pooled) != 90:
        raise ValueError(f"Stage 0A1 expected 90 pooled model-seed rows, found {len(stage0a1_pooled)}")
    stage0a1_summary = summarize_pooled(stage0a1_pooled)
    stage0a1_decision = select_candidates(stage0a1_summary, include_window=False)
    write_outputs(stage0a1_pooled, stage0a1_per_ticker, stage0a1_summary, (
        "stage0a1_pooled", "stage0a1_per_ticker", "stage0a1_summary"
    ))
    print(json.dumps(stage0a1_decision, indent=2))
    display(stage0a1_summary.sort_values(["basic_gate", "macro_f1_mean"], ascending=[False, False]))
else:
    print("RUN_STAGE0A1 is False; Stage 0A1 not run.")
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Stage 0A2 - Window Sensitivity

Frozen grid:

```text
model        = LightGBM
label_feature_pairs = union(mean_label_feature, lcb_label_feature)
window_size  = 5, 10, 20
seeds        = 101, 202, 303, 404, 505
```

Expected pooled model-seed rows: `15` or `30`. If Stage 0A1 produces zero
basic-gate cells, do not run Stage 0A2.

The word `candidates` here means "allowed by the frozen Stage 0A rules", not a
final thesis claim.
"""
        )
    )
    cells.append(
        code_cell(
            """
stage0a2_decision = None

if RUN_STAGE0A2:
    if stage0a1_decision is None:
        raise RuntimeError("Run Stage 0A1 in this session before Stage 0A2.")
    if stage0a1_decision["stage0_result"] == "do_not_decide_config":
        stage0a2_decision = {
            "stage0_result": "do_not_decide_config",
            "reason": "zero Stage 0A1 cells passed basic_gate",
            "candidate_count": 0,
        }
        print(json.dumps(stage0a2_decision, indent=2))
    else:
        label_feature_pairs = stage0a1_decision["candidates"]
        stage0a2_specs = [
            {
                "stage": "Stage 0A2",
                "model": "lightgbm",
                "label_config": pair["label_config"],
                "feature_set": pair["feature_set"],
                "window_size": window_size,
                "seed": seed,
            }
            for pair in label_feature_pairs
            for window_size in STAGE0A2_WINDOW_SIZES
            for seed in MODEL_SEEDS
        ]
        expected_rows = len(label_feature_pairs) * len(STAGE0A2_WINDOW_SIZES) * len(MODEL_SEEDS)
        if expected_rows not in (15, 30):
            raise ValueError(f"Stage 0A2 expected 15 or 30 rows by design, got planned {expected_rows}")
        stage0a2_pooled, stage0a2_per_ticker = run_stage_grid("Stage 0A2", stage0a2_specs)
        if len(stage0a2_pooled) != expected_rows:
            raise ValueError(f"Stage 0A2 expected {expected_rows} pooled rows, found {len(stage0a2_pooled)}")
        stage0a2_summary = summarize_pooled(stage0a2_pooled)
        stage0a2_decision = select_candidates(stage0a2_summary, include_window=True)
        write_outputs(stage0a2_pooled, stage0a2_per_ticker, stage0a2_summary, (
            "stage0a2_pooled", "stage0a2_per_ticker", "stage0a2_summary"
        ))
        with OUTPUT_FILES["stage0_candidates"].open("w", encoding="utf-8") as handle:
            json.dump(stage0a2_decision, handle, indent=2)
        print(json.dumps(stage0a2_decision, indent=2))
        display(stage0a2_summary.sort_values(["basic_gate", "macro_f1_mean"], ascending=[False, False]))
else:
    print("RUN_STAGE0A2 is False; Stage 0A2 not run.")
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Stage 0B - Deep-Model Second-View

Frozen grid:

```text
candidates = mean_candidate, lcb_candidate
models     = LogReg, LightGBM, simple GRU, MS-DLinear+TCN
seeds      = 101, 202, 303, 404, 505
```

Expected pooled model-seed rows: max `40`. Expected deep-model runs: max `20`.

Stage 0B is not a second selection pass. If deep models disagree, mark
`deep_model_disagrees=True`; do not rerank or retract Stage 0A candidates.

The `lcb_candidate` field may be null. It is only populated when measured
validation rows pass the pre-registered LCB gate.
"""
        )
    )
    cells.append(
        code_cell(
            """
if RUN_STAGE0B:
    if stage0a2_decision is None:
        raise RuntimeError("Run Stage 0A2 in this session before Stage 0B.")
    if stage0a2_decision["stage0_result"] == "do_not_decide_config":
        print("Stage 0 result is do_not_decide_config; Stage 0B must not run.")
    else:
        candidates = stage0a2_decision["candidates"]
        models = STAGE0B_MODELS if RUN_STAGE0B_DEEP_MODELS else ("logreg", "lightgbm")
        if not RUN_STAGE0B_DEEP_MODELS:
            print("RUN_STAGE0B_DEEP_MODELS is False; running tabular Stage 0B shape only.")
        else:
            first_candidate = candidates[0]
            input_dim = len(FEATURE_SETS[first_candidate["feature_set"]])
            run_torch_shape_smoke(input_dim=input_dim, window_size=int(first_candidate["window_size"]))
        stage0b_specs = [
            {
                "stage": "Stage 0B",
                "model": model,
                "label_config": candidate["label_config"],
                "feature_set": candidate["feature_set"],
                "window_size": int(candidate["window_size"]),
                "seed": seed,
            }
            for candidate in candidates
            for model in models
            for seed in MODEL_SEEDS
        ]
        if len(stage0b_specs) > 40:
            raise ValueError(f"Stage 0B planned too many model-seed rows: {len(stage0b_specs)}")
        deep_runs = sum(1 for spec in stage0b_specs if spec["model"] in {"simple_gru", "ms_dlinear_tcn"})
        if deep_runs > 20:
            raise ValueError(f"Stage 0B planned too many deep model runs: {deep_runs}")
        stage0b_pooled, stage0b_per_ticker = run_stage_grid("Stage 0B", stage0b_specs)
        stage0b_summary = summarize_pooled(stage0b_pooled)
        stage0b_summary["deep_model_disagrees"] = (
            stage0b_summary["model"].isin(["simple_gru", "ms_dlinear_tcn"])
            & ~stage0b_summary["basic_gate"]
        )
        write_outputs(stage0b_pooled, stage0b_per_ticker, stage0b_summary, (
            "stage0b_pooled", "stage0b_per_ticker", "stage0b_summary"
        ))
        display(stage0b_summary)
else:
    print("RUN_STAGE0B is False; Stage 0B not run.")
"""
        )
    )

    cells.append(
        markdown_cell(
            """
## Interpretation Boundary

Use `delta_macro_f1_vs_dummy` and `delta_balanced_accuracy_vs_dummy` for
absolute signal over the stratified dummy comparator. Use
`positive_ticker_count` and `top_ticker_gain_share` to detect ticker
concentration.

Candidate rules:

```text
macro_f1_lcb_95 = macro_f1_mean - t_critical_one_sided_95(seed_count)
                  * macro_f1_std / sqrt(seed_count)
basic_gate := delta_macro_f1_vs_dummy_mean > 0
              AND macro_f1_lcb_95 > dummy_macro_f1_mean
lcb_eligible := basic_gate
                   AND delta_balanced_accuracy_vs_dummy_mean > 0
                   AND top_ticker_gain_share < 0.50
                   AND positive_ticker_count >= 3
mean_candidate = argmax over basic_gate cells of macro_f1_mean
lcb_candidate = argmax over lcb_eligible cells of macro_f1_lcb_95
```

If zero Stage 0A cells pass `basic_gate`, the Stage 0 result is
`do_not_decide_config`; do not run Stage 0B and do not start Stage 1.

`lcb_eligible` and `lcb_candidate` are rule names. They are not claims until
the corresponding validation rows satisfy the rule.

Post-run interpretation guardrails:

- A selected row is a validation-selected record, not expected holdout/test
  performance. Do not describe the selected macro F1 as a promised future
  score.
- If Stage 0A2 margins between neighboring windows are below about 0.002 macro
  F1, describe the chosen window as protocol-selected rather than clearly
  dominant.
- If LogReg and LightGBM are close in Stage 0B, describe the signal as
  tabular/linear-detectable under fixed defaults. Do not conclude that deep
  model families are ineffective from Stage 0B alone.
- Inspect `stage0b_pooled.csv` for LogReg `fit_status` or convergence warnings
  before relying on the LogReg number in downstream wording.
- Inspect per-ticker CSVs before making ticker-breadth claims beyond the
  displayed summary.
"""
        )
    )

    nb.cells = cells
    return nb


def main() -> None:
    nb = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
