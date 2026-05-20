# %% [markdown]
# # P1B.11b Candidate A multi-seed LSTM no-trade-band rerun
#
# Generated locally for Colab execution.
#
# Do not commit experiment outputs.
#
# Requires Google Drive data at `/content/drive/MyDrive/stockdata/`.
#
# Requires GitHub repo `zkc768/hf_stock_clf`.
#
# Uses seeds=[42, 43, 44].
#
# No TCN/DLinear/Notebook 03.
#
# This file is a Colab-ready code draft for manual upload. It does not contain
# final experiment results.

# %%
from __future__ import annotations

import gc
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, time as dtime, timezone
from pathlib import Path

from google.colab import drive
from IPython.display import display
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# %% [markdown]
# ## 1. Title / Scope / Warnings

# %%
RUN_START_TIME = datetime.now(timezone.utc)

SELECTION_BIAS_DISCLOSURE = (
    "No-trade-band binary classification estimates P(sign(r) | X, |r| > tau), "
    "not P(sign(r) | X). Retained-subset metrics must be interpreted together "
    "with coverage/window retention."
)

print("P1B.11b Candidate A multi-seed LSTM no-trade-band rerun")
print("Generated locally for Colab execution.")
print("Do not commit experiment outputs.")
print("Requires Google Drive data at /content/drive/MyDrive/stockdata/.")
print("Requires GitHub repo zkc768/hf_stock_clf.")
print("Uses seeds=[42, 43, 44].")
print("No TCN/DLinear/Notebook 03.")
print(SELECTION_BIAS_DISCLOSURE)


# %% [markdown]
# ## 2. Drive Mount

# %%
# Mount Drive only when running inside a notebook kernel (not in a subprocess).
# When called via subprocess.run, get_ipython() returns None and drive.mount()
# crashes with AttributeError on the kernel.
try:
    from IPython import get_ipython

    if get_ipython() is not None:
        drive.mount("/content/drive")
    else:
        print("Running outside IPython kernel; Drive assumed already mounted.")
except Exception:
    print("Running outside IPython kernel; Drive assumed already mounted.")


# %% [markdown]
# ## 3. Repo Clone / Pull / Commit Guard

# %%
REPO_URL = "https://github.com/zkc768/hf_stock_clf.git"
REPO_DIR = Path("/content/hf_stock_clf")
BRANCH = "master"
REQUIRED_COMMIT = "208d1e3"
REQUIRED_COMMITS = {
    "e2e2869": "fix(dataset): align window labels to prediction point",
    "fc7b863": "docs(phase1b): record label-alignment fixed smoke",
    "208d1e3": "docs(phase1b): record LSTM rerun results",
}
SENSITIVE_TOKENS = []


def get_clone_url() -> tuple[str, str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    try:
        from google.colab import userdata

        colab_token = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
        if colab_token:
            token = colab_token
    except (ImportError, KeyError) as exc:
        print(
            "No Colab GH_TOKEN/GITHUB_TOKEN secret available; using env token or public clone "
            f"({type(exc).__name__})."
        )

    if token:
        SENSITIVE_TOKENS.append(token)
        actual_url = f"https://x-access-token:{token}@github.com/zkc768/hf_stock_clf.git"
        return actual_url, REPO_URL

    return REPO_URL, REPO_URL


def sanitize_command_output(output: str) -> str:
    sanitized = output
    sanitized = re.sub(
        r"https://x-access-token:[^@\s]+@github\.com/zkc768/hf_stock_clf\.git",
        REPO_URL,
        sanitized,
    )
    for token in SENSITIVE_TOKENS:
        if token:
            sanitized = sanitized.replace(token, "<redacted-token>")
    for token_name in ["GH_TOKEN", "GITHUB_TOKEN"]:
        token = os.environ.get(token_name)
        if token:
            sanitized = sanitized.replace(token, "<redacted-token>")
    sanitized = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:<redacted-token>@", sanitized)
    return sanitized


def run_command(
    command: list[str],
    cwd: Path | None = None,
    safe_command: list[str] | None = None,
) -> str:
    display_command = safe_command if safe_command is not None else command
    print(f"$ {' '.join(str(part) for part in display_command)}")
    # GIT_TERMINAL_PROMPT=0 prevents git from hanging on interactive auth prompts
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ | {"GIT_TERMINAL_PROMPT": "0"},
    )
    output = completed.stdout or ""
    safe_output = sanitize_command_output(output)
    if safe_output.strip():
        print(safe_output.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"exit_code: {completed.returncode}\n"
            f"command: {' '.join(str(part) for part in display_command)}\n"
            f"output:\n{safe_output}"
        )
    return output


if REPO_DIR.exists():
    print(f"Removing old clone at {REPO_DIR}")
    shutil.rmtree(REPO_DIR)

clone_url, safe_clone_url = get_clone_url()

run_command(
    ["git", "clone", clone_url, str(REPO_DIR)],
    safe_command=["git", "clone", safe_clone_url, str(REPO_DIR)],
)
run_command(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR)
run_command(["git", "checkout", "-B", BRANCH, f"origin/{BRANCH}"], cwd=REPO_DIR)
run_command(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO_DIR)

GIT_LOG_ONELINE_5 = run_command(["git", "log", "--oneline", "-5"], cwd=REPO_DIR)
GIT_COMMIT_HASH = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).strip()

for commit_hash, description in REQUIRED_COMMITS.items():
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_hash}^{{commit}}"],
        cwd=str(REPO_DIR),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"Commit exists check: {commit_hash} {exists.returncode == 0} ({description})")
    if commit_hash == REQUIRED_COMMIT and exists.returncode != 0:
        raise RuntimeError(f"Required commit {REQUIRED_COMMIT} is missing from local history.")

ancestor = subprocess.run(
    ["git", "merge-base", "--is-ancestor", REQUIRED_COMMIT, "HEAD"],
    cwd=str(REPO_DIR),
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
if ancestor.returncode != 0:
    raise RuntimeError(
        f"Required commit {REQUIRED_COMMIT} is not an ancestor of HEAD={GIT_COMMIT_HASH}."
    )
print(f"PASS: required commit {REQUIRED_COMMIT} exists and is in HEAD history.")

# Keep token-authenticated URL so re-runs don't fail on private repos
run_command(
    ["git", "remote", "set-url", "origin", clone_url],
    cwd=REPO_DIR,
    safe_command=["git", "remote", "set-url", "origin", safe_clone_url],
)
run_command(["git", "remote", "-v"], cwd=REPO_DIR)

print(f"Repo guard PASS: HEAD={GIT_COMMIT_HASH}")


# %% [markdown]
# ## 4. Import Cache Guard

# %%
for module_name in list(sys.modules):
    if module_name == "ml_utils" or module_name.startswith("ml_utils."):
        del sys.modules[module_name]

repo_path = str(REPO_DIR)
sys.path = [path for path in sys.path if path != repo_path]
sys.path.insert(0, repo_path)

import ml_utils
from ml_utils.checkpoint import load_checkpoint
from ml_utils.config import DataConfig
from ml_utils.dataset import WindowedClassificationDataset
from ml_utils.dataset import fit_scaler_on_train
from ml_utils.dataset import make_no_trade_band_labels
from ml_utils.dataset import make_time_splits
from ml_utils.dataset import transform_split
from ml_utils.dataset import trim_labels_at_split_boundary
from ml_utils.metrics import always_predict_baseline_metrics
from ml_utils.metrics import dummy_baseline_metrics
from ml_utils.models.lstm_classifier import LSTMClassifier
from ml_utils.seed import seed_everything
from ml_utils.trainer import Trainer
from ml_utils.trainer import evaluate

dataset_source = inspect.getfile(WindowedClassificationDataset)
print(f"ml_utils imported from: {ml_utils.__file__}")
print(f"WindowedClassificationDataset imported from: {dataset_source}")
assert dataset_source.startswith(str(REPO_DIR)), dataset_source
assert "ml_utils.models.tcn_classifier" not in sys.modules
assert "ml_utils.models.dlinear_classifier" not in sys.modules
print("Import cache guard PASS: using fresh Colab clone ml_utils.")
print("No TCN/DLinear imports needed for this run.")


# %% [markdown]
# ## 5. Config

# %%
CONFIG = {
    "seeds": [42, 43, 44],
    "tickers": ["CSCO", "JPM", "KO", "MSFT", "WMT"],
    "raw_data_dir": "/content/drive/MyDrive/stockdata/Dow_30_1min",
    "output_dir": "/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_p1b11b_candidate_a_multiseed",
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "model_name": "LSTMClassifier",
    "num_epochs": 20,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "hidden_size": 64,
    "dropout": 0.20,
    "weight_decay": 0.0,
    "early_stop_patience": 5,
    "num_layers": 2,
    "num_classes": 2,
    "num_workers": 0,
    "scaler_type": "standard",
    "reset_output_dir_on_start": False,
    "reset_candidate_dir_on_rerun": True,
}

CANDIDATES = {
    "A": {"name": "main", "window_size": 12, "label_horizon_k": 12, "threshold_bps": 5},
}

TIMESTAMP_COL = "timestamp"
TICKER_COL = "ticker"
LABEL_COL = "label"
PRICE_COL = "close"
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_pctb",
    "rolling_std_20",
    "obv_roc",
]

RAW_COLUMNS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)
SUSPICIOUS_F1_THRESHOLD = 0.95
SUSPICIOUS_DELTA_THRESHOLD = 0.30

RAW_DATA_DIR = Path(CONFIG["raw_data_dir"])
OUTPUT_BASE_DIR = Path(CONFIG["output_dir"])
P1B11B_RUN_ID = f"run_{RUN_START_TIME.strftime('%Y%m%dT%H%M%SZ')}"
OUTPUT_DIR = OUTPUT_BASE_DIR / P1B11B_RUN_ID
REQUIRED_ARTIFACT_NAMES = [
    "per_seed_ticker_results.csv",
    "per_seed_summary.csv",
    "overall_multiseed_summary.csv",
    "run_manifest.json",
    "window_count_check.json",
]
OLD_OUTPUT_DIRS = {
    Path("/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs"),
    Path("/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_LEAKY_INVALID"),
    Path("/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed"),
    Path("/content/drive/MyDrive/stockdata/phase1b_lstm_rerun_outputs_alignment_fixed_full_ad"),
}
EXPECTED_POOLED_WINDOW_COUNTS = {"train": 213384, "val": 11903, "test": 19190}
assert OUTPUT_BASE_DIR not in OLD_OUTPUT_DIRS
assert str(OUTPUT_BASE_DIR).endswith("phase1b_lstm_rerun_p1b11b_candidate_a_multiseed")

if CONFIG.get("reset_output_dir_on_start", True) and OUTPUT_DIR.exists():
    print(f"Resetting current P1B.11b output directory before run: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
existing_artifacts = [
    str(OUTPUT_DIR / artifact_name)
    for artifact_name in REQUIRED_ARTIFACT_NAMES
    if (OUTPUT_DIR / artifact_name).exists()
]
if existing_artifacts:
    raise FileExistsError(
        "Fresh output guard failed. Refusing to overwrite or append existing artifacts: "
        f"{existing_artifacts}"
    )

DATA_CONFIG = DataConfig(
    tickers=CONFIG["tickers"],
    data_dir=str(RAW_DATA_DIR),
    timestamp_col=TIMESTAMP_COL,
    price_col=PRICE_COL,
    label_mode="no_trade_band",
    threshold_bps=5.0,
    feature_cols=FEATURE_COLS,
    train_ratio=CONFIG["train_ratio"],
    val_ratio=CONFIG["val_ratio"],
    timezone_policy="naive",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training config: {json.dumps(CONFIG, indent=2)}")
print(f"Candidate grid: {json.dumps(CANDIDATES, indent=2)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Device: {DEVICE}")
print("Seed policy PASS: seeds=[42, 43, 44].")


# %% [markdown]
# ## 6. Utility Functions

# %%
def candidate_output_dir(candidate_id: str) -> Path:
    output_dir = OUTPUT_DIR / f"candidate_{candidate_id}"
    if output_dir.exists() and any(output_dir.iterdir()):
        if CONFIG.get("reset_candidate_dir_on_rerun", True):
            print(f"Resetting existing candidate output directory before rerun: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Candidate output directory is not fresh and will not be appended: {output_dir}"
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Candidate output directory is not fresh and will not be appended: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def class_distribution(labels: np.ndarray) -> dict[str, float | int]:
    count = int(labels.shape[0])
    up = int(np.sum(labels == 1))
    down = int(np.sum(labels == 0))
    return {
        "n": count,
        "up": up,
        "down": down,
        "up_pct": float(up / count) if count else np.nan,
        "down_pct": float(down / count) if count else np.nan,
    }


def collect_dataset_labels(dataset: WindowedClassificationDataset) -> np.ndarray:
    labels = [int(dataset[index][1].item()) for index in range(len(dataset))]
    return np.asarray(labels, dtype=np.int64)


def make_loader(
    dataset: WindowedClassificationDataset,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=shuffle,
        num_workers=CONFIG["num_workers"],
    )


def to_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=to_jsonable), encoding="utf-8")


def print_candidate_header(candidate_id: str, candidate: dict, seed: int | None = None) -> None:
    seed_text = f"seed={seed}" if seed is not None else f"seeds={CONFIG['seeds']}"
    print("\n" + "=" * 92)
    print(
        f"Candidate {candidate_id} | {candidate['name']} | "
        f"window_size={candidate['window_size']} | "
        f"label_horizon_k={candidate['label_horizon_k']} | "
        f"threshold_bps={candidate['threshold_bps']} | {seed_text}"
    )
    print("=" * 92)


# %% [markdown]
# ## 7. Data Loading and 5-Min Resampling

# %%
def raw_path_for_ticker(ticker: str) -> Path:
    txt_path = RAW_DATA_DIR / f"{ticker}.txt"
    if txt_path.exists():
        return txt_path
    raise FileNotFoundError(f"Missing raw 1-min txt file: {txt_path}")


def load_one_minute_data(ticker: str) -> pd.DataFrame:
    input_path = raw_path_for_ticker(ticker)
    frame = pd.read_csv(input_path, header=None, names=RAW_COLUMNS)
    frame = frame[frame["Date"].astype(str).str.lower() != "date"].reset_index(drop=True)
    frame[TIMESTAMP_COL] = pd.to_datetime(
        frame["Date"].astype(str) + " " + frame["Time"].astype(str),
        format="%m/%d/%Y %H:%M",
    )
    frame = frame.drop(columns=["Date", "Time"])
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    numeric_cols = ["open", "high", "low", "close", "volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="raise")
    times = frame[TIMESTAMP_COL].dt.time
    regular_hours = frame[(times >= MARKET_OPEN) & (times <= MARKET_CLOSE)]
    return regular_hours[[TIMESTAMP_COL, *numeric_cols]].reset_index(drop=True)


def resample_to_five_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    resampled = (
        frame.set_index(TIMESTAMP_COL)
        .resample("5min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close", "volume"])
        .reset_index()
    )
    times = resampled[TIMESTAMP_COL].dt.time
    return resampled[(times >= MARKET_OPEN) & (times <= MARKET_CLOSE)].reset_index(drop=True)


five_minute_frames = {}
for ticker in CONFIG["tickers"]:
    one_minute = load_one_minute_data(ticker)
    five_minute = resample_to_five_minutes(one_minute)
    five_minute_frames[ticker] = five_minute
    print(
        f"{ticker}: 5m_rows={len(five_minute):,} "
        f"start={five_minute[TIMESTAMP_COL].iloc[0]} "
        f"end={five_minute[TIMESTAMP_COL].iloc[-1]}"
    )


# %% [markdown]
# ## 8. Technical Indicator Construction

# %%
def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy(deep=True)

    ema12 = result["close"].ewm(span=12, adjust=False).mean()
    ema26 = result["close"].ewm(span=26, adjust=False).mean()
    result["macd"] = ema12 - ema26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    close_delta = result["close"].diff()
    gain = close_delta.clip(lower=0)
    loss = (-close_delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    result["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    rolling_mean_20 = result["close"].rolling(20).mean()
    rolling_std_price_20 = result["close"].rolling(20).std()
    upper_band = rolling_mean_20 + 2.0 * rolling_std_price_20
    lower_band = rolling_mean_20 - 2.0 * rolling_std_price_20
    result["bb_pctb"] = (result["close"] - lower_band) / (upper_band - lower_band)
    result["rolling_std_20"] = result["close"].pct_change().rolling(20).std()

    obv_direction = np.sign(result["close"].diff()).copy()
    obv_direction.iloc[0] = 0
    obv = (obv_direction * result["volume"]).cumsum()
    result["obv_roc"] = obv.pct_change(5)

    result[FEATURE_COLS] = result[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=FEATURE_COLS).reset_index(drop=True)


feature_frames = {}
for ticker in CONFIG["tickers"]:
    before_rows = len(five_minute_frames[ticker])
    feature_frame = add_technical_indicators(five_minute_frames[ticker])
    feature_frame[TICKER_COL] = ticker
    feature_frames[ticker] = feature_frame
    print(
        f"{ticker}: 5m_rows={before_rows:,} ti_rows={len(feature_frame):,} "
        f"ti_drop={before_rows - len(feature_frame):,} "
        f"feature_nan={int(feature_frame[FEATURE_COLS].isna().sum().sum())}"
    )

del five_minute_frames
gc.collect()


# %% [markdown]
# ## 9. Candidate Dataset Construction

# %%
def make_window_dataset(
    frame: pd.DataFrame,
    window_size: int,
    label_horizon_k: int,
) -> WindowedClassificationDataset:
    return WindowedClassificationDataset(
        df=frame,
        feature_cols=FEATURE_COLS,
        label_col=LABEL_COL,
        ticker_col=TICKER_COL,
        timestamp_col=TIMESTAMP_COL,
        window_size=window_size,
        label_horizon_k=label_horizon_k,
        stride=1,
    )


def label_frame_for_candidate(ticker: str, candidate: dict) -> tuple[pd.DataFrame, dict]:
    labeled_frame, diagnostics = make_no_trade_band_labels(
        feature_frames[ticker],
        price_col=PRICE_COL,
        k=int(candidate["label_horizon_k"]),
        threshold_bps=float(candidate["threshold_bps"]),
        timestamp_col=TIMESTAMP_COL,
    )
    retained = diagnostics["n_up"] + diagnostics["n_down"]
    retained_pct = retained / diagnostics["n_total"] if diagnostics["n_total"] else np.nan
    print(
        f"{ticker}: labels up={diagnostics['n_up']:,} down={diagnostics['n_down']:,} "
        f"neutral={diagnostics['n_neutral']:,} cross_day={diagnostics['n_cross_day']:,} "
        f"retained_pct={retained_pct:.4%}"
    )
    return labeled_frame, diagnostics


def split_trim_scale_candidate(candidate_id: str, candidate: dict) -> dict:
    labeled_frames = {}
    label_diagnostics = {}
    train_frames = {}
    val_frames = {}
    test_frames = {}
    horizon = int(candidate["label_horizon_k"])

    for ticker in CONFIG["tickers"]:
        labeled_frames[ticker], label_diagnostics[ticker] = label_frame_for_candidate(
            ticker,
            candidate,
        )
        train_frame, val_frame, test_frame = make_time_splits(
            labeled_frames[ticker],
            train_ratio=DATA_CONFIG.train_ratio,
            val_ratio=DATA_CONFIG.val_ratio,
            timestamp_col=TIMESTAMP_COL,
            timezone_policy=DATA_CONFIG.timezone_policy,
        )
        train_frames[ticker] = trim_labels_at_split_boundary(
            train_frame,
            label_horizon_k=horizon,
            timestamp_col=TIMESTAMP_COL,
        )
        val_frames[ticker] = trim_labels_at_split_boundary(
            val_frame,
            label_horizon_k=horizon,
            timestamp_col=TIMESTAMP_COL,
        )
        test_frames[ticker] = trim_labels_at_split_boundary(
            test_frame,
            label_horizon_k=horizon,
            timestamp_col=TIMESTAMP_COL,
        )

    pooled_train_frame = pd.concat(
        [train_frames[ticker] for ticker in CONFIG["tickers"]],
        ignore_index=True,
    )
    scaler = fit_scaler_on_train(
        pooled_train_frame,
        DATA_CONFIG.feature_cols,
        scaler_type=CONFIG["scaler_type"],
    )

    transformed = {"train": {}, "val": {}, "test": {}}
    for ticker in CONFIG["tickers"]:
        transformed["train"][ticker] = transform_split(
            train_frames[ticker],
            scaler,
            DATA_CONFIG.feature_cols,
        )
        transformed["val"][ticker] = transform_split(
            val_frames[ticker],
            scaler,
            DATA_CONFIG.feature_cols,
        )
        transformed["test"][ticker] = transform_split(
            test_frames[ticker],
            scaler,
            DATA_CONFIG.feature_cols,
        )

    window_size = int(candidate["window_size"])
    datasets = {"train": {}, "val": {}, "test": {}}
    labels = {"train": {}, "val": {}, "test": {}}
    for split_name in ["train", "val", "test"]:
        for ticker in CONFIG["tickers"]:
            dataset = make_window_dataset(
                transformed[split_name][ticker],
                window_size=window_size,
                label_horizon_k=horizon,
            )
            if len(dataset) == 0:
                raise ValueError(
                    f"Candidate {candidate_id} has zero {split_name} windows for {ticker}"
                )
            datasets[split_name][ticker] = dataset
            labels[split_name][ticker] = collect_dataset_labels(dataset)

    pooled_train_dataset = make_window_dataset(
        pd.concat(
            [transformed["train"][ticker] for ticker in CONFIG["tickers"]],
            ignore_index=True,
        ),
        window_size=window_size,
        label_horizon_k=horizon,
    )
    pooled_val_dataset = make_window_dataset(
        pd.concat(
            [transformed["val"][ticker] for ticker in CONFIG["tickers"]],
            ignore_index=True,
        ),
        window_size=window_size,
        label_horizon_k=horizon,
    )
    if len(pooled_train_dataset) == 0 or len(pooled_val_dataset) == 0:
        raise ValueError(f"Candidate {candidate_id} has empty pooled train/val dataset")

    for ticker in CONFIG["tickers"]:
        train_dist = class_distribution(labels["train"][ticker])
        val_dist = class_distribution(labels["val"][ticker])
        test_dist = class_distribution(labels["test"][ticker])
        print(
            f"{candidate_id} {ticker}: "
            f"train_windows={train_dist['n']:,} up={train_dist['up']:,} down={train_dist['down']:,} | "
            f"val_windows={val_dist['n']:,} up={val_dist['up']:,} down={val_dist['down']:,} | "
            f"test_windows={test_dist['n']:,} up={test_dist['up']:,} down={test_dist['down']:,}"
        )

    pooled_train_labels = collect_dataset_labels(pooled_train_dataset)
    pooled_val_labels = collect_dataset_labels(pooled_val_dataset)
    pooled_window_counts = {
        "train": len(pooled_train_dataset),
        "val": len(pooled_val_dataset),
        "test": sum(len(datasets["test"][ticker]) for ticker in CONFIG["tickers"]),
    }
    print(
        f"{candidate_id}: pooled train_windows={pooled_window_counts['train']:,} "
        f"val_windows={pooled_window_counts['val']:,} "
        f"test_windows={pooled_window_counts['test']:,}"
    )

    return {
        "candidate_id": candidate_id,
        "candidate": candidate,
        "label_diagnostics": label_diagnostics,
        "datasets": datasets,
        "labels": labels,
        "pooled_train_dataset": pooled_train_dataset,
        "pooled_val_dataset": pooled_val_dataset,
        "pooled_train_labels": pooled_train_labels,
        "pooled_val_labels": pooled_val_labels,
        "pooled_window_counts": pooled_window_counts,
    }


def assert_expected_window_counts(seed: int, prepared: dict) -> dict[str, int]:
    observed = prepared["pooled_window_counts"]
    expected = EXPECTED_POOLED_WINDOW_COUNTS
    if observed != expected:
        raise RuntimeError(
            "Candidate A pooled window counts differ from P1B.10. "
            "This indicates pipeline drift from P1B.10. "
            f"seed={seed} observed train/val/test="
            f"{observed['train']}/{observed['val']}/{observed['test']} "
            f"expected train/val/test={expected['train']}/{expected['val']}/{expected['test']}"
        )
    print(
        f"Window count guard PASS seed={seed}: "
        f"train={observed['train']:,} val={observed['val']:,} test={observed['test']:,}"
    )
    return observed


# %% [markdown]
# ## 10. Baseline Evaluation

# %%
def baseline_summary(y_train: np.ndarray, y_eval: np.ndarray, seed: int) -> dict[str, float]:
    stratified_scores = []
    for offset in range(10):
        metrics = dummy_baseline_metrics(
            y_train=y_train,
            y_test=y_eval,
            strategy="stratified",
            random_state=seed + offset,
        )
        stratified_scores.append(metrics["macro_f1"])

    prior_metrics = dummy_baseline_metrics(
        y_train=y_train,
        y_test=y_eval,
        strategy="prior",
        random_state=seed,
    )
    always_up_metrics = always_predict_baseline_metrics(y_eval, constant_label=1)
    always_down_metrics = always_predict_baseline_metrics(y_eval, constant_label=0)
    return {
        "dummy_stratified_macro_f1_mean": float(np.mean(stratified_scores)),
        "dummy_stratified_macro_f1_std": float(np.std(stratified_scores)),
        "dummy_prior_macro_f1": float(prior_metrics["macro_f1"]),
        "always_up_macro_f1": float(always_up_metrics["macro_f1"]),
        "always_down_macro_f1": float(always_down_metrics["macro_f1"]),
    }


def split_distribution_columns(
    prepared: dict,
    ticker: str,
) -> dict[str, float | int]:
    train_dist = class_distribution(prepared["labels"]["train"][ticker])
    val_dist = class_distribution(prepared["labels"]["val"][ticker])
    test_dist = class_distribution(prepared["labels"]["test"][ticker])
    return {
        "n_train_windows": train_dist["n"],
        "n_val_windows": val_dist["n"],
        "n_test_windows": test_dist["n"],
        "train_windows": train_dist["n"],
        "val_windows": val_dist["n"],
        "test_windows": test_dist["n"],
        "train_up_pct": train_dist["up_pct"],
        "train_down_pct": train_dist["down_pct"],
        "val_up_pct": val_dist["up_pct"],
        "val_down_pct": val_dist["down_pct"],
        "test_up_pct": test_dist["up_pct"],
        "test_down_pct": test_dist["down_pct"],
    }


# %% [markdown]
# ## 11. LSTM Training and Evaluation

# %%
def build_lstm() -> LSTMClassifier:
    return LSTMClassifier(
        input_size=len(DATA_CONFIG.feature_cols),
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"],
    )


def evaluate_ticker_rows(
    model: LSTMClassifier,
    criterion: nn.Module,
    prepared: dict,
    candidate_output: Path,
    seed: int,
    best_epoch: int,
    best_val_macro_f1: float,
    training_time_seconds: float,
    val_macro_f1: float,
) -> list[dict]:
    rows = []
    candidate_id = prepared["candidate_id"]
    candidate = prepared["candidate"]

    for ticker in CONFIG["tickers"]:
        test_loader = make_loader(prepared["datasets"]["test"][ticker], shuffle=False)
        test_metrics, y_test, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=DEVICE,
        )
        baselines = baseline_summary(
            prepared["labels"]["train"][ticker],
            y_test,
            seed=seed,
        )
        delta = test_metrics["macro_f1"] - baselines["dummy_stratified_macro_f1_mean"]
        suspicious = (
            val_macro_f1 >= SUSPICIOUS_F1_THRESHOLD
            or test_metrics["macro_f1"] >= SUSPICIOUS_F1_THRESHOLD
            or delta >= SUSPICIOUS_DELTA_THRESHOLD
        )
        row = {
            "candidate": candidate_id,
            "candidate_id": candidate_id,
            "candidate_name": candidate["name"],
            "ticker": ticker,
            "seed": seed,
            "window_size": candidate["window_size"],
            "label_horizon_k": candidate["label_horizon_k"],
            "threshold_bps": candidate["threshold_bps"],
            **split_distribution_columns(prepared, ticker),
            "label_n_total": int(prepared["label_diagnostics"][ticker]["n_total"]),
            "label_n_retained": int(
                prepared["label_diagnostics"][ticker]["n_up"]
                + prepared["label_diagnostics"][ticker]["n_down"]
            ),
            "label_retained_pct": float(
                (
                    prepared["label_diagnostics"][ticker]["n_up"]
                    + prepared["label_diagnostics"][ticker]["n_down"]
                )
                / prepared["label_diagnostics"][ticker]["n_total"]
            ),
            "label_n_up": int(prepared["label_diagnostics"][ticker]["n_up"]),
            "label_n_down": int(prepared["label_diagnostics"][ticker]["n_down"]),
            "label_n_neutral": int(prepared["label_diagnostics"][ticker]["n_neutral"]),
            "label_n_cross_day": int(prepared["label_diagnostics"][ticker]["n_cross_day"]),
            "label_n_tail": int(prepared["label_diagnostics"][ticker]["n_tail"]),
            **baselines,
            "model_macro_f1": float(test_metrics["macro_f1"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "model_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "model_precision_macro": float(test_metrics["precision_macro"]),
            "model_recall_macro": float(test_metrics["recall_macro"]),
            "delta_macro_f1_vs_dummy": float(delta),
            "confusion_matrix": json.dumps(test_metrics["confusion_matrix"].tolist()),
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val_macro_f1),
            "val_macro_f1": float(val_macro_f1),
            "training_time_seconds": float(training_time_seconds),
            "suspicious_status": bool(suspicious),
        }
        rows.append(row)
        print(
            f"{candidate_id} {ticker}: "
            f"dummy_stratified={row['dummy_stratified_macro_f1_mean']:.4f}+/-"
            f"{row['dummy_stratified_macro_f1_std']:.4f} "
            f"dummy_prior={row['dummy_prior_macro_f1']:.4f} "
            f"always_up={row['always_up_macro_f1']:.4f} "
            f"always_down={row['always_down_macro_f1']:.4f} "
            f"lstm_macro_f1={row['model_macro_f1']:.4f} "
            f"balanced_accuracy={row['model_balanced_accuracy']:.4f} "
            f"delta={row['delta_macro_f1_vs_dummy']:.4f} "
            f"confusion_matrix={row['confusion_matrix']}"
        )

    pd.DataFrame(rows).to_csv(candidate_output / f"results_{candidate_id}.csv", index=False)
    return rows


def train_candidate(prepared: dict, candidate_output: Path, seed: int) -> list[dict]:
    seed_everything(seed, deterministic=True)
    model = build_lstm()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(prepared["pooled_train_dataset"], shuffle=True)
    val_loader = make_loader(prepared["pooled_val_dataset"], shuffle=False)
    checkpoint_dir = candidate_output / "checkpoints" / f"seed_{seed}"

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,
        device=DEVICE,
        checkpoint_dir=str(checkpoint_dir),
        monitor_metric="val_macro_f1",
        monitor_mode="max",
        early_stop_patience=CONFIG["early_stop_patience"],
        grad_clip=None,
        verbose=True,
    )

    started = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG["num_epochs"],
    )
    elapsed = time.time() - started

    load_checkpoint(
        path=str(checkpoint_dir / "best.pt"),
        model=model,
        optimizer=None,
        scheduler=None,
        device=DEVICE,
        weights_only=True,
    )

    val_metrics, _, _ = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=DEVICE,
    )
    best_epoch = history["best_epoch"]
    best_val_macro_f1 = history["best_metric"]
    print(
        f"{prepared['candidate_id']}: best_epoch={best_epoch} "
        f"best_val_macro_f1={best_val_macro_f1:.4f} "
        f"val_macro_f1_loaded_best={val_metrics['macro_f1']:.4f} "
        f"training_time_seconds={elapsed:.1f}"
    )

    rows = evaluate_ticker_rows(
        model=model,
        criterion=criterion,
        prepared=prepared,
        candidate_output=candidate_output,
        seed=seed,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro_f1,
        training_time_seconds=elapsed,
        val_macro_f1=val_metrics["macro_f1"],
    )

    del model, optimizer, criterion, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return rows


# %% [markdown]
# ## 12. Candidate A Multi-Seed Loop

# %%
all_rows = []
window_count_check = {
    "expected_pooled_counts": EXPECTED_POOLED_WINDOW_COUNTS,
    "pooled_counts_by_seed": {},
}
MANIFEST = {
    "phase": "P1B.11b",
    "source": "checkpoints/p1b10_colab_code/p1b10_lstm_full_ad_colab.py",
    "source_of_truth": "P1B.10",
    "git_head": GIT_COMMIT_HASH,
    "git_commit_hash": GIT_COMMIT_HASH,
    "git_log_oneline_5": GIT_LOG_ONELINE_5,
    "required_commit": REQUIRED_COMMIT,
    "candidate": "A",
    "window_size": CANDIDATES["A"]["window_size"],
    "label_horizon_k": CANDIDATES["A"]["label_horizon_k"],
    "threshold_bps": CANDIDATES["A"]["threshold_bps"],
    "seeds": CONFIG["seeds"],
    "model_config": {
        "model_name": CONFIG["model_name"],
        "hidden_size": CONFIG["hidden_size"],
        "num_layers": CONFIG["num_layers"],
        "dropout": CONFIG["dropout"],
        "learning_rate": CONFIG["learning_rate"],
        "batch_size": CONFIG["batch_size"],
        "num_epochs": CONFIG["num_epochs"],
        "early_stop_patience": CONFIG["early_stop_patience"],
        "weight_decay": CONFIG["weight_decay"],
        "num_classes": CONFIG["num_classes"],
    },
    "train_val_test_split": {
        "train_ratio": CONFIG["train_ratio"],
        "val_ratio": CONFIG["val_ratio"],
        "test_ratio": CONFIG["test_ratio"],
    },
    "regular_hours_filter": {
        "market_open": "09:30",
        "market_close": "16:00",
        "inclusive": True,
    },
    "feature_cols": FEATURE_COLS,
    "output_dir": str(OUTPUT_DIR),
    "raw_data_dir": str(RAW_DATA_DIR),
    "candidate_grid": CANDIDATES,
    "training_config": CONFIG,
    "start_time": RUN_START_TIME.isoformat(),
    "end_time": None,
    "runtime": None,
    "old_leaky_results_invalid_warning": (
        "Old P1B.9 leaky outputs are invalid and must not be used as final results."
    ),
    "p1b9d_smoke_not_final_warning": (
        "P1B.9d Candidate A alignment-fixed smoke is not the P1B.11b multi-seed result."
    ),
    "selection_bias_disclosure": SELECTION_BIAS_DISCLOSURE,
}

candidate_id = "A"
candidate = CANDIDATES[candidate_id]
candidate_dir = candidate_output_dir(candidate_id)
print_candidate_header(candidate_id, candidate)

for seed in CONFIG["seeds"]:
    seed = int(seed)
    print_candidate_header(candidate_id, candidate, seed=seed)
    seed_dir = candidate_dir / f"seed_{seed}"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        if CONFIG.get("reset_candidate_dir_on_rerun", True):
            print(f"Resetting existing seed output directory before rerun: {seed_dir}")
            shutil.rmtree(seed_dir)
        else:
            raise FileExistsError(
                f"Seed output directory is not fresh and will not be appended: {seed_dir}"
            )
    seed_dir.mkdir(parents=True, exist_ok=True)

    prepared_candidate = split_trim_scale_candidate(candidate_id, candidate)
    counts = assert_expected_window_counts(seed, prepared_candidate)
    window_count_check["pooled_counts_by_seed"][str(seed)] = dict(counts)

    print(f"Starting Candidate A seed={seed}")
    candidate_rows = train_candidate(prepared_candidate, seed_dir, seed)
    print(f"Finished Candidate A seed={seed}")
    all_rows.extend(candidate_rows)

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(seed_dir / f"results_{candidate_id}_seed_{seed}.csv", index=False)
    print(f"Saved candidate {candidate_id} seed={seed} artifacts under {seed_dir}")

    if candidate_frame["suspicious_status"].any():
        print("LOUD WARNING: suspicious metric guard triggered.")
        print(candidate_frame[candidate_frame["suspicious_status"]])
        pd.DataFrame(all_rows).to_csv(
            OUTPUT_DIR / "per_seed_ticker_results.csv",
            index=False,
        )
        partial_end_time = datetime.now(timezone.utc)
        MANIFEST["end_time"] = partial_end_time.isoformat()
        MANIFEST["runtime"] = str(partial_end_time - RUN_START_TIME)
        write_json(OUTPUT_DIR / "run_manifest.json", MANIFEST)
        write_json(OUTPUT_DIR / "window_count_check.json", window_count_check)
        raise RuntimeError(
            "Suspicious metric guard triggered. Do not interpret high F1 as success; "
            "pause and inspect leakage."
        )


# %% [markdown]
# ## 13. Save Artifacts

# %%
all_results = pd.DataFrame(all_rows)
required_columns = [
    "candidate",
    "candidate_id",
    "candidate_name",
    "ticker",
    "seed",
    "window_size",
    "label_horizon_k",
    "threshold_bps",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
    "train_windows",
    "val_windows",
    "test_windows",
    "train_up_pct",
    "train_down_pct",
    "val_up_pct",
    "val_down_pct",
    "test_up_pct",
    "test_down_pct",
    "label_n_total",
    "label_n_retained",
    "label_retained_pct",
    "label_n_up",
    "label_n_down",
    "label_n_neutral",
    "label_n_cross_day",
    "label_n_tail",
    "dummy_stratified_macro_f1_mean",
    "dummy_stratified_macro_f1_std",
    "dummy_prior_macro_f1",
    "always_up_macro_f1",
    "always_down_macro_f1",
    "model_macro_f1",
    "balanced_accuracy",
    "model_balanced_accuracy",
    "model_precision_macro",
    "model_recall_macro",
    "delta_macro_f1_vs_dummy",
    "confusion_matrix",
    "best_epoch",
    "best_val_macro_f1",
    "training_time_seconds",
    "suspicious_status",
]
missing_columns = [column for column in required_columns if column not in all_results.columns]
if missing_columns:
    raise ValueError(f"Missing required result columns: {missing_columns}")

all_results = all_results[required_columns + [c for c in all_results.columns if c not in required_columns]]
all_results.to_csv(OUTPUT_DIR / "per_seed_ticker_results.csv", index=False)

per_seed_summary = (
    all_results.groupby(["candidate", "candidate_id", "candidate_name", "seed"], as_index=False)
    .agg(
        mean_model_macro_f1=("model_macro_f1", "mean"),
        mean_dummy_stratified_macro_f1=("dummy_stratified_macro_f1_mean", "mean"),
        mean_delta_macro_f1_vs_dummy=("delta_macro_f1_vs_dummy", "mean"),
        positive_delta_tickers=("delta_macro_f1_vs_dummy", lambda values: int((values > 0).sum())),
        n_tickers=("ticker", "nunique"),
        total_train_windows=("train_windows", "sum"),
        total_val_windows=("val_windows", "sum"),
        total_test_windows=("test_windows", "sum"),
        suspicious_status=("suspicious_status", "max"),
    )
)
overall_multiseed_summary = pd.DataFrame(
    [
        {
            "candidate": "A",
            "candidate_id": "A",
            "candidate_name": CANDIDATES["A"]["name"],
            "mean_delta_across_seeds": float(per_seed_summary["mean_delta_macro_f1_vs_dummy"].mean()),
            "std_delta_across_seeds": float(per_seed_summary["mean_delta_macro_f1_vs_dummy"].std()),
            "mean_model_macro_f1_across_seeds": float(per_seed_summary["mean_model_macro_f1"].mean()),
            "std_model_macro_f1_across_seeds": float(per_seed_summary["mean_model_macro_f1"].std()),
            "positive_delta_tickers_total": int((all_results["delta_macro_f1_vs_dummy"] > 0).sum()),
            "total_seed_ticker_runs": int(len(all_results)),
            "suspicious_status": bool(all_results["suspicious_status"].max()),
        }
    ]
)

per_seed_summary.to_csv(OUTPUT_DIR / "per_seed_summary.csv", index=False)
overall_multiseed_summary.to_csv(OUTPUT_DIR / "overall_multiseed_summary.csv", index=False)

RUN_END_TIME = datetime.now(timezone.utc)
MANIFEST["end_time"] = RUN_END_TIME.isoformat()
MANIFEST["runtime"] = str(RUN_END_TIME - RUN_START_TIME)
write_json(OUTPUT_DIR / "run_manifest.json", MANIFEST)
write_json(OUTPUT_DIR / "window_count_check.json", window_count_check)

print(f"Saved: {OUTPUT_DIR / 'per_seed_ticker_results.csv'}")
print(f"Saved: {OUTPUT_DIR / 'per_seed_summary.csv'}")
print(f"Saved: {OUTPUT_DIR / 'overall_multiseed_summary.csv'}")
print(f"Saved: {OUTPUT_DIR / 'run_manifest.json'}")
print(f"Saved: {OUTPUT_DIR / 'window_count_check.json'}")


# %% [markdown]
# ## 14. Suspicious Metric Guard

# %%
suspicious_rows = all_results[all_results["suspicious_status"]]
if not suspicious_rows.empty:
    print("LOUD WARNING: suspicious metric guard triggered after artifact save.")
    print(suspicious_rows)
    raise RuntimeError(
        "Suspicious metric guard triggered. Pause and inspect leakage before interpretation."
    )

print("Suspicious metric guard PASS: Candidate A did not cross configured suspicious thresholds.")


# %% [markdown]
# ## 15. Final Summary Printout

# %%
print(SELECTION_BIAS_DISCLOSURE)
print("\nPer-seed summary:")
display(per_seed_summary.round(4))
print("\nOverall multi-seed summary:")
display(overall_multiseed_summary.round(4))
print("\nFinal results are saved artifacts, not committed repo files.")
print("P1B.11b Candidate A multi-seed Colab run complete only if all cells above executed without guard failure.")
