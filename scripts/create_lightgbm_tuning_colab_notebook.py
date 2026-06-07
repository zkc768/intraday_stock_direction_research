"""Generate the Notebook 05 LightGBM tuning Colab notebook.

The generated notebook is standalone for Colab. It copies the active Stage 0
raw-data-first loading, feature, label, split, scaling, window, and base helper
logic from notebooks/02_config_screening_colab.ipynb, then adds Notebook 05
LightGBM-only train-inner HPO, 04D entry-gate checks, finalist selection,
official-validation confirmation, and decision-record cells. It does not
execute the notebook.
"""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_config_screening_colab.ipynb"
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "05_lightgbm_tuning_colab.ipynb"
RUN_ALL_NOTEBOOK = PROJECT_ROOT / "notebooks" / "05_lightgbm_tuning_colab_run_all.ipynb"


TITLE_MD = """\
# Notebook 05 LightGBM Tuning - Validation Only

Protocol: `docs/NOTEBOOK05_LIGHTGBM_TUNING_PROTOCOL_2026-06-04.md`

Scope: `validation_only`

Research question:

```text
Given Notebook 04D's manual Exit A decision and the fixed official candidate,
does train-inner LightGBM hyperparameter tuning produce a stable validation-only
gain without repeatedly searching the official validation partition?
```

Official candidate:

```text
candidate_id  = stage0_official
label_config  = h03_bps1p5
horizon_k     = 3
threshold_bps = 1.5
feature_set   = price_volume_time
window_size   = 20
```

Notebook 05 parts:

```text
05S = schema smoke, no fitting, no selection
05A = read-only Notebook 04D entry gate and operator Exit A acceptance
05B = train-inner chronological LightGBM HPO
05C = finalist selection from train-inner HPO only
05D = official-validation confirmation of default + train-inner finalists
05E = decision record and allowed wording
```

Hard boundaries:

- no project helper package, prior notebook, or archived helper is imported as
  active logic;
- no holdout/test rows are loaded, transformed, windowed, scored, summarized,
  displayed, or used for wording decisions;
- Notebook 04D is not treated as automatic authorization;
- Notebook 05 requires an explicit operator Exit A acceptance flag before any
  fitting stage;
- HPO uses only chronological train-inner folds inside the official train
  partition;
- official validation confirms the train-inner winner and fixed finalists; it
  must not choose a different official-validation-best winner;
- no confidence threshold or selective/no-trade coverage threshold is selected
  here. That remains reserved for a later separately pre-registered notebook.
"""


CONFIG_CODE = r"""
TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
RESULT_SCOPE = "validation_only"

INSTALL_LIGHTGBM_IF_MISSING = False
INSTALL_TORCH_IF_MISSING = False

RUN_05A_TO_05E_FULL_PIPELINE = False
RUN_05S_SCHEMA_SMOKE = False
RUN_05A_04D_ENTRY_GATE = False
RUN_05B_TRAIN_INNER_HPO = False
RUN_05C_SELECT_FINALISTS = False
RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION = False
RUN_05E_DECISION_RECORD = False
BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE = False
NOTEBOOK05_LOCAL_CHECKPOINT_RESUME = True
NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_MINUTES = 30
NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_COMPLETED_UNITS = 25

if RUN_05A_TO_05E_FULL_PIPELINE:
    RUN_05A_04D_ENTRY_GATE = True
    RUN_05B_TRAIN_INNER_HPO = True
    RUN_05C_SELECT_FINALISTS = True
    RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION = True
    RUN_05E_DECISION_RECORD = True

OPERATOR_SELECTED_EXIT = ""
OPERATOR_ACCEPTS_EXIT_A = False
REQUIRED_OPERATOR_EXIT_A = "Exit A - Proceed To 05 LightGBM Tuning"

DRIVE_PROJECT_FOLDER_ID = "15IZ_sOEyyAKmGCUIOv_u17SwQmFX3nG_"
NOTEBOOK04_DRIVE_RESULTS_FOLDER_ID = "1bDhF9glvEwJC_lmp7XRlDGZA4nS-u9Vr"
NOTEBOOK04_DRIVE_RESULTS_FOLDER_NAME = "notebook04_controlled_followup_results"
NOTEBOOK05_DRIVE_BACKUP_FOLDER_NAME = "notebook05_lightgbm_tuning_results"

OFFICIAL_VALIDATION_SEEDS = (606, 707, 808, 909, 1010)

NOTEBOOK05_CANDIDATE = {
    "candidate_id": "stage0_official",
    "label_config": "h03_bps1p5",
    "horizon_k": 3,
    "threshold_bps": 1.5,
    "feature_set": "price_volume_time",
    "window_size": 20,
    "source": "official_stage0_candidate_from_notebook02_and_notebook04d_exit_a",
}

BASELINE_MODELS = ("stratified_dummy", "always_up_dummy")
LIGHTGBM_PROFILES = ("default_lgbm_04", "lightgbm_trial")

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

MAX_TRAIN_ROWS = None
RANDOM_SUBSAMPLE_SEED = 42

LGBM_DEFAULT_PARAMS_04 = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "class_weight": "balanced",
    "verbosity": -1,
}
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
}

HPO_METHOD = "random_search"
HPO_BUDGET = 100
HPO_RNG_SEED = 260605
INNER_DUMMY_SEED = 260605
INNER_FOLD_COUNT = 3
MAX_FIT_ROWS_BEFORE_CONFIRMATION = 300
N_FINALISTS = 5
PRIMARY_SELECTION = "train_inner_winner"
EARLY_STOPPING_ROUNDS = 50
MIN_FINALIST_MEDIAN_BEST_ITERATION = 20
PROMOTION_MIN_DELTA_MACRO_F1_VS_DEFAULT = 0.001
PROMOTION_MIN_POSITIVE_TICKER_COUNT = 4
PROMOTION_MAX_TOP_TICKER_GAIN_SHARE = 0.35
PROMOTION_MAX_MACRO_F1_STD = 0.0025

HPO_MAX_ESTIMATORS_CHOICES = (400, 800, 1200, 1600, 2000)
HPO_MAX_DEPTH_CHOICES = (3, 4, 5, 6, 8)
HPO_NUM_LEAVES_CHOICES = (7, 15, 31, 63)
HPO_MIN_CHILD_SAMPLES_CHOICES = (20, 50, 100, 200, 400)

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

OUTPUT_DIR = Path("/content/notebook05_lightgbm_tuning_results")
PREDICTION_DIR = OUTPUT_DIR / "predictions"
NOTEBOOK04_CONTEXT_DIR = Path("/content/notebook04_controlled_followup_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK04_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = {
    "entry": OUTPUT_DIR / "notebook05_entry_decision.json",
    "hpo_search_manifest": OUTPUT_DIR / "notebook05_hpo_search_manifest.csv",
    "inner_fold_manifest": OUTPUT_DIR / "notebook05_inner_fold_manifest.csv",
    "inner_hpo_results": OUTPUT_DIR / "notebook05_inner_hpo_results.csv",
    "inner_hpo_summary": OUTPUT_DIR / "notebook05_inner_hpo_summary.csv",
    "finalists": OUTPUT_DIR / "notebook05_finalists.csv",
    "official_pooled": OUTPUT_DIR / "notebook05_official_validation_pooled.csv",
    "official_per_ticker": OUTPUT_DIR / "notebook05_official_validation_per_ticker.csv",
    "official_summary": OUTPUT_DIR / "notebook05_official_validation_summary.csv",
    "decision": OUTPUT_DIR / "notebook05_decision_record.json",
    "run_manifest": OUTPUT_DIR / "notebook05_run_manifest.json",
    "backup_manifest": OUTPUT_DIR / "notebook05_drive_backup_manifest.json",
}

NOTEBOOK04_FILES = {
    "context": NOTEBOOK04_CONTEXT_DIR / "notebook04_context_checks.json",
    "summary": NOTEBOOK04_CONTEXT_DIR / "notebook04_summary.csv",
    "selective": NOTEBOOK04_CONTEXT_DIR / "notebook04_selective_coverage.csv",
    "decision": NOTEBOOK04_CONTEXT_DIR / "notebook04_decision_matrix.csv",
    "run_manifest": NOTEBOOK04_CONTEXT_DIR / "notebook04_run_manifest.json",
}

RUN_SWITCHES = {
    "RUN_05S_SCHEMA_SMOKE": RUN_05S_SCHEMA_SMOKE,
    "RUN_05A_TO_05E_FULL_PIPELINE": RUN_05A_TO_05E_FULL_PIPELINE,
    "RUN_05A_04D_ENTRY_GATE": RUN_05A_04D_ENTRY_GATE,
    "RUN_05B_TRAIN_INNER_HPO": RUN_05B_TRAIN_INNER_HPO,
    "RUN_05C_SELECT_FINALISTS": RUN_05C_SELECT_FINALISTS,
    "RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION": RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION,
    "RUN_05E_DECISION_RECORD": RUN_05E_DECISION_RECORD,
    "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE": BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE,
    "NOTEBOOK05_LOCAL_CHECKPOINT_RESUME": NOTEBOOK05_LOCAL_CHECKPOINT_RESUME,
    "NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_MINUTES": NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_MINUTES,
    "NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_COMPLETED_UNITS": NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_COMPLETED_UNITS,
}

print("Notebook 05 scope:", RESULT_SCOPE)
print("Notebook 05 candidate:", NOTEBOOK05_CANDIDATE)
print("Run switches:", RUN_SWITCHES)
"""


NOTEBOOK05_HELPERS_CODE = r"""
import hashlib


NOTEBOOK05_STATE = {
    "entry_decision": None,
    "backup_folder_id": None,
    "last_drive_checkpoint_utc": None,
}

T_CRITICAL_ONE_SIDED_95_LOCAL = {
    1: 0.0,
    2: 6.314,
    3: 2.920,
    4: 2.353,
    5: 2.132,
    6: 2.015,
    7: 1.943,
    8: 1.895,
    9: 1.860,
    10: 1.833,
}


def stable_hash(values):
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def sample_id_hash(values):
    return stable_hash(np.asarray(values).astype(str))


def class_count_fields_05(y_values, prefix):
    y = np.asarray(y_values).astype(int)
    n = int(len(y))
    class0_n = int((y == 0).sum())
    class1_n = int((y == 1).sum())
    positive_rate = float(class1_n / n) if n else np.nan
    return {
        f"{prefix}class0_n": class0_n,
        f"{prefix}class1_n": class1_n,
        f"{prefix}positive_rate": positive_rate,
    }


def series_str_05(values):
    return pd.Series(values).astype(str).reset_index(drop=True)


def series_datetime_str_05(values):
    return pd.Series(pd.to_datetime(values)).astype(str).reset_index(drop=True)


def make_notebook05_sample_ids(owner_values, timestamp_values, y_values, split_name):
    owner = series_str_05(owner_values)
    timestamp = series_datetime_str_05(timestamp_values)
    expected_len = len(y_values)
    if len(owner) != len(timestamp) or len(owner) != expected_len:
        raise ValueError(
            f"Cannot build {split_name}_sample_id; owner/timestamp/y length mismatch: "
            f"owner={len(owner)}, timestamp={len(timestamp)}, y={expected_len}"
        )
    return np.array(
        [
            f"{ticker}|{bar_time}|row{row_index:08d}"
            for row_index, (ticker, bar_time) in enumerate(zip(owner, timestamp))
        ],
        dtype=object,
    )


def t_critical_one_sided_95_local(n):
    n = int(n)
    return T_CRITICAL_ONE_SIDED_95_LOCAL.get(n, 1.645)


def drive_query_literal(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def build_drive_service():
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Drive API is unavailable. Open this notebook in Google Colab and "
            "authenticate when prompted; do not use Drive mounting for Notebook 05 data."
        ) from exc
    auth.authenticate_user()
    return build("drive", "v3")


def find_latest_drive_file_by_suffix(service, folder_id, filename_suffix):
    escaped_parent = drive_query_literal(folder_id)
    query = f"{escaped_parent} in parents and trashed = false"
    files = []
    page_token = None
    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id,name,mimeType,createdTime,modifiedTime)",
            pageSize=100,
            pageToken=page_token,
        ).execute()
        files.extend(
            item
            for item in response.get("files", [])
            if str(item.get("name", "")).endswith(filename_suffix)
        )
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    if not files:
        raise FileNotFoundError(
            f"No Drive file ending with {filename_suffix!r} found in folder "
            f"{NOTEBOOK04_DRIVE_RESULTS_FOLDER_NAME} ({folder_id})."
        )
    return sorted(
        files,
        key=lambda item: (str(item.get("modifiedTime", "")), str(item.get("name", ""))),
        reverse=True,
    )[0]


def download_drive_file(service, file_id, target_path):
    from googleapiclient.http import MediaIoBaseDownload

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)
    with target_path.open("wb") as output:
        downloader = MediaIoBaseDownload(output, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return target_path


def ensure_latest_notebook04_artifacts_from_drive():
    service = build_drive_service()
    downloaded = {}
    for name, target in NOTEBOOK04_FILES.items():
        found = find_latest_drive_file_by_suffix(service, NOTEBOOK04_DRIVE_RESULTS_FOLDER_ID, target.name)
        download_drive_file(service, found["id"], target)
        downloaded[name] = {
            "drive_id": found["id"],
            "drive_name": found["name"],
            "local_path": str(target),
        }
        print(f"Downloaded Notebook 04 {name}: {found['name']} -> {target}")
    return downloaded


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def required_dataframe(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required Notebook 05 input missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required Notebook 05 input is empty: {name} at {path}")
    return frame


def artifact_file_hash(path):
    path = Path(path)
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_fixed_candidate_fields(record, prefix=""):
    expected = NOTEBOOK05_CANDIDATE
    checks = {
        "label_config": expected["label_config"],
        "feature_set": expected["feature_set"],
        "window_size": expected["window_size"],
    }
    for field, expected_value in checks.items():
        value = record.get(field)
        if pd.isna(value):
            raise ValueError(f"{prefix}{field} is missing in Notebook 04 context.")
        if field == "window_size":
            value = int(value)
            expected_value = int(expected_value)
        if value != expected_value:
            raise ValueError(f"{prefix}{field} drifted: {value!r} != {expected_value!r}")
    if "horizon_k" in record and not pd.isna(record["horizon_k"]):
        if int(record["horizon_k"]) != int(expected["horizon_k"]):
            raise ValueError("horizon_k drifted from the official candidate.")
    if "threshold_bps" in record and not pd.isna(record["threshold_bps"]):
        if float(record["threshold_bps"]) != float(expected["threshold_bps"]):
            raise ValueError("threshold_bps drifted from the official candidate.")


def validate_context_official_candidate(context):
    if "official_candidate" not in context:
        raise ValueError("Notebook 04 context is missing official_candidate; Notebook 05 must stop.")
    official_candidate = context["official_candidate"]
    if isinstance(official_candidate, dict):
        validate_fixed_candidate_fields(
            official_candidate,
            prefix="notebook04_context_checks.official_candidate.",
        )
        return
    text = str(official_candidate)
    required_fragments = (
        NOTEBOOK05_CANDIDATE["label_config"],
        NOTEBOOK05_CANDIDATE["feature_set"],
        f"window_size={NOTEBOOK05_CANDIDATE['window_size']}",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(
            "Notebook 04 context official_candidate does not match Notebook 05 fixed candidate; "
            f"missing fragments: {missing}"
        )


def assert_notebook05_entry_gate(download_if_missing=True):
    if download_if_missing and not all(path.exists() for path in NOTEBOOK04_FILES.values()):
        downloaded = ensure_latest_notebook04_artifacts_from_drive()
    else:
        downloaded = {
            name: {"local_path": str(path), "drive_id": None, "drive_name": path.name}
            for name, path in NOTEBOOK04_FILES.items()
        }

    for name, path in NOTEBOOK04_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Required Notebook 04D artifact missing for 05A: {path}")

    context = read_json(NOTEBOOK04_FILES["context"])
    run_manifest = read_json(NOTEBOOK04_FILES["run_manifest"])
    summary = required_dataframe(NOTEBOOK04_FILES["summary"], "notebook04_summary")
    selective = required_dataframe(NOTEBOOK04_FILES["selective"], "notebook04_selective_coverage")
    decision = required_dataframe(NOTEBOOK04_FILES["decision"], "notebook04_decision_matrix")

    if context.get("scope") != RESULT_SCOPE:
        raise ValueError(f"Notebook 04 context scope is not {RESULT_SCOPE}: {context.get('scope')!r}")
    if bool(context.get("holdout_test_authorized")):
        raise ValueError("Notebook 04 context authorizes holdout/test; Notebook 05 must stop.")
    if run_manifest.get("scope") != RESULT_SCOPE:
        raise ValueError(f"Notebook 04 run manifest scope is not {RESULT_SCOPE}: {run_manifest.get('scope')!r}")
    if bool(run_manifest.get("holdout_test_authorized")):
        raise ValueError("Notebook 04 run manifest authorizes holdout/test; Notebook 05 must stop.")
    validate_context_official_candidate(context)

    if OPERATOR_SELECTED_EXIT != REQUIRED_OPERATOR_EXIT_A:
        raise ValueError(
            "Notebook 05 requires OPERATOR_SELECTED_EXIT to be exactly "
            f"{REQUIRED_OPERATOR_EXIT_A!r}."
        )
    if OPERATOR_ACCEPTS_EXIT_A is not True:
        raise ValueError("Notebook 05 requires OPERATOR_ACCEPTS_EXIT_A = True before any fit.")

    lightgbm_rows = summary.loc[summary["model"].astype(str).eq("lightgbm")].copy()
    if len(lightgbm_rows) != 1:
        raise ValueError(f"Expected exactly one Notebook 04 lightgbm summary row, found {len(lightgbm_rows)}.")
    lightgbm_row = lightgbm_rows.iloc[0].to_dict()
    validate_fixed_candidate_fields(lightgbm_row, prefix="notebook04_summary.")
    if not bool(lightgbm_row.get("basic_gate_pass", False)):
        raise ValueError("Notebook 04 LightGBM did not pass the basic gate; Notebook 05 must stop.")
    allowed_stability = {"confirmed_or_improved", "marginal_drop_note_only"}
    if str(lightgbm_row.get("fresh_seed_stability_tag")) not in allowed_stability:
        raise ValueError("Notebook 04 LightGBM fresh-seed stability tag does not authorize Exit A.")

    if "exit" not in decision.columns:
        raise ValueError("Notebook 04 decision matrix is missing the exit column.")
    if not decision["exit"].astype(str).eq(REQUIRED_OPERATOR_EXIT_A).any():
        raise ValueError("Notebook 04 decision matrix does not include Exit A.")
    if "holdout_test_authorized" not in decision.columns:
        raise ValueError("Notebook 04 decision matrix is missing holdout_test_authorized.")
    if decision["holdout_test_authorized"].astype(bool).any():
        raise ValueError("At least one Notebook 04D exit authorizes holdout/test; Notebook 05 must stop.")

    entry = {
        "scope": RESULT_SCOPE,
        "entry_source": "notebook04_04d_decision_gate",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "operator_selected_exit": OPERATOR_SELECTED_EXIT,
        "operator_accepts_exit_a": bool(OPERATOR_ACCEPTS_EXIT_A),
        "holdout_test_authorized": False,
        "hpo_authorized": True,
        "authorized_model_family": "lightgbm",
        "authorized_candidate": NOTEBOOK05_CANDIDATE,
        "candidate": NOTEBOOK05_CANDIDATE,
        "notebook04_lightgbm_macro_f1_mean": float(lightgbm_row.get("macro_f1_mean", np.nan)),
        "notebook04_lightgbm_delta_macro_f1_vs_stratified_dummy_mean": float(
            lightgbm_row.get("delta_macro_f1_vs_stratified_dummy_mean", np.nan)
        ),
        "notebook04_selective_rows_read_for_boundary_only": int(len(selective)),
        "notebook04_artifacts": downloaded,
        "notebook04_artifact_hashes": {
            name: artifact_file_hash(path) for name, path in NOTEBOOK04_FILES.items()
        },
    }
    with OUTPUT_FILES["entry"].open("w", encoding="utf-8") as handle:
        json.dump(entry, handle, indent=2)
    NOTEBOOK05_STATE["entry_decision"] = entry
    print("Notebook 05 entry gate passed. HPO is authorized for LightGBM only.")
    return entry


def validate_dataset_candidate(dataset):
    if dataset["label_config"] != NOTEBOOK05_CANDIDATE["label_config"]:
        raise ValueError("Dataset label_config drifted from Notebook 05 candidate.")
    if int(dataset["horizon_k"]) != int(NOTEBOOK05_CANDIDATE["horizon_k"]):
        raise ValueError("Dataset horizon_k drifted from Notebook 05 candidate.")
    if float(dataset["threshold_bps"]) != float(NOTEBOOK05_CANDIDATE["threshold_bps"]):
        raise ValueError("Dataset threshold_bps drifted from Notebook 05 candidate.")
    if dataset["feature_set"] != NOTEBOOK05_CANDIDATE["feature_set"]:
        raise ValueError("Dataset feature_set drifted from Notebook 05 candidate.")
    if int(dataset["window_size"]) != int(NOTEBOOK05_CANDIDATE["window_size"]):
        raise ValueError("Dataset window_size drifted from Notebook 05 candidate.")


def get_notebook05_dataset():
    dataset = get_dataset(
        NOTEBOOK05_CANDIDATE["label_config"],
        NOTEBOOK05_CANDIDATE["feature_set"],
        NOTEBOOK05_CANDIDATE["window_size"],
    )
    validate_dataset_candidate(dataset)
    return dataset


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def zero_or_log_uniform(rng, low, high):
    if rng.random() < 0.5:
        return 0.0
    return log_uniform(rng, low, high)


def sample_lgbm_hpo_params(trial_number, rng):
    for _ in range(100):
        max_depth = int(rng.choice(HPO_MAX_DEPTH_CHOICES))
        num_leaves = int(rng.choice(HPO_NUM_LEAVES_CHOICES))
        if max_depth > 0 and num_leaves > 2 ** max_depth:
            continue
        params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": log_uniform(rng, 0.005, 0.08),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": int(rng.choice(HPO_MIN_CHILD_SAMPLES_CHOICES)),
            "subsample": float(rng.uniform(0.50, 1.00)),
            "subsample_freq": 1,
            "colsample_bytree": float(rng.uniform(0.50, 1.00)),
            "reg_alpha": zero_or_log_uniform(rng, 1e-4, 10.0),
            "reg_lambda": zero_or_log_uniform(rng, 1e-4, 20.0),
            "min_split_gain": float(rng.uniform(0.0, 0.10)),
            "max_estimators": int(rng.choice(HPO_MAX_ESTIMATORS_CHOICES)),
            "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
        }
        params["trial_id"] = f"lgbm_hpo_{int(trial_number):03d}"
        return params
    raise RuntimeError(f"Could not sample a valid LightGBM HPO trial for {trial_number}.")


def build_hpo_search_manifest():
    rng = np.random.default_rng(HPO_RNG_SEED)
    rows = []
    for trial_number in range(HPO_BUDGET):
        params = sample_lgbm_hpo_params(trial_number, rng)
        rows.append({
            "trial_id": params["trial_id"],
            "trial_number": int(trial_number),
            "scope": "train_inner_hpo_manifest",
            "hpo_method": HPO_METHOD,
            "hpo_budget": int(HPO_BUDGET),
            "hpo_rng_seed": int(HPO_RNG_SEED),
            "inner_dummy_seed": int(INNER_DUMMY_SEED),
            **{k: v for k, v in params.items() if k != "trial_id"},
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_FILES["hpo_search_manifest"], index=False)
    return manifest


def make_train_inner_folds_05(dataset):
    timestamps = pd.to_datetime(dataset["train_timestamp"])
    dates = pd.Series(timestamps).dt.normalize()
    unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
    if len(unique_dates) < INNER_FOLD_COUNT + 1:
        raise ValueError("Not enough train dates for Notebook 05 train-inner folds.")
    chunks = [pd.Series(chunk) for chunk in np.array_split(unique_dates.to_numpy(), INNER_FOLD_COUNT + 1)]
    folds = []
    rows = []
    for fold_index in range(1, INNER_FOLD_COUNT + 1):
        train_dates = pd.concat(chunks[:fold_index], ignore_index=True)
        validation_dates = chunks[fold_index].reset_index(drop=True)
        train_mask = dates.isin(set(train_dates)).to_numpy()
        validation_mask = dates.isin(set(validation_dates)).to_numpy()
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Empty train-inner fold {fold_index}.")
        train_classes = Counter(np.asarray(dataset["y_train"])[train_mask].astype(int))
        validation_classes = Counter(np.asarray(dataset["y_train"])[validation_mask].astype(int))
        if len(validation_classes) < 2:
            raise ValueError(f"Notebook 05 inner fold {fold_index} validation target has one class.")
        if len(train_classes) < 2:
            raise ValueError(f"Notebook 05 inner fold {fold_index} train target has one class.")
        ticker_boundaries = {}
        owners = np.asarray(dataset["train_owner"]).astype(str)
        for ticker in TICKERS:
            ticker_train_dates = dates[train_mask & (owners == ticker)]
            ticker_validation_dates = dates[validation_mask & (owners == ticker)]
            ticker_boundaries[ticker] = {
                "inner_train_start": str(ticker_train_dates.min()) if len(ticker_train_dates) else "",
                "inner_train_end": str(ticker_train_dates.max()) if len(ticker_train_dates) else "",
                "inner_validation_start": str(ticker_validation_dates.min()) if len(ticker_validation_dates) else "",
                "inner_validation_end": str(ticker_validation_dates.max()) if len(ticker_validation_dates) else "",
                "fold_train_n": int((train_mask & (owners == ticker)).sum()),
                "fold_validation_n": int((validation_mask & (owners == ticker)).sum()),
            }
        fold = {
            "inner_fold_id": int(fold_index),
            "train_mask": train_mask,
            "validation_mask": validation_mask,
        }
        folds.append(fold)
        rows.append({
            "inner_fold_id": int(fold_index),
            "scope": "train_inner_validation",
            "inner_train_start": str(train_dates.min()),
            "inner_train_end": str(train_dates.max()),
            "inner_validation_start": str(validation_dates.min()),
            "inner_validation_end": str(validation_dates.max()),
            "fold_train_n": int(train_mask.sum()),
            "fold_validation_n": int(validation_mask.sum()),
            "fold_train_class0_n": int(train_classes.get(0, 0)),
            "fold_train_class1_n": int(train_classes.get(1, 0)),
            "fold_validation_class0_n": int(validation_classes.get(0, 0)),
            "fold_validation_class1_n": int(validation_classes.get(1, 0)),
            "inner_purge_horizon_bars": int(NOTEBOOK05_CANDIDATE["horizon_k"]),
            "ticker_boundaries_json": json.dumps(ticker_boundaries, sort_keys=True),
        })
    fold_manifest = pd.DataFrame(rows)
    fold_manifest.to_csv(OUTPUT_FILES["inner_fold_manifest"], index=False)
    return folds, fold_manifest


def stratified_dummy_predictions_05(y_train, n_rows, seed):
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return dummy.predict(np.zeros((n_rows, 1))).astype(int)


def always_up_predictions_05(n_rows):
    return np.ones(int(n_rows), dtype=int)


def fit_lightgbm_params_05(x_train, y_train, params, seed, x_eval=None, y_eval=None, use_inner_early_stopping=False):
    lgb = ensure_lightgbm()
    n_estimators = int(params.get("n_estimators", params.get("max_estimators", 200)))
    fit_params = {
        "boosting_type": params.get("boosting_type", "gbdt"),
        "objective": params.get("objective", "binary"),
        "n_estimators": n_estimators,
        "learning_rate": float(params["learning_rate"]),
        "max_depth": int(params["max_depth"]),
        "num_leaves": int(params["num_leaves"]),
        "min_child_samples": int(params.get("min_child_samples", 20)),
        "subsample": float(params.get("subsample", 1.0)),
        "subsample_freq": int(params.get("subsample_freq", 1)),
        "colsample_bytree": float(params.get("colsample_bytree", params.get("feature_fraction", 1.0))),
        "reg_alpha": float(params.get("reg_alpha", 0.0)),
        "reg_lambda": float(params.get("reg_lambda", 0.0)),
        "min_split_gain": float(params.get("min_split_gain", 0.0)),
        "class_weight": params.get("class_weight", "balanced"),
        "random_state": int(seed),
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**fit_params)
    callbacks = []
    eval_set = None
    if use_inner_early_stopping:
        if x_eval is None or y_eval is None:
            raise ValueError("Inner early stopping requires an inner validation set.")
        callbacks = [
            lgb.early_stopping(int(params.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS)), verbose=False),
            lgb.log_evaluation(period=0),
        ]
        eval_set = [(x_eval, y_eval)]
    start_fit = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        eval_metric="binary_logloss" if eval_set is not None else None,
        callbacks=callbacks,
    )
    fit_seconds = time.perf_counter() - start_fit
    return model, fit_seconds


def run_one_inner_trial_fold(dataset, trial_params, fold, seed):
    x_train = dataset["x_train"][fold["train_mask"]]
    y_train = dataset["y_train"][fold["train_mask"]]
    x_validation = dataset["x_train"][fold["validation_mask"]]
    y_validation = dataset["y_train"][fold["validation_mask"]]
    dummy_seed = int(INNER_DUMMY_SEED) + int(fold["inner_fold_id"])
    start_predict = None
    try:
        model, fit_seconds = fit_lightgbm_params_05(
            x_train,
            y_train,
            trial_params,
            seed=seed,
            x_eval=x_validation,
            y_eval=y_validation,
            use_inner_early_stopping=True,
        )
        best_iteration = getattr(model, "best_iteration_", None) or int(trial_params["max_estimators"])
        start_predict = time.perf_counter()
        pred = model.predict(x_validation)
        predict_seconds = time.perf_counter() - start_predict
        metrics = evaluate_predictions(y_validation, pred)
        stratified_pred = stratified_dummy_predictions_05(y_train, len(y_validation), dummy_seed)
        always_up_pred = always_up_predictions_05(len(y_validation))
        stratified_metrics = evaluate_predictions(y_validation, stratified_pred)
        always_up_metrics = evaluate_predictions(y_validation, always_up_pred)
        return {
            "run_failed": False,
            "failure_reason": "",
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "lightgbm_seed": int(seed),
            "stratified_dummy_seed": int(dummy_seed),
            "best_iteration": int(best_iteration),
            "fold_train_n": int(len(y_train)),
            "fold_validation_n": int(len(y_validation)),
            "fold_train_class0_n": int((y_train == 0).sum()),
            "fold_train_class1_n": int((y_train == 1).sum()),
            "fold_validation_class0_n": int((y_validation == 0).sum()),
            "fold_validation_class1_n": int((y_validation == 1).sum()),
            "stratified_dummy_macro_f1": stratified_metrics["macro_f1"],
            "stratified_dummy_balanced_accuracy": stratified_metrics["balanced_accuracy"],
            "always_up_dummy_macro_f1": always_up_metrics["macro_f1"],
            "always_up_dummy_balanced_accuracy": always_up_metrics["balanced_accuracy"],
            "delta_macro_f1_vs_stratified_dummy": metrics["macro_f1"] - stratified_metrics["macro_f1"],
            "delta_balanced_accuracy_vs_stratified_dummy": metrics["balanced_accuracy"] - stratified_metrics["balanced_accuracy"],
            **metrics,
        }
    except Exception as exc:
        return {
            "run_failed": True,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "fit_seconds": np.nan,
            "predict_seconds": np.nan,
            "lightgbm_seed": int(seed),
            "stratified_dummy_seed": int(dummy_seed),
            "best_iteration": np.nan,
            "fold_train_n": int(len(y_train)),
            "fold_validation_n": int(len(y_validation)),
            "fold_train_class0_n": int((y_train == 0).sum()),
            "fold_train_class1_n": int((y_train == 1).sum()),
            "fold_validation_class0_n": int((y_validation == 0).sum()),
            "fold_validation_class1_n": int((y_validation == 1).sum()),
            "stratified_dummy_macro_f1": np.nan,
            "stratified_dummy_balanced_accuracy": np.nan,
            "always_up_dummy_macro_f1": np.nan,
            "always_up_dummy_balanced_accuracy": np.nan,
            "delta_macro_f1_vs_stratified_dummy": np.nan,
            "delta_balanced_accuracy_vs_stratified_dummy": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "accuracy": np.nan,
        }


def summarize_inner_hpo_results(results):
    if results.empty:
        return pd.DataFrame()
    rows = []
    param_columns = [
        "learning_rate",
        "max_depth",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
        "max_estimators",
        "early_stopping_rounds",
    ]
    for trial_id, group in results.groupby("trial_id", sort=False):
        successful = group.loc[~group["run_failed"].astype(bool)].copy()
        record = {"trial_id": trial_id, "scope": "train_inner_hpo_summary"}
        for column in param_columns:
            record[column] = group.iloc[0][column]
        record["inner_successful_fold_count"] = int(len(successful))
        record["inner_failed_fold_count"] = int(group["run_failed"].astype(bool).sum())
        if successful.empty:
            for column in (
                "inner_macro_f1_mean",
                "inner_macro_f1_std",
                "inner_macro_f1_min",
                "inner_macro_f1_max",
                "inner_balanced_accuracy_mean",
                "inner_stratified_dummy_macro_f1_mean",
                "inner_delta_macro_f1_vs_stratified_dummy_mean",
                "inner_delta_macro_f1_vs_stratified_dummy_min",
                "inner_lcb_macro_f1",
                "inner_positive_fold_count",
                "median_best_iteration",
            ):
                record[column] = np.nan
            record["eligible_for_finalist"] = False
        else:
            macro_mean = float(successful["macro_f1"].mean())
            macro_std = sample_std(successful["macro_f1"])
            success_count = int(len(successful))
            delta_mean = float(successful["delta_macro_f1_vs_stratified_dummy"].mean())
            delta_min = float(successful["delta_macro_f1_vs_stratified_dummy"].min())
            record.update({
                "inner_macro_f1_mean": macro_mean,
                "inner_macro_f1_std": macro_std,
                "inner_macro_f1_min": float(successful["macro_f1"].min()),
                "inner_macro_f1_max": float(successful["macro_f1"].max()),
                "inner_balanced_accuracy_mean": float(successful["balanced_accuracy"].mean()),
                "inner_stratified_dummy_macro_f1_mean": float(successful["stratified_dummy_macro_f1"].mean()),
                "inner_delta_macro_f1_vs_stratified_dummy_mean": delta_mean,
                "inner_delta_macro_f1_vs_stratified_dummy_min": delta_min,
                "inner_lcb_macro_f1": float(
                    macro_mean
                    - t_critical_one_sided_95_local(success_count) * macro_std / math.sqrt(max(success_count, 1))
                ),
                "inner_positive_fold_count": int((successful["delta_macro_f1_vs_stratified_dummy"] > 0).sum()),
                "median_best_iteration": int(
                    np.clip(
                        round(float(successful["best_iteration"].median())),
                        1,
                        int(group.iloc[0]["max_estimators"]),
                    )
                ),
            })
            record["eligible_for_finalist"] = bool(
                record["inner_successful_fold_count"] == INNER_FOLD_COUNT
                and record["inner_delta_macro_f1_vs_stratified_dummy_mean"] > 0
                and record["inner_delta_macro_f1_vs_stratified_dummy_min"] >= -0.002
                and record["inner_positive_fold_count"] >= 2
                and record["median_best_iteration"] >= MIN_FINALIST_MEDIAN_BEST_ITERATION
            )
        rows.append(record)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_FILES["inner_hpo_summary"], index=False)
    return summary


def run_train_inner_hpo_05b():
    assert_notebook05_entry_gate(download_if_missing=True)
    dataset = get_notebook05_dataset()
    search_manifest = build_hpo_search_manifest()
    folds, fold_manifest = make_train_inner_folds_05(dataset)
    if NOTEBOOK05_LOCAL_CHECKPOINT_RESUME and OUTPUT_FILES["inner_hpo_results"].exists():
        existing_results = pd.read_csv(OUTPUT_FILES["inner_hpo_results"])
        missing = {"trial_id", "inner_fold_id"} - set(existing_results.columns)
        if missing:
            raise ValueError(f"Existing 05B checkpoint is missing columns: {sorted(missing)}")
        rows = existing_results.to_dict("records")
    else:
        rows = []
    completed = {
        (str(row["trial_id"]), int(row["inner_fold_id"]))
        for row in rows
    }
    for _, trial in search_manifest.iterrows():
        trial_params = trial.to_dict()
        seed = int(HPO_RNG_SEED) + int(trial["trial_number"])
        for fold in folds:
            key = (str(trial["trial_id"]), int(fold["inner_fold_id"]))
            if key in completed:
                print("05B checkpoint skip", trial["trial_id"], "fold", fold["inner_fold_id"])
                continue
            print("05B", trial["trial_id"], "fold", fold["inner_fold_id"])
            result = run_one_inner_trial_fold(dataset, trial_params, fold, seed=seed)
            rows.append({
                "trial_id": trial["trial_id"],
                "trial_number": int(trial["trial_number"]),
                "inner_fold_id": int(fold["inner_fold_id"]),
                "scope": "train_inner_validation",
                **{column: trial[column] for column in search_manifest.columns if column not in {"scope"}},
                **result,
            })
            completed.add(key)
            pd.DataFrame(rows).to_csv(OUTPUT_FILES["inner_hpo_results"], index=False)
            maybe_backup_notebook05_checkpoint(
                "checkpoint_05B_train_inner_hpo",
                completed_units=len(completed),
                include_predictions=False,
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FILES["inner_hpo_results"], index=False)
    summary = summarize_inner_hpo_results(results)
    write_run_manifest()
    return search_manifest, fold_manifest, results, summary


def select_finalists_05c():
    if not OUTPUT_FILES["inner_hpo_summary"].exists():
        raise FileNotFoundError("Notebook 05 inner HPO summary is missing. Run 05B first.")
    summary = pd.read_csv(OUTPUT_FILES["inner_hpo_summary"])
    if summary.empty:
        finalists = pd.DataFrame()
        finalists.to_csv(OUTPUT_FILES["finalists"], index=False)
        return finalists
    eligible = summary.loc[summary["eligible_for_finalist"].astype(bool)].copy()
    if eligible.empty:
        finalists = pd.DataFrame()
        finalists.to_csv(OUTPUT_FILES["finalists"], index=False)
        return finalists
    eligible["max_depth_positive_sort"] = eligible["max_depth"].apply(lambda value: int(value) if int(value) > 0 else 999)
    eligible = eligible.sort_values(
        [
            "inner_lcb_macro_f1",
            "inner_macro_f1_std",
            "num_leaves",
            "max_depth_positive_sort",
            "median_best_iteration",
            "trial_id",
        ],
        ascending=[False, True, True, True, True, True],
    ).head(N_FINALISTS).copy()
    eligible.insert(0, "finalist_rank", np.arange(1, len(eligible) + 1))
    eligible["profile_id"] = eligible["trial_id"].map(lambda value: str(value).replace("lgbm_hpo_", "lightgbm_trial_"))
    eligible["profile_role"] = eligible["finalist_rank"].map(lambda rank: "train_inner_winner" if rank == 1 else "train_inner_finalist")
    eligible["selected_profile_source"] = PRIMARY_SELECTION
    eligible["holdout_test_authorized"] = False
    eligible.to_csv(OUTPUT_FILES["finalists"], index=False)
    return eligible


def lgbm_params_from_profile(profile):
    if profile["profile_id"] == "default_lgbm_04":
        params = dict(LGBM_DEFAULT_PARAMS_04)
        params["n_estimators"] = int(params["n_estimators"])
        return params
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "learning_rate": float(profile["learning_rate"]),
        "max_depth": int(profile["max_depth"]),
        "num_leaves": int(profile["num_leaves"]),
        "min_child_samples": int(profile["min_child_samples"]),
        "subsample": float(profile["subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(profile["colsample_bytree"]),
        "reg_alpha": float(profile["reg_alpha"]),
        "reg_lambda": float(profile["reg_lambda"]),
        "min_split_gain": float(profile["min_split_gain"]),
        "class_weight": "balanced",
        "verbosity": -1,
        "n_estimators": int(profile["median_best_iteration"]),
    }
    return params


def profile_rows_for_confirmation():
    if not OUTPUT_FILES["finalists"].exists():
        raise FileNotFoundError("Notebook 05 finalists artifact is missing. Run 05C before 05D.")
    rows = [{
        "profile_id": "default_lgbm_04",
        "profile_role": "default_lgbm_04",
        "selected_by_train_inner": False,
        "trial_id": "",
        "median_best_iteration": int(LGBM_DEFAULT_PARAMS_04["n_estimators"]),
        **LGBM_DEFAULT_PARAMS_04,
    }]
    finalists = pd.read_csv(OUTPUT_FILES["finalists"])
    if not finalists.empty:
        for _, row in finalists.iterrows():
            record = row.to_dict()
            record["selected_by_train_inner"] = bool(record.get("profile_role") == "train_inner_winner")
            rows.append(record)
    return pd.DataFrame(rows)


def save_probability_artifact_05(dataset, profile_id, seed, y_pred, prob_up):
    prob_up = np.asarray(prob_up, dtype=float)
    confidence = np.maximum(prob_up, 1.0 - prob_up)
    payload = {
        "validation_sample_id": np.asarray(dataset["validation_sample_id"]).astype(str),
        "ticker": np.asarray(dataset["validation_owner"]).astype(str),
        "timestamp": dataset["validation_timestamp"].astype("datetime64[ns]").astype(str),
        "y_true": dataset["y_validation"].astype(int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "prob_up": prob_up,
        "confidence": confidence,
    }
    artifact_path = PREDICTION_DIR / f"{profile_id}__seed{int(seed)}.npz"
    np.savez_compressed(artifact_path, **payload)
    return artifact_path


def metric_row_05(y_true, y_pred, stratified_pred, always_up_pred):
    metrics = evaluate_predictions(y_true, y_pred)
    stratified_metrics = evaluate_predictions(y_true, stratified_pred)
    always_up_metrics = evaluate_predictions(y_true, always_up_pred)
    metrics.update({
        "stratified_dummy_macro_f1": stratified_metrics["macro_f1"],
        "stratified_dummy_balanced_accuracy": stratified_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_stratified_dummy": metrics["macro_f1"] - stratified_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_stratified_dummy": metrics["balanced_accuracy"] - stratified_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_stratified_dummy_same_rows": metrics["macro_f1"] - stratified_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_stratified_dummy_same_rows": metrics["balanced_accuracy"] - stratified_metrics["balanced_accuracy"],
        "always_up_dummy_macro_f1": always_up_metrics["macro_f1"],
        "always_up_dummy_balanced_accuracy": always_up_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_always_up_dummy": metrics["macro_f1"] - always_up_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_always_up_dummy": metrics["balanced_accuracy"] - always_up_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_always_up_dummy_same_rows": metrics["macro_f1"] - always_up_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_always_up_dummy_same_rows": metrics["balanced_accuracy"] - always_up_metrics["balanced_accuracy"],
    })
    return metrics


def official_profile_fit_rows(dataset, profile, seed):
    profile = dict(profile)
    profile_id = str(profile["profile_id"])
    profile_role = str(profile["profile_role"])
    params = lgbm_params_from_profile(profile)
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_validation = dataset["x_validation"]
    y_validation = dataset["y_validation"]
    train_owner = np.asarray(dataset["train_owner"]).astype(str)
    pooled_train_class_counts = class_count_fields_05(y_train, "train_")
    model, fit_seconds = fit_lightgbm_params_05(
        x_train,
        y_train,
        params,
        seed=seed,
        use_inner_early_stopping=False,
    )
    start_predict = time.perf_counter()
    y_pred = model.predict(x_validation).astype(int)
    prob_up = model.predict_proba(x_validation)[:, 1].astype(float)
    predict_seconds = time.perf_counter() - start_predict
    artifact_path = save_probability_artifact_05(dataset, profile_id, seed, y_pred, prob_up)
    stratified_pred = stratified_dummy_predictions_05(y_train, len(y_validation), seed)
    always_up_pred = always_up_predictions_05(len(y_validation))
    pooled_metrics = metric_row_05(y_validation, y_pred, stratified_pred, always_up_pred)
    sample_hash = sample_id_hash(dataset["validation_sample_id"])
    per_ticker_rows = []
    for ticker in TICKERS:
        mask = dataset["validation_owner"] == ticker
        if not mask.any():
            continue
        train_mask = train_owner == ticker
        ticker_train_class_counts = class_count_fields_05(y_train[train_mask], "train_")
        per_ticker_metrics = metric_row_05(
            y_validation[mask],
            y_pred[mask],
            stratified_pred[mask],
            always_up_pred[mask],
        )
        per_ticker_rows.append({
            "stage": "05D_official_validation_confirmation",
            "candidate_id": NOTEBOOK05_CANDIDATE["candidate_id"],
            "profile_id": profile_id,
            "profile_role": profile_role,
            "seed": int(seed),
            "label_config": dataset["label_config"],
            "horizon_k": int(dataset["horizon_k"]),
            "threshold_bps": float(dataset["threshold_bps"]),
            "feature_set": dataset["feature_set"],
            "window_size": int(dataset["window_size"]),
            "ticker_or_pooled": ticker,
            "scope": RESULT_SCOPE,
            "n": int(mask.sum()),
            "train_n": int((dataset["train_owner"] == ticker).sum()),
            "validation_n": int(mask.sum()),
            **ticker_train_class_counts,
            "validation_sample_id_hash": sample_hash,
            "sample_id_mismatch_count": 0,
            "selected_by_train_inner": bool(profile.get("selected_by_train_inner", False)),
            "official_validation_used_for_selection": False,
            "selected_profile_source": PRIMARY_SELECTION,
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "prediction_artifact": str(artifact_path),
            **per_ticker_metrics,
        })
    positive = [row["delta_macro_f1_vs_stratified_dummy"] for row in per_ticker_rows if row["delta_macro_f1_vs_stratified_dummy"] > 0]
    positive_ticker_count = int(len(positive))
    top_ticker_gain_share = float(max(positive) / sum(positive)) if positive else 0.0
    for row in per_ticker_rows:
        row["positive_ticker_count"] = positive_ticker_count
        row["top_ticker_gain_share"] = top_ticker_gain_share
    pooled_row = {
        "stage": "05D_official_validation_confirmation",
        "candidate_id": NOTEBOOK05_CANDIDATE["candidate_id"],
        "profile_id": profile_id,
        "profile_role": profile_role,
        "seed": int(seed),
        "label_config": dataset["label_config"],
        "horizon_k": int(dataset["horizon_k"]),
        "threshold_bps": float(dataset["threshold_bps"]),
        "feature_set": dataset["feature_set"],
        "window_size": int(dataset["window_size"]),
        "ticker_or_pooled": "pooled",
        "scope": RESULT_SCOPE,
        "n": int(len(y_validation)),
        "train_n": int(len(y_train)),
        "validation_n": int(len(y_validation)),
        **pooled_train_class_counts,
        "validation_sample_id_hash": sample_hash,
        "sample_id_mismatch_count": 0,
        "selected_by_train_inner": bool(profile.get("selected_by_train_inner", False)),
        "official_validation_used_for_selection": False,
        "selected_profile_source": PRIMARY_SELECTION,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "prediction_artifact": str(artifact_path),
        "positive_ticker_count": positive_ticker_count,
        "top_ticker_gain_share": top_ticker_gain_share,
        **pooled_metrics,
    }
    return pooled_row, per_ticker_rows


def dummy_official_rows_05(dataset, model_name, seed):
    y_train = dataset["y_train"]
    y_validation = dataset["y_validation"]
    train_owner = np.asarray(dataset["train_owner"]).astype(str)
    pooled_train_class_counts = class_count_fields_05(y_train, "train_")
    if model_name == "stratified_dummy":
        y_pred = stratified_dummy_predictions_05(y_train, len(y_validation), seed)
    elif model_name == "always_up_dummy":
        y_pred = always_up_predictions_05(len(y_validation))
    else:
        raise ValueError(f"Unsupported dummy model: {model_name}")
    stratified_pred = stratified_dummy_predictions_05(y_train, len(y_validation), seed)
    always_up_pred = always_up_predictions_05(len(y_validation))
    metrics = metric_row_05(y_validation, y_pred, stratified_pred, always_up_pred)
    sample_hash = sample_id_hash(dataset["validation_sample_id"])
    pooled_row = {
        "stage": "05D_official_validation_confirmation",
        "candidate_id": NOTEBOOK05_CANDIDATE["candidate_id"],
        "profile_id": model_name,
        "profile_role": model_name,
        "seed": int(seed),
        "label_config": dataset["label_config"],
        "horizon_k": int(dataset["horizon_k"]),
        "threshold_bps": float(dataset["threshold_bps"]),
        "feature_set": dataset["feature_set"],
        "window_size": int(dataset["window_size"]),
        "ticker_or_pooled": "pooled",
        "scope": RESULT_SCOPE,
        "n": int(len(y_validation)),
        "train_n": int(len(y_train)),
        "validation_n": int(len(y_validation)),
        **pooled_train_class_counts,
        "validation_sample_id_hash": sample_hash,
        "sample_id_mismatch_count": 0,
        "selected_by_train_inner": False,
        "official_validation_used_for_selection": False,
        "selected_profile_source": "dummy_baseline",
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "prediction_artifact": "",
        "positive_ticker_count": np.nan,
        "top_ticker_gain_share": np.nan,
        **metrics,
    }
    per_ticker_rows = []
    for ticker in TICKERS:
        mask = dataset["validation_owner"] == ticker
        if not mask.any():
            continue
        train_mask = train_owner == ticker
        ticker_train_class_counts = class_count_fields_05(y_train[train_mask], "train_")
        per_ticker_metrics = metric_row_05(
            y_validation[mask],
            y_pred[mask],
            stratified_pred[mask],
            always_up_pred[mask],
        )
        per_ticker_rows.append({
            **pooled_row,
            "ticker_or_pooled": ticker,
            "n": int(mask.sum()),
            "train_n": int((dataset["train_owner"] == ticker).sum()),
            "validation_n": int(mask.sum()),
            **ticker_train_class_counts,
            **per_ticker_metrics,
        })
    return pooled_row, per_ticker_rows


def summarize_official_validation_05d(pooled):
    if pooled.empty:
        return pd.DataFrame()
    rows = []
    for profile_id, group in pooled.groupby("profile_id", sort=False):
        successful = group.copy()
        record = {
            "profile_id": profile_id,
            "profile_role": str(group.iloc[0]["profile_role"]),
            "scope": RESULT_SCOPE,
            "seed_count": int(successful["seed"].nunique()),
            "selected_by_train_inner": bool(successful["selected_by_train_inner"].astype(bool).any()),
            "selected_profile_source": str(group.iloc[0]["selected_profile_source"]),
            "validation_sample_id_hash": str(group.iloc[0]["validation_sample_id_hash"]),
            "sample_id_mismatch_count": int(successful["sample_id_mismatch_count"].sum()),
            "macro_f1_mean": float(successful["macro_f1"].mean()),
            "macro_f1_std": sample_std(successful["macro_f1"]),
            "balanced_accuracy_mean": float(successful["balanced_accuracy"].mean()),
            "stratified_dummy_macro_f1_mean": float(successful["stratified_dummy_macro_f1"].mean()),
            "delta_macro_f1_vs_stratified_dummy_mean": float(successful["delta_macro_f1_vs_stratified_dummy"].mean()),
            "always_up_dummy_macro_f1_mean": float(successful["always_up_dummy_macro_f1"].mean()),
            "delta_macro_f1_vs_always_up_dummy_mean": float(successful["delta_macro_f1_vs_always_up_dummy"].mean()),
            "positive_ticker_count": np.nan,
            "top_ticker_gain_share": np.nan,
        }
        record["macro_f1_lcb_95"] = float(
            record["macro_f1_mean"]
            - t_critical_one_sided_95_local(record["seed_count"])
            * record["macro_f1_std"]
            / math.sqrt(max(record["seed_count"], 1))
        )
        concentration = successful["positive_ticker_count"].dropna()
        if not concentration.empty:
            record["positive_ticker_count"] = int(round(float(concentration.mean())))
        gain_share = successful["top_ticker_gain_share"].dropna()
        if not gain_share.empty:
            record["top_ticker_gain_share"] = float(gain_share.mean())
        rows.append(record)
    summary = pd.DataFrame(rows)
    default_rows = summary.loc[summary["profile_id"].eq("default_lgbm_04")]
    if not default_rows.empty:
        default_macro = float(default_rows.iloc[0]["macro_f1_mean"])
        summary["delta_macro_f1_vs_default_lgbm_04"] = summary["macro_f1_mean"] - default_macro
    else:
        summary["delta_macro_f1_vs_default_lgbm_04"] = np.nan
    summary = summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    summary["official_validation_rank_by_macro_f1"] = np.arange(1, len(summary) + 1)
    summary["official_validation_diagnostic_rank_by_macro_f1"] = np.arange(1, len(summary) + 1)
    official_best = summary.iloc[0]["profile_id"] if not summary.empty else ""
    train_inner_rows = summary.loc[summary["selected_by_train_inner"].astype(bool)]
    train_inner_winner = train_inner_rows.iloc[0]["profile_id"] if not train_inner_rows.empty else ""
    summary["selected_by_official_validation"] = summary["profile_id"].eq(official_best)
    summary["official_validation_diagnostic_best"] = summary["profile_id"].eq(official_best)
    summary["official_validation_ranking_disagrees_with_train_inner"] = bool(
        train_inner_winner and official_best != train_inner_winner
    )
    summary.to_csv(OUTPUT_FILES["official_summary"], index=False)
    return summary


def run_official_validation_confirmation_05d():
    assert_notebook05_entry_gate(download_if_missing=True)
    dataset = get_notebook05_dataset()
    profiles = profile_rows_for_confirmation()
    if NOTEBOOK05_LOCAL_CHECKPOINT_RESUME and OUTPUT_FILES["official_pooled"].exists():
        existing_pooled = pd.read_csv(OUTPUT_FILES["official_pooled"])
        missing = {"profile_id", "seed"} - set(existing_pooled.columns)
        if missing:
            raise ValueError(f"Existing 05D pooled checkpoint is missing columns: {sorted(missing)}")
        pooled_rows = existing_pooled.to_dict("records")
    else:
        pooled_rows = []
    if NOTEBOOK05_LOCAL_CHECKPOINT_RESUME and OUTPUT_FILES["official_per_ticker"].exists():
        existing_per_ticker = pd.read_csv(OUTPUT_FILES["official_per_ticker"])
        missing = {"profile_id", "seed", "ticker_or_pooled"} - set(existing_per_ticker.columns)
        if missing:
            raise ValueError(f"Existing 05D per-ticker checkpoint is missing columns: {sorted(missing)}")
        per_ticker_rows = existing_per_ticker.to_dict("records")
    else:
        per_ticker_rows = []
    completed = {
        (str(row["profile_id"]), int(row["seed"]))
        for row in pooled_rows
    }

    def write_05d_checkpoint():
        pooled_checkpoint = pd.DataFrame(pooled_rows)
        per_ticker_checkpoint = pd.DataFrame(per_ticker_rows)
        pooled_checkpoint.to_csv(OUTPUT_FILES["official_pooled"], index=False)
        per_ticker_checkpoint.to_csv(OUTPUT_FILES["official_per_ticker"], index=False)
        summarize_official_validation_05d(pooled_checkpoint)
        write_run_manifest()

    for seed in OFFICIAL_VALIDATION_SEEDS:
        for model_name in BASELINE_MODELS:
            key = (str(model_name), int(seed))
            if key in completed:
                print("05D checkpoint skip", model_name, "seed", seed)
                continue
            print("05D", model_name, "seed", seed)
            pooled_row, ticker_rows = dummy_official_rows_05(dataset, model_name, seed)
            pooled_rows.append(pooled_row)
            per_ticker_rows.extend(ticker_rows)
            completed.add(key)
            write_05d_checkpoint()
            maybe_backup_notebook05_checkpoint(
                "checkpoint_05D_official_validation_confirmation",
                completed_units=len(completed),
                include_predictions=True,
            )
        for _, profile in profiles.iterrows():
            key = (str(profile["profile_id"]), int(seed))
            if key in completed:
                print("05D checkpoint skip", profile["profile_id"], "seed", seed)
                continue
            print("05D", profile["profile_id"], "seed", seed)
            pooled_row, ticker_rows = official_profile_fit_rows(dataset, profile.to_dict(), seed)
            pooled_rows.append(pooled_row)
            per_ticker_rows.extend(ticker_rows)
            completed.add(key)
            write_05d_checkpoint()
            maybe_backup_notebook05_checkpoint(
                "checkpoint_05D_official_validation_confirmation",
                completed_units=len(completed),
                include_predictions=True,
            )
    pooled = pd.DataFrame(pooled_rows)
    per_ticker = pd.DataFrame(per_ticker_rows)
    summary = summarize_official_validation_05d(pooled)
    pooled.to_csv(OUTPUT_FILES["official_pooled"], index=False)
    per_ticker.to_csv(OUTPUT_FILES["official_per_ticker"], index=False)
    write_run_manifest()
    return pooled, per_ticker, summary


def official_validation_status_05e(official_summary, selected_profile_id):
    if official_summary.empty:
        raise ValueError("Notebook 05 official validation summary is empty. Run 05D before 05E.")
    selected_rows = official_summary.loc[official_summary["profile_id"].astype(str).eq(str(selected_profile_id))]
    if selected_rows.empty:
        raise ValueError(f"Selected train-inner profile is missing from official validation summary: {selected_profile_id}")
    default_rows = official_summary.loc[official_summary["profile_id"].astype(str).eq("default_lgbm_04")]
    if default_rows.empty:
        raise ValueError("default_lgbm_04 is missing from official validation summary.")
    selected = selected_rows.iloc[0]
    default = default_rows.iloc[0]
    selected_delta_vs_default = float(selected["delta_macro_f1_vs_default_lgbm_04"])
    selected_macro_std = float(selected["macro_f1_std"])
    default_macro_std = float(default["macro_f1_std"])
    status_checks = {
        "all_official_validation_seeds_present": int(selected["seed_count"]) == len(OFFICIAL_VALIDATION_SEEDS),
        "beats_stratified_dummy_mean": float(selected["delta_macro_f1_vs_stratified_dummy_mean"]) > 0,
        "lcb_beats_stratified_dummy_mean": float(selected["macro_f1_lcb_95"]) > float(selected["stratified_dummy_macro_f1_mean"]),
        "beats_default_by_practical_margin": selected_delta_vs_default >= PROMOTION_MIN_DELTA_MACRO_F1_VS_DEFAULT,
        "enough_positive_tickers": int(selected["positive_ticker_count"]) >= PROMOTION_MIN_POSITIVE_TICKER_COUNT,
        "not_ticker_concentrated": float(selected["top_ticker_gain_share"]) <= PROMOTION_MAX_TOP_TICKER_GAIN_SHARE,
        "seed_std_within_limit": selected_macro_std <= max(PROMOTION_MAX_MACRO_F1_STD, 3.0 * default_macro_std),
    }
    if all(status_checks.values()):
        return "promote_train_inner_winner", status_checks
    if selected_delta_vs_default < PROMOTION_MIN_DELTA_MACRO_F1_VS_DEFAULT:
        return "no_practical_tuning_gain", status_checks
    official_best_rows = official_summary.loc[official_summary["selected_by_official_validation"].astype(bool)]
    official_best = str(official_best_rows.iloc[0]["profile_id"]) if not official_best_rows.empty else ""
    if official_best == "default_lgbm_04":
        return "retain_default_lgbm_04", status_checks
    return "validation_rejects_train_inner_winner", status_checks


def write_decision_record_05e():
    entry = NOTEBOOK05_STATE.get("entry_decision")
    if entry is None and OUTPUT_FILES["entry"].exists():
        entry = read_json(OUTPUT_FILES["entry"])
    finalists = pd.read_csv(OUTPUT_FILES["finalists"]) if OUTPUT_FILES["finalists"].exists() else pd.DataFrame()
    if not OUTPUT_FILES["official_summary"].exists():
        raise FileNotFoundError("Notebook 05 official validation summary is missing. Run 05D before 05E.")
    official_summary = pd.read_csv(OUTPUT_FILES["official_summary"])
    n_finalists_found = int(len(finalists)) if not finalists.empty else 0
    finalist_count_below_target = bool(n_finalists_found < int(N_FINALISTS))
    promotion_checks = {}
    if finalists.empty:
        decision = "no_train_inner_hpo_candidate"
        selected_profile_id = "default_lgbm_04"
        selected_profile_source = "default_context_only"
        official_validation_status = "no_train_inner_hpo_candidate"
    else:
        winner = finalists.sort_values("finalist_rank").iloc[0]
        selected_profile_id = str(winner["profile_id"])
        selected_profile_source = PRIMARY_SELECTION
        official_validation_status, promotion_checks = official_validation_status_05e(
            official_summary,
            selected_profile_id,
        )
        decision = official_validation_status
    official_best_profile_id = ""
    official_disagrees = False
    if not official_summary.empty:
        best = official_summary.sort_values("official_validation_rank_by_macro_f1").iloc[0]
        official_best_profile_id = str(best["profile_id"])
        official_disagrees = bool(official_best_profile_id != selected_profile_id)
    downstream_primary_profile_id = (
        selected_profile_id
        if official_validation_status == "promote_train_inner_winner"
        else "default_lgbm_04"
    )
    record = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "decision": decision,
        "selected_profile_id": selected_profile_id,
        "selected_profile_source": selected_profile_source,
        "train_inner_selected_profile_id": selected_profile_id,
        "official_validation_status": official_validation_status,
        "promotion_checks": promotion_checks,
        "downstream_primary_profile_id": downstream_primary_profile_id,
        "retained_default_lgbm_04": bool(downstream_primary_profile_id == "default_lgbm_04"),
        "n_finalists_found": n_finalists_found,
        "n_finalists_target": int(N_FINALISTS),
        "finalist_count_below_target": finalist_count_below_target,
        "official_validation_best_profile_id": official_best_profile_id,
        "official_validation_ranking_disagrees_with_train_inner": official_disagrees,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "allowed_wording": [
            "Notebook 05 performs validation-only LightGBM hyperparameter tuning under a train-inner chronological HPO design.",
            "The selected tuned configuration is the train-inner HPO winner, not the official-validation-best finalist.",
            "Official validation assigns a pre-registered promotion, retention, or rejection status without selecting a new winner.",
            "Notebook 05 does not authorize holdout/test evaluation.",
        ],
        "forbidden_wording": [
            "The tuned model is final.",
            "The tuned model is holdout-ready.",
            "The tuned model significantly beats LogReg.",
            "The tuned model proves LightGBM is superior to deep learning.",
            "The official-validation-best finalist is selected.",
            "Selective coverage is now the final trading threshold.",
        ],
        "candidate": NOTEBOOK05_CANDIDATE,
        "entry_decision": entry,
        "budget_tracker": budget_tracker_05(),
    }
    with OUTPUT_FILES["decision"].open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    write_run_manifest()
    return record


def budget_tracker_05():
    inner_results = pd.read_csv(OUTPUT_FILES["inner_hpo_results"]) if OUTPUT_FILES["inner_hpo_results"].exists() else pd.DataFrame()
    finalists = pd.read_csv(OUTPUT_FILES["finalists"]) if OUTPUT_FILES["finalists"].exists() else pd.DataFrame()
    official_pooled = pd.read_csv(OUTPUT_FILES["official_pooled"]) if OUTPUT_FILES["official_pooled"].exists() else pd.DataFrame()
    return {
        "hpo_method": HPO_METHOD,
        "hpo_budget": int(HPO_BUDGET),
        "inner_fold_count": int(INNER_FOLD_COUNT),
        "train_inner_lightgbm_fits_planned": int(HPO_BUDGET * INNER_FOLD_COUNT),
        "train_inner_lightgbm_fits_completed": int(len(inner_results)) if not inner_results.empty else 0,
        "train_inner_hpo_complete": bool(len(inner_results) == HPO_BUDGET * INNER_FOLD_COUNT),
        "n_finalists_available": int(len(finalists)) if not finalists.empty else 0,
        "official_validation_lightgbm_rows_planned": int((1 + N_FINALISTS) * len(OFFICIAL_VALIDATION_SEEDS)),
        "official_validation_lightgbm_rows_completed": int(
            len(official_pooled.loc[~official_pooled["profile_id"].isin(BASELINE_MODELS)])
        ) if not official_pooled.empty else 0,
        "official_validation_dummy_rows_completed": int(
            len(official_pooled.loc[official_pooled["profile_id"].isin(BASELINE_MODELS)])
        ) if not official_pooled.empty else 0,
        "holdout_test_authorized": False,
    }


def write_run_manifest():
    manifest = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "candidate": NOTEBOOK05_CANDIDATE,
        "run_switches": RUN_SWITCHES,
        "operator_selected_exit": OPERATOR_SELECTED_EXIT,
        "operator_accepts_exit_a": bool(OPERATOR_ACCEPTS_EXIT_A),
        "primary_selection": PRIMARY_SELECTION,
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "budget_tracker": budget_tracker_05(),
        "output_files": {name: str(path) for name, path in OUTPUT_FILES.items()},
    }
    with OUTPUT_FILES["run_manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def find_or_create_drive_folder(service, folder_name, parent_id):
    escaped_parent = drive_query_literal(parent_id)
    escaped_name = drive_query_literal(folder_name)
    query = f"name = {escaped_name} and mimeType = 'application/vnd.google-apps.folder' and {escaped_parent} in parents and trashed = false"
    response = service.files().list(q=query, spaces="drive", fields="files(id,name)").execute()
    folders = response.get("files", [])
    if folders:
        return folders[0]
    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    return service.files().create(body=metadata, fields="id,name").execute()


def upload_local_file_to_drive(service, local_path, folder_id, uploaded_name):
    from googleapiclient.http import MediaFileUpload

    media = MediaFileUpload(str(local_path), resumable=True)
    metadata = {"name": uploaded_name, "parents": [folder_id]}
    return service.files().create(body=metadata, media_body=media, fields="id,name").execute()


def backup_notebook05_outputs(reason, include_predictions=False):
    if not BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE:
        print("Notebook 05 Drive backup skipped; BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE is False.")
        return None
    service = build_drive_service()
    backup_folder = find_or_create_drive_folder(service, NOTEBOOK05_DRIVE_BACKUP_FOLDER_NAME, DRIVE_PROJECT_FOLDER_ID)
    NOTEBOOK05_STATE["backup_folder_id"] = backup_folder["id"]
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uploaded = []
    for path in OUTPUT_FILES.values():
        if not Path(path).exists() or Path(path).name == OUTPUT_FILES["backup_manifest"].name:
            continue
        uploaded_name = f"{timestamp}__{reason}__{Path(path).name}"
        drive_file = upload_local_file_to_drive(service, path, backup_folder["id"], uploaded_name)
        uploaded.append({"local_path": str(path), "drive_id": drive_file["id"], "drive_name": drive_file["name"]})
    if include_predictions:
        for path in sorted(PREDICTION_DIR.glob("*.npz")):
            uploaded_name = f"{timestamp}__{reason}__predictions__{path.name}"
            drive_file = upload_local_file_to_drive(service, path, backup_folder["id"], uploaded_name)
            uploaded.append({"local_path": str(path), "drive_id": drive_file["id"], "drive_name": drive_file["name"]})
    manifest = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "reason": reason,
        "local_output_dir": str(OUTPUT_DIR),
        "backup_folder": backup_folder,
        "uploaded_files": uploaded,
        "run_switches": RUN_SWITCHES,
        "holdout_test_authorized": False,
    }
    with OUTPUT_FILES["backup_manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    drive_file = upload_local_file_to_drive(
        service,
        OUTPUT_FILES["backup_manifest"],
        backup_folder["id"],
        f"{timestamp}__{reason}__{OUTPUT_FILES['backup_manifest'].name}",
    )
    manifest["backup_manifest_drive_file"] = drive_file
    print("Notebook 05 Drive backup complete:", reason)
    return manifest


def maybe_backup_notebook05_checkpoint(reason, completed_units, include_predictions=False, force=False):
    if not BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE:
        return None
    completed_units = int(completed_units)
    unit_interval = int(NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_COMPLETED_UNITS)
    minute_interval = float(NOTEBOOK05_DRIVE_CHECKPOINT_BACKUP_EVERY_MINUTES)
    now = pd.Timestamp.utcnow()
    last = NOTEBOOK05_STATE.get("last_drive_checkpoint_utc")
    due_to_units = unit_interval > 0 and completed_units > 0 and completed_units % unit_interval == 0
    due_to_time = last is None or (now - last).total_seconds() >= minute_interval * 60.0
    if not (force or due_to_units or due_to_time):
        return None
    NOTEBOOK05_STATE["last_drive_checkpoint_utc"] = now
    return backup_notebook05_outputs(reason, include_predictions=include_predictions)


def run_notebook05_schema_smoke():
    if HPO_BUDGET != 100 or INNER_FOLD_COUNT != 3:
        raise ValueError("Notebook 05 HPO budget or fold count drifted from protocol.")
    rng = np.random.default_rng(HPO_RNG_SEED)
    trial = sample_lgbm_hpo_params(0, rng)
    if trial["trial_id"] != "lgbm_hpo_000":
        raise ValueError("HPO trial id generation failed.")
    if trial["max_depth"] > 0 and trial["num_leaves"] > 2 ** trial["max_depth"]:
        raise ValueError("Invalid LightGBM HPO constraint passed sampling.")
    dummy_hash = sample_id_hash(["sample_a", "sample_b"])
    if len(dummy_hash) != 64:
        raise ValueError("sample_id_hash did not produce a sha256 digest.")
    print("Notebook 05 schema smoke passed. No model fit was performed.")
"""


STAGE05S_MD = """\
## 05S - Schema Smoke

05S checks local schema, search-space sampling, sample-id hashing, and fixed
budget constants. It does not fit any model and does not read holdout/test.
"""


STAGE05S_CODE = r"""
if RUN_05S_SCHEMA_SMOKE:
    run_notebook05_schema_smoke()
else:
    print("RUN_05S_SCHEMA_SMOKE is False; schema smoke not run.")
"""


STAGE05A_MD = """\
## 05A - Notebook 04D Entry Gate

05A reads Notebook 04D artifacts and requires an explicit operator Exit A
acceptance. Passing 05A authorizes LightGBM HPO only; it does not authorize
holdout/test.
"""


STAGE05A_CODE = r"""
if RUN_05A_04D_ENTRY_GATE:
    entry_decision = assert_notebook05_entry_gate(download_if_missing=True)
    display(pd.DataFrame([entry_decision]))
    backup_notebook05_outputs("completed_05A_04D_entry_gate")
else:
    print("RUN_05A_04D_ENTRY_GATE is False; 04D entry gate not run.")
"""


STAGE05B_MD = """\
## 05B - Train-Inner Chronological LightGBM HPO

05B runs exactly 100 random-search LightGBM trials across 3 chronological
train-inner folds. It does not use official validation rows for HPO.
"""


STAGE05B_CODE = r"""
if RUN_05B_TRAIN_INNER_HPO:
    hpo_search_manifest, inner_fold_manifest, inner_hpo_results, inner_hpo_summary = run_train_inner_hpo_05b()
    display(inner_hpo_summary.sort_values("inner_lcb_macro_f1", ascending=False).head(10))
    backup_notebook05_outputs("completed_05B_train_inner_hpo")
else:
    print("RUN_05B_TRAIN_INNER_HPO is False; train-inner HPO not run.")
"""


STAGE05C_MD = """\
## 05C - Finalist Selection

05C selects finalists only from train-inner HPO results. The rank-1 finalist is
the `train_inner_winner`. Official validation cannot replace it.
"""


STAGE05C_CODE = r"""
if RUN_05C_SELECT_FINALISTS:
    finalists = select_finalists_05c()
    display(finalists)
    backup_notebook05_outputs("completed_05C_select_finalists")
else:
    print("RUN_05C_SELECT_FINALISTS is False; finalists not selected.")
"""


STAGE05D_MD = """\
## 05D - Official Validation Confirmation

05D evaluates default LightGBM, the train-inner winner, the remaining finalists
up to `N_FINALISTS`, and same-row dummy baselines on official validation. It
does not use official validation for early stopping or profile selection.
"""


STAGE05D_CODE = r"""
if RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION:
    official_pooled, official_per_ticker, official_summary = run_official_validation_confirmation_05d()
    display(official_summary)
    backup_notebook05_outputs("completed_05D_official_validation_confirmation", include_predictions=True)
else:
    print("RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION is False; official validation confirmation not run.")
"""


STAGE05E_MD = """\
## 05E - Decision Record

05E writes the validation-only decision record and allowed wording. It does not
authorize holdout/test and does not choose any selective confidence threshold.
"""


STAGE05E_CODE = r"""
if RUN_05E_DECISION_RECORD:
    decision_record = write_decision_record_05e()
    display(pd.DataFrame([decision_record]))
    backup_notebook05_outputs("completed_05E_decision_record")
else:
    print("RUN_05E_DECISION_RECORD is False; decision record not written.")
"""


INTERPRETATION_MD = """\
## Interpretation Boundary

Notebook 05 is `validation_only`.

Allowed wording:

```text
Notebook 05 performs validation-only LightGBM hyperparameter tuning under a
train-inner chronological HPO design. The official validation partition is used
only to confirm the pre-selected train-inner winner and a fixed number of
finalists.
```

```text
The selected tuned configuration is the train-inner HPO winner, not the
official-validation-best finalist.
```

```text
Notebook 05 does not authorize holdout/test evaluation.
```

Forbidden wording:

```text
The tuned model is final.
The tuned model is holdout-ready.
The tuned model significantly beats LogReg.
The tuned model proves LightGBM is superior to deep learning.
The official-validation-best finalist is selected.
Selective coverage is now the final trading threshold.
```

Notebook 06 is reserved for separately pre-registered prediction-time
abstention / high-confidence coverage calibration. Notebook 05 does not select
coverage thresholds.
"""


def dedent_code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def assignment_value(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment for {name}")


def validate_code_cells(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell_{index}")


def validate_notebook(nb: nbformat.NotebookNode) -> None:
    nbformat.validate(nb)
    validate_code_cells(nb)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    if sum(len(getattr(cell, "outputs", [])) for cell in code_cells) != 0:
        raise AssertionError("Generated notebook contains saved outputs.")
    if [cell.get("execution_count") for cell in code_cells] != [None] * len(code_cells):
        raise AssertionError("Generated notebook contains execution counts.")
    config_source = code_cells[1].source
    for name in (
        "INSTALL_LIGHTGBM_IF_MISSING",
        "INSTALL_TORCH_IF_MISSING",
        "RUN_05A_TO_05E_FULL_PIPELINE",
        "RUN_05S_SCHEMA_SMOKE",
        "RUN_05A_04D_ENTRY_GATE",
        "RUN_05B_TRAIN_INNER_HPO",
        "RUN_05C_SELECT_FINALISTS",
        "RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION",
        "RUN_05E_DECISION_RECORD",
        "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE",
        "OPERATOR_ACCEPTS_EXIT_A",
    ):
        if assignment_value(config_source, name) is not False:
            raise AssertionError(f"{name} must default to False.")
    source = "\n".join(cell.source for cell in code_cells)
    forbidden = ("from intraday_research", "baseline_helpers", "train_test_split", "drive.mount", "runpy")
    present = [text for text in forbidden if text in source]
    if present:
        raise AssertionError(f"Forbidden active-code strings found: {present}")
    compact_source = source.lower().replace(" ", "")
    if (
        "holdout_test_authorized=true" in compact_source
        or '"holdout_test_authorized":true' in compact_source
        or "'holdout_test_authorized':true" in compact_source
    ):
        raise AssertionError("Generated notebook contains a holdout/test authorization path.")
    if "selected_profile_source = \"official_validation_best\"" in source:
        raise AssertionError("Generated notebook can select an official-validation-best finalist.")
    required = (
        "validation_sample_id",
        "make_notebook05_sample_ids",
        "selected_by_official_validation",
        "official_validation_status",
        "PROMOTION_MIN_DELTA_MACRO_F1_VS_DEFAULT",
    )
    missing = [text for text in required if text not in source]
    if missing:
        raise AssertionError(f"Generated notebook is missing Notebook 05 repair fields: {missing}")


def build_notebook() -> nbformat.NotebookNode:
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    setup_code = source.cells[1].source.replace(
        "INSTALL_LIGHTGBM_IF_MISSING = True",
        "INSTALL_LIGHTGBM_IF_MISSING = False",
    ).replace(
        "drive.mount",
        "Drive mounting",
    )
    data_loading_code = source.cells[4].source.replace(
        "RUN_ANY_STAGE = bool(RUN_STAGE0S or RUN_STAGE0A1 or RUN_STAGE0A2 or RUN_STAGE0B)",
        "RUN_ANY_STAGE = bool(RUN_05B_TRAIN_INNER_HPO or RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION)",
    ).replace(
        'print("All RUN_STAGE0* switches are False; data loading skipped.")',
        'print("All Notebook 05 fitting switches are False; data loading skipped.")',
    ).replace(
        "drive.mount",
        "Drive mounting",
    )
    base_helpers_code = source.cells[8].source.replace(
        "Enable a RUN_STAGE0* switch and rerun data loading first.",
        "Enable RUN_05B_TRAIN_INNER_HPO or RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION and rerun data loading first.",
    ).replace(
        "drive.mount",
        "Drive mounting",
    ).replace(
        '        "train_owner": train_owner,\n        "x_validation": x_validation,',
        '        "train_owner": train_owner,\n        "train_timestamp": train_timestamp,\n        "train_sample_id": make_notebook05_sample_ids(train_owner, train_timestamp, y_train, "train"),\n        "x_validation": x_validation,',
    ).replace(
        '        "validation_owner": validation_owner,\n        "x_train_seq": x_train_seq,',
        '        "validation_owner": validation_owner,\n        "validation_timestamp": validation_timestamp,\n        "validation_sample_id": make_notebook05_sample_ids(validation_owner, validation_timestamp, y_validation, "validation"),\n        "x_train_seq": x_train_seq,',
    ).replace(
        '        "train_owner_seq": train_owner_seq,\n        "x_validation_seq": x_validation_seq,',
        '        "train_owner_seq": train_owner_seq,\n        "train_timestamp_seq": train_timestamp_seq,\n        "train_sample_id_seq": make_notebook05_sample_ids(train_owner_seq, train_timestamp_seq, y_train_seq, "train_seq"),\n        "x_validation_seq": x_validation_seq,',
    ).replace(
        '        "validation_owner_seq": validation_owner_seq,\n        "prep_seconds": time.perf_counter() - start,',
        '        "validation_owner_seq": validation_owner_seq,\n        "validation_timestamp_seq": validation_timestamp_seq,\n        "validation_sample_id_seq": make_notebook05_sample_ids(validation_owner_seq, validation_timestamp_seq, y_validation_seq, "validation_seq"),\n        "prep_seconds": time.perf_counter() - start,',
    )

    nb = new_notebook()
    nb.metadata = source.metadata
    nb.cells = [
        new_markdown_cell(TITLE_MD),
        new_code_cell(setup_code),
        new_code_cell(dedent_code(CONFIG_CODE)),
        new_markdown_cell("## Notebook 05 LightGBM Tuning Helpers\n\nThis layer adds 04D entry-gate checks, train-inner HPO, finalist selection, official-validation confirmation, and decision records."),
        new_code_cell(dedent_code(NOTEBOOK05_HELPERS_CODE)),
        new_markdown_cell(STAGE05S_MD),
        new_code_cell(dedent_code(STAGE05S_CODE)),
        new_markdown_cell(STAGE05A_MD),
        new_code_cell(dedent_code(STAGE05A_CODE)),
        new_markdown_cell(source.cells[3].source.replace("Stage 0", "Notebook 05")),
        new_code_cell(data_loading_code),
        new_markdown_cell(source.cells[5].source),
        new_code_cell(source.cells[6].source),
        new_markdown_cell("## Notebook 05 Base Helpers\n\nThis section copies active Stage 0 metric, dataset, and LightGBM helper definitions. Notebook 05 uses only the LightGBM and dummy paths below."),
        new_code_cell(base_helpers_code),
        new_markdown_cell(STAGE05B_MD),
        new_code_cell(dedent_code(STAGE05B_CODE)),
        new_markdown_cell(STAGE05C_MD),
        new_code_cell(dedent_code(STAGE05C_CODE)),
        new_markdown_cell(STAGE05D_MD),
        new_code_cell(dedent_code(STAGE05D_CODE)),
        new_markdown_cell(STAGE05E_MD),
        new_code_cell(dedent_code(STAGE05E_CODE)),
        new_markdown_cell(INTERPRETATION_MD),
    ]

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    validate_notebook(nb)
    return nb


def write_run_all_notebook() -> None:
    """Create the explicit Colab upload copy with full 05 run and Drive backup on."""
    nb = nbformat.read(TARGET_NOTEBOOK, as_version=4)
    replacements = {
        "RUN_05A_TO_05E_FULL_PIPELINE = False": "RUN_05A_TO_05E_FULL_PIPELINE = True",
        "RUN_05A_04D_ENTRY_GATE = False": "RUN_05A_04D_ENTRY_GATE = True",
        "RUN_05B_TRAIN_INNER_HPO = False": "RUN_05B_TRAIN_INNER_HPO = True",
        "RUN_05C_SELECT_FINALISTS = False": "RUN_05C_SELECT_FINALISTS = True",
        "RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION = False": "RUN_05D_OFFICIAL_VALIDATION_CONFIRMATION = True",
        "RUN_05E_DECISION_RECORD = False": "RUN_05E_DECISION_RECORD = True",
        "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE = False": "BACKUP_NOTEBOOK05_TO_GOOGLE_DRIVE = True",
        'OPERATOR_SELECTED_EXIT = ""': (
            'OPERATOR_SELECTED_EXIT = "Exit A - Proceed To 05 LightGBM Tuning"'
        ),
        "OPERATOR_ACCEPTS_EXIT_A = False": "OPERATOR_ACCEPTS_EXIT_A = True",
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell.source = source
        cell.outputs = []
        cell.execution_count = None
    nbformat.write(nb, RUN_ALL_NOTEBOOK)
    print(f"Wrote {RUN_ALL_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")


def main() -> None:
    nb = build_notebook()
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, TARGET_NOTEBOOK)
    print(f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")
    write_run_all_notebook()


if __name__ == "__main__":
    main()
