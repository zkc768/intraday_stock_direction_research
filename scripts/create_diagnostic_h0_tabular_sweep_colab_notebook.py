"""Generate the Diagnostic H0 tabular sweep Colab notebook.

The generated notebook is standalone for Colab: it copies the active Stage 0
data-loading, feature, label, split, scaling, and window code from
notebooks/02_config_screening_colab.ipynb, then adds H0-only diagnostic cells.
It does not execute the notebook.
"""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_config_screening_colab.ipynb"
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_diagnostic_h0_tabular_sweep_colab.ipynb"


TITLE_MD = """\
# Diagnostic H0 Tabular Sweep - Validation Only Diagnostic

Protocol: `docs/DIAGNOSTIC_H0_TABULAR_SWEEP_PROTOCOL_2026-06-04.md`

Scope: `validation_only_diagnostic`

This notebook is a post-Stage 0 diagnostic. It does not replace official Stage
0A2/0B outputs, does not authorize holdout/test use, and must not be used as a
thesis result claim.

Entry conditions:

1. `02_config_screening_colab.ipynb` has completed Stage 0A2.
2. Stage 0B has completed, or its blocker has been explicitly recorded.
3. `/content/stage0_config_screening_results/` contains Stage 0 output files.
4. Holdout/test remains closed.

If those Stage 0 files are unavailable in a fresh Colab runtime, the notebook
will first search for an extracted Stage 0 artifact folder. If neither official
outputs nor extracted artifacts are available, it can still run in
`part0_standalone` mode. In that mode Part 0 defines the baseline for this
diagnostic run, but the output cannot claim official Stage 0A2 reproduction.

This notebook is configured for one-pass Colab execution after Stage 0B output
files exist: `RUN_H0_FULL_SEQUENCE = True` runs Part 0, Part 1, Part 2, and
Part 3 in order. Google Drive backup is enabled by default and copies outputs
after each completed part.
"""


CONFIG_CODE = r"""\
TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
MODEL_SEEDS = (101, 202, 303, 404, 505)
H0_FRESH_SEEDS = (606, 707, 808, 909, 1010)

RUN_H0_FULL_SEQUENCE = True
RUN_H0_PART0 = RUN_H0_FULL_SEQUENCE
RUN_H0_PART1 = RUN_H0_FULL_SEQUENCE
RUN_H0_PART2 = RUN_H0_FULL_SEQUENCE
RUN_H0_PART3 = RUN_H0_FULL_SEQUENCE
RUN_H0_ROUND2_IF_TRIGGERED = True

BACKUP_TO_GOOGLE_DRIVE = True
BACKUP_FAILURE_IS_FATAL = True
ALLOW_STANDALONE_H0_BASELINE = True
DRIVE_BACKUP_DIR = Path(
    "/content/drive/MyDrive/intraday_stock_direction_research/diagnostic_h0_tabular_sweep"
)

DIAGNOSTIC_NAME = "diagnostic_h0_tabular_sweep"
RESULT_SCOPE = "validation_only_diagnostic"

H0_LABEL_CONFIG = "h03_bps1p5"
H0_FEATURE_SET = "price_volume_time"
H0_BASELINE_MODEL = "lightgbm"
H0_BASELINE_PROFILE = "profile_B"
H0_BASELINE_WINDOW = 20
H0_BASELINE_TOLERANCE = 1e-4

H0_WEAK_DELTA = 0.005
H0_STRONG_DELTA = 0.010
H0_MIN_POSITIVE_TICKERS = 4
H0_MAX_TOP_TICKER_GAIN_SHARE = 0.50
H0_LOGREG_WARNING_RATE_ABORT = 0.05

H0_PART1_LGBM_WINDOWS = (6, 12, 16, 20, 24, 28, 32, 48, 64)
H0_PART1_LOGREG_WINDOWS = (12, 20, 24, 32, 48)
H0_ROUND2_TRIGGER_WINDOWS = (24, 28, 32)
H0_ROUND2_WINDOWS = (18, 20, 22, 24, 26, 28, 30, 32, 36)
H0_PART2_WINDOWS = (20, 24, 32)

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

LGBM_PROFILES = {
    "profile_A": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 7,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
    },
    "profile_B": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
    },
    "profile_C": {
        "n_estimators": 300,
        "learning_rate": 0.02,
        "max_depth": 8,
        "num_leaves": 63,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
    },
    "profile_D1": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 100,
    },
    "profile_D2": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    "profile_D3": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "min_child_samples": 100,
        "reg_lambda": 1.0,
    },
}

LGBM_PARAMS = LGBM_PROFILES[H0_BASELINE_PROFILE].copy()
LOGREG_PROFILES = {
    "logreg_l2_c1": {"penalty": "l2", "C": 1.0, "solver": "liblinear"},
}

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

STAGE0_OUTPUT_DIR = Path("/content/stage0_config_screening_results")
STAGE0_FILES = {
    "stage0a2_pooled": STAGE0_OUTPUT_DIR / "stage0a2_pooled.csv",
    "stage0a2_summary": STAGE0_OUTPUT_DIR / "stage0a2_summary.csv",
    "stage0_candidates": STAGE0_OUTPUT_DIR / "stage0_candidates.json",
    "stage0b_summary": STAGE0_OUTPUT_DIR / "stage0b_summary.csv",
}
STAGE0_ARTIFACT_DIR_CANDIDATES = (
    Path("artifacts/stage0_desktop_02_config_screening_2026-06-04"),
    Path("/content/artifacts/stage0_desktop_02_config_screening_2026-06-04"),
    Path("/content/stage0_desktop_02_config_screening_2026-06-04"),
    Path("/content/drive/MyDrive/intraday_stock_direction_research/artifacts/stage0_desktop_02_config_screening_2026-06-04"),
    Path("/content/drive/MyDrive/intraday_stock_direction_research/stage0_desktop_02_config_screening_2026-06-04"),
    Path("/content/drive/MyDrive/stage0_desktop_02_config_screening_2026-06-04"),
)
STAGE0_ARTIFACT_FILES = {
    "stage0a2_summary": "stage0a2_table1.csv",
    "stage0_candidates": "stage0_decision_blocks.json",
    "stage0b_summary": "stage0b_table1.csv",
}

OUTPUT_DIR = Path("/content/diagnostic_h0_tabular_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILES = {
    "rules": OUTPUT_DIR / "diagnostic_h0_pre_committed_rules.json",
    "part0": OUTPUT_DIR / "diagnostic_h0_part0_sanity.csv",
    "part1": OUTPUT_DIR / "diagnostic_h0_part1_window_sweep.csv",
    "part2": OUTPUT_DIR / "diagnostic_h0_part2_lgbm_profiles.csv",
    "part3": OUTPUT_DIR / "diagnostic_h0_part3_confirmation.csv",
    "summary": OUTPUT_DIR / "diagnostic_h0_summary.csv",
    "per_ticker": OUTPUT_DIR / "diagnostic_h0_per_ticker.csv",
}

PRECOMMITTED_RULES = {
    "diagnostic_name": DIAGNOSTIC_NAME,
    "scope": RESULT_SCOPE,
    "baseline_source": str(STAGE0_FILES["stage0a2_summary"]),
    "candidate_source": str(STAGE0_FILES["stage0_candidates"]),
    "artifact_dir_candidates": [str(path) for path in STAGE0_ARTIFACT_DIR_CANDIDATES],
    "allow_standalone_h0_baseline": ALLOW_STANDALONE_H0_BASELINE,
    "baseline_cell": {
        "model": H0_BASELINE_MODEL,
        "label_config": H0_LABEL_CONFIG,
        "feature_set": H0_FEATURE_SET,
        "window_size": H0_BASELINE_WINDOW,
        "seeds": MODEL_SEEDS,
    },
    "part0_tolerance": H0_BASELINE_TOLERANCE,
    "weak_delta": H0_WEAK_DELTA,
    "strong_delta": H0_STRONG_DELTA,
    "positive_ticker_count_min": H0_MIN_POSITIVE_TICKERS,
    "top_ticker_gain_share_max": H0_MAX_TOP_TICKER_GAIN_SHARE,
    "round2_trigger_windows": H0_ROUND2_TRIGGER_WINDOWS,
    "round2_windows": H0_ROUND2_WINDOWS,
    "fresh_seeds": H0_FRESH_SEEDS,
}

with OUTPUT_FILES["rules"].open("w", encoding="utf-8") as handle:
    json.dump(PRECOMMITTED_RULES, handle, indent=2)

display(pd.DataFrame([
    {"feature_set": name, "n_features": len(features), "features": ", ".join(features)}
    for name, features in FEATURE_SETS.items()
]))
print("Stage 0 output directory:", STAGE0_OUTPUT_DIR)
print("Stage 0 artifact candidates:", [str(path) for path in STAGE0_ARTIFACT_DIR_CANDIDATES])
print("Diagnostic H0 output directory:", OUTPUT_DIR)
print("Drive backup enabled:", BACKUP_TO_GOOGLE_DRIVE)
print("Drive backup directory:", DRIVE_BACKUP_DIR)
print("Standalone H0 baseline fallback enabled:", ALLOW_STANDALONE_H0_BASELINE)
print("Run switches:", {
    "RUN_H0_FULL_SEQUENCE": RUN_H0_FULL_SEQUENCE,
    "RUN_H0_PART0": RUN_H0_PART0,
    "RUN_H0_PART1": RUN_H0_PART1,
    "RUN_H0_PART2": RUN_H0_PART2,
    "RUN_H0_PART3": RUN_H0_PART3,
    "RUN_H0_ROUND2_IF_TRIGGERED": RUN_H0_ROUND2_IF_TRIGGERED,
})
"""


H0_HELPERS_CODE = r"""\
H0_STATE = {
    "baseline": None,
    "part0_passed": False,
}


def ensure_drive_backup_dir():
    if not BACKUP_TO_GOOGLE_DRIVE:
        return None
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    except ModuleNotFoundError as exc:
        message = "Google Drive backup requested, but google.colab is unavailable."
        if BACKUP_FAILURE_IS_FATAL:
            raise RuntimeError(message) from exc
        print("WARNING:", message)
        return None
    DRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    return DRIVE_BACKUP_DIR


def try_mount_google_drive(reason):
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        return True
    except ModuleNotFoundError:
        print(f"Google Drive mount unavailable for {reason}; continuing without mount.")
        return False


def backup_h0_outputs(reason):
    if not BACKUP_TO_GOOGLE_DRIVE:
        return []
    backup_dir = ensure_drive_backup_dir()
    if backup_dir is None:
        return []
    copied = []
    try:
        for path in OUTPUT_FILES.values():
            if path.exists():
                target = backup_dir / path.name
                shutil.copy2(path, target)
                copied.append(str(target))
        manifest = {
            "diagnostic_name": DIAGNOSTIC_NAME,
            "reason": reason,
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "local_output_dir": str(OUTPUT_DIR),
            "drive_backup_dir": str(backup_dir),
            "copied_files": copied,
        }
        manifest_path = backup_dir / "diagnostic_h0_backup_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        copied.append(str(manifest_path))
        print("Backed up H0 outputs to Drive:", backup_dir)
        return copied
    except Exception as exc:
        message = f"Google Drive backup failed after {reason}: {exc}"
        if BACKUP_FAILURE_IS_FATAL:
            raise RuntimeError(message) from exc
        print("WARNING:", message)
        return copied


def ensure_required_stage0_files():
    missing = [str(path) for path in STAGE0_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Diagnostic H0 requires completed Stage 0 outputs. Missing: "
            + ", ".join(missing)
        )


def missing_stage0_files():
    return [str(path) for path in STAGE0_FILES.values() if not path.exists()]


def find_stage0_artifact_dir():
    def valid_artifact_dir(path):
        return all((path / filename).exists() for filename in STAGE0_ARTIFACT_FILES.values())

    for path in STAGE0_ARTIFACT_DIR_CANDIDATES:
        if valid_artifact_dir(path):
            return path

    if any(str(path).startswith("/content/drive/") for path in STAGE0_ARTIFACT_DIR_CANDIDATES):
        try_mount_google_drive("Stage 0 artifact lookup")
        for path in STAGE0_ARTIFACT_DIR_CANDIDATES:
            if valid_artifact_dir(path):
                return path
    return None


def candidate_payload_contains_expected(payload):
    expected = {
        "label_config": H0_LABEL_CONFIG,
        "feature_set": H0_FEATURE_SET,
        "window_size": H0_BASELINE_WINDOW,
    }
    for candidate in payload.get("candidates", []):
        candidate_with_window = {
            "label_config": candidate.get("label_config"),
            "feature_set": candidate.get("feature_set"),
            "window_size": int(candidate.get("window_size")) if "window_size" in candidate else None,
        }
        if candidate_with_window == expected:
            return True
    return False


def read_stage0_artifact_candidate_payload(artifact_dir):
    path = artifact_dir / STAGE0_ARTIFACT_FILES["stage0_candidates"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocks = payload if isinstance(payload, list) else [payload]
    for block in reversed(blocks):
        if candidate_payload_contains_expected(block):
            return block
    raise RuntimeError(
        "Stage 0 artifact decision blocks do not contain the intended H0 baseline candidate."
    )


def read_h0_baseline_from_artifacts():
    artifact_dir = find_stage0_artifact_dir()
    if artifact_dir is None:
        return None

    read_stage0_artifact_candidate_payload(artifact_dir)
    stage0b_path = artifact_dir / STAGE0_ARTIFACT_FILES["stage0b_summary"]
    stage0b = pd.read_csv(stage0b_path)
    if stage0b.empty:
        raise RuntimeError(f"Stage 0B artifact is empty: {stage0b_path}")

    summary_path = artifact_dir / STAGE0_ARTIFACT_FILES["stage0a2_summary"]
    summary = pd.read_csv(summary_path)
    summary_mask = (
        summary["model"].eq(H0_BASELINE_MODEL)
        & summary["label_config"].eq(H0_LABEL_CONFIG)
        & summary["feature_set"].eq(H0_FEATURE_SET)
        & summary["window_size"].astype(int).eq(H0_BASELINE_WINDOW)
    )
    if int(summary_mask.sum()) != 1:
        raise RuntimeError(
            "Expected exactly one artifact Stage 0A2 summary row for the H0 baseline, "
            f"found {int(summary_mask.sum())} in {summary_path}."
        )
    row = summary.loc[summary_mask].iloc[0].to_dict()
    seed_count = int(row.get("seed_count", 0))
    if seed_count != len(MODEL_SEEDS):
        raise RuntimeError(
            f"Artifact baseline seed_count mismatch: expected {len(MODEL_SEEDS)}, found {seed_count}."
        )
    baseline = {
        "macro_f1_mean": float(row["macro_f1_mean"]),
        "summary_row": row,
        "source": f"stage0_extracted_artifact:{summary_path}",
        "artifact_dir": str(artifact_dir),
    }
    H0_STATE["baseline"] = baseline
    print("Using extracted Stage 0 artifact baseline:", summary_path)
    print("H0 baseline macro_f1_mean:", baseline["macro_f1_mean"])
    return baseline


def read_stage0_candidates():
    ensure_required_stage0_files()
    with STAGE0_FILES["stage0_candidates"].open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("stage0_result") == "do_not_decide_config":
        raise RuntimeError("Stage 0 reported do_not_decide_config; Diagnostic H0 must not run.")
    candidates = payload.get("candidates", [])
    expected = {
        "label_config": H0_LABEL_CONFIG,
        "feature_set": H0_FEATURE_SET,
        "window_size": H0_BASELINE_WINDOW,
    }
    if expected not in candidates:
        raise RuntimeError(
            "The intended H0 baseline is not an official Stage 0A2 candidate. "
            f"Expected {expected}; found {candidates}. Revise the H0 protocol before running."
        )
    return payload


def read_h0_baseline():
    missing = missing_stage0_files()
    if missing:
        artifact_baseline = read_h0_baseline_from_artifacts()
        if artifact_baseline is not None:
            return artifact_baseline
        if not ALLOW_STANDALONE_H0_BASELINE:
            raise FileNotFoundError(
                "Diagnostic H0 requires completed Stage 0 outputs. Missing: "
                + ", ".join(missing)
            )
        baseline = {
            "macro_f1_mean": None,
            "summary_row": {},
            "source": "part0_standalone_pending",
            "missing_stage0_files": missing,
        }
        H0_STATE["baseline"] = baseline
        print("WARNING: Stage 0 output files are missing; using standalone H0 baseline mode.")
        print("Missing Stage 0 files:", missing)
        print("Part 0 will define h0_baseline_macro_f1 for this diagnostic run.")
        return baseline

    read_stage0_candidates()
    summary = pd.read_csv(STAGE0_FILES["stage0a2_summary"])
    pooled = pd.read_csv(STAGE0_FILES["stage0a2_pooled"])

    summary_mask = (
        summary["model"].eq(H0_BASELINE_MODEL)
        & summary["label_config"].eq(H0_LABEL_CONFIG)
        & summary["feature_set"].eq(H0_FEATURE_SET)
        & summary["window_size"].astype(int).eq(H0_BASELINE_WINDOW)
    )
    if int(summary_mask.sum()) != 1:
        raise RuntimeError(
            "Expected exactly one Stage 0A2 summary row for the H0 baseline, "
            f"found {int(summary_mask.sum())}."
        )
    row = summary.loc[summary_mask].iloc[0].to_dict()

    pooled_mask = (
        pooled["model"].eq(H0_BASELINE_MODEL)
        & pooled["label_config"].eq(H0_LABEL_CONFIG)
        & pooled["feature_set"].eq(H0_FEATURE_SET)
        & pooled["window_size"].astype(int).eq(H0_BASELINE_WINDOW)
    )
    observed_seeds = tuple(sorted(int(seed) for seed in pooled.loc[pooled_mask, "seed"].unique()))
    if observed_seeds != MODEL_SEEDS:
        raise RuntimeError(
            f"Baseline pooled seeds mismatch: expected {MODEL_SEEDS}, found {observed_seeds}."
        )

    baseline = {
        "macro_f1_mean": float(row["macro_f1_mean"]),
        "summary_row": row,
        "source": str(STAGE0_FILES["stage0a2_summary"]),
    }
    H0_STATE["baseline"] = baseline
    return baseline


def require_h0_baseline():
    if H0_STATE["baseline"] is None:
        return read_h0_baseline()
    return H0_STATE["baseline"]


def require_part0_passed():
    if not H0_STATE["part0_passed"]:
        raise RuntimeError("Run Diagnostic H0 Part 0 successfully before this part.")
    return require_h0_baseline()


def fit_predict_logreg_h0(dataset, seed, params):
    x_train, y_train = subsample_rows_uniformly(dataset["x_train"], dataset["y_train"], MAX_TRAIN_ROWS, seed=seed)
    max_iter = 2000
    model = LogisticRegression(
        solver=params.get("solver", "liblinear"),
        penalty=params.get("penalty", "l2"),
        C=float(params.get("C", 1.0)),
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


def fit_predict_lightgbm_h0(dataset, seed, params):
    lgb = ensure_lightgbm()
    x_train, y_train = subsample_rows_uniformly(dataset["x_train"], dataset["y_train"], MAX_TRAIN_ROWS, seed=seed)
    lgbm_params = dict(params)
    lgbm_params.setdefault("class_weight", "balanced")
    model = lgb.LGBMClassifier(
        **lgbm_params,
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


def fit_predict_h0_model(dataset, spec):
    model_name = spec["model"]
    seed = int(spec["seed"])
    params = spec.get("params", {})
    if model_name == "logreg":
        return fit_predict_logreg_h0(dataset, seed, params)
    if model_name == "lightgbm":
        return fit_predict_lightgbm_h0(dataset, seed, params)
    raise ValueError(f"Diagnostic H0 supports only tabular models, got {model_name!r}.")


def concentration_from_per_ticker_h0(per_ticker_rows):
    deltas = [row["per_ticker_delta_macro_f1_vs_dummy"] for row in per_ticker_rows]
    positive = [float(delta) for delta in deltas if pd.notna(delta) and delta > 0]
    positive_ticker_count = int(len(positive))
    top_ticker_gain_share = float(max(positive) / sum(positive)) if positive else 0.0
    return positive_ticker_count, top_ticker_gain_share


def run_one_h0_spec(spec):
    dataset = get_dataset(spec["label_config"], spec["feature_set"], spec["window_size"])
    prep_seconds = float(dataset["prep_seconds"])
    pred, fit_seconds, predict_seconds, train_n, fit_status = fit_predict_h0_model(dataset, spec)
    pooled_metrics = evaluate_predictions(dataset["y_validation"], pred)
    pooled_dummy = dummy_metrics(dataset["y_train"], dataset["y_validation"], int(spec["seed"]))
    params_json = json.dumps(spec.get("params", {}), sort_keys=True)

    base_fields = {
        "diagnostic_name": DIAGNOSTIC_NAME,
        "stage": "Diagnostic H0",
        "part": spec["part"],
        "round": spec.get("round", "none"),
        "model": spec["model"],
        "profile": spec["profile"],
        "label_config": spec["label_config"],
        "horizon_k": dataset["horizon_k"],
        "threshold_bps": dataset["threshold_bps"],
        "feature_set": spec["feature_set"],
        "window_size": int(spec["window_size"]),
        "seed": int(spec["seed"]),
        "scope": RESULT_SCOPE,
        "params_json": params_json,
    }

    per_ticker_rows = []
    for ticker in TICKERS:
        val_mask = dataset["validation_owner"] == ticker
        train_mask = dataset["train_owner"] == ticker
        if not val_mask.any():
            continue
        ticker_metrics = evaluate_predictions(dataset["y_validation"][val_mask], pred[val_mask])
        ticker_dummy = dummy_metrics(dataset["y_train"][train_mask], dataset["y_validation"][val_mask], int(spec["seed"]))
        per_ticker_rows.append({
            **base_fields,
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

    positive_ticker_count, top_ticker_gain_share = concentration_from_per_ticker_h0(per_ticker_rows)
    for row in per_ticker_rows:
        row["positive_ticker_count"] = positive_ticker_count
        row["top_ticker_gain_share"] = top_ticker_gain_share

    pooled_row = {
        **base_fields,
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


def run_h0_grid(specs):
    pooled_rows = []
    per_ticker_rows = []
    for idx, spec in enumerate(specs, start=1):
        print(
            f"{idx}/{len(specs)}",
            spec["part"],
            spec["model"],
            spec["profile"],
            spec["label_config"],
            spec["feature_set"],
            "window",
            spec["window_size"],
            "seed",
            spec["seed"],
        )
        pooled, per_ticker = run_one_h0_spec(spec)
        pooled_rows.append(pooled)
        per_ticker_rows.extend(per_ticker)
    return pd.DataFrame(pooled_rows), pd.DataFrame(per_ticker_rows)


def summarize_h0_pooled(pooled, baseline_macro_f1):
    if pooled.empty:
        return pd.DataFrame()
    rows = []
    keys = [
        "diagnostic_name",
        "part",
        "round",
        "model",
        "profile",
        "label_config",
        "horizon_k",
        "threshold_bps",
        "feature_set",
        "window_size",
        "scope",
    ]
    for key_values, group in pooled.groupby(keys, sort=False):
        record = dict(zip(keys, key_values))
        seed_count = int(group["seed"].nunique())
        macro_std = sample_std(group["macro_f1"])
        bal_std = sample_std(group["balanced_accuracy"])
        macro_mean = float(group["macro_f1"].mean())
        delta_vs_base = macro_mean - float(baseline_macro_f1)
        record.update({
            "seed_count": seed_count,
            "baseline_macro_f1": float(baseline_macro_f1),
            "baseline_source": str((H0_STATE.get("baseline") or {}).get("source", "unknown")),
            "macro_f1_mean": macro_mean,
            "macro_f1_std": macro_std,
            "macro_f1_lcb_95": float(
                macro_mean - t_critical_one_sided_95(seed_count) * macro_std / math.sqrt(max(seed_count, 1))
            ),
            "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
            "balanced_accuracy_std": bal_std,
            "dummy_macro_f1_mean": float(group["dummy_macro_f1"].mean()),
            "dummy_balanced_accuracy_mean": float(group["dummy_balanced_accuracy"].mean()),
            "delta_macro_f1_vs_dummy_mean": float(group["delta_macro_f1_vs_dummy"].mean()),
            "delta_balanced_accuracy_vs_dummy_mean": float(group["delta_balanced_accuracy_vs_dummy"].mean()),
            "delta_macro_f1_vs_base": delta_vs_base,
            "n_mean": float(group["n"].mean()),
            "positive_ticker_count": int(round(group["positive_ticker_count"].mean())),
            "top_ticker_gain_share": float(group["top_ticker_gain_share"].mean()),
            "prep_seconds_mean": float(group["prep_seconds"].mean()),
            "fit_seconds_mean": float(group["fit_seconds"].mean()),
            "predict_seconds_mean": float(group["predict_seconds"].mean()),
            "total_seconds_mean": float(group["total_seconds"].mean()),
            "fit_status_values": ",".join(sorted(str(value) for value in group["fit_status"].dropna().unique())),
        })
        record["preliminary_gate"] = bool(
            record["delta_macro_f1_vs_base"] >= H0_WEAK_DELTA
            and record["delta_macro_f1_vs_dummy_mean"] > 0
            and record["positive_ticker_count"] >= H0_MIN_POSITIVE_TICKERS
            and record["top_ticker_gain_share"] <= H0_MAX_TOP_TICKER_GAIN_SHARE
        )
        if record["delta_macro_f1_vs_base"] < H0_WEAK_DELTA:
            record["interpretation"] = "indistinguishable_from_base"
        elif record["delta_macro_f1_vs_base"] < H0_STRONG_DELTA:
            record["interpretation"] = "weak_diagnostic_preference_note_only"
        else:
            record["interpretation"] = "strong_diagnostic_signal_requires_new_branch"
        if record["part"] == "part3_confirmation":
            record["confirmation_status"] = (
                "confirmed"
                if record["delta_macro_f1_vs_base"] >= H0_WEAK_DELTA
                else "lottery_not_confirmed"
            )
        else:
            record["confirmation_status"] = "not_selected_for_confirmation"
        rows.append(record)
    return pd.DataFrame(rows)


def existing_part_frames(kind):
    paths = {
        "part0": OUTPUT_FILES["part0"],
        "part1": OUTPUT_FILES["part1"],
        "part2": OUTPUT_FILES["part2"],
        "part3": OUTPUT_FILES["part3"],
    }
    frames = []
    for path in paths.values():
        if path.exists():
            frames.append(pd.read_csv(path))
    return frames


def write_h0_outputs(part_key, pooled, per_ticker, baseline_macro_f1):
    pooled.to_csv(OUTPUT_FILES[part_key], index=False)
    per_ticker_frames = []
    existing_per_ticker = OUTPUT_FILES["per_ticker"]
    if existing_per_ticker.exists():
        prior = pd.read_csv(existing_per_ticker)
        prior = prior.loc[~prior["part"].eq(pooled["part"].iloc[0])]
        per_ticker_frames.append(prior)
    per_ticker_frames.append(per_ticker)
    pd.concat(per_ticker_frames, ignore_index=True).to_csv(OUTPUT_FILES["per_ticker"], index=False)

    pooled_frames = existing_part_frames("pooled")
    combined = pd.concat(pooled_frames, ignore_index=True) if pooled_frames else pooled
    summary = summarize_h0_pooled(combined, baseline_macro_f1)
    summary.to_csv(OUTPUT_FILES["summary"], index=False)
    print("wrote", str(OUTPUT_FILES[part_key]), str(OUTPUT_FILES["summary"]), str(OUTPUT_FILES["per_ticker"]))
    backup_h0_outputs(f"completed_{part_key}")
    return summary


def make_lgbm_spec(part, round_name, profile, window_size, seed):
    return {
        "part": part,
        "round": round_name,
        "model": "lightgbm",
        "profile": profile,
        "label_config": H0_LABEL_CONFIG,
        "feature_set": H0_FEATURE_SET,
        "window_size": int(window_size),
        "seed": int(seed),
        "params": LGBM_PROFILES[profile].copy(),
    }


def make_logreg_spec(part, round_name, profile, window_size, seed):
    return {
        "part": part,
        "round": round_name,
        "model": "logreg",
        "profile": profile,
        "label_config": H0_LABEL_CONFIG,
        "feature_set": H0_FEATURE_SET,
        "window_size": int(window_size),
        "seed": int(seed),
        "params": LOGREG_PROFILES[profile].copy(),
    }


def build_part0_specs():
    return [
        make_lgbm_spec("part0_sanity", "part0", H0_BASELINE_PROFILE, H0_BASELINE_WINDOW, seed)
        for seed in MODEL_SEEDS
    ]


def build_part1_specs():
    specs = [
        make_lgbm_spec("part1_window_sweep", "round1", H0_BASELINE_PROFILE, window_size, seed)
        for window_size in H0_PART1_LGBM_WINDOWS
        for seed in MODEL_SEEDS
    ]
    specs.extend(
        make_logreg_spec("part1_window_sweep", "round1", "logreg_l2_c1", window_size, seed)
        for window_size in H0_PART1_LOGREG_WINDOWS
        for seed in MODEL_SEEDS
    )
    return specs


def build_round2_specs():
    return [
        make_lgbm_spec("part1_window_sweep", "round2", H0_BASELINE_PROFILE, window_size, seed)
        for window_size in H0_ROUND2_WINDOWS
        for seed in MODEL_SEEDS
    ]


def build_part2_specs():
    return [
        make_lgbm_spec("part2_lgbm_profiles", "part2", profile, window_size, seed)
        for profile in LGBM_PROFILES
        for window_size in H0_PART2_WINDOWS
        for seed in MODEL_SEEDS
    ]


def round2_is_triggered(summary):
    if summary.empty:
        return False
    mask = (
        summary["part"].eq("part1_window_sweep")
        & summary["round"].eq("round1")
        & summary["model"].eq("lightgbm")
        & summary["window_size"].astype(int).isin(H0_ROUND2_TRIGGER_WINDOWS)
        & (summary["delta_macro_f1_vs_base"] >= H0_WEAK_DELTA)
        & (summary["positive_ticker_count"] >= H0_MIN_POSITIVE_TICKERS)
        & (summary["top_ticker_gain_share"] <= H0_MAX_TOP_TICKER_GAIN_SHARE)
    )
    return bool(mask.any())


def spec_from_summary_row(row, seed):
    if row["model"] == "lightgbm":
        return make_lgbm_spec("part3_confirmation", "fresh_seed", row["profile"], int(row["window_size"]), seed)
    if row["model"] == "logreg":
        return make_logreg_spec("part3_confirmation", "fresh_seed", row["profile"], int(row["window_size"]), seed)
    raise ValueError(f"Unsupported confirmation model: {row['model']}")


def build_confirmation_specs():
    if not OUTPUT_FILES["summary"].exists():
        raise FileNotFoundError("No H0 summary found. Run Part 1 and/or Part 2 first.")
    summary = pd.read_csv(OUTPUT_FILES["summary"])
    eligible = summary.loc[
        summary["preliminary_gate"].astype(bool)
        & ~summary["part"].eq("part3_confirmation")
    ].sort_values("delta_macro_f1_vs_base", ascending=False)
    top = eligible.head(3)
    if top.empty:
        print("No cells cleared preliminary gates; Part 3 has no specs.")
        return []
    specs = []
    for _, row in top.iterrows():
        for seed in H0_FRESH_SEEDS:
            specs.append(spec_from_summary_row(row, seed))
    display(top)
    return specs


def assert_logreg_warning_rate_ok(pooled):
    mask = pooled["model"].eq("logreg")
    if not mask.any():
        return
    warning_rate = float(pooled.loc[mask, "fit_status"].eq("convergence_warning").mean())
    if warning_rate > H0_LOGREG_WARNING_RATE_ABORT:
        raise RuntimeError(
            f"LogReg convergence warning rate {warning_rate:.3f} exceeds "
            f"{H0_LOGREG_WARNING_RATE_ABORT:.3f}; stop and inspect."
        )
"""


PART0_MD = """\
## Part 0 - Sanity Check

Part 0 rereads the official Stage 0A2 baseline dynamically and reruns the exact
baseline cell in this new notebook. It must reproduce the official
`macro_f1_mean` within `1e-4`, otherwise H0 aborts.
"""


PART0_CODE = r"""\
if RUN_H0_PART0:
    baseline = read_h0_baseline()
    print("H0 baseline macro_f1_mean before Part 0:", baseline["macro_f1_mean"])
    part0_pooled, part0_per_ticker = run_h0_grid(build_part0_specs())
    provisional_baseline = (
        float(part0_pooled["macro_f1"].mean())
        if baseline["macro_f1_mean"] is None
        else float(baseline["macro_f1_mean"])
    )
    part0_summary = summarize_h0_pooled(part0_pooled, provisional_baseline)
    display(part0_summary)
    part0_value = float(part0_summary["macro_f1_mean"].iloc[0])
    if baseline["macro_f1_mean"] is None:
        baseline = {
            **baseline,
            "macro_f1_mean": part0_value,
            "source": "part0_standalone",
        }
        H0_STATE["baseline"] = baseline
        print("Standalone H0 baseline macro_f1_mean:", part0_value)
        print("This run cannot claim reproduction of official Stage 0A2 output.")
    else:
        diff = abs(part0_value - baseline["macro_f1_mean"])
        if diff > H0_BASELINE_TOLERANCE:
            part0_pooled.to_csv(OUTPUT_FILES["part0"], index=False)
            backup_h0_outputs("part0_failed_baseline_reproduction")
            raise RuntimeError(
                f"Part 0 failed baseline reproduction: diff={diff:.8f}, "
                f"tolerance={H0_BASELINE_TOLERANCE}."
            )
    H0_STATE["part0_passed"] = True
    h0_summary = write_h0_outputs("part0", part0_pooled, part0_per_ticker, baseline["macro_f1_mean"])
else:
    print("RUN_H0_PART0 is False; Part 0 sanity check not run.")
"""


PART1_MD = """\
## Part 1 - Window Sweep

Part 1 runs sparse window anchors. Round 2 dense sweep is conditional and only
runs when the pre-committed trigger is met and `RUN_H0_ROUND2_IF_TRIGGERED` is
also set to True.
"""


PART1_CODE = r"""\
if RUN_H0_PART1:
    baseline = require_part0_passed()
    part1_specs = build_part1_specs()
    part1_pooled, part1_per_ticker = run_h0_grid(part1_specs)
    assert_logreg_warning_rate_ok(part1_pooled)
    part1_summary = summarize_h0_pooled(part1_pooled, baseline["macro_f1_mean"])
    display(part1_summary.sort_values("delta_macro_f1_vs_base", ascending=False))

    if round2_is_triggered(part1_summary):
        print("Round 2 trigger met.")
        if RUN_H0_ROUND2_IF_TRIGGERED:
            round2_pooled, round2_per_ticker = run_h0_grid(build_round2_specs())
            part1_pooled = pd.concat([part1_pooled, round2_pooled], ignore_index=True)
            part1_per_ticker = pd.concat([part1_per_ticker, round2_per_ticker], ignore_index=True)
            part1_summary = summarize_h0_pooled(part1_pooled, baseline["macro_f1_mean"])
            display(part1_summary.sort_values("delta_macro_f1_vs_base", ascending=False))
        else:
            print("RUN_H0_ROUND2_IF_TRIGGERED is False; Round 2 not run.")
    else:
        print("Round 2 trigger not met.")

    h0_summary = write_h0_outputs("part1", part1_pooled, part1_per_ticker, baseline["macro_f1_mean"])
else:
    print("RUN_H0_PART1 is False; Part 1 window sweep not run.")
"""


PART2_MD = """\
## Part 2 - LightGBM Profile Sweep

Part 2 runs the neutral LightGBM profiles from the H0 protocol over windows
20, 24, and 32.
"""


PART2_CODE = r"""\
if RUN_H0_PART2:
    baseline = require_part0_passed()
    part2_pooled, part2_per_ticker = run_h0_grid(build_part2_specs())
    part2_summary = summarize_h0_pooled(part2_pooled, baseline["macro_f1_mean"])
    display(part2_summary.sort_values("delta_macro_f1_vs_base", ascending=False))
    h0_summary = write_h0_outputs("part2", part2_pooled, part2_per_ticker, baseline["macro_f1_mean"])
else:
    print("RUN_H0_PART2 is False; Part 2 LightGBM profile sweep not run.")
"""


PART3_MD = """\
## Part 3 - Fresh-Seed Confirmation

Part 3 selects the top 3 preliminary-gate cells from existing H0 summary output
and reruns only those cells with fresh seeds.
"""


PART3_CODE = r"""\
if RUN_H0_PART3:
    baseline = require_part0_passed()
    confirmation_specs = build_confirmation_specs()
    if confirmation_specs:
        part3_pooled, part3_per_ticker = run_h0_grid(confirmation_specs)
        part3_summary = summarize_h0_pooled(part3_pooled, baseline["macro_f1_mean"])
        part3_summary["confirmation_status"] = np.where(
            part3_summary["delta_macro_f1_vs_base"] >= H0_WEAK_DELTA,
            "confirmed",
            "lottery_not_confirmed",
        )
        display(part3_summary.sort_values("delta_macro_f1_vs_base", ascending=False))
        h0_summary = write_h0_outputs("part3", part3_pooled, part3_per_ticker, baseline["macro_f1_mean"])
    else:
        print("No confirmation specs generated; Part 3 skipped.")
else:
    print("RUN_H0_PART3 is False; Part 3 confirmation not run.")
"""


INTERPRETATION_MD = """\
## Interpretation Boundary

Diagnostic H0 is `validation_only_diagnostic`.

Allowed wording:

```text
Diagnostic H0 found a validation-only diagnostic preference for [cell], but the
result is post-Stage 0 and cannot replace the official Stage 0 selection without
a new pre-registered branch.
```

Forbidden wording:

```text
Window 24 is the best configuration.
The official Stage 0 winner should be replaced.
This result is ready for holdout/test.
The thesis model is selected.
```

Cells with `delta_macro_f1_vs_base < 0.005` are treated as indistinguishable
from the official baseline. Cells between `0.005` and `0.010` are weak
diagnostic notes only. Cells at or above `0.010` require a new pre-registered
branch before use.
"""


def dedent_code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def validate_code_cells(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell_{index}")


def main() -> None:
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    data_loading_code = source.cells[4].source.replace(
        "RUN_ANY_STAGE = bool(RUN_STAGE0S or RUN_STAGE0A1 or RUN_STAGE0A2 or RUN_STAGE0B)",
        "RUN_ANY_STAGE = bool(RUN_H0_PART0 or RUN_H0_PART1 or RUN_H0_PART2 or RUN_H0_PART3)",
    ).replace(
        'print("All RUN_STAGE0* switches are False; data loading skipped.")',
        'print("All RUN_H0* switches are False; data loading skipped.")',
    )

    nb = new_notebook()
    nb.metadata = source.metadata
    nb.cells = [
        new_markdown_cell(TITLE_MD),
        new_code_cell(source.cells[1].source),
        new_code_cell(dedent_code(CONFIG_CODE)),
        new_markdown_cell(source.cells[3].source),
        new_code_cell(data_loading_code),
        new_markdown_cell(source.cells[5].source),
        new_code_cell(source.cells[6].source),
        new_markdown_cell("## Diagnostic H0 Model And Metrics Helpers\n\nThis section reuses Stage 0 helper definitions, then H0 overrides the tabular run layer below."),
        new_code_cell(source.cells[8].source),
        new_markdown_cell("## Diagnostic H0 Run Helpers\n\nThese helpers implement the H0 baseline read, Part 0 sanity gate, window sweep, LightGBM profiles, and fresh-seed confirmation."),
        new_code_cell(dedent_code(H0_HELPERS_CODE)),
        new_markdown_cell(PART0_MD),
        new_code_cell(dedent_code(PART0_CODE)),
        new_markdown_cell(PART1_MD),
        new_code_cell(dedent_code(PART1_CODE)),
        new_markdown_cell(PART2_MD),
        new_code_cell(dedent_code(PART2_CODE)),
        new_markdown_cell(PART3_MD),
        new_code_cell(dedent_code(PART3_CODE)),
        new_markdown_cell(INTERPRETATION_MD),
    ]

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    validate_code_cells(nb)
    nbformat.write(nb, TARGET_NOTEBOOK)
    print(f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
