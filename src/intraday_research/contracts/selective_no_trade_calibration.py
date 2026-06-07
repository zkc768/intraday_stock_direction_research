from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NOTEBOOK06_SCOPE = "validation_only"
COVERAGE_GRID = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
DECISION_COVERAGE_GRID = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40)
MIN_INTERPRETABLE_COVERAGE = 0.30
MIN_DECISION_DELTA_MACRO_F1 = 0.005
MIN_POSITIVE_SEED_COUNT = 4
MIN_POSITIVE_DECISION_COVERAGE_COUNT = 4
NOT_SUPPORTED_FAILURE_COVERAGE_COUNT = 4
INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MIN = 1
INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MAX = 3
INCONCLUSIVE_NOISE_COVERAGE_COUNT = 4
INCONCLUSIVE_MIXED_SEED_COVERAGE_COUNT = 3
INCONCLUSIVE_WARNING_COVERAGE_COUNT = 2
RANDOM_ABSTENTION_REPEATS = 100
RANDOM_ABSTENTION_BASE_SEED = 260606
CALIBRATION_BIN_COUNT = 20
CALIBRATION_PRIMARY_BINNING = "quantile"
CALIBRATION_SENSITIVITY_BINNING = "uniform"
PRIMARY_CONFIDENCE_COLUMN = "confidence"
FLOAT_TOLERANCE = 1e-9
PLOT_DPI = 300
PLOT_FIGSIZE = (8.0, 6.0)
T_CRITICAL_ONE_SIDED_95 = {
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

NOTEBOOK05_REQUIRED_FILES = {
    "entry": "notebook05_entry_decision.json",
    "decision": "notebook05_decision_record.json",
    "run_manifest": "notebook05_run_manifest.json",
    "official_summary": "notebook05_official_validation_summary.csv",
    "official_pooled": "notebook05_official_validation_pooled.csv",
    "official_per_ticker": "notebook05_official_validation_per_ticker.csv",
}

HARD_REQUIRED_DECISION_FIELDS = (
    "scope",
    "holdout_test_authorized",
    "selective_threshold_selected",
    "selected_profile_id",
    "selected_profile_source",
)

REQUIRED_OFFICIAL_POOLED_FIELDS = (
    "profile_id",
    "profile_role",
    "seed",
    "ticker_or_pooled",
    "train_n",
    "validation_n",
    "train_class0_n",
    "train_class1_n",
    "train_positive_rate",
    "validation_sample_id_hash",
    "sample_id_mismatch_count",
    "prediction_artifact",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "stratified_dummy_macro_f1",
    "delta_macro_f1_vs_stratified_dummy",
    "always_up_dummy_macro_f1",
    "delta_macro_f1_vs_always_up_dummy",
    "scope",
)

REQUIRED_NPZ_KEYS = (
    "y_true",
    "y_pred",
    "prob_up",
    "validation_sample_id",
    "ticker",
    "timestamp",
    "confidence",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Notebook 05 file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Notebook 05 JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Notebook 05 file: {path}")
    return pd.read_csv(path)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_hash(values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _is_false(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"false", "0", "no", "n"}


def _is_true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _require_false(record: dict[str, Any], field: str, source: Path, *, required: bool) -> None:
    if field not in record:
        if required:
            raise ValueError(f"{source} is missing required field: {field}")
        return
    if not _is_false(record[field]):
        raise ValueError(f"{field} is not false in {source}")


def _require_scope(record: dict[str, Any], source: Path, *, required: bool) -> None:
    if "scope" not in record:
        if required:
            raise ValueError(f"{source} is missing required field: scope")
        return
    if str(record["scope"]) != NOTEBOOK06_SCOPE:
        raise ValueError(f"scope is not {NOTEBOOK06_SCOPE} in {source}")


def _check_no_holdout_or_test_path(raw_path: Any) -> None:
    text = str(raw_path)
    lowered = text.replace("\\", "/").lower()
    parts = [part for part in lowered.split("/") if part]
    if any(("holdout" in part or "test" in part) for part in parts):
        raise ValueError(f"Prediction artifact path may not contain holdout/test: {text}")


def _resolve_prediction_artifact(notebook05_dir: Path, raw_path: Any) -> Path:
    if pd.isna(raw_path) or not str(raw_path).strip():
        raise ValueError("Selected primary profile is missing prediction_artifact")
    _check_no_holdout_or_test_path(raw_path)
    raw_text = str(raw_path)
    raw = Path(raw_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.append(notebook05_dir / raw)
    candidates.append(notebook05_dir / "predictions" / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing prediction artifact: "
        + raw_text
        + " (checked "
        + ", ".join(str(candidate) for candidate in candidates)
        + ")"
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _validate_train_prior_columns(pooled: pd.DataFrame, source: Path) -> None:
    class0 = pd.to_numeric(pooled["train_class0_n"], errors="raise")
    class1 = pd.to_numeric(pooled["train_class1_n"], errors="raise")
    rate = pd.to_numeric(pooled["train_positive_rate"], errors="raise")
    total = class0 + class1
    if (total <= 0).any():
        raise ValueError(f"{source} has non-positive train class total")
    expected = class1 / total
    mismatch = np.abs(rate.to_numpy(dtype=float) - expected.to_numpy(dtype=float)) > FLOAT_TOLERANCE
    if bool(np.any(mismatch)):
        raise ValueError(
            f"{source} has train_positive_rate inconsistent with train class counts"
        )


def resolve_notebook06_primary_profile(decision_record: dict, pooled: pd.DataFrame) -> str:
    source = str(decision_record.get("selected_profile_source", "")).strip().lower()
    if "official_validation_best" in source:
        raise ValueError("selected_profile_source is official_validation_best; 06 cannot continue")

    downstream = str(decision_record.get("downstream_primary_profile_id", "")).strip()
    if downstream:
        return downstream

    retained_default = _is_true(decision_record.get("retained_default_lgbm_04", False))
    official_status = str(decision_record.get("official_validation_status", "")).strip()
    if retained_default or official_status == "retain_default_lgbm_04":
        if "profile_id" in pooled.columns and (pooled["profile_id"].astype(str) == "default_lgbm_04").any():
            return "default_lgbm_04"
        raise ValueError("Notebook 05 retained default_lgbm_04, but pooled rows do not contain it")

    selected = str(decision_record.get("selected_profile_id", "")).strip()
    if not selected:
        raise ValueError("Notebook 05 decision record is missing selected_profile_id")
    return selected


def load_notebook06_prediction_artifact(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifact: {path}")
    # Notebook 05 prediction artifacts may contain string identifier arrays
    # saved by older notebook copies as object dtype. These files are produced
    # by Notebook 05 itself and immediately revalidated below for required keys,
    # equal lengths, unique sample ids, probability bounds, and sample hashes.
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required .npz arrays: {missing}")
        payload = {key: data[key] for key in REQUIRED_NPZ_KEYS}

    lengths = {key: len(np.asarray(value)) for key, value in payload.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path} has unequal .npz array lengths: {lengths}")
    sample_ids = np.asarray(payload["validation_sample_id"]).astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError(f"{path} has duplicated validation_sample_id")

    prob_up = np.asarray(payload["prob_up"], dtype=float)
    confidence = np.asarray(payload["confidence"], dtype=float)
    expected_confidence = np.maximum(prob_up, 1.0 - prob_up)
    if not np.allclose(confidence, expected_confidence, atol=FLOAT_TOLERANCE, rtol=0.0):
        raise ValueError(
            f"{path} confidence differs from max(prob_up, 1 - prob_up) by more than FLOAT_TOLERANCE"
        )
    if np.any((prob_up < -FLOAT_TOLERANCE) | (prob_up > 1.0 + FLOAT_TOLERANCE)):
        raise ValueError(f"{path} has prob_up outside [0, 1]")

    payload["validation_sample_id"] = sample_ids
    payload["ticker"] = np.asarray(payload["ticker"]).astype(str)
    payload["timestamp"] = np.asarray(payload["timestamp"]).astype(str)
    payload["y_true"] = np.asarray(payload["y_true"]).astype(int)
    payload["y_pred"] = np.asarray(payload["y_pred"]).astype(int)
    payload["prob_up"] = prob_up
    payload["confidence"] = confidence
    return payload


def build_canonical_prediction_frame(npz_payload: dict, metadata: dict) -> pd.DataFrame:
    y_true = np.asarray(npz_payload["y_true"]).astype(int)
    y_pred = np.asarray(npz_payload["y_pred"]).astype(int)
    prob_up = np.asarray(npz_payload["prob_up"], dtype=float)
    frame = pd.DataFrame(
        {
            "profile_id": str(metadata.get("profile_id", "")),
            "profile_role": str(metadata.get("profile_role", "")),
            "seed": int(metadata.get("seed", -1)),
            "validation_sample_id": np.asarray(npz_payload["validation_sample_id"]).astype(str),
            "ticker": np.asarray(npz_payload["ticker"]).astype(str),
            "timestamp": np.asarray(npz_payload["timestamp"]).astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_up": prob_up,
            "y_prob_up": prob_up,
            "confidence": np.asarray(npz_payload["confidence"], dtype=float),
            "correct": (y_true == y_pred).astype(int),
        }
    )
    frame["prediction_artifact"] = str(metadata.get("prediction_artifact", ""))
    return frame


def assert_notebook06_artifact_contract(notebook05_dir: Path) -> dict:
    notebook05_dir = Path(notebook05_dir)
    if not notebook05_dir.exists():
        raise FileNotFoundError(f"Missing Notebook 05 artifact directory: {notebook05_dir}")

    paths = {
        key: notebook05_dir / file_name
        for key, file_name in NOTEBOOK05_REQUIRED_FILES.items()
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing required Notebook 05 file: {path}")

    prediction_dir = notebook05_dir / "predictions"
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Missing required Notebook 05 prediction directory: {prediction_dir}")

    entry = _read_json(paths["entry"])
    decision = _read_json(paths["decision"])
    run_manifest = _read_json(paths["run_manifest"])
    for field in HARD_REQUIRED_DECISION_FIELDS:
        if field not in decision:
            raise ValueError(f"{paths['decision']} is missing required field: {field}")

    _require_scope(decision, paths["decision"], required=True)
    _require_scope(run_manifest, paths["run_manifest"], required=True)
    _require_scope(entry, paths["entry"], required=False)
    _require_false(decision, "holdout_test_authorized", paths["decision"], required=True)
    _require_false(run_manifest, "holdout_test_authorized", paths["run_manifest"], required=True)
    _require_false(entry, "holdout_test_authorized", paths["entry"], required=False)
    _require_false(decision, "selective_threshold_selected", paths["decision"], required=True)
    _require_false(run_manifest, "selective_threshold_selected", paths["run_manifest"], required=True)
    _require_false(entry, "selective_threshold_selected", paths["entry"], required=False)

    pooled = _read_csv(paths["official_pooled"])
    per_ticker = _read_csv(paths["official_per_ticker"])
    official_summary = _read_csv(paths["official_summary"])
    if pooled.empty:
        raise ValueError(f"{paths['official_pooled']} is empty")
    _require_columns(pooled, REQUIRED_OFFICIAL_POOLED_FIELDS, paths["official_pooled"])
    _validate_train_prior_columns(pooled, paths["official_pooled"])
    if not (pooled["scope"].astype(str) == NOTEBOOK06_SCOPE).all():
        raise ValueError(f"scope is not {NOTEBOOK06_SCOPE} in {paths['official_pooled']}")
    mismatch_count = pd.to_numeric(pooled["sample_id_mismatch_count"], errors="raise")
    if not (mismatch_count == 0).all():
        raise ValueError(f"official pooled sample_id_mismatch_count is not zero in {paths['official_pooled']}")
    if "official_validation_used_for_selection" in pooled.columns:
        used = pooled["official_validation_used_for_selection"].map(_is_true)
        if bool(used.any()):
            raise ValueError("official validation was marked as used for selection in pooled rows")
    if "selected_profile_source" in pooled.columns:
        bad_source = pooled["selected_profile_source"].astype(str).str.lower().str.contains(
            "official_validation_best", regex=False, na=False
        )
        if bool(bad_source.any()):
            raise ValueError("official pooled rows contain selected_profile_source=official_validation_best")

    primary_profile_id = resolve_notebook06_primary_profile(decision, pooled)
    primary_rows = pooled[pooled["profile_id"].astype(str) == primary_profile_id].copy()
    if primary_rows.empty:
        raise ValueError(f"Primary profile {primary_profile_id} is absent from {paths['official_pooled']}")
    primary_rows = primary_rows[
        primary_rows["prediction_artifact"].astype(str).str.strip().astype(bool)
    ].copy()
    if primary_rows.empty:
        raise ValueError(f"Primary profile {primary_profile_id} has no prediction_artifact rows")

    sample_hashes = []
    prediction_paths = []
    canonical_frames = []
    for _, row in primary_rows.iterrows():
        artifact_path = _resolve_prediction_artifact(notebook05_dir, row["prediction_artifact"])
        payload = load_notebook06_prediction_artifact(artifact_path)
        computed_hash = _stable_hash(payload["validation_sample_id"])
        row_hash = str(row["validation_sample_id_hash"])
        if computed_hash != row_hash:
            raise ValueError(
                f"{artifact_path} validation_sample_id_hash differs from official pooled row"
            )
        validation_n = int(pd.to_numeric(row["validation_n"], errors="raise"))
        if len(payload["validation_sample_id"]) != validation_n:
            raise ValueError(f"{artifact_path} row count differs from validation_n")
        sample_hashes.append(computed_hash)
        prediction_paths.append(artifact_path)
        canonical_frames.append(build_canonical_prediction_frame(payload, row.to_dict()))

    first_payload = load_notebook06_prediction_artifact(prediction_paths[0])
    first_order = first_payload["validation_sample_id"]
    for artifact_path in prediction_paths[1:]:
        payload = load_notebook06_prediction_artifact(artifact_path)
        if not np.array_equal(payload["validation_sample_id"], first_order):
            raise ValueError(f"{artifact_path} validation_sample_id order differs across seeds")
    if len(set(sample_hashes)) != 1:
        raise ValueError("validation_sample_id_hash differs across primary-profile seed artifacts")

    non_dummy = pooled[
        pooled["prediction_artifact"].astype(str).str.strip().astype(bool)
    ]["profile_id"].astype(str)
    secondary_profile_ids = sorted(set(non_dummy.tolist()) - {primary_profile_id})

    return {
        "notebook05_dir": str(notebook05_dir),
        "primary_profile_id": primary_profile_id,
        "primary_profile_source": str(decision.get("selected_profile_source", "")),
        "secondary_profile_ids": secondary_profile_ids,
        "required_files_present": True,
        "required_columns_present": True,
        "required_npz_arrays_present": True,
        "sample_id_hash": sample_hashes[0],
        "sample_id_mismatch_count": int(mismatch_count.max()),
        "prediction_artifact_count": len(prediction_paths),
        "prediction_artifacts": [str(path) for path in prediction_paths],
        "holdout_test_authorized": False,
        "selective_threshold_selected": False,
        "notebook05_entry_decision_sha256": _sha256_file(paths["entry"]),
        "notebook05_decision_record_sha256": _sha256_file(paths["decision"]),
        "notebook05_run_manifest_sha256": _sha256_file(paths["run_manifest"]),
        "official_pooled_rows": int(len(pooled)),
        "official_per_ticker_rows": int(len(per_ticker)),
        "official_summary_rows": int(len(official_summary)),
        "canonical_prediction_rows": int(sum(len(frame) for frame in canonical_frames)),
        "contract_passed": True,
        "failure_reason": "",
    }


def calibration_bins(values: np.ndarray, outcomes: np.ndarray, n_bins: int, strategy: str) -> list[dict]:
    scores = np.asarray(values, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(scores) != len(y):
        raise ValueError("values and outcomes must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError(f"Unsupported calibration binning strategy: {strategy}")
    if len(scores) == 0:
        return [
            {
                "bin_index": i,
                "bin_strategy": strategy,
                "bin_lower_edge": np.nan,
                "bin_upper_edge": np.nan,
                "bin_count": 0,
                "bin_weight": 0.0,
                "bin_avg_score": np.nan,
                "bin_avg_outcome": np.nan,
                "bin_signed_gap": np.nan,
                "bin_abs_gap": np.nan,
            }
            for i in range(n_bins)
        ]

    rows = []
    if strategy == "uniform":
        for i in range(n_bins):
            lower = i / n_bins
            upper = (i + 1) / n_bins
            if i == n_bins - 1:
                mask = (scores >= lower) & (scores <= upper)
            else:
                mask = (scores >= lower) & (scores < upper)
            rows.append(_calibration_bin_row(i, strategy, lower, upper, scores[mask], y[mask], len(scores)))
        return rows

    order = np.argsort(scores, kind="mergesort")
    for i, indices in enumerate(np.array_split(order, n_bins)):
        if len(indices) == 0:
            rows.append(_calibration_bin_row(i, strategy, np.nan, np.nan, scores[indices], y[indices], len(scores)))
            continue
        rows.append(
            _calibration_bin_row(
                i,
                strategy,
                float(np.min(scores[indices])),
                float(np.max(scores[indices])),
                scores[indices],
                y[indices],
                len(scores),
            )
        )
    return rows


def _calibration_bin_row(
    index: int,
    strategy: str,
    lower: float,
    upper: float,
    scores: np.ndarray,
    outcomes: np.ndarray,
    total_count: int,
) -> dict:
    count = int(len(scores))
    if count == 0:
        avg_score = np.nan
        avg_outcome = np.nan
        signed_gap = np.nan
        abs_gap = np.nan
    else:
        avg_score = float(np.mean(scores))
        avg_outcome = float(np.mean(outcomes))
        signed_gap = avg_score - avg_outcome
        abs_gap = abs(signed_gap)
    return {
        "bin_index": int(index),
        "bin_strategy": strategy,
        "bin_lower_edge": float(lower) if not pd.isna(lower) else np.nan,
        "bin_upper_edge": float(upper) if not pd.isna(upper) else np.nan,
        "bin_count": count,
        "bin_weight": float(count / total_count) if total_count else 0.0,
        "bin_avg_score": avg_score,
        "bin_avg_outcome": avg_outcome,
        "bin_signed_gap": signed_gap,
        "bin_abs_gap": abs_gap,
    }


def ece_from_bins(rows: list[dict]) -> float:
    total = sum(int(row["bin_count"]) for row in rows)
    if total == 0:
        return float("nan")
    value = 0.0
    for row in rows:
        count = int(row["bin_count"])
        if count == 0:
            continue
        value += (count / total) * abs(float(row["bin_signed_gap"]))
    return float(value)


def risk_coverage_curve(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    conf = np.asarray(confidence, dtype=float)
    if not (len(y) == len(pred) == len(conf)):
        raise ValueError("y_true, y_pred, and confidence must have equal lengths")
    n = len(y)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "retained_n",
                "coverage",
                "selective_risk",
                "selective_accuracy",
                "error_count",
                "min_retained_confidence",
            ]
        )
    order = np.argsort(-conf, kind="mergesort")
    errors = (y[order] != pred[order]).astype(int)
    cumulative_errors = np.cumsum(errors)
    retained_n = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "retained_n": retained_n,
            "coverage": retained_n / n,
            "selective_risk": cumulative_errors / retained_n,
            "selective_accuracy": 1.0 - (cumulative_errors / retained_n),
            "error_count": cumulative_errors,
            "min_retained_confidence": conf[order],
        }
    )


def aurc_from_curve(curve: pd.DataFrame) -> float:
    if curve.empty:
        return float("nan")
    coverage = curve["coverage"].to_numpy(dtype=float)
    risk = curve["selective_risk"].to_numpy(dtype=float)
    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([risk[0]], risk))
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def selective_retained_indices(
    confidence: np.ndarray,
    validation_sample_id: np.ndarray,
    coverage_target: float,
) -> np.ndarray:
    conf = np.asarray(confidence, dtype=float)
    sample_ids = np.asarray(validation_sample_id).astype(str)
    if len(conf) != len(sample_ids):
        raise ValueError("confidence and validation_sample_id must have equal lengths")
    if not (0.0 < float(coverage_target) <= 1.0):
        raise ValueError("coverage_target must be in (0, 1]")
    n = len(conf)
    if n == 0:
        return np.asarray([], dtype=int)
    retained_n = int(math.ceil(float(coverage_target) * n))
    retained_n = max(1, min(n, retained_n))
    order = np.lexsort((sample_ids, -conf))
    return order[:retained_n].astype(int)


def same_row_stratified_dummy_predict(
    train_class0_n: int,
    train_class1_n: int,
    n_validation: int,
    seed: int,
) -> np.ndarray:
    class0_n = int(train_class0_n)
    class1_n = int(train_class1_n)
    n = int(n_validation)
    if class0_n < 0 or class1_n < 0:
        raise ValueError("train class counts must be non-negative")
    total = class0_n + class1_n
    if total <= 0:
        raise ValueError("train class counts must have positive total")
    if n < 0:
        raise ValueError("n_validation must be non-negative")
    positive_rate = class1_n / total
    rng = np.random.default_rng(int(seed))
    return rng.choice(np.asarray([0, 1], dtype=int), size=n, p=[1.0 - positive_rate, positive_rate])


def ticker_stratified_random_abstention(
    retained_count_by_ticker: dict[str, int],
    ticker_array: np.ndarray,
    base_seed: int,
    repeat_count: int,
) -> np.ndarray:
    tickers = np.asarray(ticker_array).astype(str)
    repeats = int(repeat_count)
    if repeats <= 0:
        raise ValueError("repeat_count must be positive")
    masks = np.zeros((repeats, len(tickers)), dtype=bool)
    normalized_counts = {str(key): int(value) for key, value in retained_count_by_ticker.items()}

    base_mask = np.zeros(len(tickers), dtype=bool)
    all_counts_are_full = True
    for ticker, retained_count in normalized_counts.items():
        indices = np.flatnonzero(tickers == ticker)
        if retained_count < 0 or retained_count > len(indices):
            raise ValueError(f"Invalid retained count for ticker {ticker}: {retained_count}")
        if retained_count < len(indices):
            all_counts_are_full = False
        if retained_count == len(indices):
            base_mask[indices] = True
    if all_counts_are_full:
        return np.repeat(base_mask[None, :], repeats, axis=0)

    for repeat in range(repeats):
        rng = np.random.default_rng(int(base_seed) + repeat)
        mask = base_mask.copy()
        for ticker, retained_count in normalized_counts.items():
            indices = np.flatnonzero(tickers == ticker)
            if retained_count == len(indices):
                continue
            if retained_count > 0:
                chosen = rng.choice(indices, size=retained_count, replace=False)
                mask[chosen] = True
        masks[repeat] = mask
    return masks


def concentration_metrics(retained_frame: pd.DataFrame, eligible_frame: pd.DataFrame) -> dict:
    retained_n = int(len(retained_frame))
    eligible_n = int(len(eligible_frame))
    result = {
        "eligible_n": eligible_n,
        "retained_n": retained_n,
        "retained_rate": float(retained_n / eligible_n) if eligible_n else np.nan,
        "top_ticker_retained_share": np.nan,
        "ticker_hhi": np.nan,
        "ticker_entropy_normalized": np.nan,
        "retained_ticker_count": 0,
        "top_day_retained_share": np.nan,
        "top_time_bucket_retained_share": np.nan,
    }
    if retained_n == 0:
        return result

    if "ticker" in retained_frame.columns:
        ticker_share = retained_frame["ticker"].astype(str).value_counts(normalize=True)
        result["top_ticker_retained_share"] = float(ticker_share.max())
        result["ticker_hhi"] = float(np.sum(np.square(ticker_share.to_numpy(dtype=float))))
        ticker_count = int(len(ticker_share))
        result["retained_ticker_count"] = ticker_count
        if ticker_count > 1:
            entropy = -float(np.sum(ticker_share * np.log(ticker_share)))
            result["ticker_entropy_normalized"] = float(entropy / math.log(ticker_count))
        else:
            result["ticker_entropy_normalized"] = 0.0

    if "timestamp" in retained_frame.columns:
        timestamp = pd.to_datetime(retained_frame["timestamp"], errors="coerce")
        valid_timestamp = timestamp.dropna()
        if not valid_timestamp.empty:
            day_share = valid_timestamp.dt.date.value_counts(normalize=True)
            result["top_day_retained_share"] = float(day_share.max())
            minute_bucket = valid_timestamp.dt.strftime("%H:%M").value_counts(normalize=True)
            result["top_time_bucket_retained_share"] = float(minute_bucket.max())

    if "confidence" in retained_frame.columns:
        result["mean_confidence"] = float(pd.to_numeric(retained_frame["confidence"], errors="coerce").mean())
    if "y_pred" in retained_frame.columns:
        result["retained_pred1_rate"] = float((retained_frame["y_pred"].astype(int) == 1).mean())
    if "correct" in retained_frame.columns:
        result["retained_accuracy"] = float(pd.to_numeric(retained_frame["correct"], errors="coerce").mean())
    return result


def aggregate_across_seeds(per_seed_metrics: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    if per_seed_metrics.empty:
        return pd.DataFrame()
    group_keys = [
        column
        for column in ("profile_id", "profile_role", "coverage_target")
        if column in per_seed_metrics.columns
    ]
    if not group_keys:
        raise ValueError("per_seed_metrics must include at least one grouping column")

    rows = []
    for keys, group in per_seed_metrics.groupby(group_keys, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        t_critical = T_CRITICAL_ONE_SIDED_95.get(row["seed_count"], 1.833)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_n"] = int(len(values))
            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_lcb95_one_sided"] = np.nan
                row[f"{metric}_positive_seed_count"] = 0
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_lcb95_one_sided"] = float(mean - t_critical * std / math.sqrt(len(values)))
            row[f"{metric}_positive_seed_count"] = int(np.sum(values > 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_decision_outcome(
    per_coverage_aggregated: pd.DataFrame,
    guardrails: pd.DataFrame,
    constants: dict,
) -> dict:
    minimum_delta = float(constants.get("MIN_DECISION_DELTA_MACRO_F1", MIN_DECISION_DELTA_MACRO_F1))
    min_positive_seed_count = int(constants.get("MIN_POSITIVE_SEED_COUNT", MIN_POSITIVE_SEED_COUNT))
    min_positive_coverage_count = int(
        constants.get("MIN_POSITIVE_DECISION_COVERAGE_COUNT", MIN_POSITIVE_DECISION_COVERAGE_COUNT)
    )

    if per_coverage_aggregated.empty:
        return {
            "decision": "not_supported",
            "reason": "no_coverage_rows",
            "positive_decision_coverage_count": 0,
            "failed_guardrail_count": 0,
        }

    delta_col = constants.get("decision_delta_mean_column")
    if not delta_col or delta_col not in per_coverage_aggregated.columns:
        candidates = [
            column
            for column in per_coverage_aggregated.columns
            if "delta_macro_f1" in column and column.endswith("_mean")
        ]
        if not candidates:
            raise ValueError("No decision delta mean column is available")
        delta_col = candidates[0]

    seed_count_col = delta_col.replace("_mean", "_positive_seed_count")
    decision_rows = per_coverage_aggregated.copy()
    if "coverage_target" in decision_rows.columns:
        decision_rows = decision_rows[
            decision_rows["coverage_target"].astype(float).isin(DECISION_COVERAGE_GRID)
        ]

    if seed_count_col in decision_rows.columns:
        enough_seed_support = decision_rows[seed_count_col].fillna(0).astype(int) >= min_positive_seed_count
    else:
        enough_seed_support = pd.Series(False, index=decision_rows.index)
    positive_delta = pd.to_numeric(decision_rows[delta_col], errors="coerce") >= minimum_delta
    positive_count = int((positive_delta & enough_seed_support).sum())

    failed_guardrail_count = 0
    if guardrails is not None and not guardrails.empty:
        if "guardrail_pass" in guardrails.columns:
            failed_guardrail_count += int((~guardrails["guardrail_pass"].map(_is_true)).sum())
        if "failed" in guardrails.columns:
            failed_guardrail_count += int(guardrails["failed"].map(_is_true).sum())

    if failed_guardrail_count >= int(constants.get("NOT_SUPPORTED_FAILURE_COVERAGE_COUNT", NOT_SUPPORTED_FAILURE_COVERAGE_COUNT)):
        return {
            "decision": "not_supported",
            "reason": "guardrail_failures",
            "positive_decision_coverage_count": positive_count,
            "failed_guardrail_count": failed_guardrail_count,
        }
    if positive_count >= min_positive_coverage_count and failed_guardrail_count == 0:
        return {
            "decision": "promote_selective_no_trade_for_validation_only_reporting",
            "reason": "delta_and_seed_support_pass",
            "positive_decision_coverage_count": positive_count,
            "failed_guardrail_count": failed_guardrail_count,
        }
    if positive_count == 0:
        return {
            "decision": "not_supported",
            "reason": "no_positive_decision_coverage",
            "positive_decision_coverage_count": positive_count,
            "failed_guardrail_count": failed_guardrail_count,
        }
    if INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MIN <= positive_count <= INCONCLUSIVE_LOW_POSITIVE_COVERAGE_MAX:
        return {
            "decision": "inconclusive_low_positive_coverage",
            "reason": "too_few_positive_decision_coverages",
            "positive_decision_coverage_count": positive_count,
            "failed_guardrail_count": failed_guardrail_count,
        }
    return {
        "decision": "inconclusive_mixed_seed_or_guardrail_warning",
        "reason": "mixed_support",
        "positive_decision_coverage_count": positive_count,
        "failed_guardrail_count": failed_guardrail_count,
    }
