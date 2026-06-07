"""Generate the Notebook 04 controlled follow-up Colab notebook.

The generated notebook is standalone for Colab. It copies the active Stage 0
data-loading, feature, label, split, scaling, and base model helper code from
notebooks/02_config_screening_colab.ipynb, then adds Notebook 04-only
fresh-seed confirmation, prediction persistence, selective coverage, and manual
gate cells. It does not execute the notebook.
"""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_config_screening_colab.ipynb"
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "04_controlled_followup_colab.ipynb"


TITLE_MD = """\
# Notebook 04 Controlled Follow-Up - Validation Only

Protocol: `docs/CONTROLLED_FOLLOWUP_PROTOCOL_2026-06-04.md`

Scope: `validation_only`

Research question:

```text
Given the official Stage 0 candidate and Notebook 03 selected-branch context,
does a fixed small follow-up panel still show validation-only signal under
fresh seeds, and does any model show useful within-model selective confidence
structure?
```

Official candidate:

```text
candidate_id  = stage0_official
label_config  = h03_bps1p5
feature_set   = price_volume_time
window_size   = 20
```

Fixed model panel:

```text
stratified_dummy
always_up_dummy
logreg
lightgbm
standalone_tcn
ms_dlinear_tcn
```

Fresh seeds:

```text
606, 707, 808, 909, 1010
```

Hard boundaries:

- no project helper package, prior notebook, or archived helper is imported as
  active logic;
- no holdout/test rows are loaded, transformed, windowed, scored, summarized,
  displayed, or used for wording decisions;
- 04S and 04A do not fit real models;
- 04B fits only the fixed panel on the official candidate and writes per-sample
  validation prediction artifacts;
- 04C reads only saved 04B prediction artifacts and evaluates within-model
  selective coverage;
- 04D writes a manual decision matrix and does not authorize any next notebook;
- 04E is optional and reads only saved prediction artifacts.

Run profile:

```text
04S schema smoke: CPU
04A read-only context check: CPU
04B fresh-seed panel: T4 recommended
04C selective coverage: CPU after artifacts exist
04D manual gate decision: CPU after artifacts exist
04E bootstrap CI: optional, CPU after artifacts exist
```

Tabular-only completion is a partial diagnostic, not a completed Notebook 04
gate.
"""


CONFIG_CODE = r"""\
TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
FRESH_SEEDS = (606, 707, 808, 909, 1010)
MODEL_SEEDS = FRESH_SEEDS
RESULT_SCOPE = "validation_only"

INSTALL_LIGHTGBM_IF_MISSING = False
INSTALL_TORCH_IF_MISSING = False

RUN_04S_SCHEMA_SMOKE = False
RUN_04A_READ_CONTEXT = False
RUN_04B_FRESH_SEED_PANEL = False
RUN_04C_SELECTIVE_COVERAGE = False
RUN_04D_GATE_DECISION = False
RUN_04E_BOOTSTRAP_CI = False
BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE = False

DRIVE_PROJECT_FOLDER_ID = "15IZ_sOEyyAKmGCUIOv_u17SwQmFX3nG_"
NOTEBOOK03_DRIVE_RESULTS_FOLDER_ID = "1qQbkwV07X6L_D_WtRYrHDmZ3KXjsju9r"
NOTEBOOK03_DRIVE_RESULTS_FOLDER_NAME = "notebook03_model_family_screening_results"
NOTEBOOK04_DRIVE_BACKUP_FOLDER_NAME = "notebook04_controlled_followup_results"

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 260604
SELECTIVE_COVERAGE_GRID = (1.00, 0.80, 0.60, 0.40, 0.20, 0.10)
MIN_PRACTICAL_DELTA_MACRO_F1 = 0.005

NOTEBOOK04_CANDIDATE = {
    "candidate_id": "stage0_official",
    "label_config": "h03_bps1p5",
    "feature_set": "price_volume_time",
    "window_size": 20,
    "source": "official_stage0_candidate_from_notebook02",
}

BASELINE_MODELS = ("stratified_dummy", "always_up_dummy")
TABULAR_MODELS = ("logreg", "lightgbm")
SEQUENCE_MODELS = ("standalone_tcn", "ms_dlinear_tcn")
REAL_MODELS = TABULAR_MODELS + SEQUENCE_MODELS
MODEL_PANEL = BASELINE_MODELS + REAL_MODELS

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

LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
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

OUTPUT_DIR = Path("/content/notebook04_controlled_followup_results")
PREDICTION_DIR = OUTPUT_DIR / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILES = {
    "context": OUTPUT_DIR / "notebook04_context_checks.json",
    "pooled": OUTPUT_DIR / "notebook04_pooled.csv",
    "per_ticker": OUTPUT_DIR / "notebook04_per_ticker.csv",
    "summary": OUTPUT_DIR / "notebook04_summary.csv",
    "prediction_manifest": OUTPUT_DIR / "notebook04_prediction_manifest.csv",
    "run_manifest": OUTPUT_DIR / "notebook04_run_manifest.json",
    "selective": OUTPUT_DIR / "notebook04_selective_coverage.csv",
    "decision": OUTPUT_DIR / "notebook04_decision_matrix.csv",
    "bootstrap": OUTPUT_DIR / "notebook04_bootstrap_ci.csv",
}

NOTEBOOK03_SELECTION_CANDIDATES = (
    Path("/content/notebook03_model_family_screening_results/notebook03_validation_selection.json"),
    Path("/content/notebook03_validation_selection.json"),
)
NOTEBOOK03_SUMMARY_CANDIDATES = (
    Path("/content/notebook03_model_family_screening_results/notebook03_summary.csv"),
    Path("/content/notebook03_summary.csv"),
)
H0_SUMMARY_CANDIDATES = (
    Path("/content/diagnostic_h0_tabular_sweep/diagnostic_h0_summary.csv"),
    Path("/content/diagnostic_h0_summary.csv"),
)

NOTEBOOK04_STATE = {"prediction_manifest_rows": []}

display(pd.DataFrame([NOTEBOOK04_CANDIDATE]))
print("Notebook 04 output directory:", OUTPUT_DIR)
print("Notebook 03 Drive results folder:", NOTEBOOK03_DRIVE_RESULTS_FOLDER_NAME)
print("Notebook 04 Drive backup folder:", NOTEBOOK04_DRIVE_BACKUP_FOLDER_NAME)
print("Model panel:", MODEL_PANEL)
print("Fresh seeds:", FRESH_SEEDS)
print("Run switches:", {
    "RUN_04S_SCHEMA_SMOKE": RUN_04S_SCHEMA_SMOKE,
    "RUN_04A_READ_CONTEXT": RUN_04A_READ_CONTEXT,
    "RUN_04B_FRESH_SEED_PANEL": RUN_04B_FRESH_SEED_PANEL,
    "RUN_04C_SELECTIVE_COVERAGE": RUN_04C_SELECTIVE_COVERAGE,
    "RUN_04D_GATE_DECISION": RUN_04D_GATE_DECISION,
    "RUN_04E_BOOTSTRAP_CI": RUN_04E_BOOTSTRAP_CI,
    "BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE": BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE,
})
"""


NOTEBOOK04_HELPERS_CODE = r"""\
def assert_sample_alignment(y_flat, y_seq, owner_flat, owner_seq, timestamp_flat, timestamp_seq, split_name):
    if len(y_flat) != len(y_seq):
        raise ValueError(f"{split_name} tabular and sequence labels have different lengths.")
    if not np.array_equal(y_flat, y_seq):
        raise ValueError(f"{split_name} tabular and sequence labels are not aligned.")
    if not np.array_equal(owner_flat, owner_seq):
        raise ValueError(f"{split_name} tabular and sequence ticker owners are not aligned.")
    if not np.array_equal(timestamp_flat, timestamp_seq):
        raise ValueError(f"{split_name} tabular and sequence timestamps are not aligned.")


def make_validation_sample_ids(tickers, timestamps):
    ids = []
    for ticker, timestamp in zip(tickers, timestamps):
        ids.append(f"{ticker}__{pd.Timestamp(timestamp).isoformat()}")
    return np.asarray(ids, dtype=object)


# This get_dataset intentionally overrides the Stage 0 copied helper above.
# It preserves validation timestamps and stable sample ids for prediction
# artifacts and selective-coverage same-row checks.
def get_dataset(label_config, feature_set, window_size):
    key = (label_config, feature_set, int(window_size))
    if key in DATASET_CACHE:
        dataset = DATASET_CACHE[key].copy()
        dataset["prep_seconds"] = 0.0
        return dataset
    if not raw_data:
        raise RuntimeError("raw_data is empty. Enable a Notebook 04 data-loading run switch and rerun setup first.")
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
        raise ValueError(f"No windows available for {label_config} / {feature_set} / window={window_size}")
    assert_sample_alignment(
        y_train,
        y_train_seq,
        train_owner,
        train_owner_seq,
        train_timestamp,
        train_timestamp_seq,
        split_name="train",
    )
    assert_sample_alignment(
        y_validation,
        y_validation_seq,
        validation_owner,
        validation_owner_seq,
        validation_timestamp,
        validation_timestamp_seq,
        split_name="validation",
    )
    validation_sample_id = make_validation_sample_ids(validation_owner, validation_timestamp)
    if len(np.unique(validation_sample_id)) != len(validation_sample_id):
        raise ValueError("Validation sample ids are not unique; inspect ticker/timestamp alignment.")
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
        "train_timestamp": train_timestamp,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "validation_owner": validation_owner,
        "validation_timestamp": validation_timestamp,
        "validation_sample_id": validation_sample_id,
        "x_train_seq": x_train_seq,
        "y_train_seq": y_train_seq,
        "train_owner_seq": train_owner_seq,
        "train_timestamp_seq": train_timestamp_seq,
        "x_validation_seq": x_validation_seq,
        "y_validation_seq": y_validation_seq,
        "validation_owner_seq": validation_owner_seq,
        "validation_timestamp_seq": validation_timestamp_seq,
        "prep_seconds": time.perf_counter() - start,
    }
    DATASET_CACHE[key] = dataset.copy()
    return dataset


def prediction_diagnostics(y_true, y_pred):
    metrics = evaluate_predictions(y_true, y_pred)
    y_pred = np.asarray(y_pred).astype(int)
    metrics["pred_up_pct"] = float((y_pred == 1).mean()) if len(y_pred) else np.nan
    metrics["pred_down_pct"] = float((y_pred == 0).mean()) if len(y_pred) else np.nan
    return metrics


def stratified_dummy_prediction_payload(dataset, seed):
    start = time.perf_counter()
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(dataset["y_train"]), 1)), dataset["y_train"])
    y_pred = dummy.predict(np.zeros((len(dataset["y_validation"]), 1))).astype(int)
    probability = dummy.predict_proba(np.zeros((len(dataset["y_validation"]), 1)))
    class_index = list(dummy.classes_).index(1) if 1 in dummy.classes_ else None
    prob_up = probability[:, class_index] if class_index is not None else np.zeros(len(y_pred), dtype=float)
    return y_pred, prob_up.astype(float), 0.0, time.perf_counter() - start, len(dataset["y_train"]), "baseline_stratified"


def always_up_prediction_payload(dataset, seed):
    start = time.perf_counter()
    y_pred = np.ones(len(dataset["y_validation"]), dtype=int)
    prob_up = np.ones(len(dataset["y_validation"]), dtype=float)
    return y_pred, prob_up, 0.0, time.perf_counter() - start, len(dataset["y_train"]), "baseline_always_up"


def fit_predict_logreg_04(dataset, seed):
    x_train, y_train = subsample_rows_uniformly(dataset["x_train"], dataset["y_train"], MAX_TRAIN_ROWS, seed=seed)
    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=seed,
    )
    start_fit = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start_fit
    start_predict = time.perf_counter()
    y_pred = model.predict(dataset["x_validation"]).astype(int)
    prob_up = model.predict_proba(dataset["x_validation"])[:, 1].astype(float)
    predict_seconds = time.perf_counter() - start_predict
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    max_iter_reached = bool((model.n_iter_ >= 2000).any())
    fit_status = "converged" if not convergence_warnings and not max_iter_reached else "convergence_warning"
    return y_pred, prob_up, fit_seconds, predict_seconds, len(y_train), fit_status


def fit_predict_lightgbm_04(dataset, seed):
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
    y_pred = model.predict(dataset["x_validation"]).astype(int)
    prob_up = model.predict_proba(dataset["x_validation"])[:, 1].astype(float)
    predict_seconds = time.perf_counter() - start_predict
    return y_pred, prob_up, fit_seconds, predict_seconds, len(y_train), "not_applicable"


def make_standalone_tcn(input_dim, seed):
    torch = set_global_seed(seed)
    nn = torch.nn

    class CausalConvBlock(nn.Module):
        def __init__(self, channels_in, channels_out, kernel_size, dropout):
            super().__init__()
            self.padding = kernel_size - 1
            self.conv = nn.Conv1d(channels_in, channels_out, kernel_size, padding=self.padding)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.residual = nn.Conv1d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()

        def forward(self, x):
            conv_out = self.conv(x)
            if self.padding:
                conv_out = conv_out[:, :, :-self.padding]
            return self.dropout(self.activation(conv_out)) + self.residual(x)

    class StandaloneTCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            channels = (input_dim,) + tuple(TORCH_TCN_CHANNELS)
            self.blocks = nn.Sequential(*[
                CausalConvBlock(channels[idx], channels[idx + 1], TORCH_TCN_KERNEL_SIZE, TORCH_DROPOUT)
                for idx in range(len(channels) - 1)
            ])
            self.head = nn.Linear(channels[-1], 2)

        def forward(self, x):
            z = x.transpose(1, 2)
            z = self.blocks(z)
            return self.head(z[:, :, -1])

    return StandaloneTCNClassifier()


def make_sequence_model_04(model_name, input_dim, window_size, seed):
    if model_name == "standalone_tcn":
        return make_standalone_tcn(input_dim, seed)
    if model_name == "ms_dlinear_tcn":
        return make_ms_dlinear_tcn(input_dim, window_size, seed)
    raise ValueError(f"Unsupported Notebook 04 sequence model: {model_name}")


def fit_predict_torch_sequence_04(dataset, seed, model_name):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_sequence_model_04(model_name, input_dim, window_size, seed)
    model.to(device)
    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    counts = np.bincount(y_train.astype(int), minlength=2).astype(float)
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
            if not torch.isfinite(loss):
                raise ValueError(f"{model_name} loss became non-finite.")
            loss.backward()
            optimizer.step()
    fit_seconds = time.perf_counter() - start_fit

    start_predict = time.perf_counter()
    model.eval()
    prob_parts = []
    with torch.no_grad():
        for start in range(0, len(x_validation), TORCH_BATCH_SIZE):
            batch = torch.tensor(x_validation[start : start + TORCH_BATCH_SIZE], dtype=torch.float32, device=device)
            logits = model(batch)
            prob_parts.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    prob_up = np.concatenate(prob_parts).astype(float)
    y_pred = (prob_up >= 0.5).astype(int)
    predict_seconds = time.perf_counter() - start_predict
    return y_pred, prob_up, fit_seconds, predict_seconds, len(y_train), f"fixed_epochs_device_{device}"


def fit_predict_model_04(dataset, model_name, seed):
    if model_name == "stratified_dummy":
        return stratified_dummy_prediction_payload(dataset, seed)
    if model_name == "always_up_dummy":
        return always_up_prediction_payload(dataset, seed)
    if model_name == "logreg":
        return fit_predict_logreg_04(dataset, seed)
    if model_name == "lightgbm":
        return fit_predict_lightgbm_04(dataset, seed)
    if model_name in SEQUENCE_MODELS:
        return fit_predict_torch_sequence_04(dataset, seed, model_name)
    raise ValueError(f"Unknown Notebook 04 model: {model_name}")


def run_notebook04_shape_smoke():
    synthetic_sample_id = np.asarray(["CSCO__synthetic_0", "JPM__synthetic_1"], dtype=object)
    synthetic_ticker = np.asarray(["CSCO", "JPM"], dtype=object)
    synthetic_timestamp = np.asarray(["2026-06-04T09:30:00", "2026-06-04T09:35:00"], dtype=object)
    synthetic_y_true = np.asarray([0, 1], dtype=int)
    synthetic_y_pred = np.asarray([0, 1], dtype=int)
    synthetic_prob_up = np.asarray([0.25, 0.75], dtype=float)
    artifact = {
        "validation_sample_id": synthetic_sample_id,
        "ticker": synthetic_ticker,
        "timestamp": synthetic_timestamp,
        "y_true": synthetic_y_true,
        "y_pred": synthetic_y_pred,
        "prob_up": synthetic_prob_up,
        "confidence": np.maximum(synthetic_prob_up, 1.0 - synthetic_prob_up),
    }
    validate_prediction_payload(artifact)
    torch = ensure_torch()
    for model_name in SEQUENCE_MODELS:
        model = make_sequence_model_04(model_name, input_dim=3, window_size=20, seed=606)
        model.eval()
        x = torch.zeros((2, 20, 3), dtype=torch.float32)
        with torch.no_grad():
            logits = model(x)
        if tuple(logits.shape) != (2, 2):
            raise ValueError(f"{model_name} smoke output shape mismatch: {tuple(logits.shape)}")
    print("Notebook 04 schema and torch shape smoke passed.")


def validate_prediction_payload(payload):
    required = ("validation_sample_id", "ticker", "timestamp", "y_true", "y_pred", "prob_up", "confidence")
    lengths = []
    for name in required:
        if name not in payload:
            raise ValueError(f"Prediction payload missing array: {name}")
        lengths.append(len(payload[name]))
    if len(set(lengths)) != 1:
        raise ValueError(f"Prediction payload lengths disagree: {dict(zip(required, lengths))}")
    prob_up = np.asarray(payload["prob_up"], dtype=float)
    confidence = np.asarray(payload["confidence"], dtype=float)
    if np.isnan(prob_up).any() or np.isnan(confidence).any():
        raise ValueError("Prediction payload contains NaN probability or confidence values.")
    if ((prob_up < 0.0) | (prob_up > 1.0)).any():
        raise ValueError("prob_up values must be within [0.0, 1.0].")
    if ((confidence < 0.5) | (confidence > 1.0)).any():
        raise ValueError("confidence values must be within [0.5, 1.0].")


def save_prediction_artifact(dataset, model_name, seed, y_pred, prob_up):
    prob_up = np.asarray(prob_up, dtype=float)
    confidence = np.maximum(prob_up, 1.0 - prob_up)
    payload = {
        "validation_sample_id": dataset["validation_sample_id"],
        "ticker": dataset["validation_owner"].astype(object),
        "timestamp": dataset["validation_timestamp"].astype("datetime64[ns]").astype(str).astype(object),
        "y_true": dataset["y_validation"].astype(int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "prob_up": prob_up,
        "confidence": confidence,
    }
    validate_prediction_payload(payload)
    artifact_path = PREDICTION_DIR / f"{model_name}__seed{int(seed)}.npz"
    np.savez_compressed(artifact_path, **payload)
    row = {
        "model": model_name,
        "seed": int(seed),
        "path": str(artifact_path),
        "row_count": int(len(payload["y_true"])),
        "selective_eligible": bool(model_name in REAL_MODELS),
        "prob_up_source": "predict_proba[:, 1]" if model_name in TABULAR_MODELS else (
            "softmax(logits)[:, 1]" if model_name in SEQUENCE_MODELS else "baseline_probability"
        ),
        "scope": RESULT_SCOPE,
    }
    NOTEBOOK04_STATE["prediction_manifest_rows"].append(row)
    return artifact_path, row


def stratified_dummy_predictions_same_rows(y_train, retained_n, seed):
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return dummy.predict(np.zeros((retained_n, 1))).astype(int)


def concentration_from_per_ticker(per_ticker_rows):
    deltas = [row["delta_macro_f1_vs_stratified_dummy"] for row in per_ticker_rows]
    positive = [float(delta) for delta in deltas if pd.notna(delta) and delta > 0]
    positive_ticker_count = int(len(positive))
    top_ticker_gain_share = float(max(positive) / sum(positive)) if positive else 0.0
    return positive_ticker_count, top_ticker_gain_share


def row_metrics(y_true, y_pred, stratified_pred, always_up_pred):
    metrics = prediction_diagnostics(y_true, y_pred)
    stratified_metrics = evaluate_predictions(y_true, stratified_pred)
    always_up_metrics = evaluate_predictions(y_true, always_up_pred)
    metrics.update({
        "stratified_dummy_macro_f1": stratified_metrics["macro_f1"],
        "stratified_dummy_balanced_accuracy": stratified_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_stratified_dummy": metrics["macro_f1"] - stratified_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_stratified_dummy": metrics["balanced_accuracy"] - stratified_metrics["balanced_accuracy"],
        "always_up_dummy_macro_f1": always_up_metrics["macro_f1"],
        "always_up_dummy_balanced_accuracy": always_up_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_always_up_dummy": metrics["macro_f1"] - always_up_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_always_up_dummy": metrics["balanced_accuracy"] - always_up_metrics["balanced_accuracy"],
    })
    return metrics


def run_one_notebook04_seed(model_name, seed):
    dataset = get_dataset(
        NOTEBOOK04_CANDIDATE["label_config"],
        NOTEBOOK04_CANDIDATE["feature_set"],
        NOTEBOOK04_CANDIDATE["window_size"],
    )
    prep_seconds = float(dataset.get("prep_seconds", 0.0))
    try:
        y_pred, prob_up, fit_seconds, predict_seconds, train_n, fit_status = fit_predict_model_04(dataset, model_name, seed)
        artifact_path, _ = save_prediction_artifact(dataset, model_name, seed, y_pred, prob_up)
        stratified_pred, _, _, _, _, _ = stratified_dummy_prediction_payload(dataset, seed)
        always_up_pred, _, _, _, _, _ = always_up_prediction_payload(dataset, seed)
        pooled_metrics = row_metrics(dataset["y_validation"], y_pred, stratified_pred, always_up_pred)
        per_ticker_rows = []
        for ticker in TICKERS:
            val_mask = dataset["validation_owner"] == ticker
            if not val_mask.any():
                continue
            ticker_metrics = row_metrics(
                dataset["y_validation"][val_mask],
                np.asarray(y_pred)[val_mask],
                stratified_pred[val_mask],
                always_up_pred[val_mask],
            )
            per_ticker_rows.append({
                "stage": "04B_fresh_seed_panel",
                "candidate_id": NOTEBOOK04_CANDIDATE["candidate_id"],
                "model": model_name,
                "seed": int(seed),
                "label_config": dataset["label_config"],
                "horizon_k": dataset["horizon_k"],
                "threshold_bps": dataset["threshold_bps"],
                "feature_set": dataset["feature_set"],
                "window_size": int(dataset["window_size"]),
                "scope": RESULT_SCOPE,
                "ticker_or_pooled": ticker,
                "n": int(val_mask.sum()),
                "train_n": int((dataset["train_owner"] == ticker).sum()),
                "validation_n": int(val_mask.sum()),
                "run_failed": False,
                "failure_reason": "",
                "fit_status": fit_status,
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "prep_seconds": prep_seconds,
                "total_seconds": prep_seconds + float(fit_seconds) + float(predict_seconds),
                "prediction_artifact": str(artifact_path),
                **ticker_metrics,
            })
        positive_ticker_count, top_ticker_gain_share = concentration_from_per_ticker(per_ticker_rows)
        for row in per_ticker_rows:
            row["positive_ticker_count"] = positive_ticker_count
            row["top_ticker_gain_share"] = top_ticker_gain_share
        pooled_row = {
            "stage": "04B_fresh_seed_panel",
            "candidate_id": NOTEBOOK04_CANDIDATE["candidate_id"],
            "model": model_name,
            "seed": int(seed),
            "label_config": dataset["label_config"],
            "horizon_k": dataset["horizon_k"],
            "threshold_bps": dataset["threshold_bps"],
            "feature_set": dataset["feature_set"],
            "window_size": int(dataset["window_size"]),
            "scope": RESULT_SCOPE,
            "ticker_or_pooled": "pooled",
            "n": int(len(dataset["y_validation"])),
            "train_n": int(train_n),
            "validation_n": int(len(dataset["y_validation"])),
            "run_failed": False,
            "failure_reason": "",
            "fit_status": fit_status,
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "prep_seconds": prep_seconds,
            "total_seconds": prep_seconds + float(fit_seconds) + float(predict_seconds),
            "prediction_artifact": str(artifact_path),
            "positive_ticker_count": positive_ticker_count,
            "top_ticker_gain_share": top_ticker_gain_share,
            **pooled_metrics,
        }
    except Exception as exc:
        pooled_row = {
            "stage": "04B_fresh_seed_panel",
            "candidate_id": NOTEBOOK04_CANDIDATE["candidate_id"],
            "model": model_name,
            "seed": int(seed),
            "label_config": NOTEBOOK04_CANDIDATE["label_config"],
            "horizon_k": LABEL_CONFIGS[NOTEBOOK04_CANDIDATE["label_config"]]["horizon_k"],
            "threshold_bps": LABEL_CONFIGS[NOTEBOOK04_CANDIDATE["label_config"]]["threshold_bps"],
            "feature_set": NOTEBOOK04_CANDIDATE["feature_set"],
            "window_size": int(NOTEBOOK04_CANDIDATE["window_size"]),
            "scope": RESULT_SCOPE,
            "ticker_or_pooled": "pooled",
            "n": 0,
            "train_n": 0,
            "validation_n": 0,
            "run_failed": True,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "fit_status": "run_failed",
        }
        for column in (
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "stratified_dummy_macro_f1",
            "stratified_dummy_balanced_accuracy",
            "delta_macro_f1_vs_stratified_dummy",
            "delta_balanced_accuracy_vs_stratified_dummy",
            "always_up_dummy_macro_f1",
            "always_up_dummy_balanced_accuracy",
            "delta_macro_f1_vs_always_up_dummy",
            "delta_balanced_accuracy_vs_always_up_dummy",
            "pred_up_pct",
            "pred_down_pct",
            "fit_seconds",
            "predict_seconds",
            "prep_seconds",
            "total_seconds",
            "positive_ticker_count",
            "top_ticker_gain_share",
        ):
            pooled_row[column] = np.nan
        per_ticker_rows = []
    return pooled_row, per_ticker_rows


def summarize_notebook04(pooled):
    if pooled.empty:
        return pd.DataFrame()
    rows = []
    keys = ["candidate_id", "model", "label_config", "horizon_k", "threshold_bps", "feature_set", "window_size", "scope"]
    notebook03_summary = read_optional_notebook03_summary()
    for key_values, group in pooled.groupby(keys, sort=False):
        record = dict(zip(keys, key_values))
        successful = group.loc[~group["run_failed"].astype(bool)].copy()
        seed_count = int(successful["seed"].nunique())
        n_failed = int(group["run_failed"].astype(bool).sum())
        record["seed_count"] = seed_count
        record["n_failed_seeds"] = n_failed
        record["run_failed"] = bool(n_failed > 0)
        record["failure_reason"] = "; ".join(sorted(set(group.loc[group["run_failed"].astype(bool), "failure_reason"].dropna().astype(str))))
        if successful.empty:
            for column in (
                "macro_f1_mean",
                "macro_f1_std",
                "macro_f1_lcb_95",
                "balanced_accuracy_mean",
                "stratified_dummy_macro_f1_mean",
                "delta_macro_f1_vs_stratified_dummy_mean",
                "delta_balanced_accuracy_vs_stratified_dummy_mean",
                "always_up_dummy_macro_f1_mean",
                "delta_macro_f1_vs_always_up_dummy_mean",
                "positive_ticker_count",
                "top_ticker_gain_share",
                "fresh_minus_03",
            ):
                record[column] = np.nan
            record["basic_gate_pass"] = False
            record["fresh_seed_stability_tag"] = "run_failed"
        else:
            macro_std = sample_std(successful["macro_f1"])
            macro_mean = float(successful["macro_f1"].mean())
            record.update({
                "macro_f1_mean": macro_mean,
                "macro_f1_std": macro_std,
                "macro_f1_lcb_95": float(
                    macro_mean - t_critical_one_sided_95(seed_count) * macro_std / math.sqrt(max(seed_count, 1))
                ),
                "balanced_accuracy_mean": float(successful["balanced_accuracy"].mean()),
                "stratified_dummy_macro_f1_mean": float(successful["stratified_dummy_macro_f1"].mean()),
                "delta_macro_f1_vs_stratified_dummy_mean": float(successful["delta_macro_f1_vs_stratified_dummy"].mean()),
                "delta_balanced_accuracy_vs_stratified_dummy_mean": float(successful["delta_balanced_accuracy_vs_stratified_dummy"].mean()),
                "always_up_dummy_macro_f1_mean": float(successful["always_up_dummy_macro_f1"].mean()),
                "delta_macro_f1_vs_always_up_dummy_mean": float(successful["delta_macro_f1_vs_always_up_dummy"].mean()),
                "n_mean": float(successful["n"].mean()),
                "positive_ticker_count": int(round(successful["positive_ticker_count"].mean())),
                "top_ticker_gain_share": float(successful["top_ticker_gain_share"].mean()),
            })
            record["basic_gate_pass"] = bool(
                record["model"] in REAL_MODELS
                and n_failed == 0
                and record["macro_f1_lcb_95"] > record["stratified_dummy_macro_f1_mean"]
                and record["delta_macro_f1_vs_stratified_dummy_mean"] > 0
                and record["positive_ticker_count"] >= 3
                and record["top_ticker_gain_share"] <= 0.50
            )
            notebook03_macro = lookup_notebook03_macro(notebook03_summary, record["model"])
            record["notebook03_macro_f1_mean"] = notebook03_macro
            record["fresh_minus_03"] = macro_mean - notebook03_macro if pd.notna(notebook03_macro) else np.nan
            if pd.isna(record["fresh_minus_03"]):
                record["fresh_seed_stability_tag"] = "notebook03_reference_missing"
            elif record["fresh_minus_03"] >= -0.001:
                record["fresh_seed_stability_tag"] = "confirmed_or_improved"
            elif -0.003 < record["fresh_minus_03"] < -0.001:
                record["fresh_seed_stability_tag"] = "marginal_drop_note_only"
            else:
                record["fresh_seed_stability_tag"] = "failed_fresh_seed_confirmation"
            record["positive_shift_review"] = bool(pd.notna(record["fresh_minus_03"]) and record["fresh_minus_03"] > 0.003)
        rows.append(record)
    return pd.DataFrame(rows)


def drive_query_literal(value):
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def find_latest_drive_file_by_suffix(service, folder_id, filename_suffix):
    escaped_parent = drive_query_literal(folder_id)
    query = f"{escaped_parent} in parents and trashed = false"
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name,mimeType,createdTime,modifiedTime)",
        pageSize=100,
    ).execute()
    files = [
        item
        for item in response.get("files", [])
        if str(item.get("name", "")).endswith(filename_suffix)
    ]
    if not files:
        raise FileNotFoundError(
            f"No Drive file ending with {filename_suffix!r} found in folder "
            f"{NOTEBOOK03_DRIVE_RESULTS_FOLDER_NAME} ({folder_id})."
        )
    return sorted(files, key=lambda item: str(item.get("name", "")), reverse=True)[0]


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


def ensure_latest_notebook03_context_from_drive():
    selection_target = NOTEBOOK03_SELECTION_CANDIDATES[0]
    summary_target = NOTEBOOK03_SUMMARY_CANDIDATES[0]
    if selection_target.exists() and summary_target.exists():
        return {
            "selection": str(selection_target),
            "summary": str(summary_target),
            "source": "local_existing",
        }

    service = build_drive_service()
    downloaded = {}
    if not selection_target.exists():
        selection_file = find_latest_drive_file_by_suffix(
            service,
            NOTEBOOK03_DRIVE_RESULTS_FOLDER_ID,
            "notebook03_validation_selection.json",
        )
        download_drive_file(service, selection_file["id"], selection_target)
        downloaded["selection"] = {
            "drive_name": selection_file["name"],
            "local_path": str(selection_target),
        }
        print("Downloaded Notebook 03 selection:", selection_file["name"], "->", selection_target)

    if not summary_target.exists():
        summary_file = find_latest_drive_file_by_suffix(
            service,
            NOTEBOOK03_DRIVE_RESULTS_FOLDER_ID,
            "notebook03_summary.csv",
        )
        download_drive_file(service, summary_file["id"], summary_target)
        downloaded["summary"] = {
            "drive_name": summary_file["name"],
            "local_path": str(summary_target),
        }
        print("Downloaded Notebook 03 summary:", summary_file["name"], "->", summary_target)

    return downloaded


def read_optional_notebook03_summary():
    if not any(path.exists() for path in NOTEBOOK03_SUMMARY_CANDIDATES):
        try:
            ensure_latest_notebook03_context_from_drive()
        except Exception as exc:
            print("Notebook 03 summary auto-download skipped:", type(exc).__name__, exc)
    for path in NOTEBOOK03_SUMMARY_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def lookup_notebook03_macro(summary, model_name):
    if summary.empty or "model" not in summary.columns or "macro_f1_mean" not in summary.columns:
        return np.nan
    rows = summary.loc[summary["model"].eq(model_name)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["macro_f1_mean"])


def write_run_manifest(pooled, per_ticker, summary, prediction_manifest):
    manifest = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "candidate": NOTEBOOK04_CANDIDATE,
        "model_panel": list(MODEL_PANEL),
        "fresh_seeds": list(FRESH_SEEDS),
        "row_counts": {
            "pooled": int(len(pooled)),
            "per_ticker": int(len(per_ticker)),
            "summary": int(len(summary)),
            "prediction_manifest": int(len(prediction_manifest)),
        },
        "holdout_test_authorized": False,
        "run_switches": {
            "RUN_04S_SCHEMA_SMOKE": RUN_04S_SCHEMA_SMOKE,
            "RUN_04A_READ_CONTEXT": RUN_04A_READ_CONTEXT,
            "RUN_04B_FRESH_SEED_PANEL": RUN_04B_FRESH_SEED_PANEL,
            "RUN_04C_SELECTIVE_COVERAGE": RUN_04C_SELECTIVE_COVERAGE,
            "RUN_04D_GATE_DECISION": RUN_04D_GATE_DECISION,
            "RUN_04E_BOOTSTRAP_CI": RUN_04E_BOOTSTRAP_CI,
        },
    }
    with OUTPUT_FILES["run_manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def find_or_create_drive_folder(service, folder_name, parent_id):
    escaped_name = drive_query_literal(folder_name)
    escaped_parent = drive_query_literal(parent_id)
    query = (
        f"name = {escaped_name} and {escaped_parent} in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name,webViewLink)",
        pageSize=10,
    ).execute()
    folders = response.get("files", [])
    if folders:
        return folders[0]
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(body=metadata, fields="id,name,webViewLink").execute()


def upload_local_file_to_drive(service, local_path, folder_id, uploaded_name):
    import mimetypes
    from googleapiclient.http import MediaFileUpload

    local_path = Path(local_path)
    mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    metadata = {"name": uploaded_name, "parents": [folder_id]}
    return service.files().create(
        body=metadata,
        media_body=media,
        fields="id,name,webViewLink",
    ).execute()


def notebook04_existing_output_paths(include_predictions=False, prediction_model_name=None):
    paths = [path for path in OUTPUT_FILES.values() if path.exists()]
    if include_predictions:
        if prediction_model_name:
            paths.extend(sorted(PREDICTION_DIR.glob(f"{prediction_model_name}__seed*.npz")))
        else:
            paths.extend(sorted(PREDICTION_DIR.glob("*.npz")))
    return paths


def backup_notebook04_outputs(reason, include_predictions=False, prediction_model_name=None):
    if not BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE:
        return []
    paths = notebook04_existing_output_paths(
        include_predictions=include_predictions,
        prediction_model_name=prediction_model_name,
    )
    if not paths:
        print("Notebook 04 backup skipped; no local output files exist yet.")
        return []

    service = build_drive_service()
    backup_folder = find_or_create_drive_folder(
        service,
        NOTEBOOK04_DRIVE_BACKUP_FOLDER_NAME,
        DRIVE_PROJECT_FOLDER_ID,
    )
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uploaded = []
    for path in paths:
        uploaded_name = f"{timestamp}__{reason}__{path.name}"
        drive_file = upload_local_file_to_drive(service, path, backup_folder["id"], uploaded_name)
        uploaded.append({
            "local_path": str(path),
            "uploaded_name": uploaded_name,
            "drive_file": drive_file,
        })
        print("Uploaded Notebook 04 artifact:", uploaded_name)

    manifest = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "reason": reason,
        "include_predictions": bool(include_predictions),
        "prediction_model_name": prediction_model_name,
        "backup_folder": backup_folder,
        "uploaded": uploaded,
        "holdout_test_authorized": False,
    }
    manifest_path = OUTPUT_DIR / f"{timestamp}__{reason}__notebook04_drive_backup_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    manifest_file = upload_local_file_to_drive(
        service,
        manifest_path,
        backup_folder["id"],
        manifest_path.name,
    )
    uploaded.append({
        "local_path": str(manifest_path),
        "uploaded_name": manifest_path.name,
        "drive_file": manifest_file,
    })
    print("Uploaded Notebook 04 backup manifest:", manifest_path.name)
    return uploaded


def read_first_existing_json(candidates, description):
    missing = []
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return path, json.load(handle)
        missing.append(str(path))
    raise FileNotFoundError(f"Missing {description}. Checked: {'; '.join(missing)}")


def run_context_check_04a():
    drive_context_download = ensure_latest_notebook03_context_from_drive()
    selection_path, selection = read_first_existing_json(NOTEBOOK03_SELECTION_CANDIDATES, "Notebook 03 selection JSON")
    if selection.get("scope") != RESULT_SCOPE:
        raise ValueError(f"Notebook 03 selection scope must be {RESULT_SCOPE}; found {selection.get('scope')!r}.")
    if selection.get("holdout_test_authorized") is not False:
        raise ValueError("Notebook 03 selection must have holdout_test_authorized set to false.")
    candidate_ok = (
        NOTEBOOK04_CANDIDATE["label_config"] == "h03_bps1p5"
        and NOTEBOOK04_CANDIDATE["feature_set"] == "price_volume_time"
        and int(NOTEBOOK04_CANDIDATE["window_size"]) == 20
    )
    if not candidate_ok:
        raise ValueError(f"Notebook 04 official candidate drifted: {NOTEBOOK04_CANDIDATE}")
    h0_status = "not_found"
    for path in H0_SUMMARY_CANDIDATES:
        if path.exists():
            h0 = pd.read_csv(path)
            if "scope" in h0.columns and not h0["scope"].astype(str).str.contains("diagnostic", case=False).all():
                raise ValueError(f"H0 file has non-diagnostic scope values: {path}")
            if "confirmation_status" in h0.columns:
                selected_like = h0["confirmation_status"].astype(str).str.contains("selected", case=False, na=False)
                if selected_like.any() and not h0["confirmation_status"].astype(str).str.contains("not_selected", case=False, na=False).all():
                    raise ValueError(f"H0 file appears to contain selecting confirmation status: {path}")
            h0_status = f"diagnostic_read_only:{path}"
            break
    context = {
        "scope": RESULT_SCOPE,
        "notebook03_selection_json": str(selection_path),
        "notebook03_drive_context_download": drive_context_download,
        "official_candidate": NOTEBOOK04_CANDIDATE,
        "h0_status": h0_status,
        "model_panel": list(MODEL_PANEL),
        "fresh_seeds": list(FRESH_SEEDS),
        "holdout_test_authorized": False,
        "checks_passed": True,
    }
    with OUTPUT_FILES["context"].open("w", encoding="utf-8") as handle:
        json.dump(context, handle, indent=2)
    return context


def write_notebook04_panel_outputs(pooled_rows, per_ticker_rows):
    pooled = pd.DataFrame(pooled_rows)
    per_ticker = pd.DataFrame(per_ticker_rows)
    summary = summarize_notebook04(pooled)
    prediction_manifest = pd.DataFrame(NOTEBOOK04_STATE["prediction_manifest_rows"])
    pooled.to_csv(OUTPUT_FILES["pooled"], index=False)
    per_ticker.to_csv(OUTPUT_FILES["per_ticker"], index=False)
    summary.to_csv(OUTPUT_FILES["summary"], index=False)
    prediction_manifest.to_csv(OUTPUT_FILES["prediction_manifest"], index=False)
    write_run_manifest(pooled, per_ticker, summary, prediction_manifest)
    return pooled, per_ticker, summary, prediction_manifest


def run_fresh_seed_panel_04b():
    pooled_rows = []
    per_ticker_rows = []
    NOTEBOOK04_STATE["prediction_manifest_rows"] = []
    pooled = per_ticker = summary = prediction_manifest = None
    for model_name in MODEL_PANEL:
        for seed in FRESH_SEEDS:
            print("04B", model_name, "seed", seed)
            pooled_row, ticker_rows = run_one_notebook04_seed(model_name, seed)
            pooled_rows.append(pooled_row)
            per_ticker_rows.extend(ticker_rows)
        pooled, per_ticker, summary, prediction_manifest = write_notebook04_panel_outputs(
            pooled_rows,
            per_ticker_rows,
        )
        backup_notebook04_outputs(
            f"checkpoint_04B_after_{model_name}",
            include_predictions=True,
            prediction_model_name=model_name,
        )
    backup_notebook04_outputs("completed_04B_fresh_seed_panel")
    return pooled, per_ticker, summary, prediction_manifest


def load_prediction_npz(path):
    with np.load(path, allow_pickle=True) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    validate_prediction_payload(payload)
    return payload


def load_prediction_artifact(model_name, seed):
    manifest = pd.read_csv(OUTPUT_FILES["prediction_manifest"])
    rows = manifest.loc[manifest["model"].eq(model_name) & manifest["seed"].astype(int).eq(int(seed))]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one prediction artifact for {model_name} seed {seed}; found {len(rows)}")
    return load_prediction_npz(rows.iloc[0]["path"])


def precision_for_label(y_true, y_pred, label):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pred_mask = y_pred == int(label)
    if not pred_mask.any():
        return np.nan
    return float((y_true[pred_mask] == int(label)).mean())


def selective_rows_for_artifact(model_name, seed, model_payload, stratified_payload, always_up_payload):
    if not np.array_equal(model_payload["validation_sample_id"], stratified_payload["validation_sample_id"]):
        raise ValueError(f"Stratified dummy sample ids do not match {model_name} seed {seed}.")
    if not np.array_equal(model_payload["validation_sample_id"], always_up_payload["validation_sample_id"]):
        raise ValueError(f"Always-up dummy sample ids do not match {model_name} seed {seed}.")
    order = np.lexsort((model_payload["validation_sample_id"].astype(str), -model_payload["confidence"].astype(float)))
    rows = []
    validation_n = len(order)
    for coverage in SELECTIVE_COVERAGE_GRID:
        retained_n = max(1, int(math.ceil(validation_n * float(coverage))))
        idx = order[:retained_n]
        y_true = model_payload["y_true"][idx].astype(int)
        y_pred = model_payload["y_pred"][idx].astype(int)
        stratified_pred = stratified_payload["y_pred"][idx].astype(int)
        always_up_pred = always_up_payload["y_pred"][idx].astype(int)
        metrics = row_metrics(y_true, y_pred, stratified_pred, always_up_pred)
        retained_ticker = model_payload["ticker"][idx].astype(str)
        ticker_counts = pd.Series(retained_ticker).value_counts()
        class_values = pd.Series(y_true).value_counts()
        pred_values = pd.Series(y_pred).value_counts()
        warnings_list = []
        if coverage <= 0.20:
            warnings_list.append("low_coverage_exploratory")
        if int(ticker_counts.min()) < 500:
            warnings_list.append("per_ticker_retained_n_low")
        if float(ticker_counts.max() / retained_n) > 0.40:
            warnings_list.append("ticker_concentration_warning")
        auc = np.nan
        if len(class_values) == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(y_true, model_payload["prob_up"][idx].astype(float)))
            except ValueError:
                auc = np.nan
        else:
            warnings_list.append("auc_not_defined")
        rows.append({
            "model": model_name,
            "seed": int(seed),
            "coverage": float(coverage),
            "retained_n": int(retained_n),
            "retained_pct": float(retained_n / validation_n),
            "class0_n": int(class_values.get(0, 0)),
            "class1_n": int(class_values.get(1, 0)),
            "pred0_n": int(pred_values.get(0, 0)),
            "pred1_n": int(pred_values.get(1, 0)),
            "selective_error": float(1.0 - metrics["accuracy"]),
            "precision_down": precision_for_label(y_true, y_pred, 0),
            "precision_up": precision_for_label(y_true, y_pred, 1),
            "auc": auc,
            "delta_macro_f1_vs_stratified_dummy_same_rows": metrics["delta_macro_f1_vs_stratified_dummy"],
            "delta_macro_f1_vs_always_up_dummy_same_rows": metrics["delta_macro_f1_vs_always_up_dummy"],
            "max_ticker_retained_share": float(ticker_counts.max() / retained_n),
            "min_ticker_retained_n": int(ticker_counts.min()),
            "warnings": "|".join(warnings_list),
            "scope": RESULT_SCOPE,
            **{k: metrics[k] for k in ("macro_f1", "balanced_accuracy", "accuracy")},
        })
    return rows


def run_selective_coverage_04c():
    if not OUTPUT_FILES["prediction_manifest"].exists():
        raise FileNotFoundError(f"Prediction manifest missing: {OUTPUT_FILES['prediction_manifest']}")
    rows = []
    for model_name in REAL_MODELS:
        for seed in FRESH_SEEDS:
            model_payload = load_prediction_artifact(model_name, seed)
            stratified_payload = load_prediction_artifact("stratified_dummy", seed)
            always_up_payload = load_prediction_artifact("always_up_dummy", seed)
            rows.extend(selective_rows_for_artifact(model_name, seed, model_payload, stratified_payload, always_up_payload))
    selective = pd.DataFrame(rows)
    selective.to_csv(OUTPUT_FILES["selective"], index=False)
    return selective


def run_gate_decision_04d():
    for name in ("context", "summary", "selective"):
        if not OUTPUT_FILES[name].exists():
            raise FileNotFoundError(f"04D requires {name} artifact: {OUTPUT_FILES[name]}")
    summary = pd.read_csv(OUTPUT_FILES["summary"])
    selective = pd.read_csv(OUTPUT_FILES["selective"])
    rows = []
    for exit_name in ("Exit A - Proceed To 05 LightGBM Tuning", "Exit B - Proceed To 05 MS-DLinear+TCN Design Review", "Exit C - Stop Modeling And Write Weak-Signal Result", "Exit D - Inconclusive, Pre-Register One New Diagnostic"):
        rows.append({
            "exit": exit_name,
            "operator_selected": False,
            "manual_operator_decision_required": True,
            "holdout_test_authorized": False,
            "notes": "Read Notebook 04 summary and selective coverage before manually selecting exactly one exit.",
        })
    lightgbm = summary.loc[summary["model"].eq("lightgbm")]
    msd = summary.loc[summary["model"].eq("ms_dlinear_tcn")]
    tcn = summary.loc[summary["model"].eq("standalone_tcn")]
    checks = {
        "lightgbm_basic_gate": bool((not lightgbm.empty) and bool(lightgbm.iloc[0].get("basic_gate_pass", False))),
        "ms_dlinear_tcn_basic_gate": bool((not msd.empty) and bool(msd.iloc[0].get("basic_gate_pass", False))),
        "ms_dlinear_tcn_not_failed_fresh_seed": bool((not msd.empty) and msd.iloc[0].get("fresh_seed_stability_tag") != "failed_fresh_seed_confirmation"),
        "standalone_tcn_reference_available": bool(not tcn.empty),
        "selective_rows_available": bool(not selective.empty),
        "pre04_design_review_source_required_for_exit_b": True,
    }
    decision = pd.DataFrame(rows)
    for key, value in checks.items():
        decision[key] = value
    decision.to_csv(OUTPUT_FILES["decision"], index=False)
    return decision


def bootstrap_macro_f1(y_true, y_pred, seed):
    rng = np.random.default_rng(seed)
    values = []
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        values.append(f1_score(y_true[idx], y_pred[idx], labels=[0, 1], average="macro", zero_division=0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_bootstrap_ci_04e():
    rows = []
    for model_name in REAL_MODELS:
        for seed in FRESH_SEEDS:
            payload = load_prediction_artifact(model_name, seed)
            lower, upper = bootstrap_macro_f1(payload["y_true"], payload["y_pred"], BOOTSTRAP_SEED + int(seed))
            rows.append({
                "model": model_name,
                "seed": int(seed),
                "macro_f1_bootstrap_ci_lower": lower,
                "macro_f1_bootstrap_ci_upper": upper,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED + int(seed),
                "scope": RESULT_SCOPE,
                "interpretation": "diagnostic_only_row_level_bootstrap",
            })
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(OUTPUT_FILES["bootstrap"], index=False)
    return bootstrap
"""


STAGE04S_MD = """\
## 04S - Schema Smoke

04S uses tiny synthetic arrays to check the prediction artifact schema,
selective-coverage helper shape, and torch model forward shapes. It does not
load real validation results, fit real data, or write selection decisions.
"""


STAGE04S_CODE = r"""\
if RUN_04S_SCHEMA_SMOKE:
    run_notebook04_shape_smoke()
else:
    print("RUN_04S_SCHEMA_SMOKE is False; schema smoke not run.")
"""


STAGE04A_MD = """\
## 04A - Read-Only Context Check

04A reads Notebook 03 selection context and optional diagnostic H0 outputs. It
does not fit models.
"""


STAGE04A_CODE = r"""\
if RUN_04A_READ_CONTEXT:
    context = run_context_check_04a()
    backup_notebook04_outputs("completed_04A_context_check")
    display(pd.DataFrame([context]))
else:
    print("RUN_04A_READ_CONTEXT is False; context check not run.")
"""


STAGE04B_MD = """\
## 04B - Fresh-Seed Confirmation Panel

04B fits the fixed model panel on the fixed official candidate using fresh
seeds. It writes pooled, per-ticker, summary, run-manifest, prediction-manifest,
and per-sample prediction artifacts.
"""


STAGE04B_CODE = r"""\
if RUN_04B_FRESH_SEED_PANEL:
    pooled, per_ticker, summary, prediction_manifest = run_fresh_seed_panel_04b()
    display(summary.sort_values(["basic_gate_pass", "macro_f1_mean"], ascending=[False, False]))
    display(prediction_manifest)
else:
    print("RUN_04B_FRESH_SEED_PANEL is False; fresh-seed panel not run.")
"""


STAGE04C_MD = """\
## 04C - Within-Model Selective Coverage Diagnostic

04C reads only 04B prediction artifacts. Selective coverage is within-model
only, and retained-subset dummy deltas use the exact retained sample ids.
"""


STAGE04C_CODE = r"""\
if RUN_04C_SELECTIVE_COVERAGE:
    selective = run_selective_coverage_04c()
    backup_notebook04_outputs("completed_04C_selective_coverage")
    display(selective)
else:
    print("RUN_04C_SELECTIVE_COVERAGE is False; selective coverage not run.")
"""


STAGE04D_MD = """\
## 04D - Manual Gate Decision

04D creates a decision matrix. It does not auto-authorize Notebook 05 or
holdout/test evaluation. The operator must read the matrix and manually select
one exit outside this cell.
"""


STAGE04D_CODE = r"""\
if RUN_04D_GATE_DECISION:
    decision = run_gate_decision_04d()
    backup_notebook04_outputs("completed_04D_gate_decision")
    display(decision)
else:
    print("RUN_04D_GATE_DECISION is False; manual gate decision not run.")
"""


STAGE04E_MD = """\
## Optional 04E - Bootstrap CI Appendix

04E is off by default. If enabled, it reads only 04B prediction artifacts and
computes diagnostic row-level bootstrap confidence intervals for full-coverage
macro F1.

Important caveat: adjacent sliding-window samples in this time-series setup are
autocorrelated. Row-level bootstrap intervals are therefore diagnostic variance
warnings only, not formal independent-sample inference.
"""


STAGE04E_CODE = r"""\
if RUN_04E_BOOTSTRAP_CI:
    bootstrap = run_bootstrap_ci_04e()
    backup_notebook04_outputs("completed_04E_bootstrap_ci")
    display(bootstrap)
else:
    print("RUN_04E_BOOTSTRAP_CI is False; bootstrap CI appendix not run.")
"""


INTERPRETATION_MD = """\
## Interpretation Boundary

Notebook 04 is `validation_only`.

Required interpretation points after a run:

- whether the fixed official Stage 0 candidate passed fresh-seed gates for any
  real model;
- whether each real model was confirmed, marginal, failed fresh-seed
  confirmation, or missing Notebook 03 reference context;
- pooled and per-ticker results;
- dummy baseline deltas on the same target rows;
- selective coverage profiles and ticker concentration warnings;
- sample counts and missing or failed artifact rows;
- explicit statement that holdout/test remains closed.

Allowed wording:

```text
Notebook 04 provides validation-only fresh-seed confirmation and within-model
selective-coverage diagnostics for the fixed official Stage 0 candidate. The
result does not authorize holdout/test evaluation.
```

Forbidden wording:

```text
The best model is ready for holdout.
The model is profitable.
The high-confidence subset is tradable.
Window 32 replaces the official Stage 0 candidate.
Notebook 04 tuned the final model.
The selective curve proves one model is globally better than another.
```
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
        "RUN_04S_SCHEMA_SMOKE",
        "RUN_04A_READ_CONTEXT",
        "RUN_04B_FRESH_SEED_PANEL",
        "RUN_04C_SELECTIVE_COVERAGE",
        "RUN_04D_GATE_DECISION",
        "RUN_04E_BOOTSTRAP_CI",
        "BACKUP_NOTEBOOK04_TO_GOOGLE_DRIVE",
    ):
        if assignment_value(config_source, name) is not False:
            raise AssertionError(f"{name} must default to False.")
    source = "\n".join(cell.source for cell in code_cells)
    forbidden = ("from intraday_research", "baseline_helpers", "train_test_split", "drive.mount(")
    present = [text for text in forbidden if text in source]
    if present:
        raise AssertionError(f"Forbidden active-code strings found: {present}")
    if "holdout_test_authorized = True" in source or '"holdout_test_authorized": True' in source:
        raise AssertionError("Generated notebook contains a holdout/test authorization path.")


def build_notebook() -> nbformat.NotebookNode:
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    setup_code = source.cells[1].source.replace(
        "INSTALL_LIGHTGBM_IF_MISSING = True",
        "INSTALL_LIGHTGBM_IF_MISSING = False",
    )
    data_loading_code = source.cells[4].source.replace(
        "RUN_ANY_STAGE = bool(RUN_STAGE0S or RUN_STAGE0A1 or RUN_STAGE0A2 or RUN_STAGE0B)",
        "RUN_ANY_STAGE = bool(RUN_04B_FRESH_SEED_PANEL)",
    ).replace(
        'print("All RUN_STAGE0* switches are False; data loading skipped.")',
        'print("All Notebook 04 data-loading switches are False; data loading skipped.")',
    )

    nb = new_notebook()
    nb.metadata = source.metadata
    nb.cells = [
        new_markdown_cell(TITLE_MD),
        new_code_cell(setup_code),
        new_code_cell(dedent_code(CONFIG_CODE)),
        new_markdown_cell(source.cells[3].source.replace("Stage 0", "Notebook 04")),
        new_code_cell(data_loading_code),
        new_markdown_cell(source.cells[5].source),
        new_code_cell(source.cells[6].source),
        new_markdown_cell("## Notebook 04 Base Helpers\n\nThis section copies active Stage 0 metric, dataset, tabular, and sequence helper definitions. The following cell overrides only the Notebook 04 orchestration and artifact layer."),
        new_code_cell(source.cells[8].source),
        new_markdown_cell("## Notebook 04 Controlled Follow-Up Helpers\n\nThis layer adds fresh-seed confirmation, prediction artifact persistence, selective coverage, context checks, manual decision matrix, and bootstrap diagnostics."),
        new_code_cell(dedent_code(NOTEBOOK04_HELPERS_CODE)),
        new_markdown_cell(STAGE04S_MD),
        new_code_cell(dedent_code(STAGE04S_CODE)),
        new_markdown_cell(STAGE04A_MD),
        new_code_cell(dedent_code(STAGE04A_CODE)),
        new_markdown_cell(STAGE04B_MD),
        new_code_cell(dedent_code(STAGE04B_CODE)),
        new_markdown_cell(STAGE04C_MD),
        new_code_cell(dedent_code(STAGE04C_CODE)),
        new_markdown_cell(STAGE04D_MD),
        new_code_cell(dedent_code(STAGE04D_CODE)),
        new_markdown_cell(STAGE04E_MD),
        new_code_cell(dedent_code(STAGE04E_CODE)),
        new_markdown_cell(INTERPRETATION_MD),
    ]

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    validate_notebook(nb)
    return nb


def main() -> None:
    nb = build_notebook()
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, TARGET_NOTEBOOK)
    print(f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
