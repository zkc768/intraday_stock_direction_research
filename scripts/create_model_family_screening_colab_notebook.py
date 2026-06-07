"""Generate the Notebook 03 model-family screening Colab notebook.

The generated notebook is standalone for Colab. It copies the active Stage 0
data-loading, feature, label, split, scaling, and window code from
notebooks/02_config_screening_colab.ipynb, then adds Notebook 03-only model
family screening cells. It does not execute the notebook.
"""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "notebooks" / "02_config_screening_colab.ipynb"
TARGET_NOTEBOOK = PROJECT_ROOT / "notebooks" / "03_model_family_screening_colab.ipynb"


TITLE_MD = """\
# Notebook 03 Model-Family Screening - Validation Only

Protocol: `docs/MODEL_FAMILY_SCREENING_PROTOCOL_2026-06-04.md`

Scope: `validation_only`

Research question:

```text
Given the official Stage 0 configuration, does any fixed-default model family
show a stable validation-only signal over simple dummy baselines?
```

Official Stage 0 candidate:

```text
candidate_id  = stage0_official
label_config  = h03_bps1p5
feature_set   = price_volume_time
window_size   = 20
```

This notebook is not a label, feature, threshold, window, or architecture
screen. The final holdout/test remains closed.

Stage 0 context carried forward:

- window 20 is protocol-selected and defensible, not empirically dominant;
- Stage 0B showed LogReg and LightGBM nearly tied, with frozen deep defaults
  below them and `deep_model_disagrees=False`;
- the Stage 0 selected macro F1 is validation-selected and is not expected
  holdout/test performance;
- Notebook 03 must produce its own pooled and per-ticker results.

H0 diagnostic context:

- H0 diagnostic reportedly found `window=32` with a small positive diagnostic
  lift of `+0.002265` macro F1 versus the official baseline;
- this is below the H0 `+0.005` gate;
- H0 therefore remains diagnostic-only and cannot replace the official Stage 0
  candidate.

Hard boundaries:

- raw input comes only from the five Drive `.txt` files listed in this notebook;
- no project helper package or prior notebook is imported as the active path;
- no holdout/test rows are loaded, transformed, windowed, scored, summarized, or
  used for wording decisions;
- train and validation are chronological;
- preprocessing is fit on pooled train rows only after per-ticker splits;
- labels are invalidated at train/validation and validation/closed-holdout
  boundaries;
- windows are generated per ticker, per split, and per trading day;
- every model row is compared with a stratified dummy comparator on the same
  validation target rows;
- all heavy run switches default to `False`.

Notebook 03 stages:

```text
03S = schema/sample-alignment smoke, no model selection
03A = tabular panel: stratified dummy, always-up dummy, LogReg, LightGBM
03B = sequence panel: LSTM, GRU, TCN, DLinear, MS-DLinear+TCN
03C = bootstrap CI only for rows tagged candidate_signal
```

Run profile after 03S passes:

```python
RUN_OVERNIGHT_03A_03B_PROFILE = True
```

That single switch enables 03A, 03B, 03C, and Drive API backup. It keeps 03S and
the H0 appendix off. Leave the switch at `False` when sharing or reviewing the
notebook without running training.

The H0 appendix is read-only and non-selecting. It does not fit models, does not
load raw data, and does not write Notebook 03 selection artifacts.
"""


CONFIG_CODE = r"""\
TICKERS = ("CSCO", "JPM", "KO", "MSFT", "WMT")
MODEL_SEEDS = (101, 202, 303, 404, 505)
RESULT_SCOPE = "validation_only"

RUN_03S_SCHEMA_SMOKE = False
RUN_03A_TABULAR_PANEL = False
RUN_03B_SEQUENCE_PANEL = False
RUN_03C_BOOTSTRAP_CI = False
RUN_H0_DIAGNOSTIC_NOTE = False
BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE = False
BACKUP_FAILURE_IS_FATAL = True
RUN_OVERNIGHT_03A_03B_PROFILE = False

if RUN_OVERNIGHT_03A_03B_PROFILE:
    RUN_03S_SCHEMA_SMOKE = False
    RUN_03A_TABULAR_PANEL = True
    RUN_03B_SEQUENCE_PANEL = True
    RUN_03C_BOOTSTRAP_CI = True
    RUN_H0_DIAGNOSTIC_NOTE = False
    BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE = True

BOOTSTRAP_CI_FULL_PANEL = False
BOOTSTRAP_CI_FOR_CANDIDATES = True
BOOTSTRAP_RESAMPLES = 1000

MIN_PRACTICAL_DELTA_MACRO_F1 = 0.005

NOTEBOOK03_CANDIDATES = (
    {
        "candidate_id": "stage0_official",
        "label_config": "h03_bps1p5",
        "feature_set": "price_volume_time",
        "window_size": 20,
        "source": "completed_stage0_desktop_review_2026-06-04",
    },
)

BASELINE_MODELS = ("stratified_dummy", "always_up_dummy")
TABULAR_MODELS = ("logreg", "lightgbm")
SEQUENCE_MODELS = (
    "vanilla_lstm",
    "simple_gru",
    "standalone_tcn",
    "standard_dlinear",
    "ms_dlinear_tcn",
)
MODEL_PANEL = BASELINE_MODELS + TABULAR_MODELS + SEQUENCE_MODELS

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
TORCH_LSTM_HIDDEN_DIM = 32
TORCH_LSTM_NUM_LAYERS = 1
STANDARD_DLINEAR_MOVING_AVG_KERNEL = 5

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

OUTPUT_DIR = Path("/content/notebook03_model_family_screening_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILES = {
    "pooled": OUTPUT_DIR / "notebook03_pooled.csv",
    "per_ticker": OUTPUT_DIR / "notebook03_per_ticker.csv",
    "summary": OUTPUT_DIR / "notebook03_summary.csv",
    "selection": OUTPUT_DIR / "notebook03_validation_selection.json",
    "manifest": OUTPUT_DIR / "notebook03_run_manifest.json",
}
DRIVE_BACKUP_PROJECT_FOLDER_ID = "15IZ_sOEyyAKmGCUIOv_u17SwQmFX3nG_"
DRIVE_BACKUP_FOLDER_NAME = "notebook03_model_family_screening_results"

H0_OUTPUT_CANDIDATES = (
    Path("/content/diagnostic_h0_tabular_sweep/diagnostic_h0_summary.csv"),
    Path("/content/diagnostic_h0_tabular_sweep/diagnostic_h0_part1_window_sweep.csv"),
)
H0_DIAGNOSTIC_WINDOW = 32
H0_APPENDIX_STATUS = "not_loaded"

NOTEBOOK03_STATE = {
    "pooled_predictions": {},
    "h0_cross_window_appendix": H0_APPENDIX_STATUS,
}

display(pd.DataFrame(NOTEBOOK03_CANDIDATES))
print("Notebook 03 output directory:", OUTPUT_DIR)
print("Model panel:", MODEL_PANEL)
print("Run switches:", {
    "RUN_03S_SCHEMA_SMOKE": RUN_03S_SCHEMA_SMOKE,
    "RUN_03A_TABULAR_PANEL": RUN_03A_TABULAR_PANEL,
    "RUN_03B_SEQUENCE_PANEL": RUN_03B_SEQUENCE_PANEL,
    "RUN_03C_BOOTSTRAP_CI": RUN_03C_BOOTSTRAP_CI,
    "RUN_H0_DIAGNOSTIC_NOTE": RUN_H0_DIAGNOSTIC_NOTE,
    "BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE": BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE,
    "RUN_OVERNIGHT_03A_03B_PROFILE": RUN_OVERNIGHT_03A_03B_PROFILE,
})
print("Drive backup folder:", DRIVE_BACKUP_FOLDER_NAME)
"""


NOTEBOOK03_HELPERS_CODE = r"""\
# This get_dataset intentionally overrides the 02-copied helper above.
# It adds timestamp-level sample-alignment checks and a Notebook 03-specific
# error message. Execute the notebook top-to-bottom so this stricter version is
# active before 03S/03A/03B.
def get_dataset(label_config, feature_set, window_size):
    key = (label_config, feature_set, int(window_size))
    if key in DATASET_CACHE:
        dataset = DATASET_CACHE[key].copy()
        dataset["prep_seconds"] = 0.0
        return dataset
    if not raw_data:
        raise RuntimeError("raw_data is empty. Enable a Notebook 03 run switch and rerun data loading first.")
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


def assert_sample_alignment(y_flat, y_seq, owner_flat, owner_seq, timestamp_flat, timestamp_seq, split_name):
    if len(y_flat) != len(y_seq):
        raise ValueError(f"{split_name} tabular and sequence labels have different lengths.")
    if not np.array_equal(y_flat, y_seq):
        raise ValueError(f"{split_name} tabular and sequence labels are not aligned.")
    if not np.array_equal(owner_flat, owner_seq):
        raise ValueError(f"{split_name} tabular and sequence ticker owners are not aligned.")
    if not np.array_equal(timestamp_flat, timestamp_seq):
        raise ValueError(f"{split_name} tabular and sequence timestamps are not aligned.")


def model_role(model_name):
    if model_name in BASELINE_MODELS:
        return "baseline"
    if model_name in TABULAR_MODELS:
        return "candidate_model"
    if model_name == "ms_dlinear_tcn":
        return "candidate_model"
    return "controlled_baseline"


def confusion_counts(y_true, predictions):
    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(predictions).astype(int)
    return {
        "cm_tn": int(((y_true == 0) & (predictions == 0)).sum()),
        "cm_fp": int(((y_true == 0) & (predictions == 1)).sum()),
        "cm_fn": int(((y_true == 1) & (predictions == 0)).sum()),
        "cm_tp": int(((y_true == 1) & (predictions == 1)).sum()),
    }


def prediction_diagnostics(y_true, predictions):
    metrics = evaluate_predictions(y_true, predictions)
    predictions = np.asarray(predictions).astype(int)
    pred_up_pct = float((predictions == 1).mean()) if len(predictions) else np.nan
    pred_down_pct = float((predictions == 0).mean()) if len(predictions) else np.nan
    metrics.update({
        "pred_up_pct": pred_up_pct,
        "pred_down_pct": pred_down_pct,
        "one_class_collapse": bool(pred_up_pct > 0.95 or pred_down_pct > 0.95),
        **confusion_counts(y_true, predictions),
    })
    return metrics


def stratified_dummy_predictions(y_train, y_validation, seed):
    if len(y_train) == 0 or len(y_validation) == 0:
        return np.full(len(y_validation), np.nan)
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return dummy.predict(np.zeros((len(y_validation), 1))).astype(int)


def always_up_predictions(y_validation):
    return np.ones(len(y_validation), dtype=int)


def make_vanilla_lstm(input_dim, seed):
    set_global_seed(seed)
    torch = ensure_torch()
    import torch.nn as nn

    class VanillaLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                TORCH_LSTM_HIDDEN_DIM,
                num_layers=TORCH_LSTM_NUM_LAYERS,
                batch_first=True,
            )
            self.dropout = nn.Dropout(TORCH_DROPOUT)
            self.head = nn.Linear(TORCH_LSTM_HIDDEN_DIM, 2)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(self.dropout(output[:, -1, :]))

    return VanillaLSTMClassifier()


def make_standalone_tcn(input_dim, seed):
    set_global_seed(seed)
    torch = ensure_torch()
    import torch.nn as nn

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


def make_standard_dlinear(input_dim, window_size, seed):
    set_global_seed(seed)
    torch = ensure_torch()
    import torch.nn as nn

    class StandardDLinearClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            kernel_size = STANDARD_DLINEAR_MOVING_AVG_KERNEL
            self.left_pad = (kernel_size - 1) // 2
            self.right_pad = kernel_size - 1 - self.left_pad
            self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
            self.seasonal_linear = nn.Linear(window_size, 1)
            self.trend_linear = nn.Linear(window_size, 1)
            self.head = nn.Linear(input_dim * 2, 2)

        def moving_average(self, x):
            front = x[:, 0:1, :].repeat(1, self.left_pad, 1)
            end = x[:, -1:, :].repeat(1, self.right_pad, 1)
            padded = torch.cat([front, x, end], dim=1)
            trend = self.avg_pool(padded.transpose(1, 2)).transpose(1, 2)
            return trend

        def forward(self, x):
            trend = self.moving_average(x)
            seasonal = x - trend
            seasonal_out = self.seasonal_linear(seasonal.transpose(1, 2)).squeeze(-1)
            trend_out = self.trend_linear(trend.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([seasonal_out, trend_out], dim=1))

    return StandardDLinearClassifier()


def make_sequence_model_03(model_name, input_dim, window_size, seed):
    if model_name == "vanilla_lstm":
        return make_vanilla_lstm(input_dim, seed)
    if model_name == "simple_gru":
        return make_simple_gru(input_dim, seed)
    if model_name == "standalone_tcn":
        return make_standalone_tcn(input_dim, seed)
    if model_name == "standard_dlinear":
        return make_standard_dlinear(input_dim, window_size, seed)
    if model_name == "ms_dlinear_tcn":
        return make_ms_dlinear_tcn(input_dim, window_size, seed)
    raise ValueError(f"Unsupported sequence model: {model_name}")


def run_notebook03_shape_smoke(input_dim, window_size):
    torch = ensure_torch()
    for model_name in SEQUENCE_MODELS:
        model = make_sequence_model_03(model_name, input_dim, window_size, seed=101)
        model.eval()
        x = torch.zeros((2, window_size, input_dim), dtype=torch.float32)
        with torch.no_grad():
            out = model(x)
        if tuple(out.shape) != (2, 2):
            raise ValueError(f"{model_name} smoke output shape mismatch: {tuple(out.shape)}")
    print("Notebook 03 torch shape smoke passed for", SEQUENCE_MODELS)


def fit_predict_torch_sequence_03(dataset, seed, model_name):
    torch = ensure_torch()
    from torch.utils.data import DataLoader, TensorDataset

    x_train, y_train, train_owner = subsample_rows_with_owner(
        dataset["x_train_seq"],
        dataset["y_train_seq"],
        dataset["train_owner_seq"],
        MAX_TRAIN_ROWS,
        seed,
    )
    x_validation = dataset["x_validation_seq"]
    train_n = len(y_train)
    input_dim = x_train.shape[-1]
    window_size = x_train.shape[1]
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_sequence_model_03(model_name, input_dim, window_size, seed)
    model.to(device)
    counts = np.bincount(y_train.astype(int), minlength=2).astype(float)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=TORCH_LEARNING_RATE, weight_decay=TORCH_WEIGHT_DECAY)
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train.astype(int), dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=TORCH_BATCH_SIZE, shuffle=True, generator=generator)
    fit_start = time.perf_counter()
    model.train()
    for _ in range(TORCH_EPOCHS):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name} produced non-finite loss.")
            loss.backward()
            optimizer.step()
    fit_seconds = time.perf_counter() - fit_start
    pred_start = time.perf_counter()
    model.eval()
    preds = []
    with torch.no_grad():
        for start_idx in range(0, len(x_validation), TORCH_BATCH_SIZE):
            xb = torch.tensor(x_validation[start_idx:start_idx + TORCH_BATCH_SIZE], dtype=torch.float32, device=device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    predictions = np.concatenate(preds)
    if np.isnan(predictions).any():
        raise RuntimeError(f"{model_name} produced NaN predictions.")
    predict_seconds = time.perf_counter() - pred_start
    return predictions, fit_seconds, predict_seconds, train_n, f"fit_ok_device_{device}"


def fit_predict_model_03(dataset, model_name, seed):
    if model_name == "stratified_dummy":
        start = time.perf_counter()
        predictions = stratified_dummy_predictions(dataset["y_train"], dataset["y_validation"], seed)
        return predictions, 0.0, time.perf_counter() - start, len(dataset["y_train"]), "baseline_stratified"
    if model_name == "always_up_dummy":
        start = time.perf_counter()
        predictions = always_up_predictions(dataset["y_validation"])
        return predictions, 0.0, time.perf_counter() - start, len(dataset["y_train"]), "baseline_always_up"
    if model_name == "logreg":
        return fit_predict_logreg(dataset, seed)
    if model_name == "lightgbm":
        return fit_predict_lightgbm(dataset, seed)
    if model_name in SEQUENCE_MODELS:
        return fit_predict_torch_sequence_03(dataset, seed, model_name)
    raise ValueError(f"Unknown Notebook 03 model: {model_name}")


def row_from_predictions(stage, candidate, model_name, seed, dataset, y_true, predictions, strat_pred, always_pred, ticker, train_n, timings, fit_status):
    metrics = prediction_diagnostics(y_true, predictions)
    strat_metrics = prediction_diagnostics(y_true, strat_pred)
    always_metrics = prediction_diagnostics(y_true, always_pred)
    prep_seconds, fit_seconds, predict_seconds = timings
    return {
        "stage": stage,
        "model": model_name,
        "candidate_id": candidate["candidate_id"],
        "model_role": model_role(model_name),
        "label_config": dataset["label_config"],
        "horizon_k": dataset["horizon_k"],
        "threshold_bps": dataset["threshold_bps"],
        "feature_set": dataset["feature_set"],
        "window_size": int(dataset["window_size"]),
        "seed": int(seed),
        "scope": RESULT_SCOPE,
        "ticker_or_pooled": ticker,
        "n": int(len(y_true)),
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "stratified_dummy_macro_f1": strat_metrics["macro_f1"],
        "stratified_dummy_balanced_accuracy": strat_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_stratified_dummy": metrics["macro_f1"] - strat_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_stratified_dummy": metrics["balanced_accuracy"] - strat_metrics["balanced_accuracy"],
        "always_up_dummy_macro_f1": always_metrics["macro_f1"],
        "always_up_dummy_balanced_accuracy": always_metrics["balanced_accuracy"],
        "delta_macro_f1_vs_always_up_dummy": metrics["macro_f1"] - always_metrics["macro_f1"],
        "delta_balanced_accuracy_vs_always_up_dummy": metrics["balanced_accuracy"] - always_metrics["balanced_accuracy"],
        "pred_up_pct": metrics["pred_up_pct"],
        "pred_down_pct": metrics["pred_down_pct"],
        "one_class_collapse": metrics["one_class_collapse"],
        "cm_tn": metrics["cm_tn"],
        "cm_fp": metrics["cm_fp"],
        "cm_fn": metrics["cm_fn"],
        "cm_tp": metrics["cm_tp"],
        "prep_seconds": float(prep_seconds),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(prep_seconds + fit_seconds + predict_seconds),
        "fit_status": fit_status,
        "run_failed": False,
        "failure_reason": "",
        "train_n": int(train_n),
        "positive_ticker_count": np.nan,
        "top_ticker_gain_share": np.nan,
    }


def failure_rows(stage, candidate, model_name, seed, dataset, failure_reason):
    base = {
        "stage": stage,
        "model": model_name,
        "candidate_id": candidate["candidate_id"],
        "model_role": model_role(model_name),
        "label_config": dataset["label_config"],
        "horizon_k": dataset["horizon_k"],
        "threshold_bps": dataset["threshold_bps"],
        "feature_set": dataset["feature_set"],
        "window_size": int(dataset["window_size"]),
        "seed": int(seed),
        "scope": RESULT_SCOPE,
        "fit_status": "run_failed",
        "run_failed": True,
        "failure_reason": str(failure_reason)[:500],
        "train_n": int(len(dataset["y_train"])),
        "positive_ticker_count": np.nan,
        "top_ticker_gain_share": np.nan,
    }
    metric_fields = {
        "macro_f1": np.nan,
        "balanced_accuracy": np.nan,
        "accuracy": np.nan,
        "stratified_dummy_macro_f1": np.nan,
        "stratified_dummy_balanced_accuracy": np.nan,
        "delta_macro_f1_vs_stratified_dummy": np.nan,
        "delta_balanced_accuracy_vs_stratified_dummy": np.nan,
        "always_up_dummy_macro_f1": np.nan,
        "always_up_dummy_balanced_accuracy": np.nan,
        "delta_macro_f1_vs_always_up_dummy": np.nan,
        "delta_balanced_accuracy_vs_always_up_dummy": np.nan,
        "pred_up_pct": np.nan,
        "pred_down_pct": np.nan,
        "one_class_collapse": np.nan,
        "cm_tn": np.nan,
        "cm_fp": np.nan,
        "cm_fn": np.nan,
        "cm_tp": np.nan,
        "prep_seconds": float(dataset.get("prep_seconds", np.nan)),
        "fit_seconds": np.nan,
        "predict_seconds": np.nan,
        "total_seconds": np.nan,
    }
    pooled = {**base, **metric_fields, "ticker_or_pooled": "pooled", "n": int(len(dataset["y_validation"]))}
    per_ticker = []
    for ticker in TICKERS:
        mask = dataset["validation_owner"] == ticker
        per_ticker.append({**base, **metric_fields, "ticker_or_pooled": ticker, "n": int(mask.sum())})
    return pooled, per_ticker


def concentration_from_rows(per_ticker_rows):
    deltas = [
        row["delta_macro_f1_vs_stratified_dummy"]
        for row in per_ticker_rows
        if not pd.isna(row["delta_macro_f1_vs_stratified_dummy"])
    ]
    if not deltas:
        return 0, np.nan
    positive = [delta for delta in deltas if delta > 0]
    positive_count = len(positive)
    total_gain = sum(max(delta, 0.0) for delta in deltas)
    top_share = max([max(delta, 0.0) for delta in deltas], default=0.0) / total_gain if total_gain > 0 else 0.0
    return positive_count, top_share


def run_one_notebook03_seed(stage, candidate, model_name, seed):
    dataset = get_dataset(candidate["label_config"], candidate["feature_set"], candidate["window_size"])
    prep_seconds = float(dataset.get("prep_seconds", 0.0))
    try:
        predictions, fit_seconds, predict_seconds, train_n, fit_status = fit_predict_model_03(dataset, model_name, seed)
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        pooled, per_ticker = failure_rows(stage, candidate, model_name, seed, dataset, exc)
        return pooled, per_ticker
    y_validation = dataset["y_validation"]
    if len(predictions) != len(y_validation):
        raise ValueError(f"{model_name} prediction length mismatch: {len(predictions)} vs {len(y_validation)}")
    strat_pred = stratified_dummy_predictions(dataset["y_train"], y_validation, seed)
    always_pred = always_up_predictions(y_validation)
    pooled = row_from_predictions(
        stage,
        candidate,
        model_name,
        seed,
        dataset,
        y_validation,
        predictions,
        strat_pred,
        always_pred,
        "pooled",
        train_n,
        (prep_seconds, fit_seconds, predict_seconds),
        fit_status,
    )
    per_ticker_rows = []
    for ticker in TICKERS:
        val_mask = dataset["validation_owner"] == ticker
        train_mask = dataset["train_owner"] == ticker
        ticker_strat_pred = stratified_dummy_predictions(
            dataset["y_train"][train_mask],
            dataset["y_validation"][val_mask],
            seed,
        )
        ticker_always_pred = always_up_predictions(dataset["y_validation"][val_mask])
        if model_name == "stratified_dummy":
            ticker_model_predictions = ticker_strat_pred
        elif model_name == "always_up_dummy":
            ticker_model_predictions = ticker_always_pred
        else:
            ticker_model_predictions = predictions[val_mask]
        per_ticker_rows.append(row_from_predictions(
            stage,
            candidate,
            model_name,
            seed,
            dataset,
            dataset["y_validation"][val_mask],
            ticker_model_predictions,
            ticker_strat_pred,
            ticker_always_pred,
            ticker,
            int(train_mask.sum()),
            (prep_seconds, fit_seconds, predict_seconds),
            fit_status,
        ))
    positive_count, top_share = concentration_from_rows(per_ticker_rows)
    pooled["positive_ticker_count"] = positive_count
    pooled["top_ticker_gain_share"] = top_share
    for row in per_ticker_rows:
        row["positive_ticker_count"] = positive_count
        row["top_ticker_gain_share"] = top_share
    NOTEBOOK03_STATE["pooled_predictions"][(candidate["candidate_id"], model_name, int(seed))] = {
        "y_true": y_validation.copy(),
        "predictions": np.asarray(predictions).copy(),
    }
    return pooled, per_ticker_rows


def run_notebook03_grid(stage, model_names):
    pooled_rows = []
    per_ticker_rows = []
    specs = [
        (candidate, model_name, seed)
        for candidate in NOTEBOOK03_CANDIDATES
        for model_name in model_names
        for seed in MODEL_SEEDS
    ]
    for idx, (candidate, model_name, seed) in enumerate(specs, start=1):
        print(
            f"{idx}/{len(specs)}",
            stage,
            candidate["candidate_id"],
            model_name,
            candidate["label_config"],
            candidate["feature_set"],
            "window",
            candidate["window_size"],
            "seed",
            seed,
        )
        pooled, per_ticker = run_one_notebook03_seed(stage, candidate, model_name, seed)
        pooled_rows.append(pooled)
        per_ticker_rows.extend(per_ticker)
    return pd.DataFrame(pooled_rows), pd.DataFrame(per_ticker_rows)


def summarize_notebook03_pooled(pooled):
    if pooled.empty:
        return pd.DataFrame()
    rows = []
    keys = ["candidate_id", "model", "model_role", "label_config", "horizon_k", "threshold_bps", "feature_set", "window_size", "scope"]
    for key_values, group in pooled.groupby(keys, sort=False):
        record = dict(zip(keys, key_values))
        n_failed = int(group["run_failed"].astype(bool).sum())
        successful = group.loc[~group["run_failed"].astype(bool)].copy()
        seed_count = int(successful["seed"].nunique())
        record["n_failed_seeds"] = n_failed
        record["seed_count"] = seed_count
        if successful.empty:
            record.update({
                "macro_f1_mean": np.nan,
                "macro_f1_std": np.nan,
                "macro_f1_lcb_95": np.nan,
                "balanced_accuracy_mean": np.nan,
                "balanced_accuracy_std": np.nan,
                "stratified_dummy_macro_f1_mean": np.nan,
                "stratified_dummy_macro_f1_std": np.nan,
                "delta_macro_f1_vs_stratified_dummy_mean": np.nan,
                "delta_balanced_accuracy_vs_stratified_dummy_mean": np.nan,
                "always_up_dummy_macro_f1_mean": np.nan,
                "delta_macro_f1_vs_always_up_dummy_mean": np.nan,
                "n_mean": np.nan,
                "positive_ticker_count": 0,
                "top_ticker_gain_share": np.nan,
                "pred_up_pct_mean": np.nan,
                "pred_down_pct_mean": np.nan,
                "one_class_collapse_any": np.nan,
                "run_failed": True,
                "failure_reason": "; ".join(str(value) for value in group["failure_reason"].dropna().unique())[:500],
                "signal_strength_tag": "run_failed",
                "macro_f1_bootstrap_ci_lower": np.nan,
                "macro_f1_bootstrap_ci_upper": np.nan,
            })
            rows.append(record)
            continue

        macro_std = sample_std(successful["macro_f1"])
        bal_std = sample_std(successful["balanced_accuracy"])
        dummy_std = sample_std(successful["stratified_dummy_macro_f1"])
        one_class_any = bool(successful["one_class_collapse"].astype(bool).any())
        macro_mean = float(successful["macro_f1"].mean())
        macro_lcb = float(macro_mean - t_critical_one_sided_95(seed_count) * macro_std / math.sqrt(max(seed_count, 1)))
        record.update({
            "macro_f1_mean": macro_mean,
            "macro_f1_std": macro_std,
            "macro_f1_lcb_95": macro_lcb,
            "balanced_accuracy_mean": float(successful["balanced_accuracy"].mean()),
            "balanced_accuracy_std": bal_std,
            "stratified_dummy_macro_f1_mean": float(successful["stratified_dummy_macro_f1"].mean()),
            "stratified_dummy_macro_f1_std": dummy_std,
            "delta_macro_f1_vs_stratified_dummy_mean": float(successful["delta_macro_f1_vs_stratified_dummy"].mean()),
            "delta_balanced_accuracy_vs_stratified_dummy_mean": float(successful["delta_balanced_accuracy_vs_stratified_dummy"].mean()),
            "always_up_dummy_macro_f1_mean": float(successful["always_up_dummy_macro_f1"].mean()),
            "delta_macro_f1_vs_always_up_dummy_mean": float(successful["delta_macro_f1_vs_always_up_dummy"].mean()),
            "n_mean": float(successful["n"].mean()),
            "positive_ticker_count": int(round(successful["positive_ticker_count"].mean())),
            "top_ticker_gain_share": float(successful["top_ticker_gain_share"].mean()),
            "pred_up_pct_mean": float(successful["pred_up_pct"].mean()),
            "pred_down_pct_mean": float(successful["pred_down_pct"].mean()),
            "one_class_collapse_any": one_class_any,
            "run_failed": False,
            "failure_reason": "; ".join(str(value) for value in group.loc[group["run_failed"].astype(bool), "failure_reason"].dropna().unique())[:500],
            "macro_f1_bootstrap_ci_lower": np.nan,
            "macro_f1_bootstrap_ci_upper": np.nan,
        })
        record["basic_gate"] = bool(
            record["delta_macro_f1_vs_stratified_dummy_mean"] > 0
            and record["macro_f1_lcb_95"] > record["stratified_dummy_macro_f1_mean"]
        )
        record["lcb_eligible"] = bool(
            record["basic_gate"]
            and record["delta_balanced_accuracy_vs_stratified_dummy_mean"] > 0
            and record["top_ticker_gain_share"] < 0.50
            and record["positive_ticker_count"] >= 3
        )
        seed_unstable = bool(record["macro_f1_std"] >= MIN_PRACTICAL_DELTA_MACRO_F1 / 2)
        record["seed_unstable"] = seed_unstable
        record["signal_strength_tag"] = signal_strength_tag(record)
        rows.append(record)
    return pd.DataFrame(rows)


def signal_strength_tag(record):
    if int(record["n_failed_seeds"]) >= 3:
        return "run_failed"
    if record["model"] in BASELINE_MODELS:
        return "no_signal"
    if not bool(record.get("basic_gate", False)):
        return "no_signal"
    if (
        bool(record["one_class_collapse_any"])
        or float(record["top_ticker_gain_share"]) >= 0.50
        or int(record["positive_ticker_count"]) < 3
        or bool(record["seed_unstable"])
    ):
        return "unstable_signal"
    if (
        bool(record.get("lcb_eligible", False))
        and float(record["delta_macro_f1_vs_stratified_dummy_mean"]) >= MIN_PRACTICAL_DELTA_MACRO_F1
        and not bool(record["one_class_collapse_any"])
        and not bool(record["seed_unstable"])
    ):
        return "candidate_signal"
    return "near_dummy"


def bootstrap_candidate_rows(summary):
    if summary.empty or not (BOOTSTRAP_CI_FULL_PANEL or BOOTSTRAP_CI_FOR_CANDIDATES):
        return summary
    summary = summary.copy()
    for idx, row in summary.iterrows():
        if not BOOTSTRAP_CI_FULL_PANEL and row["signal_strength_tag"] != "candidate_signal":
            continue
        lowers = []
        uppers = []
        missing_prediction_keys = []
        for seed in MODEL_SEEDS:
            key = (row["candidate_id"], row["model"], int(seed))
            payload = NOTEBOOK03_STATE["pooled_predictions"].get(key)
            if payload is None:
                missing_prediction_keys.append(key)
                continue
            rng = np.random.default_rng(int(seed) + 9000)
            y_true = payload["y_true"]
            predictions = payload["predictions"]
            values = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                sample_idx = rng.integers(0, len(y_true), size=len(y_true))
                values.append(f1_score(y_true[sample_idx], predictions[sample_idx], labels=[0, 1], average="macro", zero_division=0))
            lowers.append(float(np.quantile(values, 0.025)))
            uppers.append(float(np.quantile(values, 0.975)))
        if missing_prediction_keys and row["signal_strength_tag"] == "candidate_signal":
            raise RuntimeError(
                "Bootstrap requested for candidate_signal, but in-memory predictions are missing. "
                "Run 03C in the same Colab runtime after 03A/03B, or rerun the model panel. "
                f"Missing keys: {missing_prediction_keys}"
            )
        if lowers and uppers:
            summary.loc[idx, "macro_f1_bootstrap_ci_lower"] = min(lowers)
            summary.loc[idx, "macro_f1_bootstrap_ci_upper"] = max(uppers)
    return summary


def existing_frame(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def update_notebook03_outputs(new_pooled, new_per_ticker):
    pooled = existing_frame(OUTPUT_FILES["pooled"])
    per_ticker = existing_frame(OUTPUT_FILES["per_ticker"])
    if not pooled.empty:
        pooled = pooled.loc[~pooled["stage"].isin(new_pooled["stage"].unique())]
    if not per_ticker.empty:
        per_ticker = per_ticker.loc[~per_ticker["stage"].isin(new_per_ticker["stage"].unique())]
    pooled = pd.concat([pooled, new_pooled], ignore_index=True) if not pooled.empty else new_pooled
    per_ticker = pd.concat([per_ticker, new_per_ticker], ignore_index=True) if not per_ticker.empty else new_per_ticker
    summary = summarize_notebook03_pooled(pooled)
    selection = build_selection_record(summary)
    pooled.to_csv(OUTPUT_FILES["pooled"], index=False)
    per_ticker.to_csv(OUTPUT_FILES["per_ticker"], index=False)
    summary.to_csv(OUTPUT_FILES["summary"], index=False)
    with OUTPUT_FILES["selection"].open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
    write_run_manifest(pooled, per_ticker, summary)
    backup_notebook03_outputs(reason_from_stages(new_pooled["stage"].unique()))
    print("wrote Notebook 03 outputs:", [str(path) for path in OUTPUT_FILES.values()])
    return pooled, per_ticker, summary, selection


def reason_from_stages(stages):
    return "completed_" + "_".join(str(stage).replace(" ", "_") for stage in sorted(stages))


def build_selection_record(summary):
    all_panel_tags = {}
    selected = []
    if summary.empty:
        status = "no_rows"
    else:
        for _, row in summary.iterrows():
            key = f"{row['candidate_id']}::{row['model']}"
            all_panel_tags[key] = row["signal_strength_tag"]
            if row["signal_strength_tag"] == "candidate_signal" and row["model"] not in BASELINE_MODELS:
                selected.append({
                    "candidate_id": row["candidate_id"],
                    "label_config": row["label_config"],
                    "feature_set": row["feature_set"],
                    "window_size": int(row["window_size"]),
                    "model": row["model"],
                    "signal_strength_tag": row["signal_strength_tag"],
                    "fixed_params": fixed_params_for_model(row["model"]),
                    "validation_macro_f1_mean": nullable_float(row["macro_f1_mean"]),
                    "validation_macro_f1_lcb_95": nullable_float(row["macro_f1_lcb_95"]),
                    "delta_macro_f1_vs_stratified_dummy_mean": nullable_float(row["delta_macro_f1_vs_stratified_dummy_mean"]),
                })
        status = "candidate_signal_found" if selected else "no_validation_signal_under_current_config"
    dummy_rows = summary.loc[summary["model"].isin(BASELINE_MODELS)] if not summary.empty else pd.DataFrame()
    dummy_baselines = {
        f"{row['candidate_id']}::{row['model']}": {
            "macro_f1_mean": nullable_float(row["macro_f1_mean"]),
            "balanced_accuracy_mean": nullable_float(row["balanced_accuracy_mean"]),
        }
        for _, row in dummy_rows.iterrows()
    }
    return {
        "scope": RESULT_SCOPE,
        "selection_status": status,
        "official_candidates": list(NOTEBOOK03_CANDIDATES),
        "selected_branches": selected,
        "dummy_baselines": dummy_baselines,
        "all_panel_tags": all_panel_tags,
        "h0_cross_window_appendix": NOTEBOOK03_STATE.get("h0_cross_window_appendix", "not_loaded"),
        "h0_window32_participates_in_selection": False,
        "holdout_test_authorized": False,
    }


def nullable_float(value):
    if pd.isna(value):
        return None
    return float(value)


def fixed_params_for_model(model_name):
    if model_name == "logreg":
        return {"solver": "liblinear", "class_weight": "balanced", "max_iter": 2000}
    if model_name == "lightgbm":
        return dict(LGBM_PARAMS)
    if model_name in SEQUENCE_MODELS:
        return {
            "epochs": TORCH_EPOCHS,
            "batch_size": TORCH_BATCH_SIZE,
            "learning_rate": TORCH_LEARNING_RATE,
            "weight_decay": TORCH_WEIGHT_DECAY,
            "dropout": TORCH_DROPOUT,
        }
    return {}


def write_run_manifest(pooled, per_ticker, summary):
    manifest = {
        "scope": RESULT_SCOPE,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "output_dir": str(OUTPUT_DIR),
        "run_switches": {
            "RUN_03S_SCHEMA_SMOKE": RUN_03S_SCHEMA_SMOKE,
            "RUN_03A_TABULAR_PANEL": RUN_03A_TABULAR_PANEL,
            "RUN_03B_SEQUENCE_PANEL": RUN_03B_SEQUENCE_PANEL,
            "RUN_03C_BOOTSTRAP_CI": RUN_03C_BOOTSTRAP_CI,
            "RUN_H0_DIAGNOSTIC_NOTE": RUN_H0_DIAGNOSTIC_NOTE,
            "BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE": BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE,
            "RUN_OVERNIGHT_03A_03B_PROFILE": RUN_OVERNIGHT_03A_03B_PROFILE,
        },
        "row_counts": {
            "pooled": int(len(pooled)),
            "per_ticker": int(len(per_ticker)),
            "summary": int(len(summary)),
        },
        "holdout_test_authorized": False,
        "h0_cross_window_appendix": NOTEBOOK03_STATE.get("h0_cross_window_appendix", "not_loaded"),
        "drive_backup": {
            "enabled": bool(BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE),
            "project_folder_id": DRIVE_BACKUP_PROJECT_FOLDER_ID,
            "folder_name": DRIVE_BACKUP_FOLDER_NAME,
        },
    }
    with OUTPUT_FILES["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_drive_service_for_backup():
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Drive backup requested, but the Colab Drive API client is unavailable. "
            "Open this notebook in Google Colab and authenticate when prompted."
        ) from exc
    auth.authenticate_user()
    return build("drive", "v3")


def drive_query_literal(value):
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def find_or_create_drive_folder(service, folder_name, parent_id):
    escaped_name = drive_query_literal(folder_name)
    escaped_parent = drive_query_literal(parent_id)
    query = (
        "mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{escaped_name}' "
        f"and '{escaped_parent}' in parents "
        "and trashed = false"
    )
    response = service.files().list(
        q=query,
        fields="files(id, name, webViewLink)",
        pageSize=10,
        spaces="drive",
    ).execute()
    matches = response.get("files", [])
    if matches:
        return matches[0]
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(
        body=metadata,
        fields="id, name, webViewLink",
    ).execute()


def upload_local_file_to_drive(service, local_path, folder_id, uploaded_name):
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise RuntimeError("googleapiclient MediaFileUpload is unavailable in this Colab runtime.") from exc
    media = MediaFileUpload(str(local_path), resumable=False)
    metadata = {"name": uploaded_name, "parents": [folder_id]}
    return service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, webViewLink",
    ).execute()


def backup_notebook03_outputs(reason):
    if not BACKUP_NOTEBOOK03_TO_GOOGLE_DRIVE:
        return []
    try:
        service = build_drive_service_for_backup()
        backup_folder = find_or_create_drive_folder(
            service,
            DRIVE_BACKUP_FOLDER_NAME,
            DRIVE_BACKUP_PROJECT_FOLDER_ID,
        )
        timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        uploaded = []
        for path in OUTPUT_FILES.values():
            if not path.exists():
                continue
            uploaded_name = f"{timestamp}__{reason}__{path.name}"
            drive_file = upload_local_file_to_drive(
                service,
                path,
                backup_folder["id"],
                uploaded_name,
            )
            uploaded.append({"local_path": str(path), "drive_file": drive_file})
        backup_manifest = {
            "scope": RESULT_SCOPE,
            "reason": reason,
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "local_output_dir": str(OUTPUT_DIR),
            "drive_project_folder_id": DRIVE_BACKUP_PROJECT_FOLDER_ID,
            "drive_backup_folder": backup_folder,
            "uploaded_files": uploaded,
            "holdout_test_authorized": False,
        }
        manifest_path = OUTPUT_DIR / f"{timestamp}__{reason}__notebook03_drive_backup_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(backup_manifest, handle, indent=2)
        manifest_file = upload_local_file_to_drive(
            service,
            manifest_path,
            backup_folder["id"],
            manifest_path.name,
        )
        uploaded.append({"local_path": str(manifest_path), "drive_file": manifest_file})
        print("Backed up Notebook 03 outputs to Drive folder:", backup_folder)
        return uploaded
    except Exception as exc:
        message = f"Notebook 03 Drive backup failed after local outputs were written: {exc}"
        if BACKUP_FAILURE_IS_FATAL:
            raise RuntimeError(message) from exc
        print("WARNING:", message)
        return []
"""


STAGE03S_MD = """\
## 03S - Schema And Sample-Alignment Smoke

This stage builds the official candidate dataset, asserts tabular/sequence
sample alignment, and shape-checks the sequence models. It does not fit models
and does not participate in selection.
"""


STAGE03S_CODE = r"""\
if RUN_03S_SCHEMA_SMOKE:
    for candidate in NOTEBOOK03_CANDIDATES:
        dataset = get_dataset(candidate["label_config"], candidate["feature_set"], candidate["window_size"])
        display(pd.DataFrame([{
            "candidate_id": candidate["candidate_id"],
            "label_config": dataset["label_config"],
            "feature_set": dataset["feature_set"],
            "window_size": dataset["window_size"],
            "train_n": len(dataset["y_train"]),
            "validation_n": len(dataset["y_validation"]),
            "n_features": len(dataset["feature_columns"]),
        }]))
        run_notebook03_shape_smoke(
            input_dim=len(dataset["feature_columns"]),
            window_size=int(dataset["window_size"]),
        )
else:
    print("RUN_03S_SCHEMA_SMOKE is False; schema smoke not run.")
"""


STAGE03A_MD = """\
## 03A - Tabular Panel

Fixed panel:

```text
stratified_dummy
always_up_dummy
logreg
lightgbm
```

This stage is CPU-oriented and writes/updates the Notebook 03 pooled,
per-ticker, summary, and selection JSON artifacts.
"""


STAGE03A_CODE = r"""\
if RUN_03A_TABULAR_PANEL:
    notebook03a_pooled, notebook03a_per_ticker = run_notebook03_grid(
        "03A_tabular_panel",
        BASELINE_MODELS + TABULAR_MODELS,
    )
    pooled, per_ticker, summary, selection = update_notebook03_outputs(notebook03a_pooled, notebook03a_per_ticker)
    display(summary.sort_values(["candidate_id", "signal_strength_tag", "macro_f1_mean"], ascending=[True, True, False]))
    print(json.dumps(selection, indent=2))
else:
    print("RUN_03A_TABULAR_PANEL is False; tabular panel not run.")
"""


STAGE03B_MD = """\
## 03B - Sequence Panel

Fixed panel:

```text
vanilla_lstm
simple_gru
standalone_tcn
standard_dlinear
ms_dlinear_tcn
```

This stage is GPU-oriented. If a sequence model fails, the notebook records a
`run_failed=True` audit row and does not change architecture, batch size,
epochs, or hyperparameters inside Notebook 03.
"""


STAGE03B_CODE = r"""\
if RUN_03B_SEQUENCE_PANEL:
    first_candidate = NOTEBOOK03_CANDIDATES[0]
    first_dataset = get_dataset(
        first_candidate["label_config"],
        first_candidate["feature_set"],
        first_candidate["window_size"],
    )
    run_notebook03_shape_smoke(
        input_dim=len(first_dataset["feature_columns"]),
        window_size=int(first_dataset["window_size"]),
    )
    notebook03b_pooled, notebook03b_per_ticker = run_notebook03_grid(
        "03B_sequence_panel",
        SEQUENCE_MODELS,
    )
    pooled, per_ticker, summary, selection = update_notebook03_outputs(notebook03b_pooled, notebook03b_per_ticker)
    display(summary.sort_values(["candidate_id", "signal_strength_tag", "macro_f1_mean"], ascending=[True, True, False]))
    print(json.dumps(selection, indent=2))
else:
    print("RUN_03B_SEQUENCE_PANEL is False; sequence panel not run.")
"""


STAGE03C_MD = """\
## 03C - Bootstrap CI For Candidate Signals

Bootstrap CI is a diagnostic on validation predictions, not a replacement for
the pre-registered gates. The default only bootstraps rows already tagged
`candidate_signal`.
"""


STAGE03C_CODE = r"""\
if RUN_03C_BOOTSTRAP_CI:
    if not OUTPUT_FILES["summary"].exists():
        raise FileNotFoundError(f"Notebook 03 summary is missing: {OUTPUT_FILES['summary']}")
    if not OUTPUT_FILES["pooled"].exists():
        raise FileNotFoundError(f"Notebook 03 pooled output is missing: {OUTPUT_FILES['pooled']}")
    if not OUTPUT_FILES["per_ticker"].exists():
        raise FileNotFoundError(f"Notebook 03 per-ticker output is missing: {OUTPUT_FILES['per_ticker']}")
    pooled = pd.read_csv(OUTPUT_FILES["pooled"])
    per_ticker = pd.read_csv(OUTPUT_FILES["per_ticker"])
    summary = pd.read_csv(OUTPUT_FILES["summary"])
    if summary["signal_strength_tag"].eq("candidate_signal").any() or BOOTSTRAP_CI_FULL_PANEL:
        summary = bootstrap_candidate_rows(summary)
        selection = build_selection_record(summary)
        summary.to_csv(OUTPUT_FILES["summary"], index=False)
        with OUTPUT_FILES["selection"].open("w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2)
        write_run_manifest(pooled, per_ticker, summary)
        backup_notebook03_outputs("completed_03C_bootstrap_ci")
        print("Updated bootstrap CI columns in", OUTPUT_FILES["summary"])
        display(summary)
        print(json.dumps(selection, indent=2))
    else:
        print("No candidate_signal rows found; bootstrap CI not run.")
else:
    print("RUN_03C_BOOTSTRAP_CI is False; bootstrap CI not run.")
"""


H0_APPENDIX_MD = """\
## Diagnostic-Only H0 Window 32 Context - Non-Selecting Appendix

This appendix is intentionally outside Notebook 03 selection.

Rules:

- `RUN_H0_DIAGNOSTIC_NOTE` defaults to `False`.
- If enabled, this cell reads only existing H0 CSV output.
- It does not load raw data.
- It does not fit models.
- It does not write `notebook03_pooled.csv`, `notebook03_summary.csv`, or
  `notebook03_validation_selection.json`.
- It does not enter `basic_gate`, `lcb_eligible`, `signal_strength_tag`, or
  `selected_branches`.
- It reads only local H0 outputs under `/content/diagnostic_h0_tabular_sweep`.
  If H0 was run in a separate runtime, copy the H0 CSV into that local directory
  before enabling this appendix. Notebook 03 does not use MyDrive fallback paths.

Recommended operation:

1. Run 03S/03A/03B/03C as needed and let Notebook 03 write its selection JSON.
2. If you want H0 context in the same Colab session, set
   `RUN_H0_DIAGNOSTIC_NOTE=True` and run only this appendix cell afterward.

The purpose is narrative context only: H0 window 32 remains diagnostic-only and
non-selecting because the reported lift was below the H0 gate.

Recorded appendix result:

| source | model/profile | window_size | macro_f1_mean | baseline_macro_f1 | delta_macro_f1_vs_base | dummy_macro_f1_mean | delta_macro_f1_vs_dummy_mean | positive_ticker_count | top_ticker_gain_share | interpretation | confirmation_status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| diagnostic_h0_summary.csv, part1_window_sweep | lightgbm/profile_B | 32 | 0.529270 | 0.527005 | 0.002265 | 0.499602 | 0.029668 | 5 | 0.250014 | noise_level_positive_no_action | not_selected_for_confirmation |

This recorded appendix result does not replace the official Stage 0
`window_size=20` candidate and does not alter Notebook 03 `selected_branches`.
"""


H0_APPENDIX_CODE = r"""\
def find_h0_appendix_file():
    for path in H0_OUTPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "RUN_H0_DIAGNOSTIC_NOTE=True requires one existing H0 output file. Missing checked paths: "
        + "; ".join(str(path) for path in H0_OUTPUT_CANDIDATES)
    )


def h0_appendix_view(frame):
    window_column = "window_size" if "window_size" in frame.columns else None
    if window_column is None:
        raise ValueError("H0 appendix file has no window_size column.")
    mask = frame[window_column].astype(int).eq(H0_DIAGNOSTIC_WINDOW)
    for column, value in (
        ("label_config", "h03_bps1p5"),
        ("feature_set", "price_volume_time"),
    ):
        if column in frame.columns:
            mask &= frame[column].eq(value)
    if "model" in frame.columns:
        mask &= frame["model"].eq("lightgbm")
    view = frame.loc[mask].copy()
    preferred_columns = [
        "diagnostic_name",
        "part",
        "round",
        "model",
        "profile",
        "label_config",
        "feature_set",
        "window_size",
        "macro_f1_mean",
        "baseline_macro_f1",
        "delta_macro_f1_vs_base",
        "dummy_macro_f1_mean",
        "delta_macro_f1_vs_dummy_mean",
        "positive_ticker_count",
        "top_ticker_gain_share",
        "interpretation",
        "confirmation_status",
    ]
    columns = [column for column in preferred_columns if column in view.columns]
    return view[columns] if columns else view


if RUN_H0_DIAGNOSTIC_NOTE:
    h0_path = find_h0_appendix_file()
    h0_frame = pd.read_csv(h0_path)
    h0_view = h0_appendix_view(h0_frame)
    if h0_view.empty:
        raise ValueError(f"No H0 window={H0_DIAGNOSTIC_WINDOW} row found in {h0_path}.")
    NOTEBOOK03_STATE["h0_cross_window_appendix"] = "loaded_read_only"
    print("Loaded H0 appendix read-only from:", h0_path)
    print(
        "H0 appendix is diagnostic-only and non-selecting. "
        "It does not update Notebook 03 gates, tags, selected_branches, or selection JSON."
    )
    display(h0_view)
else:
    NOTEBOOK03_STATE["h0_cross_window_appendix"] = "not_loaded"
    print(
        "H0 cross-window context disabled. Set RUN_H0_DIAGNOSTIC_NOTE=True to view "
        "read-only local H0 window=32 rows after 03 selection is written. "
        "This remains diagnostic-only and non-selecting."
    )
"""


INTERPRETATION_MD = """\
## Interpretation Boundary

Notebook 03 is `validation_only`.

Required interpretation points after a run:

- whether the official candidate produced any `candidate_signal`;
- every model's `signal_strength_tag`;
- pooled and per-ticker results;
- dummy baseline deltas;
- one-class collapse diagnostics;
- sample counts;
- runtime/failure reasons;
- Stage 0 caveats: marginal window-20 advantage, LogReg/LightGBM near-tie,
  LogReg convergence status, and validation-selection bias;
- explicit statement that holdout/test remains closed.

Allowed wording:

```text
This validation-only screen found no candidate_signal under the current Stage 0
configuration. This is a valid negative result and does not justify post-hoc
model expansion inside Notebook 03.
```

```text
This validation-only screen found candidate_signal for the listed branch under
fixed defaults. This does not authorize holdout/test evaluation or post-hoc
tuning; it only identifies a branch for controlled follow-up design.
```

Forbidden wording:

```text
The model works.
The model beats the market.
The best model is ready for holdout.
The weak delta is still promising because it is the best row.
```
"""


def dedent_code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def validate_code_cells(nb: nbformat.NotebookNode) -> None:
    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"cell_{index}")


def build_notebook() -> nbformat.NotebookNode:
    source = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    data_loading_code = source.cells[4].source.replace(
        "RUN_ANY_STAGE = bool(RUN_STAGE0S or RUN_STAGE0A1 or RUN_STAGE0A2 or RUN_STAGE0B)",
        "RUN_ANY_STAGE = bool(RUN_03S_SCHEMA_SMOKE or RUN_03A_TABULAR_PANEL or RUN_03B_SEQUENCE_PANEL)",
    ).replace(
        'print("All RUN_STAGE0* switches are False; data loading skipped.")',
        'print("All Notebook 03 run switches are False; data loading skipped.")',
    )

    nb = new_notebook()
    nb.metadata = source.metadata
    nb.cells = [
        new_markdown_cell(TITLE_MD),
        new_code_cell(source.cells[1].source),
        new_code_cell(dedent_code(CONFIG_CODE)),
        new_markdown_cell(source.cells[3].source.replace("Stage 0", "Notebook 03")),
        new_code_cell(data_loading_code),
        new_markdown_cell(source.cells[5].source),
        new_code_cell(source.cells[6].source),
        new_markdown_cell("## Notebook 03 Base Model Helpers\n\nThis section copies active Stage 0 metric, dataset, tabular, and Stage 0 sequence helper definitions. The following cell overrides only the Notebook 03 model-family layer."),
        new_code_cell(source.cells[8].source),
        new_markdown_cell("## Notebook 03 Model-Family Helpers\n\nThis layer adds the fixed Notebook 03 model panel, row schema, failure rows, signal tags, selection JSON, bootstrap diagnostics, and sample-alignment assertions."),
        new_code_cell(dedent_code(NOTEBOOK03_HELPERS_CODE)),
        new_markdown_cell(STAGE03S_MD),
        new_code_cell(dedent_code(STAGE03S_CODE)),
        new_markdown_cell(STAGE03A_MD),
        new_code_cell(dedent_code(STAGE03A_CODE)),
        new_markdown_cell(STAGE03B_MD),
        new_code_cell(dedent_code(STAGE03B_CODE)),
        new_markdown_cell(STAGE03C_MD),
        new_code_cell(dedent_code(STAGE03C_CODE)),
        new_markdown_cell(H0_APPENDIX_MD),
        new_code_cell(dedent_code(H0_APPENDIX_CODE)),
        new_markdown_cell(INTERPRETATION_MD),
    ]

    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    validate_code_cells(nb)
    return nb


def main() -> None:
    nb = build_notebook()
    TARGET_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, TARGET_NOTEBOOK)
    print(f"Wrote {TARGET_NOTEBOOK.relative_to(PROJECT_ROOT)} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
