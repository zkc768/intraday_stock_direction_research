# ml_utils Library Construction Plan v2

> Purpose: Lightweight PyTorch utility library for high-frequency stock direction binary classification. The MVP sprint targets end-to-end pipeline on 5 stocks at 5-minute bars; later extensible to 80 NASDAQ stocks.
>
> v2 updates: Per-module reference tiers (reference implementation vs AI-authored), anti-spaghetti agent rules, per-module standard prompt templates.
>
> Authoritative hierarchy: when this plan conflicts with AGENTS.md, AGENTS.md takes precedence. If the conflict affects a function signature or test, stop and ask.

---

## 0. Design Principles and Scope

### 0.1 Core Principles

1. **No hardcoded data parameters**: stock count, bar frequency, window length, label horizon, feature column names, train ratio — all passed via config or arguments.
2. **Data layer and model layer decoupled**: `dataset.py` outputs a tensor interface transparent to models; models do not know or care about data origin.
3. **Temporal order is inviolable**: all splits, scaler fitting, and window generation must follow strict chronological order.
4. **Labels computed upfront**: labels are generated in the pandas stage and stored as a column; `Dataset.__getitem__` only performs index lookup.
5. **Notebook first, then extract**: each module is first validated in a notebook, then packaged as a `.py` file.
6. **Reference correct implementations, but do not copy code blocks**: leak-prevention, TCN causal convolution, DLinear decomposition — these error-prone parts must be informed by reference implementations. Trainer, checkpoint, metrics — lower risk, cleaner to write from scratch.
7. **Code you can maintain > 200 extra lines that work but you cannot understand.**

### 0.2 First Phase Scope

- Data: 5 stocks (CSCO, JPM, KO, MSFT, WMT), 5-minute bars
- Models: LSTM classifier (minimum baseline), TCN classifier, DLinear-style classifier
- Task: binary classification — label is whether the future k-bar average return > 0
- Evaluation: macro F1, balanced accuracy, confusion matrix, 4 baselines (see §5.2)

### 0.3 Out of First-Phase Scope

- No NLP / sentiment / text data pipeline
- No Transformer / GPT / FinGPT
- No trading strategy / reinforcement learning
- No PyTorch Lightning / Hydra integration
- No streaming inference / ONNX export
- No multi-GPU distributed training
- No attention pooling / mean pooling heads (use last time step)
- No callback / plugin / hook systems

---

## 1. Project Directory Structure

```text
hf_stock_clf/
├── data/                          User-managed, library does not touch
│   ├── CSCO.csv
│   ├── JPM.csv
│   ├── KO.csv
│   ├── MSFT.csv
│   └── WMT.csv
├── ml_utils/
│   ├── __init__.py
│   ├── config.py                  Config dataclass definitions
│   ├── seed.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── checkpoint.py
│   ├── metrics.py
│   └── models/
│       ├── __init__.py
│       ├── lstm_classifier.py
│       ├── tcn_classifier.py
│       └── dlinear_classifier.py
├── notebooks/
│   ├── 01_smoke_test_single_stock.ipynb
│   ├── 02_pooled_5_stocks.ipynb
│   └── 03_model_comparison.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_seed.py
│   ├── test_metrics.py
│   ├── test_dataset_leakage.py
│   ├── test_label_generation.py
│   ├── test_window_boundaries.py
│   ├── test_checkpoint.py
│   └── test_models_shape.py
├── reference_excerpts/            Extracted from external repos, read-only reference
│   ├── ltsf_data_loader.py        from cure-lab/LTSF-Linear
│   ├── pytorch_tcn_core.py        from paul-krug/pytorch-tcn
│   └── ltsf_dlinear_model.py      from cure-lab/LTSF-Linear
├── checkpoints/                   gitignore
├── docs/
│   └── ml_utils_construction_plan_v2.md
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 2. Module Reference Strategy (Key Change)

Instead of producing a review markdown for each of 5 repositories, we tier modules by "reference value" and operate precisely.

### 2.1 Tier 1 — Must Read Reference Implementation

These modules are easy to get wrong when written from scratch and hard to self-diagnose. **Mandatory: download reference files to `reference_excerpts/`; the agent must read them before writing code.**

| Module | Reference Source | Specific File | What to Learn |
|--------|-----------------|---------------|---------------|
| `dataset.py` | `cure-lab/LTSF-Linear` | `data_provider/data_loader.py` (`Dataset_Custom` class), `data_provider/data_factory.py` | Train-only scaler fit, border1s/border2s time-split index pattern, within-stock window construction |
| `models/tcn_classifier.py` | `paul-krug/pytorch-tcn` | `pytorch_tcn/tcn.py` (`TemporalBlock`, `Chomp1d`, `TCN`) | Causal convolution padding + chomp pairing, exponential dilation growth, residual connection structure |
| `models/dlinear_classifier.py` | `cure-lab/LTSF-Linear` | `models/DLinear.py` (`moving_avg`, `series_decomp`, `Model`) | Two-end replicate padding for moving average, trend/seasonal decomposition, individual vs shared linear |

### 2.2 Tier 2 — Reference Structure, No Code Copying

Scan the README and entry files to understand design patterns; writing from scratch is cleaner.

| Module | Reference Source | What to Scan |
|--------|-----------------|--------------|
| `trainer.py` | `victoresque/pytorch-template` | `base/base_trainer.py` — monitor / best checkpoint / early stop field design; do not copy implementation |
| `models/lstm_classifier.py` | `Yutsuro/pytorch-time-series-classification` | `tisc/modules/LSTM.py` — "LSTM + LayerNorm + take last time step" simple composition |
| `checkpoint.py` | `victoresque/pytorch-template` | `base/base_trainer.py` `_save_checkpoint` — dict field inventory |

### 2.3 Tier 3 — Write Directly, Not Worth Mining

| Module | Direct Reference |
|--------|-----------------|
| `seed.py` | PyTorch official reproducibility docs |
| `metrics.py` | sklearn `f1_score`, `balanced_accuracy_score`, `confusion_matrix`, `DummyClassifier` docs |
| `config.py` | Python stdlib `dataclasses` docs |

### 2.4 Deprecated Actions

- ~~Have the agent analyze each of 5 repos and produce a review markdown~~ — too heavy, most reviews are low value
- ~~Use `thuml/Time-Series-Library` as a mining target~~ — Tier 1 already has LTSF-Linear; TSLib's classification branch has too many issues
- ~~Clone all 5 repos locally~~ — only need specific files from the first two

### 2.5 Actual Operating Steps

```text
1. Clone two repos into temporary directories:
   - cure-lab/LTSF-Linear
   - paul-krug/pytorch-tcn

2. Copy 3 files to project reference_excerpts/, renaming:
   - LTSF-Linear/data_provider/data_loader.py → reference_excerpts/ltsf_data_loader.py
   - LTSF-Linear/models/DLinear.py            → reference_excerpts/ltsf_dlinear_model.py
   - pytorch-tcn/pytorch_tcn/tcn.py           → reference_excerpts/pytorch_tcn_core.py

3. Add license header comment at the top of each file, stating origin and original license

4. Delete the two temporary clone directories

5. reference_excerpts/ is read-only; our own code must never import from it
```

---

## 3. Configuration Management

A single `config.py` centrally defines all dataclass configs, avoiding scattered literal values. No Hydra or OmegaConf.

### 3.1 `ml_utils/config.py`

Define the following dataclasses:

```python
@dataclass
class DataConfig:
    tickers: list[str]              # e.g. ["CSCO", "JPM", "KO", "MSFT", "WMT"]
    data_dir: str
    timestamp_col: str              # default "timestamp"
    price_col: str                  # column for computing returns, default "close"
    feature_cols: list[str]
    bars_per_day: int               # for 5-min US regular session, approx 78
    train_ratio: float              # default 0.7
    val_ratio: float                # default 0.15
    timezone_policy: str            # "naive" or "utc", default "naive"; used by dataset.py for Stage 1 schema validation

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("DataConfig.tickers must be non-empty")
        if not self.feature_cols:
            raise ValueError("DataConfig.feature_cols must be non-empty")
        if not (0 < self.train_ratio < 1):
            raise ValueError(f"DataConfig.train_ratio must be in (0,1), got {self.train_ratio}")
        if not (0 < self.val_ratio < 1):
            raise ValueError(f"DataConfig.val_ratio must be in (0,1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1, got "
                f"{self.train_ratio} + {self.val_ratio}"
            )
        if self.timezone_policy not in {"naive", "utc"}:
            raise ValueError(f"timezone_policy must be 'naive' or 'utc', got {self.timezone_policy!r}")
        if self.bars_per_day <= 0:
            raise ValueError(f"bars_per_day must be > 0, got {self.bars_per_day}")

@dataclass
class WindowConfig:
    window_size: int                # input window length, e.g. 60 bars
    label_horizon_k: int            # future k bars, e.g. 12 or 24
    stride: int                     # default 1

@dataclass
class TrainConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip: float | None
    early_stop_patience: int
    monitor_metric: str             # default "val_macro_f1"
    monitor_mode: str               # "max" or "min"
    device: str                     # "cuda" or "cpu"
    seed: int

@dataclass
class ModelConfig:
    name: str                       # "lstm" / "tcn" / "dlinear"
    params: dict                    # model-specific hyperparameters
```

**All dataclasses must implement `__post_init__`** (consistent with AGENTS.md §6.5).

Minimum validation:

- `WindowConfig` checks `window_size > 0`, `label_horizon_k > 0`, `stride > 0`
- `TrainConfig` checks `batch_size > 0`, `num_epochs > 0`, `learning_rate > 0`, `monitor_mode in {"max", "min"}`, `device in {"cpu", "cuda"}`

Invalid values must immediately `raise ValueError` with an error message specifying the field name and actual value.

**Forbidden behavior**: any literal such as `78`, `12`, `0.7`, `["CSCO", ...]` appearing inside `dataset.py` / `trainer.py`.

---

## 4. Agent Operating Contract (Anti-Spaghetti Rules)

**Before each agent coding session, prepend the following as a prompt prefix.**

### 4.1 Standard Prompt Prefix (Copy Directly)

```text
You are implementing one module of a PyTorch financial time series library.
Strict rules — violating any of these means stop and ask:

[Scope]
1. Implement exactly one file: {target_file}
2. Do not create, modify, or touch any other file.
3. Do not add convenience scripts, examples, or README updates.
4. If you think another file needs to change, stop and ask.

[Imports]
5. Allowed: torch, torch.nn, torch.optim, torch.utils.data, numpy, pandas,
   sklearn.preprocessing, sklearn.metrics, dataclasses, typing, pathlib,
   random, os. Module-specific allowances are in the spec below.
6. Forbidden: pytorch_lightning, hydra, omegaconf, wandb, tensorboardX,
   any *_logger, any optimizer libraries beyond torch.optim.
7. If you think you need a new import, stop and ask.

[Line budget]
8. Hard cap on file size: {line_budget} lines.
9. Going over budget means you are over-engineering. Refactor or ask.

[Tensor shape discipline]
10. Every function that takes or returns tensors must document shapes in
    the docstring. Format: "x: torch.Tensor of shape (batch, seq_len, features)"
11. Add shape assertions at function entry for non-trivial transforms:
    assert x.dim() == 3, f"Expected 3D, got {x.shape}"

[No future code]
12. Forbidden patterns:
    - TODO comments
    - "if config.use_xxx:" branches where use_xxx is not in current config
    - Abstract base classes with only one concrete subclass
    - Callback / hook / plugin systems
    - Empty try/except blocks
    - "for future extensibility" comments
13. If the spec does not require it, do not build it.

[No silent fixes]
14. If you encounter a bug or unexpected behavior, do not silently work around it.
    Stop, describe the issue, and wait for my decision.
15. Do not catch and ignore exceptions. Do not add "safety" try/except blocks.
    Let things fail loudly.

[Reference handling]
16. For modules with a reference file in reference_excerpts/:
    a. First, read the reference file in full.
    b. Summarize in 3-5 bullet points what you learned.
    c. Only then write your own implementation following our spec.
    d. Do not copy code blocks. Reimplement in our naming convention.
    e. In a comment at the top of your file, list which ideas came from where.

[API verification]
17. Before using any non-trivial library API, verify the function signature
    against the version pinned in requirements.txt. Do not rely on memory.
    Examples of APIs that change across versions:
    - sklearn.metrics signatures
    - torch.optim variants (RAdam, NAdam availability)
    - pandas resample / groupby behavior

[Test-first]
18. Write tests/{test_file} FIRST based on the spec.
19. Show me the tests. Wait for approval.
20. Only after approval, implement the module.
21. All tests must pass without modifying the tests. If a test seems wrong,
    stop and ask — do not "fix" the test.
```

### 4.2 Post-Completion Self-Review Process

**Key: use a brand-new agent session for the review** to avoid motivated reasoning from the coding session.

```text
You are doing a code review of the attached file: {target_file}
You did not write this code. Be critical.

Check specifically for:
1. Hardcoded values (numbers, lists, paths) that should be parameters
2. Hidden assumptions about data shape, frequency, or stock count
3. Places where train data could leak into val/test
4. Unused imports or dead code
5. Comments that don't match what the code actually does
6. Off-by-one errors in window/index computation
7. Tensor shape mismatches between docstring and implementation
8. Silent except blocks or defensive code without justification
9. Code paths that are unreachable given current configs

Output format:
- One issue per line, prefixed by line number
- Severity: BLOCKER / WARNING / NIT
- Do not propose fixes. Just list issues.
```

### 4.3 Process Overview

```text
Per module:
1. Open NEW agent session (fresh context)
2. Paste: standard prefix (§4.1) + module spec (§5.x) + reference file if applicable
3. Agent writes tests first → user approves
4. Agent writes implementation
5. Run tests locally → fix failures
6. Open ANOTHER NEW agent session
7. Paste self-review prompt (§4.2) + the implementation file
8. User reads review, fixes BLOCKER and WARNING issues manually or in a third session
9. Move to next module
```

---

## 5. Module Specifications

### 5.1 `ml_utils/seed.py`

- **Reference Tier**: Tier 3 (write directly)
- **Line Budget**: 50 lines
- **Test File**: `tests/test_seed.py`

**Responsibility**: Global randomness control.

**Single function**:

```python
def seed_everything(seed: int, deterministic: bool = False) -> None
```

**Implementation notes**:

- Set `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- Set `os.environ["PYTHONHASHSEED"]`
- When `deterministic=True`:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True, warn_only=True)`
- Docstring must state: deterministic mode does not guarantee full reproducibility across PyTorch versions, platforms, or CPU/GPU

**Not implemented**:

- No DataLoader `worker_init_fn` wrapper
- Returns nothing

**Acceptance criteria**:

- Same seed produces identical `torch.randn(5)` outputs on consecutive calls
- Different seeds produce different outputs

---

### 5.2 `ml_utils/metrics.py`

- **Reference Tier**: Tier 3 (write directly)
- **Line Budget**: 250 lines
- **Test File**: `tests/test_metrics.py`

**Responsibility**: Evaluation metric computation and baseline generation. All functions are pure computation with no side effects.

**Function inventory**:

```python
def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict
```
Returns a dict containing: `accuracy`, `macro_f1`, `balanced_accuracy`, `precision_macro`, `recall_macro`, `confusion_matrix` (2x2 numpy array), `classification_report` (dict form).

```python
def dummy_baseline_metrics(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    strategy: str,
    random_state: int,
) -> dict
```

- Uses `sklearn.dummy.DummyClassifier`
- MVP only implements `"stratified"` and `"prior"` strategies
- `DummyClassifier.fit(X=zeros, y=y_train)`, `y_eval` used for evaluation; fitting on y_eval is forbidden

```python
def always_predict_baseline_metrics(
    y_eval: np.ndarray,
    constant_label: int,
) -> dict
```

- Predicts all `constant_label` (0 or 1), returns metrics dict
- `constant_label = 1` corresponds to always_up; `constant_label = 0` corresponds to always_down

```python
def compute_baseline_table(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    n_stratified_seeds: int = 10,
) -> pd.DataFrame
```

Fixed 4 baseline rows (consistent with AGENTS.md §2.4):

| baseline_name    | Implementation |
|------------------|---------------|
| dummy_stratified | Run n_stratified_seeds random_states, return mean ± std |
| dummy_prior      | Run once (deterministic) |
| always_up        | constant_label=1 |
| always_down      | constant_label=0 |

Returns a DataFrame with columns at minimum: `baseline_name`, `macro_f1_mean`, `macro_f1_std`, `balanced_accuracy_mean`, `balanced_accuracy_std`, `confusion_matrix`. For deterministic baselines, std columns are 0 or NaN (documented in docstring).

```python
def format_metrics_table(
    model_metrics: dict[str, dict],
    baseline_table: pd.DataFrame,
) -> pd.DataFrame
```

Merges model metrics and baseline table. **Must output a `delta_macro_f1_vs_dummy` column**, defined as:

```
delta_macro_f1_vs_dummy = model_macro_f1 - dummy_stratified_macro_f1_mean
```

Returns a pd.DataFrame with column order determined by the notebook 03 spec.

**Implementation notes**:

- All metric computation uses sklearn, no custom implementations
- `confusion_matrix` always uses `labels=[0, 1]` to avoid shape changes when a class is absent
- `classification_report` uses `zero_division=0`

**Forbidden behavior**:

- No direct printing; functions return strings or dicts
- No plotting (confusion matrix visualization is done in notebooks)

**Acceptance criteria**:

- Given fixed `y_true`, `y_pred`, output values match hand computation
- Baseline functions run without errors on extreme class-imbalance data

---

### 5.3 `ml_utils/dataset.py`

- **Reference Tier**: Tier 1 (must read reference implementation)
- **Reference File**: `reference_excerpts/ltsf_data_loader.py`
- **Line Budget**: 500 lines
- **Test Files**: `tests/test_dataset_leakage.py`, `tests/test_label_generation.py`, `tests/test_window_boundaries.py`

**This is the most important module in the entire library. Write tests in full before implementing.**

#### 5.3.1 Design Assumptions and Staged Schema (consistent with AGENTS.md §3.4)

Input data format: one DataFrame per stock, keyed by ticker as `dict[str, pd.DataFrame]`.

Schema validation is executed in three stages. Each stage must be explicitly validated before proceeding to the next; failures raise `ValueError` specifying ticker, column name, and row number.

**Stage 1 — Raw input (before label generation)**:

- `timestamp_col` satisfies `pandas.api.types.is_datetime64_any_dtype`
- Timezone behavior controlled by `DataConfig.timezone_policy`:
  - `"naive"` → `timestamp_col.dt.tz is None`
  - `"utc"` → tz-aware and normalized to UTC
- Timestamps strictly increasing per ticker, no duplicates
- `timestamp_col`, `price_col`, and all `feature_cols` columns exist and have **no NaN**
- `price_col` is strictly positive
- `volume` column (if used) is non-negative
- All `feature_cols` columns are numeric dtype
- Technical indicators must be precomputed by the caller in the data preparation stage

**Stage 2 — Labeled frame (after label column added, before trim)**:

- All Stage 1 constraints still hold
- New `label` column present; NaN is allowed, but **only** in the last k rows per ticker
- No fillna, no drop NaN

**Stage 3 — Post-trim frame (after trim, before window construction)**:

- All Stage 2 constraints still hold
- Label NaN may now also come from split boundary trim and cross-trading-day horizon filtering
- When constructing windows, all starting points where label==NaN must be skipped

#### 5.3.2 Functions and Classes

**(1) Label Generation**

```python
def make_binary_labels_from_future_avg_return(
    df: pd.DataFrame,
    price_col: str,
    k: int,
) -> pd.DataFrame
```

The formula is strictly fixed as arithmetic average (consistent with AGENTS.md §2.2.1; variants are forbidden):

1. Bar-to-bar return: `r_t = (price_{t+1} - price_t) / price_t`
2. Future k returns **arithmetic average**: `future_avg_r_t = mean(r_{t+1}, ..., r_{t+k})`
3. Label: `label_t = 1 if future_avg_r_t > 0 else 0`

Class semantics (library-wide unified):

- `class 0 = "non_up"` (`future_avg_r_t <= 0`, includes decline and flat)
- `class 1 = "up"` (`future_avg_r_t > 0`)

Precise boundaries:

- `future_avg_r_t == 0` is assigned to class 0
- The last k rows have label as NaN; **these NaN are preserved**, later handled by `trim_labels_at_split_boundary`; fillna is forbidden
- **Each stock is processed independently**

Forbidden variants:

- Do not implement geometric average `(price_{t+k}/price_t)^(1/k) - 1` as an option
- Do not implement total return `(price_{t+k} - price_t)/price_t` as an option
- Do not threshold labels into three classes (up / flat / down)
- To switch formulas → stop and ask

**(2) Time Split**

```python
def make_time_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    timestamp_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

- Sort by `timestamp_col` ascending
- Split by row count ratio
- Return three DataFrames, non-overlapping rows, union equals original df
- **Each stock is processed independently**

**(3) Scaler Fitting**

```python
def fit_scaler_on_train(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    scaler_type: str = "standard",
) -> sklearn.preprocessing.TransformerMixin
```

- `scaler_type` supports `"standard"` (default) and `"minmax"`
- **Must fit on the merged train_df**: merge all stocks' train segments before fitting; a single global scaler is shared
- Do not modify the input df in place

**(4) Apply Scaler**

```python
def transform_split(
    df: pd.DataFrame,
    scaler: sklearn.preprocessing.TransformerMixin,
    feature_cols: list[str],
) -> pd.DataFrame
```

- Returns a new DataFrame (no in-place modification)
- Only transforms `feature_cols`; other columns are preserved

**(5) Boundary Handling**

```python
def trim_labels_at_split_boundary(
    df: pd.DataFrame,
    label_horizon_k: int,
) -> pd.DataFrame
```

- Sets the label of the last k rows to NaN (marks boundary-crossing samples as invalid)
- **Must be applied per-ticker**, not after concatenation
- Occurs after `make_time_splits` and before `WindowedClassificationDataset`

**(6) Windowed Dataset Class**

```python
class WindowedClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        ticker_col: str,
        timestamp_col: str,
        window_size: int,
        label_horizon_k: int,
        stride: int = 1,
    ): ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
```

**Core logic**:

- Initialization only builds the index, no feature computation
- For each ticker, generates valid window starting points:
  - Starting point t must satisfy `t + window_size - 1 + label_horizon_k < len(ticker_df)`
  - Label at starting point t is not NaN
  - All time steps within the window belong to the same ticker
- Merges all tickers' valid starting points into a list of `(ticker, local_start_idx)`
- `__getitem__` returns:
  - `x`: `torch.FloatTensor` shape `(window_size, num_features)`
  - `y`: `torch.LongTensor` scalar
- Internally pre-converts each ticker's feature matrix and label array to numpy cache

#### 5.3.3 Calling Order

```text
1. Load each ticker's raw df, sort by timestamp
2. For each ticker: make_binary_labels_from_future_avg_return
3. For each ticker: make_time_splits
4. For each ticker, each split: trim_labels_at_split_boundary
5. Concatenate all tickers' train into train_df; same for val and test
6. fit_scaler_on_train(train_df, feature_cols)
7. transform_split applied to train_df, val_df, test_df
8. Construct one WindowedClassificationDataset from each df
9. Wrap as DataLoader: train shuffle=True, val/test shuffle=False
```

#### 5.3.4 Forbidden Behavior

- Do not compute returns or labels inside `__getitem__`
- Do not perform time split after merging all stocks
- Do not fit scaler before splitting
- No cross-ticker windows
- No cross-split-boundary windows
- Do not store raw DataFrame references inside the dataset

#### 5.3.5 Required Tests

Tests are distributed across three files (implementation sessions may modify all three, per AGENTS.md §7.1):

`tests/test_dataset_leakage.py`:

1. Time order test: each ticker's train max timestamp < val min timestamp < test min timestamp
2. Scaler isolation test: `scaler.mean_` equals the train segment mean, not train+val or full-dataset mean
3. Cross-ticker test: no window spans across tickers

`tests/test_label_generation.py`:

4. Label formula correctness: hand-computed future_avg_r matches function output to 1e-9 precision
5. Zero-value classification test: when `future_avg_r == 0`, label = 0 (class 0 / non_up)
6. Tail NaN test: each ticker's last k rows have label as NaN, **not filled by fillna**

`tests/test_window_boundaries.py`:

7. Boundary invalid label test (跨边界 invalid 标注测试): label-horizon samples that cross split boundaries are marked as label NaN by `trim_labels_at_split_boundary`, and `WindowedClassificationDataset` must not generate windows from those NaN label starting points.
8. **Cross-trading-day window test**: `test_no_window_crosses_trading_day` — all timestamps within any window have identical `.dt.date`
9. **Cross-trading-day label horizon test**: `test_no_label_horizon_crosses_trading_day` — the `date(t)` of starting point t equals `date(t + window_size - 1 + label_horizon_k)`
10. Stage 1 schema validation: construct illegal raw DataFrames (tz mismatch, price contains 0, timestamp non-increasing, feature contains NaN) and verify `raise ValueError` with field name in message
11. Stage 2 schema validation: label NaN appearing in non-tail-k positions raises error
12. Stage 3 schema validation: window construction skips label==NaN starting points


---

### 5.4 `ml_utils/models/lstm_classifier.py`

- **Reference Tier**: Tier 2 (scan reference)
- **Reference Scan**: `Yutsuro/pytorch-time-series-classification/tisc/modules/LSTM.py`
- **Line Budget**: 150 lines
- **Test File**: `tests/test_models_shape.py`

**Class signature**:

```python
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Architecture**:

- `nn.LSTM(..., batch_first=True, dropout=dropout if num_layers > 1 else 0)`
- LayerNorm on last hidden state
- `nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)`
- Takes last time step `x[:, -1, :]`

**Input/output conventions**:

- Input: `(batch, seq_len, input_size)`
- Output logits: `(batch, num_classes)`

**Forbidden behavior**:

- No built-in softmax / sigmoid
- First version does not implement attention pooling or mean pooling
- Does not read any global configuration

**Acceptance criteria**:

- Given `(32, 60, 7)` input, output shape is `(32, 2)`
- Backward pass completes without error

---

### 5.5 `ml_utils/trainer.py`

- **Reference Tier**: Tier 2 (scan reference)
- **Reference Scan**: `victoresque/pytorch-template/base/base_trainer.py` (field design only)
- **Line Budget**: 350 lines
- **Test File**: `tests/test_trainer_smoke.py`

**Functions and class**:

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float | None = None,
) -> dict
```
Returns `{"loss": float, "accuracy": float}`.

```python
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]
```
Returns `(metrics_dict, y_true, y_pred)`.

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        device: str,
        checkpoint_dir: str,
        monitor_metric: str = "val_macro_f1",
        monitor_mode: str = "max",
        early_stop_patience: int = 10,
        grad_clip: float | None = None,
    ): ...

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> dict
```

**Implementation notes**:

- One log line per epoch: `epoch X | train_loss=... | val_loss=... | val_macro_f1=... | best=...`
- Best checkpoint saved to `checkpoint_dir/best.pt`
- Last checkpoint saved to `checkpoint_dir/last.pt`
- ReduceLROnPlateau steps on monitor metric; other schedulers step per epoch

**Forbidden behavior**:

- No built-in AMP, no built-in multi-GPU
- No tensorboard / wandb integration
- Do not compute baselines inside the trainer

**Acceptance criteria**:

- On a toy dataset, fitting for 10 epochs shows monotonically decreasing loss
- Early stopping triggers normal exit
- Best checkpoint file exists and can be loaded

---

### 5.6 `ml_utils/checkpoint.py`

- **Reference Tier**: Tier 2 (scan reference)
- **Reference Scan**: `victoresque/pytorch-template/base/base_trainer.py` `_save_checkpoint`
- **Line Budget**: 150 lines
- **Test File**: `tests/test_checkpoint.py`

**Functions**:

```python
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_metric: float,
    extra: dict | None = None,
) -> None
```

Saved content:
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch": ...,
    "best_metric": ...,
    "rng_state": {"python", "numpy", "torch", "cuda"},
    "extra": ...,  # user can store scaler, feature_cols, label_schema, config dump
}
```

```python
def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: str = "cpu",
    weights_only: bool = False,
) -> dict
```

**Implementation notes**:

- Does not invent file naming conventions; caller passes full path

**Forbidden behavior**:

- Do not store raw data in the checkpoint
- Do not store the entire model object (only state_dict)

**Acceptance criteria**:

- After save → load, model output is identical to pre-save output

---

### 5.7 `ml_utils/models/tcn_classifier.py`

- **Reference Tier**: Tier 1 (must read reference implementation)
- **Reference File**: `reference_excerpts/pytorch_tcn_core.py`
- **Line Budget**: 200 lines
- **Test File**: `tests/test_models_shape.py`

**Class signature**:

```python
class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: list[int],         # e.g. [32, 32, 64, 64]
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_classes: int = 2,
        causal: bool = True,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Architecture**:

- Internal `TemporalBlock`: `Conv1d → Chomp1d → ReLU → Dropout → Conv1d → Chomp1d → ReLU → Dropout`, with residual connection
- Dilation grows exponentially per layer: layer i dilation = `2 ** i`
- padding = `(kernel_size - 1) * dilation`, paired with Chomp1d to trim right-side padding
- Classification head: `backbone → x[:, :, -1] → Linear(num_channels[-1], num_classes)`

**Input/output conventions**:

- External input: `(batch, seq_len, input_size)`
- Forward internally transposes → `(batch, input_size, seq_len)`
- Output logits: `(batch, num_classes)`

**Forbidden behavior**:

- No streaming / buffer / ONNX implementation
- First version only does causal
- No built-in weight_norm

**Acceptance criteria**:

- Given `(32, 60, 7)` input, output shape is `(32, 2)`
- Causal property verification: changing the last input time step affects output; changing any earlier time step also affects output

---

### 5.8 `ml_utils/models/dlinear_classifier.py`

- **Reference Tier**: Tier 1 (must read reference implementation)
- **Reference File**: `reference_excerpts/ltsf_dlinear_model.py`
- **Line Budget**: 200 lines
- **Test File**: `tests/test_models_shape.py`

**Class signature**:

```python
class DLinearClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int,
        num_classes: int = 2,
        moving_avg_kernel: int = 25,
        individual: bool = False,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Architecture**:

- `series_decomp`: `AvgPool1d(kernel_size=moving_avg_kernel, stride=1, padding=...)` extracts trend; original sequence minus trend gives seasonal
- Two branches: `Linear(seq_len, seq_len)` each processing trend and seasonal, sum gives `(batch, seq_len, input_size)`
- `individual=True`: each feature channel gets its own set of linear layers
- Classification head: `flatten → Linear(seq_len * input_size, num_classes)`
- Moving average boundary uses two-end replicate padding

**Input/output conventions**:

- Input: `(batch, seq_len, input_size)`
- Output logits: `(batch, num_classes)`

**Forbidden behavior**:

- Do not retain forecasting `pred_len` output
- Do not hardcode `moving_avg_kernel=25`
- Do not implement NLinear / Linear variants

**Acceptance criteria**:

- Given `(32, 60, 7)` input, output shape is `(32, 2)`
- Both `individual=True` and `False` run successfully

---

## 6. Implementation Order and Process

**Strictly follow this order. Each module gets its own independent agent session.**

| Step | Module / Task | Reference Tier | Line Budget | Estimate |
|------|---------------|----------------|-------------|----------|
| step 0   | Preparation: clone reference repos, extract required reference files into reference_excerpts/, then remove temporary clones | — | — | 20 min |
| step 0.3 | User fills docs/ENVIRONMENT.md with PYTHON_INTERPRETER and locks requirements.txt | — | — | 30 min |
| step 0.5 | Codex runs AGENTS.md §14.1 readiness audit as a read-only session and returns a report | — | — | 15 min |
| step 0.7 | Codex runs AGENTS.md §14.2 testing infrastructure setup, creating tests/conftest.py and pytest.ini | — | — | 45 min |
| step 1   | `config.py` | Tier 3 | 100 | 30 min |
| step 2   | `seed.py` | Tier 3 | 50 | 30 min |
| step 3   | `metrics.py` | Tier 3 | 250 | 1.5 h |
| step 4   | `dataset.py` (with full tests) | **Tier 1** | 500 | 4 h |
| step 5   | `models/lstm_classifier.py` | Tier 2 | 150 | 1 h |
| step 6   | `checkpoint.py` | Tier 2 | 150 | 1 h |
| step 7   | `trainer.py` | Tier 2 | 350 | 2.5 h |
| step 8   | **Notebook 01**: single-stock smoke test (CSCO) | — | — | 1.5 h |
| step 9   | **Notebook 02**: 5-stock pooled LSTM smoke report and boundary/trading-day leakage review | — | — | 1.5 h |
| step 10  | `models/tcn_classifier.py` | **Tier 1** | 200 | 2 h |
| step 11  | `models/dlinear_classifier.py` | **Tier 1** | 200 | 2 h |
| step 12  | **Notebook 03**: three-model + 4-baseline full comparison table | — | — | 1.5 h |

**Total estimate**: Approximately 21.5 hours, including 1.5 hours of preparation work.

### 6.1 Single-Module Standard Process

```text
For module M:

1. Open new agent session A1
2. Prompt = §4.1 standard prefix + §5.x module spec + (if Tier 1) reference file content
3. Have A1 write tests/test_M.py first
4. You review the tests, confirm all spec requirements are covered
5. Approve → A1 writes ml_utils/M.py
6. Run tests locally → all green → proceed to 7; red → have A1 fix, but tests must not be modified
7. Open new agent session A2 (fresh context)
8. Prompt = §4.2 self-review template + complete M.py content
9. You read the review, fix BLOCKER and WARNING items in a third session A3, or manually
10. Re-run tests → all green → module complete
11. Do not continue to the next module in A1; close A1, open B1 for the next module
```

### 6.2 MVP_YES — Sprint Scope Lock

This section defines the MVP sprint scope lock.

**MVP_YES**: This sprint is locked as MVP (Minimum Viable Product), LSTM-only end-to-end.

This sprint only advances the lstm 通道 (LSTM channel). 不允许写任何 tcn/dlinear 未来代码 (No TCN/DLinear future code is allowed during MVP).

**MVP implementation scope** (files to be created during this sprint):

- `ml_utils/config.py`
- `ml_utils/seed.py`
- `ml_utils/metrics.py`
- `ml_utils/dataset.py`
- `ml_utils/checkpoint.py`
- `ml_utils/models/lstm_classifier.py`
- `ml_utils/trainer.py`

**Deferred to Phase 1B**:

- `ml_utils/models/tcn_classifier.py`
- `ml_utils/models/dlinear_classifier.py`
- Corresponding tests for TCN and DLinear
- `reference_excerpts/pytorch_tcn_core.py`
- `reference_excerpts/ltsf_dlinear_model.py`

**Constraints**:

- Do not create tcn/dlinear files during MVP, including empty placeholders
- Do not add "for future tcn/dlinear use" branches in `trainer.py` or `config.py`
- Only after the W5.4 integration test passes can Phase 1B start

---

## 7. Leak-Prevention Sanity Checks (Must Run in Notebook 02)

### 7.1 Time Boundary Check
```python
for ticker in tickers:
    assert train[ticker]["timestamp"].max() < val[ticker]["timestamp"].min()
    assert val[ticker]["timestamp"].max() < test[ticker]["timestamp"].min()
```

### 7.2 Scaler Isolation Check
```python
np.testing.assert_allclose(
    scaler.mean_,
    train_df[feature_cols].mean().values,
    rtol=1e-5,
)
```

### 7.3 Boundary Invalid Label Check
```python
for ticker in tickers:
    n_nan = train_df[train_df["ticker"] == ticker]["label"].isna().sum()
    assert n_nan >= label_horizon_k
```


### 7.5 Window Ticker Consistency

Guaranteed internally by `WindowedClassificationDataset` assertions; notebook does not re-test.

---

## 8. Future Expansion Interface Reservations

To avoid rewrites when scaling to 80 NASDAQ stocks:

1. **Frequency-agnostic**: `bars_per_day` in config
2. **Stock-count-agnostic**: processing logic uses `dict[ticker, df]` rather than 5 fixed variables
3. **Feature-count-agnostic**: `input_size = len(feature_cols)` derived
4. **Window-length-agnostic**: `window_size` and `label_horizon_k` in config
5. **Scaler type switchable**: start with StandardScaler, add RobustScaler later if needed

When expanding, the **only new code needed**:

- A new data loader function to read 80 stocks' actual file format into `dict[str, pd.DataFrame]`
- Possibly a gradient accumulation option

---

## 9. requirements.txt Version Locking

```text
torch==2.X.X
numpy==1.X.X
pandas==2.X.X
scikit-learn==1.X.X
pytest==8.X.X
```

Fill in specific minor version numbers after determining them. **Use of `>=` or unspecified versions is forbidden.**

---

## 10. One-Sentence Summary

**Build end-to-end with 5 stocks; Tier 1 modules must read reference implementations, Tier 2 scan briefly, Tier 3 write directly; each module in its own agent session, tests first then implementation, then a fresh session for self-review; all data parameters go through config, all tensor shapes go in docstrings; the first-version completion criterion is notebook 03 outputting a three-model + 4-baseline comparison table with delta_macro_f1_vs_dummy, and boundary / trading-day leakage tests pass. This sprint is MVP_YES: LSTM channel only, Phase 1B adds TCN and DLinear.**

---

## Appendix A: Standard Prompt Templates (Copy Per Module)

Below is the complete prompt example for `dataset.py`; for other modules, substitute `{target_file}`, `{line_budget}`, and §5.x content accordingly.

````text
You are implementing one module of a PyTorch financial time series library.
Strict rules — violating any of these means stop and ask:

[Scope]
1. Implement exactly one file: ml_utils/dataset.py
2. Do not create, modify, or touch any other file (except tests/test_dataset_leakage.py,
   tests/test_label_generation.py, tests/test_window_boundaries.py).
3. Do not add convenience scripts, examples, or README updates.

[Imports]
Allowed: torch, torch.utils.data, numpy, pandas, pandas.api.types,
sklearn.preprocessing, dataclasses, typing, pathlib.
Forbidden: pytorch_lightning, hydra, wandb, anything not above.

[Line budget]
Hard cap: 500 lines for ml_utils/dataset.py.

[Tensor shapes]
Every function with tensors must document shapes in the docstring.
Add shape assertions on entry for non-trivial transforms.

[No future code]
No TODOs, no callback systems, no "for future extensibility" branches.

[No silent fixes]
If you hit a bug, stop and ask. No try/except to swallow errors.

[Reference]
First read the attached reference_excerpts/ltsf_data_loader.py.
Summarize what you learned about: (1) train-only scaler fit,
(2) border1s/border2s time split, (3) within-stock windowing.
Do not copy code blocks. Reimplement in our naming convention.
Add a header comment listing which ideas came from the reference.

[Spec]
{paste §5.3 here, including all 6 functions/classes and the calling order}

[Tests first]
Write the three test files FIRST. Show me. Wait for approval before
implementing dataset.py. All tests must pass without modifying tests.

[API verification]
Verify sklearn.preprocessing.StandardScaler API against scikit-learn 1.X.X
before using it.
````

Appendix A ends. Before each module, assemble the prompt using this template and start a new session.
