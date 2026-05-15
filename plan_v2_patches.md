# plan v2 同步 patches

> 配套 AGENTS.md v4（2026-05-14）。每条 patch 给出在 plan v2 中的位置、原文锚点、替换文本。
> 用法：在 `docs/ml_utils_construction_plan_v2.md` 中按【原文】定位段落，整段替换为【替换为】。
> 用 markdown 文本搜索 `【原文】` 行首关键词即可定位。

---

## Patch 1 — §5.3.2 (1) 标签生成

**位置**：plan v2 §5.3.2 "(1) 标签生成" 整段。

**【原文】**（在 plan v2 中查找以下段落）：

````
**(1) 标签生成**

```python
def make_binary_labels_from_future_avg_return(
    df: pd.DataFrame,
    price_col: str,
    k: int,
) -> pd.DataFrame
```
- 计算 bar-to-bar return：`r_t = (price_{t+1} - price_t) / price_t`
- 计算未来 k 个 return 的均值：`future_avg_r_t = mean(r_{t+1}, ..., r_{t+k})`
- 等价做法：`(price_{t+k} / price_t)^(1/k) - 1`，二选一在 docstring 说明
- 添加列 `label = (future_avg_r > 0).astype(int)`
- 最后 k 行 label 为 NaN，**保留这些 NaN**
- **每只股票单独调用**
````

**【替换为】**：

````
**(1) 标签生成**

```python
def make_binary_labels_from_future_avg_return(
    df: pd.DataFrame,
    price_col: str,
    k: int,
) -> pd.DataFrame
```

公式严格固定为算术平均（与 AGENTS.md §2.2.1 一致，禁止变体）：

1. bar-to-bar return：`r_t = (price_{t+1} - price_t) / price_t`
2. 未来 k 个 return 的**算术平均**：`future_avg_r_t = mean(r_{t+1}, ..., r_{t+k})`
3. label：`label_t = 1 if future_avg_r_t > 0 else 0`

类别语义（全库统一）：

- `class 0 = "non_up"`（`future_avg_r_t <= 0`，含下跌和平盘）
- `class 1 = "up"`（`future_avg_r_t > 0`）

精确边界：

- `future_avg_r_t == 0` 归入 class 0
- 最后 k 行 label 为 NaN，**保留这些 NaN**，后续由 `trim_labels_at_split_boundary` 处理；不许 fillna
- **每只股票单独调用**

禁止变体：

- 不许实现几何平均 `(price_{t+k}/price_t)^(1/k) - 1` 作为可选项
- 不许实现总收益 `(price_{t+k} - price_t)/price_t` 作为可选项
- 不许把 label 阈值化为三分类（up / flat / down）
- 要切换公式 → 停下问
````

---

## Patch 2 — §5.3.1 设计假设

**位置**：plan v2 §5.3.1 整段。

**【原文】**：

```
#### 5.3.1 设计假设

- 输入数据格式：每只股票一份 DataFrame，列至少包含 `timestamp`（已转 datetime 或可排序）、`close`、若干特征列
- 多股票：用户传入 `dict[str, pd.DataFrame]`，key 为 ticker
- 数据已清洗：无 NaN、无重复时间戳、按时间升序
```

**【替换为】**：

```
#### 5.3.1 设计假设与 schema 分阶段（与 AGENTS.md §3.4 一致）

输入数据格式：每只股票一份 DataFrame，key 为 ticker 的 `dict[str, pd.DataFrame]`。

schema 验证分三阶段执行，每阶段进入下一阶段前必须显式 validate，失败 raise ValueError 并指明 ticker、列名、行号。

**Stage 1 — Raw input（label 生成前）**：

- `timestamp_col` 满足 `pandas.api.types.is_datetime64_any_dtype`
- timezone 行为由 `DataConfig.timezone_policy` 控制：
  - `"naive"` → `timestamp_col.dt.tz is None`
  - `"utc"`   → tz-aware 且 normalized 到 UTC
- 每个 ticker 内 timestamp 严格递增、无重复
- `timestamp_col`、`price_col`、所有 `feature_cols` 列存在且**无 NaN**
- `price_col` 严格正数
- `volume` 列（若使用）非负
- 所有 `feature_cols` 列为数值 dtype
- 技术指标必须由调用方在数据准备阶段提前算好

**Stage 2 — Labeled frame（label 列加好后、trim 前）**：

- Stage 1 所有约束仍成立
- 新增 `label` 列；允许 NaN，**只允许**出现在每只 ticker 最后 k 行
- 不许 fillna、不许 drop NaN

**Stage 3 — Post-trim frame（trim 后、构造窗口前）**：

- Stage 2 所有约束仍成立
- label NaN 现在还可能来自 split 边界 trim 和跨交易日 horizon 过滤
- 构造窗口时所有 label==NaN 的起点必须被跳过
```

---

## Patch 3 — §3.1 DataConfig（含 __post_init__ 要求）

**位置**：plan v2 §3.1 的 `DataConfig` 整段。同时本 patch 在末尾追加 "全部 dataclass 必须实现 __post_init__" 的总要求。

**【原文】**：

```python
@dataclass
class DataConfig:
    tickers: list[str]              # ["CSCO", "JPM", "KO", "MSFT", "WMT"]
    data_dir: str                   # 数据文件所在目录
    timestamp_col: str              # 默认 "timestamp"
    price_col: str                  # 用于算 return 的列，默认 "close"
    feature_cols: list[str]         # 输入特征列名列表
    bars_per_day: int               # 5-min 美股常规盘约 78
    train_ratio: float              # 默认 0.7
    val_ratio: float                # 默认 0.15
    # test_ratio 自动 = 1 - train - val
```

**【替换为】**：

```python
@dataclass
class DataConfig:
    tickers: list[str]              # 例如 ["CSCO", "JPM", "KO", "MSFT", "WMT"]
    data_dir: str
    timestamp_col: str              # 默认 "timestamp"
    price_col: str                  # 用于算 return 的列，默认 "close"
    feature_cols: list[str]
    bars_per_day: int               # 5-min 美股常规盘约 78
    train_ratio: float              # 默认 0.7
    val_ratio: float                # 默认 0.15
    timezone_policy: str            # "naive" 或 "utc"，默认 "naive"，由 dataset.py 用于 Stage 1 schema 验证

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
```

并在 §3.1 末尾（在所有 dataclass 定义之后）追加：

```
**所有 dataclass 必须实现 `__post_init__`**（与 AGENTS.md §6.5 一致）。
最低验证：`WindowConfig` 检查 window_size / label_horizon_k / stride 均 > 0；
`TrainConfig` 检查 batch_size > 0、num_epochs > 0、learning_rate > 0、
monitor_mode ∈ {"max", "min"}、device ∈ {"cpu", "cuda"}。
非法值立即 raise ValueError，错误信息指出字段名和实际值。
```

---

## Patch 4 — §5.3.5 必须通过的测试

**位置**：plan v2 §5.3.5 整段。

**【原文】**：

```
#### 5.3.5 必须通过的测试

写在 `tests/test_dataset_leakage.py` 里：

1. **时间顺序测试**：每只 ticker 的 train 最大时间 < val 最小时间 < test 最小时间
2. **Scaler 不污染测试**：scaler.mean_ 应等于 train 的均值，而非 train+val 或全量的均值
3. **跨边界丢弃测试**：每只 ticker 的 train 末尾至少 k 行 label 应为 NaN
4. **跨 ticker 测试**：不存在跨 ticker 的窗口
5. **Shuffled label 检查**（最重要）：把 train label 随机打乱后训练，val macro F1 应接近 0.5
```

**【替换为】**：

```
#### 5.3.5 必须通过的测试

测试分布在三个文件中（实现会话允许同时修改三者，配套 AGENTS.md §7.1）：

`tests/test_dataset_leakage.py`：

1. 时间顺序测试：每只 ticker 的 train 最大时间 < val 最小时间 < test 最小时间
2. Scaler 不污染测试：`scaler.mean_` 等于 train 的均值，而非 train+val 或全量
3. 跨 ticker 测试：不存在跨 ticker 的窗口

`tests/test_label_generation.py`：

4. 标签公式正确性：用手算的 future_avg_r 与函数输出比对到 1e-9 精度
5. 零值归类测试：`future_avg_r == 0` 时 label = 0（class 0 / non_up）
6. 末尾 NaN 测试：每只 ticker 最后 k 行 label 为 NaN，**未被 fillna**

`tests/test_window_boundaries.py`：

7. 跨边界 invalid 标注测试：标签区间跨 split 边界的样本 label 被标注为 NaN（由 trim_labels_at_split_boundary 添加），且 Dataset 构造时不会以这些 NaN 起点生成窗口
8. **跨交易日窗口测试**：`test_no_window_crosses_trading_day` —— 任何窗口内部所有时间戳的 `.dt.date` 一致
9. **跨交易日 label horizon 测试**：`test_no_label_horizon_crosses_trading_day` —— 起点 t 的 `date(t)` 与 `date(t + window_size - 1 + label_horizon_k)` 一致
10. Stage 1 schema validation：构造非法 raw DataFrame（tz 不符、price 含 0、timestamp 不递增、feature 含 NaN），验证 raise ValueError 并消息含字段名
11. Stage 2 schema validation：label 在非末尾 k 行处出现 NaN 时 raise
12. Stage 3 schema validation：构造窗口时跳过 label==NaN 起点

**shuffled-label sanity check 不在 pytest 中实现**（与 AGENTS.md §13.2 一致）。
该检查放在 notebook 02 中作为人工可读的最终汇报数字，pytest 不重复。
```

---

## Patch 5 — §5.2 metrics.py 函数清单

**位置**：plan v2 §5.2 函数清单中的 dummy / always-predict / format-table 三段。

**【原文】**：

````
```python
def dummy_baseline_metrics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    strategy: str = "stratified",
    random_state: int = 0,
) -> dict
```
使用 sklearn `DummyClassifier`，strategy 支持 `"stratified"`、`"most_frequent"`、`"prior"`、`"uniform"`。

```python
def always_predict_baseline_metrics(
    y_test: np.ndarray,
    constant_label: int,
) -> dict
```
预测全 `constant_label`（0 或 1），返回 metrics dict。

```python
def format_metrics_table(metrics_dict: dict[str, dict]) -> str
```
输入：`{"lstm": {...}, "dummy_stratified": {...}, "always_up": {...}}`，输出对齐的文本表格。
````

**【替换为】**：

````
```python
def dummy_baseline_metrics(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    strategy: str,
    random_state: int,
) -> dict
```

- 使用 `sklearn.dummy.DummyClassifier`
- 第一阶段只用 `"stratified"` 和 `"prior"` 两种 strategy；`"most_frequent"`、`"uniform"` 不实现
- `DummyClassifier.fit(X=zeros, y=y_train)`，`y_eval` 用于评估；禁止用 y_eval 拟合

```python
def always_predict_baseline_metrics(
    y_eval: np.ndarray,
    constant_label: int,
) -> dict
```
- 预测全 `constant_label`（0 或 1），返回 metrics dict
- `constant_label = 1` 对应 always_up；`constant_label = 0` 对应 always_down

```python
def compute_baseline_table(
    y_train: np.ndarray,
    y_eval: np.ndarray,
    n_stratified_seeds: int = 10,
) -> pd.DataFrame
```

固定 4 个 baseline 行（与 AGENTS.md §2.4 一致）：

| baseline_name      | 实现                                                       |
|--------------------|-----------------------------------------------------------|
| dummy_stratified   | 跑 n_stratified_seeds 个 random_state，返回 mean ± std    |
| dummy_prior        | 跑 1 次（确定性）                                          |
| always_up          | constant_label=1                                          |
| always_down        | constant_label=0                                          |

返回 DataFrame 列至少包含：
`baseline_name`、`macro_f1_mean`、`macro_f1_std`、`balanced_accuracy_mean`、
`balanced_accuracy_std`、`confusion_matrix`。
对确定性 baseline，std 列为 0 或 NaN（在 docstring 中说明）。

```python
def format_metrics_table(
    model_metrics: dict[str, dict],
    baseline_table: pd.DataFrame,
) -> pd.DataFrame
```

合并模型指标和 baseline 表，**强制输出 `delta_macro_f1_vs_dummy` 列**，定义为：

```
delta_macro_f1_vs_dummy = model_macro_f1 - dummy_stratified_macro_f1_mean
```

返回 pd.DataFrame，列固定顺序由 notebook 03 spec 决定。
````

---
## §6.x MVP_YES

本 sprint（hf_stock_clf v4.1）锁定为 MVP_YES：

实施范围（W2–W5）：
- ml_utils/config.py
- ml_utils/seed.py
- ml_utils/metrics.py
- ml_utils/dataset.py
- ml_utils/checkpoint.py
- ml_utils/models/lstm_classifier.py
- ml_utils/trainer.py

推迟至 Phase 1B：
- ml_utils/models/tcn_classifier.py
- ml_utils/models/dlinear_classifier.py
- 对应测试
- reference_excerpts/pytorch_tcn_core.py
- reference_excerpts/ltsf_dlinear_model.py

约束：
- 不允许在 MVP 阶段创建 tcn / dlinear 文件（含空文件、占位）
- trainer.py / config.py 不预留 "for tcn/dlinear future use" 分支
- 通过 W5.4 integration test 后才启动 Phase 1B

## Patch 6 — §6 实施顺序，插入 step 0.5

**位置**：plan v2 §6.1 单模块标准流程之前的步骤表。

**【原文】**（表格前两行 + 表格中的 step 0/1）：

```
| 0   | 准备：clone 两个仓库，提取 3 个参考文件到 `reference_excerpts/`，删除 clone 目录 | — | — | 20 min |
| 0.3 | 用户填 `docs/ENVIRONMENT.md`（PYTHON_INTERPRETER）+ 锁定 `requirements.txt` | — | — | 30 min |
| 0.5 | Codex 跑 AGENTS.md §14.1 readiness audit（只读会话，返回报告） | — | — | 15 min |
| 0.7 | Codex 跑 AGENTS.md §14.2 testing infrastructure（创建 `tests/conftest.py` + `pytest.ini`） | — | — | 45 min |
| 1   | `config.py` | 第三档 | 100 | 30 min |
```
本流程新增 step 0.3、step 0.5、step 0.7。
**【替换为】**：

```
| 步骤 | 模块/任务 | 借鉴档次 | 行预算 | 估时 |
|------|----------|----------|--------|------|
| 0   | 准备：clone 两个仓库，提取 3 个参考文件到 `reference_excerpts/`，删除 clone 目录 | — | — | 20 min |
| 0.3 | 用户填 `docs/ENVIRONMENT.md`（PYTHON_INTERPRETER）+ 锁定 `requirements.txt` | — | — | 30 min |
| 0.5 | Codex 跑 AGENTS.md §14.1 readiness audit（只读会话，返回报告） | — | — | 15 min |
| 0.7 | Codex 跑 AGENTS.md §14.2 testing infrastructure（创建 `tests/conftest.py` + `pytest.ini`） | — | — | 45 min |
| 1   | `config.py` | 第三档 | 100 | 30 min |
```

并把表格末尾 "**总估时**：约 20 小时" 改为 "**总估时**：约 21.5 小时（含准备工作 1.5 h）"。

---

## 验证 patch 应用是否完整

应用完 6 个 patch 后，在 plan v2 中以下关键字应**全部消失**：

```
等价做法
二选一在 docstring 说明
"most_frequent"、"prior"、"uniform"
Shuffled label 检查
data_dir: str                   # 数据文件所在目录   ← 旧 DataConfig 注释
```

以下关键字应**全部出现**：

```
timezone_policy
__post_init__
test_no_window_crosses_trading_day
test_no_label_horizon_crosses_trading_day
class 0 = "non_up"
compute_baseline_table
delta_macro_f1_vs_dummy
step 0.3
step 0.5
step 0.7
```

每条 grep 通过 → plan v2 与 AGENTS.md v4 对齐。


