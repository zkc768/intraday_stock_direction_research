# AGENTS.md — hf_stock_clf / ml_utils
<!-- AGENTS_VERSION: v4.2 -->

> Codex 进入此项目目录时自动加载。本文件是所有 agent 会话的硬约束基线。
> 模块级实施细节参见 `docs/ml_utils_construction_plan_v2.md`。
> **与 plan v2 冲突时，以 AGENTS.md 为准**；若冲突影响 function signature 或测试，停下问。
> 违反本文件任何"硬规定"项 = 停下问，不许自行决定。

## 1. 项目身份

- **项目名**：hf_stock_clf — 高频股票方向二分类轻量库
- **库名**：ml_utils — 第一性原理实现的 PyTorch 工具集
- **目标**：5 分钟 bar / 5 只股票上跑通端到端 baseline，后期扩展到 80 只 NASDAQ
- **学术背景**：Northeastern 毕设，建立严格 baseline 后再做 EMD / attention 等高级方法
- **判断基准**：本项目是"窄而干净"的库，不是通用框架。所有"是否值得做"的判断必须围绕本范围

## 2. 第一阶段范围

### 2.1 数据

- 频率：5 分钟 bar
- 股票：CSCO, JPM, KO, MSFT, WMT（5 只，写入 config，不 hardcode）
- 特征：OHLCV + 技术指标（MACD 族、RSI-14、Bollinger %B、rolling std、OBV 变化率）
- 后期扩展：80 只 NASDAQ → 所有处理逻辑必须股票数量无关、频率无关、特征数量无关

### 2.2 任务

- 二分类（不是三分类、不是回归、不是 multi-label）
- 标签：未来 k 个 bar 平均 return 是否 > 0，k ∈ {12, 24}
- 标签在 pandas 阶段计算完存为列，`Dataset.__getitem__` 只做索引查表

### 2.2.1 标签公式（固定，禁止变体）

二分类 label 必须按以下三步计算，每只股票单独执行：

1. bar-to-bar return：`r_t = (price_{t+1} - price_t) / price_t`
2. 未来 k 个 return 的**算术平均**：`future_avg_r_t = mean(r_{t+1}, ..., r_{t+k})`
3. `label_t = 1 if future_avg_r_t > 0 else 0`

类别语义（全库统一，metrics / baseline / confusion matrix 都遵循）：

```
class 0 = "non_up"   ← future_avg_r_t <= 0  (含下跌和平盘)
class 1 = "up"       ← future_avg_r_t > 0
```

`sklearn.metrics.confusion_matrix` 调用时必须显式 `labels=[0, 1]`，禁止依赖默认值。

精确边界：

- `future_avg_r_t == 0` → `label = 0`（归入 non_up）
- 最后 k 行 `future_avg_r` 为 NaN → label 也为 NaN。**保留这些 NaN 作为后续阶段的 invalid marker**：
  - `trim_labels_at_split_boundary` 在 split 边界**追加 NaN 标注**，不删除行
  - 跨交易日 horizon 同样通过 NaN 标注实现（具体函数边界见 §3.4 Stage 3 与 plan v2 §5.3）
  - `WindowedClassificationDataset` 构造窗口时跳过所有 label==NaN 的起点
  - 任何阶段均不许 `fillna`、不许 `dropna` 删行
- 禁止变体：几何平均 `(price_{t+k}/price_t)^(1/k) - 1`、总收益 `(price_{t+k} - price_t)/price_t`、阈值化为三分类（up/flat/down）
- 要切换公式 → 停下问

### 2.3 模型

- 三个对比模型：LSTM classifier、TCN classifier、DLinear-style classifier
- Baseline 数量与命名由 §2.4 唯一定义（固定 4 个：dummy_stratified、dummy_prior、always_up、always_down），本节不重复声明

### 2.4 评估

- 主指标：macro F1、balanced accuracy
- 辅助：confusion matrix、precision/recall macro、classification_report
- **accuracy 不是主指标** — 类别可能不平衡，accuracy 容易误导

**4 个 baseline**（数量固定，命名固定）：

```
1. dummy_stratified  ← sklearn.dummy.DummyClassifier(strategy="stratified")，跑 10 个 random_state 取 mean ± std
2. dummy_prior       ← sklearn.dummy.DummyClassifier(strategy="prior")，确定性，1 次
3. always_up         ← 全预测 class 1
4. always_down       ← 全预测 class 0
```

dummy baseline 必须 fit 在 `y_train` 上、evaluate 在 `y_eval` 上，禁止用 test 分布拟合。

`delta_macro_f1_vs_dummy = model_macro_f1 - dummy_stratified_macro_f1_mean` 是判断模型是否真有信号的核心数字，必须出现在最终对比表。

完整对比表 schema（model_name / ticker / seed / 各指标 / delta）由 plan v2 §5.2 metrics.py 和 notebook 03 spec 定义。

### 2.5 第一阶段明确不做

| 类别 | 不做内容 |
|------|---------|
| 框架集成 | PyTorch Lightning、Hydra、OmegaConf、wandb、tensorboardX、mlflow |
| 模型 | Transformer、FinGPT、attention pooling、mean pooling |
| 工程 | streaming inference、ONNX 导出、multi-GPU 分布式、AMP |
| 数据 | NLP、sentiment、文本管线 |
| 抽象 | callback、plugin、hook 系统、JSON config 注册器、动态 import |
| 训练 | reinforcement learning、trading strategy 回测 |

遇到 spec 没要求但你想做的事 → 停下问。

## 3. 不可违反的硬约束（BLOCKER）

任何一条触发 → 立即停止实现，回到方案讨论。

### 3.1 时间顺序与泄漏防控

1. train / val / test 切分必须按时间顺序进行，禁止任何形式的 random shuffle 切分
2. Scaler 只在 train segment fit，再 transform 到 val / test
3. 多股票合并 fit 一个全局 scaler，但合并前每只股票必须各自完成时间切分
4. 标签窗口（未来 k bar 均 return）跨越 split 边界的样本必须被**标记为 invalid 并跳过**：实现上通过 `label=NaN` 标注（由 `trim_labels_at_split_boundary` 在 split 边界追加 NaN），`WindowedClassificationDataset` 构造窗口时跳过所有 `label==NaN` 起点。**不许 dropna 删行**（与 §3.4 Stage 2/3、§2.2.1 一致）
5. 训练集内部允许 DataLoader 的 batch-level shuffle，但仅限"窗口已切进 train 之后"
6. pooled 多股票时，窗口生成必须 per-ticker，禁止跨股票窗口
7. `__init__` 或 setup 阶段禁止对全量数据计算统计量
8. 禁止任何 forward-looking 特征（如向前看的 rolling mean）进入输入
9. 输入窗口和 label horizon 均不许跨交易日边界。**无 config 开关，无 escape hatch**。
   - 交易日边界由 `timestamp_col.dt.date` 判定
   - 必须有测试：`test_no_window_crosses_trading_day`、`test_no_label_horizon_crosses_trading_day`
   - 触发原因：5-min intraday 数据跨日会把隔夜跳空、不同 regime 和不同流动性状态混入同一窗口
   - 第一阶段保持此硬约束。要允许跨日 → 走未来的设计变更流程，不在第一阶段开 flag

### 3.2 不 hardcode

`dataset.py` / `trainer.py` / 模型文件等内部禁止出现以下字面量：

- `78`、`12`、`0.7`、`0.15` 等魔法数字
- `["CSCO", "JPM", ...]` 等股票列表
- `"close"`、`"timestamp"`、`"volume"` 等列名

所有数据相关参数必须通过 `config.py` 中的 dataclass 传入。

例外（**不**算违反硬约束）：

- `config.py` 的 dataclass 默认值：`timestamp_col: str = "timestamp"`、`price_col: str = "close"`、`train_ratio: float = 0.7` 等是**合法默认值**，不是 hardcode。这是 config 自己定义默认的地方
- `tests/` 下的合成 fixture：deterministic 测试必须用显式常量（如 fixture 内 `timestamp_col="timestamp"`、构造 5 个 ticker 的 dict），fixture 的本质就是固定输入，不受此禁令约束
- `notebooks/` 顶层 cell 中的配置值：notebook 是 orchestration，其顶层 cell 本身就是 config 来源

简言之：禁的是 `ml_utils/<module>.py` 实现里的字面量；config 自己定的默认值、tests 里的 fixture 常量、notebook 顶层的 orchestration 参数都不禁。

### 3.3 数据层与模型层解耦

- `dataset.py` 输出的张量接口对模型透明
- 模型不知道也不关心数据来源、来自哪只股票、什么时间段
- 模型 `forward` 内部禁止 `.to(device)`、禁止读取全局状态
- 模型不内置 softmax / sigmoid，loss 函数处理

### 3.4 输入 DataFrame Schema 契约

外部传给 `ml_utils` 的每只股票 DataFrame 必须满足以下分阶段约束。`dataset.py` 在对应阶段必须**显式 validate**，违反立即 `raise ValueError` 并指明 ticker、列名、行号。禁止 trust-but-don't-verify。

**Stage 1 — Raw input（label 生成前）**：

- timestamp 列必须满足 `pandas.api.types.is_datetime64_any_dtype`（不能用 `dtype == "datetime64[ns]"` 判定，那会漏掉 tz-aware 类型）
- timezone 行为由 `DataConfig.timezone_policy` 控制：
  - `"naive"` → `timestamp_col.dt.tz is None`
  - `"utc"`   → tz-aware 且 normalized 到 UTC
- timestamp 列名来自 `config.DataConfig.timestamp_col`，默认 `"timestamp"`
- 每个 ticker 内 timestamp 严格递增、无重复
- `timestamp_col`、`price_col`、所有 `feature_cols` 列存在且**无 NaN**
- `price_col` 严格正数
- `volume` 列（若使用）非负
- 所有 `feature_cols` 列为数值 dtype（float32 / float64 / int）
- 技术指标必须由调用方在数据准备阶段**提前算好**，作为输入列传入；`ml_utils` 不内置特征工程

**Stage 2 — Labeled frame（label 列加好后、trim 前）**：

- Stage 1 所有约束仍成立
- 新增 `label` 列；允许 NaN，**只允许**出现在以下位置：
  1. 每只 ticker 的最后 k 行（来自 `make_binary_labels_from_future_avg_return` 的自然末尾）
- 此阶段不许 fillna、不许 drop NaN——保留 NaN 作为后续 trim 的依据

**Stage 3 — Post-trim frame（trim 后、构造窗口前）**：

- Stage 2 所有约束仍成立
- label NaN 现在还可以来自：
  2. split 边界（由 `trim_labels_at_split_boundary` 标注）
  3. 跨交易日边界（label horizon 跨日的样本，由跨日过滤标注）
- 构造窗口时，所有 label==NaN 的起点必须被跳过（dataset.py 内部保证）

列名大小写约定：全小写（`open`、`high`、`low`、`close`、`volume`），由 config 锁定。

## 4. 环境与版本

```
Python: 3.10 或 3.11
PyTorch: 2.x (具体小版本由 requirements.txt 锁定)
CUDA: 12.x (Colab T4 / L4 / A100)
numpy: 1.x
pandas: 2.x
scikit-learn: 1.x
pytest: 8.x
```

**requirements.txt 禁止使用 `>=`，全部锁死小版本**。

### 4.1 启动前必须敲定的字段（TODO，由用户填写）

版本管理采取**单一 source of truth**：

- `requirements.txt` 是**唯一**的依赖版本权威，所有版本用 `==` 锁到具体小版本
- `docs/ENVIRONMENT.md` **只**记录 PYTHON_INTERPRETER 和环境校验命令，**不**重复列出版本
- 两个文件若出现版本声明冲突 → 停下问

未敲定以下两项前，**禁止开始任何 `ml_utils/*` 实现会话**；只允许做 §14 的只读 readiness audit。

```
PYTHON_INTERPRETER  ← 本地 Codex 使用的 Python 完整路径
                     例如 E:\codex_workspace\_envs\py311_shared\python.exe
                     写入 docs/ENVIRONMENT.md

requirements.txt    ← 锁定后的依赖清单
                     生成方式: <PYTHON_INTERPRETER> -m pip freeze > requirements.txt 之后
                     人工清洗到 ml_utils 实际需要的包
```

模板：见 `docs/ENVIRONMENT.md`（与本文件一同交付，由用户填写 PYTHON_INTERPRETER 后激活）。

## 5. 目录结构

```
hf_stock_clf/
├── AGENTS.md                    本文件
├── README.md
├── requirements.txt
├── ml_utils/                    库源码
│   ├── __init__.py
│   ├── config.py
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
├── notebooks/                   实验 notebook
├── tests/                       pytest 测试
├── reference_excerpts/          外部参考代码 (只读)
├── docs/                        项目文档 (含 ml_utils_construction_plan_v2.md)
├── data/                        用户自管，库不负责，进 .gitignore
└── checkpoints/                 进 .gitignore
```

**禁止行为**：

- 自己代码不许 import `reference_excerpts/` 下任何文件
- 不创建 `utils.py`、`helpers.py`、`common.py` 这类"杂物间"文件
- 不在根目录新增 convenience script

## 6. 代码规范

### 6.1 依赖白名单

允许的 import：

- `torch`、`torch.nn`、`torch.optim`、`torch.utils.data`
- `numpy`、`pandas`
- `pandas.api.types`（用于 §3.4 timestamp dtype 校验）
- `sklearn.preprocessing`、`sklearn.metrics`、`sklearn.dummy`、`sklearn.base`
- stdlib：`dataclasses`、`typing`、`pathlib`、`random`、`os`、`json`、`math`、`copy`、`collections`、`datetime`、`inspect`

模块特定额外允许（必须在该模块文件顶部注释中说明）：

- `checkpoint.py`：可用 `torch.save` / `torch.load`，不许用 `pickle` 直接 dump 数据
- `tests/`：可用 `tempfile`、`pytest` fixture

禁止的 import（违反 = 停下问）：

- `pytorch_lightning`、`lightning`
- `hydra`、`omegaconf`
- `wandb`、`tensorboardX`、`mlflow`
- 任何 `*_logger` 第三方
- `torch.optim` 之外的 optimizer 库

如认为需要新增 import，停下问。

### 6.2 Tensor 形状纪律

- 每个收 / 发 tensor 的函数必须在 docstring 中写明形状：
  ```
  x: torch.Tensor of shape (batch, seq_len, features)
  ```
- 非平凡变换的函数入口必须加 shape assertion：
  ```python
  assert x.dim() == 3, f"Expected 3D, got {x.shape}"
  ```
- 全库统一约定输入形状 `(batch, seq_len, features)` = NLC
- Conv1d 需要 NCL → 由模型 `forward` 内部 transpose 处理，不由 dataset 处理

### 6.3 行预算

每个模块都有硬性行数上限，见 `docs/ml_utils_construction_plan_v2.md` 第 5 节。

超出行预算 = 在过度工程化 → 重构或停下问。

### 6.4 注释风格

- 用祈使句描述当前块在做什么（例如 `# Create the testing data set`）
- 不写括号补语、不写元注释（如"修改了 X"、"为了灵活性"）
- 不用装饰性分隔符（`# ===`、`# -----`）

### 6.5 Config dataclass 自验证

`config.py` 中每个 dataclass 必须实现 `__post_init__`，在非法值上立即 `raise ValueError`，错误信息必须指出违反字段名和实际值。

最低验证清单（不止于这些；如有自然需要更多）：

```
DataConfig:
- tickers 非空
- feature_cols 非空
- 0 < train_ratio < 1
- 0 < val_ratio < 1
- train_ratio + val_ratio < 1

WindowConfig:
- window_size > 0
- label_horizon_k > 0
- stride > 0

TrainConfig:
- batch_size > 0
- num_epochs > 0
- learning_rate > 0
- monitor_mode in {"max", "min"}
- device in {"cpu", "cuda"}
```

config 不是"装参数的盒子"，是第一道防错。

### 6.6 库函数 print 纪律

`ml_utils/` 下所有非 trainer 函数禁止直接 `print`。需要日志的：

- trainer 类入口（如 `Trainer.fit`）可有 `verbose: bool = False` 参数，True 时每 epoch 输出一行紧凑日志
- 默认 `verbose=False`（测试期间不许输出噪音）
- 其他底层函数（`train_one_epoch`、`evaluate`、`compute_classification_metrics` 等）一律不许 print，只返回数据结构

模块特定的日志格式细节见 plan v2 对应 §5.x。

## 7. Agent 工作合同（反屎山规则）

### 7.1 范围纪律

实现会话允许修改的文件**只有以下**：

1. `target_file`：单个 `ml_utils/<module>.py`
2. `test_file(s)`：当前模块对应的全部测试文件（可以多个，例如 `dataset.py` 对应三个 test_*.py）

`target_file` 和 `test_file(s)` 的精确清单由 pre-flight（§9.1）从 plan v2 §5.x 中抽出，allowed_files 等于 `[target_file, *test_file(s)]`。

其他动作的归属：

- `docs/CHANGELOG.md` 更新 → 不属于实现会话，由用户在独立的 finalization step 中做（见 §11）
- 修改另一个模块文件 → 停下问
- 创建新文件、新目录 → 停下问
- 修改 `requirements.txt`、`AGENTS.md`、plan v2 → 停下问
- 修改 `tests/conftest.py` / `pytest.ini` → 不属于实现会话，由 §14.2 testing infrastructure 一次性 step 处理
- 添加 convenience script、example、README 更新 → 停下问
- **为未授权 future module 写测试文件 → 禁止**。只有当本会话 prompt 的 `allowed_files` 显式列出该 `tests/<test_file>` 时，才允许创建该测试文件；否则 `tests/test_<future_module>.py` 不许出现。

### 7.D Test-first 通用约束

本节仅适用于 **test-writing / test-first session**。

本节也称为 test-first lazy import rule：test-first 测试必须通过 lazy import 确保 `pytest --collect-only` 不会在目标模块尚未实现时失败。

适用场景：

| Session 类型 | 是否适用 | 说明 |
|---|---:|---|
| test-first session | 是 | 例如 W2.1 / W3.1 / W4.A.1 / W4.B.1 / W4.C.1 / W4.D.1 / W5.1 |
| implementation session | 否 | 目标模块已进入实现阶段，测试文件通常已存在 |
| review session | 否 | review session 只读，不创建测试、不实现代码 |
| log sync / docs sync session | 否 | 只改日志或文档状态，不创建测试 |

在目标模块尚未实现前，测试文件不得在 module top-level import 目标模块。

禁止：

```python
from ml_utils.seed import seed_everything
import ml_utils.seed
```

允许：

```python
def test_seed_everything_sets_pythonhashseed():
    from ml_utils.seed import seed_everything

    seed_everything(123)
```

或：

```python
def _seed_everything():
    from ml_utils.seed import seed_everything

    return seed_everything
```

原因：test-first session 只要求测试文件可被 pytest collect；目标模块尚未实现时，module top-level import 会导致 collection 阶段失败，阻断先写测试、后实现的流程。

一旦目标模块实现完成，implementation / review session 可按普通测试习惯 import，但仍不得绕过本会话 prompt 的 `allowed_files`。

### 7.2 无未来代码

禁止模式：

- TODO 注释
- `if config.use_xxx:` 分支，其中 `use_xxx` 不在当前 config 中
- 只有一个具体子类的抽象基类
- callback / hook / plugin 系统
- 空 try/except 块
- "for future extensibility" 类注释

spec 不要求的，就不要建。

### 7.3 无静默修复

1. 遇到 bug 或意外行为，不允许静默 work around → 停下，描述问题，等待决策
2. 不允许 catch 并忽略异常
3. 不允许加"防御性" try/except → 让错误大声失败

### 7.4 参考实现处理

对使用 `reference_excerpts/` 下参考文件的模块：

1. 先完整读参考文件
2. 用 3-5 个 bullet 总结学到了什么
3. 再按 spec 写自己的实现
4. 不许复制代码块 → 用本项目命名风格重写
5. 在文件顶部注释中列出哪些思想借鉴自哪里

### 7.5 测试先行

1. 先写 pre-flight 中列出的全部 `test_file(s)`（与 §7.1、§9.1 一致；可能多个，如 dataset.py 对应三个）
2. 展示测试给用户，等待批准
3. 批准后才实现 `target_file`
4. 所有测试必须不修改测试本身就通过
5. 若测试看起来错了，停下问 — 不许"修复"测试

### 7.5.1 测试质量门槛

每个模块的测试必须覆盖以下类别。命中不全 → review 时算 BLOCKER。

- **正常情况**：典型输入下输出符合 spec
- **边界情况**：empty input、最小 / 最大尺寸、k 边界值（如 k=1）、stride 边界
- **错误情况**：非法输入必须 raise（与 §6.5 config 验证、§3.4 schema 验证配套）
- **不可变性**（适用时）：函数返回新对象，不原地修改输入 DataFrame / 张量
- **确定性期望值**：用手算结果断言具体数值，禁止只断言 shape 或 dtype
- **泄漏专项**（仅 `dataset.py`）：参见 plan v2 §5.3.5

禁止：只测 `import` 是否成功；只测 shape 不测值；用 random 输入跑一遍看不 raise 就算过。

### 7.6 API 验证

使用任何非平凡库 API 前，必须对照 `requirements.txt` 锁定的版本验证函数签名。不允许凭记忆使用 API。

常见易变 API：

- `sklearn.metrics` 各函数签名（`f1_score` 的 `average` 参数等）
- `torch.optim` 变体（RAdam、NAdam 可用性）
- `pandas` 的 `resample` / `groupby` 行为

## 8. 反模式速查表

每写完一个模块，对照检查。命中任何一条 → 在向用户汇报时显式说明命中编号，或修掉。

### 时间序列泄漏类（命中 = BLOCKER）

- **L1**：全量数据 fit scaler 后再切分 train / val / test
- **L2**：用 `random.shuffle` / `np.random.permutation` 切 train / val / test
- **L3**：用 `sklearn.model_selection.train_test_split`（默认 `shuffle=True`）切时序
- **L4**：标签窗口跨 train / val 或 val / test 边界仍保留
- **L5**：多股票合并后再切窗口 → 产生跨股票窗口
- **L6**：`__init__` 或 setup 对全量数据计算统计量
- **L7**：用 future 信息计算的特征（forward-looking rolling mean）进入输入

### 评估误导类（命中 = WARNING）

- **E1**：主指标仅用 accuracy，无 macro F1 / balanced accuracy
- **E2**：val 和 test 加载同一份数据
- **E3**：早停或 best-checkpoint 基于 val accuracy 而非 macro F1
- **E4**：没有 baseline 对比（dummy / always-positive / always-negative）
- **E5**：混淆矩阵未输出或仅在 docstring 提及未实际计算
- **E6**：类别不平衡场景使用未加权 cross-entropy 且无说明

### 工程债务类（命中 = WARNING）

- **W1**：`collate_fn` 用 lambda（`num_workers>0` 会 pickling 失败）
- **W2**：全局可变状态（module-level 字典、单例 config）
- **W3**：硬编码股票代码、日期范围、特征列名、窗口大小
- **W4**：try / except 吞异常无日志
- **W5**：设备处理混乱（`.cuda()` / `.to(device)` 混用，或在 `forward` 内 `.to()`）
- **W6**：requirements 用 `>=` 未锁版本

### 依赖膨胀类（命中 = BLOCKER）

- **D1**：主功能模块依赖 Lightning / Hydra / wandb
- **D2**：plotting / 可视化依赖渗入核心训练逻辑
- **D3**：同一仓库混合训练 / 推理服务 / UI / 策略回测

## 9. 会话工作流程

### 9.1 Pre-flight（实现前强制完成）

任何模块实现会话开始前必须完成以下检查。任一步骤失败 → 停下问，不许凭记忆继续。

1. 读完整的 `AGENTS.md`（本文件）
2. 读 `docs/ENVIRONMENT.md`，获取 PYTHON_INTERPRETER 和锁定版本（见 §4.1）
3. 读 `docs/ml_utils_construction_plan_v2.md` 中对应模块的 §5.x 段
4. 从 §5.x 中抽出并显式确认八项：
   - `target_file`（实现路径）
   - `test_file(s)`（测试路径，可多个）
   - `plan_section`（§5.x 编号）
   - `line_budget`（行预算）
   - `reference_tier`（第一档 / 第二档 / 第三档）
   - `required_reference_excerpts`（若第一档，列出 `reference_excerpts/` 下必读文件）
   - `acceptance_criteria`（plan v2 中"验收标准"段）
   - `allowed_files`（本会话允许修改的全部文件清单，应等于 `[target_file, *test_files]`）
5. 若是第一档模块，确认 `required_reference_excerpts` 中每个文件都已存在
6. 任一项缺失或不一致 → 停下问

不许只加载 AGENTS.md 就开始写。不许凭记忆补"plan v2 大概是这样"。

### 9.2 单模块标准流程

```
对模块 M：

1. 开新会话 A1 (fresh context)
2. 完成 §9.1 pre-flight
3. A1 先写 pre-flight 中列出的全部 `test_file(s)` — 用户审查测试覆盖
4. 批准 → A1 写 `ml_utils/M.py`（target_file）
5. 按 §9.4 跑测试 — 红 → A1 修，不许改测试
6. 全绿 → 开新会话 A2 做 self-review (见 §9.3)
7. 处理 A2 给出的 BLOCKER / WARNING
8. 重跑测试 → 全绿 → 模块完成
9. 关闭 A1，下一个模块开新会话 B1
```
**Exception**：`config.py` 的 spec source 是 `docs/ml_utils_construction_plan_v2.md` §3.1，不是 §5.x。W2 config sessions 按 §3.1 抽取 fields / line_budget / acceptance_criteria，§9.1 step 3-4 中"对应模块的 §5.x 段"对 config.py 改读 §3.1。

### 9.3 Self-Review 流程

完成实现后，**必须换全新会话**做 review，避免写代码时的 motivated reasoning。

Review 检查清单：

1. 应作为参数的硬编码值（数字、列表、路径）
2. 关于数据形状、频率、股票数的隐藏假设
3. train 数据能泄漏到 val / test 的地方
4. 未使用的 import 或死代码
5. 注释与代码实际行为不一致
6. 窗口 / 索引计算的 off-by-one
7. docstring 与实现之间的 tensor 形状不一致
8. 无理由的静默 except 或防御性代码
9. 在当前 config 下不可达的代码路径

输出格式：

- 一行一个 issue，前缀行号
- 严重度：BLOCKER / WARNING / NIT
- 不提建议修复方式 — 只列 issue

### 9.4 测试命令与验收门槛

所有测试统一通过 `docs/ENVIRONMENT.md` 中声明的 `$PYTHON_INTERPRETER` 调用，**不许用裸 `pytest`，也不许用 `python -m pytest`**。两者都会被 PATH 解析到非预期解释器（系统 Python / conda base / VSCode 默认环境 / Cursor 内置 Python），与 §14.1 审计通过的解释器不一致。

跨平台调用形式：

```bash
# Git Bash / Linux / macOS
"$PYTHON_INTERPRETER" -m pytest <args>
```

```powershell
# PowerShell (Windows)
& $env:PYTHON_INTERPRETER -m pytest <args>
```

下文所有测试命令均按上述形式调用；为简洁起见，仅列出 Git Bash 版本，PowerShell 版本按上面规则换算。

测试分类：

- **unit test**（默认）：synthetic 数据、确定性、不训练模型，必须秒级完成
- **integration test**：端到端 trainer smoke 这类需要训练但仍然短（< 1 min）的测试；标注 `@pytest.mark.integration`，需手工触发
- **shuffled-label sanity check 不放在 pytest 里**，统一放在 notebook 02（见 §13）。原因：训练模型有随机性，作为 pytest 慢测试容易假阳/假阴
- marker 注册在 `pytest.ini`（由 §14.2 一次性创建）

模块自身测试（每次实现后必跑，必须全绿）：

```bash
"$PYTHON_INTERPRETER" -m pytest <test_file_1> <test_file_2> ... -q
```

其中 `test_file_*` 来自 pre-flight 的 `test_file(s)` 清单。例如 `dataset.py` 的命令是：

```bash
"$PYTHON_INTERPRETER" -m pytest tests/test_dataset_leakage.py tests/test_label_generation.py tests/test_window_boundaries.py -q
```

已实现模块的全量回归（必须全绿）：

```bash
"$PYTHON_INTERPRETER" -m pytest tests/ -q -m "not integration"
```

注意：未实现模块不许存在测试文件（§7.1），所以"已实现模块的全量"等于当前 `tests/` 下全部 non-integration 测试。

integration 测试（手工触发，trainer 完成后跑）：

```bash
"$PYTHON_INTERPRETER" -m pytest tests/ -q -m "integration"
```

Agent 向用户汇报测试结果时必须包含：

- 命令原文
- 通过 / 失败 / 跳过的具体数量
- 失败测试名称（若有）

禁止只说 "tests pass" 或 "all green"。

### 9.5 测试数据规则

- 所有 `tests/` 下的 unit test 必须使用 **synthetic in-memory DataFrame**
- 测试代码禁止读取 `data/` 下任何文件
- 真实股票数据只允许在 `notebooks/` 中使用，或在显式标注的 integration test 中使用
- 合成数据 fixture 集中放在 `tests/conftest.py`，由 §14.2 testing infrastructure step 一次性创建

合成数据 fixture 必须**分层**，对应 §3.4 三阶段 schema：

| Fixture 名 | 满足的 schema | 用途 |
|------|------|------|
| `raw_price_df` | Stage 1（无 label 列） | label 生成、scaler 等单元测试 |
| `raw_multi_ticker_dict` | Stage 1（dict[ticker, df]） | 多股票切分、合并测试 |
| `labeled_df_with_tail_nan` | Stage 2（含 label 列，末尾 k 行 NaN） | trim、窗口边界测试 |
| `split_df_after_trim` | Stage 3（label NaN 在末尾 + split 边界 + 跨日处） | Dataset 构造、窗口生成测试 |

合成数据生成规则：

- timestamp 用 `pd.date_range` 生成，频率明示，例如 `freq="5min"`
- 价格列用固定 seed 的 random walk 或几何布朗运动生成，保证 Stage 1 的 `price_col > 0` 约束
- 跨多个交易日的 fixture 必须用 `business_day` + 盘中 time 拼接，模拟真实开盘/收盘边界，以便测试跨日过滤
- 每个 fixture 在 docstring 中显式声明属于哪个 stage

### 9.6 上下文漂移防控

长会话中 agent 容易：

- 把支线任务做成主线（在子任务上花掉大半步骤）
- 输出风格变化（开头结构化，后面变自然语言）
- 反复在同一类操作中循环

防控措施：

1. 每个模块一个独立会话，不在同一会话中跨模块
2. 长任务执行 5-8 步后**或**自觉思路开始模糊时，主动暂停，按以下模板显式列出当前状态：

   ```
   原始目标:     <一句话复述本会话最初接到的任务>
   当前子任务:   <现在正在做的具体步骤>
   已修改文件:   <列出至今 touch 过的文件>
   下一动作:     <下一个具体操作>
   ```

3. 单会话超过 10 轮工具调用 → 总结当前状态后开新会话

### 9.7 Git 工作流

Agent 在实现会话中：

- **禁止** `git commit`、`git push`、`git checkout -b` / 创建分支
- **禁止** 修改 `.gitignore`、`.git/` 下任何内容
- 会话开始时跑一次 `git status --short`，确认起始状态
- 会话结束时跑一次 `git diff --stat`，向用户汇报：
  - 被修改文件列表
  - 跑过的测试命令
  - 测试结果
  - 未解决问题（若有）

提交、push、合并 PR 等动作一律由用户手工执行，agent 不参与。

## 10. License 合规

`reference_excerpts/` 下任何文件：

- 顶部必须保留原 license header
- 自己代码绝不 import 这些文件
- 借鉴思想可以，搬代码块不可以

外部仓库 license 处理规则（**统一禁止代码块拷贝**，仅按 license 区分阅读与使用方式）：

| License 类型 | 处理 |
|------|------|
| MIT / Apache-2.0 | 思想可借鉴，代码块不可拷贝；必须用本项目命名风格重写 |
| GPL-3.0 / 系列 | 思想可借鉴，代码块不可拷贝；理由：避免本库被 GPL 传染 |
| 无 license | GitHub 默认保留所有权利 → 仅可阅读思路，不许任何形式代码搬运 |

无论 license 是什么，agent 一律不许从 `reference_excerpts/` 复制代码块到 `ml_utils/`。"有限拷贝"、"短片段拷贝"、"加 attribution 后拷贝" 都不算合规。要复用 → 用自己的命名和结构重写。

## 11. 文档与产出物

模块实现会话内产出（agent 负责）：

1. `ml_utils/<module>.py` — 实现（`target_file`）
2. `tests/test_<module>*.py` — 当前模块的全部测试文件（`test_file(s)`，可能多个，由 pre-flight 从 plan v2 §5.x 抽出）

模块实现会话外产出（用户在独立的 finalization step 中手工或在另一个简短会话里做）：

3. `docs/CHANGELOG.md` 追加一行：`[YYYY-MM-DD] <module>.py 完成，covered by N 个测试`

这条切分是为了与 §7.1 的"实现会话只许碰两个文件"对齐。CHANGELOG 不属于实现 scope，agent 在实现会话里不许 touch 它。

不需要产出：

- 单独的 module README
- 设计文档（设计已在 plan v2 中）
- example script（notebook 中体现）

## 12. 模块完成验收标准

一个模块只有同时满足以下七项才算"完成"。任一项不满足都不许进入下一模块。

1. 仅修改 §7.1 允许的文件（实现 + 该模块全部测试文件）
2. 测试先行流程完整（§7.5）：测试先写、用户批准、再实现
3. 模块自身测试（`"$PYTHON_INTERPRETER" -m pytest <test_file_1> <test_file_2> ... -q`，跨平台命令见 §9.4）全绿
4. 已实现模块的全量回归 `"$PYTHON_INTERPRETER" -m pytest tests/ -q -m "not integration"`（跨平台命令见 §9.4）全绿，或已知失败已向用户明示
5. self-review（§9.3）已在 fresh 会话中完成
6. self-review 给出的所有 BLOCKER / WARNING 已解决，或用户明确同意 defer
7. `git diff --stat` 已向用户汇报，列出修改清单

不允许 agent 自己宣布"完成"。完成与否由用户在收到上述七项汇报后判定。

## 13. Notebook 规则

`notebooks/` 下任何 `.ipynb` 必须遵守：

### 13.1 代码结构

1. 只做 orchestration、display、最终结果聚合；禁止复制 `ml_utils/` 核心函数到 notebook 内部
2. 可以 `import ml_utils.xxx`，但不许重新定义同名函数
3. 每个 notebook 必须有顶部 flag 控制重训练：`FULL_RUN = False` 或 `RUN_TRAINING = False`，默认关，启用时才跑完整训练
4. 主要输出收敛到 pandas DataFrame，最终结果表必须以 DataFrame 形式呈现，不许只散 print 数字
5. 中间 print 只允许 shape 检查和最终 metric；禁止打印整段中间张量内容

### 13.2 必含内容

6. **notebook 02 必须包含 shuffled-label sanity check**（在 notebook 内部直接实现：打乱 train label → 训练 → 报告 val macro F1 应 ≈ 0.5）。这是项目内唯一的 shuffled-label 检查载体，不在 pytest 中重复
7. notebook 03 必须包含三模型 × 4 个 baseline 对比表，列名固定遵循 plan v2 §5.2 metrics.py 输出（含 delta_macro_f1_vs_dummy）

### 13.3 交付前清理

8. 提交或分享 notebook 前必须清理大输出：
   - 删除整段训练日志、模型 summary、原始张量 dump
   - 只保留最终结果表、关键 metric 数字、必要的图
   - 用 `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace <notebook>` 或手工 Clear All Outputs 之后再选择性 re-run 关键 cell

违反任何一条 → notebook self-review 时算 BLOCKER。

## 14. 首次启动准备工作（一次性）

第一次让 Codex 接触本项目时，按以下两步顺序执行。这两步走完之前，**不许**进入 plan v2 §6 步骤 1 的 `config.py` 实现。

### 14.1 项目就绪审计（只读会话）

第一个会话**只做只读审计**，不写任何代码、不创建任何文件。审计输出报告，由用户决定是否进入 §14.2。

审计清单（11 项，按下面给出的两套 shell 命令任选一套）：

```
A. tree / 目录结构对照 §5
B. AGENTS.md 第一行可读
C. docs/ml_utils_construction_plan_v2.md 存在
D. docs/ENVIRONMENT.md 存在，读出 PYTHON_INTERPRETER
E. requirements.txt 存在，所有行用 ==，无 >=
F. reference_excerpts/ 下三个第一档参考文件齐全
   (ltsf_data_loader.py、ltsf_dlinear_model.py、pytorch_tcn_core.py)
G. <PYTHON_INTERPRETER> --version 与 §4 要求一致
H. <PYTHON_INTERPRETER> -m pytest --version 可调用
I. <PYTHON_INTERPRETER> -c "import torch, numpy, pandas, sklearn" 各包导入成功
J. git status --short 工作目录状态可见
K. tests/conftest.py 与 pytest.ini 是否存在（不存在不算失败，提示需要走 §14.2）
```

**Git Bash / Linux / macOS** 命令版本：

```bash
tree -L 2                                              # A
head -n 1 AGENTS.md                                    # B
test -f docs/ml_utils_construction_plan_v2.md          # C
test -f docs/ENVIRONMENT.md && cat docs/ENVIRONMENT.md # D
test -f requirements.txt && grep -E ">=" requirements.txt && echo BAD || echo OK  # E
ls reference_excerpts/                                 # F
"<PYTHON_INTERPRETER>" --version                       # G
"<PYTHON_INTERPRETER>" -m pytest --version             # H
"<PYTHON_INTERPRETER>" -c "import torch, numpy, pandas, sklearn"  # I
git status --short                                     # J
test -f tests/conftest.py; test -f pytest.ini          # K
```

**PowerShell（Windows）** 命令版本：

```powershell
Get-ChildItem -Depth 2                                                 # A
Get-Content AGENTS.md -TotalCount 1                                    # B
Test-Path docs/ml_utils_construction_plan_v2.md                        # C
Test-Path docs/ENVIRONMENT.md; Get-Content docs/ENVIRONMENT.md         # D
Test-Path requirements.txt; Select-String -Path requirements.txt -Pattern ">="  # E
Get-ChildItem reference_excerpts/                                      # F
& "<PYTHON_INTERPRETER>" --version                                     # G
& "<PYTHON_INTERPRETER>" -m pytest --version                           # H
& "<PYTHON_INTERPRETER>" -c "import torch, numpy, pandas, sklearn"     # I
git status --short                                                     # J
Test-Path tests/conftest.py; Test-Path pytest.ini                      # K
```

任何 A-J 项失败 → 报告给用户，**不许**自行修复（修复属于配置工作，由用户做）。K 项失败不算失败，但提示用户进入 §14.2。

### 14.2 测试基础设施搭建（一次性专项会话）

§14.1 通过后，开一个独立会话**只**用于创建以下两个文件。本会话不许动 `ml_utils/*` 或任何模块测试文件。

允许修改的文件清单（exactly two）：

```
tests/conftest.py
pytest.ini
```

`tests/conftest.py` 内容范围：

- 按 §9.5 创建分层 fixture：`raw_price_df`、`raw_multi_ticker_dict`、`labeled_df_with_tail_nan`、`split_df_after_trim`
- 每个 fixture 在 docstring 中说明属于 §3.4 的哪个 Stage
- 不写任何模块特定逻辑

`pytest.ini` 内容范围：

- 注册 `integration` marker
- 配置 testpaths = tests
- 不引入其他设置

完成后，下一个会话才能进入 plan v2 §6 步骤 1 的 `config.py` 实现。

## 15. 一句话总结

**这是一个小而干净的库；硬约束是时间顺序泄漏防控；每个文件单独会话先测试再实现；写完换会话 self-review；所有数据参数走 config，所有 tensor 形状进 docstring；不该建的不建，遇到问题大声失败。**
