# PROJECT_OVERVIEW.md — hf_stock_clf

> **意图**：新 agent 会话进入本项目的快速导航文件。
> 先读本文件（5 分钟），再读 `AGENTS.md`（硬约束），然后开始工作。
>
> **最后更新**：2026-05-18
> **当前阶段**：Phase 1B — Step 3 capacity profiling next

---

## 1. 项目一句话

**高频股票 5 分钟 bar 方向二分类 PyTorch 工具库 (`ml_utils`)**，Northeastern 毕设项目。

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/zkc768/hf_stock_clf (private) |
| **默认分支** | `master` |
| **Phase 1B 分支** | `phase-1b` |

---

## 2. 当前状态

| 维度 | 状态 |
|---|---|
| **阶段** | Phase 1B — Steps 1-2 complete，Step 3 capacity profiling next |
| **已实现模型** | 仅 LSTM（smoke test 通） |
| **Notebook** | 00 (Colab setup) ✅, 01 (single-stock LSTM) ✅, 02 (pooled LSTM) ✅ |
| **Tests** | 107 passing, 1 known scheduler-order warning |
| **Git** | latest local commits: `c7162c2` labels, `d9352fc` config；no push performed for these commits |
| **Phase 1B 待建** | TCN, DLinear, Multi-scale DLinear+TCN |

---

## 3. 目录结构（实际）

```
hf_stock_clf/
├── AGENTS.md                       ← Agent 硬约束基线（必读）
├── README.md
├── requirements.txt                ← 锁定依赖版本（==，无>=）
├── pytest.ini
│
├── ml_utils/                       ← 库源码（PyTorch 工具集）
│   ├── __init__.py
│   ├── config.py                   ← DataConfig / WindowConfig / TrainConfig / ModelConfig
│   ├── seed.py                     ← seed_everything()
│   ├── dataset.py                  ← 标签构造 / 时序拆分 / 缩放 / 窗口Dataset
│   ├── trainer.py                  ← Trainer + train_one_epoch + evaluate
│   ├── checkpoint.py               ← save_checkpoint / load_checkpoint
│   ├── metrics.py                  ← 分类指标 + 4 个 baseline
│   └── models/
│       ├── __init__.py
│       └── lstm_classifier.py      ← LSTMClassifier（唯一已实现模型）
│
├── tests/                          ← pytest 测试（107 tests, 1 known scheduler-order warning）
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_seed.py
│   ├── test_metrics.py
│   ├── test_dataset_leakage.py
│   ├── test_label_generation.py
│   ├── test_window_boundaries.py
│   ├── test_checkpoint.py
│   ├── test_no_trade_band_labels.py
│   ├── test_models_shape.py
│   └── test_trainer_smoke.py
│
├── notebooks/
│   ├── 00_colab_setup.ipynb       ← Colab 启动（clone + Drive mount）
│   ├── 01_smoke_test_single_stock_lstm.ipynb
│   └── 02_smoke_test_pooled_lstm.ipynb
│
├── PHASE_1B_PLAN.md                ← Phase 1B 执行计划（12 步）
├── PHASE_1B_SESSION_CONTEXT.md     ← Phase 1B 会话上下文（防漂移）
├── PHASE_1B_RESEARCH_HANDOFF.md    ← Phase 1B 研究交接
│
├── docs/
│   ├── PROJECT_OVERVIEW.md         ← 本文件
│   ├── ENVIRONMENT.md              ← Python 解释器路径 + 环境验证
│   ├── SKILLS_USAGE.md             ← Skills 使用指南
│   └── ml_utils_construction_plan_v2.md ← 构建计划（spec 权威）
│
├── reference_excerpts/             ← 外部参考（只读，禁止 import）
│   ├── ltsf_data_loader.py
│   ├── ltsf_dlinear_model.py
│   └── pytorch_tcn_core.py
│
├── data/                           ← 股票 CSV（.gitignore）
└── checkpoints/                    ← 模型检查点（.gitignore）
```

---

## 4. 快速启动

### 本地开发

```bash
cd E:\codex_workspace\projects\hf_stock_clf
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests/ -q -m "not integration"
```

### Colab

1. 打开 `notebooks/00_colab_setup.ipynb` → Run All（clone + Drive mount）
2. 打开实验 notebook → Run All
3. 数据在 `MyDrive/stockdata/`

### 推送更新

```bash
git add -A && git commit -m "..." && git push
# Colab 端重跑 00_colab_setup.ipynb 即自动 git pull
```

---

## 5. 文档导航（agent 必读顺序）

| 顺序 | 文件 | 内容 |
|---|---|---|
| 1 | `docs/PROJECT_OVERVIEW.md` | 本文件，项目全景 |
| 2 | `AGENTS.md` | 硬约束基线 |
| 3 | `docs/ENVIRONMENT.md` | Python 路径 + 环境 |
| 4 | `docs/ml_utils_construction_plan_v2.md` | 模块 spec + 验收标准 |
| 5 | `PHASE_1B_PLAN.md` | Phase 1B 执行步骤 + 模型架构 spec |
| 6 | `PHASE_1B_SESSION_CONTEXT.md` | 决策记录 + 禁止事项 + 文件清单 |

**优先级规则**（AGENTS.md）：当前 sprint prompt > AGENTS.md > memory > construction_plan_v2 > SKILLS_USAGE

---

## 6. 模型状态

| 模型 | 文件 | 状态 |
|---|---|---|
| LSTMClassifier | `ml_utils/models/lstm_classifier.py` | ✅ smoke test 通 |
| TCNClassifier | `ml_utils/models/tcn_classifier.py` | Phase 1B Step 6 |
| DLinearClassifier | `ml_utils/models/dlinear_classifier.py` | Phase 1B Step 7 |
| MS-DLinear+TCN | `ml_utils/models/ms_dlinear_tcn_classifier.py` | Phase 1B Step 10 |

外部参考仓库：`E:\codex_workspace\projects\hf_stock_ml_references2\repos\`（TSLib, Autoformer, FEDformer, LTSF-Linear, mlfinlab）

---

## 7. 核心技术约定

| 约定 | 详情 |
|---|---|
| **输入形状** | `(batch, seq_len, features)` = NLC |
| **Legacy 标签** | `make_binary_labels_from_future_avg_return` → `future_avg_r`, `label`; class 0=non_up, class 1=up; exact zero maps to 0.0; last k rows are NaN |
| **Phase 1B 标签** | `make_no_trade_band_labels(..., timestamp_col=None)` → `future_avg_r`, `label` plus diagnostics; class 0=down, class 1=up, neutral=NaN |
| **Config label fields** | `DataConfig.label_mode` in `{"legacy_binary", "no_trade_band"}`; `threshold_bps >= 0` |
| **时序拆分** | chronological，train→val→test 按时间顺序，禁止 shuffle |
| **Scaler** | fit on train only，transform val/test |
| **跨日处理** | 窗口 + label horizon 均不许跨交易日，NaN 标注后跳过 |
| **设备** | `TrainConfig.device` 统一管理，`forward` 内禁止 `.to(device)` |
| **Loss** | `CrossEntropyLoss(reduction='none')`，模型输出 raw logits |
| **主指标** | macro F1 + balanced accuracy（非 accuracy） |
| **Baseline** | dummy_stratified, dummy_prior, always_up, always_down |

---

## 8. Phase 1B 方向

详见 `PHASE_1B_PLAN.md`。当前 no-trade-band 标签和 config label fields 已完成；下一步是 capacity profiling。TCN、DLinear、Notebook 03 和改进模型仍未实现。
