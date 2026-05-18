# Phase 1B 上下文管理文件

更新时间：2026-05-18
用途：新窗口开局时提供给评审 AI，防止任务漂移
性质：只记录已确认的事实和决策，不记录未执行的计划细节

---

## 1. 项目一句话状态

```
MVP ml_utils 库：完成；当前 non-integration suite 107 passed, 1 known scheduler-order warning
Phase 1B：Step 1 no-trade-band labels 和 Step 2 config extension 已完成并提交
下一个动作：Step 3 capacity profiling
```

---

## 2. 关键决策记录（已确认，不可回退）

### 2.1 开发路线

- Phase 1B 走 ml_utils PyTorch 库路线，不走 Colab TF/Keras 实验路线
- 所有新代码进 ml_utils，test-first，遵循 ml_utils_construction_plan_v2.md 的 Wave 模型
- 实验在 notebooks 里调用 ml_utils 跑，不在 notebook 里写独立模型代码
- Colab TF/Keras 旧 notebooks（binary_clf_comparison 等）是参考资料，不是生产路径

### 2.2 数据与环境

- 代码：本地 E:\codex_workspace\projects\hf_stock_clf，推到 GitHub private repo
- 数据：Google Drive /content/drive/MyDrive/stockdata/
- 同步方式：GitHub clone 到 Colab，Colab 挂载 Drive 拿数据
- 本地开发跑 pytest，Colab 跑实验

### 2.3 标签升级

- 旧函数 make_binary_labels_from_future_avg_return 不改
- 已新增 make_no_trade_band_labels（对称阈值，neutral 标 NaN）
- 实际 API：`make_no_trade_band_labels(df, price_col, k, threshold_bps, timestamp_col=None) -> tuple[pd.DataFrame, dict[str, int]]`
- 输出列：`future_avg_r`, `label`
- diagnostics：`n_total`, `n_tail`, `n_cross_day`, `n_neutral`, `n_up`, `n_down`
- legacy exact-zero `future_avg_r` -> label 0.0；no-trade-band exact zero at threshold_bps=0 -> neutral NaN
- `timestamp_col=None` 禁用跨日过滤；提供 `timestamp_col` 时按 `.dt.date` 做跨日 invalidation
- DataConfig 已有 `label_mode="legacy_binary"` 和 `threshold_bps=0.0`，label_mode 只允许 `"legacy_binary"` / `"no_trade_band"`
- 第一轮只做 fixed bps：[5, 10, 15, 20, 30]
- 波动率缩放、三分类、threshold lists 均 defer
- 新语义：class 0=down, class 1=up, neutral=NaN/skipped
- 必须报告 retained coverage，不能只报 F1

### 2.4 DLinear 实现约束（来自 research report 3）

- 标准 DLinear baseline：单个奇数核（默认 5），individual=False，flatten+linear 分类头
- 偶数核在 TSLib/Autoformer 实现里会导致长度失配（L-1），只有 FEDformer 风格不对称 padding 能支持
- 多尺度 improved model：如果用 [3,6,12,24] 必须实现 FEDformer-style padding，否则用全奇数 [3,7,13,25]
- 多尺度融合用 FEDformer-style per-scale scalar softmax，不用 TSLib 的 simple averaging
- 写作措辞："DLinear adapted for classification, following Time-Series-Library"

### 2.5 Window / horizon

- 不预锁任何 window_size，profiling 之后才锁
- 候选：window=[12, 24, 60]，k=[12, 24]
- window=60 + k=24 = 84 > 78 bars/day → 理论 INFEASIBLE
- window=60 + k=12 需要 profiling 验证
- 78 bars/day 是理论值，需从数据验证实际 P5

### 2.6 Selection bias

- drop-neutral 估计的是 P(sign(r) | X, |r| > τ)，不是 P(sign(r) | X)
- 这是条件分类任务，不是全市场方向预测
- 必须在报告中声明

---

## 3. 执行步骤（顺序已确定）

```
Step 0:  Preflight — 创建 GitHub private repo, push, 创建 phase-1b branch
Step 1:  ml_utils 加 make_no_trade_band_labels（完成，c7162c2）
Step 2:  ml_utils config 扩展（完成，d9352fc）
Step 3:  capacity profiling（下一步）
Step 4:  根据 profiling 结果锁定 window/k/threshold/kernel strategy
Step 5:  下载 TCN + DLinear reference 代码到 reference_excerpts/
Step 6:  ml_utils 建 tcn_classifier.py（construction plan §5.7, test-first）
Step 7:  ml_utils 建 dlinear_classifier.py（§5.8 + research 约束, test-first）
Step 8:  Notebook 03 — baseline 对比（import ml_utils）
Step 9:  下载 FEDformer reference 代码
Step 10: ml_utils 建 ms_dlinear_tcn_classifier.py（test-first）
Step 11: Full experiment notebook
Step 12: 给 Ian 汇报结果
```

当前位置：Step 3 未开始；TCN/DLinear/improved model/Notebook 03 仍然 absent。

---

## 4. 项目文件清单

### 已完成的 MVP production files

```
ml_utils/config.py
ml_utils/seed.py
ml_utils/metrics.py
ml_utils/dataset.py
ml_utils/checkpoint.py
ml_utils/models/lstm_classifier.py
ml_utils/trainer.py
tests/（107 tests passing, 1 known scheduler-order warning）
notebooks/01_smoke_test_single_stock_lstm.ipynb
notebooks/02_smoke_test_pooled_lstm.ipynb
```

### Phase 1B 已创建/修改的文件

```
tests/test_no_trade_band_labels.py          （Step 1 完成）
ml_utils/dataset.py                         （Step 1 新增 make_no_trade_band_labels）
ml_utils/config.py                          （Step 2 新增 label_mode / threshold_bps）
tests/test_config.py                        （Step 2 覆盖 config label fields）
```

### Phase 1B 后续需要创建的文件

```
ml_utils/models/tcn_classifier.py          （Step 6）
ml_utils/models/dlinear_classifier.py       （Step 7）
ml_utils/models/ms_dlinear_tcn_classifier.py（Step 10）
notebooks/03_baseline_comparison.ipynb      （Step 8）
```

### Phase 1B 需要下载的 reference files

```
reference_excerpts/tslib_dlinear_model.py        （Step 5, before DLinear impl）
reference_excerpts/tslib_autoformer_encdec.py     （Step 5）
reference_excerpts/fedformer_autoformer_encdec.py （Step 9, before improved model）
reference_excerpts/tslib_exp_classification.py     （Step 9）
```

### 已有的 reference files（来自 MVP）

```
reference_excerpts/ltsf_data_loader.py
reference_excerpts/ltsf_dlinear_model.py
reference_excerpts/pytorch_tcn_core.py
```

### Phase 1B 计划文件（已写好，放入 project knowledge）

```
PHASE_1B_PLAN.md
PHASE_1B_RESEARCH_HANDOFF.md
deep-research-report (3).md
```

### 仍然 absent（正确状态）

```
ml_utils/models/tcn_classifier.py
ml_utils/models/dlinear_classifier.py
notebooks/03_*.ipynb
```

---

## 5. 研究状态

```
deep-research-report (1).md：早期探索，已被后续报告覆盖
deep-research-report (2).md：Phase 1B 文献综述，有效
deep-research-report (3).md：Phase 1B 工程参考，最精确，有效
进一步文献研究：不需要，STOP
```

---

## 6. 禁止事项（防漂移）

- 不要在 Colab 里用 TF/Keras 写 Phase 1B 模型代码
- 不要跳过 test-first 直接写实现
- 不要在 profiling 之前锁定 window_size
- 不要修改旧的 make_binary_labels_from_future_avg_return
- 不要在标准 DLinear baseline 里用偶数核或多尺度
- 不要把未验证的计划写进 hf_stock_clf_after_w7_memory.md
- 不要在没有 profiling 数据的情况下给 Ian 发邮件
- 不要做新的 Deep Research
- 不要做 readiness audit / 流程设计迭代 — 直接执行 Step 0

---

## 7. 新窗口开场白

复制以下内容到新窗口：

```
这是我当前 hf_stock_clf / ml_utils 项目的 Phase 1B 上下文文件。

你的角色是评审：批判式审查我的方案和代码，给 Codex 写可执行 prompt。

当前状态：
- MVP ml_utils 库已完成；Phase 1B Steps 1-2 已提交
- Phase 1B 计划已同步（PHASE_1B_PLAN.md, PHASE_1B_RESEARCH_HANDOFF.md）
- Research A 已完成（3 份 deep research report）
- 最新验证：107 passed, 1 known pre-existing scheduler-order warning
- 下一个动作：Step 3 capacity profiling

关键约束：
- 走 ml_utils PyTorch 库路线，不走 Colab TF/Keras
- test-first，遵循 ml_utils_construction_plan_v2.md
- 数据在 Drive，代码推 GitHub，Colab clone 后 import ml_utils
- DLinear 标准 baseline 必须用单奇数核、individual=False
- 偶数核需要 FEDformer-style padding，不能直接用 TSLib moving_avg
- profiling 之前不锁 window_size

请先确认你理解了上下文，然后帮我执行 Step 3，或者我告诉你我已经完成了哪一步。
```

---

## 8. Ian 的原始要求（原文摘要）

1. 改进标签：用 real no-trade band，删除小收益样本，剩余分为 Up/Down，试几个阈值
2. 保留 LSTM, TCN, standard DLinear 作为 baselines
3. 构建 stock-aware multi-scale DLinear + residual TCN branch
   - 多尺度 MA 分解（windows 3, 6, 12, 24）
   - DLinear 线性层处理 trend
   - 小 TCN 分支处理 residual（dilation 1, 2, 4）
