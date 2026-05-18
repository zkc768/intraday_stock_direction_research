# hf_stock_clf 项目 Skills 使用说明

更新时间：2026-05-18

项目路径：

E:\codex_workspace\projects\hf_stock_clf

> 本文件是 skills 使用指南，不是硬约束源。若与 `AGENTS.md`、`SPRINT_LOG.md`、当前 memory 或当前 sprint prompt 冲突，以后者为准。

本文用于记录 `hf_stock_clf / ml_utils` 项目中可用 skills 的推荐用途和禁用场景。使用任何 skill 时，仍必须以当前 sprint scope、`AGENTS.md`、`SPRINT_LOG.md` 和当前 memory 为最高优先级。

## 当前阶段

当前项目处于 W7 cleanup / docs governance 阶段。

已完成：

- MVP LSTM 工具库
- Notebook 01 / 02 smoke phase
- W6 final validation record
- W7.2 pandas FutureWarning cleanup

当前不要做：

- 不要直接进入 Phase 1B
- 不要创建 Notebook 03
- 不要创建 TCN / DLinear 文件
- 不要创建 TCN / DLinear tests
- 不要运行 heavy training
- 不要用 notebook workaround 掩盖 `ml_utils` public API 问题

## 已安装 skills

### pytest-testing

用途：

- focused pytest 设计
- warning cleanup regression tests
- dtype、immutability、split boundary、NaN label marker 断言
- 小范围测试维护

不要用于：

- 自动扩大测试范围
- 给尚未进入的 Phase 1B 提前写 tests
- 用 full pytest 或 notebook execution 替代 readiness audit

### python-expert-best-practices-code-review

用途：

- 小 patch 后做 fresh code review
- 检查 public API 是否被无意修改
- 检查是否违反 `AGENTS.md`
- 检查 pandas / numpy / torch 代码是否有 dtype、copy、shape、device 或 leakage 风险

不要用于：

- 泛泛重构整个代码库
- 推动 callback / plugin / hook 等项目明确不做的抽象
- 把审查和实现混在同一轮

### documenting-python-libraries

用途：

- README / MVP documentation pass
- 记录 `ml_utils` 稳定 public API
- 整理 data contract、label semantics、metrics、baseline semantics
- 总结 notebook smoke phase 结果

不要用于：

- 在代码行为未稳定时提前写长文档
- 把未来 TCN / DLinear / Notebook 03 写成已经完成
- 编造结果、指标、路径或实验结论

### deep-learning-pytorch

用途：

- Phase 1B readiness audit PASS 之后，用于 TCN / DLinear / PyTorch 模型实现
- 检查 input/output shape、loss/logits 约定、device handling
- 帮助写 tiny synthetic smoke tests

当前阶段不要用于：

- 不要创建 `ml_utils/models/tcn_classifier.py`
- 不要创建 `ml_utils/models/dlinear_classifier.py`
- 不要创建 Notebook 03
- 不要直接进入模型 implementation

### time-series-analysis

用途：

- 审查时间序列窗口、horizon、split boundary
- 审查标签是否 forward-looking
- 后续研究 trend / residual decomposition

不要用于：

- 改变当前固定二分类 label 公式
- 引入三分类 no-trade band，除非当前 sprint 明确允许设计变更
- 允许 random split、跨股票窗口、跨交易日窗口

### aeon

用途：

- Phase 1B 或更后面作为 time-series classification 概念参考
- 对照时间序列分类任务、shape、baseline 思路

不要用于：

- 直接加入项目依赖
- 修改 `requirements.txt`
- 把当前小型 PyTorch 工具库变成通用时间序列框架

## 已有本地 skills，也适合本项目

### pytorch-ml-utils-builder

用途：

- 在已有明确 audit 结论和 patch scope 后，维护 `ml_utils`
- 小范围维护 `seed.py`、`dataset.py`、`metrics.py`、`checkpoint.py`、`trainer.py`、`models/`
- 保护 ML validity：chronological split、train-only scaler fit、shape checks、named metrics

限制：

- 必须先读 `AGENTS.md`
- 必须使用项目指定 Python
- 不跑 heavy training
- 不复制外部 repo 代码

### notebook-code-reviewer

用途：

- 只读审查 Notebook 01 / 02 的 hidden state、指标、路径、checkpoint、泄漏风险
- 未来 Notebook 03 创建前或创建后做只读审查

限制：

- 当前不要创建 Notebook 03
- notebook 不能掩盖 `ml_utils` public API 问题

### github-repo-miner

用途：

- Phase 1B readiness / reference audit
- 只读审查 TCN / DLinear / Time-Series-Library 等外部参考 repo

限制：

- 当前 W7 cleanup 阶段不需要启动
- 不要把外部 repo 结构直接搬进本项目

## 推荐使用顺序

### W7 cleanup

推荐：

- pytest-testing
- python-expert-best-practices-code-review
- pytorch-ml-utils-builder

目标：

- 只读定位 warning
- 设计最小 patch
- 修复后只跑 focused tests
- fresh review 后再决定是否记录到 `SPRINT_LOG.md`

### W7 documentation pass

推荐：

- documenting-python-libraries
- python-expert-best-practices-code-review

目标：

- 更新 README
- 写清 MVP scope、data contract、label semantics、metrics、baseline、notebook smoke status
- 不写未来模型已完成的表述

### Phase 1B readiness / reference audit

仅在用户明确决定进入 Phase 1B 后使用：

- github-repo-miner
- time-series-analysis
- deep-learning-pytorch
- aeon

目标：

- 只读审查 TCN / DLinear / Time-Series-Library 参考实现
- 明确 license、shape、API、测试边界
- 形成 Phase 1B plan

### Phase 1B implementation 之后

推荐：

- deep-learning-pytorch
- pytest-testing
- python-expert-best-practices-code-review
- pytorch-ml-utils-builder

目标：

- 每个模型单独 patch
- tiny synthetic smoke tests 优先
- 不跑 heavy training，除非用户明确要求

## 项目红线

无论使用哪个 skill，都必须遵守：

- 先读项目 `AGENTS.md`
- 使用项目指定 Python：`E:\codex_workspace\_envs\py311_shared\python.exe`
- 不编造结果、指标、路径、实验结论
- 不删除 raw data
- 不修改 active project 外文件，除非用户明确要求
- 不跑 heavy training，除非用户明确要求
- 当前不要直接进入 Phase 1B
- 当前不要创建 Notebook 03
- 当前不要创建 TCN / DLinear 文件
- 审查、实现、提交不要混在同一轮

## 新会话开场模板

如果希望 Codex 在后续会话按这些 skills 工作，可以这样开头：

请先读取 `E:\codex_workspace\projects\hf_stock_clf\AGENTS.md` 和当前 memory。
本项目是 `hf_stock_clf / ml_utils`，高频股票方向二分类 PyTorch 工具库。
当前优先级仍以 memory 中的 sprint scope 为准，不要自行扩大范围。

本项目可用 skills：

- pytest-testing：用于 focused pytest 设计和测试维护
- python-expert-best-practices-code-review：用于小 patch code review
- documenting-python-libraries：用于 README / library docs
- deep-learning-pytorch：仅在 Phase 1B readiness PASS 后用于 TCN / DLinear / PyTorch 模型
- time-series-analysis：用于时间序列窗口、horizon、split / leakage 审查
- aeon：用于时间序列分类参考，不要直接引入依赖
- pytorch-ml-utils-builder：用于 `ml_utils` 小范围构建或维护
- notebook-code-reviewer：用于 notebook 只读审查
- github-repo-miner：用于外部 ML repo 只读审查

请严格遵守 `AGENTS.md`，先报告 scope，再行动。
