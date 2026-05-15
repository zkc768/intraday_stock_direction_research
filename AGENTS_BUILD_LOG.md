# AGENTS.md 迭代记忆文件

> 用途：把 hf_stock_clf / ml_utils 项目的 AGENTS.md 演进过程、关键决策、被拒提案、未完事项凝固下来。
> 给谁看：(a) 你本人下次继续改 AGENTS.md 时调取；(b) 在新的 Claude / Codex 会话里贴上这个文件可以快速 onboard。
> 维护方式：每次大改 AGENTS.md 后，更新 §2 当前版本快照、§3 决策记录、§5 待办。其他章节稳定。

---

## 1. 项目身份卡

```
项目名:        hf_stock_clf
库名:          ml_utils
任务:          高频股票方向二分类（5-min bar、5 只股票 → 后期 80 只 NASDAQ）
学术背景:      Northeastern 毕设，supervisor: Professor Ian
当前阶段:      Phase: AGENTS.md 设计 + plan v2 同步，未开始 ml_utils 任何模块实现
工作模式:      本地 Codex CLI（AGENTS.md 自动加载）+ Google Colab Pro（notebooks）
```

---

## 2. 当前文件清单与版本快照

| 文件 | 路径 | 版本 / 状态 | 用途 |
|------|------|-----------|------|
| AGENTS.md | 仓库根 | **v4**（2026-05-14） | Codex 每会话自动加载的硬约束基线 |
| ENVIRONMENT.md | docs/ | **模板已交付，待用户填** | 记 PYTHON_INTERPRETER + 校验命令 |
| ml_utils_construction_plan_v2.md | docs/ | **待打 6 个 patch**（见 plan_v2_patches.md） | 模块级实施细节 |
| plan_v2_patches.md | docs/（或临时目录） | 已交付 | 同步 plan v2 与 AGENTS.md v4 的补丁集 |
| requirements.txt | 仓库根 | **未生成** | 依赖版本单一 source of truth |
| tests/conftest.py | tests/ | **未生成** | 分层 fixture，等 §14.2 step 创建 |
| pytest.ini | 仓库根 | **未生成** | `integration` marker 注册，等 §14.2 step 创建 |
| reference_excerpts/ | 仓库根 | **未确认状态** | 三个第一档参考文件 |

### AGENTS.md v4 章节地图（775 行）

```
§1   项目身份
§2   第一阶段范围
  §2.1   数据 / §2.2 任务 / §2.2.1 标签公式 / §2.3 模型 / §2.4 评估 / §2.5 不做项
§3   不可违反的硬约束（BLOCKER）
  §3.1 时间顺序泄漏防控（9 条）/ §3.2 不 hardcode / §3.3 数据模型解耦 / §3.4 DataFrame schema 三阶段
§4   环境与版本
  §4.1 启动前 TODO（PYTHON_INTERPRETER + requirements.txt）
§5   目录结构
§6   代码规范
  §6.1 import 白名单 / §6.2 tensor 形状 / §6.3 行预算 / §6.4 注释 / §6.5 config 自验证 / §6.6 print 纪律
§7   Agent 工作合同
  §7.1 范围 / §7.2 无未来代码 / §7.3 无静默修复 / §7.4 参考实现 / §7.5 测试先行 / §7.5.1 测试质量门槛 / §7.6 API 验证
§8   反模式速查表 L1-L7 / E1-E6 / W1-W6 / D1-D3
§9   会话工作流程
  §9.1 pre-flight / §9.2 单模块流程 / §9.3 self-review / §9.4 测试命令 / §9.5 测试数据规则 / §9.6 漂移防控 / §9.7 git
§10  License 合规（统一禁止代码块拷贝）
§11  文档与产出物（CHANGELOG 在会话外）
§12  模块完成验收标准（7 项）
§13  Notebook 规则（13.1 结构 / 13.2 必含 / 13.3 交付前清理）
§14  首次启动准备工作
  §14.1 readiness audit（双 shell 命令）/ §14.2 testing infrastructure setup
§15  一句话总结
```

---

## 3. 关键决策日志（一经固化不要回头讨论）

| 决策点 | 已固定值 | 出处 |
|------|---------|------|
| Label 公式 | 算术平均 `mean(r_{t+1..t+k})`，**禁止**几何平均、总收益、三分类 | AGENTS §2.2.1，plan v2 patch 1 |
| 零收益归类 | `future_avg_r == 0` → class 0 (non_up) | AGENTS §2.2.1 |
| 类别语义 | 0 = non_up（含下跌和平盘）, 1 = up | AGENTS §2.2.1 |
| 跨交易日 | 无条件禁止，**无 config escape hatch**；测试名固定 | AGENTS §3.1 第 9 条 |
| Schema 校验 | 三阶段（raw / labeled / post-trim） | AGENTS §3.4，plan v2 patch 2 |
| Timezone | dtype 用 `is_datetime64_any_dtype`，policy 由 `DataConfig.timezone_policy` 控制 | AGENTS §3.4 |
| 版本 source of truth | `requirements.txt` 单一权威，ENVIRONMENT.md 不重复列版本 | AGENTS §4.1 |
| Baseline 数量 | 固定 4 个：dummy_stratified（10 seed）、dummy_prior、always_up、always_down | AGENTS §2.4，plan v2 patch 5 |
| 核心指标 | macro F1 + balanced accuracy；`delta_macro_f1_vs_dummy` 强制出现 | AGENTS §2.4 |
| 测试 marker | 用 `integration`，**不**用 `slow` | AGENTS §9.4 |
| Shuffled-label sanity check | **只**在 notebook 02 实现，不在 pytest 重复 | AGENTS §13.2，plan v2 patch 4 |
| Tensor 形状约定 | 库统一 NLC `(batch, seq_len, features)`；Conv1d 转置在模型内部做 | AGENTS §6.2 |
| 实现会话允许文件 | `[target_file, *test_file(s)]`；CHANGELOG / conftest / pytest.ini **不**在内 | AGENTS §7.1 |
| 未实现模块的测试 | 禁止提前存在 | AGENTS §7.1 末条 |
| Self-review | 必须 fresh 会话；只列 issue 不修 | AGENTS §9.3 |
| Git 行为 | 禁 commit / push / 建分支；开始 `git status`，结束 `git diff --stat` | AGENTS §9.7 |
| License 处理 | 一律不拷贝代码块；MIT / Apache 也不行 | AGENTS §10 |
| AGENTS vs plan v2 冲突 | AGENTS 为准；若影响 signature 或测试，停下问 | AGENTS preamble |

---

## 4. 评审反馈历史（共 3 轮）

每轮反馈都做了「采纳 / 部分采纳 / 拒绝」三类标注。下面只记**拒绝**和**部分采纳**的决策，避免下一轮重新提起。

### 轮次 1 → v2

全部 8 条采纳，无拒绝项。

### 轮次 2 → v3

**拒绝**：

- **三阶段会话切分**（测试 / 实现 / CHANGELOG 各开独立会话） — 拒绝理由：会话切碎丢失上下文；§7.1 + §11 的"两文件 + 用户批准 gate"已足够。
- **每 3 步强制打印漂移检查** — 拒绝理由：仪式化，违反 §7.2 反屎山精神；评审者自己也警告"会变啰嗦"。§9.6 改为"5-8 步或自觉模糊时"+ 4 字段模板。

**部分采纳**：

- baseline 协议完整 column 表 — AGENTS 只保留 delta_f1 硬性要求；完整 schema 归 plan v2 metrics.py
- trainer 日志规则 — AGENTS 只保留通用规则 §6.6；模块特定细节归 plan v2

### 轮次 3 → v4

无拒绝项；评审者基本在找我自己引入的内部冲突，全部已修。

**plan v2 未同步**（无法直接改 plan v2）：

- 已生成 `plan_v2_patches.md`（6 个 patch），需用户人工应用。
- patch 应用完成的验证 grep 词表见该文件末尾。

---

## 5. 待办清单（按依赖顺序）

启动 Codex 写代码前必须按顺序完成：

```
TODO 1  用户  填写 docs/ENVIRONMENT.md 中的 PYTHON_INTERPRETER         ← 阻塞
TODO 2  用户  生成 requirements.txt（干净 venv + pip freeze + 清洗）    ← 阻塞
TODO 3  用户  应用 plan_v2_patches.md 中的 6 个 patch 到 plan v2        ← 阻塞
TODO 4  用户  把 AGENTS.md / ENVIRONMENT.md / plan v2 放进仓库 git commit
TODO 5  Codex  跑 AGENTS §14.1 readiness audit（只读会话）              ← 不阻塞但必经
TODO 6  Codex  跑 AGENTS §14.2 testing infrastructure setup            ← 不阻塞但必经
TODO 7  Codex  实施 plan v2 §6 step 1：config.py（第一个真正的模块）
```

TODO 5 之前都不会写任何 `ml_utils/*` 代码。

### AGENTS.md 自身的可能继续改动方向（非紧急）

如果你在 TODO 5 之前还想继续动 AGENTS.md，这些是评审过程中提到、目前**没采纳**或**最低限度采纳**、有空再加无空也行的项：

- §10 License：可考虑独立成 `docs/LICENSE_NOTES.md`，AGENTS 只留一句话指向
- §13 Notebook 规则：可独立成 `docs/NOTEBOOK_RULES.md`
- §7.6 API 验证：可加"必须展示 `inspect.signature` 输出"作为证据要求（评审轮 1 item 10，当时未采纳，理由是怕仪式化）
- §6.5 config 验证：可考虑同时添加 WindowConfig / TrainConfig 完整 __post_init__ 示例（目前只有 AGENTS §6.5 给清单，plan v2 patch 3 给 DataConfig 完整代码）
- Trainer / metrics 接口边界更明确：可在 plan v2 §5.5 trainer.py 中规定 "trainer.evaluate 必须调用 metrics.compute_classification_metrics"（评审轮 2 item 15）

不建议改动方向（已明确拒绝过）：

- 不要再把会话切分得更碎（测试 / 实现 / CHANGELOG 三独立会话）
- 不要再加"每 N 步强制打印 X"类周期性仪式
- 不要在 AGENTS.md 重新引入 cross_day_allowed escape hatch

---

## 6. 反屎山核心信条（核心三条，凡新增规则先过这三关）

任何新加规则必须同时通过：

1. **可执行**：agent 看完就知道做什么，不需要再推断
2. **可验证**：用户能在一次审计中查到是否被违反
3. **不仪式化**：不为了"看起来负责"而要求 agent 反复输出 meta-commentary

任何想加的规则不过这三关 → 不要加。

---

## 7. 工程上下文（不太会变）

- 数据：5 只股票 5-min bar，OHLCV + 技术指标（MACD 族、RSI-14、Bollinger %B、rolling std、OBV ROC）
- 模型：LSTM / TCN / DLinear 三个 classifier
- 锚定论文：Parmar et al. (2018), Jin et al. (2019)
- 之前已完成阶段：Phase 1-2 + hyperparam tuning（结论 vanilla LSTM 不能可靠提取信号，macro F1 0.24-0.30）
- 当前 Phase：建库为下一轮（pooled multi-stock + multi-model 对比）做工具底座
- 后期 Phase：EMD 去噪 + attention（来自 Jin 2019）

---

## 8. 给下一次 Claude 会话的开局提示

如果你在新 Claude 会话里要继续这个项目，按这个顺序贴文件：

```
1. 本文件（AGENTS_BUILD_LOG.md）—— 让 Claude 知道项目现状和已固化决策
2. AGENTS.md 当前版本（v4 或更新）
3. 评审者新一轮反馈（如果有）
```

然后用类似下面的开场白：

> 我在做 hf_stock_clf / ml_utils 项目的 AGENTS.md 迭代。当前是 v4。
> 这是迭代记忆文件，是当前 AGENTS.md，是评审者新一轮反馈。
> 请按和之前一样的方式：先列判断表（采纳 / 部分采纳 / 拒绝 + 理由），再做 str_replace 改动，最后给我汇总。
> 不要重新讨论 §6 反屎山核心信条已经拒绝过的方向。

这样可以避免新会话重新犯老问题，也保留之前几轮的判断脉络。

---

## 9. Gate 2 sync — AGENTS.md v4.2

- AGENTS.md v4.2 sync completed.
- AGENTS_VERSION marker introduced on line 2 of AGENTS.md:

```markdown
<!-- AGENTS_VERSION: v4.2 -->
```

- config.py 例外 introduced in AGENTS.md §9.1:
  - config.py 的 spec source 是 docs/ml_utils_construction_plan_v2.md §3.1，不是 §5.x。
  - W2 config sessions 按 §3.1 抽取 fields / line_budget / acceptance_criteria。
- PYTHON_INTERPRETER rule:
  - All local Python and pytest commands must use the interpreter recorded in docs/ENVIRONMENT.md.
  - Current expected interpreter: `PYTHON_INTERPRETER = E:\codex_workspace\_envs\py311_shared\python.exe`
- MVP_YES / Phase 1B decision summary:
  - MVP scope is LSTM-only end-to-end.
  - TCN and DLinear are deferred to Phase 1B.
