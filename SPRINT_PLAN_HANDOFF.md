# SPRINT_PLAN_HANDOFF.md v4.1 — ml_utils MVP 实施前 gate 文件

> 文件用途：把 ml_utils 实施前的所有 gate、决策与近期 5 个 session prompt 凝固到一处。
> 给谁看：(a) 用户在新窗口继续推进时调取；(b) 在新 Claude 会话里贴上本文件 + AGENTS.md v4.2 + AGENTS_BUILD_LOG.md 即可 onboard。
> 项目根：`E:\codex_workspace\projects\hf_stock_clf\`。本文件中所有引用路径均相对此根（Windows 风格），不引用任何 sandbox 路径。
> 维护方式：每次大状态变化后更新 §1 状态快照、§4 gate 表进度、§9 SPRINT_LOG.md 镜像。
> 与 AGENTS_BUILD_LOG.md 的分工：AGENTS_BUILD_LOG 记 AGENTS.md 演进；本文件记 ml_utils 实施层 sprint 计划。

> **v4.1 修订摘要**（与 v4 差异，1 结构升级 + 1 conditional WARNING 落地）：
> (1) 新增 §7.D test-first 通用约束（lazy import 规则），含适用范围表与 §B 引用模板；§10.3 W2.1 简化为一行引用
> (2) §1 状态快照中 AGENTS.md v4.2 行追加 §9.1 例外条款待办（config.py spec source 例外，避免 W2 pre-flight 机械找 §5.x config 时停下问）

> **v4 修订摘要**（与 v3 差异，2 BLOCKER + 4 WARNING）：
> (1) §6.B 同步 W2.2 allowed_files 删除 `tests/test_config.py`（与 §10.4 对齐）
> (2) §10.3 W2.1 加 test-first 通用约束：禁止 module top-level import 目标模块，必须 lazy import 至 function 体内（v4.1 已抽出为 §7.D）
> (3) Gate 7 + §9 SPRINT_LOG 模板限定为第三方依赖审计，stdlib 排除
> (4) §12 与 §2.3 中 `v2` → `v3/v4` 引用同步
> (5) §10.1 W0.1 加 shell 选择条款（Git Bash vs PowerShell 二选一，整轮一致）

> **v3 修订摘要**（与 v2 差异，保留作为历史记录）：
> (1) W1.1 plan v2 anchor 改为 AGENTS §14.2，fixture 名按 §14.2 verbatim
> (2) W2.1/W2.2/W2.3 plan v2 anchor 由 `§5.2` 改为 `§3.1`（config 真实编号）
> (3) Gate 6/8 死锁解开：SPRINT_LOG 创建提前到 whitelist audit 之前
> (4) Gate 7 拆 MVP 必需（仅 `ltsf_data_loader.py`）/ Phase 1B 推迟
> (5) W0.1 第 10 项允许 `__init__.py` 两个
> (6) AGENTS_VERSION 显式 marker；W1.1 collect-only 不写死 0；W2.1 五类按 §7.5.1 verbatim；W2.2 移除 test_config.py

---

## §0. MVP 决策（已决，不可回退至"未决"状态）

**Decision: MVP_YES**

本 sprint 采用 MVP 路径：先跑通

```
ml_utils/config.py
ml_utils/seed.py
ml_utils/metrics.py
ml_utils/dataset.py
ml_utils/checkpoint.py
ml_utils/models/lstm_classifier.py
ml_utils/trainer.py
```

形成可测试、可复现、可端到端训练的 LSTM binary classification baseline。

**Deferred scope（Phase 1B，本 sprint 内禁触）**：
- `ml_utils/models/tcn_classifier.py`
- `ml_utils/models/dlinear_classifier.py`
- 对应 test 文件
- `reference_excerpts/pytorch_tcn_core.py` 与 `reference_excerpts/ltsf_dlinear_model.py` **本 sprint 不要求 vendor**；Phase 1B 启动前再补

**强制规则**：

1. 本 sprint 期间不允许在 `ml_utils/` 创建 `tcn_classifier.py` / `dlinear_classifier.py` 的空文件或占位文件
2. `trainer.py` / `config.py` 中不得预留 "for tcn/dlinear future use" 的分支
3. `tests/` 不得提前生成 tcn / dlinear 测试骨架
4. 任何"为未来扩展"的 if / elif / config flag → 立即 STOP

**plan v2 同步要求**：

`docs/ml_utils_construction_plan_v2.md` §6 必须在 patch 应用阶段同步增加一段：

```
§6.x MVP_YES 说明（本 sprint）

本轮实施仅落地 lstm 通道。tcn / dlinear 推迟至 Phase 1B。
本节存在的目的是让 AGENTS §7.2 "无未来代码" 与 plan v2 "三模型对比"
的最终目标共存：MVP 阶段允许只实现 lstm，但不允许写任何 tcn/dlinear
未来代码（占位文件、未使用 if 分支、未注册 config 字段等）。
```

未在 plan v2 同步登记 MVP_YES → §4 Gate 3 FAIL。

**最终目标不变**：项目最终仍是 lstm / tcn / dlinear 三模型对比。MVP 只是第一轮实施路径选择，用于先验证数据管线、指标体系、训练循环、checkpoint 与测试基础设施。

---

## §1. 状态快照（2026-05-15）

| 项 | 状态 | 备注 |
|---|---|---|
| AGENTS.md | **v4.2 待落地到项目根** | 项目根路径 `E:\codex_workspace\projects\hf_stock_clf\AGENTS.md`，需替换项目内 v4；落地时第 2 行加 `<!-- AGENTS_VERSION: v4.2 -->` marker；**§9.1 pre-flight 末尾加 config.py 例外条款**（"Exception: config.py 的 spec source 是 plan v2 §3.1，不是 §5.x；W2 config sessions 按 §3.1 抽取 fields / line_budget / acceptance criteria"），否则 W2.x pre-flight 会因机械找 §5.x config 而停下问 |
| AGENTS_BUILD_LOG.md | 待补 v4.2 节点 | §2 表格 / §3 决策日志 / §4 评审反馈轮次 4 |
| ENVIRONMENT.md | **未填**（仍是 `<在这里填写完整路径>`） | 阻塞所有 ml_utils 实施 |
| requirements.txt | 未生成 | 用 PYTHON_INTERPRETER `pip freeze` 后人工锁 `==` 版本 |
| plan_v2_patches.md | 已生成，**未应用到 plan v2** | 6 patch + patch 4 test #7 话术改 + §0 MVP_YES 段同步 |
| docs/ml_utils_construction_plan_v2.md | 存在但未与 AGENTS.md v4.2 同步 | 待 6 patch 应用 + MVP 段补入 |
| reference_excerpts/ltsf_data_loader.py | **未确认到位** | MVP 必需（dataset.py 用），plan v2 §2.1 第一档 |
| reference_excerpts/pytorch_tcn_core.py | 推迟 | Phase 1B 必需，MVP 不阻塞 |
| reference_excerpts/ltsf_dlinear_model.py | 推迟 | Phase 1B 必需，MVP 不阻塞 |
| tests/conftest.py & pytest.ini | 未生成 | 由 W1.1 创建，fixture 名按 AGENTS §14.2 verbatim |
| SPRINT_LOG.md | **未创建** | 阻塞段 Gate 6 创建最小骨架（§9 模板，先于 whitelist audit）|
| ml_utils/__init__.py & ml_utils/models/__init__.py | 状态未知 | W0.1 #10 允许这两个存在 |
| 其他 ml_utils/ 模块 | 全未开始 | 0 行 production code |

**阻塞链**（顺序与 §4 一致，依次必须 PASS）：

```
MVP 决策 → AGENTS v4.2 → BUILD_LOG v4.2 → plan v2 patches
       → ENVIRONMENT → requirements → SPRINT_LOG → whitelist audit
       → MVP reference_excerpts → atomic commit → W0.1 audit
```

---

## §2. AGENTS.md v4.2 改动记录（指向项目根，不指向 sandbox）

落地动作：把 v4.2 全文写入 `E:\codex_workspace\projects\hf_stock_clf\AGENTS.md`。本节只记改动清单，全文不在此重复。

### 2.1 v4 → v4.1（上一批 3 项）

| 章节 | 改动 |
|---|---|
| §2.3 模型 | "三个 baseline：DummyClassifier(strategy=prior/stratified)、always-up、always-down" → "Baseline 数量与命名由 §2.4 唯一定义（4 个）" |
| §2.2.1 末尾 NaN 一条 | 单行"必须由 trim_labels_at_split_boundary 处理后丢弃" → 4 子条（trim 追加 NaN 不删行 / 跨日 NaN 标注 / Dataset 跳过 NaN 起点 / 任何阶段禁 fillna 禁 dropna），与 §3.4 Stage 2/3 对齐 |
| §4.1 line 200 | "模板：参见仓库根目录 `ENVIRONMENT.md`" → "模板：见 `docs/ENVIRONMENT.md`" |

### 2.2 v4.1 → v4.2（本批 7 项）

| 章节 | 类型 | 改动 |
|---|---|---|
| §3.1 #4 | rewrite | "样本必须丢弃" → "标记为 invalid 并跳过 + label=NaN + Dataset 跳过 NaN 起点" |
| §3.2 末尾 | append | 三条 hardcode ban 例外说明：config dataclass 默认值 / tests fixture / notebooks orchestration |
| §7.5 | rewrite | 单测试文件话术 → `test_file(s)` 复数 + 指向 §7.1 §9.1 |
| §9.2 #3-4 | rewrite | `tests/test_M.py` → pre-flight 列出的全部 test_file(s) |
| §9.4 | rewrite + expand | 引入 `$PYTHON_INTERPRETER` 跨平台调用规则（Git Bash + PowerShell），所有命令换皮，禁裸 python 禁裸 pytest |
| §11 #1-2 | rewrite | 单文件 → `tests/test_<module>*.py` + 引用 pre-flight |
| §12 #3-4 | rewrite | `python -m pytest` → `"$PYTHON_INTERPRETER" -m pytest` + 指向 §9.4 |

行数：v4 776 → v4.1 779 → v4.2 801。

### 2.3 关键决策表新增条目（待补入 AGENTS_BUILD_LOG §3）

| 决策点 | 已固定值 | 出处 |
|---|---|---|
| 测试命令解释器 | 所有 pytest 通过 `$PYTHON_INTERPRETER` 显式调用，禁裸 `python -m pytest` 和裸 `pytest` | AGENTS §9.4 v4.2 |
| Hardcode ban 例外 | config dataclass 默认值、tests fixture、notebooks 顶层 cell 不受 §3.2 约束 | AGENTS §3.2 v4.2 |
| 测试文件复数 | test_file(s) 全 AGENTS 一致复数（§7.1/§7.5/§9.2/§11/§12 已对齐） | AGENTS v4.2 |
| 跨 split 边界处理 | "丢弃" 话术全库换为 "标记 invalid + Dataset 跳过 NaN 起点"，与 §3.4 Stage 2/3 同源 | AGENTS §3.1 #4 v4.2 |
| MVP 路径决策 | MVP_YES：第一轮 lstm-only，tcn/dlinear 推迟至 Phase 1B；plan v2 §6.x 同步登记 | SPRINT_PLAN_HANDOFF v4.1 §0 |

---

## §3. 连带不一致（在 plan v2 patch 阶段一并处理）

### 3.1 patch 4 test #7 话术

`plan_v2_patches.md` 的 Patch 4 test #7 文本：

```
7. 跨边界丢弃测试：标签区间跨 split 边界的样本被丢弃
```

与 v4.2 §3.1 #4 新话术不齐（"丢弃" 已禁用）。Patch 应用时改为：

```
7. 跨边界 invalid 标注测试：标签区间跨 split 边界的样本 label 被标注为 NaN
   （由 trim_labels_at_split_boundary 添加），且 Dataset 构造时不会以这些
   NaN 起点生成窗口
```

测试实现本身不变。

### 3.2 plan v2 §6.x MVP_YES 段补入

按 §0 末尾给出的 §6.x 文本，作为 patch 应用阶段的第 7 个 patch（plan_v2_patches.md 原本只有 6 patch，加 MVP 段后为 7 patch）。

---

## §4. TODO 顺序（gate 化）

每行必须显式有 PASS 判据与 FAIL 动作。FAIL 即 STOP，不允许 "报告失败后继续"。

| Step | Gate | PASS 判据 | FAIL 动作 |
|---|---|---|---|
| 0 | MVP decision | 本文件 §0 存在 `Decision: MVP_YES` 或 `Decision: MVP_NO` | STOP，用户必须先决策；本版已固定 MVP_YES |
| 1 | AGENTS.md v4.2 | `AGENTS.md` 存在且 `grep "<!-- AGENTS_VERSION: v4.2 -->" AGENTS.md` 命中 ≥ 1 行（marker 在第 2 行）| STOP，先按 v3 修订附记 1 在 AGENTS.md 第 2 行追加 `<!-- AGENTS_VERSION: v4.2 -->` 标记 |
| 2 | AGENTS_BUILD_LOG v4.2 | `E:\codex_workspace\projects\hf_stock_clf\AGENTS_BUILD_LOG.md` grep 同时命中：`v4.2`、`测试文件复数`、`PYTHON_INTERPRETER`、`invalid` | STOP，先补 build log §2/§3/§4 |
| 3 | plan v2 patches applied | 跑 `plan_v2_patches.md` 末尾验证 grep 词表全部通过（旧词 0 命中、新词全命中）；§6.x MVP_YES 段已补入，含 `MVP_YES` / `Phase 1B` / `lstm 通道` / `不允许写任何 tcn/dlinear 未来代码`；`ml_utils/models/tcn_classifier.py` 与 `ml_utils/models/dlinear_classifier.py` 不存在 | STOP，先同步 plan v2 |
| 4 | ENVIRONMENT active | `docs/ENVIRONMENT.md` grep `<在这里填写完整路径>` 0 命中，且至少 1 行包含真实 Windows 路径（`E:\...\python.exe` 或 `C:\...\python.exe`） | STOP，先填解释器 |
| 5 | requirements locked | `requirements.txt` 存在，所有非空非注释行均含 `==`，`grep ">=" requirements.txt` 0 命中 | STOP，先锁版本 |
| 6 | SPRINT_LOG ready | `SPRINT_LOG.md` 存在，含 §9 模板的 5 段：当前状态 / 已合并模块清单 / 当前 git 状态 / requirements vs 白名单交叉审计差异（空槽位）/ 进行中 session 注意事项 | STOP，按 §9 创建最小骨架 |
| 7 | requirements whitelist audit | SPRINT_LOG.md "requirements vs 白名单交叉审计差异" 段填入两侧清单（**仅审计第三方依赖**，stdlib 不在 requirements.txt 范围内，按 AGENTS §6.1 排除 `dataclasses` / `typing` / `pathlib` / `random` / `os` / `json` / `math` / `copy` / `collections` / `datetime` / `inspect` / `tempfile`）：`requirements 有但第三方白名单无` / `第三方白名单有但 requirements 无`；空集必须显式写 `无差异`，不允许留模板占位符 | STOP，先做交叉审计 |
| 8 | MVP reference_excerpts | `reference_excerpts/ltsf_data_loader.py` 存在（MVP 必需，dataset.py 用，plan v2 §2.1 第一档）。`pytorch_tcn_core.py` 与 `ltsf_dlinear_model.py` 推迟到 Phase 1B 启动前 vendor，**不在本 Gate 检查范围** | STOP，先补 `ltsf_data_loader.py` |
| 9 | atomic commit | `git status --short` 输出为空（或仅含明确允许的未跟踪文件，如 `.venv/`） | STOP，先 commit 或人工解释 |
| 10 | W0.1 ready | Gate 0–9 全部 PASS | 进入 W0.1 readiness audit |

**执行约束**：

- 用户在阻塞段任一 Gate FAIL → 不允许跳到下一 Gate
- 不允许在 Gate FAIL 状态下启动任何 Codex session
- W0.1 是 Gate 10 之后的第一个 agent session，本身仍是只读 audit，**不修改任何文件**

**v3 修订附记 1（AGENTS_VERSION marker）**：

AGENTS.md v4.2 落地到项目根时，**第 2 行必须加上**：

```
<!-- AGENTS_VERSION: v4.2 -->
```

放在第 2 行（紧跟 `# AGENTS.md — hf_stock_clf / ml_utils` 之下）。这是 Gate 1 唯一可靠的机器可检索标识，避免泛 grep `v4.2` 把正文里出现的版本号也算上。每次 AGENTS.md 升级版本时同步更新这一行。

**v3 修订附记 2（reference_excerpts 二层化）**：

按 plan v2 §2 三档分类：

- **第一档（vendor 入 `reference_excerpts/`）**：`ltsf_data_loader.py`（dataset 用，MVP 阻塞）；`pytorch_tcn_core.py` + `ltsf_dlinear_model.py`（Phase 1B 启动前 vendor，MVP 不阻塞）
- **第二档（不入 `reference_excerpts/`，仅 URL 备查）**：pytorch-template `base/base_trainer.py`（trainer / checkpoint 用）；Yutsuro `tisc/modules/LSTM.py`（lstm 用）
- **第三档（无需参考文件）**：seed / metrics / config

不要把第二档文件 vendor 进 `reference_excerpts/`。Phase 1B 启动前用户单独补 vendor 第一档的 tcn / dlinear 两个文件。

---

## §5. Wave 模型（参考用，MVP_YES 已剪裁）

```
W0  pre-flight gate    │ 只读 audit
W1  infra gate         │ conftest + pytest.ini
W2  config.py          │ 串行（所有后续模块依赖）
W3  seed.py            │ 串行（小模块，紧跟 W2）
W4  并行段（MVP_YES：仅 metrics / checkpoint / dataset / lstm）
W5  trainer.py         │ 串行（依赖 W4 全部）
W6  Phase 1B 起点      │ MVP 通过 W5 integration test 后启动 tcn + dlinear
```

**并行原则**：同时最多 2 个 active 模块，且必须错位状态（一个写测试 + 一个实现 / 一个实现 + 一个 review），禁止两个同时在 implementation。

**每个模块最少 3 个 session**：

```
S1  test-writing       (allowed: test_file(s))
    ↓ 用户审批 gate (离线)
S2  implementation     (allowed: target_file + test_file(s))
    ↓ fresh session gate
S3  self-review        (allowed: [], 只输出 issue list)
    ↓
S4  fix-up (可选)      (allowed: target_file + test_file(s))
```

`dataset.py` S2 拆 a/b/c 三个 sub-session：

- W4.B.2a：schema validation Stage 1/2/3 + `make_binary_labels_from_future_avg_return`
- W4.B.2b：`trim_labels_at_split_boundary` + 跨日 NaN 标注 + per-ticker split
- W4.B.2c：`WindowedClassificationDataset`（NaN 起点跳过）

---

## §6. Session 清单（两层）

### §6.A 完整 wave 表（参考用，本轮不立即生成 prompt）

| # | Session | 阶段 | 输出 | 估算 tool calls | 可并行 |
|---|---|---|---|---|---|
| W0.1 | audit | 只读 | 11 项审计报告 | 5-10 | — |
| W1.1 | infra-impl | 一次性 | conftest.py + pytest.ini | 5-8 | — |
| W2.1 | config-test | 测试 | tests/test_config.py | 3-5 | — |
| W2.2 | config-impl | 实现 | ml_utils/config.py + 测试全绿 | 5-8 | — |
| W2.3 | config-review | review | issue list | 2-4 | — |
| W3.1 | seed-test | 测试 | tests/test_seed.py | 2-3 | 可与 W2.3 并行 |
| W3.2 | seed-impl | 实现 | ml_utils/seed.py | 3-5 | — |
| W3.3 | seed-review | review | issue list | 2-3 | — |
| W4.A.1 | metrics-test | 测试 | tests/test_metrics.py | 4-6 | 可与 W4.B.1 并行 |
| W4.A.2 | metrics-impl | 实现 | ml_utils/metrics.py | 6-9 | 可与 W4.B.2 并行 |
| W4.A.3 | metrics-review | review | — | 2-4 | — |
| W4.B.1 | dataset-test | 测试 | 3 个 test 文件 | 8-10 | 占一并行槽 |
| W4.B.2a-c | dataset-impl | 实现 | dataset.py 三段 | 6-8 each | — |
| W4.B.3 | dataset-review | review | issue list | 4-6 | — |
| W4.B.4 | dataset-fixup | 修 issue | — | 3-6 | — |
| W4.C.1-3 | checkpoint × 3 | test/impl/review | | 3-5 each | 可与 W4.D 并行 |
| W4.D.1-3 | lstm × 3 | test/impl/review | | 4-7 each | — |
| W5.1-4 | trainer × 4 | test/impl/review/fixup | integration test 在此 | 6-10 each | — |
| ~~W4.E~~ | ~~tcn × 3~~ | **Phase 1B（推迟）** | — | — | — |
| ~~W4.F~~ | ~~dlinear × 3~~ | **Phase 1B（推迟）** | — | — | — |

**MVP_YES 估时**：5-8 个工作日（含用户审批时间），约 22 个 session；`dataset.py` 若边界问题（NaN 标注、per-ticker split、跨日 horizon）需要额外迭代，单独 +1-2 天。原 v2 写 4-6 天偏乐观，已修正。

### §6.B 近期真正产出的 5 个 prompt（§10 给出 B+C 草图）

```
W0.1   readiness audit            （只读，allowed_files=[]）
W1.1   testing infrastructure     （allowed_files=[tests/conftest.py, pytest.ini]）
W2.1   config-test                （allowed_files=[tests/test_config.py]）
W2.2   config-impl                （allowed_files=[ml_utils/config.py]，test_config.py 只读）
W2.3   config-review              （allowed_files=[], 输出 issue list）
```

**只生成这 5 个**。W3 及以后等 W2 完成后再生成；plan 早出会随实施学习作废，维护成本不值。

---

## §7. 通用 Prompt 三段拆分

每个 session prompt = §7.A 固定前缀（直接复用） + §7.B 本会话 B 段 + §7.C 本会话 C 段。

§7.A 不在每个 prompt 里重复哲学；AGENTS.md 是 contract，Codex 进项目目录自动加载，§7.A 只指向不复述。

### §7.A 固定前缀（约 18 行，全 session 共用）

```
你是 hf_stock_clf 项目的 implementation agent。
项目根 E:\codex_workspace\projects\hf_stock_clf\ 的 AGENTS.md 是硬约束基线。

# 启动前强制步骤（不要假设 AGENTS.md 已自动加载）

第 0 步：显式读取项目根 AGENTS.md，确认第 2 行有 `<!-- AGENTS_VERSION: v4.2 -->`。
        若 marker 缺失或版本号不匹配 → 停下问，不要继续。
第 1-7 步：按 AGENTS.md §9.1 完成 pre-flight，把输出贴给我，等我说 "go" 才进入 B 段。
第 7 步追加：读 SPRINT_LOG.md "已合并模块清单"，确认本会话可 import 范围。

# 通用硬禁令

修改 allowed_files 之外的文件、git commit/push/建分支/改 .gitignore、
写 TODO / 占位 / 假抽象 / "for future extensibility"、静默修复 / try-except 吞异常、
凭记忆使用 API（sklearn / torch.optim / pandas resample/groupby 尤甚）、
用裸 python 或 pytest（必须 "$PYTHON_INTERPRETER" -m pytest）—— 任一触发 → 停下问。

# 漂移与退出

超过 8 轮工具调用 → 按 AGENTS §9.6 输出 4 字段总结然后等指示。
退出条件见本 prompt §C。
```

§7.A 不复述 §9.1 内部步骤、不复述测试命令模板、不复述 review 输出格式——这些都在 AGENTS.md。

### §7.B 当前 session 目标段（每个 session 独有）

模板字段（每个 prompt 填四块）：

```
# 本会话身份
模块/阶段:        {module / phase}
plan v2 段:       {§x.y}
allowed_files:    {完整 Windows 路径清单}

# 本会话目标
{一两句话写清楚本轮做什么，不复述 plan v2}

# 本会话独有 do/don't
{只列与本 session 强相关的特异约束；通用规则不重复}
```

### §7.C 验收输出段（每个 session 独有）

模板字段：

```
# 必须输出
{该 session 的 deliverable，例如："tests/test_config.py 全文 + 测试运行结果"}

# 收尾命令（按顺序跑）
"$PYTHON_INTERPRETER" -m pytest tests/test_<module>*.py -v
git status --short
git diff --stat

# 收尾汇报格式
- 修改文件清单（路径绝对到 E:\... 或相对 repo root）
- 跑过的测试命令原文
- 测试通过/失败/跳过的具体数量（禁止只写 "tests pass"）
- 未解决问题（若有）
- 本会话 PASS / FAIL 自评（按本 prompt §C "必须输出" 是否全部满足）
```

### §7.D Test-first 通用约束（v4.1 新增）

**仅适用于 test-writing session**（W2.1 / W3.1 / W4.A.1 / W4.B.1 等所有"先写测试"轮）。Implementation 与 review session 不受此节约束。

**规则**：在目标模块尚未实现前，测试文件不得在 module top-level import 目标模块。

禁止：

```python
# tests/test_config.py 顶部
from ml_utils.config import DataConfig         # ← BLOCKER, 会触发 collection error
import ml_utils.config                          # ← 同上
```

必须改为 test function 或局部 helper 内 lazy import：

```python
# tests/test_config.py
import pytest

def test_normal_default_construction():
    from ml_utils.config import DataConfig     # ← lazy import, OK
    cfg = DataConfig()
    assert cfg.k == 12

def _build_invalid_config():
    from ml_utils.config import DataConfig
    return DataConfig(k=-1)

def test_error_negative_k():
    with pytest.raises(ValueError):
        _build_invalid_config()
```

**为何这是硬约束**：

`pytest --collect-only` 在 collection 阶段会 import 测试文件。Module top-level 写 `from ml_utils.X import Y` 时，如果 `ml_utils.X` 还不存在（test-first session 的定义），collection 立即 ImportError，整个 session 收尾必然 FAIL。Lazy import 把 ImportError 推迟到测试执行阶段——而 test-first session 的退出条件只检查 collect-only 通过，不要求 test pass。

**implementation session 后**：测试不需要回头改成 top-level import；lazy import 不影响测试运行正确性，只是少量重复 import 开销。保持 lazy 形式可以让 test-first session 与 implementation session 的代码结构差异最小，diff 干净。

**适用范围一次性写死**（未来 prompt 直接引用本节，不再复制此段）：

| Session | 是否适用 §7.D |
|---|---|
| W2.1 config-test | 是 |
| W3.1 seed-test | 是 |
| W4.A.1 metrics-test | 是 |
| W4.B.1 dataset-test (三个测试文件) | 是 |
| W4.C.1 checkpoint-test | 是 |
| W4.D.1 lstm-test | 是 |
| W5.1 trainer-test | 是 |
| W*.2 implementation sessions | 否（目标模块此时已创建） |
| W*.3 review sessions | 否（不写代码） |

**§B 引用模板**（未来 test-first prompt 在 do/don't 段只需写一行）：

```
- 遵守 SPRINT_PLAN_HANDOFF.md §7.D test-first 通用约束（lazy import 至 function 体内）
```

不必再复制完整规则。

---

## §8. W0.1 11 项审计清单（写死，不再"自由发挥"）

W0.1 是只读 readiness audit，`allowed_files = []`，**不允许修改任何文件**，仅输出报告。

报告必须按顺序覆盖以下 11 项，每项格式为 `[PASS|FAIL|N/A] 项描述 — 证据`：

```
1.  AGENTS.md 是否位于项目根，且为 v4.2 或更新
    证据：grep "<!-- AGENTS_VERSION: v4.2 -->" AGENTS.md 命中行号

2.  AGENTS_BUILD_LOG.md 是否已补 v4.2 节点
    证据：grep "v4.2" / "测试文件复数" / "PYTHON_INTERPRETER" / "invalid"

3.  docs/ml_utils_construction_plan_v2.md 是否已应用全部 plan_v2_patches
    证据：跑 plan_v2_patches.md 末尾的验证 grep 词表；附加 §6.x MVP_YES 段
          grep 命中 "MVP_YES" / "Phase 1B" / "lstm 通道"

4.  patch 4 test #7 是否已从"丢弃"改为 "invalid 标注 + Dataset 跳过 NaN 起点"
    证据：grep plan v2 相关位置无"丢弃"出现 + 命中新话术

5.  docs/ENVIRONMENT.md 是否已填真实 PYTHON_INTERPRETER，且不含占位符
    证据：grep "<在这里填写完整路径>" 0 命中 + grep "python.exe" ≥ 1 命中

6.  requirements.txt 是否存在，是否全部使用 == 锁版本，是否无 >=
    证据：wc -l + grep "==" 行数 + grep ">=" 0 命中

7.  SPRINT_LOG.md 是否存在且 5 段完整；requirements vs §6.1 第三方白名单
    交叉审计差异段是否已填（含 "无差异" 或具体清单），不允许留模板占位符
    （stdlib 不在审计范围）
    证据：ls SPRINT_LOG.md + grep 五段标题命中数 + 审计段内容片段

8.  reference_excerpts/ltsf_data_loader.py 是否存在（MVP 第一档必需）
    证据：test -f reference_excerpts/ltsf_data_loader.py
    说明：pytorch_tcn_core.py / ltsf_dlinear_model.py 推迟至 Phase 1B，
          本项不检查它们，即使缺失也不算 FAIL

9.  tests/conftest.py 与 pytest.ini 是否尚未提前污染，或状态是否符合 W1.1 预期
    证据：ls tests/ 输出；若存在则报告内容是否仅为空骨架（不应有任何 fixture）

10. ml_utils 是否仅包含允许的 __init__.py，未出现未经批准的 production module
    允许存在：
      - ml_utils/__init__.py
      - ml_utils/models/__init__.py
    除此之外，任何 ml_utils/**/*.py 文件在 W0.1 前出现均 FAIL。
    证据：find ml_utils -name "*.py" 的完整输出与允许清单对比

11. git status 是否 clean；如不 clean，列出所有未提交文件
    证据：git status --short 完整输出
```

**FAIL 处理**：W0.1 报告任一项 FAIL → 用户回到 §4 对应 Gate 修复 → 重跑 W0.1 → 全 PASS 后才能进入 W1.1。

W0.1 不允许给修复建议，只报事实。修复方案由用户决定。

---

## §9. SPRINT_LOG.md 最小骨架（用户在 Gate 6 创建此文件）

文件位置：`E:\codex_workspace\projects\hf_stock_clf\SPRINT_LOG.md`

初始内容（最小骨架，复制粘贴即可）：

```markdown
# SPRINT_LOG.md (ml_utils MVP 实施进度)

## 1. 当前状态

最近完成:     —（W0 之前）
正在进行:     阻塞段 Gate 0-10 用户侧执行
下一步候选:   W0.1 readiness audit
MVP 决策:     MVP_YES（lstm-only, tcn/dlinear 推迟至 Phase 1B）

## 2. 已合并模块清单 (Codex 可以安全 import)

（空，无任何 ml_utils 模块已合并）

## 3. 当前 git 状态

- 工作目录干净: 待 Gate 9 确认
- 当前分支: main
- 与 origin: 待确认

## 4. requirements vs 白名单交叉审计差异（Gate 7 用户填）

**仅审计第三方依赖。stdlib（dataclasses / typing / pathlib / random / os / json / math / copy / collections / datetime / inspect / tempfile）不在 requirements.txt 范围，不参与审计。**

第三方白名单参考（按 AGENTS §6.1）：`torch` / `numpy` / `pandas` / `scikit-learn`（含 `sklearn.preprocessing` / `sklearn.metrics` / `sklearn.dummy` / `sklearn.base`）/ `pytest`。

requirements 有但第三方白名单无:
- （待填；若空集，删除"待填"改写 `无差异`）

第三方白名单有但 requirements 无:
- （待填；若空集，删除"待填"改写 `无差异`）

## 5. 进行中 session 注意事项

- 阻塞段未完成，所有 agent session 禁止启动
- W0.1 启动前必须 §4 Gate 0-9 全 PASS
```

每个 session 结束后用户手工更新（约 30 秒）：
- "最近完成" 改为刚完成的 session ID
- "已合并模块清单" 追加（仅在 review PASS + commit 后追加）
- "当前 git 状态" 同步

---

## §10. 近期 5 个 prompt 的 B+C 草图

§7.A 固定前缀直接复用，下方只列每个 session 的 §B + §C。复制时把 §7.A + 对应 §B + §C 拼起来即一个完整 prompt。

### §10.1 W0.1 readiness audit

**§B**：
```
# 本会话身份
模块/阶段:        W0.1 — readiness audit (只读)
plan v2 段:       — (无，本会话不实现任何模块)
allowed_files:    []  # 严格空集，禁止修改任何文件

# 本会话目标
按 SPRINT_PLAN_HANDOFF.md §8 列出的 11 项，依序输出 readiness audit 报告。
每项格式 [PASS|FAIL|N/A] 项描述 — 证据。
不给修复建议，只报事实。

# 本会话独有 do/don't
- 不允许写任何文件（包括 SPRINT_LOG.md 更新——那是用户的活）
- 不允许 git commit / git add
- 不允许凭记忆判断：每项必须给出 grep / ls / git 命令的真实输出片段作为证据
- 任一项无法确认（命令失败、文件不存在），写 FAIL 并说明，不写 "应该是 PASS"
- **Shell 选择**：§8 的 PASS 判据用 `grep` / `test -f` / `find` 默认 Git Bash 语法。
  若当前 shell 是 PowerShell，必须把命令等价改写为 PowerShell（`Select-String` /
  `Test-Path` / `Get-ChildItem -Recurse`）并在每项证据中注明所用 shell。
  跨平台命令对照见 AGENTS §9.4。两种 shell 输出语义需等价；选定一种后整轮一致，
  禁止半途切换。
```

**§C**：
```
# 必须输出
11 项审计报告（按 §8 顺序），每项 PASS/FAIL/N/A + 证据。
末尾给一句总结：[全 PASS | 有 N 项 FAIL，编号 X/Y/Z]

# 收尾命令
git status --short

# 收尾汇报格式
- 11 项 PASS/FAIL 计数
- 若有 FAIL，列出编号与简短原因
- 本会话 PASS / FAIL 自评：报告完整覆盖 11 项 = PASS；缺项或臆测 = FAIL
```

---

### §10.2 W1.1 testing infrastructure

**§B**：
```
# 本会话身份
模块/阶段:        W1.1 — testing infrastructure (一次性 impl)
spec 来源:        AGENTS.md §14.2 （Plan v2 无 testing infra 段，本 W 不指向 plan v2）
allowed_files:    
  - E:\codex_workspace\projects\hf_stock_clf\tests\conftest.py
  - E:\codex_workspace\projects\hf_stock_clf\pytest.ini
  # 严格两个文件，不许动其他任何文件

# 本会话目标
按 AGENTS §14.2 verbatim 创建 conftest.py + pytest.ini。
fixture 必须使用 §14.2 指定的 4 个名字（不许改名、不许加新 fixture）：
  - raw_price_df             —— 单 ticker，原始 OHLCV，对应 §3.4 Stage 1
  - raw_multi_ticker_dict    —— 多 ticker dict，对应 §3.4 Stage 1
  - labeled_df_with_tail_nan —— 末尾 k 行 label==NaN，对应 §3.4 Stage 2
  - split_df_after_trim      —— 经 trim_labels_at_split_boundary 处理后的样本，对应 §3.4 Stage 3
每个 fixture 在 docstring 内明示对应 §3.4 Stage。
pytest.ini 范围（按 §14.2 verbatim）：注册 integration marker + testpaths = tests，不引入其他设置。

# 本会话独有 do/don't
- fixture 名字必须与 AGENTS §14.2 完全一致，禁止"tiny_df / small_df"等自创命名
- fixture 必须用 @pytest.fixture，不要 module-level 全局对象
- 不要写任何 sample test 验证 fixture——fixture 由后续模块 test 间接验证
- conftest.py 不许 import ml_utils（此时 ml_utils 仍为空）
- pytest.ini 不许加 --strict-markers / addopts 等额外 flag（保持最小化）
- 不许 import torch / sklearn 等任何重依赖到 conftest 顶层
- fixture 内构造数据用显式常量（pandas / numpy），不许调任何 ml_utils 函数
```

**§C**：
```
# 必须输出
- conftest.py 全文
- pytest.ini 全文
- "$PYTHON_INTERPRETER" -m pytest --collect-only tests 的输出（要求：无 collection error）

# 收尾命令
"$PYTHON_INTERPRETER" -m pytest --collect-only tests
git status --short
git diff --stat

# 收尾汇报格式
- 修改文件清单（应严格只有 2 个：tests/conftest.py + pytest.ini）
- pytest --collect-only 输出：
    若 tests/ 下当前除 conftest.py 外无 test_*.py → 报告 "0 tests collected, no errors"
    若已有 test_*.py（不应有，但若存在）→ 报告实际 collected 数量，并标红"W0.1 #9 应已 FAIL"
    任何 collection error → FAIL，回头修
- fixture 名清单（应与 AGENTS §14.2 四个名字完全一致，verbatim）
- 本会话 PASS / FAIL 自评（fixture 名匹配 + collect-only 无 error = PASS）
```

---

### §10.3 W2.1 config-test

**§B**：
```
# 本会话身份
模块/阶段:        W2.1 — config-test (test-writing)
plan v2 段:       §3.1 ml_utils/config.py — 请在 pre-flight 中读出 8 项并贴出
                  （重要：plan v2 §5.x 是 seed/metrics/dataset/lstm/trainer/checkpoint/tcn/dlinear，
                   config 不在 §5.x，而在 §3.1，不要走错）
allowed_files:    
  - E:\codex_workspace\projects\hf_stock_clf\tests\test_config.py
  # 严格一个文件

# 本会话目标
按 AGENTS §7.5.1 测试质量门槛六类为 plan v2 §3.1 定义的 dataclass 写测试。
六类（verbatim 按 AGENTS §7.5.1）：
  - 正常情况          (典型输入下输出符合 spec)
  - 边界情况          (empty / 最小最大尺寸 / 边界值)
  - 错误情况          (非法输入必须 raise，与 §6.5 config 验证配套)
  - 不可变性          (config 为 value-type dataclass，本类 N/A 时显式标注)
  - 确定性期望值      (用手算值断言具体数值，禁止只断言类型)
  - 泄漏专项          (仅 dataset.py 涉及，本模块 N/A，显式标注)
每个 dataclass 的 __post_init__ 字段逐条验证（任一字段越界 → ValueError）。

# 本会话独有 do/don't
- 不许实现 ml_utils/config.py（本 session 仅写测试，目标文件不在 allowed_files）
- 不许凭记忆假设字段名：必须先从 plan v2 §3.1 抽出确切字段列表
- 测试中所有 magic number（边界值）必须有注释说明依据
- 不许 mock 任何东西：config 是纯 dataclass，无 IO，无外部依赖
- 测试函数命名按 AGENTS §7.5.1 类别分组前缀：test_normal_* / test_boundary_* / test_error_* / test_deterministic_*
- 遵守 SPRINT_PLAN_HANDOFF.md §7.D test-first 通用约束（lazy import 至 function 体内，不许 module top-level `from ml_utils.config import ...`）
```

**§C**：
```
# 必须输出
- tests/test_config.py 全文
- 测试 case 数量与六类分布表：
    正常情况=X, 边界情况=Y, 错误情况=Z, 不可变性=N/A, 确定性期望值=W, 泄漏专项=N/A

# 收尾命令
"$PYTHON_INTERPRETER" -m pytest tests/test_config.py --collect-only -v
git status --short
git diff --stat

# 收尾汇报格式
- 修改文件清单（应严格只有 1 个）
- collected tests 数量 + 六类分布
- 任何"我从 plan v2 §3.1 抽出但不确定字段名/类型"的疑问，列出来等用户确认
- 本会话 PASS / FAIL 自评（pytest --collect-only 无 error + 六类全覆盖或显式 N/A = PASS）
```

---

### §10.4 W2.2 config-impl

**§B**：
```
# 本会话身份
模块/阶段:        W2.2 — config-impl (implementation)
plan v2 段:       §3.1 ml_utils/config.py
                  （重要：config 在 plan v2 §3.1，不是 §5.x）
allowed_files:    
  - E:\codex_workspace\projects\hf_stock_clf\ml_utils\config.py
  # 严格一个文件。tests/test_config.py 不在 allowed_files，本会话只读。
line_budget:      plan v2 §3.1 行预算（pre-flight 中读出并贴出实际值；超 → STOP）

# 本会话目标
实现 ml_utils/config.py，让 tests/test_config.py 全绿。
所有 dataclass + 必要的 __post_init__ 验证（按 plan v2 §3.1 spec）。
不允许新增字段、不允许新增导出。

# 本会话独有 do/don't
- tests/test_config.py 不在 allowed_files；不许改测试任何一个字节（即使 typo）
  若发现测试 typo → STOP，报告给用户，由用户决定是否单独修
- 测试失败 → 调整实现，不要改测试
- 不许写任何"未来可能用到"的字段（MVP_YES：无 tcn/dlinear 相关字段）
- 不许 import ml_utils 其他模块（config 是基础层）
- 第三方依赖仅允许 dataclasses（stdlib）+ AGENTS §6.1 白名单内的
- 测试若 6 类中"错误情况 / 边界情况"失败，调整 __post_init__ 而非测试断言
```

**§C**：
```
# 必须输出
- ml_utils/config.py 全文
- 测试运行结果：passed / failed / skipped 计数（每个 dataclass 分组）

# 收尾命令
"$PYTHON_INTERPRETER" -m pytest tests/test_config.py -v
git status --short
git diff --stat

# 收尾汇报格式
- 修改文件清单（应严格只有 1 个：ml_utils/config.py）
- 若 git diff 显示 tests/test_config.py 有改动 → FAIL，本会话违反 allowed_files
- pytest 完整 passed/failed/skipped 数量
- 行数 vs line_budget 的对比
- 若 test 中有原本 PASS 现 FAIL 的 case → 标红，必须解释
- 本会话 PASS / FAIL 自评（全绿 + 行数 ≤ line_budget + tests/test_config.py 无 diff = PASS）
```

---

### §10.5 W2.3 config-review

**§B**：
```
# 本会话身份
模块/阶段:        W2.3 — config-review (self-review, fresh session)
plan v2 段:       §3.1 ml_utils/config.py
allowed_files:    []  # 严格空集，纯只读 review

# 本会话目标
按 AGENTS §9.3 九项 checklist 对 ml_utils/config.py + tests/test_config.py 做
self-review，输出 issue list。**不修任何代码**。

# 本会话独有 do/don't
- 这是 fresh 会话：不许引用 W2.2 的任何"我当时是这么想的"
- 假装代码是别人写的，主动找 motivated reasoning
- 不给修复建议，只列 issue
- 不修代码（修复留 W2.4 fixup session 或下一轮）
```

**§C**：
```
# 必须输出
issue list，格式：

[BLOCKER|WARNING|NIT] config.py:Lxx — 简短描述
[BLOCKER|WARNING|NIT] test_config.py:Lyy — 简短描述
...

末尾计数：BLOCKER=N, WARNING=M, NIT=K

# 收尾命令
git status --short    # 应为空，本会话不应有任何改动

# 收尾汇报格式
- issue 计数（按严重度分）
- 若 git status 非空 → 立刻 STOP 并报告，本会话违反 allowed_files=[]
- 本会话 PASS / FAIL 自评（git status 干净 + 报告完整 = PASS）
```

---

## §11. 反屎山核心信条（与 AGENTS_BUILD_LOG §6 同源）

任何新加规则必须同时通过：

1. **可执行**：agent 看完就知道做什么，不需要再推断
2. **可验证**：用户能在一次审计中查到是否被违反
3. **不仪式化**：不为了"看起来负责"而要求 agent 反复输出 meta-commentary

不过三关 → 不要加。

**不建议改动方向**（已多轮拒绝）：

- 不要把 session 切得更碎（test / impl / CHANGELOG 三独立会话已被拒）
- 不要加"每 N 步强制打印 X"类周期性仪式
- 不要在 AGENTS.md 重新引入 cross_day_allowed escape hatch
- 不要把 shuffled-label sanity check 加进 pytest（已固定在 notebook 02）
- 不要在 MVP 阶段创建 tcn/dlinear 占位文件或预留分支
- 不要让任何 Gate 退化回"建议"状态（SPRINT_LOG / MVP / reference_excerpts）

---

## §12. 新会话开场提示（直接复制粘贴）

```
我在做 hf_stock_clf / ml_utils 项目的 MVP 实施准备阶段。
MVP_YES 已决：第一轮仅 lstm 端到端，tcn/dlinear 推迟至 Phase 1B。
项目根：E:\codex_workspace\projects\hf_stock_clf\
AGENTS.md 当前应为 v4.2，第 2 行须有 `<!-- AGENTS_VERSION: v4.2 -->` marker
（若项目里仍是 v4 或缺 marker，先用我上传的版本覆盖项目根并补 marker）。

我贴四个文件给你：
1. SPRINT_PLAN_HANDOFF.md v4.1  ← 本文件，当前状态 + Gate 表 + 5 个近期 prompt
2. AGENTS.md v4.2             ← 硬约束基线
3. AGENTS_BUILD_LOG.md        ← AGENTS.md 演进历史
4. plan_v2_patches.md         ← 6+1 个待应用的 patch（含 MVP §6.x 段）

当前进度: 见 §1 状态快照 + §4 Gate 表。
我接下来要做的: <从 §4 Gate 中选当前未 PASS 的最早一项>

约束: 
- 不要重新讨论 AGENTS_BUILD_LOG §6 已拒绝过的方向
- 不要把 MVP_YES 重新讨论成"未决"
- 任何对 AGENTS.md / plan v2 / Gate 顺序的修改建议要先列判断表
- 反屎山原则：可执行 / 可验证 / 不仪式化
```
