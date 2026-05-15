# SPRINT_LOG.md — hf_stock_clf / ml_utils

最近更新：2026-05-15

## 当前状态

最近完成:     Gate 0-6（手工段）
正在进行:     Gate 7-9（Codex 接手）
下一步候选:   W0.1 readiness audit

## 已合并模块清单（Codex 可以安全 import）

无（ml_utils 仅含 __init__.py × 2）。

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| 0 MVP_YES | PASS | SPRINT_PLAN §1 / plan v2 §6 |
| 1 AGENTS v4.2 + marker + §9.1 例外 | PASS | AGENTS.md L2 + §9.1 末尾 |
| 2 AGENTS_BUILD_LOG v4.2 | PASS | BUILD_LOG §2/§3/§4 |
| 3 plan_v2_patches 应用 + MVP_YES + patch4#7 | PASS | grep 词表 100% 通过 |
| 4 ENVIRONMENT 填实 | PASS | Python / pytest / imports / pip check passed |
| 5 requirements 锁定 | PASS | requirements.txt 5 行核心依赖，全部 ==，torch 保留 +cpu |
| 6 SPRINT_LOG 创建 | PASS | 本文件 |
| 7 whitelist audit | PENDING | 由 Codex 填入 requirements vs 白名单交叉审计差异 |
| 8 reference_excerpts ltsf_data_loader.py | PENDING | 由 Codex 填入 Reference excerpts |
| 9 atomic commit | PENDING | 由 Codex 填入 Atomic commits |

## 当前 git 状态

- 工作目录干净：待 Gate 0-6 commit 后确认
- 当前分支：待确认
- 与 origin：待确认

## requirements vs 白名单交叉审计差异 / Requirements whitelist audit

待 Gate 7 填写（Codex）。

- requirements 有但第三方白名单无：待填写
- 第三方白名单有但 requirements 无：待填写

## Reference excerpts

待 Gate 8 填写（Codex）。

- reference_excerpts/ltsf_data_loader.py：待确认
- reference_excerpts/pytorch_tcn_core.py：Phase 1B 前再 vendor，MVP 不阻塞
- reference_excerpts/ltsf_dlinear_model.py：Phase 1B 前再 vendor，MVP 不阻塞

## 进行中 session 注意事项

- Gate 0-6 已由用户手工执行
- Gate 7-9 才允许 Codex 接手
- 不允许启动 W0.1，直到 Gate 0-9 全部 PASS
- 不允许创建任何 ml_utils production code
- 不允许创建 tests/conftest.py 或 pytest.ini，直到 Gate 7-9 完成且进入 W1.1

## Atomic commits

本项目 Gate 阶段采用两次 commit 边界：

- C1：Gate 0-6，由用户手工 commit
- C2：Gate 7-9，由 Codex session commit

| Commit | Gate range | Owner | Commit hash | Message |
|---|---|---|---|---|
| C1 | 0-6 | user manual | 待 Gate 0-6 commit 后生成 | chore(gate): apply Gate 0-6 sprint contract updates |
| C2 | 7-9 | Codex session | 待 Codex 在 Gate 9 完成自身 commit 后回填 | chore(gate): complete Gate 7-9 reference audit and sprint log |