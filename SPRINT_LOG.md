# SPRINT_LOG.md — hf_stock_clf / ml_utils

最近更新：2026-05-15

## 当前状态

最近完成:     W4.1 metrics-test
当前阶段:     W4.2 metrics-impl ready
下一步:       W4.2 metrics-impl
备注:         当前只允许实现 ml_utils/metrics.py；tests/test_metrics.py 只读，不允许修改
## 已合并模块清单（Codex 可以安全 import）

- `ml_utils/config.py`
- `ml_utils/seed.py`

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
| 7 whitelist audit | PASS | requirements 5 个核心依赖均在 AGENTS.md §6.1 白名单或 tests scope |
| 8 reference_excerpts ltsf_data_loader.py | PASS | vendored from local LTSF-Linear data_provider/data_loader.py |
| 9 atomic commit | PASS | Gate 9 commit created by Codex session; hash not written back |

## Workstream status

| Item | Status | Evidence |
|---|---|---|
| Gate 0-6 | PASS | 手工段已完成 |
| Gate 7-9 | PASS | Gate status 表中 7-9 均为 PASS |
| W0.1 readiness audit | PASS | readiness audit 已完成；项目可进入 §14.2 testing infrastructure |
| W1.1 testing infrastructure | PASS | pytest.ini 与 tests/conftest.py 已创建；commit 249b2d8 |
| W2.1 config-test | PASS | tests/test_config.py 已创建；commit e0773f2 |
| W2.2 config-impl | PASS | ml_utils/config.py 已创建；commit 2945c81；tests/test_config.py 26 passed；W2.3 review PASS |
| W3.1 seed-test | PASS | tests/test_seed.py 已创建；commit 7e6b1ab；collect-only 5 tests collected；生成的 tests/__pycache__/ 已清理且未提交 |
| W3.2 seed-impl | READY | 当前只允许实现 ml_utils/seed.py；tests/test_seed.py 只读，不允许修改 |
| W3.2 seed-impl | PASS | commit `db0baf3`; created `ml_utils/seed.py`; implemented `seed_everything`; `tests/test_seed.py` passed 5/5; `tests/test_config.py tests/test_seed.py` passed 31/31 |
| W3.3 seed-review | PASS | Fresh review found no implementation issues; final status recovery restored clean scope before commit |
| W4.1 metrics-test | PASS | commit `092331d`; created `tests/test_metrics.py`; lazy import used; metrics collect-only collected 10 tests; config+seed+metrics collect-only collected 41 tests; `ml_utils/metrics.py` not created |
## 当前 git 状态

记录 W3.2 seed-impl 后，预期本次 docs/log step 只修改 `SPRINT_LOG.md`。

- 工作目录干净：待 Gate 0-6 commit 后确认
- 当前分支：待确认
- 与 origin：待确认

## Requirements whitelist audit

审计日期：2026-05-15
判据：AGENTS.md §6.1 依赖白名单

| Package | Version | Status | Source in AGENTS §6.1 |
|---|---|---|---|
| torch | 2.12.0+cpu | allowed | "torch、torch.nn、torch.optim、torch.utils.data" |
| numpy | 1.26.4 | allowed | "numpy、pandas" |
| pandas | 2.2.2 | allowed | "numpy、pandas" |
| scikit-learn | 1.4.2 | allowed | "sklearn.preprocessing、sklearn.metrics、sklearn.dummy、sklearn.base" |
| pytest | 8.3.5 | allowed (tests scope) | "tests/：可用 tempfile、pytest fixture" |

差异:
- requirements 有但白名单未列：无
- 白名单允许但 requirements 未列：无（核心第三方依赖均已列；`pandas.api.types` 由 pandas 提供，`sklearn.*` 由 scikit-learn 提供）

结论：PASS

## Reference excerpts

| File | Status | Source commit | License | Note |
|---|---|---|---|---|
| ltsf_data_loader.py | vendored | 0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6 | Apache-2.0 | MVP 用 |
| pytorch_tcn_core.py | deferred | n/a | n/a | Phase 1B 启动前再补 |
| ltsf_dlinear_model.py | deferred | n/a | n/a | Phase 1B 启动前再补 |

## 进行中 session 注意事项

- Gate 0-6 已由用户手工执行
- Gate 7-9 已完成并记录 PASS
- W0.1 readiness audit 已完成并记录 PASS
- W1.1 testing infrastructure 已完成并记录 PASS
- W2.1 config-test 已完成并记录 PASS
- W2.2 config-impl 已完成并记录 PASS
- W3.1 seed-test 已完成并记录 PASS
- 当前阶段为 W3.2 seed-impl ready，下一步是 W3.2 seed-impl
- 当前只允许实现 ml_utils/seed.py
- tests/test_seed.py 只读，不允许修改
- 不要声称 W3.2 已完成；ml_utils/seed.py 尚未创建

## Atomic commits

本项目 Gate 阶段采用两次 commit 边界：

- C1：Gate 0-6，由用户手工 commit
- C2：Gate 7-9，由 Codex session commit

| Commit | Gate range | Owner | Commit hash | Message |
|---|---|---|---|---|
| C1 | 0-6 | user manual | 待 Gate 0-6 commit 后生成 | chore(gate): apply Gate 0-6 sprint contract updates |
| C2 | 7-9 | Codex session | 不回填（按 Gate 9 规则，用 git log --oneline -1 验证） | chore(gate): land Gate 7-9 (whitelist audit + ltsf_data_loader vendor) for hf_stock_clf v4.1 |
