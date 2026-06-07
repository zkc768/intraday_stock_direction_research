# Notebook 08 — Resume Gates and Session Handoff

> **You are**: a chat session that just arrived to the
> `intraday_stock_direction_research` project, asked to do something with
> Notebook 08. **You have not seen prior session history.** This file is
> your briefing.
>
> **Do this in order**:
>
> 1. Read §1 (state) → §2 (triage) → §4 (red lines).
> 2. Run §3 (gate) before any Branch A work identified in §2.
> 3. Consult §5 (paths) and §6 (required reading) as you act.
>
> **Anchor commit (MVP freeze)**: `<commit-sha-of-mvp-freeze>`
>
> Replace the placeholder with the actual 40-hex SHA after the first commit.
> If you are reading this and the placeholder is still literal, ask the user
> to confirm the commit identity before treating §1 snapshot as canonical.

---

## §1 — MVP Frozen Snapshot

> **What this section is for**: tell you what's already done so you don't redo it.

After four rounds of adversarial review, Notebook 08 is frozen at MVP scope:
contract module + generator + 35-cell notebook + comprehensive contract tests.
Deep-sequence model training (DLinear / TCN / GRU / LSTM / fusion) and real
LightGBM control fitting are intentionally **not implemented** — they are
gated on the GitHub migration described in `docs/GITHUB_MIGRATION_PLAN.md`.

### Files at MVP frozen state

| Path | Lines | Status |
|---|---:|---|
| `scripts/notebook08_contract.py` | 1001 | NEW (contract module, 17 validators) |
| `scripts/create_deep_sequence_exploration_colab_notebook.py` | 1809 | NEW (generator) |
| `notebooks/08_deep_sequence_exploration_colab.ipynb` | 2742 / 35 cells | GENERATED (no outputs, no execution counts) |
| `tests/test_notebook08_artifact_contract.py` | 1395 | MODIFIED (167 tests) |
| `tests/test_notebook08_static_gate.py` | 202 | UNCHANGED (Codex range, 14 tests) |
| `docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md` | 1659 | MODIFIED (+2/-2, typo fix only) |

### Test state at MVP frozen

| Scope | Passed | Skipped | Failed |
|---|---:|---:|---:|
| N08 isolated (static gate + contract) | 181 | 0 | 0 |
| N03–N08 full face | 361 | 0 | 0 |

### Safety mechanisms folded in across 4 review rounds

These are the assertions and guards the contract module enforces. A new agent
must not weaken or bypass any of them.

- **§5.5 Pre-registration Constants Table** — 13 frozen numeric thresholds. Any change requires a new freeze document and a fresh sha256 stamp on `08x_search_space.json`.
- **§7.9 low_compute_submode B nested-fold protocol** — closes the classical stacking-leak path; `outer_fold_k` and `inner_fold_k_for_head` both required ≥ 5; outer fold scheme must be one of `rolling_origin_folds` / `purged_time_series_folds` / `embargoed_train_inner_folds`.
- **§8.3 trial ledger enums** — `fit_status` ∈ `FIT_STATUSES`; failed rows must additionally carry `failure_type` ∈ `FAILURE_TYPES`. (Round 7 finding #6)
- **§9.1 DMC + separate-session positive attestation** — absent flag is not proof. Explicit `dmc_attestation.json` OR `separate_session_attestation.json` is required for the 08F entry gate. (Round 7 finding #3)
- **§9.2 paper_safe_score z_in_tier isolation** — `full_compute` and `low_compute` form separate z-score pools; no cross-tier mixing. Tier with `< 2` completed trials contributes 0 to the penalty term.
- **§9.3 fallback rule reverse regex** — `fallback_activation_rule` cannot reference any official-validation metric (substring layer + normalized regex layer).
- **§10.1 `OPERATOR_READOUT_AUTHORIZATION_SHA` canonical recipe** — byte-for-byte: path-length prefix + content canonicalization (json_canonical sorts keys, text_lf collapses CRLF). Input order is sensitive; `allow_nan=False` is enforced.
- **§10.2 step 0 ledger append-before-read** — `notebook07_validation_budget_ledger.csv` must gain a row BEFORE any official-validation metric is read; pre-existing rows are byte-immutable. (AGENTS.md §4.3)
- **§10.4 `schema_only_stub` hard override** — when 08O CSVs are header-only, the manifest is forced to `schema_only_stub=True` and `allowed_wording_bucket="no_candidate_freezable"`. No evidence wording bucket is reachable from stub mode. (Round 7 finding #1)
- **§13.3 08O completeness gate** — `check_08o_real_readout_completeness()` requires ALL four required artifacts (`validation_readout` / `per_ticker` / `seed_summary` / `same_row_baselines`) present + non-empty + schema-complete before real-readout mode is allowed. (Round 8 finding #1)

---

## §2 — Triage Decision Tree

> **What this section is for**: when the user gives you a request, match it to ONE branch below before doing anything.

### Branch A — Substantive 08 work (RUN §3 GATE FIRST)

The user wants to make real research progress on Notebook 08: train models,
fit baselines, read official validation, build real fold pipelines. This is
**blocked on the GitHub migration**; the MVP frozen state is the correct
stopping point until migration phases 1, 2, 4, and 7 (per
`docs/GITHUB_MIGRATION_PLAN.md`) all land.

**Trigger phrases — Chinese**:

- 继续 08 / 继续构建 08 / 继续写 08 / 写 08 / 执行 08 / 做 08 / 完成 08
- 实装深度模型 / 实装 DLinear / 实装 TCN / 实装 GRU / 实装 LSTM / 实装 fusion
- 接通 LightGBM / 把 LightGBM 跑起来 / wire LightGBM
- 把 08X 真实跑起来 / 让 08X 真实训练 / real 08X training
- 08O 真实读 / 08O 跑官方 validation / 跑 08O readout
- 让 08 跑起来 / 让 08 端到端 / 把 08 接通真实数据

**Trigger phrases — English**:

- "continue 08" / "finish 08" / "complete 08" / "implement 08"
- "implement deep models in N08"
- "wire the LightGBM control"
- "make 08X actually train"
- "run the 08O readout against real data"

**Action**:

1. Run §3 Migration Gate Check first.
2. Gate **FAIL** → tell the user: *"N08 substantive work is blocked on migration Phase &lt;X&gt;. The MVP is at a clean stop point; see `docs/GITHUB_MIGRATION_PLAN.md`."* Do NOT write deep model training into `scripts/create_deep_sequence_exploration_colab_notebook.py` — that file is being relocated by migration Phase 5.
3. Gate **PASS** → consult §5 Path Mapping. Substantive work goes into `src/intraday_research/stages/deep_sequence_exploration.py::run_stage`, NOT into the legacy generator script.

### Branch B — Additive / migration-safe work (do it NOW)

The user wants to tighten safety, add tests, fix docs, or audit state. These
are additive patches that migrate cleanly via migration Phase 2 (contract
shim) and Phase 7 (test relocation).

**Trigger phrases**:

- 加 08 测试 / 补 08 test / add 08 tests / cover X with tests
- 改 08 文档 / 改 08 design / update 08 design / fix typo in 08 design
- tighten 08 safety / reviewer 提了 X / Round N #M / 第 N 轮发现 / new review finding
- 重新生成 notebook / regenerate notebook / re-run generator
- audit 08 状态 / check 08 tests / 跑一下 08 测试 / verify 08 state
- 检查 ... / 看一下 ... (read-only inquiries)
- 继续下一部分 (when prior work was Branch B)

**Action**:

1. Proceed directly using paths in §5 "MVP frozen" column.
2. Contract patches go into `scripts/notebook08_contract.py`. Test additions go into existing `tests/test_notebook08_*.py`.
3. After every change run `<PYTHON> -m pytest tests/test_notebook08_*.py` and report regressions.
4. Never bypass §4 DO NOTs even for additive work.

### Branch C — Unclear or ambiguous

The request mixes branches, references state you have not verified, or could
be interpreted two ways.

**Action**: Ask the user one focused clarifying question before touching
code. Quote §1 snapshot facts if you need to clarify state. Do NOT guess
branch assignment.

---

## §3 — Migration Gate Check

> **What this section is for**: a deterministic pass/fail check that says whether Branch A substantive work is unblocked.

Run this exact block from the project root. Each step maps to a specific
migration phase in `docs/GITHUB_MIGRATION_PLAN.md`. The first failing step
IS the answer to the user's *"why is 08 blocked?"* question.

### Bash (Git Bash on Windows, or any POSIX shell)

```bash
PROJECT_ROOT="E:/codex_workspace/projects/intraday_stock_direction_research"
PYTHON="${PYTHON:-E:/codex_workspace/_envs/py311_shared/python.exe}"

cd "$PROJECT_ROOT" || { echo "FAIL: project root not found: $PROJECT_ROOT"; exit 1; }

# Phase 1 — package scaffold importable + scope intact
"$PYTHON" -c "import intraday_research; assert intraday_research.__research_scope__ == 'validation_only', 'wrong scope'" \
  || { echo "FAIL: Phase 1 (intraday_research package not importable, or scope drift)"; exit 1; }

# Phase 2 — contract module migrated
"$PYTHON" -c "from intraday_research.contracts.deep_sequence_exploration import validate_freeze_record, check_08o_real_readout_completeness" \
  || { echo "FAIL: Phase 2 (contract not migrated to src/intraday_research/contracts/)"; exit 1; }

# Phase 4 — stage entrypoint exists
"$PYTHON" -c "from intraday_research.stages.deep_sequence_exploration import run_stage" \
  || { echo "FAIL: Phase 4 (run_stage entrypoint missing in src/intraday_research/stages/)"; exit 1; }

# Phase 7 — notebook semantically renamed
test -f notebooks/deep_sequence_exploration_colab.ipynb \
  || { echo "FAIL: Phase 7 (notebook still at legacy path notebooks/08_*.ipynb)"; exit 1; }

# Post-migration tests must be green
"$PYTHON" -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  -q \
  || { echo "FAIL: post-migration tests not green"; exit 1; }

echo "GATE PASSED. Substantive N08 work may proceed."
echo "Entry point: src/intraday_research/stages/deep_sequence_exploration.py::run_stage"
```

### PowerShell (Windows-native)

```powershell
$ProjectRoot = "E:\codex_workspace\projects\intraday_stock_direction_research"
$Python = if ($env:PYTHON) { $env:PYTHON } else { "E:\codex_workspace\_envs\py311_shared\python.exe" }

Set-Location $ProjectRoot
if (-not $?) { Write-Error "FAIL: project root not found"; exit 1 }

& $Python -c "import intraday_research; assert intraday_research.__research_scope__ == 'validation_only'"
if (-not $?) { Write-Error "FAIL: Phase 1 (package not importable or scope drift)"; exit 1 }

& $Python -c "from intraday_research.contracts.deep_sequence_exploration import validate_freeze_record, check_08o_real_readout_completeness"
if (-not $?) { Write-Error "FAIL: Phase 2 (contract not migrated)"; exit 1 }

& $Python -c "from intraday_research.stages.deep_sequence_exploration import run_stage"
if (-not $?) { Write-Error "FAIL: Phase 4 (run_stage missing)"; exit 1 }

if (-not (Test-Path "notebooks\deep_sequence_exploration_colab.ipynb")) {
    Write-Error "FAIL: Phase 7 (notebook not renamed)"; exit 1
}

& $Python -m pytest tests\contracts\test_deep_sequence_exploration_contract.py tests\notebooks\test_deep_sequence_exploration_static_gate.py -q
if (-not $?) { Write-Error "FAIL: post-migration tests not green"; exit 1 }

Write-Host "GATE PASSED. Substantive N08 work may proceed."
Write-Host "Entry point: src/intraday_research/stages/deep_sequence_exploration.py::run_stage"
```

### Reading the verdict

- All five steps succeed → tell the user GATE PASSED, then proceed per §5 post-migration paths.
- First failing step → tell the user *"blocked on Phase &lt;X&gt;"*; cite the failing import, file, or test name verbatim.
- The `PYTHON` environment override exists for systems where the default path does not apply. Do NOT silently substitute `python` from `PATH` if the override path is missing — report it as a setup failure instead.

---

## §4 — Hard DO NOTs

> **What this section is for**: red lines. Violating any of these can invalidate research conclusions or break the GitHub migration. Ordered "most likely to be misstepped by a new agent" first.

1. **Do not** continue substantive 08 work in `scripts/create_deep_sequence_exploration_colab_notebook.py`. That file is being relocated by migration Phase 5. Substantive work goes into `src/intraday_research/stages/deep_sequence_exploration.py` AFTER §3 gate passes.

2. **Do not** bundle Codex-pre-existing dirty files into 08 commits. The following five files are outside this session's scope: `.gitignore`, `AGENTS.md`, `README.md`, `docs/RESEARCH_WORKFLOW.md`, `tests/test_notebook07_static_gate.py`. Coordinate ownership with the user before staging any of them.

3. **Do not** read, transform, score, summarize, or build wording from holdout/test data, ever. (AGENTS.md §4.1 + §4.2 + §4.3)

4. **Do not** bypass `schema_only_stub=True` when 08O artifacts are header-only. `check_08o_real_readout_completeness()` is the authority; do not hard-code `same_row_dummy_present=True` or `seed_summary_present=True` to suppress the stub flag. (Round 7 finding #1 + Round 8 finding #1)

5. **Do not** commit, push, branch, amend, force-push, or rewrite history without explicit user authorization in this session. (AGENTS.md §9)

6. **Do not** run heavy training (deep-sequence forward/backward, LightGBM cross-validation on full bars) to prove basic import or contract behavior. Stubs and fixture data are sufficient and intended; that is what the contract module exists for. (`docs/GITHUB_MIGRATION_PLAN.md` stop conditions)

7. **Do not** edit `docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md` beyond additive patches. Four review rounds are baked into the document; changing existing rules requires a new freeze document and a new sha256 stamp on `08x_search_space.json` (per §5.5 Pre-registration Constants Table).

---

## §5 — Path Mapping (MVP frozen ↔ post-migration)

> **What this section is for**: tell you where the work actually goes. Wrong file location = wrong work.

| MVP frozen (now) | Post-migration target |
|---|---|
| `scripts/notebook08_contract.py` | `src/intraday_research/contracts/deep_sequence_exploration.py` |
| `scripts/create_deep_sequence_exploration_colab_notebook.py` | `scripts/notebooks/generate_deep_sequence_exploration_colab.py` |
| `notebooks/08_deep_sequence_exploration_colab.ipynb` (35 cells, heavy CONFIG / RUNTIME / CELL_* inlining) | `notebooks/deep_sequence_exploration_colab.ipynb` (thin: `from intraday_research.stages.deep_sequence_exploration import run_stage; run_stage(config)`) |
| `tests/test_notebook08_artifact_contract.py` | `tests/contracts/test_deep_sequence_exploration_contract.py` |
| `tests/test_notebook08_static_gate.py` | `tests/notebooks/test_deep_sequence_exploration_static_gate.py` |

### Artifact filenames are INVARIANT across migration

The following filenames do NOT change when notebooks are renamed in Phase 7:

- `08x_search_space.json`, `08x_trial_ledger.csv`, `08x_failure_ledger.csv`, `08x_seed_summary.csv`, `08x_candidate_compression_table.csv`, `08x_run_manifest.json`, `08x_environment_manifest.json`
- `08f_candidate_freeze_record.{json,md}`, `08f_static_gate_report.json`, `08f_no_candidate_freezable.json`, `08f_candidate_compression_audit.csv`
- `08o_validation_readout.csv`, `08o_validation_per_ticker.csv`, `08o_seed_summary.csv`, `08o_same_row_baselines.csv`, `08o_concentration_guardrails.csv`, `08o_failure_rows.csv`, `08o_decision_record.json`, `08o_run_manifest.json`
- `notebook07_validation_budget_ledger.csv` (project-level ledger)

These are stage-internal namespaces, not notebook numbers. Do NOT rename them
to `deep_sequence_exploration_*`. Downstream consumers — the ledger
append-before-read flow, paper-ready synthesis, and ledger snapshots — all
expect the existing names.

---

## §6 — Required Reading (in order)

> **What this section is for**: documents to read before answering any Branch A request. Read them in this exact order.

1. **`AGENTS.md`** — hard research rules. Read §4 (Hard Research Rules: chronology, leakage, evaluation honesty, failure behavior) and §9 (Environment And Git) at minimum. These rules override everything in this document.

2. **`docs/GITHUB_MIGRATION_PLAN.md`** — 10-phase migration spec. The §3 gate check in THIS file maps to phases 1, 2, 4, and 7 of that plan; the migration plan's stop conditions block any code path that violates the research contract.

3. **`docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md`** — 1659 lines, frozen. The N08 contract module and generator are derived from this document; substantive work must honor the section references in §1 of THIS file (§5.5, §7.9, §8.3, §9.1–§9.3, §10.1–§10.4, §13.3).

4. **`configs/pipeline.yaml`** (once written) — stage registry including `deep_sequence_exploration`. If this file does not yet exist, migration has not reached Phase 3; treat its absence as additional evidence that the §3 gate will fail at Phase 4.
