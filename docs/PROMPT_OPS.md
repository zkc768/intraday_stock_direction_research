# PromptOps for hf_stock_clf

This document defines how Codex should generate future task prompts for this
project. It is docs-only process guidance. It does not override the active user
prompt, `AGENTS.md`, or the project plans referenced by `AGENTS.md`.

## 1. Role Separation

Prompt generator mode:

- Read the current project state and relevant docs.
- Write a prompt draft only.
- Do not execute the generated prompt.
- Do not authorize the next step.
- Do not expand scope without explicit user instruction.

Execution mode:

- Execute a user-approved prompt.
- Follow the prompt exactly.
- Stop when the prompt's stop conditions are met.
- Report only what was actually inspected, changed, run, and validated.

Review mode:

- Review outputs, diffs, reports, or prompt drafts only.
- Do not edit files.
- Do not run experiments unless the review prompt explicitly permits a narrow
  read-only validation command.

Prompt generation and prompt execution must remain separate. A generated prompt
is not permission to execute it.

## 2. Non-Negotiable Generated Prompt Fields

Every generated prompt must include:

- Task id.
- Task type: `read-only`, `docs-only`, `prompt-generator`,
  `notebook-patch`, `code-implementation`, `review-only`, `commit-only`,
  `push-only`, or `Colab execution`.
- Allowed files.
- Forbidden files.
- Forbidden actions.
- Pre-flight commands.
- Validation commands.
- Stop conditions.
- Final report format.
- Explicit non-actions.

If any field cannot be filled from current context, the generated prompt must
say what is unknown and require the execution session to stop before editing.

## 3. Project-Specific Hard Defaults

Generated prompts must preserve these defaults unless the user explicitly
changes them:

- No `git add -A`.
- No automatic commit unless the task type is `commit-only`.
- No push unless the task type is `push-only`.
- No notebook execution unless the task type is an execution task.
- No training unless the task type is an execution task.
- No Colab unless the task type is a `Colab execution` task.
- No artifacts unless explicitly enabled.
- No TCN/DLinear expansion unless the current task explicitly allows it.
- No full run unless a full-run prompt is explicitly requested and reviewed.

For implementation prompts, keep `ml_utils/`, `tests/`, and notebook scope
aligned with `AGENTS.md` and the active sprint prompt. For docs-only prompts,
do not create code, scripts, notebooks, tests, data, or artifacts.

## 4. Scope Discipline

Before drafting a prompt, the prompt generator must ask:

- What is the exact next task?
- What is the smallest safe scope?
- What files may change?
- What must not change?
- What validation proves completion?
- What still remains unproven?

The generated prompt must make the answers explicit. It must not give Codex room
to choose a larger task family, add convenience scripts, broaden the model set,
or turn a smoke into a full run.

## 5. Anti-Overclaiming Rules

Generated prompts and final reports must not claim:

- Smoke success equals model signal.
- Helper path success equals full entrypoint readiness.
- Full entrypoint readiness equals full experiment validity.
- LSTM beats dummy unless multi-seed or full-scope results support it.
- TCN/DLinear validated unless they were actually run.
- Artifacts are absent unless absence was independently checked.

Reports must separate what was observed from what remains unproven.

## 6. Current Project Workflow

Use this workflow unless the user explicitly approves a different path:

```text
planning/readiness
-> prompt generation
-> execution
-> review-only audit
-> atomic commit
-> push-only verification
-> next planning
```

Notes:

- Planning/readiness and prompt generation are not execution.
- Execution should be narrow and should leave evidence for review.
- Review-only audits should not edit.
- Atomic commit prompts should stage only the reviewed files, never `git add -A`.
- Push-only prompts should verify branch/status before pushing and should not
  create new commits.

## 7. Prompt Handoff Format

Every generated prompt must start with:

```text
你现在执行 {TASK_ID} — {TASK_NAME}.
```

Every generated prompt must include these sections:

- Current state.
- Goal.
- Hard boundaries.
- Pre-flight.
- Exact task instructions.
- Validation.
- Expected final state.
- Final report format.
- Explicit non-actions.

Keep prompts copyable. A reviewer should be able to see the scope, risks,
validation, and non-actions without consulting chat history.

## 8. Prompt Generator Stop Conditions

The prompt generator must stop and report instead of drafting a runnable prompt
when:

- The git working tree is dirty and the requested prompt assumes a clean tree.
- Required project docs or source-of-truth records are missing.
- The user asks for a prompt that conflicts with `AGENTS.md`.
- The prompt would need a code script to generate safely.
- The scope cannot be expressed as explicit allowed files and forbidden files.
- The requested task would silently combine prompt generation and execution.

## 9. Final Report Expectations

Prompt-generation sessions should report:

- Overall verdict.
- Files inspected.
- Files changed or created.
- Summary of the generated prompt system or draft.
- Validation results.
- Remaining warnings.
- Recommended next step.
- Explicit non-actions.

The report must not imply that the generated prompt has been executed.
