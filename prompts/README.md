# Prompt Templates

This directory contains reusable prompt templates for Codex work on
`hf_stock_clf`.

These files are not executable code. They do not override `AGENTS.md`, the
active user prompt, or the current sprint prompt. Current sprint prompt and
`AGENTS.md` have higher priority.

Templates must be copied, filled, reviewed, and explicitly approved before
execution. A filled prompt draft is not permission for Codex to run it.

Default rules:

- Do not use `git add -A`.
- Do not commit except in a reviewed `commit-only` task.
- Do not push except in a reviewed `push-only` task.
- Do not execute notebooks or training unless the task is explicitly an
  execution task.
- Keep allowed files and forbidden files explicit.
- Keep final reports evidence-based and anti-overclaiming.

Available templates:

- `template_read_only_audit.md`
- `template_docs_only_record.md`
- `template_notebook_patch.md`
- `template_review_only.md`
- `template_commit_only.md`
- `template_push_only.md`
- `template_colab_execution.md`
- `template_next_prompt_generator.md`
