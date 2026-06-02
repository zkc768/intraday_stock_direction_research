# hf_stock_clf

Research notebook project for high-frequency stock direction classification.

Default entry points:

- `AGENTS.md` — project rules and hard research-validity constraints.
- `docs/RESEARCH_WORKFLOW.md` — notebook-first experiment workflow.
- `docs/ENVIRONMENT.md` — local Python interpreter and environment notes.

Useful references:

- `docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md` records the current mentor-aligned baseline protocol.
- `docs/ml_utils_construction_plan_v2.md` is retained for future `ml_utils` implementation work only; it is not the default research workflow.
- Historical `PM_*`, `PHASE_1B_*`, route-control, handoff, and prompt-operation documents were removed from the working tree as tracked history. Recover them from git history if a historical audit is needed.

Current working style: one clear validation-only research notebook at a time, with chronological splits, train-only preprocessing, split-boundary label invalidation, dummy-baseline comparison, and explicit result scope.
