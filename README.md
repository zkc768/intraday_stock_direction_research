# intraday_stock_direction_research

Research notebook project for high-frequency stock direction classification.

Default entry points:

- `AGENTS.md` — project rules and hard research-validity constraints.
- `docs/RESEARCH_WORKFLOW.md` — notebook-first experiment workflow.
- `notebooks/04_ian_research_memo.ipynb` — current active research notebook.
- `docs/ENVIRONMENT.md` — local Python interpreter and environment notes.

Useful references:

- `docs/BASELINE_REFERENCE.md` records the current baseline setup.
- `archive/legacy_model_runner_reference/` contains the old helper library, tests, reference excerpts, scripts, and model-runner-orchestrated notebook. It is historical reference only.
- Historical `PM_*`, `PHASE_1B_*`, route-control, handoff, and prompt-operation documents were removed from the working tree as tracked history. Recover them from git history if a historical audit is needed.
- Notebooks `00` through `03` were moved to `notebooks/archive/legacy_setup_smoke_notebooks/`; they are not active templates.

Current working style: one clear validation-only research notebook at a time, with chronological splits, train-only preprocessing, split-boundary label invalidation, dummy-baseline comparison, and explicit result scope. Active notebook work should not depend on archived helper library or old runner scripts.
