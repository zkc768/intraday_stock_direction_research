# intraday_stock_direction_research

Research notebook project for high-frequency stock direction classification.

Default entry points:

- `AGENTS.md` - project rules and hard research-validity constraints.
- `docs/RESEARCH_WORKFLOW.md` - notebook-first experiment workflow.
- `notebooks/01_research_direction_colab.ipynb` - raw-data-first research direction and Colab opening context.
- `notebooks/02_config_screening_colab.ipynb` - active Stage 0 configuration-screening notebook.
- `docs/ENVIRONMENT.md` - local Python interpreter and environment notes.

Useful references:

- `docs/CONFIG_SCREENING_FREEZE_2026-06-04.md` locks the active raw-data-first Stage 0 configuration-screening route.
- `notebooks/01_research_direction_colab.ipynb` records the Colab opening workflow and raw-data manifest.
- `notebooks/02_config_screening_colab.ipynb` is the active validation-only Stage 0 notebook.

Current working style: one clear validation-only research notebook at a time, with chronological splits, train-only preprocessing, split-boundary label invalidation, dummy-baseline comparison, and explicit result scope. Active notebook work should be raw-data-first and self-contained. The active route is `01_research_direction_colab.ipynb` then `02_config_screening_colab.ipynb`; future model-family screening belongs in a later `03_model_family_screening_colab.ipynb` only after Stage 0 selects candidates.
