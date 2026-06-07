"""Raw-data ingestion + label/feature/window helpers for the N08 #5C pipeline.

Submodules:
  - ``labels``    no-trade-band binary labels (#5C-1)
  - ``raw_bars``  5-min pre-aggregated per-ticker CSV loader (#5C-3)
  - ``features``, ``splits``, ``windows``  arrive in sibling commits
    #5C-2 / #5C-4 / #5C-5.

Validation-only scope (AGENTS.md section 4.1); no holdout/test data is read
by anything in this subpackage.
"""
