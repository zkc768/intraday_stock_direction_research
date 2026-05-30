# Untracked Cleanup Manifest - 2026-05-30

## Scope

This manifest records the remaining untracked `hf_stock_clf` files after the
controlled docs and Phase 1B local-reporting commits through:

```text
7d00690 feat(phase1b): add local reporting pipeline
```

Rules for this cleanup pass:

- Do not delete raw data.
- Do not run heavy training.
- Do not execute notebooks.
- Do not stage broad dirty-tree groups with `git add .`.
- Keep generated cache files out of commits unless explicitly approved.

## Current Git State

Last verified state:

```text
## master...origin/master
?? .codegraph/changes.journal
?? .codegraph/graph.db
?? .codegraph/graph.db-shm
?? .codegraph/graph.db-wal
?? notebooks/04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb
?? "notebooks/Binary classification comparison_Zhang (2).ipynb"
?? notebooks/P1B.21d_notebook03_ticker_axis_narrow_smoke.ipynb
```

Remote sync check:

```text
origin/master...master = 0 0
```

## Inventory And Disposition

| Path | Type | Evidence | Disposition |
|---|---|---|---|
| `.codegraph/changes.journal` | generated codegraph cache | 37-byte journal file observed in untracked status | Do not commit. Keep untracked or ignore after user confirms `.gitignore` change. |
| `.codegraph/graph.db` | generated codegraph SQLite cache | about 2.1 MB graph database observed in untracked status | Do not commit. Keep untracked or ignore after user confirms `.gitignore` change. |
| `.codegraph/graph.db-shm` | generated SQLite sidecar | created by local graph access | Do not commit. Safe cache noise; ignore/delete only after confirmation. |
| `.codegraph/graph.db-wal` | generated SQLite sidecar | created by local graph access | Do not commit. Safe cache noise; ignore/delete only after confirmation. |
| `notebooks/04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb` | research notebook prototype | 15 cells, 0 executed cells, 0 outputs; flags are `FULL_RUN=False`, `RUN_DATA_DIAGNOSTICS=False`, `RUN_TRAINING=False`, `WRITE_ARTIFACTS=False` | Candidate for a later notebook-only review. Do not mix with code/docs commits. |
| `notebooks/Binary classification comparison_Zhang (2).ipynb` | external/Zhang comparison notebook | 43 cells, 32 executed code cells, 58 outputs, about 340,799 output JSON chars; writes results to `/content/drive/...` | Do not commit as-is. First extract provenance/results, then clear large outputs or archive. |
| `notebooks/P1B.21d_notebook03_ticker_axis_narrow_smoke.ipynb` | smoke-run provenance notebook | 5 cells, 2 executed code cells, 5 outputs, about 46,941 output JSON chars; flags include `FULL_RUN=True`, `RUN_TRAINING=True`, `WRITE_ARTIFACTS=False` | Do not commit as-is. First extract the smoke evidence, then clear outputs and reset training flags before any notebook commit. |

## Notebook Safety Notes

The three notebooks were parsed with `nbformat` and were not executed.

Observed risks:

- `P1B.21d_notebook03_ticker_axis_narrow_smoke.ipynb` has training flags set to
  true in the saved code cell.
- `Binary classification comparison_Zhang (2).ipynb` contains large embedded
  outputs and Colab/Drive paths.
- `04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb` is structurally clean but
  still belongs in a separate notebook-review commit because it is a prototype,
  not part of the Phase 1B reporting pipeline.

## Validation Evidence

Commands already run in this cleanup pass:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile scripts\phase1b_local\build_paper_tables.py scripts\phase1b_local\local_baseline_matrix.py scripts\phase1b_local\summarize_runs.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_phase1b_local_runner.py tests\test_phase1b_paper_tables.py tests\test_phase1b_report_summarizer.py -q
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\ -q -m "not integration"
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_phase1b_local_runner.py tests\test_phase1b_paper_tables.py tests\test_phase1b_report_summarizer.py -q -m "not integration"
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\ -q --collect-only
E:\codex_workspace\_envs\py311_shared\python.exe -c "<nbformat structural read for the three untracked notebooks>"
```

Observed results:

- Phase 1B local-reporting tests: `83 passed`.
- Full non-integration regression: `235 passed, 1 warning`.
- Phase 1B tests under `-m "not integration"`: `83 passed`.
- Pytest collection: `235 tests collected`.
- Notebook structural read: `notebook structure ok: 3`.

The warning is the existing PyTorch scheduler ordering warning in
`tests/test_checkpoint.py::test_scheduler_state_is_restored_when_scheduler_is_provided`.

## Recommended Next Actions

1. Leave `.codegraph/*` untracked for now.
2. If clean status noise matters, approve a separate `.gitignore` change for
   `.codegraph/`.
3. For `P1B.21d_notebook03_ticker_axis_narrow_smoke.ipynb`, extract the smoke
   evidence into a tracked doc before clearing outputs or changing flags.
4. For `Binary classification comparison_Zhang (2).ipynb`, extract provenance
   and result summaries before deciding whether to clear outputs or archive it.
5. For `04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb`, run a notebook-only
   review pass before any commit.

## Stop Rule

No remaining untracked notebook or cache file should be committed without a
separate confirmation or a notebook-specific review/cleanup step.
