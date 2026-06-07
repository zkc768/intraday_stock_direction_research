# Reproducibility

This document is the operator reference for reproducing the validation-only
research pipeline of `intraday_stock_direction_research` from a pinned git
commit plus a documented raw-data manifest. Local and Colab paths are both
covered. No published metrics or release claims are made here.

## Research contract

Scope: validation-only. The closed holdout/test interval is not opened,
transformed, scored, summarized, or used for selection or wording. The
package exposes `intraday_research.__research_scope__ == "validation_only"`
and `intraday_research.__holdout_test_authorized__ is False` as runtime
guards. Every research decision (features, labels, thresholds, model family,
wording) is made from train + validation only. See `AGENTS.md` sections 4.1
through 4.3 for chronology, evaluation honesty, and the ledger contract.

## Local install

Use the project Python interpreter. In this document the literal interpreter
path is intentionally written as the placeholder `<py311_shared_python>`:
set this to your project Python (e.g. the workspace-documented `py311_shared`
interpreter that the project's `AGENTS.md §9` standardises on).

The current `pyproject.toml` declares only the optional `[deep]` extras;
runtime deps (`numpy`, `pandas`, `scikit-learn`, `lightgbm`, `pytest`,
`nbformat`) are pinned in `requirements.txt` and not yet propagated into
`pyproject.toml`. Fresh-env reproducibility from `pip install -e .` alone
is therefore NOT yet supported; until those deps move in, use:

```bash
<py311_shared_python> -m pip install -r requirements.txt
<py311_shared_python> -m pip install -e . --no-deps
<py311_shared_python> -m pytest tests/test_package_import.py -q
```

The workspace `py311_shared` project Python already has those deps
installed; in that case skip the first install line.

The optional deep-learning dependency group adds `torch`:

```bash
<py311_shared_python> -m pip install -e ".[deep]"
```

No other implicit installs are performed by this repository; CI / Colab is
expected to pin its install command rather than re-resolve dependencies.

## Colab exact-commit install

Colab notebooks install the package from an exact git commit pinned in the
notebook itself, and record the same commit in the run manifest.

```python
%pip install -q "git+https://github.com/<user>/intraday_stock_direction_research.git@<commit_sha>"
```

`<user>` and `<commit_sha>` above are placeholders. The commit SHA must be a
40-character hex; floating refs are not acceptable (see next section). See
`AGENTS.md §7.2` for the package-first Colab boundary.

## Forbidden install patterns

The following install strings are forbidden in any committed notebook, in
any run manifest, and in any external instruction:

- the literal refs `@main`, `@master`, `@HEAD`, or any other branch / tag
  that is not an immutable commit SHA
- bare `intraday_research` (the package is not published to PyPI and must
  not be assumed installable from a public index)
- any `pip install` line whose URL has no `@` ref, or whose `@` ref resolves
  to a moving branch tip

Violations are detectable by a release-time check against the committed
notebooks: any `git+https://...@<ref>` substring whose `<ref>` is not a
40-char hex SHA fails the install-pattern gate.

## Raw data manifest policy

Raw five-stock 5-minute bar data is not committed to GitHub. The repository
tracks a documented manifest only; the bytes live in Google Drive (active
research) or in a public data repository (archival release). See
`docs/ARTIFACT_POLICY.md` for the GitHub / Drive / Zenodo / Colab storage
boundary, the manifest policy, and the test-fixture rules.

A Colab run obtains raw data through the manifest, downloads it to a local
runtime directory, and writes outputs first to that runtime directory; any
Drive backup is an explicit, optional step. Mounting `MyDrive` in the
default setup cell is not part of this workflow.

## What can and cannot be rebuilt

Rebuildable from `<commit_sha>` plus an authorised raw-data manifest:

- the validation-only artifacts produced by every stage listed in
  `configs/pipeline.yaml`, with the scope and ledger guards described in
  `AGENTS.md §4`.
- the package, configs, contracts, and tests checked into the repository.

Not rebuildable from this repository alone, by design:

- the closed holdout/test split (no read, no transform, no score, no
  summary path exists, and the package guard refuses authorisation).
- machine-local compute environments not captured in `pyproject.toml` or
  the `[deep]` extras group.
- any ledger snapshot that has not been curated into
  `artifacts/ledger_snapshots/` per the artifact policy.

No published numerical results are claimed in this document; consult the
thesis manuscript or release notes for any wording-bound, scope-tagged
research statements.

## Known non-determinism and stage order

Stage execution order is the canonical order from `configs/pipeline.yaml`:

1. `config_screening`
2. `model_family_screening`
3. `controlled_followup`
4. `lightgbm_tuning`
5. `selective_no_trade_calibration`
6. `validation_synthesis_gap_audit`
7. `deep_sequence_exploration`

Known non-determinism sources:

- per-stage seeds documented in each `configs/stages/<name>.yaml` once
  populated; defaults frozen by the corresponding stage design doc.
- library-level non-determinism in LightGBM and any PyTorch path; reruns
  must not be used as a way to alter recorded results, and any rerun
  decision must be scope-tagged.

## Validation-budget ledger policy

Per `AGENTS.md §4.3`, the cross-notebook validation-budget ledger
(`notebook07_validation_budget_ledger.csv`) is append-only across N07, N08,
and any thesis chapter that cites an official-validation metric. Any
downstream read of an official-validation metric must append a row recording
the intent BEFORE the read; pre-existing rows must not be modified, dropped,
or reordered. A read without a prior append is a contract violation
detected by
`tests/test_notebook08_artifact_contract.py::validate_08o_ledger_append_precedes_read`.
Runtime copies of the ledger live under ignored runtime output directories.
Canonical and release snapshots are tracked under
`artifacts/ledger_snapshots/` after review (see `docs/ARTIFACT_POLICY.md`);
any ledger row referenced by N08 outputs or by the thesis manuscript must
have either a tracked snapshot or a manifest plus checksum on record.
`AGENTS.md §4.3` designates `notebook07_validation_budget_ledger.csv` as
the project-level source of truth for the validation budget; this document
does not redefine that status.
