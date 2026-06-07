# Artifact Policy

This policy defines what belongs in Git, what belongs in external storage, and
how Colab runs should move data and outputs. It exists to keep the repository
small, auditable, and reproducible without committing raw market data or runtime
artifacts.

This document does not authorize deleting raw data, changing validation results,
touching holdout/test data, or rewriting notebook logic. It only defines storage
and version-control boundaries.

## Repository Boundary

GitHub should contain the research code and the information needed to reproduce
the work from authorized data sources:

- source code under `src/`, `scripts/`, and generated notebooks with empty
  outputs;
- configuration files and pipeline registries under `configs/`;
- protocol, design, reproducibility, and mapping documents under `docs/`;
- data manifests, file identifiers, schemas, and checksums;
- small test fixtures under `tests/fixtures/`;
- curated code-management artifacts;
- canonical or release validation-budget ledger snapshots when they are
  intentionally reviewed and committed.

GitHub should not contain:

- raw stock data;
- copied Google Drive data files;
- Colab runtime output directories;
- Drive backup outputs;
- run-copy notebooks such as `*_run_all.ipynb`, `*_full_run.ipynb`, or
  `*_drive_core_full_run.ipynb`;
- model checkpoints and trained-model binaries;
- large arrays or intermediate tables such as `.npy`, `.npz`, `.parquet`, or
  `.feather`;
- temporary caches, experiment tracking folders, or local notebook backups.

## Raw Data

Raw data is not committed to Git. The active pattern is:

```text
GitHub:      code, configs, manifests, schemas, checksums, docs
Drive/Zenodo: raw data and large reproducibility assets
Colab:       install exact git commit, read manifest, download raw data, run
```

During active work, Google Drive may hold the five approved source ticker files
and explicit file IDs. For public archival release, use a proper data repository
such as Zenodo, OSF, or Hugging Face Datasets when license terms allow it.

The repository may track files such as:

```text
data/README.md
data/raw/MANIFEST.md
data/manifests/*.yaml
data/sample/*
tests/fixtures/*
```

The repository must not track the raw ticker data itself. If a required data
file is missing, report the exact missing path or manifest entry instead of
silently substituting another file.

## Colab Runs

Colab notebooks should be reproducible from a pinned code version and explicit
data manifest:

1. install `intraday_research` from an exact git commit or versioned archive;
2. record the repo URL, git commit, config hash, notebook hash, input artifacts,
   output artifacts, validation scope, and `holdout_test_contact` in a run
   manifest;
3. read raw data through the manifest or documented user-provided local path;
4. write outputs first to the Colab runtime filesystem;
5. copy outputs to Drive only through an explicit optional backup step.

Committed notebooks should have empty outputs unless the file is explicitly a
review artifact. Run-copy notebooks and backup notebooks are local/runtime
artifacts, not canonical source.

## Results And Runtime Artifacts

Use `results/<stage_name>/` for reproducible stage outputs and
`artifacts/` for supporting review or code-management artifacts. Runtime
subdirectories are ignored by default:

```text
artifacts/runtime_caches/
artifacts/colab_runs/
artifacts/drive_backups/
artifacts/tmp/
artifacts/run_manifests/
```

Curated artifacts may be tracked when they are intentionally small and reviewed:

```text
artifacts/code_management/
artifacts/ledger_snapshots/
```

Do not move a runtime artifact into a tracked location unless it has a clear
research purpose, stable schema, and reviewable provenance.

## Validation-Budget Ledger

The validation-budget ledger is a research-governance artifact, not an ordinary
runtime log. Per `AGENTS.md` section 4.3, any downstream read of an
official-validation metric must append a ledger row before the read, and
pre-existing rows must not be modified, dropped, or reordered.

Recommended policy:

- runtime ledgers created during exploratory Colab execution may live under
  ignored runtime output directories;
- canonical or release ledger snapshots may be tracked under
  `artifacts/ledger_snapshots/` after review;
- ledger snapshots must preserve prefix invariance;
- no code path may use the ledger to justify post-hoc model, threshold, feature,
  or wording changes.

## Test Fixtures

Small synthetic or truncated fixtures may be committed when they are needed for
tests. They must be small enough for normal Git history, must not contain
licensed raw market data unless redistribution is allowed, and must not be used
as research evidence.

Allowed examples:

```text
tests/fixtures/tiny_ohlcv.csv
tests/fixtures/notebook07_minimal_ledger.csv
tests/fixtures/sample_manifest.json
```

Disallowed examples:

```text
tests/fixtures/full_CSCO_5min_history.csv
tests/fixtures/copied_drive_raw_data.csv
tests/fixtures/real_holdout_predictions.csv
```

## Review Checklist

Before staging artifact-policy related changes, run:

```bash
git status --short
git diff -- .gitignore docs/ARTIFACT_POLICY.md
git diff --check
git check-ignore -v data/raw/example.csv
git check-ignore -v data/raw/MANIFEST.md
git check-ignore -v notebooks/example_run_all.ipynb
git check-ignore -v tests/fixtures/tiny.csv
```

Expected behavior:

- raw data examples are ignored;
- manifests are not ignored;
- run-copy notebooks are ignored;
- test fixtures are not ignored;
- no notebook, script, test, raw data, or validation output is modified by this
  policy change.
