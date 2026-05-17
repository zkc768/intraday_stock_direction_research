# NEXT_SESSION_BRIEF.md — hf_stock_clf / ml_utils

## Current status

Project root:

```text
E:\codex_workspace\projects\hf_stock_clf
```

Latest confirmed state:

- W5.3 trainer implementation review: PASS
- MVP full validation audit: PASS
- MVP full validation log update: PASS
- Latest commit: `ba07403 docs(log): record MVP full validation`
- Full pytest: `86 passed, 1 warning`
- Collect-only: `86 tests collected`
- `pip check`: No broken requirements found

MVP production files complete:

```text
ml_utils/config.py
ml_utils/seed.py
ml_utils/metrics.py
ml_utils/dataset.py
ml_utils/checkpoint.py
ml_utils/models/lstm_classifier.py
ml_utils/trainer.py
```

Phase 1B files remain absent and must stay absent for now:

```text
ml_utils/models/tcn_classifier.py
ml_utils/models/dlinear_classifier.py
```

`notebooks/` is still empty.

## Current phase

Current phase is W6 notebook integration planning.

W6.0 notebook cell plan has been produced and is approved with amendments.

Next implementation step:

```text
W6.1 create Notebook 01 single-stock LSTM smoke
```

Do not start final handoff yet.

Do not start TCN / DLinear.

Do not create Notebook 02 yet.

## W6.0 approved amendments

### 1. Public API escalation rule

Notebook smoke is integration validation, not a patch layer.

If notebook construction or execution exposes a missing or unsuitable `ml_utils` public API, STOP and return to the corresponding `ml_utils` module for a production-code fix, tests, and review.

Forbidden notebook workarounds:

- monkey patching
- temporary wrappers
- copying internal `ml_utils` logic
- accessing private functions or attributes
- reimplementing trainer loops
- reimplementing metrics or baseline logic

### 2. Pooled scaler workflow is not approved yet

Notebook 02 is deferred until the public `dataset.py` API is confirmed to support the intended pooled workflow.

The intended Notebook 02 design says: per-ticker split, then fit one global scaler on combined train segments.

Before creating Notebook 02, confirm whether public `dataset.py` API naturally supports this.

If not, STOP and return to `dataset.py`; do not implement global scaler composition manually in the notebook.

### 3. Shuffled-label sanity check needs concrete threshold

Notebook 02 shuffled-label sanity check must not use vague language like “close to random.”

Proposed rule:

```text
PASS if shuffled_label_macro_f1 <= 1.10 * dummy_stratified_macro_f1_mean
WARNING if shuffled_label_macro_f1 <= 1.20 * dummy_stratified_macro_f1_mean
FAIL if shuffled_label_macro_f1 > 1.20 * dummy_stratified_macro_f1_mean
```

The dummy baseline must be fit on train labels and evaluated on the same validation/test target used for the shuffled-label model.

### 4. Reproducibility is required

Notebook 01 and Notebook 02 must include a top-level seed parameter, for example:

```python
SEED = 42
seed_everything(SEED)
```

The notebook must print one compact line confirming the seed.

## W6.1 scope

W6.1 creates only:

```text
notebooks/01_smoke_test_single_stock_lstm.ipynb
```

Allowed:

- create Notebook 01
- inspect public `ml_utils` API signatures
- write notebook cells according to W6.0 plan
- include clear user-facing `DATA_PATH` config

Forbidden:

- modifying `ml_utils/*`
- modifying tests
- creating Notebook 02
- creating TCN / DLinear files or cells
- creating helper `.py` scripts
- using test fixtures as fake real data
- running training
- git add / commit / push

## Notebook 01 required direction

Notebook 01 must validate whether the MVP components can be naturally orchestrated end to end:

```text
DataConfig / WindowConfig / TrainConfig / ModelConfig
→ seed_everything
→ label generation
→ time split
→ train-only scaler fit
→ transform split
→ boundary trim
→ WindowedClassificationDataset
→ DataLoader
→ LSTMClassifier
→ Trainer.fit
→ evaluation metrics
→ baseline comparison
→ compact final table
```

Notebook 01 must not claim final benchmark results.

Notebook 01 must not include TCN or DLinear.

Notebook 01 must not create synthetic data to fake a successful smoke test.

Notebook 01 may include a user-editable data path such as:

```python
DATA_PATH = Path("data/CSCO.csv")
```

If the real data file is missing, the notebook should raise a clear `FileNotFoundError` and tell the user to update `DATA_PATH`.

## Notebook 01 output discipline

Notebook 01 should keep output compact.

Allowed:

- one compact config / seed summary
- one compact data loading summary
- one compact label distribution summary
- one compact split / scaler summary
- one compact dataset / DataLoader shape summary
- one compact model / trainer setup summary
- one compact final comparison table
- one compact confusion matrix table if useful

Forbidden:

- `df.head()` dumps
- dumping tensor values
- dumping full model summary
- printing full `classification_report` text
- repeated debug prints
- long exploratory outputs
- plotting by default

## API blocker rule

If API blocker appears, use this STOP format:

```markdown
## API BLOCKER

- module:
- public API problem:
- why notebook workaround is forbidden:
- suggested return step:
- files that likely need tests updated:
```

Examples of API blockers:

- `Trainer.fit` cannot be called cleanly from notebook without private attributes
- checkpoint path / monitor metric / device cannot be passed through public API
- `dataset.py` public API cannot naturally complete label / split / scaler / trim / window flow
- `metrics.py` public API cannot produce a compact comparison table
- `LSTMClassifier` does not accept the DataLoader batch shape
- notebook would need to copy internal `ml_utils` logic to proceed

## Next action

Next action is W6.1:

```text
Create notebooks/01_smoke_test_single_stock_lstm.ipynb
```

Before W6.1, ensure git working tree is clean.

After W6.1, review the notebook manually before running it.

Do not proceed to Notebook 02 until Notebook 01 is reviewed and any API blockers are resolved.

Do not proceed to final handoff until notebook integration evidence exists.