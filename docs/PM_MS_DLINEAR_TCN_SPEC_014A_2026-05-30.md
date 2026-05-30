# PM-MS-DLINEAR-TCN-SPEC-014A

Date: 2026-05-30
Owner: PM route control
Status: spec lock before test-first work

## Goal

Move Ian's combined MS-DLinear+TCN route from raw notebook idea to a
test-first implementation gate. This packet is not an implementation, not a
training approval, and not a model-quality claim.

The next valid progress step after this packet is a narrow test-first session
for the future combined model class. Runner integration, notebook cleanup,
training, and result claims remain separate tasks.

## Current State

- `ml_utils.models.dlinear_classifier.DLinearClassifier` exists as a separate
  baseline model.
- `ml_utils.models.tcn_classifier.TCNClassifier` exists as a separate baseline
  model.
- No canonical combined MS-DLinear+TCN model exists in `ml_utils/models`.
- The local runner currently maps torch model names to separate `lstm`, `tcn`,
  and `dlinear` model paths.
- Notebook 04 records a stock-aware multi-scale DLinear plus residual TCN idea,
  but it is an untracked prototype and is `raw_material_only`.

Forbidden claims:

- Do not claim combined MS-DLinear+TCN exists.
- Do not claim Notebook 04 is an implementation.
- Do not claim any performance, robustness, or component effect.

## Source Roles

| Source | Allowed role | Forbidden role |
| --- | --- | --- |
| `ml_utils/models/dlinear_classifier.py` | Existing DLinear shape and decomposition contract | Final combined architecture proof |
| `ml_utils/models/tcn_classifier.py` | Existing causal TCN shape and residual-block contract | Final combined architecture proof |
| `notebooks/04_ian_no_trade_band_multiscale_dlinear_tcn.ipynb` | Raw design input only | Evidence, result claim, canonical implementation |
| `docs/MENTOR_CLEAN_V1_PROTOCOL_LOCK_2026-05-30.md` | Feature/scaler/threshold protocol lock | Permission to train or tune |
| `docs/ml_utils_construction_plan_v2.md` | Existing model/test discipline | Combined model spec, because no combined section exists yet |

## Architecture Decision Table

| Decision | Lock |
| --- | --- |
| Future model file | `ml_utils/models/ms_dlinear_tcn_classifier.py` |
| Future class | `MultiScaleDLinearTCNClassifier` |
| Future runner model name | `ms_dlinear_tcn` |
| Input shape | `(batch, seq_len, input_size)` |
| Output shape | raw logits `(batch, num_classes)` |
| Sequence length | Fixed by `seq_len`, because the DLinear side is fixed-sequence |
| Dataset API | Unchanged in v0 |
| Stock embedding | Blocked from v0; requires a separate design change |
| Multi-scale moving averages | Positive odd kernels only, default candidate `(3, 5, 9, 15)` |
| TCN branch | Causal residual Conv1d branch with internal NLC to NCL transpose |
| Fusion | Late fusion of DLinear representation and TCN last-step representation |
| Threshold policy | Model does not choose thresholds |
| Scaler policy | Model does not fit scalers |
| Softmax/sigmoid | Forbidden in model; loss/metrics consume logits |

## V0 Model Contract

The future combined model should accept only the window tensor:

```text
x: torch.Tensor of shape (batch, seq_len, input_size)
```

and return:

```text
logits: torch.Tensor of shape (batch, num_classes)
```

The model must validate:

- `x` is 3D.
- `x.shape[1] == seq_len`.
- `x.shape[2] == input_size`.
- `moving_avg_kernels` is a non-empty sequence of positive odd integers.
- `tcn_channels` is a non-empty sequence of positive integers.
- `dropout` satisfies `0 <= dropout < 1`.
- `num_classes > 0`.

The model must not:

- Receive ticker ids in v0.
- Move tensors to a device inside `forward`.
- Read global config or global state.
- Create a threshold, scaler, label, or feature.
- Contain softmax or sigmoid modules.

## Minimal Architecture

V0 should keep the branches simple and testable:

1. Multi-scale DLinear branch.
   - For each allowed moving-average kernel, decompose the input sequence into
     seasonal and trend components.
   - Preserve `(batch, seq_len, input_size)` within each scale before the final
     branch representation.
   - Do not use even kernels unless a later approved spec defines exact padding
     semantics.

2. Causal TCN branch.
   - Consume the original clean feature window `x`.
   - Internally transpose from NLC to NCL.
   - Use causal padding/chomp behavior consistent with the existing TCN route.
   - Return a fixed-size last-step representation.

3. Fusion head.
   - Concatenate the DLinear branch representation and TCN representation.
   - Feed the concatenated representation into a final `Linear` classifier.
   - Return logits only.

## Stock Embedding Decision

Stock embedding is blocked from v0.

Reason:

- `AGENTS.md` requires the model to be transparent to data source.
- `WindowedClassificationDataset.__getitem__` currently returns only `(x, y)`.
- `Trainer` and evaluation paths call `model(x)`.
- Adding ticker ids would require dataset, trainer, evaluation, runner, and test
  changes in one wider design task.

Future unblock requires a separate prompt that states:

- Dataset output contract.
- Ticker vocabulary and split safety.
- Trainer/evaluate API changes.
- Leakage checks proving ticker identity is not derived from future outcomes.

## Protocol Constraints

The combined model route inherits the mentor clean protocol:

- `feature_set_id = mentor_clean_v1`
- `decision_time_policy = post_bar_close_completed_bar`
- `scaler_id = standard_pooled_train_only_v1`
- `threshold_source = fixed_pre_registered_5bps`
- `threshold_bps = 5.0`

The model class itself must remain unaware of those policies. They belong to
data preparation, runner metadata, and validation reports.

## Test-First Stage

The next allowed test-first file is:

```text
tests/test_ms_dlinear_tcn_classifier.py
```

The future implementation file is:

```text
ml_utils/models/ms_dlinear_tcn_classifier.py
```

The test-first file must use lazy imports so pytest collection succeeds before
the future implementation exists. It must not top-level import the future
module.

Allowed tests in the first test file:

- Forward output shape for `(batch, seq_len, input_size) -> (batch, 2)`.
- Custom `num_classes` output shape.
- Reject non-3D input.
- Reject wrong `seq_len`.
- Reject wrong `input_size`.
- Reject empty `moving_avg_kernels`.
- Reject even moving-average kernels.
- Reject empty `tcn_channels`.
- Reject invalid dropout.
- Assert outputs are raw logits, not probabilities.
- Assert the model has no softmax or sigmoid modules.
- Assert gradients flow through both branch and fusion parameters.

Defer to a separate runner task:

- `build_model("ms_dlinear_tcn", ...)` wiring.
- CLI `--models ms_dlinear_tcn` support.
- Phase 1B local runner metadata.
- Any validation or smoke command.

Expected validation at the test-first stage:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_ms_dlinear_tcn_classifier.py
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_ms_dlinear_tcn_classifier.py --collect-only -q
```

Running the tests before implementation may be red because the future module
does not exist. That red state is acceptable only for the test-first gate and
must not be reported as a passing implementation.

## Stop Rules

Stop before editing if any next task would:

- Modify `ml_utils` and tests in the same step without an explicit
  implementation prompt.
- Modify runner and model files in the same step.
- Execute notebooks.
- Train torch models.
- Read test metrics to select thresholds.
- Commit `.codegraph/*`.
- Commit untracked notebooks.
- Add stock embedding to v0.
- Use even moving-average kernels without a separate padding spec.
- Add attention, Transformer, model registry, dynamic import, callbacks, or
  plugin abstractions.

## Commit Policy

Commits are allowed only as atomic commits after validation:

1. Spec lock commit:
   - stage only this document.
2. Test-first commit:
   - stage only `tests/test_ms_dlinear_tcn_classifier.py`.

Forbidden:

- `git add .`
- `git add -A`
- staging `.codegraph/*`
- staging notebooks
- combining docs, tests, implementation, and runner changes in one commit

## Next Prompt

```text
你现在执行 PM-MS-DLINEAR-TCN-TESTS-014B -- Create test-first combined model tests.

Task type: test-first / tests-only / no implementation.

Allowed files:
- tests/test_ms_dlinear_tcn_classifier.py

Forbidden files:
- ml_utils/**
- scripts/**
- notebooks/**
- docs/**
- requirements.txt
- .gitignore

Goal:
- Create a lazy-import pytest file for the future
  MultiScaleDLinearTCNClassifier.
- Keep collection green before the implementation exists.
- Do not implement the model.

Validation:
- E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile tests\test_ms_dlinear_tcn_classifier.py
- E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_ms_dlinear_tcn_classifier.py --collect-only -q

Expected state:
- Test collection passes.
- Direct execution may fail until the model is implemented.
- No training, notebook execution, runner edits, or result claims.
```
