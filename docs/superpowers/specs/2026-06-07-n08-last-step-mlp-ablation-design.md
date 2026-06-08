# N08 #5D-7 — Last-Step MLP Sequence Ablation (`deep_sequence/controls.py`) Design

> Status: design 2026-06-07. The final unimplemented §7.1 family — a CONTROL /
> ablation, not a deep sequence model. Tooling: inline design + `humanize:ask-codex`
> review. Coexistence: model package co-developed with a parallel Codex session;
> check `git status` clean before editing.

## 1. Goal & Scope

Implement `LastStepMLPSequenceAblation.fit` / `.predict_proba` (currently
`NotImplementedError` scaffolds in `controls.py`) as a tiny MLP trained on the
LAST bar of each window (`X[:, -1, :]`) only — the deliberate
sequence-vs-last-step ablation that deep families must beat (design §14.5
complexity penalty; §11.1 / §9.4). It is the last entry in the interface test's
`_NOT_YET_IMPLEMENTED_FAMILIES`.

It is the 6th thin subclass of `_SequenceTorchClassifier` (`_torch_base.py`): a
single torch module trained on `X`, so it inherits fit / `_train` /
chronological-tail early stop / determinism with global-state restore /
`_validate_x` / `predict_proba`. It supplies only `_validate_axes` +
`_build_module`.

**Out of scope (unchanged):** data-agnostic body — no data/folds/HPO/08F/08O; no
frozen-config edit. The "deep must beat the ablation" gate is the future 08F
orchestrator's job, not this body's.

## 2. Architecture
`_LastStepMLPModule(nn.Module)`: `forward(x (b, L, C))` → take `x[:, -1, :]`
(`(b, C)`, the last completed bar) → `Linear(C, hidden_size) → ReLU →
Dropout(dropout) → Linear(hidden_size, 2)` → `(b, 2)` logits. The rest of the
window is intentionally discarded — that IS the ablation (isolates the
last-step contribution). `softmax` in the inherited `predict_proba`, float64.

**No causality concern:** using only `x[:, -1, :]` (the window's last completed
bar) reads no future bar — there is no per-timestep leak path, so no causal gate
is needed (unlike TCN/GRU/LSTM, which read the whole window).

## 3. Search Axes (spec-introduced; mirror SmallFusionMLP's MLP axes)
| axis | values | validation |
|---|---|---|
| `hidden_size` | 8, 16, 32 | exact int in frozen set (reject bool/numpy) |
| `dropout` | 0.0, 0.05, 0.10 | exact float in frozen set |

Strict exact-type (these are user-specified frozen search axes — unlike the
losses' caller-provided numpy stats; consistent with GRU/LSTM/SmallFusionMLP
axis discipline). `random_state` validated at fit (inherited). Training kwargs
(`max_epochs` etc.) come from the class with the shared defaults.

**08X-eligibility:** `last_step_mlp_sequence_ablation` IS already in
`SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES` (it is a control). But `hidden_size` /
`dropout` are spec-introduced sub-axes; they need a config / search-space mirror
before an 08X run varies them (parallels the GRU/LSTM/SmallFusionMLP gate). The
class docstring states this.

## 4. Files
- **Modify** `src/intraday_research/models/deep_sequence/controls.py` (implement
  `LastStepMLPSequenceAblation` as a `_SequenceTorchClassifier` subclass + add
  `_LastStepMLPModule`). `LastStepLightGBMControl` (already implemented, #5A) is
  untouched.
- **Create** `tests/stages/models/test_last_step_mlp_ablation.py`.
- **Modify** `tests/stages/models/test_models_deep_sequence_interface.py` (remove
  `LastStepMLPSequenceAblation` from `_NOT_YET_IMPLEMENTED_FAMILIES` — it becomes
  empty; the two NotImplementedError-anchor tests become empty-parametrized
  skipped placeholders [fine in this repo — no `empty_parameter_set_mark` config].
  Update the file docstring so it no longer claims scaffold-`NotImplementedError`
  checks are its purpose — all 10 families are now implemented; Codex P3).
- No change to `_torch_base.py`, the other models, configs, contracts, notebooks.

## 5. Testing (`tests/stages/models/test_last_step_mlp_ablation.py`, mirror prior model tests)
- Protocol conformance + proba contract (shape (n,2), float64, rows sum 1).
- Determinism (bit-exact same seed; global-state restore).
- **Last-step-only ablation (the key test):** perturbing ALL bars EXCEPT the last
  (`X[:, :-1, :]`) must leave `predict_proba` BIT-IDENTICAL — the HARD invariant
  proving the model reads only `X[:, -1, :]` (the opposite of the deep models'
  sequence-signal test). A secondary check perturbs the last bar with a LARGE
  change and asserts predictions differ — kept robust (large perturbation) to
  avoid flakiness on learned weights (Codex P2; the non-last-bar invariant is the
  primary proof).
- Axis coverage: `hidden_size ∈ {8,16,32}`, `dropout ∈ {0.0,0.05,0.10}`.
- Guards: off-grid / wrong-type / bool `hidden_size` & `dropout`; shared base
  guards (random_state at fit, X-not-3D, y single-class, predict-before-fit,
  predict shape-drift); empty-batch predict `(0,2)` via the inherited path.
- Early-stop bookkeeping (`actual_epochs_` / `early_stop_reason_` /
  `internal_val_n_`).
- The interface `_NOT_YET_IMPLEMENTED_FAMILIES`-emptied state still passes (the
  protocol-conformance test over `_ALL_FAMILIES` covers it now).

## 6. Open Decisions for Review
1. **Axis values** (`hidden_size ∈ {8,16,32}`, `dropout ∈ {0.0,0.05,0.10}`) are a
   spec-introduced reviewable default mirroring SmallFusionMLP; user may adjust.
2. **`_SequenceTorchClassifier` subclass** (module extracts the last step) vs a
   bespoke non-base class — the subclass reuses the whole training contract for
   free and is the right shape; confirm.
3. **No causal gate** (last-step MLP reads only the last completed bar, no future
   path) — confirm sufficiency.
