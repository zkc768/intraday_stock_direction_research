# N08 #5D-3 — Shallow GRU Classifier Body + `_SequenceTorchClassifier` Shared-Trainer Refactor

> Status: design 2026-06-07. Third deep model body of #5 Half 2, and the
> rule-of-three trigger for the shared-trainer refactor that the DLinear (§10.2)
> and TCN (§10.2) specs explicitly deferred to "the 3rd model (GRU)".
> Tooling: inline design + `humanize:ask-codex` review (Humanize swarm
> unavailable — Claude subagents fail to spawn under the cc-switch model remap).

## 1. Goal & Scope

Two coupled changes, designed and reviewed together because the GRU is the
abstraction's third concrete shape:

1. **Refactor (behavior-preserving):** hoist the training/predict machinery that
   DLinear and TCN currently duplicate verbatim into a private
   `_SequenceTorchClassifier` base. DLinear, TCN, and GRU become thin subclasses
   that supply only `_validate_axes` + `_build_module`.
2. **GRU:** implement `ShallowGRUClassifier.fit` / `.predict_proba` (currently
   `NotImplementedError` scaffolds) as a thin subclass of that base, honoring the
   §7.1 `shallow_gru` family.

**Out of scope (unchanged from DLinear/TCN):** no data loading, folds, train/val
splitting, HPO, trial ledger, 08F/08O, `run_stage`. Data-agnostic bodies only.
No edit to the frozen config YAML, contracts, notebooks, or data modules.

**Sequencing requirement (Codex design review P2):** the refactor and the GRU
must land as TWO sequential checkpoints, NOT one indistinguishable change, so a
refactor regression cannot be masked by new GRU tests:
- **Checkpoint 1 (refactor):** create `_torch_base.py`, convert ONLY DLinear +
  TCN to subclasses, run `test_dlinear.py` + `test_tcn.py` + the 158-test models
  gate UNCHANGED and green. Commit.
- **Checkpoint 2 (GRU):** add `ShallowGRUClassifier` + `test_gru.py` + the
  interface-list edit. Run three gates. Commit.

## 2. The Refactor — `_SequenceTorchClassifier`

**Placement: new module `src/intraday_research/models/deep_sequence/_torch_base.py`**
(private). It does NOT go in `base.py`: that module's own docstring states
"This module declares the protocol only; nothing here trains a model, reads data,
or writes artifacts." A torch training base belongs in its own private module;
`base.py` stays the pure `SequenceClassifier` protocol. The base class name keeps
a leading underscore (`_SequenceTorchClassifier`) — internal machinery, never in
`__init__.__all__`, never instantiated by the orchestrator.

**Base owns (hoisted verbatim from the identical DLinear/TCN code):**

- `__init__(*, random_state, max_epochs, learning_rate, batch_size,
  early_stopping_patience, early_stopping_fraction, weight_decay)` — stores the
  shared training kwargs + the post-fit state attrs (`_model`, `_window_size`,
  `_n_features`, `actual_epochs_`, `early_stop_reason_`, `internal_val_n_`), then
  calls `self._validate_axes()` **then** `self._validate_training_kwargs()`.
  Axes-first preserves the original DLinear/TCN validation order (both validate
  axes before training kwargs), so single-bad-value guard tests are unaffected.
- `_validate_training_kwargs()` — exact-type + range validation for the six
  shared training kwargs (`max_epochs`, `learning_rate`, `batch_size`,
  `early_stopping_patience`, `early_stopping_fraction`, `weight_decay`), same
  messages as today. NOT `random_state` (that stays a fit-time check, so the
  model is constructible with defaults for the protocol/orchestrator).
- `_validate_x(X, *, where)` (static) — 3-D float, finite, positive
  window/feature dims; returns a contiguous float32 copy. Uses the
  **DLinear-compatible messages** (Codex P2): `"...X must be a 3-D ndarray..."`,
  `"...X window_size and n_features must be positive..."`, `"...X contains
  NaN/inf"` — DLinear's empty-axis test matches `"window_size and n_features"`,
  and TCN has no conflicting empty-axis assertion, so the DLinear wording is the
  safe shared choice.
- `fit(X, y)` — fit-time `random_state` int check; `y` validation (1-D, integer,
  `{0,1}`, both classes present, length match); determinism global-state
  save/restore (deterministic-algorithms flag + `warn_only` sub-state + RNG
  stream) in `try/finally`; `torch.manual_seed`; `_early_stop_split`;
  `model = self._build_module(window_size=W, n_features=C)`; `_train`; store
  post-fit state. The `_build_module` call is the only model-specific line and it
  runs INSIDE the seeded block so init draws from the seeded stream.
- `_early_stop_split(y)` — chronological-tail split (AGENTS.md §4.1 forbids random
  internal splits); fit = leading rows, val = trailing slice; falls back to
  no-split when too small.
- `_train(model, x_arr, y_arr, fit_idx, val_idx)` — Adam + CrossEntropy;
  mini-batches in caller order with NO shuffle (§4.1); chronological-tail early
  stop with best-state restore; sets `actual_epochs_` / `early_stop_reason_`.
- `predict_proba(X)` — fitted check, shape-drift check, softmax → float64.
  Error messages use `type(self).__name__` so each subclass keeps its own class
  name in the message (tests match on stable substrings like `"before fit"` /
  `"differs from the fitted"`, which are preserved).
- `_forward_features(X)` — delegates to `self._model.forward_features(tensor)`
  (hoisted out of TCN). **Guarded (Codex P2):** because inheritance makes
  `hasattr(DLinearClassifier(), "_forward_features")` true while `_DLinearModule`
  has no `forward_features`, the base first checks
  `hasattr(self._model, "forward_features")` and raises a deliberate
  `NotImplementedError` naming `type(self).__name__` if absent — so DLinear fails
  loud and clear, not with a confusing bare `AttributeError`. TCN/GRU modules
  define `forward_features` and inherit the working path.

**Subclass hooks (base raises `NotImplementedError`):**

- `_validate_axes(self) -> None` — model-specific exact-type + frozen-set axis
  validation (the only per-model validation).
- `_build_module(self, *, window_size, n_features) -> nn.Module` — construct the
  torch module. DLinear uses both args; TCN/GRU use `n_features` only.

**Regression guard:** this is behavior-preserving. The existing `test_dlinear.py`
(39) and `test_tcn.py` (43) plus the whole models gate (158) MUST stay green
byte-for-byte WITHOUT test edits. No test references a method that moves in a way
that breaks: `_early_stop_split` and `_forward_features` are inherited and called
on instances; the DLinear chronological-split lock test still resolves.

## 3. GRU Architecture (shallow, causal-by-construction → binary logits)

Forward over `x` `(b, L, C)`:

1. `nn.GRU(input_size=C, hidden_size, num_layers, batch_first=True,
   bidirectional=False, dropout=0.0)` → output sequence `(b, L, H)`.
   `forward_features(x)` returns this sequence (exposed for the causal test).
2. **Head** (`head`):
   - `"last_step"`: `out[:, -1, :]` → `(b, H)`. Note this is the GRU *output*
     (hidden state after integrating the whole window), not the last *input*
     bar — so it still reads sequence structure.
   - `"attention_pooling_pre_frozen"`: `Linear(H, 1)` scores → `softmax` over the
     time axis `L` → weighted sum → `(b, H)`. Same semantics + naming as TCN §3
     (head shape frozen; attention over within-window time only, no
     cross-window/day mixing).
3. `nn.Dropout(dropout)` on the pooled features → `Linear(H, 2)`.

`softmax(logits, dim=1)` in `predict_proba`, returned float64.

**Causal-by-construction (§4.1):** a unidirectional GRU is a strict left-to-right
recurrence — `output[t]` depends only on inputs `≤ t`. `bidirectional` is a fixed
`False` axis; a bidirectional GRU would leak future bars into earlier timesteps
and is forbidden. No future-aware mechanism exists; the head selects/pools within
the window only.

**Dropout semantics (deliberate, and a pytest.ini-trap avoidance):**
PyTorch's `nn.GRU(dropout=p)` applies dropout only *between* stacked layers and
emits a `UserWarning` when `num_layers == 1`. To give the `dropout` axis uniform,
layer-count-independent meaning AND avoid that warning, the module passes
`dropout=0.0` to `nn.GRU` and realizes the `dropout` axis as a single
`nn.Dropout` on the pooled features before the classifier (mirrors DLinear's
explicit `nn.Dropout`). Decision flagged in §10.

## 4. Search Axes — the key open decision

**Unlike DLinear (§7.2) and TCN (§7.3), the N08 design freezes NO GRU axis table.**
§7.1 lists `shallow_gru` only as a candidate family; the config YAML
(`configs/stages/deep_sequence_exploration.yaml`) has `dlinear:` and `tcn:` axis
blocks but NO `gru:`/`shallow_gru:` block. The GRU search space is therefore
genuinely unfrozen upstream. This spec PROPOSES a frozen "shallow" search space —
consistent with the sibling frozen-set discipline (off-grid fails loud in the
body, keeping config hashes unambiguous) — and flags it as the primary open
decision (§10.1). The proposal is derived from the scaffold defaults
(`hidden_size=32, num_layers=1, dropout=0.0, head="last_step"`) plus the
"shallow" constraint.

| axis | values | validation |
|---|---|---|
| `hidden_size` | 16, 32, 64 | exact int in frozen set (reject bool) |
| `num_layers` | 1, 2 | exact int in frozen set |
| `dropout` | 0.0, 0.05, 0.10, 0.20 | exact float in set (mirrors TCN) |
| `head` | "last_step", "attention_pooling_pre_frozen" | exact str in set (mirrors TCN) |
| `bidirectional` | False (fixed) | exact bool `is False`; reject `True`, `0`, `"false"` — §4.1 causal red line (parallels TCN `causal is True`) |

`random_state` validated at `fit` (as siblings).

**08X-eligibility gate (Codex P1 — contract risk resolution).** Codex correctly
flags that body-side validation alone does NOT make the orchestrated 08X
search/audit surface frozen: 08X requires a finite, logged, sha-stamped search
space before trial 0, and DLinear/TCN axes are already mirrored into the config
(`dlinear:`/`tcn:` blocks) while GRU's are not. To bound that risk WITHOUT editing
the frozen config artifact in this body PR:

- The body validates its proposed frozen set purely for **fail-loud + hash-stable
  construction** (same discipline as siblings) — it makes a GRU instance
  self-consistent, nothing more.
- **GRU is body-constructible but NOT 08X-search-eligible** until its axes are
  mirrored into `configs/stages/deep_sequence_exploration.yaml` (a `gru:` /
  `shallow_gru:` block) AND `08x_search_space.json`, and sha-stamped before
  trial 0. That config/search-space freeze is a **future, in-scope change owned by
  the 08X harness piece**, not this model-body PR. The `ShallowGRUClassifier`
  docstring states this gate explicitly so no one mistakes the body-side tuples
  for an 08X-ready frozen search space.

This parallels TCN's `num_blocks=4` resolution: document the upstream gap loudly
rather than silently papering over it. The proposed axis VALUES below remain a
reviewable default (the upstream froze none); changing them is a trivial edit to
the frozen tuples (+ the future config mirror).

## 5. Training Protocol

Inherited verbatim from `_SequenceTorchClassifier` (§2): Adam, CrossEntropy,
chronological-tail internal early-stop split with no mini-batch shuffle,
determinism with full global-state save/restore (incl. RNG stream + `warn_only`),
`random_state` required at fit, training kwargs (`max_epochs`, `learning_rate`,
`batch_size`, `early_stopping_patience`, `early_stopping_fraction`,
`weight_decay`) with the same defaults + exact-type ranges. The module is built
under `torch.manual_seed(random_state)`; GRU weight init draws from the seeded
stream.

**Determinism risk — de-risked.** The Codex design review RAN a tiny same-seed
`nn.GRU` smoke under `torch.use_deterministic_algorithms(True)` on `torch
2.12.0+cpu` and got bit-exact equality (max diff 0.0). The bit-exact determinism
test remains the gate, but CPU GRU non-determinism is not anticipated; no
`warn_only` relaxation or env var is needed.

## 6. Error Modes (fail-loud)

All shared modes via the base (axis exact-type/frozen-set through `_validate_axes`;
training kwargs; `random_state` at fit; X 3-D float finite; predict-before-fit;
y 1-D int `{0,1}` + both classes + length; predict shape-drift) PLUS GRU-specific:

| # | condition | exception |
|---|---|---|
| G1 | `bidirectional` not exactly `False` (the `is False` singleton check rejects `True`, `0`, AND `1` — Codex wants all three tested) | `ValueError` (construction) |
| G2 | `hidden_size` / `num_layers` / `dropout` / `head` off-grid or wrong exact-type | `ValueError` (construction) |

## 7. Testing

### 7a. Refactor regression (no new behavior)

`test_dlinear.py` + `test_tcn.py` + the models gate stay green UNCHANGED. They are
the regression guard for the hoist. (Only `dlinear.py` and `tcn.py` source change,
to subclass the base; their test files are not edited.)

### 7b. GRU — `tests/stages/models/test_gru.py` (~30–35 tests), mirroring DLinear/TCN:

1. **Protocol conformance / proba contract** (shape `(n,2)`, float64, rows sum 1).
2. **Determinism + global-state restore** (bit-exact same seed; restores
   deterministic flag + `warn_only` + RNG stream).
3. **Causal-by-construction (light §4.1 gate):** expose `_forward_features` (GRU
   output seq `(b, L, H)`); build two inputs identical up to time `t`, arbitrary
   after; assert output rows `≤ t` are bit-identical (time is axis 1 for GRU vs
   axis 2 for TCN). Lighter than TCN's gate (parametrize only
   `head ∈ {last_step, attention_pooling}` × `num_layers ∈ {1, 2}` rather than
   the full conv-axis sweep) because a unidirectional GRU is causal by
   construction — the test guards against a `bidirectional=True` regression.
4. **Sequence-only signal** (slope sign, identical last *input* bar) → train acc
   `> 0.6`, proving the recurrence uses temporal structure.
5. **Axis coverage:** `hidden_size ∈ {16,32,64}`, `num_layers ∈ {1,2}`,
   `dropout ∈ {0,.05,.10,.20}`, `head ∈ {last_step, attention_pooling}`; each
   fits + shape-conforms.
6. **Guards:** `bidirectional` ∈ {`True`, `0`, `1`} all rejected at construction
   (the `is False` singleton check); off-grid / wrong-type `hidden_size` /
   `num_layers` / `dropout` / `head`; plus the shared X/y/predict guards asserted
   through GRU (random_state-at-fit, X-not-3D, y-single-class, predict-before-fit,
   predict-shape-drift).
7. **Early-stop bookkeeping** (`actual_epochs_`, `early_stop_reason_`,
   `internal_val_n_`; small-n no-internal-val path).
8. **No-warning gate:** fit with `dropout=0.10, num_layers=1` inside
   `warnings.catch_warnings()` + `simplefilter("error")` and assert it does NOT
   raise — directly verifies the §3 dropout decision suppresses the torch
   `num_layers==1` `UserWarning` (independent of pytest.ini's module-scoped
   filter, which would not catch a `torch.nn`-issued warning anyway).
9. **Dropout-placement assertion (Codex P3):** after fit, assert the underlying
   module's `nn.GRU.dropout == 0.0` (confirming the `dropout` axis is realized as
   pooled-feature dropout, never `nn.GRU` between-layer dropout) — for both
   `num_layers=1` and `num_layers=2`.

## 8. Files

- **Create** `src/intraday_research/models/deep_sequence/_torch_base.py`
  (`_SequenceTorchClassifier`).
- **Modify** `dlinear.py` — subclass the base; delete the hoisted
  `_validate_x`/`fit`/`_early_stop_split`/`_train`/`predict_proba` + the training
  half of `_validate_init` (renamed to `_validate_axes`); keep `_DLinearModule`,
  `_validate_axes`, `_build_module`.
- **Modify** `tcn.py` — same pattern; keep `_CausalConv1d`/`_TCNBlock`/
  `_TCNModule` (with `forward_features`), `_validate_axes`, `_build_module`; drop
  the duplicated training loop + the now-inherited `_forward_features`.
- **Modify** `gru.py` — implement `ShallowGRUClassifier` + `_GRUModule` as a thin
  subclass.
- **Create** `tests/stages/models/test_gru.py`.
- **Modify** `tests/stages/models/test_models_deep_sequence_interface.py` — move
  `ShallowGRUClassifier` out of `_NOT_YET_IMPLEMENTED_FAMILIES`.
- **No change** to `base.py`, `folds.py`, `controls.py`, `fusion.py`, `lstm.py`,
  `losses.py`, configs, contracts, notebooks, or data modules.

## 9. Dependencies

`torch` 2.12.0+cpu + numpy. `base.SequenceClassifier` (GRU still satisfies the
protocol). Downstream: the 08X harness instantiates per-trial; LSTM (#5D-4) will
be the 4th thin subclass of `_SequenceTorchClassifier`, validating the
abstraction further.

## 10. Open Decisions for Review

1. **GRU axis VALUES — the one decision still genuinely open for the user.** §7.1
   names the family but freezes no axes; the config has no `gru:` block. The
   contract risk this creates is RESOLVED by the §4 08X-eligibility gate
   (body-constructible + self-validating, but not 08X-search-eligible until the
   config/`08x_search_space.json` mirror lands). What remains is the choice of the
   proposed axis VALUES: {`hidden_size`: 16/32/64; `num_layers`: 1/2; `dropout`:
   0/.05/.10/.20; `head`: last_step / attention_pooling; `bidirectional`:
   False-fixed}. These are a reviewable default derived from the scaffold; the
   user may adjust them (a trivial edit to the frozen tuples).
2. ~~Dropout as a single head-input `nn.Dropout`~~ — **Codex-confirmed sound**
   (avoids the `num_layers==1` warning, layer-count-independent). §3 + test 9.
3. ~~`attention_pooling_pre_frozen` = learned temporal-attention pool~~ — same as
   TCN §10.1 (already accepted there). Carried over.
4. ~~Refactor placement (`_torch_base.py`, not `base.py`) + rule-of-three timing~~
   — sound; staged into two checkpoints per Codex P2 (§1).
5. ~~Light causal gate for GRU~~ — **Codex-confirmed sufficient** given exact
   `bidirectional is False` tested against True/0/1; a unidirectional GRU over a
   completed window adds no future-source path (AGENTS §4.1). §7b test 3 + 6.

## 11. Codex Design Review — absorbed (2026-06-07, gpt-5.5:high, 156s)

- **P1 (GRU axes not upstream-frozen → 08X contract risk):** resolved via the §4
  08X-eligibility gate (body self-validates for hash stability; explicitly
  08X-ineligible + docstring-flagged until config/search-space mirror; no frozen
  config edit in this PR).
- **P2 (stage the refactor):** §1 now mandates two checkpoints (refactor →
  regression-green → commit; then GRU).
- **P2 (`_forward_features` inheritance leak):** §2 base now guards
  `hasattr(self._model, "forward_features")` and raises `NotImplementedError`
  naming the subclass.
- **P2 (shared `_validate_x` strings):** §2 fixes the shared messages to the
  DLinear-compatible wording (`"window_size and n_features"`, `"3-D"`, `"NaN/inf"`).
- **P3 (dropout naming):** §7b test 9 asserts `nn.GRU.dropout == 0.0`.
- **Confirmed sound by Codex:** validation order (axes before training kwargs,
  axes set before `super().__init__()`, TCN `channels` list→tuple preserved);
  `type(self).__name__` messages; GRU dropout decision; light causal gate; CPU
  GRU determinism (Codex ran a same-seed smoke → bit-exact, max diff 0.0).
