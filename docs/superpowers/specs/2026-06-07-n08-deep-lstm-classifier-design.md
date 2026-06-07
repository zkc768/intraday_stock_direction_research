# N08 #5D-4 — Shallow LSTM Classifier Body (`deep_sequence/lstm.py`) Design

> Status: design 2026-06-07. Fourth deep model body of #5 Half 2, and the 4th
> thin subclass of `_SequenceTorchClassifier` — its job is to validate the
> abstraction with no new training code. Near-mirror of #5D-3 GRU
> (`2026-06-07-n08-deep-gru-classifier-design.md`); this spec records only the
> deltas. Tooling: inline design + `humanize:ask-codex` review.

## 1. Goal & Scope

Implement `ShallowLSTMClassifier.fit` / `.predict_proba` (currently
`NotImplementedError` scaffolds) as a thin `_SequenceTorchClassifier` subclass
honoring the §7.1 `shallow_lstm` family. Structurally identical to
`ShallowGRUClassifier` with `nn.LSTM` replacing `nn.GRU`.

**Out of scope (unchanged from siblings):** data-agnostic body — no data loading,
folds, train/val splitting, HPO, trial ledger, 08F/08O, `run_stage`; no edit to
the frozen config YAML, contracts, notebooks, or data modules.

## 2. Reuse — the entire shared contract

Inherits verbatim from `_SequenceTorchClassifier` (`_torch_base.py`):
`fit` / `_train` / `_early_stop_split` (chronological-tail, AGENTS.md §4.1 — no
random split, no mini-batch shuffle) / `_validate_x` / `predict_proba` /
determinism with full global-state save/restore / `_forward_features` (guarded
delegate) / `_validate_training_kwargs`. LSTM supplies ONLY `_validate_axes` +
`_build_module`, exactly like GRU. Subclass `__init__` sets axis attrs before
`super().__init__(training kwargs)`.

## 3. Architecture (delta from GRU §3)

`_LSTMModule`:
`nn.LSTM(input_size=C, hidden_size, num_layers, batch_first=True,
bidirectional=False, dropout=0.0)`. **No `proj_size` (Codex P3):** projections
(LSTMP) are out of scope and must remain the default `0`; a non-zero `proj_size`
re-shapes `h_t`/`output` to the projection size while `c_n` stays cell-sized,
which would break the "GRU-equivalent hidden-state sequence → head" framing. It
is not an axis (never passed), so `nn.LSTM` keeps `proj_size=0`. Forward:
`out, (h_n, c_n) = self.lstm(x)` → `out` is `(b, L, H)`, the hidden-state
sequence; the cell state `c_n` is internal and unused by the head.
`forward_features(x)` returns `out` (same shape/semantics as GRU — time axis 1).
Head (`last_step` = `out[:, -1, :]`, or `attention_pooling_pre_frozen` = learned
temporal-attention pool over time), then a single pooled-feature `nn.Dropout`,
then `Linear(H, 2)`. `softmax(dim=1)` in `predict_proba`, float64. **Every line
of this mirrors GRU §3 with `nn.GRU` → `nn.LSTM` and the 2-tuple state unpack.**

**Causal-by-construction (§4.1):** a unidirectional LSTM is the same strict
left-to-right recurrence as GRU — `output[t]` depends only on inputs `≤ t`.
`bidirectional` is a fixed-`False` axis (reject `True`/`0`/`1`); a bidirectional
LSTM would leak future bars into earlier timesteps.

**Dropout (identical to GRU):** `dropout=0.0` to `nn.LSTM`; the `dropout` axis is
realized as a single pooled-feature `nn.Dropout` — layer-count-independent and
avoids the `num_layers==1` UserWarning.

## 4. Search Axes — identical to GRU + the same 08X-eligibility gate

§7.1 lists `shallow_lstm` but freezes no axis table; the config YAML has no
`shallow_lstm:` block. Same resolution as GRU §4: the body validates its proposed
frozen set for fail-loud + hash-stable **construction only**, and **LSTM is NOT
08X-search-eligible** until its axes are mirrored into
`configs/stages/deep_sequence_exploration.yaml` + `08x_search_space.json` and
sha-stamped before trial 0 (a future, in-scope 08X-harness change). Same proposed
axis values (reviewable default):

| axis | values | validation |
|---|---|---|
| `hidden_size` | 16, 32, 64 | exact int in frozen set |
| `num_layers` | 1, 2 | exact int in frozen set |
| `dropout` | 0.0, 0.05, 0.10, 0.20 | exact float in set |
| `head` | "last_step", "attention_pooling_pre_frozen" | exact str in set |
| `bidirectional` | False (fixed) | exact bool `is False`; reject True/0/1 |

## 5. Training Protocol

Inherited verbatim (Adam, CrossEntropy, chronological-tail early stop,
determinism, `random_state` at fit, same training kwargs + defaults). Module
built under `torch.manual_seed`.

**Determinism note:** confirm `nn.LSTM` is bit-exact under
`torch.use_deterministic_algorithms(True)` on CPU (GRU was, per the GRU Codex
smoke). The bit-exact determinism test is the gate; verify at impl + Codex review.

## 6. Error Modes

Same as GRU §6: `bidirectional` not exactly `False` (reject True/0/1, the §4.1
red line); off-grid / wrong-type `hidden_size` / `num_layers` / `dropout` /
`head`; plus the shared base guards (random_state at fit, X 3-D float finite,
y 1-D int {0,1} both-classes length, predict-before-fit, predict shape-drift).

## 7. Testing (`tests/stages/models/test_lstm.py`, ~41, mirrors `test_gru.py`)

Same shapes as GRU §7b: protocol/proba; determinism + global-state restore; light
causal gate (head × num_layers, perturb future, assert output rows `≤ t`
bit-identical); sequence-only signal (slope sign, identical last input bar →
train acc > 0.6); axis coverage (hidden_size/num_layers/dropout/head); guards
(`bidirectional` ∈ {True,0,1}, off-grid/wrong-type, shared X/y/predict); early-
stop bookkeeping; no-warning gate (dropout=0.10, num_layers=1 → no dropout
UserWarning); dropout-placement assertion (`_model.lstm.dropout == 0.0`,
`_model.dropout.p == dropout`, and `_model.lstm.proj_size == 0` per Codex P3).

## 8. Files

- **Modify** `src/intraday_research/models/deep_sequence/lstm.py` (implement).
- **Create** `tests/stages/models/test_lstm.py`.
- **Modify** `tests/stages/models/test_models_deep_sequence_interface.py` (move
  `ShallowLSTMClassifier` out of `_NOT_YET_IMPLEMENTED_FAMILIES`).
- No other changes.

## 9. Dependencies

`torch` + numpy + `_SequenceTorchClassifier`. After LSTM the base has FOUR
concrete shapes (DLinear / TCN / GRU / LSTM) — the abstraction is validated.
Next pieces: fusion variants (compose DLinear + TCN), losses (§7.5), then the 08X
harness.

## 10. Open Decisions for Review

1. **LSTM axis VALUES** = the same reviewable default as GRU; the 08X-eligibility
   gate (§4) handles the contract risk. User may adjust.
2. Dropout placement, attention-pool head, causal gate, and the
   `_SequenceTorchClassifier` base are all carried from GRU/TCN precedent
   (already Codex-accepted); LSTM introduces no new design choice beyond the
   `nn.GRU`→`nn.LSTM` swap and the 2-tuple state unpack.

## 11. Codex Design Review — absorbed (2026-06-07, gpt-5.5:high, 175s)

- **No P1/P2; the GRU mirror is sound.**
- **P3 (absorbed):** freeze `proj_size=0` explicitly (§3) + assert it in the
  dropout-placement test (§7) — an LSTM-only parameter (GRU has none) that would
  break the hidden-sequence-to-head framing if set.
- **Confirmed sound by Codex:** reading `out` (not `c_n`/`h_n`) is the standard
  window-level head input and matches GRU last-step semantics; unidirectional
  LSTM `out` is strictly causal for num_layers 1 & 2 (same light causal test
  valid); CPU determinism design is correct (LSTM nondeterminism is CUDA/cuDNN-
  scoped only); the dropout mirror (`dropout=0.0` to `nn.LSTM` + pooled
  `nn.Dropout`) is correct. CPU bit-exact smoke remains the impl gate (Codex's
  local smoke was blocked by a Windows runner spawn failure).
