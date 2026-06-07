# N08 #5D-2 — TCN Classifier Body (`deep_sequence/tcn.py`) Design

> Status: design 2026-06-07. Second deep model body of #5 Half 2.
> Reuses the #5D-1 DLinear training pattern; adds causal dilated-conv blocks.
> Tooling: inline design + `humanize:ask-codex` review (Humanize swarm
> unavailable — Claude subagents fail to spawn under the cc-switch model remap).

## 1. Goal & Scope

Implement `TCNClassifier.fit` / `.predict_proba` (currently `NotImplementedError`
scaffolds) as a CPU PyTorch `SequenceClassifier` honoring the §7.3 TCN search
axes of the N08 technical design. The current scaffold has only a truthiness
guard for `causal`; this piece must replace it with exact `causal is True`
validation before adding the substantive body.

**Reuses verbatim the #5D-1 DLinear training contract** (see
`2026-06-07-n08-deep-dlinear-classifier-design.md` §5/§6): Adam + CrossEntropy;
seeded determinism that restores the global torch deterministic-algorithms flag
+ `warn_only` sub-state + RNG stream on exit; a CHRONOLOGICAL-TAIL internal
early-stop split (no random split, no mini-batch shuffle — AGENTS.md §4.1
forbids random internal splits) that gates training duration only (never
candidate selection); `random_state`
required at fit (constructible with defaults); exact-type axis validation;
exposed `actual_epochs_` / `early_stop_reason_` / `internal_val_n_`. The training
loop is duplicated for now (DLinear is the only prior peer); a shared
`_SequenceTorchClassifier` base is the planned refactor at the 3rd model (GRU),
once the abstraction has three concrete shapes to fit.

**Out of scope (unchanged from DLinear):** no data loading, folds, train/val
splitting, HPO, trial ledger, 08F/08O, `run_stage`. Data-agnostic body.

## 2. I/O Contract

Identical to `base.SequenceClassifier` / DLinear: `X` float `(n, window_size, C)`,
`y` int `{0,1}`; `fit(X,y)->self`; `predict_proba(X)->(n,2)` float64 rows
summing to 1; both classes required (single-class fit fails loud).

## 3. Architecture (causal dilated TCN → binary logits)

Forward over `x` `(b, L, C_in)`; internally transpose to `(b, C_in, L)` for
`Conv1d`.

For block `i = 0 .. num_blocks-1` (`in_ch = C_in` if `i==0` else `channels[i-1]`,
`out_ch = channels[i]`, dilation `d = dilation_base ** i`):

1. **Causal dilated conv** (×2 per block): left-pad the time axis by
   `(kernel_size - 1) * d` (PAST only), then `Conv1d(in, out_eff, kernel_size,
   dilation=d, padding=0)` → output length exactly `L`. Left-only padding is
   what makes output `t` depend solely on inputs `≤ t` (§4.1 no-future-leak).
   `out_eff = 2*out_ch` when `gating` else `out_ch`.
2. **Normalization** (`normalization`):
   - `"none"`: identity;
   - `"weight_norm"`: wrap each conv with
     `torch.nn.utils.parametrizations.weight_norm` (the **non-deprecated** API —
     the legacy `torch.nn.utils.weight_norm` emits a DeprecationWarning that
     `pytest.ini`'s `filterwarnings` promotes to a test error);
   - `"layer_norm"`: `LayerNorm` over the channel dim per timestep (transpose
     `(b,C,L)`→`(b,L,C)`, norm, transpose back) — causal-safe (per-timestep, no
     time mixing).
3. **Activation**: ReLU, OR if `gating`: split the `2*out_ch` conv output into
   `(a, b)` and use `tanh(a) * sigmoid(b)` (gated activation) → `out_ch`.
4. **Dropout** `dropout`.
5. **Residual** (always on): `res = x` if `in_ch == out_ch` else a `1×1`
   `Conv1d(in_ch, out_ch, 1)`; block output `= activation_chain(x) + res`.

After `num_blocks` blocks: `(b, channels[-1], L)`. **Head** (`head`):

- `"last_step"`: take the last timestep `out[:, :, -1]` → `Linear(channels[-1], 2)`.
- `"attention_pooling_pre_frozen"`: a learned temporal-attention pool — scores
  `Linear(channels[-1], 1)` over time → `softmax` over `L` → weighted sum →
  `(b, channels[-1])` → `Linear(channels[-1], 2)`. The pooling is a fixed part of
  the frozen architecture (no extra tunable axes); "pre_frozen" denotes that the
  head shape is frozen, not that weights are externally pinned. Attention is over
  the within-window time axis only (no cross-window/day mixing). **Open for
  Codex/user review** (decision §10.1).

`softmax(logits, dim=1)` in `predict_proba`, returned float64.

## 4. Search Axes (§7.3) — semantics + validation

| axis | values | validation |
|---|---|---|
| `num_blocks` | 2, 3, 4 | exact int; AND `== len(channels)` |
| `channels` | `(16,16)`, `(32,32)`, `(32,32,32)`, `(64,32,16)` | normalize list→tuple; exact membership in the frozen set; each element exact positive int (reject bool); `len == num_blocks` |
| `kernel_size` | 2, 3, 5 | exact int in set |
| `dilation_base` | 2 (fixed) | exact int `== 2` |
| `dropout` | 0.0, 0.05, 0.10, 0.20 | exact float in set |
| `residual` | True (fixed) | exact bool `== True` |
| `gating` | False, True | exact bool |
| `normalization` | "none","weight_norm","layer_norm" | exact str in set |
| `causal` | True (fixed) | exact bool `is True`; reject `False`, `1`, `"true"` |
| `head` | "last_step","attention_pooling_pre_frozen" | exact str in set |

All search axes use the same frozen search-space discipline as DLinear:
off-grid values fail loud in the model body, not only in the future 08X harness.
This keeps config hashes unambiguous and prevents silent exploratory architecture
drift. `random_state` is validated at `fit` (as DLinear).

## 5. Training Protocol

Verbatim reuse of DLinear §5 (Adam, CrossEntropy, chronological-tail internal
early-stop split with no mini-batch shuffle [AGENTS.md §4.1 forbids random
internal splits], determinism with full global-state save/restore incl. RNG
stream + `warn_only`, `random_state` required at fit, training kwargs `max_epochs` /
`learning_rate` / `batch_size` / `early_stopping_patience` /
`early_stopping_fraction` / `weight_decay` with the same defaults + exact-type
ranges). The module is constructed under `torch.manual_seed(random_state)` for
reproducible init; conv/weight-norm init draws from the seeded stream.

## 6. Error Modes (fail-loud)

DLinear §6 modes 1–8 (axis exact-type/frozen-set, `random_state` at fit, training
kwargs, X 3-D float finite, predict-before-fit, y 1-D int {0,1} + both classes +
length, predict shape-drift) **plus** TCN-specific:

| # | condition | exception |
|---|---|---|
| T1 | `causal` not exactly `True` (including `1`) | `ValueError` (at construction; replace scaffold truthiness check) |
| T2 | `num_blocks != len(channels)` | `ValueError` |
| T3 | `channels` not a tuple/list of exact positive ints, or not one of the §7.3 frozen tuples | `ValueError` |
| T4 | `dilation_base != 2` / `residual != True` (fixed axes) | `ValueError` |

## 7. Testing (`tests/stages/models/test_tcn.py`)

Reuses the DLinear test shapes plus the TCN-critical leakage gate.

1. **Protocol conformance / proba contract** (as DLinear).
2. **Determinism + global-state restore** (bit-exact same seed; restores
   deterministic flag + warn_only + RNG stream — the DLinear Codex P2 lesson).
3. **CAUSALITY (§4.1 red line — the key TCN gate):** fit a model; build two
   inputs identical up to time `t` and arbitrarily different at times `> t`;
   run the fitted module's private conv-stack feature path on both and assert the
   output column at time `t` is bit-identical. The implementation should expose a
   narrow `_forward_features` helper for this test; do not rely on indirect
   `predict_proba` comparisons. A symmetric/future-aware padding bug fails this.
4. **Sequence-only signal** (slope sign with identical last bar) → train acc
   > 0.6, proving the conv stack uses temporal structure.
5. **Axis coverage:** parametrize `kernel_size`∈{2,3,5}, `dropout`∈{0,.05,.10,.20},
   `gating`∈{F,T}, `normalization`∈{none,weight_norm,layer_norm},
   `head`∈{last_step,attention_pooling_pre_frozen}, and `(num_blocks,channels)`
   ∈ {(2,(16,16)),(3,(32,32,32)),(3,(64,32,16))}; each fits + shape-conforms.
6. **Guards:** all §6 modes incl. `causal=False` and `causal=1` (construction),
   `num_blocks != len(channels)`, non-int/negative/off-grid `channels`,
   `dilation_base != 2`, bad scalar axis exact-type, plus the shared
   X/y/predict guards.
7. **Early-stop bookkeeping** (as DLinear).

`weight_norm` parametrization must NOT emit a DeprecationWarning (pytest.ini
trap) — covered implicitly by the `normalization="weight_norm"` axis test
running clean under the Warning→error filter.

Estimated ~40–50 tests.

## 8. Files

- **Modify** `src/intraday_research/models/deep_sequence/tcn.py` (fill scaffolds;
  replace the truthiness-only `causal` guard with exact-type validation; add
  training kwargs).
- **Create** `tests/stages/models/test_tcn.py`.
- **Modify** `tests/stages/models/test_models_deep_sequence_interface.py` to move
  `TCNClassifier` out of `_NOT_YET_IMPLEMENTED_FAMILIES` (now implemented).
- No change to `base.py`, `dlinear.py`, `folds.py`, or any data/contract/
  notebook/config file.

## 9. Dependencies

`torch` 2.12.0+cpu + numpy. `base.SequenceClassifier`. Downstream: the 08X
harness instantiates per-trial; the planned `_SequenceTorchClassifier` refactor
(at GRU) will hoist the shared training loop out of DLinear + TCN.

## 10. Open Decisions for Review

1. **`attention_pooling_pre_frozen` = learned temporal-attention pool** (scores →
   softmax over time → weighted sum), architecture frozen, no extra axes.
   Confirm this reading of "pre_frozen" or specify the intended fixed pooling.
2. **Training loop duplicated from DLinear** (shared base deferred to GRU, the
   3rd model, per rule-of-three). Confirm the defer.
