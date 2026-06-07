# N08 #5D-5 — Fusion Variants (`deep_sequence/fusion.py`) Design

> Status: design 2026-06-07. Fifth deep piece of #5 Half 2 — the four §7.4 fusion
> variants that compose the now-implemented DLinear + TCN bodies. Tooling: inline
> design + `humanize:ask-codex` review.
> Coexistence note: the model package is co-developed with a parallel Codex
> session. `__init__.py` already lazily exports the four fusion classes; the
> `ms_dlinear_tcn` family is already 08X-search-eligible
> (`SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES`, contract commit `65e98eb`).

## 1. Goal & Scope

Implement the four `NotImplementedError` fusion scaffolds in `fusion.py` as
`SequenceClassifier`s that compose the existing `DLinearClassifier` +
`TCNClassifier`:

- `LateAverageProbabilitiesFusion` — average post-softmax probabilities.
- `DLinearLogitsPlusTCNLogitsFusion` — sum pre-softmax logits.
- `SmallFusionMLP` — small MLP head over both models' logits.
- `DLinearTrendPlusTCNResidualFusion` — DLinear trend branch + TCN residual
  branch (architectural).

§7.4 gives ONLY the four names (the design doc specifies no per-variant
mechanics) plus the gate that fusion train-inner `lcb_delta_macro_f1` vs the
better of {`dlinear_only`, `tcn_only`} MUST exceed
`FUSION_MIN_LCB_ADVANTAGE_OVER_COMPONENTS` (0.003) — enforced by the future 08F
orchestrator, NOT in these bodies. So the per-variant mechanics below are
spec-introduced (like the GRU/LSTM axes) and the ambiguous one
(`dlinear_trend_plus_tcn_residual`) is flagged for review (§11).

**Out of scope (unchanged):** data-agnostic bodies — no data, folds, train/val
splitting, HPO, ledger, 08F/08O, `run_stage`; no frozen-config edit. The 08F
"fusion must beat components" gate is the orchestrator's job, not these bodies'.

## 2. Shape — composition wrappers, NOT `_SequenceTorchClassifier` subclasses

Unlike the four single models, three fusion variants hold **two sub-models** and
combine their outputs; they do not have one torch module / one `_train` loop, so
they do NOT subclass `_SequenceTorchClassifier`. They implement the
`SequenceClassifier` protocol (`fit`/`predict_proba`) directly, delegating each
sub-model's construction/validation/determinism to `DLinearClassifier` /
`TCNClassifier`. The exception is `DLinearTrendPlusTCNResidualFusion`, which (per
the proposed §4.4 interpretation) is ONE joint module trained on `X` and so CAN
subclass `_SequenceTorchClassifier`.

**Shared `_FusionBase` (Codex P2).** To avoid three subtly-different fail-loud
paths, a non-public `_FusionBase` (evolving the scaffold's `_BaseFusionScaffold`)
centralizes the shared wrapper mechanics: `dlinear_config`/`tcn_config` dict
validation + defensive copy, nested-`random_state` rejection, `random_state`
injection, the fitted-check, a stable row-wise softmax helper, and the proba
row-sum assertion. The three wrappers compose it; the joint-module variant uses
its config/seed helpers while subclassing `_SequenceTorchClassifier` for training.

## 3. Shared fusion mechanics

### 3.1 Config forwarding + seeding
Each fusion forwards `dlinear_config` / `tcn_config` (dicts) as kwargs to the
sub-classifiers: `DLinearClassifier(**dlinear_config)` /
`TCNClassifier(**tcn_config)`. The sub-classifiers do their own exact-type frozen
axis validation, so fusion does NOT re-validate component axes (delegation).

`random_state` is the single fusion seed. To keep the whole fusion reproducible
from one seed, the fusion INJECTS its `random_state` into each sub-model
(`DLinearClassifier(random_state=self.random_state, **dlinear_config)`), and
rejects a `random_state` key inside `dlinear_config`/`tcn_config` (ambiguous
double-seed → fail loud). `random_state` is required at `fit` (int; reject
None/bool), matching the single-model contract. (Open: same-seed for both
sub-models is fine — their RNG draws differ by architecture; §11.)

### 3.2 `_predict_logits` helper on the base (additive, behavior-preserving)
Logit fusion + the MLP head need each sub-model's pre-softmax logits, not just
`predict_proba`. Add a small `_predict_logits(X) -> (n, 2) float64` to
`_SequenceTorchClassifier`: identical to `predict_proba` but returns the raw head
output WITHOUT softmax (fitted-check + shape-drift check + eval + no_grad). This
is a NEW method (no existing behavior changes), so the DLinear/TCN/GRU/LSTM test
suites stay green (regression guard). A direct unit test asserts
`softmax(_predict_logits(X)) == predict_proba(X)` for a fitted model. (Codex
confirmed: `softmax(logit_d + logit_t) == softmax(log p_d + log p_t)` for ANY
class count, so `_predict_logits` is not strictly required — but it is cleaner
and avoids `log(0)`/underflow, so it is the chosen path.)

### 3.3 Stable softmax (Codex P3)
The `_FusionBase` softmax helper subtracts the row max before `exp`
(`z = z - z.max(axis=1, keepdims=True)`), returns float64, and the proba contract
test asserts finite `(n, 2)` rows summing to 1 within `1e-6`. Used by logit fusion
(§4.2) and the MLP output (§4.3).

## 4. The four variants

### 4.1 `LateAverageProbabilitiesFusion`
- `fit`: fit `self._dlinear` + `self._tcn` on `(X, y)` (each seeded by
  `random_state`).
- `predict_proba`: `(dlinear.predict_proba(X) + tcn.predict_proba(X)) / 2.0` —
  mean of two distributions is a distribution (rows sum to 1, float64).
- No fusion-level torch training; determinism inherited from the seeded
  sub-models.

### 4.2 `DLinearLogitsPlusTCNLogitsFusion`
- `fit`: same as §4.1.
- `predict_proba`:
  `softmax(dlinear._predict_logits(X) + tcn._predict_logits(X), axis=1)` → float64.
  (Summing the two heads' pre-softmax logits, then one softmax.)

### 4.3 `SmallFusionMLP`
- Extra axes (scaffold): `mlp_hidden_size` (default 16), `mlp_dropout`
  (default 0.0).
- `fit` (**chronological-OOF default — Codex P2**): split `(X, y)` chronologically
  into a fit-prefix + a tail (the same `_early_stop_split`-style chronological tail
  discipline, no random split — AGENTS §4.1); fit `self._dlinear` + `self._tcn` on
  the PREFIX; compute their logits on the TAIL (out-of-fold — the sub-models did
  not see those rows); build `feat = concat([dlinear._predict_logits(tail),
  tcn._predict_logits(tail)], axis=1)` → `(n_tail, 4)`; train a small MLP
  `Linear(4, mlp_hidden_size) → ReLU → Dropout(mlp_dropout) →
  Linear(mlp_hidden_size, 2)` on `(feat, tail_y)` with the SAME deterministic Adam
  + CrossEntropy recipe + determinism global-state save/restore (a small bespoke
  trainer, since the MLP trains on 2-D features). The prefix-fit sub-models +
  trained MLP are kept for prediction (consistent train/predict pipeline; the base
  models intentionally use prefix data only so the MLP never sees in-sample
  sub-model logits).
- `predict_proba`: `stable_softmax(mlp(concat([dlinear._predict_logits(X),
  tcn._predict_logits(X)], axis=1)))` → float64.
- **Why OOF (Codex P2):** train-on-train stacking (MLP on logits from sub-models
  fit on the same rows) is not an AGENTS §4.1 holdout leak, but it ships an
  optimistic/overfit-prone stacker; the 08F LCB gate cannot repair body-level
  stacker optimism. So the body default is chronological-OOF. (A named
  `train_on_train` diagnostic mode may be added later if needed; not in scope for
  the default body.) Fallback when the tail cannot spare a both-class prefix +
  non-empty tail: documented in §7 (fail loud rather than silently degrade).

### 4.4 `DLinearTrendPlusTCNResidualFusion` — proposed architectural reading (§11 key decision)
The name (and DLinear's built-in trend/seasonal decomposition) suggest: DLinear
models the TREND, TCN models the RESIDUAL. Proposed JOINT module (subclasses
`_SequenceTorchClassifier`, trained on `X`):
1. Decompose `X`: `trend = causal_moving_average(X)` (a **trailing / left-pad**
   moving average — kernel from `dlinear_config`), `residual = X - trend`.
   **Causal MA is required (Codex P1):** standalone DLinear uses a CENTERED
   edge-replicate MA (pads both sides), so its `trend[t]`/`residual[t]` depend on
   FUTURE bars within the window — fine for DLinear (a window-level predictor that
   collapses the whole window), but it would make the residual non-causal. Here
   the residual feeds a per-timestep causal TCN branch, so the decomposition MUST
   be causal (left-pad by `kernel-1`, average over `[t-kernel+1 .. t]`), making
   `residual[t]` depend only on inputs `≤ t`. The residual branch then passes the
   §4.1 perturb-future causal test end-to-end. (The trend branch itself is a
   window-level temporal-linear collapse, so it needs no per-timestep causality,
   but using the causal MA keeps the residual clean.)
2. **Trend branch:** DLinear's per-component temporal-linear map on `trend` →
   `trend_logits` (2).
3. **Residual branch:** TCN causal dilated-conv stack on `residual` (axes from
   `tcn_config`; causal, §4.1-safe) → head → `residual_logits` (2).
4. `logits = trend_logits + residual_logits` → softmax. Trained as ONE module
   (one optimizer), so the two branches co-adapt (this is what distinguishes it
   from §4.1/§4.2 late fusion of independently-trained models).
This reuses the `_DLinearModule` trend path + `_TCNModule` conv path concepts in
a new joint module. It is the only variant that is a single-module
`_SequenceTorchClassifier` subclass. **Alternative readings** (sequential
residual-boosting; pure output-level combination) are possible — Codex/user to
confirm the intended semantics before implementation (§11.1).

## 5. Determinism
- §4.1/§4.2 (no fusion-level torch): determinism is fully inherited from the
  seeded sub-models (each `fit` already does the global-state save/restore);
  fusion only does numpy averaging / a single softmax.
- §4.3/§4.4 (fusion-level torch training): wrap the MLP / joint-module training
  in the same `use_deterministic_algorithms(True)` + RNG-stream + warn_only
  save/restore as the base. §4.4 inherits it from `_SequenceTorchClassifier`;
  §4.3 reuses the pattern in its bespoke MLP trainer.
- Bit-exact same-`random_state` reproducibility is the gate for all four.

## 6. Search axes / config validation / 08X-eligibility
- `dlinear_config`/`tcn_config` axis validation is delegated to the sub-models
  (fusion forwards kwargs). Fusion validates only: `random_state` at fit; no
  `random_state` inside the sub-configs; (`SmallFusionMLP`) `mlp_hidden_size`
  exact positive int + `mlp_dropout` exact float in `[0,1)`.
- **08X-eligibility:** the fusion family `ms_dlinear_tcn` is ALREADY
  08X-search-eligible (`SEARCH_ELIGIBLE_ARCHITECTURE_FAMILIES`), because its
  components' axes are frozen in config. So fusion needs NO GRU/LSTM-style
  withholding. BUT `SmallFusionMLP`'s `mlp_hidden_size`/`mlp_dropout` are
  spec-introduced (not in the frozen config). Proposed: freeze small sets
  (`mlp_hidden_size ∈ {8,16,32}`, `mlp_dropout ∈ {0.0,0.05,0.10}`) body-side for
  hash-stable construction, and document that the MLP sub-axes need a config
  mirror before an 08X run varies them (parallels the GRU/LSTM gate). Flagged
  §11.

## 7. Error modes (fail-loud)
- `random_state` not int at fit → ValueError (each variant).
- `random_state` present inside `dlinear_config`/`tcn_config` → ValueError
  (ambiguous double-seed).
- Sub-model construction errors (bad component axis) propagate from
  DLinear/TCN unchanged (delegation).
- X/y validation: delegated to the sub-models' `fit` (§4.1/§4.2) or done up front
  for the fusion-trained variants; predict-before-fit → RuntimeError;
  `SmallFusionMLP` off-grid `mlp_*` → ValueError.

## 8. Testing (`tests/stages/models/test_fusion.py`, mirror prior model tests)
Per variant: protocol conformance + proba contract (shape (n,2), float64, rows
sum 1); determinism (bit-exact same seed); a "fusion ≠ either component"
sanity (fused proba differs from each sub-model's, proving combination happens);
guards (random_state at fit, double-seed rejection, predict-before-fit,
predict shape-drift via the sub-models). Plus: `_predict_logits` base test
(`softmax(_predict_logits) == predict_proba`); §4.4 causal-leak test (the
residual TCN branch must stay causal — reuse the TCN `_forward_features`-style
perturb-future assertion); §4.3 `mlp_*` axis coverage + off-grid rejection +
`nn.GRU`-free determinism. Regression: DLinear/TCN/GRU/LSTM suites stay green
after the additive `_predict_logits` base change.

## 9. Files
- **Modify** `src/intraday_research/models/deep_sequence/_torch_base.py` (add
  `_predict_logits`).
- **Modify** `src/intraday_research/models/deep_sequence/fusion.py` (implement 4).
- **Create** `tests/stages/models/test_fusion.py`.
- **Modify** `tests/stages/models/test_models_deep_sequence_interface.py` (move
  the 4 fusion families out of `_NOT_YET_IMPLEMENTED_FAMILIES`).
- No change to `__init__.py` (already lazily exports the 4), configs, contracts,
  notebooks, data. (Coordinate: confirm Codex is idle on the model package
  before editing the interface test / `_torch_base.py`.)

## 10. Staged implementation (proposed)
Given the complexity gap, implement + commit in slices, each three-gates-green +
Codex impl-reviewed:
- **Slice 1:** `_predict_logits` base helper + `LateAverageProbabilitiesFusion` +
  `DLinearLogitsPlusTCNLogitsFusion` (lowest risk — pure composition).
- **Slice 2:** `SmallFusionMLP` (bespoke MLP trainer + stacking-leakage decision).
- **Slice 3:** `DLinearTrendPlusTCNResidualFusion` (joint module — after §11.1
  semantics are confirmed).

## 11. Open Decisions for Review
1. ~~`dlinear_trend_plus_tcn_residual` semantics~~ — **resolved**: Codex confirmed
   the joint co-trained trend+residual module is the most defensible reading
   (sequential boosting = different algorithm; output-level = already covered by
   §4.1/§4.2). Adds the **causal-MA requirement** (§4.4, Codex P1).
2. ~~`SmallFusionMLP` stacking leakage~~ — **resolved**: chronological-OOF is the
   body default (§4.3, Codex P2); train-on-train only as a future named diagnostic.
3. **`SmallFusionMLP` `mlp_*` axes** (§6): proposed freeze body-side
   (`mlp_hidden_size ∈ {8,16,32}`, `mlp_dropout ∈ {0.0,0.05,0.10}`) + document the
   config-mirror gate (parallel to GRU/LSTM). Reviewable default; user may adjust.
4. ~~`_predict_logits` on the base~~ — **resolved**: Codex confirmed cleaner/safer
   (avoids `log(0)`/underflow); the math identity holds for any class count.
5. ~~Single `random_state` injected into both sub-models~~ — **resolved**: Codex
   confirmed acceptable; rejecting a nested `random_state` is the right fail-loud.

## 12. Codex Design Review — absorbed (2026-06-07, gpt-5.5:high, 168s)
- **P1 (residual branch not causal):** §4.4 now mandates a **causal trailing
  moving average** for the decomposition so `residual[t]` depends only on inputs
  `≤ t`; the residual TCN branch passes the §4.1 perturb-future test end-to-end.
- **P2 (stacker optimism):** §4.3 default is now **chronological-OOF** (base on
  prefix, MLP on tail OOF logits, predict via prefix-fit base + MLP).
- **P2 (shared fail-loud paths):** §2 adds a non-public **`_FusionBase`**
  centralizing config validation, nested-seed rejection, fitted-check, and stable
  softmax for the three wrappers.
- **P3 (stable softmax):** §3.3 specifies max-subtracted softmax + the row-sum
  contract test.
- **Confirmed sound:** joint trend+residual reading; `_predict_logits` (math
  identity any class count); single-seed injection + nested-seed rejection;
  composition-wrapper determinism; averaging two distributions within `1e-6`; the
  staged 3-slice plan (base/logit/proba → MLP → joint trend/residual).
