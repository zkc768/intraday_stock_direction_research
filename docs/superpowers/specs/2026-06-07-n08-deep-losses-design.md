# N08 #5D-6 — Loss Functions (`deep_sequence/losses.py`) Design

> Status: design 2026-06-07. Sixth deep piece of #5 Half 2 — the five §7.5
> loss variants. Tooling: inline design + `humanize:ask-codex` review.
> Coexistence: model package co-developed with a parallel Codex session; check
> `git status` clean before editing.

## 1. Goal & Scope

Implement the five `NotImplementedError` loss scaffolds in `losses.py` per the
existing fixed signatures: each takes pre-softmax `logits` `(n, 2)` + binary
`targets` `(n,)` and returns a **scalar Python float** (the mean loss):

- `cross_entropy_loss(logits, targets, *, weight=None)`
- `weighted_cross_entropy_train_prior_loss(logits, targets, *, train_class_prior)`
- `focal_loss(logits, targets, *, gamma=2.0, alpha=None)` (Lin et al. 2017)
- `class_balanced_loss_effective_number(logits, targets, *, samples_per_class, beta=0.9999)` (Cui et al. 2019)
- `balanced_softmax_loss(logits, targets, *, train_class_prior)` (Ren et al. 2020)

**numpy→float DIAGNOSTIC losses (the scaffold's signature, and §7.5: "loss
variants are diagnostic unless train-inner selection was predeclared").** These
compute a loss VALUE (for the trial ledger / diagnostics), NOT a differentiable
torch training objective. Differentiable training-loss selection (wiring a chosen
loss into the models' `_train`) is a SEPARATE future concern (08X harness), out
of scope here. Flagged for Codex (§7.1).

**Out of scope:** no model-training wiring, no torch autograd, no data/folds/HPO/
08F/08O; no frozen-config edit. `train_class_prior` / `samples_per_class` MUST be
computed by the caller on train-inner-fit rows only (AGENTS §4.1) — these bodies
just consume the passed-in values.

## 2. Formulas (numpy, numerically stable)

Shared helper `_log_softmax(logits)`: `logits - logsumexp(logits, axis=1)` with
the max-subtraction trick (`m = logits.max(axis=1, keepdims=True)`;
`logsumexp = m + log(sum(exp(logits - m)))`). `log_p_t = log_softmax(logits)[i,
targets[i]]` is finite for finite logits (no `log(0)`).

| loss | formula (per-sample, then reduce) |
|---|---|
| `cross_entropy_loss` | `ce_i = -log_p_t`. No weight → `mean(ce_i)`; with class `weight` (2,) → weighted mean `Σ w[y_i]·ce_i / Σ w[y_i]` (PyTorch `reduction="mean"` convention). |
| `weighted_cross_entropy_train_prior_loss` | inverse-prior class weight `w[c] = 1 / prior[c]`, then the weighted CE above. (Weighted mean is scale-invariant, so no extra normalization.) |
| `focal_loss` | `p_t = exp(log_p_t)`; `fl_i = α_t · (1 - p_t)^gamma · (-log_p_t)`; `α_t = alpha if y_i==1 else (1-alpha)` when `alpha` given, else `1`; reduce `mean(fl_i)`. |
| `class_balanced_loss_effective_number` | effective number `E_c = (1 - beta^{n_c}) / (1 - beta)`; class weight `w[c] = 1 / E_c`; then the weighted CE above. (Scale-invariant → Cui's sum-to-num_classes normalization is value-irrelevant; skipped.) **Compute `E_c` stably (Codex P2):** `E_c = -expm1(n_c · log1p(beta - 1)) / (1 - beta)` to avoid cancellation as `beta → 1` (default `0.9999`). |
| `balanced_softmax_loss` | adjust logits by the log-prior: `CE(logits + log(prior), targets)` (unweighted). |

**Reduce-to-CE sanity (drives the tests):** `focal_loss(gamma=0, alpha=None) ==
cross_entropy_loss`; `balanced_softmax_loss(uniform prior) == cross_entropy_loss`
(a constant logit shift cancels in softmax CE); `weighted_…(balanced prior) ==
cross_entropy_loss` and `class_balanced_…(equal samples) == cross_entropy_loss`
(equal class weights → weighted mean = mean).

## 3. Validation (fail-loud)

Shared `_validate_logits_targets(logits, targets)`:
- `logits`: 2-D float ndarray `(n, 2)`, **`n >= 1`** (reject empty — Codex P1; an
  empty batch is a meaningless diagnostic, not a 0-loss), finite; else `ValueError`.
- `targets`: 1-D **integer (non-bool)** ndarray, `len == n`, values ⊆ `{0, 1}`;
  else `ValueError`. (bool dtype rejected — Codex note.)

Per-param (exact-type incl. bool rejection — Codex note):
- `weight` (if not None): float ndarray shape `(2,)`, finite, all `> 0`.
- `train_class_prior`: 2-tuple of exact floats, each in `(0, 1)`, summing to 1
  within a tolerance; else `ValueError` (guards `1/0`).
- `gamma`: exact float `>= 0`. `alpha` (if not None): exact float in `[0, 1]`.
- `samples_per_class`: 2-tuple of exact ints (reject bool), each `>= 1`.
  `beta`: exact float in `(0, 1)`.

## 4. Files
- **Modify** `src/intraday_research/models/deep_sequence/losses.py` (fill the 5).
- **Create** `tests/stages/models/test_losses.py`.
- No change to `__init__.py` (losses not in `__all__`/lazy map — they are module
  functions, imported as `from ...deep_sequence import losses`), configs,
  contracts, notebooks, data. No model/base change (diagnostic-only).

## 5. Testing (`tests/stages/models/test_losses.py`)
- **Known-value** checks per loss on a tiny hand-computed `(logits, targets)`
  (e.g. 2 samples) — assert the returned float matches the closed-form value.
- **Reduce-to-CE** equivalences (§2): focal γ=0, balanced-softmax uniform prior,
  weighted balanced prior, CB equal samples all `== cross_entropy_loss` (allclose).
- **Direction** checks: focal with γ>0 ≤ CE on confident-correct samples
  (down-weights easy examples); weighted/CB up-weight the minority class
  (loss rises when the minority class is mispredicted vs uniform).
- **Stability**: large-magnitude logits (e.g. ±50) do not overflow/NaN.
- **Guards**: non-(n,2) logits, non-finite logits, targets out of {0,1}, prior
  not summing to 1 / containing 0, negative gamma, beta out of (0,1),
  samples_per_class non-positive, weight wrong shape — each `ValueError`.
- **Import**: the existing `test_losses_module_exports_five_named_losses` stays
  green (names unchanged).

## 6. Dependencies
numpy only (no torch — diagnostic numpy losses). Downstream: the 08X harness
reads these for diagnostic loss columns; a future training-loss-selection piece
(if predeclared per §7.5) would wire differentiable torch equivalents into the
models' `_train` — NOT part of this piece.

## 7. Open Decisions for Review
1. **numpy→float DIAGNOSTIC vs torch differentiable** — the scaffold signature is
   `np.ndarray → float`, so I implement diagnostic value-losses. Confirm this is
   the intended role (and that training-loss selection is a separate future
   piece), vs. wanting torch/differentiable losses now.
2. **Convention choices:** focal α as `α_t = alpha if y==1 else 1-alpha`
   (Lin et al. binary convention); CB/weighted use scale-invariant weighted-mean
   (skip sum-to-num_classes normalization since it doesn't change the value);
   balanced-softmax = CE on log-prior-shifted logits. Confirm these match intent.
3. **Reductions:** weighted losses use `Σ w·ce / Σ w` (PyTorch mean convention);
   focal uses plain `mean`. Confirm (vs. dividing focal by `Σ α_t`).
