# N08 #5D-1 — DLinear Classifier Body (`deep_sequence/dlinear.py`) Design

> Status: design 2026-06-07. First piece of #5 Half 2 (deep model bodies).
> Establishes the shared PyTorch `SequenceClassifier` training pattern that
> TCN / GRU / LSTM bodies reuse.
> Tooling note: produced inline (the Humanize gen-idea/gen-plan/rlcr swarm is
> unavailable — Claude subagents fail to spawn under the active cc-switch model
> remap); independent review is Codex via `humanize:ask-codex`, as for #5C.

## 1. Goal & Scope

Implement `DLinearClassifier.fit` / `.predict_proba` (currently
`NotImplementedError` scaffolds) as a CPU PyTorch model honoring the
`SequenceClassifier` Protocol (`base.py`) and the §7.2 DLinear search axes of
`docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md`.

**In scope:** the model body only — architecture, deterministic training loop,
`fit(X, y) -> self`, `predict_proba(X) -> (n, 2)`, fail-loud input validation.

**Out of scope (later #5 Half 2 pieces, explicitly NOT here):** data loading,
fold construction (`folds.py`), train-inner/validation splitting, HPO, trial
ledger, search-space JSON, 08F/08O, `run_stage`. This body never reads data
from disk, never sees official validation or holdout, and performs no
cross-day / cross-ticker / label-horizon reasoning — those are upstream window
invariants (#5C-1/#5C-5) already baked into `X`. The class only consumes the
`(X, y)` arrays the caller hands it.

## 2. Input / Output Contract (from `base.py`)

| Arg | dtype/shape | constraint |
|---|---|---|
| `X` (fit, predict_proba) | float, `(n, window_size=20, n_features)` | finite; `window_size` is the active Stage-0 freeze (20) but the body reads it from `X.shape[1]` and does not hard-code it |
| `y` (fit) | int, `(n,)` | values in `{0, 1}` |

`fit` returns `self`. `predict_proba(X)` returns `float64 (n, 2)`, rows sum to
1, column `j` is `P(class=j)`. Both classes must be present in `y` (a single-
class fit is a fail-loud error — mirrors the #5A `LastStepLightGBMControl`
canonical-column discipline; the orchestrator handles degenerate folds, not the
model).

## 3. Architecture (DLinear → binary logits)

Forward path over a batch `x` of shape `(b, L, C)` (`L=window_size`, `C=n_features`):

1. **Input projection** (`input_projection`):
   - `"none"`: `x' = x` (`C' = C`).
   - `"linear_bottleneck"`: `x' = x @ W_proj` with `W_proj: (C, C')`,
     `C' = max(1, C // 2)` (a per-feature linear bottleneck applied identically
     at every timestep; no bias, no time mixing — keeps the projection a pure
     channel-reducer). `C'` is recorded on the fitted model.
2. **Series decomposition** (the DLinear core):
   - `trend = moving_average(x', kernel=moving_avg_kernel)` — a 1-D average pool
     over the time axis with `padding` that replicates the edge values so
     `trend` keeps length `L` (standard DLinear `series_decomp`); `kernel` must
     be odd (the §7.2 set `{3,5,7,11}` is all odd) for symmetric padding.
   - `seasonal = x' - trend`. Both `(b, L, C')`.
3. **Per-component temporal linear** — for each component `s ∈ {trend, seasonal}`
   map the time axis `L → 1`, collapsing each channel's window to one scalar:
   - `individual_channels=False`: one shared `Linear(L, 1)` applied to every
     channel (weights shared across channels).
   - `individual_channels=True`: a separate `Linear(L, 1)` per channel (a
     `(C', L)` weight bank); higher capacity, `C'`× the parameters.
   - trend and seasonal use **separate** linear banks. Output per component:
     `(b, C')`. Sum: `feat = dropout(trend_feat) + dropout(seasonal_feat)`,
     dropout rate = `seasonal_trend_dropout` (applied to each component before
     the sum; identity when rate `== 0.0`).
4. **Classification head** (`linear_head`) → 2 logits:
   - `"shared"`: `Linear(C', 2)` over the summed per-channel features `(b, C')`.
   - `"per_channel"`: a per-channel `Linear(1, 2)` bank giving `(b, C', 2)`,
     then **mean over channels** → `(b, 2)` (each channel votes, votes averaged;
     keeps the head per-channel without exploding parameters).
5. Output: raw logits `(b, 2)`. `predict_proba` applies `softmax(dim=1)` and
   returns `float64`.

This is intentionally small (parameter count ≈ `O(C' · L)` for the temporal
banks plus a tiny head), matching the §7.2 "decomposition and linear sequence
bias" role and the CPU budget.

## 4. Search Axes (§7.2) — exact semantics

| axis | values | controls |
|---|---|---|
| `moving_avg_kernel` | 3, 5, 7, 11 (odd) | decomposition trend smoothness |
| `individual_channels` | False, True | shared vs per-channel temporal `Linear(L,1)` banks |
| `linear_head` | "shared", "per_channel" | final head over summed features |
| `seasonal_trend_dropout` | 0.0, 0.05, 0.10 | dropout on trend & seasonal feats before sum |
| `input_projection` | "none", "linear_bottleneck" | optional `C → C//2` channel reducer |

These are validated fail-loud in `__init__` against the frozen value sets
**by exact type, not membership** (a frozen-search-space / config-hash
discipline requirement — Codex P1). Specifically:
- `individual_channels`: `type(x) is bool` (reject `1`/`0`, which `== True/False`);
- `linear_head`, `input_projection`: `type(x) is str` AND in the frozen set;
- `moving_avg_kernel`: `type(x) is int` (reject `bool`) AND in `{3,5,7,11}`;
- `seasonal_trend_dropout`: a real float (reject `bool`) AND in `{0.0,0.05,0.10}`.

Without exact-type checks, `individual_channels=1`, `seasonal_trend_dropout=False`,
or `moving_avg_kernel=True` would silently alias a different frozen config and
corrupt the future config hash. `random_state` (existing scaffold kwarg) seeds
all RNG.

## 5. Training Protocol (proposed; §7.2 leaves it open, §8.3/§11 constrain it)

Added as `__init__` kwargs (allowed by `base.py`: "implementations may add
family-specific keyword arguments"), with budget-tier-friendly defaults:

| kwarg | default | meaning |
|---|---|---|
| `max_epochs` | 50 | hard epoch cap (§8.3 `max_epochs`); harness lowers it per §11 tier |
| `learning_rate` | 1e-3 | Adam lr |
| `batch_size` | 256 | minibatch; full-batch when `n < batch_size` |
| `early_stopping_patience` | 5 | epochs without val-loss improvement before stop |
| `early_stopping_fraction` | 0.15 | seeded internal split of `X` used ONLY for early stopping |
| `weight_decay` | 0.0 | Adam L2 |

- **Optimizer/loss:** Adam + `CrossEntropyLoss` (the §7.5 default;
  weighted/focal/etc. are a separate later loss piece, not this body).
- **Internal early-stop split:** `fit` carves a seeded random
  `early_stopping_fraction` slice of `(X, y)` for early-stop monitoring **only**.
  This split lives entirely inside the caller's train-inner-fit rows, never
  touches the harness's train-inner-validation or any official partition, and
  gates training **duration only** — not candidate selection (08F does that).
  If `n` is too small to spare a non-empty, both-class internal val split,
  `fit` trains to `max_epochs` with `early_stop_reason="no_internal_val"`.
- **Determinism (REQUIRED — reproducibility-critical research; Codex P2):**
  `fit` seeds `torch.manual_seed(random_state)` and uses a local
  `numpy.random.default_rng(random_state)` for the early-stop split and the
  per-epoch shuffle index (no global numpy seeding). All batching is in-process
  (NO multi-worker DataLoader; if a DataLoader is used, `num_workers=0`). `fit`
  saves and **restores after returning** any global torch state it changes —
  `torch.use_deterministic_algorithms(True)` and the thread count — so it does
  not pollute later code or other tests. Two `fit` calls with the same
  `(X, y, random_state)` and params must produce bit-identical `predict_proba`.
  `random_state=None` is rejected fail-loud for a deep model (non-determinism
  would break the §13 freeze contract); the caller must pass an int seed
  (`type(random_state) is int`, reject bool).
- **Early-stop policy is part of the config surface, not a hidden detail
  (Codex P2):** it co-determines `actual_epochs_` and thus the fitted model, so
  it must be reconstructible for the future trial ledger / config hash. The
  fitted model exposes: `actual_epochs_` (int), `early_stop_reason_` (str ∈
  `{"patience", "max_epochs", "no_internal_val"}`), `internal_val_n_` (int rows
  held out for early stopping; 0 on the `no_internal_val` path), and the
  early-stop policy is fully determined by the frozen kwargs
  (`max_epochs`, `early_stopping_patience`, `early_stopping_fraction`) + the
  `random_state`-derived split seed. No ledger row is written here (that is the
  harness's job) — the body only exposes the fields so the harness can record
  them and so no hidden training budget escapes accounting.

## 6. Error Modes (fail-loud)

| # | condition | exception |
|---|---|---|
| 1 | any `__init__` axis fails its §4 **exact-type** check or is outside its frozen set (incl. `individual_channels=1`, `seasonal_trend_dropout=False`, `moving_avg_kernel=True`) | `TypeError`/`ValueError` |
| 2 | `random_state` is None, `bool`, or not exactly `int` | `TypeError`/`ValueError` |
| 3 | training kwarg wrong type (`bool` for an int/float kwarg) or out of range (`max_epochs<1`, `learning_rate<=0`, `batch_size<1`, `early_stopping_patience<1`, `early_stopping_fraction` ∉ (0,1), `weight_decay<0`) | `TypeError`/`ValueError` |
| 4 | `X` not a 3-D float ndarray, or `predict_proba` before `fit` | `ValueError` |
| 5 | `X` not finite (NaN/inf) | `ValueError` |
| 6 | `fit`: `y` not 1-D int in `{0,1}`, or `len(y) != len(X)` | `ValueError` |
| 7 | `fit`: `y` has a single class | `ValueError` |
| 8 | `predict_proba`: `X.shape[1:]` differs from the fitted `(L, C)` | `ValueError` |

## 7. Testing Strategy (`tests/models/deep_sequence/test_dlinear.py`)

No GPU; CPU only; small synthetic arrays; fast.

1. **Protocol conformance:** `isinstance(DLinearClassifier(...), SequenceClassifier)`;
   `fit` returns self; `predict_proba` shape `(n, 2)` float64, rows sum to 1.
2. **Determinism + no global-state pollution (Codex P2):** two fits with same
   seed → bit-exact `predict_proba`; different seeds → may differ; AND after
   `fit` returns, `torch.are_deterministic_algorithms_enabled()` and the torch
   thread count equal their pre-fit values (the fit restored them).
3. **Learns a SEQUENCE-ONLY signal (Codex P2):** synthetic windows whose label
   depends on the *temporal shape*, not the last step — e.g. all windows share
   the same final-bar value but differ in early-vs-late slope sign. Train
   accuracy clearly above 0.5 proves the DLinear temporal path works and is not
   a last-step shortcut (a last-step-only model would be at chance here). Kept
   loose to avoid flakiness.
4. **Search-axis coverage:** parametrize each §4 axis over its frozen set,
   assert `fit`+`predict_proba` run and shape-conform (incl. `individual_channels`
   True/False, both heads, `linear_bottleneck`, each kernel, each dropout).
5. **Guards:** all §6 error modes (bad axis value, `random_state=None`/bool,
   out-of-range training kwargs, 2-D X, non-finite X, single-class y, length
   mismatch, predict-before-fit, shape drift at predict).
6. **Early-stop bookkeeping:** a tiny `max_epochs`/`patience` run sets
   `actual_epochs_ <= max_epochs` and a valid `early_stop_reason_`; the
   `no_internal_val` path fires when `n` is too small.

Estimated ~25–35 tests. Cross-model anti-drift is not applicable (no
`baseline_v1` DLinear oracle); correctness rests on the protocol contract,
determinism, the separable-signal sanity, and Codex review.

## 8. Files

- **Modify** `src/intraday_research/models/deep_sequence/dlinear.py` (fill the
  two scaffolds + add training kwargs; keep the existing 5 axes + `random_state`).
- **Create** `tests/models/deep_sequence/test_dlinear.py` (mirror existing
  `tests/stages/models/` deep-sequence test conventions — confirm exact path in
  the plan).
- No change to `base.py`, `controls.py`, `folds.py`, or any data/contract/
  notebook/config file.

## 9. Dependencies

`torch` (2.12.0+cpu, confirmed installed) + `numpy`. `base.SequenceClassifier`
for the protocol. Nothing from `data/` (the body is data-agnostic). Downstream:
the future 08X harness instantiates this with per-trial axis values and feeds it
fold arrays; TCN/GRU/LSTM bodies reuse this piece's training-loop + validation +
determinism pattern.

## 10. Resolved Design Decisions (Codex-reviewed 2026-06-07)

1. **Internal early-stop split = random (seeded), not chronological.** It gates
   training duration only, lives within the caller's train-inner-fit rows, and
   never selects a candidate. Codex confirmed this is NOT an official-validation
   leak, with the §5 condition that the policy + `internal_val_n_` enter the
   trial-ledger/config surface (now specified). Kept as-is.
2. **Temporal collapse `L → 1`** (one learned linear filter per channel/
   component — not a last-step read). Codex confirmed this fits DLinear's
   small-model / linear-sequence-bias role; `L → H>1` is a future capacity axis,
   not the first piece. Kept as-is.
3. **`per_channel` head = mean over channels of per-channel logits** (NOT mean
   of probabilities), where "channels" are the latent channels after any
   `linear_bottleneck` projection. The `shared` head already covers
   cross-channel weighting; `per_channel` is the low-parameter ablation.
   Clarified per Codex.
