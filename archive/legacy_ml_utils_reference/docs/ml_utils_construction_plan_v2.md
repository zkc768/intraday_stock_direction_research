# ml_utils Historical Reference

> This file used to be the active construction plan for `ml_utils`.
> It is now a historical implementation reference only.
>
> The active research workflow is:
>
> 1. `AGENTS.md`
> 2. `docs/RESEARCH_WORKFLOW.md`
> 3. the current research notebook

---

## 1. Current Role of ml_utils

`ml_utils/` is a small helper library that protects stable, repeated research
logic. It is not the product, not the default workflow, and not a reason to turn
ordinary exploration into a backend project.

Use `ml_utils/` for code that must be correct every time:

- chronological split handling
- split-boundary and trading-day label invalidation
- per-ticker window construction
- train-only scaling or preprocessing helpers
- metric computation and dummy baselines
- stable model definitions already used by notebooks

Keep one-off exploration inside the notebook until it has been reused enough to
justify tests.

---

## 2. When To Modify ml_utils

Modify `ml_utils/` only when all of these are true:

1. The notebook has shown that the logic is repeated or safety-critical.
2. The exact function boundary is clear.
3. The behavior can be tested with synthetic in-memory data.
4. The change protects research validity or reduces real duplication.

Do not modify `ml_utils/` just to make a notebook look tidier.

---

## 3. Hard Invariants To Preserve

These are the useful parts of the old construction plan that still matter:

- Splits are chronological.
- Scalers and preprocessing fit train only.
- Labels may be computed per full ticker sequence, but samples whose future
  horizon crosses split boundaries or trading-day boundaries are marked invalid
  before window construction.
- Invalid labels are markers. Do not erase them with global `dropna` or
  `fillna`.
- Windows are generated per ticker and never span tickers.
- Window target index is:

```text
target_idx = start + window_size - 1
```

- If the target label is invalid, skip that window.
- Metrics must include macro F1, balanced accuracy, confusion matrix with
  explicit labels, dummy baselines, and `delta_macro_f1_vs_dummy`.
- Holdout/test access remains closed unless a separate pre-registered decision
  explicitly opens it.

---

## 4. Deprecated From the Old Plan

Do not revive these as default workflow:

- module-by-module PM implementation sessions
- long line-budget tables
- prompt templates
- route-control closeout documents
- test-first notebook exploration
- "library is the product" framing
- expanding CLI runners for ordinary research
- creating new design docs before each experiment

The old detailed construction history is recoverable from git history before
the notebook-first reset.

---

## 5. Current Practical Rule

Start in the notebook. Extract only the parts that are:

```text
reused
safety-critical
testable
small
```

If a future agent wants to change `ml_utils/`, it should first state the
notebook evidence that motivates the extraction and the exact tests that will
protect the behavior.
