# artifacts/code_management/

Migration housekeeping artifacts for the GitHub / package-first repo cleanup.
Not active research data. Safe to track in git.

## Contents

- `migration_inventory.csv` — Phase 0 snapshot of every relevant repo path,
  classified by `category` (e.g. `canonical_source`, `generated_notebook`,
  `contract_helper`, `static_gate_test`, `protocol_doc`, `raw_data`,
  `runtime_output`) and pinned to a single `action` label drawn from the
  fixed set: `KEEP_LOCAL_ONLY` / `SHIM_LATER` / `MOVE_IN_PHASE_3` /
  `MOVE_IN_FUTURE_PHASE` / `INVENTORY_ONLY` / `FLAG_FOR_REVIEW`.

## Conventions

- All `path` values are repo-relative POSIX paths.
- No Drive file IDs, Drive folder URLs, share links, gsutil paths, or
  machine-local absolute paths in any column.
- `notes` field is space-separated tags only (no commas, no embedded
  newlines). The runtime ledger row's `notes` starts with the literal
  string `append_only_per_AGENTS_4.3` per AGENTS.md §4.3.

## Regeneration

The CSV is hand-curated for now. If/when a generator script is added it
should live next to this file and write the same column schema:

```text
path, category, current_role, target_role, action, risk, notes
```

The inventory is read-only context for downstream phases; no later phase
mutates it.
