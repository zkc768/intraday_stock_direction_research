"""Skeleton for stage deep_sequence_exploration. Body migration is a separate authorized phase.

For N07/N08: see AGENTS.md §4.3 — any read of official-validation metrics
MUST append a ledger row BEFORE reading; pre-existing rows MUST NOT be
modified, dropped, or reordered.
"""
from __future__ import annotations

STAGE_NAME = "deep_sequence_exploration"
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "notebook07_validation_budget_ledger.csv",
)


def run_stage(config) -> None:
    raise NotImplementedError(
        f"Stage '{STAGE_NAME}' body not migrated yet. "
        f"Required artifacts: {REQUIRED_ARTIFACTS}. "
        f"See AGENTS.md §4.3 for ledger-append-before-read if reading official validation."
    )
