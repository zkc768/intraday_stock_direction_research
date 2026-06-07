"""Skeleton for stage controlled_followup. Body migration is a separate authorized phase."""
from __future__ import annotations

STAGE_NAME = "controlled_followup"
REQUIRED_ARTIFACTS: tuple[str, ...] = ()


def run_stage(config) -> None:
    raise NotImplementedError(
        f"Stage '{STAGE_NAME}' body not migrated yet. "
        f"Required artifacts: {REQUIRED_ARTIFACTS}. "
        f"See AGENTS.md §4.3 for ledger-append-before-read if reading official validation."
    )
