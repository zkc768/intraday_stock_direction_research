"""Skeleton for stage lightgbm_tuning. Body migration is a separate authorized phase."""
from __future__ import annotations

STAGE_NAME = "lightgbm_tuning"
REQUIRED_ARTIFACTS: tuple[str, ...] = ()


def run_stage(config) -> None:
    raise NotImplementedError(
        f"Stage '{STAGE_NAME}' body not migrated yet. "
        f"Required artifacts: {REQUIRED_ARTIFACTS}. "
        f"See AGENTS.md §4.3 for ledger-append-before-read if reading official validation."
    )
