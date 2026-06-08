"""Append-only validation-budget ledger helper for N07/N08 package stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from intraday_research.contracts.validation_synthesis_gap_audit import (
    REQUIRED_LEDGER_COLUMNS,
    validate_ledger_frame,
    validate_ledger_prefix_invariance,
)


LEDGER_COLUMNS: tuple[str, ...] = (
    "artifact",
    "notebook_stage",
    "decision_made",
    "decision_timing",
    "decision_surface",
    "model_families_considered",
    "profiles_or_trials_considered",
    "seeds_used",
    "thresholds_or_coverages_considered",
    "official_validation_rows_inspected",
    "cumulative_official_validation_inspections_across_notebooks",
    "train_inner_only_decision",
    "official_validation_informed_decision",
    "diagnostic_only_readout",
    "holdout_test_contact",
    "allowed_wording",
    "forbidden_wording",
    "risk_note",
    "appended_by_notebook",
    "appended_at_utc",
)

if set(LEDGER_COLUMNS) != REQUIRED_LEDGER_COLUMNS:
    raise RuntimeError("LEDGER_COLUMNS does not match REQUIRED_LEDGER_COLUMNS")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ledger_frame(path: Path | str) -> pd.DataFrame:
    """Read a ledger CSV without pandas NA/type inference.

    Preserving strings such as ``n/a`` and ``07`` is part of the N07 static
    contract, because the ledger is an append-only audit artifact rather than a
    numeric modeling table.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@dataclass
class ValidationBudgetLedger:
    """Durable append-only writer for ``notebook07_validation_budget_ledger.csv``."""

    path: Path | str
    appended_by_notebook: str = "07"
    now: Callable[[], str] = utc_now_iso
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False)
    _columns: tuple[str, ...] = field(default=LEDGER_COLUMNS, init=False)

    @property
    def target_path(self) -> Path:
        return Path(self.path)

    def hydrate_from_disk_if_needed(self) -> None:
        """Load existing on-disk rows once so later appends extend the prefix."""
        if self._rows:
            return
        target = self.target_path
        if not target.exists():
            return
        on_disk = read_ledger_frame(target)
        validate_ledger_frame(on_disk)
        self._columns = tuple(on_disk.columns)
        self._rows.extend(on_disk.to_dict("records"))

    def flush_to_disk(self) -> Path:
        """Write memory to disk after validating append-only prefix invariance."""
        target = self.target_path
        new_df = pd.DataFrame(self._rows, columns=list(self._columns))
        validate_ledger_frame(new_df)
        if target.exists():
            existing_df = read_ledger_frame(target)
            validate_ledger_frame(existing_df)
            validate_ledger_prefix_invariance(existing_df, new_df)
        target.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(target, index=False, lineterminator="\n")
        return target

    def append_row(
        self,
        artifact: Any,
        notebook_stage: Any,
        decision_made: Any,
        decision_timing: Any,
        decision_surface: Any,
        *,
        model_families_considered: Any = "lightgbm",
        profiles_or_trials_considered: Any = "lightgbm_winner",
        seeds_used: Any = "",
        thresholds_or_coverages_considered: Any = "n/a",
        official_validation_rows_inspected: int = 0,
        train_inner_only_decision: bool = False,
        official_validation_informed_decision: bool = False,
        diagnostic_only_readout: bool = False,
        holdout_test_contact: bool = False,
        allowed_wording: Any = "",
        forbidden_wording: Any = "",
        risk_note: Any = "",
    ) -> dict[str, Any]:
        """Append a ledger row and flush before the caller can read metrics."""
        self.hydrate_from_disk_if_needed()
        last_cumulative = (
            int(
                self._rows[-1][
                    "cumulative_official_validation_inspections_across_notebooks"
                ]
            )
            if self._rows
            else 0
        )
        cumulative = last_cumulative + int(official_validation_rows_inspected)
        row = {
            "artifact": str(artifact),
            "notebook_stage": str(notebook_stage),
            "decision_made": str(decision_made),
            "decision_timing": str(decision_timing),
            "decision_surface": str(decision_surface),
            "model_families_considered": str(model_families_considered),
            "profiles_or_trials_considered": str(profiles_or_trials_considered),
            "seeds_used": str(seeds_used),
            "thresholds_or_coverages_considered": str(
                thresholds_or_coverages_considered
            ),
            "official_validation_rows_inspected": int(
                official_validation_rows_inspected
            ),
            "cumulative_official_validation_inspections_across_notebooks": int(
                cumulative
            ),
            "train_inner_only_decision": bool(train_inner_only_decision),
            "official_validation_informed_decision": bool(
                official_validation_informed_decision
            ),
            "diagnostic_only_readout": bool(diagnostic_only_readout),
            "holdout_test_contact": bool(holdout_test_contact),
            "allowed_wording": str(allowed_wording),
            "forbidden_wording": str(forbidden_wording),
            "risk_note": str(risk_note),
            "appended_by_notebook": str(self.appended_by_notebook),
            "appended_at_utc": self.now(),
        }
        self._rows.append(row)
        self.flush_to_disk()
        return row


def append_validation_budget_ledger_row(
    path: Path | str,
    artifact: Any,
    notebook_stage: Any,
    decision_made: Any,
    decision_timing: Any,
    decision_surface: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """One-shot append helper for callers that do not keep a ledger object."""
    ledger = ValidationBudgetLedger(path)
    return ledger.append_row(
        artifact,
        notebook_stage,
        decision_made,
        decision_timing,
        decision_surface,
        **kwargs,
    )
