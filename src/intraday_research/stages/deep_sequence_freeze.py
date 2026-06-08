"""08F candidate freeze pipeline — contract gate, compression, freeze record.

Simple implementations of the three 08F run switches:
- RUN_08F_CONTRACT_GATE: check 08X outputs are complete
- RUN_08F_CANDIDATE_COMPRESSION: score candidates via §9.2 paper_safe_score
- RUN_08F_WRITE_FREEZE_RECORD: write the freeze record JSON + markdown
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intraday_research.contracts.deep_sequence_exploration import (
    CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA,
    OUTPUT_FILES_08X,
    PAPER_SAFE_SCORE_PENALTY_COMPLEXITY,
    PAPER_SAFE_SCORE_PENALTY_COMPUTE,
    PAPER_SAFE_SCORE_WEIGHT_FOLD_CONSISTENCY,
    PAPER_SAFE_SCORE_WEIGHT_LCB_DELTA,
    PAPER_SAFE_SCORE_WEIGHT_MEAN_DELTA,
    PAPER_SAFE_SCORE_WEIGHT_PER_TICKER,
    PAPER_SAFE_SCORE_WEIGHT_SEED_STABILITY,
    SEED_STABILITY_SCALE_FALLBACK,
    validate_freeze_record,
)
from intraday_research.stages.io_helpers import sha256_bytes, write_json

logger = logging.getLogger(__name__)


def run_08f_contract_gate(out: Path) -> dict[str, Any]:
    """Check that all 08X output artifacts exist and are non-empty.

    Returns a gate report dict with 'passed' bool.
    """
    report: dict[str, Any] = {"passed": True, "missing": [], "empty": []}
    for filename in OUTPUT_FILES_08X:
        path = out / filename
        if not path.exists():
            report["missing"].append(filename)
            report["passed"] = False
        elif path.stat().st_size == 0:
            report["empty"].append(filename)
            report["passed"] = False
    write_json(out / "08f_static_gate_report.json", report)
    logger.info("08F contract gate: %s", "PASSED" if report["passed"] else "FAILED")
    return report


def run_08f_candidate_compression(out: Path) -> pd.DataFrame:
    """Score candidates via §9.2 paper_safe_score; write compression table.

    Reads 08x_seed_summary.csv and 08x_per_ticker.csv, computes a composite
    score for each candidate, writes 08x_candidate_compression_table.csv.
    Returns the compression table as a DataFrame.
    """
    seed_summary = pd.read_csv(out / "08x_seed_summary.csv")
    per_ticker = pd.read_csv(out / "08x_per_ticker.csv")

    if seed_summary.empty:
        empty = pd.DataFrame(columns=[
            "candidate_id", "candidate_family", "paper_safe_score",
            "z_lcb_delta", "z_mean_delta", "z_seed_stability",
            "z_fold_consistency", "z_per_ticker",
            "complexity_penalty", "compute_penalty", "compute_tier",
        ])
        empty.to_csv(out / "08x_candidate_compression_table.csv", index=False)
        return empty

    candidates = seed_summary["candidate_id"].unique()

    # Build per-candidate metric table.
    rows: list[dict[str, Any]] = []
    for cid in candidates:
        cs = seed_summary[seed_summary["candidate_id"] == cid]
        pt = per_ticker[per_ticker["candidate_id"] == cid]
        fam = pt["candidate_family"].iloc[0] if not pt.empty else "unknown"

        lcb_delta = _metric_value(cs, "delta_macro_f1_vs_dummy", "seed_lcb_95")
        mean_delta = _metric_value(cs, "delta_macro_f1_vs_dummy", "seed_mean")
        seed_std = _metric_value(cs, "macro_f1", "seed_std")
        n_positive = int(pt["positive_delta"].sum()) if not pt.empty else 0
        n_tickers = len(pt) if not pt.empty else 1

        rows.append({
            "candidate_id": cid,
            "candidate_family": fam,
            "lcb_delta": lcb_delta,
            "mean_delta": mean_delta,
            "seed_std": seed_std,
            "n_positive_tickers": n_positive,
            "n_tickers": n_tickers,
        })

    df = pd.DataFrame(rows)

    # Z-score each dimension within the candidate pool.
    df["z_lcb_delta"] = _zscore(df["lcb_delta"])
    df["z_mean_delta"] = _zscore(df["mean_delta"])

    seed_scale = SEED_STABILITY_SCALE_FALLBACK
    df["z_seed_stability"] = _zscore(-df["seed_std"].fillna(seed_scale))
    df["z_fold_consistency"] = df["z_lcb_delta"]  # proxy: LCB already penalizes fold variance
    df["z_per_ticker"] = _zscore(df["n_positive_tickers"].astype(float))

    # §9.2 complexity/compute penalties: lightgbm control = 0 penalty, deep = full penalty.
    df["complexity_penalty"] = df["candidate_family"].apply(
        lambda f: 0.0 if f == "last_step_lightgbm_control" else PAPER_SAFE_SCORE_PENALTY_COMPLEXITY
    )
    df["compute_penalty"] = df["candidate_family"].apply(
        lambda f: 0.0 if f == "last_step_lightgbm_control" else PAPER_SAFE_SCORE_PENALTY_COMPUTE
    )
    df["compute_tier"] = "full_compute"

    df["paper_safe_score"] = (
        PAPER_SAFE_SCORE_WEIGHT_LCB_DELTA * df["z_lcb_delta"]
        + PAPER_SAFE_SCORE_WEIGHT_MEAN_DELTA * df["z_mean_delta"]
        + PAPER_SAFE_SCORE_WEIGHT_SEED_STABILITY * df["z_seed_stability"]
        + PAPER_SAFE_SCORE_WEIGHT_FOLD_CONSISTENCY * df["z_fold_consistency"]
        + PAPER_SAFE_SCORE_WEIGHT_PER_TICKER * df["z_per_ticker"]
        + df["complexity_penalty"]
        + df["compute_penalty"]
    )

    result = df[[
        "candidate_id", "candidate_family", "paper_safe_score",
        "z_lcb_delta", "z_mean_delta", "z_seed_stability",
        "z_fold_consistency", "z_per_ticker",
        "complexity_penalty", "compute_penalty", "compute_tier",
    ]].sort_values("paper_safe_score", ascending=False).reset_index(drop=True)

    result.to_csv(out / "08x_candidate_compression_table.csv", index=False)
    logger.info("08F compression: scored %d candidates", len(result))
    return result


def run_08f_write_freeze_record(
    out: Path, config: dict[str, Any],
) -> dict[str, Any]:
    """Pick primary + fallback from compression table; write freeze record.

    If no candidate meets the eligibility threshold, writes
    08f_no_candidate_freezable.json instead.
    """
    compression = pd.read_csv(out / "08x_candidate_compression_table.csv")
    seed_summary = pd.read_csv(out / "08x_seed_summary.csv")

    # Filter eligible: lcb_delta >= threshold.
    eligible_ids: list[str] = []
    for cid in compression["candidate_id"]:
        cs = seed_summary[
            (seed_summary["candidate_id"] == cid)
            & (seed_summary["metric"] == "delta_macro_f1_vs_dummy")
        ]
        if not cs.empty:
            lcb = float(cs["seed_lcb_95"].iloc[0])
            if lcb >= CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA:
                eligible_ids.append(cid)

    if not eligible_ids:
        no_freeze = {
            "reason": "no candidate meets eligibility threshold",
            "threshold": CANDIDATE_ELIGIBILITY_MIN_TRAIN_INNER_LCB_DELTA,
            "n_candidates_scored": len(compression),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(out / "08f_no_candidate_freezable.json", no_freeze)
        logger.info("08F: no candidate freezable")
        return no_freeze

    # Pick primary (highest paper_safe_score among eligible).
    eligible_comp = compression[compression["candidate_id"].isin(eligible_ids)]
    eligible_comp = eligible_comp.sort_values("paper_safe_score", ascending=False)
    primary_id = str(eligible_comp.iloc[0]["candidate_id"])
    primary_family = str(eligible_comp.iloc[0]["candidate_family"])
    primary_score = float(eligible_comp.iloc[0]["paper_safe_score"])

    # Fallback = second-best eligible, or empty.
    fallback_id = ""
    if len(eligible_comp) >= 2:
        fallback_id = str(eligible_comp.iloc[1]["candidate_id"])

    # Read search_space for frozen params.
    search_space = json.loads((out / "08x_search_space.json").read_text())
    candidate_configs = {c["candidate_id"]: c for c in search_space.get("candidates", [])}
    primary_cfg = candidate_configs.get(primary_id, {})

    freeze_record: dict[str, Any] = {
        "stage": "08F",
        "scope": "validation_only",
        "primary_candidate_id": primary_id,
        "fallback_candidate_id": fallback_id,
        "fallback_activation_rule": (
            "activate fallback if primary train-inner lcb_delta_macro_f1 "
            "drops below eligibility threshold on re-run with different seed"
        ),
        "config_hash": primary_cfg.get("config_hash", ""),
        "architecture_family": primary_family,
        "frozen_architecture_params": primary_cfg.get("model_config", {}),
        "frozen_loss": "cross_entropy",
        "frozen_hpo_method": search_space.get("hpo_method", "random_search"),
        "frozen_seed_list": search_space.get("seed_list", []),
        "frozen_metric_list": ["macro_f1", "delta_macro_f1_vs_dummy", "balanced_accuracy"],
        "frozen_wording_rule": "improvement if delta > threshold else weak_mixed",
        "paper_safe_score": primary_score,
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    validate_freeze_record(freeze_record)
    write_json(out / "08f_candidate_freeze_record.json", freeze_record)

    # Write markdown summary.
    md = (
        f"# 08F Candidate Freeze Record\n\n"
        f"- Primary: `{primary_id}` ({primary_family}), score={primary_score:.4f}\n"
        f"- Fallback: `{fallback_id or 'none'}`\n"
        f"- Generated: {freeze_record['generated_at_utc']}\n"
    )
    (out / "08f_candidate_freeze_record.md").write_text(md, encoding="utf-8")

    logger.info("08F: froze primary=%s, fallback=%s", primary_id, fallback_id)
    return freeze_record


def _metric_value(summary: pd.DataFrame, metric: str, column: str) -> float:
    row = summary[summary["metric"] == metric]
    if row.empty:
        return float("nan")
    return float(row[column].iloc[0])


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
