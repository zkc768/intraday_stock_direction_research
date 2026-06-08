"""Tests for the 08X RUN_08X_BUILD_TRAIN_INNER_FOLDS slice (#5F-2).

All synthetic — no real ticker CSV is touched.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    validate_08x_fold_results_frame,
)
from intraday_research.data.splits import PARTITION_TRAIN, PARTITION_VALIDATION
from intraday_research.stages import deep_sequence_exploration as stage
from intraday_research.stages.deep_sequence_fold_build import (
    FoldSpec,
    build_fold_plan,
    build_fold_results,
    resolve_train_inner_index,
    run_build_train_inner_folds,
)
from intraday_research.stages.deep_sequence_schema_smoke import FOLD_RESULTS_COLUMNS


def _synth_index(n_per_ticker=12, tickers=("AAA", "BBB"), partition_value=PARTITION_TRAIN):
    base = np.datetime64("2014-03-03T09:30:00", "ns")
    step = np.timedelta64(5, "m").astype("timedelta64[ns]")
    ts, tk, part = [], [], []
    for ticker in tickers:
        for i in range(n_per_ticker):
            ts.append(base + step * i)
            tk.append(ticker)
            part.append(partition_value)
    return {
        "target_timestamps": np.array(ts, dtype="datetime64[ns]"),
        "target_ticker_ids": np.array(tk, dtype=object),
        "target_partition": np.array(part, dtype=np.int8),
    }


# --------------------------------------------------------------------------
# build_fold_results (pure)
# --------------------------------------------------------------------------

def test_build_fold_results_rolling_origin():
    wi = _synth_index(n_per_ticker=12)
    plan = [FoldSpec("rolling_origin_folds", 2, 1, 2, 0)]
    df = build_fold_results(wi["target_timestamps"], wi["target_ticker_ids"], plan)
    assert list(df.columns) == list(FOLD_RESULTS_COLUMNS)
    assert len(df) == 2
    assert set(df["fold_scheme"]) == {"rolling_origin_folds"}
    assert (df["purge_gap_k"] == 1).all()
    assert (df["embargo_gap_k"] == 0).all()
    assert (df["train_inner_fit_n"] > 0).all()
    assert (df["train_inner_validation_n"] > 0).all()
    assert list(df["fold_id"]) == ["rolling_origin_folds__0", "rolling_origin_folds__1"]
    validate_08x_fold_results_frame(df, require_non_empty=True)


def test_build_fold_results_multi_scheme():
    wi = _synth_index(n_per_ticker=15)
    plan = [
        FoldSpec("rolling_origin_folds", 2, 1, 2, 0),
        FoldSpec("purged_time_series_folds", 3, 1, 1, 0),
        FoldSpec("embargoed_train_inner_folds", 3, 1, 1, 1),
    ]
    df = build_fold_results(wi["target_timestamps"], wi["target_ticker_ids"], plan)
    assert set(df["fold_scheme"]) == {
        "rolling_origin_folds",
        "purged_time_series_folds",
        "embargoed_train_inner_folds",
    }
    emb = df[df["fold_scheme"] == "embargoed_train_inner_folds"]
    assert (emb["embargo_gap_k"] == 1).all()
    non_emb = df[df["fold_scheme"] != "embargoed_train_inner_folds"]
    assert (non_emb["embargo_gap_k"] == 0).all()
    validate_08x_fold_results_frame(df, require_non_empty=True)


def test_fold_build_fails_loud_on_too_few_samples():
    wi = _synth_index(n_per_ticker=3)  # < 2*2 + 1 + 1 required for rolling origin
    plan = [FoldSpec("rolling_origin_folds", 2, 1, 2, 0)]
    with pytest.raises(ValueError):
        build_fold_results(wi["target_timestamps"], wi["target_ticker_ids"], plan)


# --------------------------------------------------------------------------
# build_fold_plan
# --------------------------------------------------------------------------

def test_build_fold_plan_defaults_to_rolling_origin():
    plan = build_fold_plan({"fold_plan": {"label_horizon_k": 1, "inner_validation_size": 2}})
    assert [spec.scheme for spec in plan] == ["rolling_origin_folds"]
    assert plan[0].n_folds == 2  # default budget tier = quick


def test_build_fold_plan_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unknown schemes"):
        build_fold_plan(
            {"fold_plan": {"selected_fold_modes": ["bogus"], "label_horizon_k": 1}}
        )


def test_build_fold_plan_requires_label_horizon_k():
    with pytest.raises(ValueError, match="label_horizon_k"):
        build_fold_plan({"fold_plan": {"selected_fold_modes": ["rolling_origin_folds"]}})


def test_build_fold_plan_rejects_bool_n_folds():
    with pytest.raises(ValueError, match="n_folds"):
        build_fold_plan(
            {"fold_plan": {"label_horizon_k": 1, "n_folds": True}}
        )


# --------------------------------------------------------------------------
# resolve_train_inner_index (train-partition filter / seam)
# --------------------------------------------------------------------------

def test_resolve_train_inner_index_filters_to_train_partition():
    train = _synth_index(n_per_ticker=10, partition_value=PARTITION_TRAIN)
    val = _synth_index(n_per_ticker=4, partition_value=PARTITION_VALIDATION)
    mixed = {
        key: np.concatenate([train[key], val[key]])
        for key in ("target_timestamps", "target_ticker_ids", "target_partition")
    }
    ts, tk = resolve_train_inner_index({}, injected_window_index=mixed)
    assert len(ts) == 20  # 10 train * 2 tickers; validation rows dropped
    assert len(tk) == 20


def test_resolve_train_inner_index_requires_window_index():
    with pytest.raises(NotImplementedError, match="later slice"):
        resolve_train_inner_index({})


def test_resolve_train_inner_index_no_train_rows_fails():
    val_only = _synth_index(n_per_ticker=6, partition_value=PARTITION_VALIDATION)
    with pytest.raises(ValueError, match="PARTITION_TRAIN"):
        resolve_train_inner_index({}, injected_window_index=val_only)


# --------------------------------------------------------------------------
# dispatch (run_build_train_inner_folds / run_stage)
# --------------------------------------------------------------------------

def _build_folds_config(wi, *, label_horizon_k=1, horizon_k=1):
    return {
        "windowed_index": wi,
        "fold_plan": {
            "selected_fold_modes": ["rolling_origin_folds"],
            "n_folds": 2,
            "label_horizon_k": label_horizon_k,
            "inner_validation_size": 2,
        },
        "frozen_candidate": {"candidate_id": "cand_007", "horizon_k": horizon_k},
    }


def test_label_horizon_provenance_mismatch_fails(tmp_path):
    wi = _synth_index(n_per_ticker=12)
    config = _build_folds_config(wi, label_horizon_k=3, horizon_k=1)
    with pytest.raises(ValueError, match="label_horizon_k"):
        run_build_train_inner_folds(config, tmp_path)


def test_provenance_requires_candidate_horizon(tmp_path):
    wi = _synth_index(n_per_ticker=12)
    config = {
        "windowed_index": wi,
        "fold_plan": {
            "selected_fold_modes": ["rolling_origin_folds"],
            "n_folds": 2,
            "label_horizon_k": 1,
            "inner_validation_size": 2,
        },
        # no frozen_candidate
    }
    with pytest.raises(ValueError, match="frozen_candidate"):
        run_build_train_inner_folds(config, tmp_path)


def test_run_stage_build_folds_writes_real_rows_and_manifest(tmp_path):
    wi = _synth_index(n_per_ticker=12)
    config = {
        "run_switches": {"RUN_08X_BUILD_TRAIN_INNER_FOLDS": True},
        **_build_folds_config(wi),
    }
    stage.run_stage(config, output_dir=tmp_path)

    df = pd.read_csv(tmp_path / "08x_fold_results.csv")
    assert not df.empty
    validate_08x_fold_results_frame(df, require_non_empty=True)
    assert set(df["fold_scheme"]) == {"rolling_origin_folds"}

    manifest = json.loads((tmp_path / "08x_run_manifest.json").read_text("utf-8"))
    assert manifest["train_inner_fold_policy"] == "rolling_origin_folds"
    assert manifest["purge_policy"] == "horizon_bar_purge_k=1"
    assert manifest["embargo_policy"] == "none"
    assert manifest["official_validation_used"] is False
    assert manifest["holdout_test_authorized"] is False
    # P1-1: smoke placeholder candidate string overwritten with the real id
    assert manifest["source_stage0_candidate"] == "cand_007"

    # full 08X bundle skeletons laid down in the clean output dir
    assert (tmp_path / "08x_search_space.json").exists()
    assert (tmp_path / "08x_environment_manifest.json").exists()


def test_run_stage_build_folds_embargo_policy_string(tmp_path):
    wi = _synth_index(n_per_ticker=15)
    config = {
        "run_switches": {"RUN_08X_BUILD_TRAIN_INNER_FOLDS": True},
        "windowed_index": wi,
        "fold_plan": {
            "selected_fold_modes": ["embargoed_train_inner_folds"],
            "n_folds": 3,
            "label_horizon_k": 1,
            "embargo_size": 2,
        },
        "frozen_candidate": {"horizon_k": 1},
    }
    stage.run_stage(config, output_dir=tmp_path)
    manifest = json.loads((tmp_path / "08x_run_manifest.json").read_text("utf-8"))
    assert manifest["train_inner_fold_policy"] == "embargoed_train_inner_folds"
    assert manifest["embargo_policy"] == "symmetric_embargo_k=2"
    # no candidate_id given -> deterministic provenance string from horizon_k
    assert manifest["source_stage0_candidate"] == "frozen_candidate_horizon_k=1"


def test_build_folds_and_smoke_mutually_exclusive(tmp_path):
    config = {
        "run_switches": {
            "RUN_08X_SCHEMA_SMOKE": True,
            "RUN_08X_BUILD_TRAIN_INNER_FOLDS": True,
        },
        **_build_folds_config(_synth_index(n_per_ticker=12)),
    }
    with pytest.raises(ValueError, match="one of"):
        stage.run_stage(config, output_dir=tmp_path)


def test_build_folds_refuses_rebuild_over_existing_trials(tmp_path):
    """Codex impl review P1-1: non-empty trial ledger blocks a fold rebuild."""
    wi = _synth_index(n_per_ticker=12)
    config = {
        "run_switches": {"RUN_08X_BUILD_TRAIN_INNER_FOLDS": True},
        **_build_folds_config(wi),
    }
    stage.run_stage(config, output_dir=tmp_path)  # first build OK
    # simulate downstream trial evidence
    pd.DataFrame([{"trial_id": "t0"}]).to_csv(
        tmp_path / "08x_trial_ledger.csv", index=False
    )
    with pytest.raises(ValueError, match="trial rows"):
        stage.run_stage(config, output_dir=tmp_path)


def test_resolve_train_inner_index_rejects_unexpected_partition_code():
    """Codex impl review P1-2: a foreign partition code fails loud, not dropped."""
    wi = _synth_index(n_per_ticker=8, partition_value=PARTITION_TRAIN)
    wi["target_partition"] = wi["target_partition"].copy()
    wi["target_partition"][0] = np.int8(7)
    with pytest.raises(ValueError, match="unexpected partition codes"):
        resolve_train_inner_index({}, injected_window_index=wi)
