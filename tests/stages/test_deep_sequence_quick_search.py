"""Tests for the #5F-6 quick-search loop (synthetic data; mostly monkeypatched
single-trial runner, plus one tiny real DLinear end-to-end fit)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    validate_trial_ledger_frame,
)
from intraday_research.data.splits import PARTITION_TRAIN
from intraday_research.stages import deep_sequence_quick_search as qs


def _windowed_index(*, per_ticker=15, t=4, f=3, seed=0, val_single_class=False):
    """Build a synthetic train-partition windowed index: 2 tickers, alternating
    classes (so train+val carry both classes), optionally forcing a single-class
    validation tail."""
    rng = np.random.default_rng(seed)
    blocks_x, blocks_y, blocks_ts, blocks_tk = [], [], [], []
    for name in ("AAA", "BBB"):
        blocks_x.append(rng.standard_normal((per_ticker, t, f)).astype(np.float64))
        labels = np.array([j % 2 for j in range(per_ticker)], dtype=np.int64)
        if val_single_class:
            labels[-3:] = 0  # force the last (validation) rows to one class
        blocks_y.append(labels)
        days = np.datetime64("2014-01-01") + np.arange(per_ticker)
        blocks_ts.append(days.astype("datetime64[ns]"))
        blocks_tk.append(np.array([name] * per_ticker, dtype=object))
    x = np.concatenate(blocks_x, axis=0)
    return {
        "X": x,
        "y": np.concatenate(blocks_y),
        "target_partition": np.full(x.shape[0], PARTITION_TRAIN, dtype=np.int8),
        "target_timestamps": np.concatenate(blocks_ts),
        "target_ticker_ids": np.concatenate(blocks_tk),
    }


def _config(
    windowed_index,
    *,
    horizon_k=3,
    n_folds=1,
    inner_validation_size=2,
    candidates=None,
    seed_list=(0, 1),
    scientific_cap=None,
    families=None,
):
    search_space: dict = {"hpo_method": "random_search", "seed_list": list(seed_list)}
    if candidates is not None:
        search_space["candidates"] = candidates
    if families is not None:
        search_space["architecture_families"] = families
    if scientific_cap is not None:
        search_space["scientific_budget_cap_total_trials"] = scientific_cap
    return {
        "windowed_index": windowed_index,
        "frozen_candidate": {"candidate_id": "frozen_x", "horizon_k": horizon_k},
        "fold_plan": {
            "selected_fold_modes": ["rolling_origin_folds"],
            "n_folds": n_folds,
            "label_horizon_k": horizon_k,
            "inner_validation_size": inner_validation_size,
            "embargo_size": 0,
        },
        "search_space": search_space,
    }


def _fake_row(*, trial_id, candidate_family, candidate_id, config_hash, fold_id, seed,
              class0=0.5, class1=0.5, macro_f1=0.6):
    """A schema-complete completed trial-ledger row for the monkeypatched runner."""
    row = {
        "trial_id": trial_id,
        "candidate_family": candidate_family,
        "candidate_id": candidate_id,
        "config_hash": config_hash,
        "fold_id": fold_id,
        "seed": int(seed),
        "budget_tier": "quick",
        "max_epochs": 1.0,
        "actual_epochs": 1.0,
        "early_stop_reason": "",
        "fit_status": "completed",
        "failure_type": "",
        "failure_message": "",
        "train_inner_fit_n": 10,
        "train_inner_validation_n": 4,
        "actual_wall_clock_seconds": 0.01,
        "peak_memory_mb": 1.0,
        "gpu_seconds_or_null": None,
        "compute_tier": "full_compute",
        "scope": "exploratory",
        "official_validation_used": False,
        "holdout_test_authorized": False,
        "macro_f1": macro_f1,
        "balanced_accuracy": 0.6,
        "accuracy": 0.6,
        "stratified_dummy_macro_f1_same_rows": 0.5,
        "delta_macro_f1_vs_dummy": macro_f1 - 0.5,
        "class0_pred_rate": class0,
        "class1_pred_rate": class1,
        "ticker_max_share": 0.5,
    }
    return row


def _recorder(class0=0.5, class1=0.5):
    """Return a stand-in for run_single_trial: a completed row whose macro_f1
    varies deterministically with the seed (so seed aggregation is testable)."""

    def _rec(X, y, ticker_ids, *, train_idx, val_idx, trial_id, candidate_family,
             candidate_id, config_hash, fold_id, seed, budget_tier, model_config):
        return _fake_row(
            trial_id=trial_id,
            candidate_family=candidate_family,
            candidate_id=candidate_id,
            config_hash=config_hash,
            fold_id=fold_id,
            seed=int(seed),
            class0=class0,
            class1=class1,
            macro_f1=0.6 + 0.05 * int(seed),
        )

    return _rec


def test_quick_search_loop_assembles_ledger(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[
            {"family": "dlinear_only", "candidate_id": f"c{i}", "model_config": {}}
            for i in range(4)
        ],
        seed_list=(0, 1),
    )
    monkeypatch.setattr(qs, "run_single_trial", _recorder())
    ledger = qs.run_quick_search(config, tmp_path)

    assert len(ledger) == 8  # 4 candidates x 1 fold x 2 seeds
    assert (ledger["fit_status"] == "completed").all()
    validate_trial_ledger_frame(ledger)

    space = json.loads((tmp_path / "08x_search_space.json").read_text())
    assert space["search_space_version"] == qs.QUICK_SEARCH_VERSION
    assert "search_space_sha256" in space
    assert len(space["candidates"]) == 4

    seed_summary = pd.read_csv(tmp_path / "08x_seed_summary.csv")
    assert len(seed_summary) == 4 * len(qs._SEED_SUMMARY_METRICS)

    manifest = json.loads((tmp_path / "08x_run_manifest.json").read_text())
    assert manifest["search_budget_tier"] == "quick"
    assert manifest["trial_count_requested"] == 8
    assert manifest["trial_count_completed"] == 8
    assert manifest["trial_count_failed"] == 0
    assert manifest["provenance"]["quick_evidence_complete"] is True
    assert manifest["provenance"]["fold_assignment_sha256"]


def test_quick_search_real_dlinear_end_to_end(tmp_path):
    config = _config(
        _windowed_index(per_ticker=18),
        candidates=[
            {
                "family": "dlinear_only",
                "candidate_id": "c0",
                "model_config": {"max_epochs": 1, "batch_size": 8},
            }
        ],
        seed_list=(0,),
    )
    ledger = qs.run_quick_search(config, tmp_path)
    assert len(ledger) == 1  # 1 candidate x 1 fold x 1 seed
    validate_trial_ledger_frame(ledger)
    assert ledger.iloc[0]["fit_status"] in {"completed", "failed"}
    assert (tmp_path / "08x_fold_results.csv").exists()


def test_quick_search_class_collapse_guard(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0,),
    )
    monkeypatch.setattr(qs, "run_single_trial", _recorder(class0=0.0, class1=1.0))
    ledger = qs.run_quick_search(config, tmp_path)

    assert (ledger["fit_status"] == "failed").all()
    assert (ledger["failure_type"] == "class_collapse").all()
    # metrics are retained on the collapsed row (auditable)
    assert ledger.iloc[0]["class1_pred_rate"] == 1.0

    failure = pd.read_csv(tmp_path / "08x_failure_ledger.csv")
    assert len(failure) == 1
    assert failure.iloc[0]["failure_type"] == "class_collapse"
    # collapsed trials are excluded from the seed summary
    assert pd.read_csv(tmp_path / "08x_seed_summary.csv").empty


def test_quick_search_skips_single_class_fold(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(val_single_class=True),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0,),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("run_single_trial must not run on an unusable fold")

    monkeypatch.setattr(qs, "run_single_trial", _boom)
    ledger = qs.run_quick_search(config, tmp_path)

    assert (ledger["fit_status"] == "skipped").all()
    assert pd.read_csv(tmp_path / "08x_failure_ledger.csv").empty  # skip != failure
    manifest = json.loads((tmp_path / "08x_run_manifest.json").read_text())
    assert manifest["trial_count_skipped"] == 1
    assert manifest["trial_count_completed"] == 0


def test_quick_search_rejects_too_many_seeds(tmp_path):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0, 1, 2),
    )
    with pytest.raises(ValueError, match="seeds"):
        qs.run_quick_search(config, tmp_path)


def test_quick_search_rejects_grid_over_scientific_cap(tmp_path):
    config = _config(
        _windowed_index(),
        candidates=[
            {"family": "dlinear_only", "candidate_id": f"c{i}", "model_config": {}}
            for i in range(4)
        ],
        seed_list=(0, 1),
        scientific_cap=1,
    )
    with pytest.raises(ValueError, match="exceeds cap"):
        qs.run_quick_search(config, tmp_path)


def test_quick_search_dependency_preflight_missing_lightgbm(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[
            {"family": "last_step_lightgbm_control", "candidate_id": "c0", "model_config": {}}
        ],
        seed_list=(0,),
    )
    monkeypatch.setitem(sys.modules, "lightgbm", None)  # force import failure
    with pytest.raises(RuntimeError, match="lightgbm"):
        qs.run_quick_search(config, tmp_path)


def test_quick_search_writes_search_space_before_trials(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0,),
    )
    space_path = tmp_path / "08x_search_space.json"

    def _rec(X, y, ticker_ids, *, trial_id, **kwargs):
        # Preregistration (P1-3): the REAL search space exists before trial 0.
        assert space_path.exists()
        assert json.loads(space_path.read_text())["search_space_version"] == (
            qs.QUICK_SEARCH_VERSION
        )
        return _fake_row(trial_id=trial_id, **{
            k: kwargs[k] for k in (
                "candidate_family", "candidate_id", "config_hash", "fold_id", "seed"
            )
        })

    monkeypatch.setattr(qs, "run_single_trial", _rec)
    qs.run_quick_search(config, tmp_path)


def test_quick_search_fold_results_drift_fails(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0,),
    )
    # search_space.json present -> the skeleton writer is skipped, so the tampered
    # fold_results.csv survives to the consistency check.
    (tmp_path / "08x_search_space.json").write_text("{}", encoding="utf-8")
    tampered = pd.DataFrame(
        [
            {
                "fold_id": "rolling_origin_folds__0",
                "fold_scheme": "rolling_origin_folds",
                "split_index": 0,
                "train_inner_fit_n": 999,
                "train_inner_validation_n": 999,
                "purge_gap_k": 3,
                "embargo_gap_k": 0,
            }
        ],
        columns=list(qs.FOLD_RESULTS_COLUMNS),
    )
    tampered.to_csv(tmp_path / "08x_fold_results.csv", index=False)
    monkeypatch.setattr(qs, "run_single_trial", _recorder())
    with pytest.raises(ValueError, match="differs|changed"):
        qs.run_quick_search(config, tmp_path)


def test_quick_search_seed_summary_lcb_math(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0, 1),
    )
    monkeypatch.setattr(qs, "run_single_trial", _recorder())  # macro_f1 = 0.6 + 0.05*seed
    qs.run_quick_search(config, tmp_path)
    summary = pd.read_csv(tmp_path / "08x_seed_summary.csv")
    macro = summary[summary["metric"] == "macro_f1"].iloc[0]
    assert macro["seed_mean"] == pytest.approx(0.625)
    assert macro["seed_std"] == pytest.approx(0.0353553, rel=1e-4)
    assert macro["seed_lcb_95"] == pytest.approx(0.576, abs=1e-3)


def test_quick_search_single_seed_lcb_is_mean(tmp_path, monkeypatch):
    config = _config(
        _windowed_index(),
        candidates=[{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}],
        seed_list=(0,),
    )
    monkeypatch.setattr(qs, "run_single_trial", _recorder())
    qs.run_quick_search(config, tmp_path)
    summary = pd.read_csv(tmp_path / "08x_seed_summary.csv")
    macro = summary[summary["metric"] == "macro_f1"].iloc[0]
    assert macro["seed_mean"] == pytest.approx(0.6)
    assert macro["seed_lcb_95"] == pytest.approx(0.6)  # n=1 -> lcb = mean
    assert pd.isna(macro["seed_std"])


def test_dispatcher_routes_to_quick_search(tmp_path):
    from intraday_research.stages import deep_sequence_exploration as ex

    config = _config(
        _windowed_index(per_ticker=18),
        candidates=[
            {
                "family": "dlinear_only",
                "candidate_id": "c0",
                "model_config": {"max_epochs": 1, "batch_size": 8},
            }
        ],
        seed_list=(0,),
    )
    config["run_switches"] = {"RUN_08X_QUICK_SEARCH": True}
    ex.run_stage(config, output_dir=tmp_path)
    ledger = pd.read_csv(tmp_path / "08x_trial_ledger.csv")
    assert len(ledger) == 1


def test_quick_search_index_sha_binds_data_identity(tmp_path, monkeypatch):
    # impl-review P1: the row-identity hash must change when the underlying data
    # changes even if the positional fold layout (and row counts) is identical.
    monkeypatch.setattr(qs, "run_single_trial", _recorder())
    cands = [{"family": "dlinear_only", "candidate_id": "c0", "model_config": {}}]

    qs.run_quick_search(_config(_windowed_index(seed=0), candidates=cands, seed_list=(0,)), tmp_path)
    prov_a = json.loads((tmp_path / "08x_run_manifest.json").read_text())["provenance"]

    out_b = tmp_path / "b"
    out_b.mkdir()
    wi_flipped = _windowed_index(seed=0)
    wi_flipped["y"] = 1 - wi_flipped["y"]  # same shape/positions, new label identity
    qs.run_quick_search(_config(wi_flipped, candidates=cands, seed_list=(0,)), out_b)
    prov_b = json.loads((out_b / "08x_run_manifest.json").read_text())["provenance"]

    assert prov_a["train_inner_index_sha256"] != prov_b["train_inner_index_sha256"]
    assert prov_a["fold_assignment_sha256"] == prov_b["fold_assignment_sha256"]
    assert prov_a["data_source_sha256"] is None  # injected window index, no raw path


def test_dispatcher_rejects_quick_plus_smoke(tmp_path):
    from intraday_research.stages import deep_sequence_exploration as ex

    config = {
        "run_switches": {"RUN_08X_QUICK_SEARCH": True, "RUN_08X_SCHEMA_SMOKE": True}
    }
    with pytest.raises(ValueError, match="only one of"):
        ex.run_stage(config, output_dir=tmp_path)
