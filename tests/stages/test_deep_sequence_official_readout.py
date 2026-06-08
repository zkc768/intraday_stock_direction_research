import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    check_08o_real_readout_completeness,
)
from intraday_research.stages.deep_sequence_official_readout import (
    build_08o_readout_frames,
    preflight_08o_prediction_rows,
    reject_holdout_test_filename,
    resolve_08o_readout_inputs,
    write_08o_readout_artifacts,
    write_08o_run_manifest,
)


def _prediction_row(seed: int, ticker: str, row_id: str, y_true: int, y_pred: int) -> dict:
    return {
        "seed": seed,
        "ticker": ticker,
        "candidate_id": "candidate_a",
        "official_validation_row_id": row_id,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _prediction_row(1, "AAA", "row_1", 0, 0),
            _prediction_row(1, "AAA", "row_2", 1, 1),
            _prediction_row(1, "BBB", "row_3", 0, 0),
            _prediction_row(1, "BBB", "row_4", 1, 1),
            _prediction_row(2, "AAA", "row_1", 0, 0),
            _prediction_row(2, "AAA", "row_2", 1, 0),
            _prediction_row(2, "BBB", "row_3", 0, 1),
            _prediction_row(2, "BBB", "row_4", 1, 1),
        ]
    )


def _freeze_record() -> dict:
    return {
        "stage": "08F",
        "scope": "diagnostic",
        "primary_candidate_id": "candidate_a",
        "fallback_candidate_id": "candidate_b",
        "fallback_activation_rule": "Activate fallback only if primary fails before scoring official validation.",
        "config_hash": "deadbeef" * 4,
        "architecture_family": "dlinear_only",
        "frozen_architecture_params": {"hidden_size": 8},
        "frozen_loss": "cross_entropy",
        "frozen_hpo_method": "random_search",
        "frozen_seed_list": [1, 2],
        "frozen_metric_list": ["macro_f1", "balanced_accuracy"],
        "frozen_wording_rule": "per AGENTS.md section 4.2.5a",
        "paper_safe_score": 0.1,
        "official_validation_used_for_selection": False,
        "holdout_test_authorized": False,
    }


def test_build_08o_readout_frames_computes_required_artifacts():
    frames = build_08o_readout_frames(_prediction_rows())

    assert set(frames) == {
        "08o_validation_readout.csv",
        "08o_validation_per_ticker.csv",
        "08o_seed_summary.csv",
        "08o_same_row_baselines.csv",
        "08o_concentration_guardrails.csv",
        "08o_failure_rows.csv",
    }
    readout = frames["08o_validation_readout.csv"]
    assert readout["seed"].tolist() == [1, 2]
    assert readout.loc[0, "macro_f1"] == pytest.approx(1.0)
    assert readout.loc[0, "delta_macro_f1_vs_stratified_dummy_same_rows"] == pytest.approx(0.5)
    assert readout.loc[1, "macro_f1"] == pytest.approx(0.5)
    assert readout.loc[1, "delta_macro_f1_vs_stratified_dummy_same_rows"] == pytest.approx(0.0)

    baselines = frames["08o_same_row_baselines.csv"]
    assert baselines.loc[0, "baseline"] == "stratified_dummy_same_rows"
    assert baselines.loc[0, "macro_f1_mean"] == pytest.approx(0.5)
    assert baselines.loc[0, "macro_f1_std"] == pytest.approx(0.0)

    seed_summary = frames["08o_seed_summary.csv"]
    assert "delta_macro_f1_vs_stratified_dummy_same_rows" in set(seed_summary["metric"])
    assert frames["08o_failure_rows.csv"].empty


def test_write_08o_readout_artifacts_passes_real_completeness_gate(tmp_path):
    verdict = write_08o_readout_artifacts(tmp_path, _prediction_rows())

    assert verdict["is_real_readout"] is True
    assert verdict["missing_artifacts"] == []
    assert verdict["empty_artifacts"] == []
    assert verdict["schema_drift"] == []
    assert check_08o_real_readout_completeness(tmp_path)["is_real_readout"] is True


def test_positive_ticker_count_excludes_zero_delta_tickers():
    rows = pd.DataFrame(
        [
            _prediction_row(1, "AAA", "row_1", 0, 0),
            _prediction_row(1, "AAA", "row_2", 1, 1),
            _prediction_row(1, "BBB", "row_3", 0, 0),
            _prediction_row(1, "BBB", "row_4", 1, 0),
            _prediction_row(1, "BBB", "row_5", 0, 1),
            _prediction_row(1, "BBB", "row_6", 1, 1),
        ]
    )

    concentration = build_08o_readout_frames(rows)["08o_concentration_guardrails.csv"]

    positive_ticker_count = concentration.loc[
        concentration["guardrail"] == "positive_ticker_count",
        "value",
    ].iloc[0]
    assert positive_ticker_count == 1


def test_prediction_preflight_records_provenance():
    provenance = preflight_08o_prediction_rows(
        _prediction_rows(),
        freeze_record=_freeze_record(),
        expected_tickers=("AAA", "BBB"),
    )

    assert provenance["candidate_id"] == "candidate_a"
    assert provenance["prediction_row_count"] == 8
    assert provenance["official_validation_row_id_count"] == 4
    assert provenance["seeds"] == [1, 2]
    assert provenance["tickers"] == ["AAA", "BBB"]
    assert provenance["same_official_rows_for_each_seed"] is True
    assert set(provenance["row_id_set_sha256_by_seed"]) == {1, 2}


def test_prediction_preflight_rejects_candidate_mismatch():
    rows = _prediction_rows()
    rows.loc[0, "candidate_id"] = "other_candidate"

    with pytest.raises(ValueError, match="exactly one candidate_id"):
        preflight_08o_prediction_rows(rows, freeze_record=_freeze_record())


def test_prediction_preflight_rejects_seed_mismatch():
    freeze_record = _freeze_record()
    freeze_record["frozen_seed_list"] = [1, 2, 3]

    with pytest.raises(ValueError, match="frozen_seed_list"):
        preflight_08o_prediction_rows(_prediction_rows(), freeze_record=freeze_record)


def test_prediction_preflight_rejects_row_id_drift_across_seeds():
    rows = _prediction_rows()
    rows.loc[(rows["seed"] == 2) & (rows["official_validation_row_id"] == "row_4"), "official_validation_row_id"] = "row_extra"

    with pytest.raises(ValueError, match="same official_validation_row_id"):
        preflight_08o_prediction_rows(rows, freeze_record=_freeze_record())


def test_write_08o_run_manifest_uses_current_completeness(tmp_path):
    write_08o_readout_artifacts(
        tmp_path,
        _prediction_rows(),
        primary_candidate_id="candidate_a",
        expected_seeds=(1, 2),
        expected_tickers=("AAA", "BBB"),
    )

    manifest = write_08o_run_manifest(
        tmp_path,
        freeze_record=_freeze_record(),
        static_input_provenance={
            "freeze_record_sha256": "f" * 64,
            "freeze_record_path": "08f_candidate_freeze_record.json",
            "decision_record_sha256": "d" * 64,
            "decision_record_path": "08o_decision_record.json",
        },
        prediction_provenance={
            "candidate_id": "candidate_a",
            "prediction_row_count": 8,
            "official_validation_row_id_count": 4,
            "seeds": [1, 2],
            "tickers": ["AAA", "BBB"],
            "same_official_rows_for_each_seed": True,
        },
        constants={
            "improvement_threshold_positive_ticker_count_min": 2,
            "tier_escalation_medium_to_aggressive_seed_std_max": 1.0,
        },
        readout_started_at_utc="2026-06-08T10:00:00Z",
    )

    assert manifest["stage"] == "08O"
    assert manifest["schema_only_stub"] is False
    assert manifest["same_row_dummy_present"] is True
    assert manifest["per_ticker_present"] is True
    assert manifest["seed_summary_present"] is True
    assert manifest["allowed_wording_bucket"] in {"improvement", "weak_mixed"}


def test_08o_readout_rejects_missing_required_prediction_column():
    rows = _prediction_rows().drop(columns=["y_pred"])

    with pytest.raises(ValueError, match="missing columns"):
        build_08o_readout_frames(rows)


def test_08o_readout_rejects_holdout_test_selection_columns():
    rows = _prediction_rows()
    rows["holdout_score"] = 0.0

    with pytest.raises(ValueError, match="forbidden columns"):
        build_08o_readout_frames(rows)


def test_08o_readout_rejects_single_class_seed():
    rows = _prediction_rows()
    rows.loc[rows["seed"] == 2, "y_true"] = 1

    with pytest.raises(ValueError, match="seed 2 y_true must contain both classes"):
        build_08o_readout_frames(rows)


def test_reject_holdout_test_filename_checks_only_filename(tmp_path):
    reject_holdout_test_filename(
        tmp_path / "official_validation_predictions.csv",
        field_name="official predictions",
    )
    with pytest.raises(ValueError, match="holdout/test tokens"):
        reject_holdout_test_filename(
            tmp_path / "holdout_predictions.csv",
            field_name="official predictions",
        )


def test_resolve_08o_readout_inputs_rejects_bad_ledger_config_type():
    config = {
        "inputs": {
            "official_validation_predictions_csv": "official_validation_predictions.csv",
            "08o_decision_record": "08o_decision_record.json",
            "08f_candidate_freeze_record": "08f_candidate_freeze_record.json",
        },
        "policy": {"validation_budget_ledger": "ledger.csv"},
    }

    with pytest.raises(ValueError, match="validation_budget_ledger"):
        resolve_08o_readout_inputs(config)
