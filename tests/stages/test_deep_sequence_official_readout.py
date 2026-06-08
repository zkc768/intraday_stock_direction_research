import pandas as pd
import pytest

from intraday_research.contracts.deep_sequence_exploration import (
    check_08o_real_readout_completeness,
)
from intraday_research.stages.deep_sequence_official_readout import (
    build_08o_readout_frames,
    reject_holdout_test_filename,
    write_08o_readout_artifacts,
)


def _prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"seed": 1, "ticker": "AAA", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "AAA", "y_true": 1, "y_pred": 1},
            {"seed": 1, "ticker": "BBB", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "BBB", "y_true": 1, "y_pred": 1},
            {"seed": 2, "ticker": "AAA", "y_true": 0, "y_pred": 0},
            {"seed": 2, "ticker": "AAA", "y_true": 1, "y_pred": 0},
            {"seed": 2, "ticker": "BBB", "y_true": 0, "y_pred": 1},
            {"seed": 2, "ticker": "BBB", "y_true": 1, "y_pred": 1},
        ]
    )


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
            {"seed": 1, "ticker": "AAA", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "AAA", "y_true": 1, "y_pred": 1},
            {"seed": 1, "ticker": "BBB", "y_true": 0, "y_pred": 0},
            {"seed": 1, "ticker": "BBB", "y_true": 1, "y_pred": 0},
            {"seed": 1, "ticker": "BBB", "y_true": 0, "y_pred": 1},
            {"seed": 1, "ticker": "BBB", "y_true": 1, "y_pred": 1},
        ]
    )

    concentration = build_08o_readout_frames(rows)["08o_concentration_guardrails.csv"]

    positive_ticker_count = concentration.loc[
        concentration["guardrail"] == "positive_ticker_count",
        "value",
    ].iloc[0]
    assert positive_ticker_count == 1


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
