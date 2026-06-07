import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import notebook06_contract as c


SAMPLE_IDS = np.asarray(
    [
        "CSCO|2020-01-02 09:35:00|row00000000",
        "CSCO|2020-01-02 09:40:00|row00000001",
        "JPM|2020-01-02 09:35:00|row00000002",
        "JPM|2020-01-02 09:40:00|row00000003",
        "MSFT|2020-01-02 09:35:00|row00000004",
        "MSFT|2020-01-02 09:40:00|row00000005",
    ],
    dtype=str,
)
TICKERS = np.asarray(["CSCO", "CSCO", "JPM", "JPM", "MSFT", "MSFT"], dtype=str)
TIMESTAMPS = np.asarray(
    [
        "2020-01-02 09:35:00",
        "2020-01-02 09:40:00",
        "2020-01-02 09:35:00",
        "2020-01-02 09:40:00",
        "2020-01-02 09:35:00",
        "2020-01-02 09:40:00",
    ],
    dtype=str,
)
Y_TRUE = np.asarray([0, 1, 1, 0, 1, 0], dtype=int)
Y_PRED = np.asarray([0, 1, 0, 0, 1, 1], dtype=int)
PROB_UP = np.asarray([0.12, 0.91, 0.48, 0.31, 0.74, 0.63], dtype=float)
CONFIDENCE = np.maximum(PROB_UP, 1.0 - PROB_UP)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_prediction(path: Path, *, sample_ids=SAMPLE_IDS, omit_key=None, confidence=CONFIDENCE) -> None:
    payload = {
        "y_true": Y_TRUE,
        "y_pred": Y_PRED,
        "prob_up": PROB_UP,
        "validation_sample_id": np.asarray(sample_ids, dtype=str),
        "ticker": TICKERS,
        "timestamp": TIMESTAMPS,
        "confidence": np.asarray(confidence, dtype=float),
    }
    if omit_key is not None:
        payload.pop(omit_key)
    np.savez_compressed(path, **payload)


def _base_records(prediction_artifact: str, profile_id: str = "lightgbm_trial_001") -> dict:
    sample_hash = c._stable_hash(SAMPLE_IDS)
    return {
        "stage": "05D_official_validation_confirmation",
        "candidate_id": "h03_bps1p5_price_volume_time_w20",
        "profile_id": profile_id,
        "profile_role": "train_inner_winner",
        "seed": 260501,
        "ticker_or_pooled": "pooled",
        "train_n": 120,
        "validation_n": len(SAMPLE_IDS),
        "train_class0_n": 72,
        "train_class1_n": 48,
        "train_positive_rate": 48 / 120,
        "validation_sample_id_hash": sample_hash,
        "sample_id_mismatch_count": 0,
        "prediction_artifact": prediction_artifact,
        "macro_f1": 0.66,
        "balanced_accuracy": 0.67,
        "accuracy": 0.67,
        "stratified_dummy_macro_f1": 0.50,
        "delta_macro_f1_vs_stratified_dummy": 0.16,
        "always_up_dummy_macro_f1": 0.40,
        "delta_macro_f1_vs_always_up_dummy": 0.26,
        "scope": "validation_only",
        "official_validation_used_for_selection": False,
        "selected_profile_source": "train_inner_hpo",
    }


def make_bundle(tmp_path: Path, *, two_seed: bool = True) -> Path:
    root = tmp_path / "notebook05_bundle"
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True)

    _write_json(
        root / "notebook05_entry_decision.json",
        {
            "scope": "validation_only",
            "holdout_test_authorized": False,
            "selective_threshold_selected": False,
        },
    )
    _write_json(
        root / "notebook05_decision_record.json",
        {
            "scope": "validation_only",
            "holdout_test_authorized": False,
            "selective_threshold_selected": False,
            "selected_profile_id": "lightgbm_trial_001",
            "selected_profile_source": "train_inner_hpo",
        },
    )
    _write_json(
        root / "notebook05_run_manifest.json",
        {
            "scope": "validation_only",
            "holdout_test_authorized": False,
            "selective_threshold_selected": False,
        },
    )

    first_path = prediction_dir / "primary_seed1.npz"
    _write_prediction(first_path)
    rows = [_base_records("predictions/primary_seed1.npz")]
    if two_seed:
        second_path = prediction_dir / "primary_seed2.npz"
        _write_prediction(second_path)
        second = _base_records("predictions/primary_seed2.npz")
        second["seed"] = 260502
        rows.append(second)
    pd.DataFrame(rows).to_csv(root / "notebook05_official_validation_pooled.csv", index=False)
    pd.DataFrame([{**rows[0], "ticker_or_pooled": "CSCO", "validation_n": 2}]).to_csv(
        root / "notebook05_official_validation_per_ticker.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "profile_id": "lightgbm_trial_001",
                "validation_sample_id_hash": c._stable_hash(SAMPLE_IDS),
                "sample_id_mismatch_count": 0,
                "scope": "validation_only",
            }
        ]
    ).to_csv(root / "notebook05_official_validation_summary.csv", index=False)
    return root


def test_contract_passes_on_minimal_valid_bundle(tmp_path):
    root = make_bundle(tmp_path)

    result = c.assert_notebook06_artifact_contract(root)

    assert result["contract_passed"] is True
    assert result["primary_profile_id"] == "lightgbm_trial_001"
    assert result["prediction_artifact_count"] == 2
    assert result["sample_id_hash"] == c._stable_hash(SAMPLE_IDS)


def test_missing_prob_up_fails_with_artifact_path(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "primary_seed1.npz"
    _write_prediction(artifact, omit_key="prob_up")

    with pytest.raises(ValueError, match=re.escape(str(artifact))):
        c.assert_notebook06_artifact_contract(root)


def test_missing_validation_sample_id_fails_with_artifact_path(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "primary_seed1.npz"
    _write_prediction(artifact, omit_key="validation_sample_id")

    with pytest.raises(ValueError, match=re.escape(str(artifact))):
        c.assert_notebook06_artifact_contract(root)


def test_duplicated_validation_sample_id_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "primary_seed1.npz"
    duplicated = SAMPLE_IDS.copy()
    duplicated[1] = duplicated[0]
    _write_prediction(artifact, sample_ids=duplicated)

    with pytest.raises(ValueError, match="duplicated validation_sample_id"):
        c.assert_notebook06_artifact_contract(root)


def test_differing_sample_id_order_across_seed_artifacts_fails(tmp_path):
    root = make_bundle(tmp_path)
    artifact = root / "predictions" / "primary_seed2.npz"
    _write_prediction(artifact, sample_ids=SAMPLE_IDS[::-1])
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    pooled.loc[pooled["seed"] == 260502, "validation_sample_id_hash"] = c._stable_hash(SAMPLE_IDS[::-1])
    pooled.to_csv(pooled_path, index=False)

    with pytest.raises(ValueError, match="validation_sample_id order differs across seeds"):
        c.assert_notebook06_artifact_contract(root)


def test_differing_sample_id_hash_against_pooled_row_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    pooled.loc[0, "validation_sample_id_hash"] = "bad_hash"
    pooled.to_csv(pooled_path, index=False)

    with pytest.raises(ValueError, match="validation_sample_id_hash differs"):
        c.assert_notebook06_artifact_contract(root)


def test_holdout_authorized_true_fails_from_decision_or_run_manifest(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    decision_path = root / "notebook05_decision_record.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["holdout_test_authorized"] = True
    _write_json(decision_path, decision)

    with pytest.raises(ValueError, match="holdout_test_authorized is not false"):
        c.assert_notebook06_artifact_contract(root)


def test_selective_threshold_selected_true_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    run_path = root / "notebook05_run_manifest.json"
    run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
    run_manifest["selective_threshold_selected"] = True
    _write_json(run_path, run_manifest)

    with pytest.raises(ValueError, match="selective_threshold_selected is not false"):
        c.assert_notebook06_artifact_contract(root)


def test_prediction_path_containing_holdout_or_test_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    pooled.loc[0, "prediction_artifact"] = "predictions/holdout_primary_seed1.npz"
    pooled.to_csv(pooled_path, index=False)

    with pytest.raises(ValueError, match="holdout/test"):
        c.assert_notebook06_artifact_contract(root)


def test_downstream_primary_profile_id_is_preferred(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "downstream_seed1.npz"
    _write_prediction(artifact)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    downstream = _base_records("predictions/downstream_seed1.npz", profile_id="default_lgbm_04")
    pooled = pd.concat([pooled, pd.DataFrame([downstream])], ignore_index=True)
    pooled.to_csv(pooled_path, index=False)
    decision_path = root / "notebook05_decision_record.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["downstream_primary_profile_id"] = "default_lgbm_04"
    _write_json(decision_path, decision)

    result = c.assert_notebook06_artifact_contract(root)

    assert result["primary_profile_id"] == "default_lgbm_04"


def test_retain_default_status_resolves_default_profile(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "default_seed1.npz"
    _write_prediction(artifact)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    default_row = _base_records("predictions/default_seed1.npz", profile_id="default_lgbm_04")
    pooled = pd.concat([pooled, pd.DataFrame([default_row])], ignore_index=True)
    pooled.to_csv(pooled_path, index=False)
    decision_path = root / "notebook05_decision_record.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["official_validation_status"] = "retain_default_lgbm_04"
    _write_json(decision_path, decision)

    result = c.assert_notebook06_artifact_contract(root)

    assert result["primary_profile_id"] == "default_lgbm_04"


def test_official_validation_best_source_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    decision_path = root / "notebook05_decision_record.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["selected_profile_source"] = "official_validation_best"
    _write_json(decision_path, decision)

    with pytest.raises(ValueError, match="official_validation_best"):
        c.assert_notebook06_artifact_contract(root)


def test_missing_train_positive_rate_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path).drop(columns=["train_positive_rate"])
    pooled.to_csv(pooled_path, index=False)

    with pytest.raises(ValueError, match="train_positive_rate"):
        c.assert_notebook06_artifact_contract(root)


def test_train_positive_rate_mismatch_fails(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    pooled_path = root / "notebook05_official_validation_pooled.csv"
    pooled = pd.read_csv(pooled_path)
    pooled.loc[0, "train_positive_rate"] = 0.99
    pooled.to_csv(pooled_path, index=False)

    with pytest.raises(ValueError, match="train_positive_rate inconsistent"):
        c.assert_notebook06_artifact_contract(root)


def test_confidence_must_equal_max_probability(tmp_path):
    root = make_bundle(tmp_path, two_seed=False)
    artifact = root / "predictions" / "primary_seed1.npz"
    _write_prediction(artifact, confidence=np.zeros_like(CONFIDENCE))

    with pytest.raises(ValueError, match="confidence differs"):
        c.assert_notebook06_artifact_contract(root)


def test_build_canonical_prediction_frame_uses_prob_up_column(tmp_path):
    payload = c.load_notebook06_prediction_artifact(
        make_bundle(tmp_path, two_seed=False) / "predictions" / "primary_seed1.npz"
    )
    frame = c.build_canonical_prediction_frame(payload, {"profile_id": "p1", "profile_role": "winner", "seed": 1})

    assert "prob_up" in frame.columns
    assert "y_prob_up" in frame.columns
    assert np.allclose(frame["prob_up"], frame["y_prob_up"])
    assert frame["correct"].tolist() == (Y_TRUE == Y_PRED).astype(int).tolist()


def test_selective_retained_indices_uses_confidence_then_sample_id_tie_break():
    confidence = np.asarray([0.8, 0.9, 0.9, 0.7])
    sample_ids = np.asarray(["b", "z", "a", "c"])

    retained = c.selective_retained_indices(confidence, sample_ids, 0.5)

    assert retained.tolist() == [2, 1]


def test_same_row_stratified_dummy_is_deterministic_and_matches_train_prior():
    first = c.same_row_stratified_dummy_predict(7000, 3000, 10000, seed=260606)
    second = c.same_row_stratified_dummy_predict(7000, 3000, 10000, seed=260606)

    assert np.array_equal(first, second)
    assert abs(float(first.mean()) - 0.30) < 0.02


def test_ticker_stratified_random_abstention_preserves_counts_and_short_circuits_full_coverage():
    tickers = np.asarray(["A", "A", "A", "B", "B"])

    masks = c.ticker_stratified_random_abstention({"A": 2, "B": 1}, tickers, 260606, 5)
    assert masks.shape == (5, 5)
    for mask in masks:
        assert int(mask[tickers == "A"].sum()) == 2
        assert int(mask[tickers == "B"].sum()) == 1

    full = c.ticker_stratified_random_abstention({"A": 3, "B": 2}, tickers, 260606, 100)
    assert full.shape == (100, 5)
    assert full.all()


def test_calibration_bins_write_empty_uniform_bins_and_ece_skips_them():
    rows = c.calibration_bins(np.asarray([0.05, 0.95]), np.asarray([0, 1]), n_bins=4, strategy="uniform")

    assert [row["bin_count"] for row in rows] == [1, 0, 0, 1]
    assert c.ece_from_bins(rows) == pytest.approx(0.05)


def test_risk_coverage_curve_and_aurc_are_finite():
    curve = c.risk_coverage_curve(Y_TRUE, Y_PRED, CONFIDENCE)

    assert curve.iloc[-1]["coverage"] == pytest.approx(1.0)
    assert math.isfinite(c.aurc_from_curve(curve))


def test_aggregate_across_seeds_uses_dynamic_lcb_and_positive_counts():
    frame = pd.DataFrame(
        {
            "profile_id": ["p1"] * 5,
            "coverage_target": [0.8] * 5,
            "seed": [1, 2, 3, 4, 5],
            "delta_macro_f1_vs_random_abstention": [0.01, 0.02, 0.03, -0.01, 0.04],
        }
    )

    aggregated = c.aggregate_across_seeds(frame, ["delta_macro_f1_vs_random_abstention"])

    assert aggregated.loc[0, "seed_count"] == 5
    assert aggregated.loc[0, "delta_macro_f1_vs_random_abstention_positive_seed_count"] == 4
    assert "delta_macro_f1_vs_random_abstention_lcb95_one_sided" in aggregated.columns


def test_evaluate_decision_outcome_uses_strict_priority():
    aggregated = pd.DataFrame(
        {
            "coverage_target": [0.9, 0.8, 0.7, 0.6],
            "delta_macro_f1_vs_random_abstention_mean": [0.02, 0.02, 0.02, 0.02],
            "delta_macro_f1_vs_random_abstention_positive_seed_count": [5, 5, 5, 5],
        }
    )

    decision = c.evaluate_decision_outcome(
        aggregated,
        pd.DataFrame({"guardrail_pass": [True, True, True, True]}),
        {
            "MIN_DECISION_DELTA_MACRO_F1": 0.005,
            "MIN_POSITIVE_SEED_COUNT": 4,
            "MIN_POSITIVE_DECISION_COVERAGE_COUNT": 4,
        },
    )

    assert decision["decision"] == "promote_selective_no_trade_for_validation_only_reporting"
