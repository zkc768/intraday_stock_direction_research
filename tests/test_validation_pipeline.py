from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from intraday_research.validation_pipeline import (
    build_walk_forward_fold_specs,
    build_validation_only_report,
    evaluate_lightgbm_last_step_adapter,
    evaluate_sklearn_logreg_last_step,
    load_ticker_csv,
    precheck_lightgbm_dependency,
    summarize_dummy_baseline,
    subsample_rows_uniformly,
)
from scripts.run_validation_only_pipeline_smoke import json_default


def make_daily_rows(ticker, start_date, n_days, bars_per_day=14):
    rows = []
    base = 100.0
    for day_index in range(n_days):
        day = pd.Timestamp(start_date) + pd.Timedelta(days=day_index)
        direction = 1.0 if day_index % 2 == 0 else -1.0
        for bar_index in range(bars_per_day):
            timestamp = day + pd.Timedelta(hours=9, minutes=30 + 5 * bar_index)
            close = base + day_index * 2.0 + direction * bar_index * 0.2
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": close - 0.05,
                    "high": close + 0.10,
                    "low": close - 0.10,
                    "close": close,
                    "volume": 1000 + day_index * 10 + bar_index,
                }
            )
    frame = pd.DataFrame(rows)
    frame["ticker"] = ticker
    return frame


def write_ticker_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.drop(columns=["ticker"]).to_csv(path, index=False)


def test_load_ticker_csv_reports_missing_raw_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing raw ticker file"):
        load_ticker_csv("AAA", data_dir=tmp_path)


def test_subsample_rows_uniformly_uses_uniform_index_stride():
    x_values = np.arange(10).reshape(10, 1)
    y_values = np.arange(10)

    x_sampled, y_sampled = subsample_rows_uniformly(x_values, y_values, max_rows=4)

    assert x_sampled[:, 0].tolist() == [0, 3, 6, 9]
    assert y_sampled.tolist() == [0, 3, 6, 9]


def test_json_default_preserves_numpy_bool_as_json_bool():
    assert json_default(np.bool_(True)) is True
    assert json_default(np.bool_(False)) is False


def test_validation_only_report_uses_train_validation_without_holdout_windows(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = pd.concat(
        [
            make_daily_rows("AAA", "2020-01-01", 3),
            make_daily_rows("AAA", "2020-01-10", 3),
            make_daily_rows("AAA", "2020-01-20", 1),
        ],
        ignore_index=True,
    )
    write_ticker_csv(data_dir / "AAA.csv", frame)

    report = build_validation_only_report(
        data_dir=data_dir,
        tickers=("AAA",),
        feature_columns=(
            "log_return",
            "close_to_open_return",
            "high_low_range",
            "time_of_day_sin",
            "time_of_day_cos",
        ),
        splits={
            "train": ("2020-01-01", "2020-01-10"),
            "validation": ("2020-01-10", "2020-01-20"),
            "closed_holdout_boundary_only": ("2020-01-20", "2020-01-21"),
        },
        horizon_k=2,
        threshold_bps=0.0,
        window_size=3,
        dummy_seeds=(41, 42),
        walk_forward_folds=2,
    )

    metadata = report["metadata"]
    assert metadata["scope"] == "validation_only"
    assert metadata["transformed_splits"] == ["train", "validation"]
    assert metadata["diagnostic_row_subsample"]["strategy"].startswith("uniform_index")
    assert "not_transformed_not_windowed_not_scored" in metadata["closed_holdout_policy"]

    balance = report["window_class_balance"]
    assert {row["split"] for row in balance} == {"train", "validation"}
    assert all(row["scope"].endswith("_window_class_balance_diagnostic") for row in balance)
    assert all(row["shape"][1:] == [3, 5] for row in balance)
    assert all(row["n_windows"] > 0 for row in balance)
    assert report["dummy_baseline_summary"]["n"] == sum(
        row["n_windows"] for row in balance if row["split"] == "validation"
    )
    assert np.isfinite(report["dummy_baseline_summary"]["macro_f1_mean"])

    mi_rows = report["mutual_information_diagnostic"]
    assert {row["split"] for row in mi_rows} == {"train", "validation"}
    assert all("not_selection" in row["scope"] for row in mi_rows)

    adapter = report["model_adapter_precheck"]
    assert adapter["lightgbm"]["adapter"] == "lightgbm"
    assert adapter["dependency_free_diagnostic"]["scope"].endswith("not_selection")
    assert adapter["dependency_free_diagnostic"]["class_weight"] == "balanced"
    assert adapter["dependency_free_diagnostic"]["feature_view"] == "last_step_only_not_sequence"
    assert not adapter["dependency_free_diagnostic"]["uses_full_window_sequence"]
    assert report["metadata"]["diagnostic_model_views"]["sklearn_logreg_last_step"][
        "class_weight"
    ] == "balanced"
    assert (
        report["metadata"]["walk_forward_contract_policy"]
        == "date_range_contract_only_no_model_scores"
    )

    ablation = report["feature_ablation_diagnostic"]
    assert len(ablation) == 6
    assert ablation[0]["ablation"] == "all_features"
    assert {row["ablation"] for row in ablation[1:]} == {"leave_one_feature_out"}

    walk_forward = report["walk_forward_contract"]
    assert {row["ticker"] for row in walk_forward} == {"AAA"}
    assert {row["fold"] for row in walk_forward} == {1, 2}
    assert all(row["chronological"] for row in walk_forward)
    assert all(row["contract_only"] for row in walk_forward)
    assert all(not row["model_scores_available"] for row in walk_forward)
    assert all(row["score_fields"] == "not_computed" for row in walk_forward)
    assert all(
        row["scope"] == "train_validation_only_walk_forward_contract_no_scores"
        for row in walk_forward
    )


def test_dummy_summary_uses_sample_std_ddof_1():
    dummy_rows = pd.DataFrame(
        {
            "macro_f1": [0.2, 0.8],
            "balanced_accuracy": [0.3, 0.9],
            "accuracy": [0.4, 0.6],
            "validation_n": [10, 10],
        }
    )

    summary = summarize_dummy_baseline(dummy_rows)

    assert summary["macro_f1_std"] == pytest.approx(dummy_rows["macro_f1"].std(ddof=1))
    assert summary["balanced_accuracy_std"] == pytest.approx(
        dummy_rows["balanced_accuracy"].std(ddof=1)
    )
    assert summary["std_estimator"] == "sample_std_ddof_1"


def test_sklearn_logreg_reports_convergence_warning_without_crashing(monkeypatch):
    class WarningLogisticRegression:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, x_values, y_values):
            warnings.warn("iteration limit hit", ConvergenceWarning)
            return self

        def predict(self, x_values):
            return np.zeros(len(x_values), dtype=int)

    windows = {
        "AAA": {
            "train": {"X": np.ones((4, 2, 2)), "y": np.array([0, 1, 0, 1])},
            "validation": {"X": np.ones((3, 2, 2)), "y": np.array([0, 1, 1])},
        }
    }
    monkeypatch.setattr(
        "intraday_research.validation_pipeline.LogisticRegression",
        WarningLogisticRegression,
    )

    result = evaluate_sklearn_logreg_last_step(windows)

    assert result["class_weight"] == "balanced"
    assert not result["converged"]
    assert result["fit_status"] == "convergence_warning"
    assert "iteration limit hit" in result["fit_error"]
    assert result["feature_view"] == "last_step_only_not_sequence"


def test_lightgbm_unavailable_result_keeps_metric_schema(monkeypatch):
    monkeypatch.setattr(
        "intraday_research.validation_pipeline.importlib.util.find_spec",
        lambda name: None if name == "lightgbm" else object(),
    )

    result = evaluate_lightgbm_last_step_adapter({})

    assert result["adapter"] == "lightgbm"
    assert not result["available"]
    assert result["model"] is None
    assert result["macro_f1"] is None
    assert result["balanced_accuracy"] is None
    assert result["accuracy"] is None
    assert result["feature_view"] == "last_step_only_not_sequence"


def test_walk_forward_contract_rejects_non_positive_fold_count():
    frame = make_daily_rows("AAA", "2020-01-01", 3)
    frame["split"] = "train"

    with pytest.raises(ValueError, match="n_folds"):
        build_walk_forward_fold_specs({"AAA": frame}, n_folds=0)


def test_lightgbm_precheck_reports_availability_or_exact_dependency_blocker():
    result = precheck_lightgbm_dependency()

    assert result["adapter"] == "lightgbm"
    assert result["scope"] == "validation_only_adapter_precheck"
    if not result["available"]:
        assert result["blocker"] == "Missing Python dependency: lightgbm"
