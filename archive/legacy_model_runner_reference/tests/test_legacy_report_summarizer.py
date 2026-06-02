import json

import pandas as pd
import pytest

from scripts.local_runner_reference import summarize_runs


def test_summarizer_writes_protocol_scoped_tables(tmp_path):
    run_dir = _write_run(
        tmp_path / "run_a",
        run_id="run_a",
        label_mode="legacy_binary",
        label_semantics="canonical_phase1_full_binary",
        zero_return_policy="class_0_non_up",
        no_trade_band_enabled=False,
        neutral_policy="not_applicable",
        threshold_bps=0.0,
    )
    output_dir = tmp_path / "report"

    outputs = summarize_runs.summarize_run_dirs([run_dir], output_dir)

    run_summary = outputs["run_summary"].iloc[0]
    assert run_summary["run_id"] == "run_a"
    assert run_summary["label_semantics"] == "canonical_phase1_full_binary"
    assert run_summary["zero_return_policy"] == "class_0_non_up"
    assert run_summary["best_pooled_model"] == "tcn"
    assert run_summary["best_pooled_delta_macro_f1_vs_dummy"] == pytest.approx(-0.002)
    assert run_summary["best_pooled_test_report_model"] == "tcn"
    assert run_summary["best_pooled_test_delta_macro_f1_vs_dummy"] == pytest.approx(-0.002)
    assert run_summary["gate_split"] == "validation"
    assert run_summary["model_expansion_gate"] == "blocked_delta_lt_0.01"
    assert run_summary["model_expansion_gate_reason"] == "validation_delta_available"

    pooled = outputs["pooled_by_model"]
    tcn_row = pooled.loc[pooled["model_name"] == "tcn"].iloc[0]
    assert tcn_row["report_split"] == "final_test_exploratory"
    assert tcn_row["dummy_stratified_macro_f1_mean"] == pytest.approx(0.5)
    assert tcn_row["val_delta_macro_f1_vs_dummy_mean"] == pytest.approx(-0.002)
    assert tcn_row["dummy_stratified_balanced_accuracy_mean"] == pytest.approx(0.51)
    assert tcn_row["macro_f1_mean"] == pytest.approx(0.498)
    assert tcn_row["seeds"] == "42,43"

    coverage = outputs["coverage_by_ticker"]
    pooled_coverage = coverage.loc[coverage["ticker"] == "pooled"].iloc[0]
    assert pooled_coverage["retained_pct"] == pytest.approx(0.847)
    assert pooled_coverage["label_n_zero_return"] == 2
    assert pooled_coverage["n_val_windows"] == 400
    assert pooled_coverage["val_up_pct"] == pytest.approx(0.5)

    assert (output_dir / "run_summary.csv").exists()
    assert (output_dir / "pooled_by_model.csv").exists()
    assert (output_dir / "by_model_ticker.csv").exists()
    assert (output_dir / "coverage_by_ticker.csv").exists()
    report_text = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "canonical_phase1_full_binary" in report_text
    assert "class_0_non_up" in report_text
    assert "final_test_exploratory" in report_text
    assert "blocked_delta_lt_0.01" in report_text


def test_gate_uses_validation_delta_not_test_delta(tmp_path):
    run_dir = _write_run(
        tmp_path / "run_a",
        run_id="run_a",
        label_mode="legacy_binary",
        label_semantics="canonical_phase1_full_binary",
        zero_return_policy="class_0_non_up",
        no_trade_band_enabled=False,
        neutral_policy="not_applicable",
        threshold_bps=0.0,
        val_deltas_by_model={"lstm": [0.020, 0.018], "tcn": [-0.003, -0.001]},
    )

    outputs = summarize_runs.summarize_run_dirs([run_dir], tmp_path / "report")

    run_summary = outputs["run_summary"].iloc[0]
    assert run_summary["best_pooled_model"] == "lstm"
    assert run_summary["best_pooled_delta_macro_f1_vs_dummy"] == pytest.approx(0.019)
    assert run_summary["best_pooled_test_report_model"] == "tcn"
    assert run_summary["best_pooled_test_delta_macro_f1_vs_dummy"] == pytest.approx(-0.002)
    assert run_summary["model_expansion_gate"] == "review_required_delta_ge_0.01"


def test_gate_closes_when_validation_delta_is_missing(tmp_path):
    run_dir = _write_run(
        tmp_path / "older_run",
        run_id="older_run",
        label_mode="legacy_binary",
        label_semantics="canonical_phase1_full_binary",
        zero_return_policy="class_0_non_up",
        no_trade_band_enabled=False,
        neutral_policy="not_applicable",
        threshold_bps=0.0,
    )
    results = pd.read_csv(run_dir / "results.csv")
    results = results.drop(
        columns=[
            "best_val_macro_f1",
            "val_dummy_stratified_macro_f1_mean",
            "val_delta_macro_f1_vs_dummy",
        ]
    )
    results.to_csv(run_dir / "results.csv", index=False)

    outputs = summarize_runs.summarize_run_dirs([run_dir], tmp_path / "report")

    run_summary = outputs["run_summary"].iloc[0]
    assert pd.isna(run_summary["best_pooled_model"])
    assert pd.isna(run_summary["best_pooled_delta_macro_f1_vs_dummy"])
    assert run_summary["best_pooled_test_report_model"] == "tcn"
    assert run_summary["model_expansion_gate"] == "closed_insufficient_validation_evidence"
    assert run_summary["model_expansion_gate_reason"] == "insufficient_validation_evidence"


def test_collect_run_dirs_scans_immediate_children(tmp_path):
    root = tmp_path / "runs"
    run_dir = _write_run(
        root / "child_run",
        run_id="child_run",
        label_mode="no_trade_band",
        label_semantics="legacy_runner_no_trade_band_diagnostic",
        zero_return_policy="neutral_nan",
        no_trade_band_enabled=True,
        neutral_policy="abs(future_avg_r) <= threshold_bps is NaN/skipped",
        threshold_bps=5.0,
    )
    (root / "not_a_run").mkdir()

    assert summarize_runs.collect_run_dirs([], [root]) == [run_dir.resolve()]


def test_optional_protocol_columns_are_backfilled_for_older_runs(tmp_path):
    run_dir = _write_run(
        tmp_path / "older_run",
        run_id="older_run",
        label_mode="no_trade_band",
        label_semantics="legacy_runner_no_trade_band_diagnostic",
        zero_return_policy="neutral_nan",
        no_trade_band_enabled=True,
        neutral_policy="abs(future_avg_r) <= threshold_bps is NaN/skipped",
        threshold_bps=5.0,
    )
    optional_columns = [
        "label_semantics",
        "zero_return_policy",
        "no_trade_band_enabled",
        "neutral_policy",
        "label_n_zero_return",
    ]
    for filename in ("results.csv", "manifest.csv"):
        frame = pd.read_csv(run_dir / filename)
        frame = frame.drop(columns=optional_columns)
        if filename == "manifest.csv":
            frame.loc[frame["ticker"] == "pooled", "retained_pct"] = pd.NA
        frame.to_csv(run_dir / filename, index=False)

    tables = summarize_runs.read_run_tables(run_dir)
    outputs = summarize_runs.summarize_run_dirs([run_dir], tmp_path / "report")

    assert tables["results"].loc[0, "label_semantics"] == "legacy_runner_no_trade_band_diagnostic"
    assert tables["manifest"].loc[0, "zero_return_policy"] == "neutral_nan"
    assert pd.isna(tables["manifest"].loc[0, "label_n_zero_return"])
    assert outputs["run_summary"].loc[0, "pooled_retained_pct"] == pytest.approx(0.847)


def test_missing_result_columns_fail_closed(tmp_path):
    run_dir = _write_run(
        tmp_path / "bad_run",
        run_id="bad_run",
        label_mode="legacy_binary",
        label_semantics="canonical_phase1_full_binary",
        zero_return_policy="class_0_non_up",
        no_trade_band_enabled=False,
        neutral_policy="not_applicable",
        threshold_bps=0.0,
    )
    results = pd.read_csv(run_dir / "results.csv")
    results = results.drop(columns=["delta_macro_f1_vs_ticker_dummy"])
    results.to_csv(run_dir / "results.csv", index=False)

    with pytest.raises(ValueError, match="delta_macro_f1_vs_ticker_dummy"):
        summarize_runs.read_run_tables(run_dir)


def _write_run(
    run_dir,
    run_id,
    label_mode,
    label_semantics,
    zero_return_policy,
    no_trade_band_enabled,
    neutral_policy,
    threshold_bps,
    val_deltas_by_model=None,
):
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "feature_set_id": "technical_v1",
        "label_mode": label_mode,
        "label_semantics": label_semantics,
        "zero_return_policy": zero_return_policy,
        "no_trade_band_enabled": no_trade_band_enabled,
        "neutral_policy": neutral_policy,
        "threshold_bps": threshold_bps,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rows = []
    test_deltas_by_model = {"lstm": [-0.020, -0.018], "tcn": [-0.003, -0.001]}
    if val_deltas_by_model is None:
        val_deltas_by_model = test_deltas_by_model
    for model_name, deltas in test_deltas_by_model.items():
        for seed, delta in zip([42, 43], deltas):
            val_delta = val_deltas_by_model[model_name][[42, 43].index(seed)]
            rows.append(
                _result_row(
                    run_id,
                    model_name,
                    ticker="pooled",
                    seed=seed,
                    delta=delta,
                    label_mode=label_mode,
                    label_semantics=label_semantics,
                    zero_return_policy=zero_return_policy,
                    no_trade_band_enabled=no_trade_band_enabled,
                    neutral_policy=neutral_policy,
                    threshold_bps=threshold_bps,
                    val_delta=val_delta,
                )
            )
            rows.append(
                _result_row(
                    run_id,
                    model_name,
                    ticker="CSCO",
                    seed=seed,
                    delta=delta - 0.010,
                    label_mode=label_mode,
                    label_semantics=label_semantics,
                    zero_return_policy=zero_return_policy,
                    no_trade_band_enabled=no_trade_band_enabled,
                    neutral_policy=neutral_policy,
                    threshold_bps=threshold_bps,
                    val_delta=val_delta,
                )
            )
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
    pd.DataFrame(
        [
            _manifest_row(
                run_id,
                ticker="CSCO",
                label_mode=label_mode,
                label_semantics=label_semantics,
                zero_return_policy=zero_return_policy,
                no_trade_band_enabled=no_trade_band_enabled,
                neutral_policy=neutral_policy,
                threshold_bps=threshold_bps,
            ),
            _manifest_row(
                run_id,
                ticker="pooled",
                label_mode=label_mode,
                label_semantics=label_semantics,
                zero_return_policy=zero_return_policy,
                no_trade_band_enabled=no_trade_band_enabled,
                neutral_policy=neutral_policy,
                threshold_bps=threshold_bps,
            ),
        ]
    ).to_csv(run_dir / "manifest.csv", index=False)
    return run_dir


def _result_row(
    run_id,
    model_name,
    ticker,
    seed,
    delta,
    label_mode,
    label_semantics,
    zero_return_policy,
    no_trade_band_enabled,
    neutral_policy,
    threshold_bps,
    val_delta,
):
    dummy_macro_f1 = 0.5
    return {
        "run_id": run_id,
        "feature_set_id": "technical_v1",
        "label_mode": label_mode,
        "label_semantics": label_semantics,
        "zero_return_policy": zero_return_policy,
        "no_trade_band_enabled": no_trade_band_enabled,
        "neutral_policy": neutral_policy,
        "threshold_bps": threshold_bps,
        "model_name": model_name,
        "ticker": ticker,
        "seed": seed,
        "n_test_windows": 1000 if ticker == "pooled" else 200,
        "n_val_windows": 400 if ticker == "pooled" else 80,
        "val_up_pct": 0.50,
        "test_up_pct": 0.52,
        "best_val_macro_f1": dummy_macro_f1 + val_delta,
        "model_macro_f1": dummy_macro_f1 + delta,
        "model_balanced_accuracy": 0.51,
        "val_dummy_stratified_macro_f1_mean": dummy_macro_f1,
        "val_delta_macro_f1_vs_dummy": val_delta,
        "delta_macro_f1_vs_dummy": delta,
        "dummy_stratified_macro_f1_mean": dummy_macro_f1,
        "dummy_stratified_macro_f1_std": 0.002,
        "dummy_stratified_balanced_accuracy_mean": 0.51,
        "dummy_stratified_balanced_accuracy_std": 0.003,
        "dummy_stratified_confusion_matrix_mean": "[[1.0, 1.0], [1.0, 1.0]]",
        "delta_macro_f1_vs_ticker_dummy": delta,
        "ticker_dummy_stratified_macro_f1_mean": dummy_macro_f1,
        "ticker_dummy_stratified_macro_f1_std": 0.002,
        "ticker_dummy_stratified_balanced_accuracy_mean": 0.51,
        "ticker_dummy_stratified_balanced_accuracy_std": 0.003,
        "ticker_dummy_stratified_confusion_matrix_mean": "[[1.0, 1.0], [1.0, 1.0]]",
        "dummy_prior_balanced_accuracy": 0.5,
        "dummy_prior_confusion_matrix": "[[0, 2], [0, 2]]",
        "always_up_balanced_accuracy": 0.5,
        "always_up_confusion_matrix": "[[0, 2], [0, 2]]",
        "always_down_balanced_accuracy": 0.5,
        "always_down_confusion_matrix": "[[2, 0], [2, 0]]",
        "ticker_dummy_prior_balanced_accuracy": 0.5,
        "ticker_dummy_prior_confusion_matrix": "[[0, 2], [0, 2]]",
        "ticker_always_up_balanced_accuracy": 0.5,
        "ticker_always_up_confusion_matrix": "[[0, 2], [0, 2]]",
        "ticker_always_down_balanced_accuracy": 0.5,
        "ticker_always_down_confusion_matrix": "[[2, 0], [2, 0]]",
        "label_n_neutral": 0,
        "label_n_zero_return": 2,
        "retained_pct": 0.847,
        "train_rows": 900,
        "val_rows": 500,
        "test_rows": 1100,
        "train_retained_labels": 800,
        "val_retained_labels": 400,
        "test_retained_labels": 1000 if ticker == "pooled" else 200,
        "train_nan_labels": 100,
        "val_nan_labels": 100,
        "test_nan_labels": 100,
        "training_time_seconds": 12.0 + seed,
        "suspicious_status": "ok",
    }


def _manifest_row(
    run_id,
    ticker,
    label_mode,
    label_semantics,
    zero_return_policy,
    no_trade_band_enabled,
    neutral_policy,
    threshold_bps,
):
    return {
        "run_id": run_id,
        "feature_set_id": "technical_v1",
        "label_mode": label_mode,
        "label_semantics": label_semantics,
        "zero_return_policy": zero_return_policy,
        "no_trade_band_enabled": no_trade_band_enabled,
        "neutral_policy": neutral_policy,
        "threshold_bps": threshold_bps,
        "ticker": ticker,
        "label_n_total": 1200,
        "label_n_retained": 1000,
        "label_n_neutral": 0,
        "label_n_cross_day": 150,
        "label_n_tail": 12,
        "label_n_zero_return": 2,
        "retained_pct": 0.847,
        "train_rows": 900,
        "val_rows": 500,
        "test_rows": 1100,
        "train_retained_labels": 800,
        "val_retained_labels": 400,
        "test_retained_labels": 1000 if ticker == "pooled" else 200,
        "train_nan_labels": 100,
        "val_nan_labels": 100,
        "test_nan_labels": 100,
        "n_train_windows": 800,
        "n_val_windows": 400 if ticker == "pooled" else 80,
        "n_test_windows": 1000 if ticker == "pooled" else 200,
        "train_up_pct": 0.51,
        "val_up_pct": 0.50,
        "test_up_pct": 0.52,
    }
