from pathlib import Path

import pandas as pd
import pytest

from scripts.phase1b_local import build_paper_tables


def test_build_paper_tables_writes_manuscript_tables(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    output_dir = tmp_path / "paper_tables"

    outputs = build_paper_tables.build_paper_tables(input_dir, output_dir)

    assert set(outputs) == {
        "paper_table_1_run_gate_summary",
        "paper_table_2_pooled_model_vs_dummy",
        "paper_table_3_canonical_ticker_delta",
        "paper_table_4_coverage_label_semantics",
        "paper_table_5_ticker_delta_counts",
        "paper_table_6_seed_ticker_stability",
        "paper_table_7_regime_shift_by_ticker",
        "paper_table_8_coverage_fragility_flags",
        "figure_delta_vs_coverage",
        "figure_ticker_delta_heatmap",
        "figure_threshold_retention_proxy",
    }
    assert (output_dir / "paper_table_1_run_gate_summary.csv").exists()
    assert (output_dir / "paper_table_2_pooled_model_vs_dummy.csv").exists()
    assert (output_dir / "paper_table_3_canonical_ticker_delta.csv").exists()
    assert (output_dir / "paper_table_4_coverage_label_semantics.csv").exists()
    assert (output_dir / "paper_table_5_ticker_delta_counts.csv").exists()
    assert (output_dir / "paper_table_6_seed_ticker_stability.csv").exists()
    assert (output_dir / "paper_table_7_regime_shift_by_ticker.csv").exists()
    assert (output_dir / "paper_table_8_coverage_fragility_flags.csv").exists()
    assert (output_dir / "figure_delta_vs_coverage.csv").exists()
    assert (output_dir / "figure_ticker_delta_heatmap.csv").exists()
    assert (output_dir / "figure_threshold_retention_proxy.csv").exists()
    assert (output_dir / "paper_tables.md").exists()

    gate = outputs["paper_table_1_run_gate_summary"]
    assert gate["regime"].tolist() == [
        "canonical_full_binary",
        "0bps_no_trade_band_diagnostic",
        "5bps_no_trade_band_diagnostic",
    ]
    assert gate.loc[0, "best_model"] == "tcn"
    assert gate.loc[0, "best_delta_macro_f1_vs_dummy"] == pytest.approx(-0.002318)
    assert gate.loc[0, "model_expansion_gate"] == "blocked_delta_lt_0.01"

    pooled = outputs["paper_table_2_pooled_model_vs_dummy"]
    canonical_tcn = pooled.loc[
        (pooled["regime"] == "canonical_full_binary")
        & (pooled["model_name"] == "tcn")
    ].iloc[0]
    assert canonical_tcn["delta_macro_f1_vs_dummy_mean"] == pytest.approx(-0.002318)
    assert canonical_tcn["retained_pct"] == pytest.approx(0.847315)

    markdown = (output_dir / "paper_tables.md").read_text(encoding="utf-8")
    assert "Protocol-Safe Paper Tables" in markdown
    assert "canonical_full_binary" in markdown
    assert "0bps_no_trade_band_diagnostic" in markdown
    assert "Seed Ticker Stability" in markdown
    assert "Regime Shift By Ticker" in markdown
    assert "Coverage Fragility Flags" in markdown
    assert "Threshold Retention Proxy" in markdown


def test_canonical_ticker_table_marks_positive_delta_false(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    ticker_table = outputs["paper_table_3_canonical_ticker_delta"]

    assert set(ticker_table["regime"]) == {"canonical_full_binary"}
    assert ticker_table["positive_delta"].tolist() == [False, False, False, False]
    assert ticker_table["delta_macro_f1_vs_ticker_dummy_mean"].max() < 0.0


def test_ticker_delta_counts_summarize_positive_and_non_positive(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    counts = outputs["paper_table_5_ticker_delta_counts"]
    canonical_lstm = counts.loc[
        (counts["regime"] == "canonical_full_binary")
        & (counts["model_name"] == "lstm")
    ].iloc[0]
    diagnostic_lstm = counts.loc[
        (counts["regime"] == "5bps_no_trade_band_diagnostic")
        & (counts["model_name"] == "lstm")
    ].iloc[0]

    assert canonical_lstm["n_tickers"] == 2
    assert canonical_lstm["n_positive_delta"] == 0
    assert canonical_lstm["n_non_positive_delta"] == 2
    assert bool(canonical_lstm["all_non_positive_delta"]) is True
    assert diagnostic_lstm["n_positive_delta"] == 1
    assert bool(diagnostic_lstm["all_non_positive_delta"]) is False


def test_seed_ticker_stability_uses_seed_level_rows(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    stability = outputs["paper_table_6_seed_ticker_stability"]
    canonical_lstm_csco = stability.loc[
        (stability["regime"] == "canonical_full_binary")
        & (stability["model_name"] == "lstm")
        & (stability["ticker"] == "CSCO")
    ].iloc[0]
    diagnostic_lstm_csco = stability.loc[
        (stability["regime"] == "5bps_no_trade_band_diagnostic")
        & (stability["model_name"] == "lstm")
        & (stability["ticker"] == "CSCO")
    ].iloc[0]

    assert canonical_lstm_csco["seeds"] == "42,43"
    assert canonical_lstm_csco["n_seeds"] == 2
    assert canonical_lstm_csco["n_positive_seed_delta"] == 0
    assert bool(canonical_lstm_csco["all_seeds_non_positive"]) is True
    assert diagnostic_lstm_csco["n_positive_seed_delta"] == 2
    assert diagnostic_lstm_csco["positive_seed_rate"] == pytest.approx(1.0)
    assert diagnostic_lstm_csco["n_suspicious_seed_rows"] == 0


def test_regime_shift_by_ticker_keeps_diagnostic_comparison_post_hoc(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    shift = outputs["paper_table_7_regime_shift_by_ticker"]
    lstm_csco = shift.loc[
        (shift["model_name"] == "lstm") & (shift["ticker"] == "CSCO")
    ].iloc[0]

    assert lstm_csco["canonical_delta"] == pytest.approx(-0.034409)
    assert lstm_csco["diagnostic_0bps_delta"] == pytest.approx(0.007889)
    assert lstm_csco["diagnostic_5bps_delta"] == pytest.approx(0.025177)
    assert lstm_csco["delta_5bps_minus_canonical"] == pytest.approx(0.059586)
    assert lstm_csco["coverage_drop_5bps"] == pytest.approx(0.0)


def test_coverage_fragility_flags_low_coverage_descriptive_scope(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    fragility = outputs["paper_table_8_coverage_fragility_flags"]
    diagnostic = fragility.loc[
        (fragility["regime"] == "5bps_no_trade_band_diagnostic")
        & (fragility["ticker"] == "pooled")
    ].iloc[0]
    canonical = fragility.loc[
        (fragility["regime"] == "canonical_full_binary")
        & (fragility["ticker"] == "pooled")
    ].iloc[0]

    assert bool(diagnostic["low_coverage_flag"]) is True
    assert bool(diagnostic["diagnostic_only"]) is True
    assert diagnostic["claim_scope"] == "diagnostic_low_coverage_descriptive_only"
    assert bool(canonical["low_coverage_flag"]) is False
    assert canonical["claim_scope"] == "canonical_descriptive"


def test_threshold_retention_proxy_is_not_confidence_curve(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    proxy = outputs["figure_threshold_retention_proxy"]

    assert proxy["proxy_kind"].unique().tolist() == [
        "threshold_retention_not_confidence"
    ]
    assert proxy["regime"].tolist() == [
        "canonical_full_binary",
        "0bps_no_trade_band_diagnostic",
        "5bps_no_trade_band_diagnostic",
    ]
    assert proxy.loc[2, "retained_pct"] == pytest.approx(0.149115)
    assert proxy.loc[2, "best_delta_macro_f1_vs_dummy"] == pytest.approx(0.001893)


def test_coverage_table_keeps_missing_zero_return_as_na(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    outputs = build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")

    coverage = outputs["paper_table_4_coverage_label_semantics"]
    diagnostic = coverage.loc[
        coverage["regime"] == "5bps_no_trade_band_diagnostic"
    ].iloc[0]

    assert pd.isna(diagnostic["label_n_zero_return"])
    assert diagnostic["label_n_neutral"] == 100
    assert diagnostic["retained_pct"] == pytest.approx(0.149115)


def test_missing_input_file_fails_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    (input_dir / "pooled_by_model.csv").unlink()

    with pytest.raises(FileNotFoundError, match="pooled_by_model.csv"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_missing_required_column_fails_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    pooled = pd.read_csv(input_dir / "pooled_by_model.csv")
    pooled = pooled.drop(columns=["delta_macro_f1_vs_dummy_mean"])
    pooled.to_csv(input_dir / "pooled_by_model.csv", index=False)

    with pytest.raises(ValueError, match="delta_macro_f1_vs_dummy_mean"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_missing_canonical_ticker_rows_fail_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    tickers = pd.read_csv(input_dir / "by_model_ticker.csv")
    tickers = tickers.loc[tickers["label_semantics"] != "canonical_phase1_full_binary"]
    tickers.to_csv(input_dir / "by_model_ticker.csv", index=False)

    with pytest.raises(ValueError, match="canonical ticker rows"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_missing_seed_result_file_fails_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    result_path = input_dir / "runs" / "canonical_run" / "results.csv"
    result_path.unlink()

    with pytest.raises(FileNotFoundError, match="results.csv"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_seed_result_missing_required_column_fails_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    result_path = input_dir / "runs" / "canonical_run" / "results.csv"
    results = pd.read_csv(result_path)
    results = results.drop(columns=["delta_macro_f1_vs_ticker_dummy"])
    results.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="delta_macro_f1_vs_ticker_dummy"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_duplicate_seed_rows_fail_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    result_path = input_dir / "runs" / "canonical_run" / "results.csv"
    results = pd.read_csv(result_path)
    results = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    results.to_csv(result_path, index=False)

    with pytest.raises(ValueError, match="duplicate seed row"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def test_invalid_probability_values_fail_closed(tmp_path):
    input_dir = _write_report_inputs(tmp_path / "report")
    coverage = pd.read_csv(input_dir / "coverage_by_ticker.csv")
    coverage.loc[0, "retained_pct"] = 1.1
    coverage.to_csv(input_dir / "coverage_by_ticker.csv", index=False)

    with pytest.raises(ValueError, match="retained_pct outside"):
        build_paper_tables.build_paper_tables(input_dir, tmp_path / "paper_tables")


def _write_report_inputs(input_dir):
    input_dir.mkdir(parents=True)
    run_root = input_dir / "runs"
    canonical_run_dir = run_root / "canonical_run"
    diagnostic_0bps_run_dir = run_root / "diagnostic_0bps"
    diagnostic_5bps_run_dir = run_root / "diagnostic_5bps"
    pd.DataFrame(
        [
            {
                "run_dir": str(canonical_run_dir),
                "run_id": "canonical_run",
                "label_mode": "legacy_binary",
                "label_semantics": "canonical_phase1_full_binary",
                "zero_return_policy": "class_0_non_up",
                "no_trade_band_enabled": False,
                "neutral_policy": "not_applicable",
                "threshold_bps": 0.0,
                "feature_set_id": "technical_v1",
                "seeds": "42,43",
                "best_pooled_model": "tcn",
                "best_pooled_delta_macro_f1_vs_dummy": -0.002318,
                "model_expansion_gate": "blocked_delta_lt_0.01",
                "n_suspicious_rows": 0,
                "pooled_retained_pct": 0.847315,
                "pooled_test_windows": 235333,
                "pooled_zero_return_rows": 13,
            },
            {
                "run_dir": str(diagnostic_0bps_run_dir),
                "run_id": "diagnostic_0bps",
                "label_mode": "no_trade_band",
                "label_semantics": "phase1b_no_trade_band_diagnostic",
                "zero_return_policy": "neutral_nan",
                "no_trade_band_enabled": True,
                "neutral_policy": "abs(future_avg_r) <= threshold_bps is NaN/skipped",
                "threshold_bps": 0.0,
                "feature_set_id": "technical_v1",
                "seeds": "42,43",
                "best_pooled_model": "tcn",
                "best_pooled_delta_macro_f1_vs_dummy": 0.004890,
                "model_expansion_gate": "blocked_delta_lt_0.01",
                "n_suspicious_rows": 0,
                "pooled_retained_pct": 0.847309,
                "pooled_test_windows": 235333,
                "pooled_zero_return_rows": pd.NA,
            },
            {
                "run_dir": str(diagnostic_5bps_run_dir),
                "run_id": "diagnostic_5bps",
                "label_mode": "no_trade_band",
                "label_semantics": "phase1b_no_trade_band_diagnostic",
                "zero_return_policy": "neutral_nan",
                "no_trade_band_enabled": True,
                "neutral_policy": "abs(future_avg_r) <= threshold_bps is NaN/skipped",
                "threshold_bps": 5.0,
                "feature_set_id": "technical_v1",
                "seeds": "42,43",
                "best_pooled_model": "lstm",
                "best_pooled_delta_macro_f1_vs_dummy": 0.001893,
                "model_expansion_gate": "blocked_delta_lt_0.01",
                "n_suspicious_rows": 0,
                "pooled_retained_pct": 0.149115,
                "pooled_test_windows": 19182,
                "pooled_zero_return_rows": pd.NA,
            },
        ]
    ).to_csv(input_dir / "run_summary.csv", index=False)

    pd.DataFrame(
        [
            _pooled_row("canonical_run", "canonical_phase1_full_binary", 0.0, "dlinear", 0.476766, -0.023092, 0.847315),
            _pooled_row("canonical_run", "canonical_phase1_full_binary", 0.0, "lstm", 0.484279, -0.015578, 0.847315),
            _pooled_row("canonical_run", "canonical_phase1_full_binary", 0.0, "tcn", 0.497540, -0.002318, 0.847315),
            _pooled_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "tcn", 0.504747, 0.004890, 0.847309),
            _pooled_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "lstm", 0.499939, 0.001893, 0.149115),
        ]
    ).to_csv(input_dir / "pooled_by_model.csv", index=False)

    pd.DataFrame(
        [
            _ticker_row("canonical_run", "canonical_phase1_full_binary", 0.0, "lstm", "CSCO", -0.034409),
            _ticker_row("canonical_run", "canonical_phase1_full_binary", 0.0, "lstm", "JPM", -0.043262),
            _ticker_row("canonical_run", "canonical_phase1_full_binary", 0.0, "tcn", "CSCO", -0.032842),
            _ticker_row("canonical_run", "canonical_phase1_full_binary", 0.0, "tcn", "JPM", -0.016494),
            _ticker_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "lstm", "CSCO", 0.007889),
            _ticker_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "lstm", "JPM", -0.016661),
            _ticker_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "tcn", "CSCO", -0.010000),
            _ticker_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "tcn", "JPM", -0.020000),
            _ticker_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "lstm", "CSCO", 0.025177),
            _ticker_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "lstm", "JPM", -0.070268),
        ]
    ).to_csv(input_dir / "by_model_ticker.csv", index=False)

    pd.DataFrame(
        [
            _coverage_row("canonical_run", "canonical_phase1_full_binary", 0.0, "pooled", 0.847315, 0, 13),
            _coverage_row("canonical_run", "canonical_phase1_full_binary", 0.0, "CSCO", 0.847493, 0, 5),
            _coverage_row("canonical_run", "canonical_phase1_full_binary", 0.0, "JPM", 0.847301, 0, 0),
            _coverage_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "pooled", 0.847309, 13, pd.NA),
            _coverage_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "CSCO", 0.847482, 5, pd.NA),
            _coverage_row("diagnostic_0bps", "phase1b_no_trade_band_diagnostic", 0.0, "JPM", 0.847301, 0, pd.NA),
            _coverage_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "pooled", 0.149115, 100, pd.NA),
            _coverage_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "CSCO", 0.847493, 100, pd.NA),
            _coverage_row("diagnostic_5bps", "phase1b_no_trade_band_diagnostic", 5.0, "JPM", 0.847301, 100, pd.NA),
        ]
    ).to_csv(input_dir / "coverage_by_ticker.csv", index=False)
    _write_seed_results(
        canonical_run_dir,
        [
            _seed_row("canonical_run", "lstm", "CSCO", 42, -0.062213),
            _seed_row("canonical_run", "lstm", "CSCO", 43, -0.006605),
            _seed_row("canonical_run", "lstm", "JPM", 42, -0.018677),
            _seed_row("canonical_run", "lstm", "JPM", 43, -0.067847),
            _seed_row("canonical_run", "tcn", "CSCO", 42, -0.032842),
            _seed_row("canonical_run", "tcn", "CSCO", 43, -0.032842),
            _seed_row("canonical_run", "tcn", "JPM", 42, -0.016494),
            _seed_row("canonical_run", "tcn", "JPM", 43, -0.016494),
        ],
    )
    _write_seed_results(
        diagnostic_0bps_run_dir,
        [
            _seed_row("diagnostic_0bps", "lstm", "CSCO", 42, 0.007889),
            _seed_row("diagnostic_0bps", "lstm", "CSCO", 43, 0.007889),
            _seed_row("diagnostic_0bps", "lstm", "JPM", 42, -0.016661),
            _seed_row("diagnostic_0bps", "lstm", "JPM", 43, -0.016661),
            _seed_row("diagnostic_0bps", "tcn", "CSCO", 42, -0.010000),
            _seed_row("diagnostic_0bps", "tcn", "CSCO", 43, -0.010000),
            _seed_row("diagnostic_0bps", "tcn", "JPM", 42, -0.020000),
            _seed_row("diagnostic_0bps", "tcn", "JPM", 43, -0.020000),
        ],
    )
    _write_seed_results(
        diagnostic_5bps_run_dir,
        [
            _seed_row("diagnostic_5bps", "lstm", "CSCO", 42, 0.023235),
            _seed_row("diagnostic_5bps", "lstm", "CSCO", 43, 0.027119),
            _seed_row("diagnostic_5bps", "lstm", "JPM", 42, -0.070268),
            _seed_row("diagnostic_5bps", "lstm", "JPM", 43, -0.070268),
        ],
    )
    return input_dir


def _pooled_row(run_id, label_semantics, threshold_bps, model_name, macro_f1, delta, retained_pct):
    return {
        "run_id": run_id,
        "label_mode": "legacy_binary" if label_semantics == "canonical_phase1_full_binary" else "no_trade_band",
        "label_semantics": label_semantics,
        "zero_return_policy": "class_0_non_up" if label_semantics == "canonical_phase1_full_binary" else "neutral_nan",
        "no_trade_band_enabled": label_semantics != "canonical_phase1_full_binary",
        "neutral_policy": "not_applicable",
        "threshold_bps": threshold_bps,
        "feature_set_id": "technical_v1",
        "model_name": model_name,
        "ticker": "pooled",
        "macro_f1_mean": macro_f1,
        "macro_f1_std": 0.01,
        "balanced_accuracy_mean": 0.51,
        "dummy_stratified_macro_f1_mean": 0.499857,
        "dummy_stratified_macro_f1_std": 0.001,
        "delta_macro_f1_vs_dummy_mean": delta,
        "delta_macro_f1_vs_dummy_std": 0.01,
        "n_test_windows": 235333,
        "retained_pct": retained_pct,
    }


def _ticker_row(run_id, label_semantics, threshold_bps, model_name, ticker, delta):
    return {
        "run_id": run_id,
        "label_mode": "legacy_binary" if label_semantics == "canonical_phase1_full_binary" else "no_trade_band",
        "label_semantics": label_semantics,
        "zero_return_policy": "class_0_non_up" if label_semantics == "canonical_phase1_full_binary" else "neutral_nan",
        "no_trade_band_enabled": label_semantics != "canonical_phase1_full_binary",
        "neutral_policy": "not_applicable",
        "threshold_bps": threshold_bps,
        "feature_set_id": "technical_v1",
        "model_name": model_name,
        "ticker": ticker,
        "delta_macro_f1_vs_ticker_dummy_mean": delta,
        "delta_macro_f1_vs_ticker_dummy_std": 0.01,
        "ticker_dummy_stratified_macro_f1_mean": 0.5,
        "macro_f1_mean": 0.5 + delta,
        "n_test_windows": 1000,
        "retained_pct": 0.847315,
    }


def _coverage_row(
    run_id,
    label_semantics,
    threshold_bps,
    ticker,
    retained_pct,
    label_n_neutral,
    label_n_zero_return,
):
    return {
        "run_id": run_id,
        "label_mode": "legacy_binary" if label_semantics == "canonical_phase1_full_binary" else "no_trade_band",
        "label_semantics": label_semantics,
        "zero_return_policy": "class_0_non_up" if label_semantics == "canonical_phase1_full_binary" else "neutral_nan",
        "no_trade_band_enabled": label_semantics != "canonical_phase1_full_binary",
        "neutral_policy": "not_applicable",
        "threshold_bps": threshold_bps,
        "feature_set_id": "technical_v1",
        "ticker": ticker,
        "label_n_neutral": label_n_neutral,
        "label_n_zero_return": label_n_zero_return,
        "retained_pct": retained_pct,
        "n_test_windows": 1000,
        "test_up_pct": 0.52,
    }


def _write_seed_results(run_dir: Path, rows):
    run_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)


def _seed_row(run_id, model_name, ticker, seed, delta):
    return {
        "run_id": run_id,
        "model_name": model_name,
        "ticker": ticker,
        "seed": seed,
        "model_macro_f1": 0.5 + delta,
        "delta_macro_f1_vs_ticker_dummy": delta,
        "n_test_windows": 1000,
        "retained_pct": 0.847315,
        "suspicious_status": "ok",
    }
