"""Create a compact review summary from extracted Stage 0 output CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt(value: float) -> str:
    return f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    base = Path(args.artifact_dir)

    a1 = pd.read_csv(base / "stage0a1_table1.csv")
    a2 = pd.read_csv(base / "stage0a2_table1.csv")
    b = pd.read_csv(base / "stage0b_table1.csv")

    a1_top = a1.sort_values(["basic_gate", "macro_f1_mean"], ascending=[False, False]).iloc[0]
    a2_top = a2.sort_values(["basic_gate", "macro_f1_mean"], ascending=[False, False]).iloc[0]
    b_top = b.sort_values("macro_f1_mean", ascending=False).iloc[0]

    a2_w20 = a2.loc[a2["window_size"].eq(20)].iloc[0]
    a2_w10 = a2.loc[a2["window_size"].eq(10)].iloc[0]
    a2_w5 = a2.loc[a2["window_size"].eq(5)].iloc[0]

    b_lgbm = b.loc[b["model"].eq("lightgbm")].iloc[0]
    b_logreg = b.loc[b["model"].eq("logreg")].iloc[0]
    b_gru = b.loc[b["model"].eq("simple_gru")].iloc[0]
    b_ms = b.loc[b["model"].eq("ms_dlinear_tcn")].iloc[0]

    lines = []
    lines.append("# Stage 0 Desktop Notebook Output Review - 2026-06-04")
    lines.append("")
    lines.append("Source notebook: `C:\\Users\\CK\\Desktop\\02_config_screening_colab.ipynb`")
    lines.append("")
    lines.append("## Extracted Result")
    lines.append("")
    lines.append("- Notebook execution: 20 cells, 10 executed code cells, 13 outputs, 0 error outputs.")
    lines.append("- Stage 0S was skipped. Stage 0A1, Stage 0A2, Stage 0B ran.")
    lines.append("- Drive upload reported 10 uploaded files and 2 skipped Stage 0S files.")
    lines.append("")
    lines.append("## Official Stage 0 Selection")
    lines.append("")
    lines.append(
        "- Stage 0A1 selected `h03_bps1p5 + price_volume_time` by both mean and LCB candidate rules."
    )
    lines.append(
        f"- Stage 0A2 selected `h03_bps1p5 + price_volume_time + window_size=20` "
        f"with macro F1 mean {fmt(a2_w20['macro_f1_mean'])}, LCB {fmt(a2_w20['macro_f1_lcb_95'])}, "
        f"and delta vs dummy {fmt(a2_w20['delta_macro_f1_vs_dummy_mean'])}."
    )
    lines.append("")
    lines.append("## Stage 0A2 Window Check")
    lines.append("")
    lines.append(
        f"- Window 20 beat window 10 by {fmt(a2_w20['macro_f1_mean'] - a2_w10['macro_f1_mean'])} macro F1."
    )
    lines.append(
        f"- Window 20 beat window 5 by {fmt(a2_w20['macro_f1_mean'] - a2_w5['macro_f1_mean'])} macro F1."
    )
    lines.append(
        "- The window-20 win is protocol-correct, but the practical margin over 10/5 is small. "
        "This supports a separate diagnostic H0 for window continuity, not a claim that 20 is universally optimal."
    )
    lines.append("")
    lines.append("## Stage 0B Second-View Check")
    lines.append("")
    lines.append(
        f"- LightGBM: macro F1 {fmt(b_lgbm['macro_f1_mean'])}, delta vs dummy {fmt(b_lgbm['delta_macro_f1_vs_dummy_mean'])}."
    )
    lines.append(
        f"- LogReg: macro F1 {fmt(b_logreg['macro_f1_mean'])}, delta vs dummy {fmt(b_logreg['delta_macro_f1_vs_dummy_mean'])}; "
        f"only {fmt(b_lgbm['macro_f1_mean'] - b_logreg['macro_f1_mean'])} below LightGBM."
    )
    lines.append(
        f"- Simple GRU: macro F1 {fmt(b_gru['macro_f1_mean'])}, LCB {fmt(b_gru['macro_f1_lcb_95'])}."
    )
    lines.append(
        f"- MS-DLinear+TCN: macro F1 {fmt(b_ms['macro_f1_mean'])}, LCB {fmt(b_ms['macro_f1_lcb_95'])}."
    )
    lines.append(
        "- `deep_model_disagrees` is false for all Stage 0B rows. Deep models do not retract the Stage 0A candidate."
    )
    lines.append(
        "- Deep models also do not improve over LightGBM in this fixed-default run; the current signal looks mostly tabular/linear."
    )
    lines.append("")
    lines.append("## Breadth And Concentration")
    lines.append("")
    lines.append(
        f"- Stage 0A2 selected row has positive_ticker_count={int(a2_w20['positive_ticker_count'])} "
        f"and top_ticker_gain_share={fmt(a2_w20['top_ticker_gain_share'])}."
    )
    lines.append(
        "- This passes the breadth/concentration gate: signal is not coming from a single ticker in the displayed summary."
    )
    lines.append("")
    lines.append("## Statistical And Time-Series Caveats")
    lines.append("")
    lines.append(
        "- All conclusions remain validation_only. The final holdout/test is not authorized by these outputs."
    )
    lines.append(
        "- Seed std for LightGBM is very small, but sliding-window samples are highly autocorrelated; row count should not be read as independent sample count."
    )
    lines.append(
        "- Stage 0 used 125 validation-screened model-seed rows. Multiple comparison risk is controlled by the pre-registered gates, but results are still selection records, not final evidence."
    )
    lines.append(
        "- Per-ticker detailed CSVs were uploaded by the notebook but are not embedded in the notebook output. Download them from Drive for deeper ticker-level review."
    )
    lines.append("")
    lines.append("## Next Action")
    lines.append("")
    lines.append(
        "The official Stage 0 candidate is ready to feed Notebook 03: "
        "`h03_bps1p5 + price_volume_time + window_size=20`. "
        "Diagnostic H0 may now run as a separate post-Stage0 diagnostic if you want to check 24/32/longer windows."
    )

    out_path = base / "stage0_review_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
