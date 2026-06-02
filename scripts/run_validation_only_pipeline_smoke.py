"""Run the baseline_v1 validation-only real-data pipeline smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intraday_research.validation_pipeline import (  # noqa: E402
    DEFAULT_TICKERS,
    build_validation_only_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validation-only baseline_v1 smoke report from data/*.csv."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    parser.add_argument("--horizon-k", type=int, default=12)
    parser.add_argument("--threshold-bps", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--diagnostic-max-train-rows", type=int, default=20000)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    report = build_validation_only_report(
        data_dir=args.data_dir,
        tickers=tuple(args.tickers),
        horizon_k=args.horizon_k,
        threshold_bps=args.threshold_bps,
        window_size=args.window_size,
        diagnostic_max_train_rows=args.diagnostic_max_train_rows,
        walk_forward_folds=args.walk_forward_folds,
    )
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    main()
