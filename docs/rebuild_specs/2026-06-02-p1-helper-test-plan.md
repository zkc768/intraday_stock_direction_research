# P1 Helper Test Plan

Date: 2026-06-02
Status: implemented as a tests-first P1 helper slice.
Scope: design tests before extracting active helpers from
`notebooks/04_ian_research_memo.ipynb`.

## Goal

Create focused synthetic tests for the safety-critical notebook logic before
moving any function into an active helper module. These tests should prove
chronology, leakage, train-only preprocessing, ticker isolation, split
isolation, and dummy-baseline behavior without reading raw data, checkpoints,
artifacts, or holdout/test results.

## Current Implementation Status

Implemented files:

```text
intraday_research/__init__.py
intraday_research/baseline_v1.py
tests/test_baseline_v1_helpers.py
```

Validation result:

```text
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_baseline_v1_helpers.py -q
36 passed after the second-round critical-review repair pass
```

The first pytest run failed during collection because the helper skeleton did
not yet expose the required functions. That failure was the expected tests-first
checkpoint before minimal implementation.

Post-review fixes added:

- single-ticker guards for label, split-boundary, and segment-window helpers;
- synthetic pooled multi-ticker rejection tests;
- stronger stratified dummy proof using train-all-zero labels and
  validation-all-one labels;
- AST import guard for active helper imports.

## Proposed Files

Created in the P1 implementation step:

```text
intraday_research/__init__.py
intraday_research/baseline_v1.py
tests/test_baseline_v1_helpers.py
```

The implementation step created tests first. The helper module then implemented
only the behavior required by those tests.

Do not create:

```text
ml_utils/
runner_utils/
scripts/local_runner*.py
PM_* docs
checkpoint readers
artifact readers
```

## Test Fixture Shape

Use only synthetic in-memory frames.

```python
import pandas as pd


def make_one_ticker_frame():
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * 8,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:30",
                    "2020-01-01 09:35",
                    "2020-01-01 09:40",
                    "2020-01-01 09:45",
                    "2020-01-02 09:30",
                    "2020-01-02 09:35",
                    "2020-01-02 09:40",
                    "2020-01-02 09:45",
                ]
            ),
            "open": [100, 101, 102, 103, 110, 111, 112, 113],
            "high": [101, 102, 103, 104, 111, 112, 113, 114],
            "low": [99, 100, 101, 102, 109, 110, 111, 112],
            "close": [100, 101, 103, 102, 110, 111, 109, 112],
            "volume": [1000, 1100, 1200, 1300, 1000, 1100, 1200, 1300],
        }
    )
```

Use a second ticker by copying this frame and changing `ticker` plus close
values. Never use project raw data in these tests.

## Test Cases

### 1. Label Horizon Uses Future One-Bar Returns

Target functions:

```python
from intraday_research.baseline_v1 import make_no_trade_band_labels
```

Test:

```python
def test_no_trade_label_uses_future_one_bar_returns_only():
    frame = make_one_ticker_frame()
    labeled = make_no_trade_band_labels(frame, horizon_k=2, threshold_bps=0.0)

    one_bar = frame["close"].pct_change()
    expected = (one_bar.shift(-1) + one_bar.shift(-2)) / 2

    assert labeled.loc[0, "future_cumulative_return"] == expected.loc[0]
    assert labeled.loc[1, "future_cumulative_return"] == expected.loc[1]
    assert pd.isna(labeled.loc[6, "label"])
    assert pd.isna(labeled.loc[7, "label"])
```

### 2. No-Trade Band Creates Invalid Markers

Target functions:

```python
from intraday_research.baseline_v1 import make_no_trade_band_labels
```

Test:

```python
def test_no_trade_band_marks_near_zero_returns_invalid_not_classed():
    frame = make_one_ticker_frame()
    frame["close"] = [100, 100.001, 100.002, 100.003, 100.004, 100.005, 100.006, 100.007]

    labeled = make_no_trade_band_labels(frame, horizon_k=2, threshold_bps=5.0)

    assert labeled["label"].isna().any()
    assert set(labeled["label"].dropna().unique()).issubset({0, 1})
```

### 3. Calendar Splits Are Half-Open

Target functions:

```python
from intraday_research.baseline_v1 import assign_calendar_split
```

Test:

```python
def test_calendar_split_boundaries_are_half_open():
    splits = {
        "train": ("2020-01-01", "2020-01-02"),
        "validation": ("2020-01-02", "2020-01-03"),
        "closed_holdout_boundary_only": ("2020-01-03", "2020-01-04"),
    }

    assert assign_calendar_split(pd.Timestamp("2020-01-01 00:00"), splits) == "train"
    assert assign_calendar_split(pd.Timestamp("2020-01-02 00:00"), splits) == "validation"
    assert assign_calendar_split(pd.Timestamp("2020-01-03 00:00"), splits) == "closed_holdout_boundary_only"
    assert assign_calendar_split(pd.Timestamp("2020-01-04 00:00"), splits) == "outside_defined_calendar"
```

### 4. Split Boundary Invalidates Horizon Crossing

Target functions:

```python
from intraday_research.baseline_v1 import add_split_and_invalidate_boundaries
```

Test:

```python
def test_split_boundary_invalidates_labels_before_next_split():
    frame = make_one_ticker_frame()
    frame["future_cumulative_return"] = 0.01
    frame["label"] = 1.0
    splits = {
        "train": ("2020-01-01 09:30", "2020-01-01 09:45"),
        "validation": ("2020-01-01 09:45", "2020-01-02 09:45"),
        "closed_holdout_boundary_only": ("2020-01-02 09:45", "2020-01-03"),
    }

    checked = add_split_and_invalidate_boundaries(frame, splits=splits, horizon_k=2)

    assert pd.isna(checked.loc[1, "label"])
    assert checked.loc[1, "invalid_cross_split"]
    assert checked.loc[3, "split"] == "validation"
```

### 5. Scaler Fits On Train Only

Target functions:

```python
from intraday_research.baseline_v1 import fit_train_only_scaler
```

Test:

```python
def test_scaler_fit_ignores_validation_and_closed_holdout_values():
    feature_columns = ["f1", "f2"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "split": ["train", "train", "train", "validation", "validation", "closed_holdout_boundary_only"],
            "f1": [1.0, 2.0, 3.0, 1000.0, 2000.0, 3000.0],
            "f2": [10.0, 20.0, 30.0, 10000.0, 20000.0, 30000.0],
        }
    )

    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    assert list(scaler.mean_) == [2.0, 20.0]
```

### 6. Transform Scope Excludes Closed Holdout

Target functions:

```python
from intraday_research.baseline_v1 import fit_train_only_scaler, transform_train_and_validation
```

Test:

```python
def test_transform_train_and_validation_does_not_transform_closed_holdout():
    feature_columns = ["f1"]
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 4,
            "split": ["train", "train", "validation", "closed_holdout_boundary_only"],
            "f1": [1.0, 2.0, 3.0, 999.0],
        }
    )
    scaler = fit_train_only_scaler({"AAA": frame}, feature_columns=feature_columns)

    transformed = transform_train_and_validation({"AAA": frame}, scaler, feature_columns=feature_columns)["AAA"]

    assert "f1_scaled" in transformed.columns
    assert pd.notna(transformed.loc[0, "f1_scaled"])
    assert pd.notna(transformed.loc[2, "f1_scaled"])
    assert pd.isna(transformed.loc[3, "f1_scaled"])
```

### 7. Windows Do Not Cross Day Or Split

Target functions:

```python
from intraday_research.baseline_v1 import build_windows_for_segment
```

Test:

```python
def test_windows_do_not_cross_day_or_split_and_skip_invalid_labels():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 09:30",
                    "2020-01-01 09:35",
                    "2020-01-01 09:40",
                    "2020-01-02 09:30",
                    "2020-01-02 09:35",
                    "2020-01-02 09:40",
                ]
            ),
            "split": ["train", "train", "train", "train", "train", "train"],
            "f1_scaled": [1, 2, 3, 4, 5, 6],
            "label": [0, 1, 1, 0, 1, float("nan")],
        }
    )

    result = build_windows_for_segment(frame, "train", feature_columns=["f1"], window_size=2)

    assert result["X"].shape[1:] == (2, 1)
    assert len(result["y"]) == 3
    assert set(result["metadata"]["target_timestamp"].dt.date) == {
        pd.Timestamp("2020-01-01").date(),
        pd.Timestamp("2020-01-02").date(),
    }
```

### 8. Model Windows Require Scaled Features

Target functions:

```python
from intraday_research.baseline_v1 import build_windows_for_segment
```

Test:

```python
import pytest


def test_model_windows_do_not_fall_back_to_raw_features():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "timestamp": pd.to_datetime(
                ["2020-01-01 09:30", "2020-01-01 09:35", "2020-01-01 09:40"]
            ),
            "split": ["train", "train", "train"],
            "f1": [1, 2, 3],
            "label": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="scaled"):
        build_windows_for_segment(frame, "train", feature_columns=["f1"], window_size=2)
```

### 9. Multi-Ticker Windows Stay Isolated

Target functions:

```python
from intraday_research.baseline_v1 import build_windows_by_ticker_and_split
```

Test:

```python
def test_windows_are_built_per_ticker_not_pooled_across_tickers():
    base = make_one_ticker_frame().head(4)
    base["split"] = "train"
    base["f1_scaled"] = [1, 2, 3, 4]
    base["label"] = [0, 1, 0, 1]
    other = base.copy()
    other["ticker"] = "BBB"
    other["f1_scaled"] = [10, 20, 30, 40]

    windows = build_windows_by_ticker_and_split(
        {"AAA": base, "BBB": other},
        feature_columns=["f1"],
        window_size=2,
    )

    assert set(windows) == {"AAA", "BBB"}
    assert windows["AAA"]["train"]["X"].shape[0] == windows["BBB"]["train"]["X"].shape[0]
    assert windows["AAA"]["train"]["X"][0, :, 0].tolist() == [1, 2]
    assert windows["BBB"]["train"]["X"][0, :, 0].tolist() == [10, 20]
```

### 10. Stratified Dummy Uses Train Labels Only

Target functions:

```python
from intraday_research.baseline_v1 import evaluate_stratified_dummy
```

Test:

```python
def test_stratified_dummy_uses_train_distribution_and_validation_labels():
    y_train = [0, 0, 0, 1]
    y_validation = [0, 1, 1, 1]

    result = evaluate_stratified_dummy(y_train, y_validation, seeds=(41, 42))

    assert set(result.columns) == {"seed", "macro_f1", "balanced_accuracy", "accuracy", "validation_n"}
    assert result["validation_n"].tolist() == [4, 4]
    assert result[["macro_f1", "balanced_accuracy", "accuracy"]].notna().all().all()
```

### 11. Active Helper Import Guard

Target files:

```text
intraday_research/baseline_v1.py
```

Test:

```python
import ast
from pathlib import Path


def test_active_helper_imports_do_not_reference_archived_paths():
    tree = ast.parse(Path("intraday_research/baseline_v1.py").read_text())
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    forbidden = {
        "archive",
        "legacy_model_runner",
        "ml_utils",
        "runner_utils",
        "train_test_split",
    }
    assert not (imported_names & forbidden)
```

## Execution Commands

After the tests and minimal helper file exist:

```powershell
E:\codex_workspace\_envs\py311_shared\python.exe -m pytest tests\test_baseline_v1_helpers.py -q
E:\codex_workspace\_envs\py311_shared\python.exe -m py_compile intraday_research\baseline_v1.py
git -C E:\codex_workspace\projects\intraday_stock_direction_research diff --check
```

Expected before implementation:

- pytest fails because `intraday_research.baseline_v1` does not exist or lacks
  functions.

Expected after minimal implementation:

- pytest passes.
- `py_compile` passes.
- `git diff --check` has no whitespace errors.

## Stop Rules

Stop before implementation if any proposed helper requires:

- importing archived runners or old helper packages;
- reading raw project data;
- reading checkpoints, artifacts, or prior validation outputs;
- opening or scoring closed holdout/test;
- changing notebook smoke reports or using their metrics as expected values;
- widening into LightGBM or MS-DLinear+TCN adapters.

## Recommended Next Step

Create the tests and the smallest `intraday_research/baseline_v1.py` skeleton in
one scoped P1 task. Run pytest once to confirm the tests fail for missing
implementation, then implement only the functions required by the tests.
