"""Import smoke tests for the N08 deep-sequence model subpackage scaffold.

Confirms every family / loss / fold builder declared by section 7.1-8.2 is
importable. Does not exercise any substantive behavior; those tests live in
the implementation half of N08 task #4.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_models_subpackage_imports():
    import intraday_research.models  # noqa: F401


def test_deep_sequence_subpackage_imports():
    import intraday_research.models.deep_sequence  # noqa: F401


def test_lightweight_folds_import_does_not_load_torch():
    """Importing non-torch helpers must not eagerly import torch model bodies."""
    code = (
        "import sys\n"
        "import intraday_research.models.deep_sequence.folds\n"
        "raise SystemExit(1 if 'torch' in sys.modules else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_each_classifier_family_importable():
    from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
    from intraday_research.models.deep_sequence.tcn import TCNClassifier
    from intraday_research.models.deep_sequence.gru import ShallowGRUClassifier
    from intraday_research.models.deep_sequence.lstm import ShallowLSTMClassifier
    from intraday_research.models.deep_sequence.controls import (
        LastStepLightGBMControl,
        LastStepMLPSequenceAblation,
    )
    from intraday_research.models.deep_sequence.fusion import (
        DLinearTrendPlusTCNResidualFusion,
        DLinearLogitsPlusTCNLogitsFusion,
        LateAverageProbabilitiesFusion,
        SmallFusionMLP,
    )
    classes = (
        DLinearClassifier,
        TCNClassifier,
        ShallowGRUClassifier,
        ShallowLSTMClassifier,
        LastStepLightGBMControl,
        LastStepMLPSequenceAblation,
        DLinearTrendPlusTCNResidualFusion,
        DLinearLogitsPlusTCNLogitsFusion,
        LateAverageProbabilitiesFusion,
        SmallFusionMLP,
    )
    # Constructor must succeed; substantive behavior is deferred to fit / predict_proba.
    for cls in classes:
        cls()


def test_losses_module_exports_five_named_losses():
    from intraday_research.models.deep_sequence import losses
    expected = {
        "cross_entropy_loss",
        "weighted_cross_entropy_train_prior_loss",
        "focal_loss",
        "class_balanced_loss_effective_number",
        "balanced_softmax_loss",
    }
    missing = expected - set(dir(losses))
    assert not missing, f"losses.py missing scaffolds for: {sorted(missing)}"


def test_folds_module_exports_three_named_builders():
    from intraday_research.models.deep_sequence import folds
    expected = {
        "rolling_origin_folds",
        "purged_time_series_folds",
        "embargoed_train_inner_folds",
    }
    missing = expected - set(dir(folds))
    assert not missing, f"folds.py missing scaffolds for: {sorted(missing)}"


def test_top_level_reexports_match_design_section_7_1():
    """__init__.py must re-export every section 7.1 family + section 7.4 fusion variant."""
    from intraday_research.models import deep_sequence as ds
    expected_classifier_attrs = {
        "DLinearClassifier",
        "TCNClassifier",
        "ShallowGRUClassifier",
        "ShallowLSTMClassifier",
        "LastStepLightGBMControl",
        "LastStepMLPSequenceAblation",
        "DLinearTrendPlusTCNResidualFusion",
        "DLinearLogitsPlusTCNLogitsFusion",
        "LateAverageProbabilitiesFusion",
        "SmallFusionMLP",
        "SequenceClassifier",
    }
    missing = expected_classifier_attrs - set(ds.__all__)
    assert not missing, f"__init__.py missing exports for: {sorted(missing)}"


def test_top_level_lazy_reexports_are_resolvable():
    from intraday_research.models import deep_sequence as ds

    assert ds.DLinearClassifier.__name__ == "DLinearClassifier"
    assert ds.TCNClassifier.__name__ == "TCNClassifier"
    assert ds.ShallowGRUClassifier.__name__ == "ShallowGRUClassifier"
    assert ds.ShallowLSTMClassifier.__name__ == "ShallowLSTMClassifier"


@pytest.mark.parametrize(
    "module_name,cls_name",
    [
        ("dlinear", "DLinearClassifier"),
        ("tcn", "TCNClassifier"),
        ("gru", "ShallowGRUClassifier"),
        ("lstm", "ShallowLSTMClassifier"),
        ("controls", "LastStepLightGBMControl"),
        ("controls", "LastStepMLPSequenceAblation"),
        ("fusion", "DLinearTrendPlusTCNResidualFusion"),
        ("fusion", "DLinearLogitsPlusTCNLogitsFusion"),
        ("fusion", "LateAverageProbabilitiesFusion"),
        ("fusion", "SmallFusionMLP"),
    ],
)
def test_each_module_exposes_documented_class(module_name, cls_name):
    import importlib
    mod = importlib.import_module(f"intraday_research.models.deep_sequence.{module_name}")
    assert hasattr(mod, cls_name), f"{module_name}.py missing class {cls_name}"
