"""Interface contract tests for the N08 deep-sequence model subpackage scaffold.

Confirms every section 7.1 classifier family / section 7.4 fusion variant:
  - satisfies the ``SequenceClassifier`` runtime-checkable protocol;
  - raises ``NotImplementedError`` on ``fit`` and ``predict_proba`` with a
    message anchoring to the design / Resume Gate.

These tests do not exercise any substantive training behavior; that is the
implementation half of N08 task #4 and is gated on Resume Gate section 3.
"""

import numpy as np
import pytest

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.controls import (
    LastStepLightGBMControl,
    LastStepMLPSequenceAblation,
)
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.fusion import (
    DLinearLogitsPlusTCNLogitsFusion,
    DLinearTrendPlusTCNResidualFusion,
    LateAverageProbabilitiesFusion,
    SmallFusionMLP,
)
from intraday_research.models.deep_sequence.gru import ShallowGRUClassifier
from intraday_research.models.deep_sequence.lstm import ShallowLSTMClassifier
from intraday_research.models.deep_sequence.tcn import TCNClassifier


# All 10 families satisfy the SequenceClassifier protocol regardless of
# whether their fit/predict_proba bodies are implemented yet.
_ALL_FAMILIES = [
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
]

# Subset that still raises NotImplementedError on fit / predict_proba. As
# families land in #5A, #5C, ..., they move out of this list and into their
# own behavioral test file (e.g. test_last_step_lightgbm_control.py for #5A).
_NOT_YET_IMPLEMENTED_FAMILIES = [
    ShallowGRUClassifier,
    ShallowLSTMClassifier,
    LastStepMLPSequenceAblation,
    DLinearTrendPlusTCNResidualFusion,
    DLinearLogitsPlusTCNLogitsFusion,
    LateAverageProbabilitiesFusion,
    SmallFusionMLP,
]


@pytest.mark.parametrize("cls", _ALL_FAMILIES)
def test_family_satisfies_sequence_classifier_protocol(cls):
    """Every family must structurally implement ``fit`` and ``predict_proba``."""
    instance = cls()
    assert isinstance(instance, SequenceClassifier), (
        f"{cls.__name__} does not satisfy SequenceClassifier protocol "
        "(missing fit and / or predict_proba)"
    )


@pytest.mark.parametrize("cls", _NOT_YET_IMPLEMENTED_FAMILIES)
def test_fit_raises_not_implemented_with_scaffold_anchor(cls):
    instance = cls()
    X = np.zeros((4, 20, 3), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(NotImplementedError, match="scaffold"):
        instance.fit(X, y)


@pytest.mark.parametrize("cls", _NOT_YET_IMPLEMENTED_FAMILIES)
def test_predict_proba_raises_not_implemented_with_scaffold_anchor(cls):
    instance = cls()
    X = np.zeros((4, 20, 3), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="scaffold"):
        instance.predict_proba(X)


def test_tcn_rejects_non_causal_construction():
    """TCN must refuse causal=False at construction time per AGENTS.md section 4.1."""
    with pytest.raises(ValueError, match="causal"):
        TCNClassifier(causal=False)


def test_sequence_classifier_protocol_is_runtime_checkable():
    """Protocol must be @runtime_checkable so the orchestrator can iterate families."""
    # An object without fit / predict_proba must NOT satisfy the protocol.
    class _Bare:
        pass
    assert not isinstance(_Bare(), SequenceClassifier)
