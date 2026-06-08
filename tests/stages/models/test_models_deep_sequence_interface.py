"""Interface contract tests for the N08 deep-sequence model subpackage.

Confirms every section 7.1 classifier family / section 7.4 fusion variant
satisfies the ``SequenceClassifier`` runtime-checkable protocol.

All 10 families now have substantive bodies (their behavior is covered by their
own test files: ``test_dlinear.py``, ``test_tcn.py``, ``test_gru.py``,
``test_lstm.py``, ``test_fusion.py``, ``test_last_step_lightgbm_control.py``,
``test_last_step_mlp_ablation.py``), so ``_NOT_YET_IMPLEMENTED_FAMILIES`` is now
empty; the scaffold-``NotImplementedError`` anchor tests below parametrize over
it and therefore no longer run any case (kept as a regression tripwire — a new
scaffold added to the list would re-activate them).
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

# Subset that still raises NotImplementedError on fit / predict_proba. Now EMPTY
# — all 10 families are implemented (each has its own behavioral test file). Kept
# as a regression tripwire: a newly added scaffold goes here to re-arm the
# NotImplementedError-anchor tests below.
_NOT_YET_IMPLEMENTED_FAMILIES: list = []


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
