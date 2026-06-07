"""Scaffold subpackage for N08 deep-sequence model families.

Source: ``docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md``.

Families (section 7.1):
  - DLinear  (section 7.2)         in ``dlinear.py``
  - TCN      (section 7.3)         in ``tcn.py``
  - shallow GRU                    in ``gru.py``
  - shallow LSTM                   in ``lstm.py``
  - fusion variants (section 7.4)  in ``fusion.py``
  - controls (last-step LightGBM + last-step MLP ablation) in ``controls.py``

Shared training pieces:
  - losses (section 7.5)           in ``losses.py``
  - fold builders (section 8.2)    in ``folds.py``
  - sklearn-style protocol         in ``base.py``

Interface convention: every classifier class implements the
``SequenceClassifier`` protocol from ``base.py`` --
``fit(X, y) -> self`` and ``predict_proba(X) -> ndarray`` -- so the N08
stage orchestrator can swap families without family-specific code paths.

All ``fit`` / ``predict_proba`` bodies raise ``NotImplementedError`` until the
deep-model implementation half of task #4 lands.
"""

from intraday_research.models.deep_sequence.base import SequenceClassifier
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.tcn import TCNClassifier
from intraday_research.models.deep_sequence.gru import ShallowGRUClassifier
from intraday_research.models.deep_sequence.lstm import ShallowLSTMClassifier
from intraday_research.models.deep_sequence.fusion import (
    DLinearTrendPlusTCNResidualFusion,
    DLinearLogitsPlusTCNLogitsFusion,
    LateAverageProbabilitiesFusion,
    SmallFusionMLP,
)
from intraday_research.models.deep_sequence.controls import (
    LastStepLightGBMControl,
    LastStepMLPSequenceAblation,
)

__all__ = [
    "SequenceClassifier",
    "DLinearClassifier",
    "TCNClassifier",
    "ShallowGRUClassifier",
    "ShallowLSTMClassifier",
    "DLinearTrendPlusTCNResidualFusion",
    "DLinearLogitsPlusTCNLogitsFusion",
    "LateAverageProbabilitiesFusion",
    "SmallFusionMLP",
    "LastStepLightGBMControl",
    "LastStepMLPSequenceAblation",
]
