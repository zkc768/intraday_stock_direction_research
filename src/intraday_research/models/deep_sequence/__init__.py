"""N08 deep-sequence model family package.

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

Interface convention: every classifier class implements the ``SequenceClassifier``
protocol from ``base.py`` -- ``fit(X, y) -> self`` and
``predict_proba(X) -> ndarray`` -- so the N08 stage orchestrator can swap
families without family-specific code paths.

Implementation status is family-specific and lands incrementally. Heavy model
modules are exported lazily so importing lightweight helpers such as
``folds.py`` or ``base.py`` does not import torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_LAZY_EXPORTS = {
    "SequenceClassifier": "intraday_research.models.deep_sequence.base",
    "DLinearClassifier": "intraday_research.models.deep_sequence.dlinear",
    "TCNClassifier": "intraday_research.models.deep_sequence.tcn",
    "ShallowGRUClassifier": "intraday_research.models.deep_sequence.gru",
    "ShallowLSTMClassifier": "intraday_research.models.deep_sequence.lstm",
    "DLinearTrendPlusTCNResidualFusion": (
        "intraday_research.models.deep_sequence.fusion"
    ),
    "DLinearLogitsPlusTCNLogitsFusion": (
        "intraday_research.models.deep_sequence.fusion"
    ),
    "LateAverageProbabilitiesFusion": "intraday_research.models.deep_sequence.fusion",
    "SmallFusionMLP": "intraday_research.models.deep_sequence.fusion",
    "LastStepLightGBMControl": "intraday_research.models.deep_sequence.controls",
    "LastStepMLPSequenceAblation": "intraday_research.models.deep_sequence.controls",
}


def __getattr__(name: str) -> Any:
    """Load public model exports on first access.

    This keeps ``import intraday_research.models.deep_sequence.folds`` torch-free
    while preserving ``from intraday_research.models.deep_sequence import
    DLinearClassifier`` for orchestrator code.
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
