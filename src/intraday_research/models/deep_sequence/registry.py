"""Family -> SequenceClassifier registry for the 08X trial runner (#5F-5 / #5F-6).

#5F-5 registered ``dlinear_only`` only. #5F-6 (the quick-search loop) extends this
to the remaining SEARCH_ELIGIBLE families:

  - ``tcn_only``                       -> TCNClassifier
  - ``ms_dlinear_tcn``                 -> DLinearTrendPlusTCNResidualFusion (the
                                          jointly-trained causal fusion module;
                                          Codex #5F-6 Q2 -- the late-fusion WRAPPER
                                          variants are deferred)
  - ``last_step_mlp_sequence_ablation``-> LastStepMLPSequenceAblation
  - ``last_step_lightgbm_control``     -> LastStepLightGBMControl (no torch base;
                                          no ``max_epochs`` -- the runner getattr-
                                          guards that field, #5F-6 Q1)

``shallow_gru`` / ``shallow_lstm`` remain section-7.1 candidate families but are
NOT 08X-search-eligible until their axes are frozen, so they stay unregistered;
``build_classifier`` raises for them (and any unknown family).
"""

from __future__ import annotations

from typing import Any

from intraday_research.models.deep_sequence.controls import (
    LastStepLightGBMControl,
    LastStepMLPSequenceAblation,
)
from intraday_research.models.deep_sequence.dlinear import DLinearClassifier
from intraday_research.models.deep_sequence.fusion import (
    DLinearTrendPlusTCNResidualFusion,
)
from intraday_research.models.deep_sequence.tcn import TCNClassifier


SEQUENCE_CLASSIFIER_REGISTRY: dict[str, type] = {
    "dlinear_only": DLinearClassifier,
    "tcn_only": TCNClassifier,
    "ms_dlinear_tcn": DLinearTrendPlusTCNResidualFusion,
    "last_step_mlp_sequence_ablation": LastStepMLPSequenceAblation,
    "last_step_lightgbm_control": LastStepLightGBMControl,
}


def build_classifier(family: str, *, random_state: int, **config: Any):
    """Instantiate the registered classifier for ``family`` with ``random_state``.

    Raises ``ValueError`` for an unregistered family.
    """
    if family not in SEQUENCE_CLASSIFIER_REGISTRY:
        raise ValueError(
            f"unknown candidate_family {family!r}; "
            f"known: {sorted(SEQUENCE_CLASSIFIER_REGISTRY)}"
        )
    return SEQUENCE_CLASSIFIER_REGISTRY[family](random_state=random_state, **config)
