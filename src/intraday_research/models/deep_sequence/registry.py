"""Family -> SequenceClassifier registry for the 08X trial runner (#5F-5).

#5F-5 registers ``dlinear_only`` only. #5F-6 (the quick-search loop) extends this
to the other SEARCH_ELIGIBLE families (incl. ``last_step_lightgbm_control``).
"""

from __future__ import annotations

from typing import Any

from intraday_research.models.deep_sequence.dlinear import DLinearClassifier


SEQUENCE_CLASSIFIER_REGISTRY: dict[str, type] = {
    "dlinear_only": DLinearClassifier,
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
