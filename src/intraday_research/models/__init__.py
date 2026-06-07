"""Reusable model implementations for the intraday research project.

Sub-packages:
  - ``deep_sequence``  Deep sequence model families consumed by stage
                       ``deep_sequence_exploration`` (N08). See
                       ``docs/NOTEBOOK08_DEEP_SEQUENCE_EXPLORATION_FREEZE_READOUT_TECHNICAL_DESIGN_2026-06-06.md``
                       sections 7.1-7.5 and 8.2.

Substantive model code is the second half of N08 task #4 and is gated on
Resume Gate Phase 1+2+4+7 passing
(``docs/NOTEBOOK08_RESUME_GATES.md`` section 3). Until that half lands, each
classifier raises ``NotImplementedError`` on ``fit`` / ``predict_proba``.
"""
