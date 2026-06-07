"""Compatibility shim — canonical code lives in
``intraday_research.contracts.validation_synthesis_gap_audit``.

See ``docs/LEGACY_NAME_MAPPING.md`` for the legacy <-> target path mapping.

All top-level public and private names are re-exported explicitly so existing
callers (notebooks, tests, generators) continue to resolve ``c.<name>`` and
``c._<name>`` without modification.
"""

from intraday_research.contracts.validation_synthesis_gap_audit import (  # noqa: F401
    ALLOWED_ROW_CLASS_VALUES,
    CONCENTRATION_WARNING_POSITIVE_TICKER_COUNT_MIN,
    CONCENTRATION_WARNING_TOP_TICKER_SHARE_MAX,
    CONDITIONAL_FINAL_COMPARISON_COLUMNS,
    FORBIDDEN_PHRASE_REGEX,
    FORBIDDEN_PHRASE_REGEX_PATTERN,
    IMPROVEMENT_LCB_MIN,
    IMPROVEMENT_TICKER_COUNT_MIN,
    NOTEBOOK07_SCOPE,
    NULL_CONTROL_ALPHA_TOTAL,
    REQUIRED_FINAL_COMPARISON_COLUMNS,
    REQUIRED_LEDGER_COLUMNS,
    REQUIRED_THESIS_KIT_FIELDS,
    REQUIRED_THRESHOLD_CHECK_FIELDS,
    SAME_ROW_DUMMY_REQUIRED_NON_NULL_COLUMNS,
    WEAK_SEED_EVIDENCE_COUNT_THRESHOLD,
    WEAK_SIGNAL_BAND_LOWER,
    WEAK_SIGNAL_BAND_UPPER,
    _LABEL_CONFIG_PATTERN,
    parse_label_config,
    validate_final_validation_comparison_frame,
    validate_ledger_frame,
    validate_ledger_prefix_invariance,
    validate_thesis_paragraph_kit,
)
