#!/usr/bin/env bash
# Resume Gate §3 (docs/NOTEBOOK08_RESUME_GATES.md) — deterministic pass/fail
# check that says whether N08 Branch A "substantive work" is unblocked.
#
# This is a CHECK, not a FIX. It does not pip-install, edit settings, or
# move files. The first failing step IS the answer to the user's
# "why is N08 blocked?" question.
#
# Usage (run from anywhere):
#     bash scripts/check_n08_resume_gate.sh
#     PYTHON=/path/to/python bash scripts/check_n08_resume_gate.sh
#
# Exit codes:
#   0  Gate passed; substantive N08 work may proceed.
#   1  At least one phase failed; the first failing message tells you which.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-E:/codex_workspace/_envs/py311_shared/python.exe}"

cd "$PROJECT_ROOT" || { echo "FAIL: project root not found: $PROJECT_ROOT"; exit 1; }

# Phase 1 — package scaffold importable + scope intact
"$PYTHON" -c "import intraday_research; assert intraday_research.__research_scope__ == 'validation_only', 'wrong scope'" \
  || { echo "FAIL: Phase 1 (intraday_research package not importable, or scope drift)"; exit 1; }

# Phase 2 — contract module migrated
"$PYTHON" -c "from intraday_research.contracts.deep_sequence_exploration import validate_freeze_record, check_08o_real_readout_completeness" \
  || { echo "FAIL: Phase 2 (contract not migrated to src/intraday_research/contracts/)"; exit 1; }

# Phase 4 — stage entrypoint exists
"$PYTHON" -c "from intraday_research.stages.deep_sequence_exploration import run_stage" \
  || { echo "FAIL: Phase 4 (run_stage entrypoint missing in src/intraday_research/stages/)"; exit 1; }

# Phase 7 — notebook semantically renamed
test -f notebooks/deep_sequence_exploration_colab.ipynb \
  || { echo "FAIL: Phase 7 (notebook still at legacy path notebooks/08_*.ipynb)"; exit 1; }

# Post-migration tests must be green
"$PYTHON" -m pytest \
  tests/contracts/test_deep_sequence_exploration_contract.py \
  tests/notebooks/test_deep_sequence_exploration_static_gate.py \
  -q \
  || { echo "FAIL: post-migration tests not green"; exit 1; }

echo "GATE PASSED. Substantive N08 work may proceed."
echo "Entry point: src/intraday_research/stages/deep_sequence_exploration.py::run_stage"
