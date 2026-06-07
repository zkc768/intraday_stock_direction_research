# Resume Gate §3 (docs\NOTEBOOK08_RESUME_GATES.md) — Windows PowerShell sibling
# of scripts\check_n08_resume_gate.sh.
#
# This is a CHECK, not a FIX. It does not pip-install, edit settings, or
# move files. The first failing step IS the answer to the user's
# "why is N08 blocked?" question.
#
# Usage (run from anywhere):
#     powershell.exe -File scripts\check_n08_resume_gate.ps1
#     $env:PYTHON = 'C:\path\to\python.exe'; powershell.exe -File scripts\check_n08_resume_gate.ps1
#
# Exit codes mirror the bash sibling: 0 pass, 1 fail at first failing phase.

$ErrorActionPreference = 'Stop'

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir '..')
$Python      = if ($env:PYTHON) { $env:PYTHON } else { 'E:\codex_workspace\_envs\py311_shared\python.exe' }

Set-Location $ProjectRoot
if (-not $?) { Write-Error 'FAIL: project root not found'; exit 1 }

& $Python -c "import intraday_research; assert intraday_research.__research_scope__ == 'validation_only'"
if (-not $?) { Write-Error 'FAIL: Phase 1 (package not importable or scope drift)'; exit 1 }

& $Python -c "from intraday_research.contracts.deep_sequence_exploration import validate_freeze_record, check_08o_real_readout_completeness"
if (-not $?) { Write-Error 'FAIL: Phase 2 (contract not migrated)'; exit 1 }

& $Python -c "from intraday_research.stages.deep_sequence_exploration import run_stage"
if (-not $?) { Write-Error 'FAIL: Phase 4 (run_stage missing)'; exit 1 }

if (-not (Test-Path 'notebooks\deep_sequence_exploration_colab.ipynb')) {
    Write-Error 'FAIL: Phase 7 (notebook not renamed)'; exit 1
}

& $Python -m pytest tests\contracts\test_deep_sequence_exploration_contract.py tests\notebooks\test_deep_sequence_exploration_static_gate.py -q
if (-not $?) { Write-Error 'FAIL: post-migration tests not green'; exit 1 }

Write-Host 'GATE PASSED. Substantive N08 work may proceed.'
Write-Host 'Entry point: src\intraday_research\stages\deep_sequence_exploration.py::run_stage'
