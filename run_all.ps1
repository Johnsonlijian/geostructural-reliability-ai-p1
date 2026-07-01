param(
  [switch]$FullDecisionUtility,
  [switch]$R12Manuscript
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
  Push-Location code
  python run_ambiguity_floor_sensitivity.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  python run_conformal_decision_metrics.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if ($FullDecisionUtility) {
    python run_conformal_decision_utility.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  } else {
    Write-Host "Skipping run_conformal_decision_utility.py; use -FullDecisionUtility for the full decision-utility audit."
  }
  python run_random_split_sensitivity.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  python run_conformal_split_sensitivity.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  python -m pytest tests -q
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Pop-Location

  python figures\paper_figures\src\build_manuscript_figures.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if ($R12Manuscript) {
    Push-Location code
    python run_predictability_ceiling.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python run_tuned_ml_challenge.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python run_overvalidation_benchmark.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python run_value_of_information.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python run_geological_residual.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Pop-Location
    python code\make_figures_reframe.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  python scripts\write_checksums.py
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  Pop-Location
}
