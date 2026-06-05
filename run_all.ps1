$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
  Push-Location code
  python run_ambiguity_floor_sensitivity.py
  python run_random_split_sensitivity.py
  python run_conformal_split_sensitivity.py
  python -m pytest tests -q
  Pop-Location

  python figures\paper_figures\source_scripts\fig1_central_mechanism_evidence.py
  python figures\paper_figures\source_scripts\fig2_3_4_composites.py
  python scripts\write_checksums.py
} finally {
  Pop-Location
}
