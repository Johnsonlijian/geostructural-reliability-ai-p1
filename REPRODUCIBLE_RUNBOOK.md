# Reproducible Runbook

## Environment

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Raw Data

Raw third-party data are not included. Download the datasets listed in `DATASETS_AND_LINKS.csv` from
their official records and place them in the indicated `code/data/raw/` folders if you need to rerun
the full raw-to-processed pipeline.

Derived processed outputs are included under `code/data/processed/` for auditability.

## Test Suite

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation:

```text
26 passed
```

## Analysis Pipeline

From `code/`, run:

```powershell
python run_baseline_cetin2018.py
python run_grouped_validation.py
python run_groundwater_ablation.py
python run_residual_fusion.py
python run_source_grouped.py
python run_baseline_cpt_geyin.py
python run_sufficiency_likelihood.py
python run_residual_sufficiency_audit.py
python run_ambiguity_floor_sensitivity.py
python run_innovation_analysis.py
python run_reliability_upgrade.py
python run_sensitivity.py
python run_cross_region.py
python run_groundwater_residual_stratification.py
python run_practical_equivalence_audit.py
python run_conformal_decision_metrics.py
```

## Figure Regeneration

From the package root:

```powershell
python figures\paper_figures\source_scripts\fig1_central_mechanism_evidence.py
python figures\paper_figures\source_scripts\fig2_3_4_composites.py
```

Outputs are written to `figures/paper_figures/output/svg/`, `pdf/`, and `png/`.

## Submission Boundary

Do not add active manuscripts, cover letters, reviewer responses, internal review rounds, or raw
third-party data to this repository. Those files remain private submission materials.
