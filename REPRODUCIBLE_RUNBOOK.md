# Reproducible Runbook

## Environment

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Conda alternative:

```powershell
conda env create -f environment.yml
conda activate geostructural-reliability-ai-p1
```

## Raw Data

Raw third-party data are not included. Download the datasets listed in `DATASETS_AND_LINKS.csv` from
their official records and place them in the indicated `code/data/raw/` folders if you need to rerun
the full raw-to-processed pipeline.

Derived processed outputs are included under `code/data/processed/` for auditability.

The grouped stress-test and CPT 2024 current-release validation result files are included as derived
outputs. Re-running `run_cg_targeted_stress_tests.py` requires the original raw SPT workbook and CPT
source file because source, site, and event blocking use raw provenance fields that cannot be
redistributed in this public package. Re-running `run_cpt2024_external_validation.py` requires the
2024 DesignSafe `GLOBALDATASET.mat` file under `code/data/raw/rateria_geyin_maurer_2024_cpt/`.

## Test Suite

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation:

```text
31 passed
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
python run_random_split_sensitivity.py
python run_conformal_split_sensitivity.py
```

The decision-utility audit is computationally heavier because it repeats conformal policy
simulations. Run it separately when a full audit is needed:

```powershell
python run_conformal_decision_utility.py
```

After retrieving the required raw source files listed in `DATASETS_AND_LINKS.csv`, the grouped
stress test can be regenerated separately:

```powershell
python run_cg_targeted_stress_tests.py
```

After retrieving the 2024 DesignSafe CPT release, the CPT 2024 current-release validation upgrade can be
regenerated separately:

```powershell
python run_cpt2024_external_validation.py
```

## Figure Regeneration

From the package root:

```powershell
python figures\paper_figures\src\build_manuscript_figures.py
```

Outputs are written to `figures/paper_figures/output/svg/`, `pdf/`, `png/`, and preview `pptx/`.
The canonical submission-facing figure sources are the SVG/PDF files plus
`figures/paper_figures/src/build_manuscript_figures.py`, `figures/paper_figures/data/`, and
`figures/paper_figures/figure_provenance.yml`. PPTX files embed PNG previews and are not fully
editable vector source files.

## Package-Level Processed-Data Reproduction

From the package root:

```powershell
.\run_all.ps1
```

This regenerates the key derived sensitivity audits that do not require undistributed raw
provenance workbooks, runs tests, regenerates figures, and writes
`DERIVED_OUTPUT_CHECKSUMS.sha256`. The decision-utility audit is skipped by default to keep the
one-command check short and can be included with:

```powershell
.\run_all.ps1 -FullDecisionUtility
```

The raw-provenance grouped stress test and CPT 2024 current-release validation scripts are
intentionally excluded from the default one-command check.
If local PowerShell execution policy blocks scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

## Submission Boundary

Do not add active manuscripts, cover letters, reviewer responses, internal review rounds, or raw
third-party data to this repository. Those files remain private submission materials.
