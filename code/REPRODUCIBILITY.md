# Reproducibility - P1 Grouped Validation Stress-Test Study

Deterministic analysis; no wet lab; public data only. The public package should distribute scripts, configuration, provenance records, derived non-sensitive outputs, and generated figures. Do not redistribute raw third-party data unless the source license explicitly permits it.

## Environment

- Python 3.11
- Required packages are listed in `requirements.txt`.
- The release package also provides `environment.yml`.
- No GPU required

## Data Acquisition

- SPT: Cetin et al. (2018), Data in Brief, DOI `10.1016/j.dib.2018.08.043`. Use `mmc2.xls` sheet `CETIN_2018` under `data/raw/cetin2018_spt/mmc2.xls`.
- CPT 2021 continuity benchmark: Geyin & Maurer (2021), DesignSafe DOI `10.17603/ds2-wftt-mv37`, PRJ-3012. Use the published MATLAB structure under `data/raw/geyin_maurer_2021_cpt/GLOBALDATASETV1.mat`.
- CPT 2024 validation upgrade: Rateria, Geyin & Maurer (2024), DesignSafe DOI `10.17603/ds2-8hvd-hd43`, PRJ-5746. Use `GLOBALDATASET.mat` under `data/raw/rateria_geyin_maurer_2024_cpt/`.
- Nisqually fixed event caveat: Rasanen, Geyin & Maurer (2022), DesignSafe DOI `10.17603/ds2-nsf8-7944`, PRJ-3758; related article DOI `10.1177/87552930231174244`. Use only as a fixed held-out event caveat because the 2024 global release includes Nisqually.
- Release provenance: see the package-level `DATASETS_AND_LINKS.csv`.
- If rerunning from raw data locally, keep optional source notes beside the downloaded raw files.

Hard rule: no record without provenance enters the analysis; no value is fabricated or imputed as a result.

## Run Order

```powershell
# 1. Engine and snapshot tests
$env:PYTHONPATH=(Get-Location).Path
python -m pytest tests -q

# 2. SPT pipeline, Cetin 2018
python run_baseline_cetin2018.py
python run_grouped_validation.py
python run_residual_fusion.py
python run_groundwater_ablation.py
python run_source_grouped.py

# 3. CPT pipeline, Geyin-Maurer 2021
python run_baseline_cpt_geyin.py

# 4. Reliability, bounded-claim audits, and robustness
python run_reliability_upgrade.py
python run_innovation_analysis.py
python run_sufficiency_likelihood.py
python run_residual_sufficiency_audit.py
python run_ambiguity_floor_sensitivity.py
python run_groundwater_residual_stratification.py
python run_practical_equivalence_audit.py
python run_conformal_decision_metrics.py
python run_random_split_sensitivity.py
python run_conformal_split_sensitivity.py
python run_sensitivity.py
python run_cross_region.py

# 4b. C&G stress tests that require undistributed raw provenance files
python run_cg_targeted_stress_tests.py
python run_cpt2024_external_validation.py

# 5. Figures, including vector manuscript figures where available
python make_figures.py
python make_figures_innovation.py
python make_figures_mechanism.py
python make_figures_crossregion.py
```

## Key Outputs

Under `data/processed/`:

- `cetin2018_baseline_metrics.json`
- `cetin2018_baseline_records.csv`
- `cetin2018_grouped_validation.json`
- `cetin2018_source_grouped.json`
- `cetin2018_groundwater_ablation.json`
- `geyin2021_cpt_records.csv`
- `geyin2021_cpt_results.json`
- `cg_targeted_stress_tests.json`
- `cpt2024_records_inverse.csv`
- `cpt2024_records_measured.csv`
- `cpt2024_external_validation.json`
- `reliability_upgrade.json`
- `innovation_analysis.json`
- `sufficiency_likelihood.json`
- `residual_sufficiency_audit.json`
- `ambiguity_floor_sensitivity.json`
- `groundwater_residual_stratification.json`
- `practical_equivalence_audit.json`
- `conformal_decision_metrics.json`
- `random_split_sensitivity.json`
- `conformal_split_sensitivity.json`
- `cross_region_transfer.json`
- `sensitivity.json`

Figures regenerate from the processed records and JSON outputs into `../figures/`.

## Determinism

All stochastic steps use fixed seeds (`random_state=0` or `default_rng(0)`). Bootstrap and conformal repetitions are fixed by script. Re-running the pipeline should reproduce the reported numbers within the exact deterministic settings encoded in the scripts.
