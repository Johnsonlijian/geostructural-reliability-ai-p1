# Reproducibility - P1 Liquefaction Mechanistic-Compression Study

Deterministic analysis; no wet lab; public data only. The public package should distribute scripts, configuration, provenance records, derived non-sensitive outputs, and generated figures. Do not redistribute raw third-party data unless the source license explicitly permits it.

## Environment

- Python 3.11
- numpy 2.4
- pandas 3.0
- scikit-learn 1.8
- scipy 1.17
- matplotlib 3.10
- xlrd 2.0
- No GPU required

## Data Acquisition

- SPT: Cetin et al. (2018), Data in Brief, DOI `10.1016/j.dib.2018.08.043`. Use `mmc2.xls` sheet `CETIN_2018` under `data/raw/cetin2018_spt/mmc2.xls`.
- CPT: Geyin & Maurer (2021), DesignSafe DOI `10.17603/ds2-wftt-mv37`, PRJ-3012. Use the published MATLAB structure under `data/raw/geyin_maurer_2021_cpt/GLOBALDATASETV1.mat`.
- Provenance files: `data/raw/cetin2018_spt/SOURCE.md` and `data/raw/geyin_maurer_2021_cpt/SOURCE.md`.

Hard rule: no record without provenance enters the analysis; no value is fabricated or imputed as a result.

## Run Order

```powershell
# 1. Engine and snapshot tests
$env:PYTHONPATH="R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-GeoStructural-Reliability-AI\code"
python -m pytest tests -q

# 2. SPT pipeline, Cetin 2018
python run_baseline_cetin2018.py
python run_grouped_validation.py
python run_residual_fusion.py
python run_groundwater_ablation.py
python run_source_grouped.py

# 3. CPT pipeline, Geyin-Maurer 2021
python run_baseline_cpt_geyin.py

# 4. Reliability, sufficiency, bounded-claim audits, and robustness
python run_reliability_upgrade.py
python run_innovation_analysis.py
python run_sufficiency_likelihood.py
python run_residual_sufficiency_audit.py
python run_ambiguity_floor_sensitivity.py
python run_groundwater_residual_stratification.py
python run_sensitivity.py
python run_cross_region.py

# 5. Figures, 300 dpi PNG previews
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
- `reliability_upgrade.json`
- `innovation_analysis.json`
- `sufficiency_likelihood.json`
- `residual_sufficiency_audit.json`
- `ambiguity_floor_sensitivity.json`
- `groundwater_residual_stratification.json`
- `cross_region_transfer.json`
- `sensitivity.json`

Figures regenerate from the processed records and JSON outputs into `../figures/`.

## Determinism

All stochastic steps use fixed seeds (`random_state=0` or `default_rng(0)`). Bootstrap and conformal repetitions are fixed by script. Re-running the pipeline should reproduce the reported numbers within the exact deterministic settings encoded in the scripts.
