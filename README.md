# Grouped Validation Stress Testing For Infrastructure Hazard Machine Learning

This is the public reproducibility package for the paper:

**Physics-anchored validation reveals limits of infrastructure hazard machine learning**

The package contains code, tests, derived outputs, figure-generation scripts, figure specifications,
and publication-figure exports. It intentionally excludes active manuscript files, cover letters,
reviewer-response drafts, internal review notes, and raw third-party datasets whose redistribution
rights are not controlled by the authors.

## Contents

- `code/`: analysis scripts, mechanistic engine, tests, and derived processed outputs.
- `figures/paper_figures/src/build_manuscript_figures.py`: deterministic script used to generate the six
  manuscript figures.
- `figures/paper_figures/data/`: source-data mirrors for the displayed figure values.
- `figures/paper_figures/figure_provenance.yml`: figure provenance, AI-image, preview-PPTX, and
  author-signoff boundary notes.
- `figures/paper_figures/output/`: SVG, PDF, PNG, and PPTX-preview figure outputs.
- `DATASETS_AND_LINKS.csv`: official source records for raw data retrieval.
- `REPRODUCIBLE_RUNBOOK.md`: end-to-end reproduction commands.
- `environment.yml`: Conda environment specification.
- `DERIVED_OUTPUT_CHECKSUMS.sha256`: checksums for derived processed outputs and figure exports.
- `RELEASE_CHECKLIST.md`: human-only release and DOI checklist.

## Data Boundary

Raw third-party datasets are not included. Users should retrieve them from the official records listed
in `DATASETS_AND_LINKS.csv`, then place them under the local paths described in the runbook. Derived
processed outputs are included so that reported tables and figures can be inspected without
redistributing the raw sources.

## Evidence Boundary

The package supports a bounded, reviewer-auditable claim: an effective-stress margin is an
operational reference coordinate for grouped validation and conformal reliability in public
liquefaction case histories. CPT manifestation remains margin-dominated in the 2021 continuity
benchmark and the superseding 2024 DesignSafe release under strong grouped machine-learning stress tests, while
SPT source blocking retains a disclosed residual-fusion caveat. The fixed leave-Nisqually-out
evaluation is included as a single-event caveat, not as independent validation. The package does not
claim mathematical sufficiency, a universal Bayes limit, or that groundwater is physically
unimportant; groundwater is treated as physically essential but largely absorbed into the margin
before residual learning is tested.

The slope-failure vignette is represented by derived non-sensitive rows in
`code/data/processed/slope_vignette_r12.csv` and uncertainty summaries in
`code/data/processed/slope_vignette_uncertainty_r12.csv`. It supports protocol traceability beyond
liquefaction as a bounded portability check.

The `postliq_settlement` helper is retained as a tested utility for future mechanics extensions. It is
not used as evidence for the P1 manuscript conclusions.

## Quick Check

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation: `31 passed`.

For a package-level processed-data reproduction check:

```powershell
.\run_all.ps1
```

This regenerates the key derived sensitivity audits, runs the test suite, redraws the six manuscript
figure exports, and updates `DERIVED_OUTPUT_CHECKSUMS.sha256`. The computationally heavier
decision-utility audit can be rerun with:

```powershell
.\run_all.ps1 -FullDecisionUtility
```

The grouped stress-test and CPT 2024 current-release validation scripts are kept in `code/` and their
derived outputs are included, but they are not part of the default one-command check because full
regeneration requires raw third-party data that are not redistributed.

If local PowerShell execution policy blocks scripts, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

## Repository Status

This package is prepared for public release at
`https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`. The author metadata, license,
source registry, runbook, figure provenance, derived outputs, and checksums have been prepared for a
submission-stage release. The repository intentionally excludes raw third-party data and active
submission manuscripts.

For double-blind review, use the anonymized reviewer package supplied with the manuscript submission.
The public repository URL and archival DOI should be inserted into the manuscript only after the
double-blind review constraint no longer applies.
