# Effective-Stress Coordinate Bounds And Conformal Decision Reliability For Liquefaction AI

This is the public reproducibility package for the paper:

**Effective-stress coordinates bound transferable machine-learning gains in liquefaction prediction**

The package contains code, tests, derived outputs, figure-generation scripts, figure specifications,
and publication-figure exports. It intentionally excludes active manuscript files, cover letters,
reviewer-response drafts, internal review notes, and raw third-party datasets whose redistribution
rights are not controlled by the authors.

## Contents

- `code/`: analysis scripts, mechanistic engine, tests, and derived processed outputs.
- `figures/paper_figures/source_scripts/`: scripts used to generate the composite paper figures.
- `figures/paper_figures/figure_specs/`: figure design briefs and evidence boundaries.
- `figures/paper_figures/output/`: SVG, PDF, and PNG figure outputs.
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

The package supports a bounded, reviewer-auditable claim: tested public predictors under
earthquake-grouped validation show a measured upper bound on recoverable gain beyond an
effective-stress margin. It does not claim mathematical sufficiency, a universal Bayes limit, or that
groundwater is physically unimportant; groundwater is treated as physically essential but largely
absorbed into the margin before residual learning is tested.

The `postliq_settlement` helper is retained as a tested utility for future mechanics extensions. It is
not used as evidence for the P1 manuscript conclusions.

## Quick Check

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation: `28 passed`.

For a package-level processed-data reproduction check:

```powershell
.\run_all.ps1
```

This regenerates the key derived sensitivity audits, runs the test suite, redraws manuscript figure
exports, and updates `DERIVED_OUTPUT_CHECKSUMS.sha256`.

If local PowerShell execution policy blocks scripts, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

## Repository Status

This is a local GitHub-ready package. Before public release, replace author placeholders in
`CITATION.cff`, confirm the license with all co-authors, create or activate the GitHub repository at
`https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`, and insert the final repository
URL/DOI into the manuscript.

The release and DOI steps are intentionally not automated in this repository because they require
human approval, author/license confirmation, and final submission timing.
