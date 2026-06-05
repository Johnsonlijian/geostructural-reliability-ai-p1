# Mechanistic-Coordinate Ceilings And Conformal Decision Reliability For Liquefaction AI

This is the public reproducibility package for the paper:

**Measured mechanistic-coordinate ceilings and conformal decision reliability for machine-learning
liquefaction prediction**

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

## Quick Check

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation: `26 passed`.

## Repository Status

This is a local GitHub-ready package. Before public release, replace author placeholders in
`CITATION.cff`, confirm the license with all co-authors, create or activate the GitHub repository at
`https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`, and insert the final repository
URL/DOI into the manuscript.
