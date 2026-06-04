# Mechanistic Sufficiency and Reliable Prediction of Seismic Liquefaction

This is the public reproducibility package for the paper:

**Mechanistic sufficiency and reliable prediction of seismic liquefaction**

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

## Quick Check

```powershell
Set-Location code
python -m pytest tests -q
```

Expected result at package preparation: `25 passed`.

## Repository Status

This is a local GitHub-ready package. Before public release, replace author placeholders in
`CITATION.cff`, confirm the license with all co-authors, create the GitHub repository under
`https://github.com/Johnsonlijian/`, and insert the final repository URL/DOI into the manuscript.
