# Event-Transfer Limits In Liquefaction Case Histories

This is the public reproducibility package for the paper:

**Event-transfer limits in liquefaction case histories: effective-stress margins, CPT
transitional-band residuals and site-investigation value**

The package contains code, tests, derived outputs, figure-generation scripts, figure specifications,
and publication-figure exports. It intentionally excludes active manuscript files, cover letters,
reviewer-response drafts, internal review notes, and raw third-party datasets whose redistribution
rights are not controlled by the authors.

## Contents

- `code/`: analysis scripts, mechanistic engine, tests, and derived processed outputs.
- `code/run_predictability_ceiling.py`, `code/run_tuned_ml_challenge.py`,
  `code/run_overvalidation_benchmark.py`, `code/run_value_of_information.py`,
  `code/run_geological_residual.py`, and `code/run_cpt2024_external_validation.py`: scripts for
  the R12 manuscript's main numerical claims.
- `code/data/processed/`: derived non-sensitive processed tables and JSON outputs used by the
  manuscript and supplementary information.
- `figures/reframe_2026-06-30/`: R12 manuscript figure exports in SVG/PDF/PNG.
- `figures/paper_figures/`: earlier reusable figure-engineering assets retained for provenance.
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

The package supports bounded, reviewer-auditable claims: random splits inflate apparent
machine-learning skill; under earthquake-grouped validation the published effective-stress margin is
not detectably exceeded by the tested tuned challengers; the CPT manifestation residual floor is
geologically structured, especially in the `Ic`-defined transitional silt-sand band; and the
fixed-rule decision value of the margin is positive while added standard predictors contribute
little cross-event decision value beyond that margin. The package does not claim a universal Bayes
limit, a site-specific calibrated decision instrument, or that multi-method site investigation is
unnecessary.

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

The public repository is
`https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`. The existing public GitHub
release `v0.1.0-submission` was verified on 2026-07-01 but is not R12-complete. The R12 source
release candidate `v0.2.0-r12-submission` was published on 2026-07-01; this metadata-aligned
package revision is released as `v0.2.1-r12-submission` after the manuscript title and claim
boundaries were tightened. A separate archival DOI has not been verified and should be added only
after a Zenodo/OSF-style archive is minted.
