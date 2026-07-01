# Release Checklist

This package is GitHub-ready. It intentionally excludes raw third-party data, active manuscript
files, cover letters, internal rounds, logs, and private submission materials.

## R12 GitHub Release State

- [x] Replace author placeholders in `CITATION.cff` and `.zenodo.json`.
- [x] Confirm the license for the single-author package.
- [x] Confirm `https://github.com/Johnsonlijian/geostructural-reliability-ai-p1` exists and public
  release `v0.1.0-submission` exists.
- [x] Publish the R12 source release candidate `v0.2.0-r12-submission` on GitHub.
- [x] Prepare the metadata-aligned R12 source release candidate `v0.2.1-r12-submission` after the
  manuscript title and claim boundaries were tightened.
- [ ] Run `powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -R12Manuscript` from the package
  root if a full local regeneration is required before release.
- [ ] Confirm the R12 figure/export refresh regenerated key figure exports and
  `DERIVED_OUTPUT_CHECKSUMS.sha256`.
- [x] Push the metadata-aligned R12 reproducibility update and create the GitHub release tag
  `v0.2.1-r12-submission`.
- [ ] Archive the release with Zenodo/Figshare/OSF and record the DOI.
- [ ] Insert the final repository URL and DOI into the manuscript data/code availability statement
  only after the R12-complete release/DOI is actually available. If no archival DOI exists, cite
  only the verified GitHub release URL and state that no DOI is asserted.

## Do Not Release

- Raw third-party data.
- Active manuscripts, cover letters, reviewer-response files, internal rounds, logs, private
  author/funding files, or submission-system exports.
