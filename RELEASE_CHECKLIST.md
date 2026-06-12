# Release Checklist

This package is GitHub-ready. It intentionally excludes raw third-party data, active manuscript
files, cover letters, internal rounds, logs, and private submission materials.

## Before GitHub Release

- [x] Replace author placeholders in `CITATION.cff` and `.zenodo.json`.
- [x] Confirm the license for the single-author package.
- [ ] Run `powershell -ExecutionPolicy Bypass -File .\run_all.ps1` from the package root if a full
  local regeneration is required before release.
- [x] Confirm the latest figure/export refresh regenerated key figure exports and
  `DERIVED_OUTPUT_CHECKSUMS.sha256`.
- [ ] Create or activate `https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`.
- [ ] Push the repository and create a GitHub release, suggested tag `v1.0.0`.
- [ ] Archive the release with Zenodo/Figshare/OSF and record the DOI.
- [ ] Insert the final repository URL and DOI into the manuscript data/code availability statement
  only after the double-blind review constraint no longer applies.

## Do Not Release

- Raw third-party data.
- Active manuscripts, cover letters, reviewer-response files, internal rounds, logs, private
  author/funding files, or submission-system exports.
