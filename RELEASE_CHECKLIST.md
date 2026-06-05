# Release Checklist

This package is GitHub-ready but not yet publicly released. Do not publish until the human-only
items below are complete.

## Before GitHub Release

- [ ] Replace author placeholders in `CITATION.cff`, `.zenodo.json`, and `LICENSE` after author
  confirmation.
- [ ] Confirm the license with all co-authors.
- [ ] Run `powershell -ExecutionPolicy Bypass -File .\run_all.ps1` from the package root.
- [ ] Confirm the run regenerated key sensitivity audits, tests, figure exports, and
  `DERIVED_OUTPUT_CHECKSUMS.sha256`.
- [ ] Create or activate `https://github.com/Johnsonlijian/geostructural-reliability-ai-p1`.
- [ ] Push the repository and create a GitHub release, suggested tag `v1.0.0`.
- [ ] Archive the release with Zenodo/Figshare/OSF and record the DOI.
- [ ] Insert the final repository URL and DOI into the manuscript data/code availability statement.

## Do Not Release

- Raw third-party data.
- Active manuscripts, cover letters, reviewer-response files, internal rounds, logs, private
  author/funding files, or submission-system exports.
