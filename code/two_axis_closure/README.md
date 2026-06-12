# Two-Axis Closure Code

- `vs_kayen2013.py`: zero-shot Vs mechanism engine for Kayen et al. (2013) Eqs. 13-17 plus the 2015 erratum sign fix. Run directly to execute self-tests.
- `run_closure.py`: E0 data gate, E1 third-instrument grouped audit, and E2 SPT--Vs critical-band closure.
- `e2_closure.py`: SPT--Vs site-paired closure from the frozen label-blind audit list.
- `e2_closure2.py`: Moss-CPT axis, pooled two-axis closure, and label-agreement robustness run.

To rerun from source, place machine-readable `kayen2013_table_s1.csv` and `moss2006_table41.csv`
under `incoming/`. Raw third-party source files are not redistributed; see the root
`DATASETS_AND_LINKS.csv` and `DATA_NOTE.md`.

Outputs are written in this folder so the committed derived CSV/JSON files remain directly
auditable. Claim boundary: the closure test is pre-specified and label-blind before outcome joining,
not an external registry record.
