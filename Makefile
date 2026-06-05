.PHONY: derived test figures checksums all

derived:
	cd code && python run_ambiguity_floor_sensitivity.py
	cd code && python run_random_split_sensitivity.py
	cd code && python run_conformal_split_sensitivity.py

test:
	cd code && python -m pytest tests -q

figures:
	python figures/paper_figures/source_scripts/fig1_central_mechanism_evidence.py
	python figures/paper_figures/source_scripts/fig2_3_4_composites.py

checksums:
	python scripts/write_checksums.py

all: derived test figures checksums
