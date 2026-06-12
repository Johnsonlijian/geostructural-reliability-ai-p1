.PHONY: derived test figures checksums cg-stress cpt2024 all

derived:
	cd code && python run_ambiguity_floor_sensitivity.py
	cd code && python run_conformal_decision_metrics.py
	cd code && python run_conformal_decision_utility.py
	cd code && python run_random_split_sensitivity.py
	cd code && python run_conformal_split_sensitivity.py

cg-stress:
	cd code && python run_cg_targeted_stress_tests.py

cpt2024:
	cd code && python run_cpt2024_external_validation.py

test:
	cd code && python -m pytest tests -q

figures:
	python figures/paper_figures/src/build_manuscript_figures.py

checksums:
	python scripts/write_checksums.py

all: derived test figures checksums
