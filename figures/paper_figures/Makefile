env-check:
	python -c "import matplotlib, yaml, numpy; print('figure env ok')"

figures:
	python src/build_manuscript_figures.py

clean:
	python -c "from pathlib import Path; [p.unlink() for d in ['output/svg','output/pdf','output/png','output/pptx'] for p in Path(d).glob('Fig*') if p.is_file()]"
