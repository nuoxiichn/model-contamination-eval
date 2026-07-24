.PHONY: test lint validate

test:
	PYTHONPATH=src python -m pytest -q

lint:
	ruff check src tests scripts

validate:
	PYTHONPATH=src python -m model_contamination.cli validate-config --config configs/benchmarks.yaml
