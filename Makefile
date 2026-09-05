.PHONY: install install-dev test lint compile-check clean

install:
	pip install -e .
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

compile-check:
	python -m compileall -q src experiments

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -maxdepth 1 -type d -name "chroma_*" -exec rm -rf {} +
