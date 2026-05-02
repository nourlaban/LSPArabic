.PHONY: install install-dev install-synthesis preprocess train evaluate infer synthesize test lint typecheck clean

PYTHON := python
SCRIPTS := scripts

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-synthesis:
	pip install -e ".[synthesis]"

preprocess:
	$(PYTHON) $(SCRIPTS)/preprocess.py

train:
	$(PYTHON) $(SCRIPTS)/train.py

evaluate:
	$(PYTHON) $(SCRIPTS)/evaluate.py

infer:
	$(PYTHON) $(SCRIPTS)/infer.py

synthesize:
	$(PYTHON) $(SCRIPTS)/synthesize.py

test:
	pytest tests/ --cov=src/lsparabic --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -s

lint:
	ruff check src/ scripts/ tests/

typecheck:
	mypy src/lsparabic/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
