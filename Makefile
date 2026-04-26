.PHONY: install test lint format clean check preprocess transfer-dataset train-phase2 smoke

# Variables
POETRY = poetry
PYTHON = $(POETRY) run python
RUFF = $(POETRY) run ruff check
RUFF_FORMAT = $(POETRY) run ruff format
PYTEST = $(POETRY) run pytest

install:
	$(POETRY) install

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint:
	$(RUFF) src/ tests/

format:
	$(RUFF_FORMAT) src/ tests/

check: lint test

clean:
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

preprocess:
	$(PYTHON) src/data/make_dataset.py

transfer-dataset:
	$(PYTHON) -m src.data.transfer_dataset --output-dir data/processed

train-phase2:
	$(PYTHON) -m src.models.train_model --dataset-path data/processed/transfer_modeling_dataset.csv --output-dir models/generated

smoke:
	$(PYTEST) tests/test_phase2_pipeline.py -v
