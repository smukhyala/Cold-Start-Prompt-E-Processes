PY ?= python3.13
VENV := .venv

.PHONY: setup test pilot smoke clean lint

setup:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

test:
	$(VENV)/bin/pytest

pilot:
	$(VENV)/bin/python -m cold_start.cli.run --config configs/pilot_toy.yaml

smoke:
	$(VENV)/bin/python -m cold_start.cli.smoke

lint:
	$(VENV)/bin/ruff check src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
