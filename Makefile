VENV_DIR ?= .venv
PYTHON ?= python3

install:
	uv sync

venv:
	$(PYTHON) -m venv $(VENV_DIR)

activate:
	@echo "Run: source $(VENV_DIR)/bin/activate"

deps:
	$(VENV_DIR)/bin/python -m pip install -U pip
	$(VENV_DIR)/bin/python -m pip install -e .

parse:
	PYTHONPATH=. uv run python scripts/parsing.py

run:
	PYTHONPATH=. uv run python app/main.py 