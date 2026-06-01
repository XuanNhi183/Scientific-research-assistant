VENV_DIR ?= .venv
PYTHON ?= python3

install:
	uv sync

venv:
	$(PYTHON) -m venv $(VENV_DIR)

activate:
	@echo "Run: source $(VENV_DIR)/bin/activate"

shell:
	@test -f $(VENV_DIR)/bin/activate || (echo "Missing $(VENV_DIR)/bin/activate. Run: make venv"; exit 1)
	@CONDA_AUTO_ACTIVATE_BASE=false bash -i -c "source $(VENV_DIR)/bin/activate; exec bash -i"

deps:
	$(VENV_DIR)/bin/python -m pip install -U pip
	$(VENV_DIR)/bin/python -m pip install -e .

parse:
	PYTHONPATH=. uv run python scripts/parsing.py

run:
	PYTHONPATH=. uv run python app/main.py 