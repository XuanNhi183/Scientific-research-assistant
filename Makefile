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
	@bash -c 'TMPFILE=$$(mktemp); cat ~/.bashrc > $$TMPFILE; echo "conda deactivate 2>/dev/null || true; source $(VENV_DIR)/bin/activate; rm -f $$TMPFILE" >> $$TMPFILE; export CONDA_AUTO_ACTIVATE_BASE=false; exec bash --rcfile $$TMPFILE -i'

deps:
	$(VENV_DIR)/bin/python -m pip install -U pip
	$(VENV_DIR)/bin/python -m pip install -e .

test:
	PYTHONPATH=. uv run python scripts/debug_search.py

run:
	PYTHONPATH=. uv run python main.py 

server:
	cd frontend && npm run dev

dataset:
	PYTHONPATH=. uv run python -m dataset_builder.build_dataset

dataset-config:
	PYTHONPATH=. uv run python -m dataset_builder.build_dataset --config $(CONFIG)
