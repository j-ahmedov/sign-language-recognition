# SignAdapt -- see PLAN.md section 7.
# Every target is runnable from a clean checkout after `make setup`.

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
SEED   ?= 0
CONFIG ?= configs

.PHONY: help setup data overlay train federated figures demo test lint format clean

CLIPS ?= 001_001_001 010_007_004 030_002_001 045_010_002 060_006_003

help:
	@echo "SignAdapt targets:"
	@echo "  setup      install the package and dev extras into .venv"
	@echo "  data       download LSA64 and extract/cache MediaPipe keypoints"
	@echo "  overlay    render keypoints back over video for visual checking"
	@echo "  train      centralized baselines E1/E2/E6                         [phase 2]"
	@echo "  federated  Flower simulation: FedAvg (E4) and FedPer (E5)         [phase 3-4]"
	@echo "  figures    regenerate every figure from committed results/*.json  [phase 5]"
	@echo "  demo       webcam -> keypoints -> caption -> virtual camera       [phase 6]"
	@echo "  test       pytest (signer-leakage guard lives here)"
	@echo "  lint       ruff check"
	@echo "  format     ruff format + import sort"

setup:
	$(PIP) install -e ".[dev,demo]"

data:
	$(PYTHON) -m signadapt.data.download --config $(CONFIG)/data.yaml
	$(PYTHON) -m signadapt.data.keypoints --config $(CONFIG)/data.yaml

overlay:
	$(PYTHON) -m signadapt.data.overlay --config $(CONFIG)/data.yaml --clips $(CLIPS) --out figures/overlays

train:
	$(PYTHON) -m signadapt.train.centralized --config $(CONFIG)/model.yaml --seed $(SEED)

federated:
	$(PYTHON) -m signadapt.federated.simulation --config $(CONFIG)/fl.yaml --seed $(SEED)

figures:
	$(PYTHON) -m signadapt.figures --results results --out figures

demo:
	$(PYTHON) -m signadapt.demo.realtime --config $(CONFIG)/model.yaml

test:
	$(PYTHON) -m pytest

lint:
	.venv/bin/ruff check src tests

format:
	.venv/bin/ruff format src tests
	.venv/bin/ruff check --fix src tests

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
