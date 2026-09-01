# SignAdapt -- see PLAN.md section 7.
# Every target is runnable from a clean checkout after `make setup`.

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/pip
SEED   ?= 0
CONFIG ?= configs

.PHONY: help setup data overlay sanity train federated local sweep figures demo test lint format clean

CLIPS ?= 001_001_001 010_007_004 030_002_001 045_010_002 060_006_003

help:
	@echo "SignAdapt targets:"
	@echo "  setup      install the package and dev extras into .venv"
	@echo "  data       download LSA64 and extract/cache MediaPipe keypoints"
	@echo "  overlay    render keypoints back over video for visual checking"
	@echo "  sanity     overfit-50-clips gate; nothing is reportable until it passes"
	@echo "  train      centralized baselines E1 and E2 (E6 lands with the k-sweep)"
	@echo "  federated  IID correctness check, then FedAvg over one client per signer"
	@echo "  local      E3 local-only: each held-out signer trains from scratch on k clips"
	@echo "  sweep      E4/E5/E6 k-shot adaptation over every leave-one-signer-out fold"
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

# EXPERIMENT=all runs the sanity gate first and refuses to report E1/E2 if it fails.
# SEEDS defaults to `seeds:` in configs/model.yaml; SEEDS="0" runs a single seed.
EXPERIMENT ?= all
SEEDS ?=

sanity:
	$(PYTHON) -m signadapt.train.centralized --config $(CONFIG)/model.yaml \
		--data-config $(CONFIG)/data.yaml --experiment sanity --seed $(SEED)

train:
	$(PYTHON) -m signadapt.train.centralized --config $(CONFIG)/model.yaml \
		--data-config $(CONFIG)/data.yaml --experiment $(EXPERIMENT) \
		$(if $(SEEDS),--seeds $(SEEDS),)

# EXPERIMENT=all runs the IID correctness check first and refuses to go on if it fails.
FL_EXPERIMENT ?= all

federated:
	$(PYTHON) -m signadapt.federated.simulation --config $(CONFIG)/fl.yaml \
		--model-config $(CONFIG)/model.yaml --data-config $(CONFIG)/data.yaml \
		--experiment $(FL_EXPERIMENT) $(if $(SEEDS),--seeds $(SEEDS),--seed $(SEED))

local:
	$(PYTHON) -m signadapt.train.local_only --config $(CONFIG)/fl.yaml \
		--model-config $(CONFIG)/model.yaml --data-config $(CONFIG)/data.yaml \
		$(if $(SEEDS),--seeds $(SEEDS),)

# METHODS/SEEDS override the sweep; pretrainings are cached under data/checkpoints/pretrain
# so re-running a method or adding a k value does not repeat the federated runs.
METHODS ?= E5 E4 E6

sweep:
	$(PYTHON) -m signadapt.personalize.adapt --config $(CONFIG)/fl.yaml \
		--model-config $(CONFIG)/model.yaml --data-config $(CONFIG)/data.yaml \
		--methods $(METHODS) $(if $(SEEDS),--seeds $(SEEDS),)

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
