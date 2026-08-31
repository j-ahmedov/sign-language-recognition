#!/usr/bin/env bash
# Reproduce every number in the thesis from a clean checkout.
# Each stage is filled in by its phase; the ordering is fixed by PLAN.md section 8.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2}"

echo "== phase 1: data =============================================="
$PYTHON -m signadapt.data.download  --config configs/data.yaml
$PYTHON -m signadapt.data.keypoints --config configs/data.yaml

echo "== phase 2: sanity gate + centralized baselines E1, E2 ========"
# --experiment all runs the overfit gate first and exits non-zero if it fails, so
# `set -e` stops the whole script rather than letting later phases build on a broken
# pipeline (PLAN.md section 8, week 3).
$PYTHON -m signadapt.train.centralized --experiment all --seeds $SEEDS

echo "== phase 3: federated FedAvg + IID correctness check + E3 ====="
# --experiment all runs the IID check first and exits non-zero if FedAvg does not
# reproduce centralized training on IID data (PLAN.md section 8, week 4).
$PYTHON -m signadapt.federated.simulation --experiment all --seeds $SEEDS
$PYTHON -m signadapt.train.local_only --seeds $SEEDS

echo "== phase 4: FedPer E5, E4, E6 and the k-sweep ================="
echo "not implemented yet (phase 4)"

echo "== phase 5: figures from committed JSON ======================="
# $PYTHON -m signadapt.figures --results results --out figures
echo "not implemented yet (phase 5)"
