#!/usr/bin/env bash
# Reproduce every number in the thesis from a clean checkout.
# Each stage is filled in by its phase; the ordering is fixed by PLAN.md section 8.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2}"

echo "== phase 1: data =============================================="
# $PYTHON -m signadapt.data.download  --config configs/data.yaml
# $PYTHON -m signadapt.data.keypoints --config configs/data.yaml
echo "not implemented yet (phase 1)"

echo "== phase 2: sanity gate + centralized baselines E1, E2 ========"
# $PYTHON -m signadapt.train.centralized --experiment sanity --config configs/model.yaml
# for s in $SEEDS; do
#   $PYTHON -m signadapt.train.centralized --experiment E1 --seed "$s"
#   $PYTHON -m signadapt.train.centralized --experiment E2 --seed "$s"
# done
echo "not implemented yet (phase 2)"

echo "== phase 3: federated FedAvg + IID correctness check + E3 ====="
echo "not implemented yet (phase 3)"

echo "== phase 4: FedPer E5, E4, E6 and the k-sweep ================="
echo "not implemented yet (phase 4)"

echo "== phase 5: figures from committed JSON ======================="
# $PYTHON -m signadapt.figures --results results --out figures
echo "not implemented yet (phase 5)"
