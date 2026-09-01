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
# One federated pretraining per (method, fold, seed), cached under data/checkpoints/pretrain.
# Budget about 90 minutes per seed on an M4.
for s in $SEEDS; do
  $PYTHON -m signadapt.personalize.adapt --methods E5 E4 E6 --seed "$s"
done

echo "== phase 5: figures from committed JSON ======================="
# Reads only results/*.json -- no model, no dataset, no GPU. This step is what makes the
# thesis reproducible from a checkout: figures/summary.json holds every number the figures
# draw, and the prose quotes that file rather than a number someone typed (PLAN.md section 7).
$PYTHON -m signadapt.figures --results results --out figures

echo "== phase 6: demo correctness gate + RQ4 latency ==============="
# Both are offline and need no camera, so the whole script still runs unattended. The live
# webcam demo is `make demo` and is the one step a person has to be present for.
$PYTHON -m signadapt.demo.realtime --verify 100
$PYTHON -m signadapt.demo.realtime --bench
