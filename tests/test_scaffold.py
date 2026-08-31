"""Structural checks on the repository layout defined in PLAN.md section 7."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "README.md",
    "pyproject.toml",
    "Makefile",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/fl.yaml",
    "src/signadapt/data/download.py",
    "src/signadapt/data/keypoints.py",
    "src/signadapt/data/normalize.py",
    "src/signadapt/data/dataset.py",
    "src/signadapt/models/encoder.py",
    "src/signadapt/models/head.py",
    "src/signadapt/models/model.py",
    "src/signadapt/train/centralized.py",
    "src/signadapt/train/local_only.py",
    "src/signadapt/train/evaluate.py",
    "src/signadapt/federated/client.py",
    "src/signadapt/federated/strategy.py",
    "src/signadapt/federated/simulation.py",
    "src/signadapt/personalize/adapt.py",
    "src/signadapt/demo/realtime.py",
    "src/signadapt/demo/virtualcam.py",
    "src/signadapt/utils/seeding.py",
    "src/signadapt/utils/results.py",
    "experiments/run_all.sh",
    "tests/test_splits.py",
    "tests/test_normalize.py",
    "tests/test_fedper.py",
]

MAKE_TARGETS = ["data", "train", "federated", "figures", "demo", "test"]


@pytest.mark.parametrize("rel", EXPECTED_FILES)
def test_layout_file_exists(rel):
    assert (ROOT / rel).is_file(), f"missing from the PLAN section 7 layout: {rel}"


@pytest.mark.parametrize("target", MAKE_TARGETS)
def test_makefile_target_exists(target):
    makefile = (ROOT / "Makefile").read_text()
    assert f"\n{target}:" in makefile


def test_third_party_dependencies_import():
    """The environment is only reproducible if these actually load on this machine."""
    for name in ("torch", "numpy", "yaml", "mediapipe", "cv2", "flwr", "matplotlib"):
        importlib.import_module(name)
