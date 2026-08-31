"""Tests for the seeding utility, config loader and JSON results logger (phase 0)."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from signadapt.utils import (
    ResultsLogger,
    apply_overrides,
    get_in,
    load_config,
    load_results,
    merge_configs,
    seed_everything,
    temporary_seed,
    torch_generator,
)

# --------------------------------------------------------------------------- seeding


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1).item())


def test_seed_everything_is_reproducible():
    seed_everything(1234)
    first = _draw()
    seed_everything(1234)
    second = _draw()
    assert first == second


def test_different_seeds_differ():
    seed_everything(0)
    a = _draw()
    seed_everything(1)
    b = _draw()
    assert a != b


def test_seed_state_is_reported():
    state = seed_everything(7)
    assert state.seed == 7
    assert isinstance(state.deterministic_algorithms, bool)


def test_temporary_seed_restores_stream():
    seed_everything(0)
    baseline = [_draw() for _ in range(3)]

    seed_everything(0)
    before = _draw()
    with temporary_seed(999):
        _draw()
    after = [_draw() for _ in range(2)]

    # The outer stream must continue as if the block had never run.
    assert [before, *after] == baseline


def test_temporary_seed_is_independent_of_history():
    """k=5 support sets must start with the same examples as k=3 (see personalize/adapt)."""
    with temporary_seed(42):
        a = [random.random() for _ in range(5)]
    for _ in range(17):
        random.random()
    with temporary_seed(42):
        b = [random.random() for _ in range(3)]
    assert a[:3] == b


def test_torch_generator_is_deterministic():
    g1, g2 = torch_generator(3), torch_generator(3)
    assert torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))


# --------------------------------------------------------------------------- config


def test_load_project_configs():
    cfg = load_config("configs/data.yaml", "configs/model.yaml", "configs/fl.yaml")
    assert get_in(cfg, "encoder.d_model") == 128
    assert get_in(cfg, "temporal.n_frames") == 64
    assert get_in(cfg, "strategy.private_prefixes") == ["head."]


def test_input_dim_matches_landmark_counts():
    """A mismatch here would silently mis-shape every batch."""
    cfg = load_config("configs/data.yaml", "configs/model.yaml")
    lm = cfg["landmarks"]
    total = lm["pose"] + lm["left_hand"] + lm["right_hand"] + lm["face"]["n_landmarks"]
    assert total == lm["total"]
    assert cfg["encoder"]["input_dim"] == total * 3
    assert cfg["encoder"]["max_len"] == cfg["temporal"]["n_frames"]
    assert cfg["head"]["n_classes"] == cfg["dataset"]["n_classes"]


def test_configured_splits_are_signer_disjoint():
    """The scaffold-level version of the guard in tests/test_splits.py."""
    splits = load_config("configs/data.yaml")["splits"]
    train, val, test = (set(splits[f"{s}_signers"]) for s in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test


def test_merge_is_deep_and_non_mutating():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = merge_configs(base, {"a": {"y": 9}})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base["a"]["y"] == 2


def test_apply_overrides_parses_scalars():
    cfg = apply_overrides({"train": {"lr": 1.0}}, ["train.lr=1e-3", "augment.enabled=false"])
    assert cfg["train"]["lr"] == pytest.approx(1e-3)
    assert cfg["augment"]["enabled"] is False


def test_apply_overrides_rejects_malformed():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({}, ["nonsense"])


# --------------------------------------------------------------------------- results


def test_results_logger_roundtrip(tmp_path):
    cfg = {"encoder": {"d_model": 128}}
    with ResultsLogger("E2", config=cfg, seed=0, tag="lsa64", results_dir=tmp_path) as log:
        log.log_record(epoch=0, val_acc=np.float32(0.5))
        log.log_record(epoch=1, val_acc=0.75)
        log.set_metrics(top1=0.75, top5=np.float64(0.9))
        path = log.path

    doc = json.loads(path.read_text())
    assert doc["experiment"] == "E2"
    assert doc["status"] == "ok"
    assert doc["seed"] == 0
    assert doc["config"] == cfg
    assert doc["metrics"]["top1"] == 0.75
    assert doc["metrics"]["top5"] == pytest.approx(0.9)
    assert [r["epoch"] for r in doc["records"]] == [0, 1]
    assert doc["environment"]["versions"]["torch"] is not None


def test_results_logger_records_failures(tmp_path):
    with pytest.raises(RuntimeError), ResultsLogger("E9", results_dir=tmp_path) as log:
        log.log_record(epoch=0)
        raise RuntimeError("boom")

    doc = json.loads(log.path.read_text())
    assert doc["status"] == "failed"
    assert "boom" in doc["error"]


def test_load_results_excludes_unfinished_runs_by_default(tmp_path):
    """Only completed runs may reach a figure."""
    ResultsLogger("E1", seed=0, results_dir=tmp_path).finish()
    with pytest.raises(RuntimeError), ResultsLogger("E1", seed=1, results_dir=tmp_path):
        raise RuntimeError("crashed")
    ResultsLogger("E1", seed=2, results_dir=tmp_path).save()  # checkpoint, never finished

    ok = load_results(tmp_path, experiment="E1")
    assert [doc["seed"] for doc in ok] == [0]
    assert len(load_results(tmp_path, experiment="E1", status=None)) == 3


def test_numpy_arrays_are_serializable(tmp_path):
    log = ResultsLogger("E1", results_dir=tmp_path)
    log.set_metrics(per_signer=np.array([0.1, 0.2]))
    doc = json.loads(log.save().read_text())
    assert doc["metrics"]["per_signer"] == [pytest.approx(0.1), pytest.approx(0.2)]
