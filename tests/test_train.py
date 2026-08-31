"""Phase 2: the training loop, the split-to-tensor path and the evaluation breakdowns.

Marked ``needs_data`` where they touch the extracted cache; the loop itself is exercised on
synthetic tensors so the essential parts still run in CI.
"""

from __future__ import annotations

import math

import pytest
import torch

from signadapt.data.dataset import KeypointDataset, load_records, make_splits
from signadapt.models.model import build_model
from signadapt.train.evaluate import EvalResult, evaluate, predict
from signadapt.train.loop import (
    evaluate_tensors,
    make_loader,
    stack_dataset,
    train_model,
)
from signadapt.utils.config import load_config
from signadapt.utils.seeding import seed_everything

CACHE_DIR = "data/cache/lsa64"


@pytest.fixture
def tiny_cfg():
    """A model config small enough to train inside a unit test."""
    cfg = load_config("configs/model.yaml")
    cfg["encoder"] |= {"d_model": 32, "n_layers": 1, "n_heads": 2, "ff_dim": 32, "max_len": 8}
    cfg["head"] |= {"in_dim": 32, "n_classes": 4}
    cfg["train"] |= {
        "epochs": 12,
        "batch_size": 8,
        "lr": 3e-3,
        "warmup_epochs": 1,
        "early_stopping_patience": 0,
        "device": "cpu",
    }
    cfg["augment"]["enabled"] = False
    return cfg


def separable_batch(n_per_class=8, n_classes=4, seed=0):
    """A trivially learnable dataset: one constant offset per class, plus noise."""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for label in range(n_classes):
        x = torch.randn(n_per_class, 8, 115, 4, generator=generator) * 0.1
        x[..., :3] += float(label)
        x[..., 3] = 1.0
        xs.append(x)
        ys.append(torch.full((n_per_class,), label))
    return torch.cat(xs), torch.cat(ys)


# ------------------------------------------------------------------------------ the loop


def test_training_reduces_loss_and_fits_a_separable_problem(tiny_cfg):
    seed_everything(0)
    model = build_model(tiny_cfg)
    data = separable_batch()
    outcome = train_model(model, data, None, tiny_cfg, device=torch.device("cpu"), seed=0)

    assert outcome.epochs_run == tiny_cfg["train"]["epochs"]
    assert outcome.history[-1]["train_loss"] < outcome.history[0]["train_loss"]
    assert evaluate_tensors(model, data, tiny_cfg, device=torch.device("cpu")).top1 > 0.9


def test_training_is_reproducible_from_the_seed(tiny_cfg):
    def run(seed: int) -> float:
        seed_everything(0)
        model = build_model(tiny_cfg)
        outcome = train_model(
            model, separable_batch(), None, tiny_cfg, device=torch.device("cpu"), seed=seed
        )
        return outcome.history[-1]["train_loss"]

    assert run(0) == run(0)


def test_validation_drives_early_stopping_and_checkpointing(tiny_cfg):
    tiny_cfg["train"] |= {"epochs": 40, "early_stopping_patience": 2, "lr": 1e-6}
    seed_everything(0)
    model = build_model(tiny_cfg)
    data = separable_batch()
    outcome = train_model(model, data, data, tiny_cfg, device=torch.device("cpu"), seed=0)

    assert outcome.epochs_run < 40, "a flat validation curve should trigger early stopping"
    assert 0 <= outcome.best_epoch < outcome.epochs_run
    assert not math.isnan(outcome.best_val)
    assert set(outcome.best_state) == set(model.state_dict())


def test_best_state_is_a_cpu_snapshot_not_a_live_reference(tiny_cfg):
    seed_everything(0)
    model = build_model(tiny_cfg)
    data = separable_batch()
    outcome = train_model(model, data, data, tiny_cfg, device=torch.device("cpu"), seed=0)
    captured = outcome.best_state["head.linear.weight"].clone()
    with torch.no_grad():
        model.head.linear.weight.add_(1.0)
    torch.testing.assert_close(captured, outcome.best_state["head.linear.weight"])


def test_frozen_encoder_does_not_move_during_training(tiny_cfg):
    """The E5 personalization setting: only the private head may change."""
    seed_everything(0)
    model = build_model(tiny_cfg)
    model.freeze_encoder(True)
    before = model.encoder_state_dict()

    train_model(model, separable_batch(), None, tiny_cfg, device=torch.device("cpu"), seed=0)

    for key, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[key].cpu())
    assert not torch.allclose(
        model.head.linear.weight.cpu(), torch.zeros_like(model.head.linear.weight.cpu())
    )


# ------------------------------------------------------------------------------ evaluation


def test_evaluation_loader_order_matches_the_index_order(tiny_cfg):
    """The per-signer breakdown is only correct if prediction i belongs to index i."""
    seed_everything(0)
    model = build_model(tiny_cfg)
    x, y = separable_batch()
    loader = make_loader(x, y, batch_size=5, shuffle=False)
    _, targets = predict(model, loader, torch.device("cpu"))
    torch.testing.assert_close(targets, y)


def test_evaluate_rejects_misaligned_indices(tiny_cfg):
    seed_everything(0)
    model = build_model(tiny_cfg)
    x, y = separable_batch()
    loader = make_loader(x, y, batch_size=8, shuffle=False)
    with pytest.raises(ValueError, match="indices"):
        evaluate(model, loader, torch.device("cpu"), records=[], indices=[0])


def test_evaluate_on_an_empty_set_is_nan(tiny_cfg):
    seed_everything(0)
    model = build_model(tiny_cfg)
    loader = make_loader(
        torch.empty(0, 8, 115, 4), torch.empty(0, dtype=torch.long), batch_size=4, shuffle=False
    )
    result = evaluate(model, loader, torch.device("cpu"))
    assert result.n == 0 and math.isnan(result.top1)


def test_eval_result_dict_carries_the_across_signer_spread():
    result = EvalResult(top1=0.9, top5=1.0, loss=0.3, n=10, per_signer={"9": 0.8, "10": 1.0})
    payload = result.to_dict()
    assert payload["across_signers"]["mean"] == pytest.approx(0.9)
    assert payload["across_signers"]["n"] == 2


# ----------------------------------------------------------------- against the real cache


@pytest.mark.needs_data
def test_stack_dataset_preserves_index_order():
    records = load_records(CACHE_DIR)
    cfg = load_config("configs/data.yaml")
    indices = tuple(range(0, 200, 37))
    dataset = KeypointDataset(records, indices, cfg)
    x, y = stack_dataset(dataset)

    assert x.shape == (len(indices), cfg["temporal"]["n_frames"], 115, 4)
    assert y.tolist() == [records[i].label for i in indices]
    assert not torch.isnan(x).any()


@pytest.mark.needs_data
def test_stack_dataset_of_an_empty_view_is_empty():
    records = load_records(CACHE_DIR)
    x, y = stack_dataset(KeypointDataset(records, (), load_config("configs/data.yaml")))
    assert len(x) == 0 and len(y) == 0


@pytest.mark.needs_data
def test_breakdowns_cover_every_signer_in_the_test_split():
    records = load_records(CACHE_DIR)
    data_cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    split = make_splits(records, data_cfg)

    indices = split.test[:64]
    data = stack_dataset(KeypointDataset(records, indices, data_cfg))
    seed_everything(0)
    result = evaluate_tensors(
        build_model(model_cfg),
        data,
        model_cfg,
        device=torch.device("cpu"),
        records=records,
        indices=indices,
    )
    assert set(result.per_signer) == {str(records[i].signer) for i in indices}
    assert set(result.per_handedness) <= {"one_handed", "two_handed"}
    assert result.n == len(indices)


# ------------------------------------------------------------------- E3 local-only (phase 3)


def test_adapt_config_disables_early_stopping_and_uses_the_personalization_lr():
    """A signer with k=1 has one clip per sign, so there is nothing to early-stop on."""
    from signadapt.train.local_only import adapt_config

    model_cfg = load_config("configs/model.yaml")
    fl_cfg = load_config("configs/fl.yaml")
    cfg = adapt_config(model_cfg, fl_cfg)

    assert cfg["train"]["lr"] == float(fl_cfg["personalization"]["adapt_lr"])
    assert cfg["train"]["epochs"] == int(fl_cfg["personalization"]["adapt_epochs"])
    assert cfg["train"]["early_stopping_patience"] == 0
    assert cfg["train"]["warmup_epochs"] <= cfg["train"]["epochs"] // 4
    assert model_cfg["train"]["early_stopping_patience"] > 0, "the centralized config stands"


def test_constant_schedule_does_not_decay():
    """A within-round cosine would decay to near zero every round and then reset."""
    from signadapt.train.loop import build_schedule

    factor = build_schedule({"scheduler": "constant"}, epochs=2)
    assert [factor(e) for e in range(2)] == [1.0, 1.0]


def test_cosine_schedule_over_two_epochs_would_never_reach_full_rate():
    """Documents the bug the constant schedule exists to avoid."""
    from signadapt.train.loop import build_schedule

    factor = build_schedule({"scheduler": "cosine", "warmup_epochs": 3}, epochs=2)
    assert max(factor(e) for e in range(2)) < 1.0


def test_unknown_scheduler_is_rejected():
    from signadapt.train.loop import build_schedule

    with pytest.raises(ValueError, match="train.scheduler"):
        build_schedule({"scheduler": "onecycle"}, epochs=10)
