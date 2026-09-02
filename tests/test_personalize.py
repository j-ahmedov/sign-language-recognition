"""Phase 4: the k-shot adaptation sweep (E4, E5, E6).

The head-privacy assertions live in ``tests/test_fedper.py``; what is checked here is that
the three methods differ in exactly the ways PLAN.md section 6 says they do and in no others,
because the whole sweep is a controlled comparison. If E4 and E5 also differed in their
learning-rate schedule, their query sets, or how many epochs they adapt for, the resulting
curve would not answer RQ3.
"""

from __future__ import annotations

import pytest
import torch

from signadapt.models.model import ENCODER_PREFIX, build_model
from signadapt.personalize.adapt import (
    METHODS,
    PRETRAIN_EPOCHS,
    adapt_and_evaluate,
    adapt_config,
)
from signadapt.utils.config import load_config
from signadapt.utils.seeding import seed_everything


@pytest.fixture
def fl_cfg():
    return load_config("configs/fl.yaml")


@pytest.fixture
def tiny_cfg():
    cfg = load_config("configs/model.yaml")
    cfg["encoder"] |= {"d_model": 32, "n_layers": 1, "n_heads": 2, "ff_dim": 32, "max_len": 8}
    cfg["head"] |= {"in_dim": 32, "n_classes": 4}
    cfg["train"] |= {"epochs": 2, "batch_size": 4, "device": "cpu"}
    cfg["augment"]["enabled"] = False
    return cfg


def clips(n=8, n_classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, 115, 4, generator=generator)
    x[..., 3] = 1.0
    return x, torch.arange(n) % n_classes


def empty():
    return torch.empty(0, 8, 115, 4), torch.empty(0, dtype=torch.long)


# ------------------------------------------------------------------- the method table


def test_the_three_methods_cover_the_plan_matrix():
    """PLAN.md section 6 names three methods; E6M is a diagnostic added on top of them."""
    assert {"E4", "E5", "E6"} <= set(METHODS)
    assert {METHODS[m]["pretrain"] for m in ("E4", "E5", "E6")} == {
        "fedavg",
        "fedper",
        "centralized",
    }
    assert set(METHODS) - {"E4", "E5", "E6"} == {"E6M"}


def test_e5_and_e6_differ_only_in_where_the_encoder_came_from():
    """That difference is the whole content of "what federation costs"."""
    e5, e6 = METHODS["E5"], METHODS["E6"]
    assert e5["load"] == e6["load"] and e5["adapt"] == e6["adapt"]
    assert e5["pretrain"] != e6["pretrain"]


def test_unknown_methods_are_rejected():
    from signadapt.personalize.adapt import run_sweep

    with pytest.raises(ValueError, match="unknown method"):
        run_sweep("E7", model_cfg={}, data_cfg={}, fl_cfg={})


# ------------------------------------------------------------------ the adapt config


def test_head_and_full_adaptation_share_everything_but_the_learning_rate(tiny_cfg, fl_cfg):
    head = adapt_config(tiny_cfg, fl_cfg, adapt="head")
    full = adapt_config(tiny_cfg, fl_cfg, adapt="full")
    assert head["train"]["epochs"] == full["train"]["epochs"]
    assert head["train"]["early_stopping_patience"] == full["train"]["early_stopping_patience"] == 0
    differing = {k for k in head["train"] if head["train"][k] != full["train"][k]}
    assert differing == {"lr"}


def test_full_finetuning_uses_the_lower_rate(fl_cfg, tiny_cfg):
    """A whole pretrained encoder at the head's rate is destroyed, not fine-tuned."""
    head = adapt_config(tiny_cfg, fl_cfg, adapt="head")["train"]["lr"]
    full = adapt_config(tiny_cfg, fl_cfg, adapt="full")["train"]["lr"]
    assert full < head
    assert full == float(fl_cfg["personalization"]["adapt_lr_full"])


def test_unknown_adapt_mode_is_rejected(tiny_cfg, fl_cfg):
    with pytest.raises(ValueError, match="adapt must be"):
        adapt_config(tiny_cfg, fl_cfg, adapt="lora")


# ------------------------------------------------------------------------- adaptation


def test_k_zero_evaluates_without_training(tiny_cfg, fl_cfg):
    """The zero-shot point: whatever the pretraining produced, applied untouched."""
    seed_everything(3)
    state = build_model(tiny_cfg).state_dict()
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="full")
    cfg["train"] |= {"epochs": 5, "device": "cpu"}

    result, seconds = adapt_and_evaluate(
        "E4", state, empty(), clips(), cfg=cfg, device=torch.device("cpu"), seed=0
    )
    assert seconds == 0.0, "k=0 must not train"
    assert result.n == 8


def test_e4_at_k_zero_inherits_the_pretrained_head(tiny_cfg, fl_cfg):
    """E4 is the only method with a meaningful zero-shot point, because it gets a head."""
    seed_everything(3)
    donor = build_model(tiny_cfg)
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="full")
    cfg["train"] |= {"device": "cpu"}
    query = clips()

    e4, _ = adapt_and_evaluate(
        "E4", donor.state_dict(), empty(), query, cfg=cfg, device=torch.device("cpu"), seed=0
    )
    donor.eval()
    with torch.no_grad():
        expected = (donor(query[0]).argmax(dim=1) == query[1]).float().mean().item()
    assert e4.top1 == pytest.approx(expected)


def test_e5_at_k_zero_is_chance_because_it_has_no_head_yet(tiny_cfg, fl_cfg):
    """Not a defect: under FedPer a head does not exist until the signer trains one."""
    seed_everything(3)
    donor = build_model(tiny_cfg)
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="head")
    cfg["train"] |= {"device": "cpu"}
    query = clips()

    e5, _ = adapt_and_evaluate(
        "E5", donor.state_dict(), empty(), query, cfg=cfg, device=torch.device("cpu"), seed=0
    )
    donor.eval()
    with torch.no_grad():
        inherited = (donor(query[0]).argmax(dim=1) == query[1]).float().mean().item()
    assert e5.top1 != inherited or inherited == 0.0


def test_adaptation_is_reproducible_from_the_seed(tiny_cfg, fl_cfg):
    seed_everything(3)
    state = build_model(tiny_cfg).state_dict()
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="head")
    cfg["train"] |= {"epochs": 3, "device": "cpu"}

    def run(seed: int) -> float:
        return adapt_and_evaluate(
            "E5", state, clips(), clips(), cfg=cfg, device=torch.device("cpu"), seed=seed
        )[0].top1

    assert run(0) == run(0)


def test_e4_adaptation_moves_the_encoder_and_e5_does_not(tiny_cfg, fl_cfg):
    """The one structural difference between the two federated methods at adaptation time."""
    seed_everything(3)
    state = {k: v.clone() for k, v in build_model(tiny_cfg).state_dict().items()}
    support, query = clips(), clips()

    moved = {}
    for method, adapt in (("E4", "full"), ("E5", "head")):
        cfg = adapt_config(tiny_cfg, fl_cfg, adapt=adapt)
        cfg["train"] |= {"epochs": 3, "device": "cpu"}
        seed_everything(0)
        model = build_model(cfg)
        payload = (
            state
            if METHODS[method]["load"] == "all"
            else {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
        )
        model.load_state_dict(payload, strict=False)
        model.freeze_encoder(METHODS[method]["adapt"] == "head")
        from signadapt.train.loop import train_model

        train_model(model, support, None, cfg, device=torch.device("cpu"), seed=0)
        moved[method] = not torch.allclose(
            model.state_dict()["encoder.input_proj.weight"].cpu(),
            state["encoder.input_proj.weight"],
        )

    assert moved["E4"], "E4 fine-tunes the whole model"
    assert not moved["E5"], "E5 keeps the federated encoder exactly as received"
    assert query[1].numel() == 8


def test_frozen_submodules_are_put_back_into_eval_mode(tiny_cfg):
    """A frozen encoder must not keep sampling dropout the head cannot adapt to."""
    from signadapt.train.loop import eval_frozen_submodules

    model = build_model(tiny_cfg)
    model.freeze_encoder(True)
    model.train()
    assert model.encoder.training and model.head.training

    frozen = eval_frozen_submodules(model)
    assert frozen == ["encoder"]
    assert not model.encoder.training
    assert model.head.training


# ------------------------------------------------------------------ the pretraining cache


def test_run_sweep_passes_its_cache_dir_down_to_the_pretraining(monkeypatch, tmp_path):
    """A cache_dir that never arrives sends every artefact to the default directory."""
    import signadapt.personalize.adapt as adapt

    seen = {}

    def fake_pretrained_state(*args: object, **kwargs: object) -> None:
        seen["cache_dir"] = kwargs.get("cache_dir", "MISSING")
        raise RuntimeError("stop here")

    monkeypatch.setattr(adapt, "pretrained_state", fake_pretrained_state)
    with pytest.raises(RuntimeError, match="stop here"):
        adapt.run_sweep(
            "E5",
            model_cfg=load_config("configs/model.yaml"),
            data_cfg=load_config("configs/data.yaml"),
            fl_cfg=load_config("configs/fl.yaml"),
            seed=0,
            results_dir=str(tmp_path),
            cache_dir=str(tmp_path / "cache"),
        )
    assert seen["cache_dir"] == str(tmp_path / "cache")


def test_a_fingerprint_changes_when_the_config_that_produced_it_changes():
    from signadapt.utils.config import config_fingerprint

    fl_cfg = load_config("configs/fl.yaml")
    base = config_fingerprint(fl_cfg["server"], fl_cfg["client"])
    fewer_rounds = config_fingerprint({**fl_cfg["server"], "num_rounds": 2}, fl_cfg["client"])
    assert base != fewer_rounds, "a 2-round smoke artefact must not match a 50-round run"
    assert base == config_fingerprint(fl_cfg["server"], fl_cfg["client"])


def test_a_stale_pretraining_is_recomputed_rather_than_reused(tmp_path, monkeypatch, capsys):
    """The regression test for a smoke run's checkpoint being served to a real experiment."""
    import signadapt.federated.simulation as sim
    from signadapt.data.dataset import Split

    fold = Split(
        name="loso-signer01", train=(0,), val=(1,), test=(2,),
        signers={"train": (2, 3), "val": (4,), "test": (1,)},
    )
    model_cfg = load_config("configs/model.yaml")
    fl_cfg = load_config("configs/fl.yaml")
    stale = dict(fl_cfg, server={**fl_cfg["server"], "num_rounds": 2})

    calls = []
    monkeypatch.setattr(sim, "partition_indices", lambda *a, **k: [])
    monkeypatch.setattr(sim, "write_partitions", lambda *a, **k: [])

    def fake_round_trip(**kwargs: object) -> dict[str, list]:
        calls.append(1)
        return {"keys": [], "best_arrays": [], "rounds": [], "prefixes": []}

    monkeypatch.setattr(sim, "run_simulation_round_trip", fake_round_trip)

    common = dict(model_cfg=model_cfg, data_cfg={}, seed=0, cache_dir=str(tmp_path))
    sim.federated_pretrain([], fold, fl_cfg=stale, **common)
    assert len(calls) == 1

    # Same filename, different config: it must recompute rather than serve the stale file.
    result = sim.federated_pretrain([], fold, fl_cfg=fl_cfg, **common)
    assert len(calls) == 2
    assert not result["cached"]
    assert "different config" in capsys.readouterr().out

    # Same config again: now the cache is legitimately warm.
    assert sim.federated_pretrain([], fold, fl_cfg=fl_cfg, **common)["cached"]
    assert len(calls) == 2


# ------------------------------------------------------------ E6M, the matched-budget E6


def test_e6m_is_e6_in_everything_but_its_pretraining_budget():
    """The comparison is only meaningful if budget is the single difference."""
    assert METHODS["E6M"] == METHODS["E6"]
    assert set(PRETRAIN_EPOCHS) == {"E6M"}
    assert "E6" not in PRETRAIN_EPOCHS


def test_e6m_matches_the_clip_presentations_the_federation_consumes():
    """50 rounds x 2 local epochs = 100 passes over the pooled training set."""
    fl_cfg = load_config("configs/fl.yaml")
    rounds = int(fl_cfg["server"]["num_rounds"])
    local_epochs = int(fl_cfg["client"]["local_epochs"])
    assert PRETRAIN_EPOCHS["E6M"] == rounds * local_epochs


def test_e6m_spends_its_whole_budget_and_keeps_e6s_selection_rule(monkeypatch):
    """Early stopping off so the budget is really spent, validation selection kept.

    Patience > 0 is what gates the break in ``train_model``, so 0 disables stopping without
    touching how the returned checkpoint is chosen -- leaving budget as the only difference.
    """
    from signadapt.personalize import adapt

    seen = {}

    def fake(records, fold, *, model_cfg, data_cfg, seed, cache_dir) -> dict:
        seen[len(seen)] = (model_cfg["train"], cache_dir)
        return {"state": {}, "cached": False}

    monkeypatch.setattr(adapt, "pretrain_centralized", fake)
    model_cfg = load_config("configs/model.yaml")
    baseline = dict(model_cfg["train"])
    for method in ("E6", "E6M"):
        adapt.pretrained_state(
            method, [], None, model_cfg=model_cfg, data_cfg={}, fl_cfg={},
            seed=0, cache_dir="cache",
        )
    (e6_train, e6_cache), (e6m_train, e6m_cache) = seen[0], seen[1]
    assert e6_train["epochs"] == baseline["epochs"]
    assert e6m_train["epochs"] == PRETRAIN_EPOCHS["E6M"]
    assert e6m_train["early_stopping_patience"] == 0
    assert e6_train["early_stopping_patience"] > 0
    # Same directory would silently overwrite the checkpoints E6's committed numbers came from.
    assert e6_cache != e6m_cache
    # And the caller's config must survive unmutated, or E6 in the same process inherits it.
    assert model_cfg["train"] == baseline


def test_e6m_never_reaches_the_figures():
    """METHOD_ORDER gates the figure loader, so a diagnostic run cannot enter a chart."""
    from signadapt.figures import METHOD_ORDER

    assert "E6M" not in METHOD_ORDER
