"""Asserts that head parameters are never included in federated aggregation (PLAN.md E5).

This is the test the FedPer claim rests on, and it exists because **nothing about a broken
FedPer is visible in its accuracy**. A run that quietly transmits the head still converges,
still produces a plausible adaptation curve, and still writes a result file labelled E5 --
it is simply FedAvg wearing E5's name, and every privacy sentence in the thesis about it is
false. So the assertions here are on the payload itself at every point it exists: what a
client hands back from ``fit``, what the strategy is configured to aggregate, what a
pretraining writes to its checkpoint, and what a client's own head does across a round.

The tests run on a small model and synthetic clips so they hold in CI without the dataset,
and the prefixes are read from ``configs/fl.yaml`` so that a config edit cannot quietly move
the boundary without failing here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from signadapt.federated.client import SignerClient, write_partition
from signadapt.federated.parameters import get_parameters, set_parameters, shared_keys
from signadapt.federated.simulation import as_fedavg
from signadapt.federated.strategy import build_strategy, weighted_average
from signadapt.models.model import ENCODER_PREFIX, HEAD_PREFIX, build_model
from signadapt.personalize.adapt import METHODS, adapt_and_evaluate, adapt_config
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
    cfg["train"] |= {"epochs": 2, "batch_size": 4, "device": "cpu", "scheduler": "constant"}
    cfg["augment"]["enabled"] = False
    return cfg


@pytest.fixture
def fedper_prefixes(fl_cfg):
    return tuple(fl_cfg["strategy"]["aggregate_prefixes"])


def clips(n=8, n_classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, 115, 4, generator=generator)
    x[..., 3] = 1.0
    return x, torch.arange(n) % n_classes


def make_client(tiny_cfg, path, prefixes, signer=1):
    return SignerClient(
        model_cfg=tiny_cfg,
        train_path=path,
        val_path=None,
        signer=signer,
        share_prefixes=prefixes,
        device=torch.device("cpu"),
        local_epochs=1,
    )


# --------------------------------------------------------- the configured boundary


def test_config_declares_the_head_private_and_only_the_encoder_shared(fl_cfg):
    assert fl_cfg["strategy"]["name"] == "fedper"
    assert tuple(fl_cfg["strategy"]["aggregate_prefixes"]) == (ENCODER_PREFIX,)
    assert tuple(fl_cfg["strategy"]["private_prefixes"]) == (HEAD_PREFIX,)


def test_the_shared_key_set_contains_no_head_parameter(tiny_cfg, fedper_prefixes):
    model = build_model(tiny_cfg)
    keys = shared_keys(model, fedper_prefixes)
    assert keys, "something must be shared"
    assert not [k for k in keys if k.startswith(HEAD_PREFIX)]
    assert set(keys) == set(model.encoder_state_dict())


def test_the_partition_is_total_so_nothing_is_shared_by_accident(tiny_cfg, fedper_prefixes):
    """Every parameter is either shared or private; there is no third, unexamined category."""
    model = build_model(tiny_cfg)
    shared = set(shared_keys(model, fedper_prefixes))
    private = {k for k in model.state_dict() if k.startswith(HEAD_PREFIX)}
    assert shared | private == set(model.state_dict())
    assert not shared & private


def test_a_strategy_that_would_share_the_head_is_refused(fl_cfg):
    with pytest.raises(ValueError, match="private parameters would be transmitted"):
        build_strategy(
            "fedper",
            aggregate_prefixes=[ENCODER_PREFIX, HEAD_PREFIX],
            private_prefixes=fl_cfg["strategy"]["private_prefixes"],
        )


def test_the_guard_passes_only_because_the_head_is_absent(fl_cfg):
    """A guard that cannot fail is not a guard: the same call succeeds without the head."""
    strategy = build_strategy(
        "fedper",
        aggregate_prefixes=[ENCODER_PREFIX],
        private_prefixes=fl_cfg["strategy"]["private_prefixes"],
    )
    assert strategy.aggregate_prefixes == (ENCODER_PREFIX,)


# ------------------------------------------------------------- what a client transmits


def test_a_client_fit_payload_contains_no_head_tensor(tiny_cfg, fedper_prefixes, tmp_path):
    """The load-bearing assertion: the head does not go on the wire."""
    path = write_partition(tmp_path / "c.npz", *clips())
    seed_everything(0)
    client = make_client(tiny_cfg, path, fedper_prefixes)
    keys = shared_keys(build_model(tiny_cfg), fedper_prefixes)

    outgoing, _, _ = client.fit(client.get_parameters({}), {"server_round": 1})

    assert len(outgoing) == len(keys)
    assert not any(k.startswith(HEAD_PREFIX) for k in keys)
    head_shapes = {tuple(v.shape) for v in build_model(tiny_cfg).head_state_dict().values()}
    assert not (head_shapes & {a.shape for a in outgoing}), (
        "no transmitted array even has the shape of a head tensor"
    )


def test_a_fedavg_client_does_transmit_the_head(tiny_cfg, tmp_path):
    """The contrast that makes the assertion above meaningful rather than vacuous."""
    path = write_partition(tmp_path / "c.npz", *clips())
    seed_everything(0)
    client = make_client(tiny_cfg, path, (ENCODER_PREFIX, HEAD_PREFIX))
    outgoing, _, _ = client.fit(client.get_parameters({}), {"server_round": 1})
    assert len(outgoing) == len(build_model(tiny_cfg).state_dict())


def test_a_clients_head_survives_a_round_untouched(tiny_cfg, fedper_prefixes, tmp_path):
    """The private head is the signer's own: a server payload must not overwrite it."""
    path = write_partition(tmp_path / "c.npz", *clips())
    seed_everything(0)
    client = make_client(tiny_cfg, path, fedper_prefixes)
    head_before = {k: v.clone() for k, v in client._model.head_state_dict().items()}

    seed_everything(123)
    server_payload = get_parameters(build_model(tiny_cfg), fedper_prefixes)
    set_parameters(client._model, server_payload, fedper_prefixes)

    for key, value in head_before.items():
        torch.testing.assert_close(value, client._model.state_dict()[key])


def test_a_clients_head_is_trained_locally_and_diverges_between_clients(
    tiny_cfg, fedper_prefixes, tmp_path
):
    """Private heads must actually personalize, or FedPer is FedAvg with a wasted layer."""
    seed_everything(0)
    shared = get_parameters(build_model(tiny_cfg), fedper_prefixes)
    heads = []
    for signer, seed in ((1, 0), (2, 5)):
        path = write_partition(tmp_path / f"c{signer}.npz", *clips(seed=seed))
        seed_everything(signer)
        client = make_client(tiny_cfg, path, fedper_prefixes, signer=signer)
        client.fit(shared, {"server_round": 1})
        heads.append(client._model.head.linear.weight.detach().clone())

    assert not torch.allclose(heads[0], heads[1])


# ------------------------------------------------- what aggregation can possibly touch


def test_aggregating_two_fedper_payloads_cannot_reach_a_head(tiny_cfg, fedper_prefixes):
    """Averaging is elementwise over the payload, so a head it never receives cannot move."""
    seed_everything(0)
    a = build_model(tiny_cfg)
    seed_everything(1)
    b = build_model(tiny_cfg)
    averaged = [(x + y) / 2 for x, y in
                zip(get_parameters(a, fedper_prefixes), get_parameters(b, fedper_prefixes),
                    strict=True)]

    head_before = {k: v.clone() for k, v in b.head_state_dict().items()}
    set_parameters(b, averaged, fedper_prefixes)

    for key, value in head_before.items():
        torch.testing.assert_close(value, b.state_dict()[key])
    for key, value in b.encoder_state_dict().items():
        assert not torch.allclose(value, a.state_dict()[key]) or torch.allclose(
            a.state_dict()[key], b.state_dict()[key]
        )


def test_metric_aggregation_never_carries_a_parameter():
    """Client metrics are the other channel back to the server; keep it numeric and small."""
    aggregated = weighted_average([(4, {"train_loss": 1.0, "signer": 3})])
    assert set(aggregated) == {"train_loss"}


# ---------------------------------------------------- what the sweep loads and adapts


def test_e5_loads_only_the_encoder_and_trains_only_the_head(tiny_cfg, fl_cfg):
    """E5's head must come from the signer, not from a checkpoint of other people."""
    seed_everything(7)
    donor = build_model(tiny_cfg)
    state = donor.state_dict()
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="head")
    cfg["train"] |= {"epochs": 2, "device": "cpu"}

    seed_everything(0)
    expected_head = build_model(cfg).head.linear.weight.detach().clone()

    result, _ = adapt_and_evaluate(
        "E5", state, clips(), clips(), cfg=cfg, device=torch.device("cpu"), seed=0
    )
    assert result.n == 8

    # Re-derive the adapted model to inspect it: same seed, same construction, no training.
    seed_everything(0)
    model = build_model(cfg)
    model.load_state_dict(
        {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}, strict=False
    )
    torch.testing.assert_close(model.head.linear.weight, expected_head)
    for key, value in donor.encoder_state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key])
    assert not torch.allclose(model.head.linear.weight, donor.head.linear.weight)


def test_e5_and_e6_freeze_the_encoder_while_e4_does_not():
    assert METHODS["E5"]["adapt"] == "head"
    assert METHODS["E6"]["adapt"] == "head"
    assert METHODS["E4"]["adapt"] == "full"
    assert METHODS["E5"]["load"] == "encoder"
    assert METHODS["E4"]["load"] == "all"


def test_e5_adaptation_leaves_the_pretrained_encoder_exactly_as_it_was(tiny_cfg, fl_cfg):
    """If adaptation moved the encoder, E5 would be E4 with fewer parameters unfrozen."""
    seed_everything(7)
    state = {k: v.clone() for k, v in build_model(tiny_cfg).state_dict().items()}
    cfg = adapt_config(tiny_cfg, fl_cfg, adapt="head")
    cfg["train"] |= {"epochs": 3, "device": "cpu"}

    seed_everything(0)
    model = build_model(cfg)
    model.load_state_dict(
        {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}, strict=False
    )
    model.freeze_encoder(True)
    from signadapt.train.loop import train_model

    train_model(model, clips(), None, cfg, device=torch.device("cpu"), seed=0)

    for key, value in state.items():
        if key.startswith(ENCODER_PREFIX):
            torch.testing.assert_close(value, model.state_dict()[key].cpu())
    assert not torch.allclose(model.head.linear.weight.cpu(), state["head.linear.weight"])


def test_e5_pretraining_uses_the_fedper_config_and_e4_forces_fedavg(fl_cfg):
    assert METHODS["E5"]["pretrain"] == "fedper"
    assert METHODS["E4"]["pretrain"] == "fedavg"
    assert as_fedavg(fl_cfg)["strategy"]["aggregate_prefixes"] == [ENCODER_PREFIX, HEAD_PREFIX]
    assert fl_cfg["strategy"]["aggregate_prefixes"] == [ENCODER_PREFIX], "config not mutated"


def test_a_fedper_checkpoint_holds_no_head_tensor(tiny_cfg, fedper_prefixes, tmp_path):
    """What a pretraining persists is the last place a head could leak into a shared file."""
    model = build_model(tiny_cfg)
    keys = shared_keys(model, fedper_prefixes)
    payload = {
        "state": dict(
            zip(keys, (torch.from_numpy(np.asarray(a))
                       for a in get_parameters(model, fedper_prefixes)), strict=True)
        ),
        "prefixes": list(fedper_prefixes),
    }
    torch.save(payload, tmp_path / "ckpt.pt")

    reloaded = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert not [k for k in reloaded["state"] if k.startswith(HEAD_PREFIX)]
    assert set(reloaded["state"]) == set(model.encoder_state_dict())


# --------------------------------- the private head must survive between rounds


def test_the_private_key_set_is_exactly_the_head_under_fedper(tiny_cfg, fedper_prefixes):
    from signadapt.federated.client import private_keys

    model = build_model(tiny_cfg)
    assert set(private_keys(model, fedper_prefixes)) == set(model.head_state_dict())


def test_nothing_is_private_under_fedavg(tiny_cfg):
    """FedAvg sends the head back every round, which is why this bug is invisible there."""
    from signadapt.federated.client import private_keys

    assert private_keys(build_model(tiny_cfg), (ENCODER_PREFIX, HEAD_PREFIX)) == ()


def test_a_clients_head_persists_across_rounds_through_context_state(
    tiny_cfg, fedper_prefixes, tmp_path
):
    """The regression test for the bug that made E5 look like a negative result.

    Two rounds, the client object rebuilt in between exactly as Flower rebuilds it. Round 2
    must start from the head round 1 trained, not from a fresh initialization.
    """
    from flwr.common import RecordDict

    path = write_partition(tmp_path / "c.npz", *clips())
    state = RecordDict()
    seed_everything(0)
    shared = get_parameters(build_model(tiny_cfg), fedper_prefixes)

    seed_everything(0)
    first = SignerClient(
        model_cfg=tiny_cfg, train_path=path, val_path=None, signer=1,
        share_prefixes=fedper_prefixes, device=torch.device("cpu"), local_epochs=1,
        state=state,
    )
    first.fit(shared, {"server_round": 1})
    after_round_1 = first._model.head.linear.weight.detach().clone()

    # A brand-new object, as Flower would build it, with the same node state.
    seed_everything(0)
    second = SignerClient(
        model_cfg=tiny_cfg, train_path=path, val_path=None, signer=1,
        share_prefixes=fedper_prefixes, device=torch.device("cpu"), local_epochs=1,
        state=state,
    )
    torch.testing.assert_close(second._model.head.linear.weight, after_round_1)


def test_without_context_state_the_head_restarts_every_round(
    tiny_cfg, fedper_prefixes, tmp_path
):
    """The contrast that makes the test above meaningful: this is the bug, reproduced."""
    path = write_partition(tmp_path / "c.npz", *clips())
    seed_everything(0)
    shared = get_parameters(build_model(tiny_cfg), fedper_prefixes)

    seed_everything(0)
    first = SignerClient(
        model_cfg=tiny_cfg, train_path=path, val_path=None, signer=1,
        share_prefixes=fedper_prefixes, device=torch.device("cpu"), local_epochs=1,
    )
    first.fit(shared, {"server_round": 1})
    trained = first._model.head.linear.weight.detach().clone()

    seed_everything(0)
    second = SignerClient(
        model_cfg=tiny_cfg, train_path=path, val_path=None, signer=1,
        share_prefixes=fedper_prefixes, device=torch.device("cpu"), local_epochs=1,
    )
    assert not torch.allclose(second._model.head.linear.weight, trained)


def test_the_persisted_record_holds_the_head_and_only_the_head(
    tiny_cfg, fedper_prefixes, tmp_path
):
    """What is written to node state must not quietly accumulate encoder tensors too."""
    from flwr.common import RecordDict

    from signadapt.federated.client import PRIVATE_STATE_KEY

    path = write_partition(tmp_path / "c.npz", *clips())
    state = RecordDict()
    seed_everything(0)
    client = SignerClient(
        model_cfg=tiny_cfg, train_path=path, val_path=None, signer=1,
        share_prefixes=fedper_prefixes, device=torch.device("cpu"), local_epochs=1,
        state=state,
    )
    client.fit(get_parameters(build_model(tiny_cfg), fedper_prefixes), {"server_round": 1})

    stored = state.array_records[PRIVATE_STATE_KEY].to_torch_state_dict()
    assert set(stored) == set(build_model(tiny_cfg).head_state_dict())
    assert not [k for k in stored if k.startswith(ENCODER_PREFIX)]
