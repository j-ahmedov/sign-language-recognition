"""Phase 3: the federated parameter boundary, the strategy and the client.

The parameter-boundary tests are the load-bearing ones. Everything the thesis claims about
privacy and about the difference between E4 and E5 is a claim about which tensors cross
this boundary, and none of it is visible in an accuracy number: a FedPer run that quietly
transmits the head still trains, still converges, and still reports plausible results.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from signadapt.data.dataset import ClipRecord
from signadapt.federated.client import SignerClient, load_partition, write_partition
from signadapt.federated.parameters import (
    assert_excludes,
    get_parameters,
    set_parameters,
    shared_keys,
)
from signadapt.federated.simulation import (
    client_model_config,
    partition_indices,
    write_partitions,
)
from signadapt.federated.strategy import RecordingFedAvg, build_strategy, weighted_average
from signadapt.models.model import ENCODER_PREFIX, HEAD_PREFIX, build_model
from signadapt.utils.config import load_config
from signadapt.utils.seeding import seed_everything

FEDAVG = ("encoder.", "head.")
FEDPER = ("encoder.",)


@pytest.fixture
def tiny_cfg():
    cfg = load_config("configs/model.yaml")
    cfg["encoder"] |= {"d_model": 32, "n_layers": 1, "n_heads": 2, "ff_dim": 32, "max_len": 8}
    cfg["head"] |= {"in_dim": 32, "n_classes": 4}
    cfg["train"] |= {"epochs": 2, "batch_size": 4, "device": "cpu", "scheduler": "constant"}
    cfg["augment"]["enabled"] = False
    return cfg


@pytest.fixture
def model(tiny_cfg):
    seed_everything(0)
    return build_model(tiny_cfg)


def fake_clips(n=8, n_classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, 115, 4, generator=generator)
    x[..., 3] = 1.0
    y = torch.arange(n) % n_classes
    return x, y


def records_for(signers=(1, 2, 3), n_classes=4, n_reps=5):
    records, index = [], 0
    for signer in signers:
        for label in range(n_classes):
            for rep in range(1, n_reps + 1):
                records.append(
                    ClipRecord(
                        clip_id=f"{label + 1:03d}_{signer:03d}_{rep:03d}",
                        path=f"/nonexistent/{index}.npy",
                        label=label,
                        signer=signer,
                        repetition=rep,
                        n_frames=8,
                        aspect=16 / 9,
                    )
                )
                index += 1
    return records


# ------------------------------------------------------- the parameter boundary


def test_fedavg_transmits_every_parameter(model):
    assert set(shared_keys(model, FEDAVG)) == set(model.state_dict())


def test_fedper_transmits_the_encoder_and_nothing_else(model):
    keys = shared_keys(model, FEDPER)
    assert set(keys) == set(model.encoder_state_dict())
    assert not any(k.startswith(HEAD_PREFIX) for k in keys)


def test_shared_keys_order_is_stable_across_model_instances(tiny_cfg):
    seed_everything(0)
    a = build_model(tiny_cfg)
    seed_everything(99)
    b = build_model(tiny_cfg)
    assert shared_keys(a, FEDPER) == shared_keys(b, FEDPER)


def test_a_prefix_matching_nothing_is_refused(model):
    """An empty payload makes every round a no-op that still reports plausible losses."""
    with pytest.raises(ValueError, match="nothing would be aggregated"):
        shared_keys(model, ("backbone.",))


def test_parameters_round_trip(tiny_cfg):
    seed_everything(0)
    a = build_model(tiny_cfg)
    seed_everything(1)
    b = build_model(tiny_cfg)
    set_parameters(b, get_parameters(a, FEDAVG), FEDAVG)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key])


def test_fedper_payload_leaves_the_receiving_head_untouched(tiny_cfg):
    """The whole point of E5: a client keeps its own classifier across rounds."""
    seed_everything(0)
    server = build_model(tiny_cfg)
    seed_everything(1)
    client = build_model(tiny_cfg)
    head_before = {k: v.clone() for k, v in client.head_state_dict().items()}

    set_parameters(client, get_parameters(server, FEDPER), FEDPER)

    for key, value in server.encoder_state_dict().items():
        torch.testing.assert_close(value, client.state_dict()[key])
    for key, value in head_before.items():
        torch.testing.assert_close(value, client.state_dict()[key])


def test_a_payload_of_the_wrong_length_is_refused(model):
    with pytest.raises(ValueError, match="payload has"):
        set_parameters(model, get_parameters(model, FEDPER)[:-1], FEDPER)


def test_a_payload_of_the_wrong_shape_is_refused(model):
    arrays = get_parameters(model, FEDPER)
    arrays[0] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="payload shape"):
        set_parameters(model, arrays, FEDPER)


def test_assert_excludes_catches_a_private_prefix():
    assert_excludes(["encoder.a", "encoder.b"], [HEAD_PREFIX])
    with pytest.raises(ValueError, match="private parameters would be transmitted"):
        assert_excludes(["encoder.a", "head.linear.weight"], [HEAD_PREFIX])


def test_build_strategy_refuses_a_fedper_config_that_would_share_the_head():
    """A misconfiguration here silently turns E5 back into E4 while still labelling it E5."""
    with pytest.raises(ValueError, match="private parameters"):
        build_strategy("fedper", aggregate_prefixes=FEDAVG, private_prefixes=(HEAD_PREFIX,))


def test_build_strategy_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("fedprox", aggregate_prefixes=FEDPER)


def test_build_strategy_records_what_it_shares():
    strategy = build_strategy("fedper", aggregate_prefixes=FEDPER, private_prefixes=(HEAD_PREFIX,))
    assert isinstance(strategy, RecordingFedAvg)
    assert strategy.aggregate_prefixes == FEDPER
    assert strategy.private_prefixes == (HEAD_PREFIX,)


# ----------------------------------------------------------------- metric aggregation


def test_weighted_average_weights_by_example_count():
    metrics = [(10, {"top1": 1.0}), (90, {"top1": 0.0})]
    assert weighted_average(metrics)["top1"] == pytest.approx(0.1)


def test_weighted_average_drops_identity_keys():
    """Averaging signer ids yields a number that looks like a metric and is not one."""
    assert "signer" not in weighted_average([(1, {"signer": 3, "top1": 0.5})])


def test_weighted_average_of_nothing_is_empty():
    assert weighted_average([]) == {}
    assert weighted_average([(0, {"top1": 1.0})]) == {}


# ------------------------------------------------------------------------ partitioning


def test_signer_partition_gives_one_client_per_signer():
    records = records_for(signers=(1, 2, 3))
    parts = partition_indices(records, tuple(range(len(records))), "signer")
    assert [client for client, _ in parts] == [1, 2, 3]
    for client, indices in parts:
        assert {records[i].signer for i in indices} == {client}


def test_iid_partition_keeps_the_same_clips_and_client_count():
    records = records_for(signers=(1, 2, 3))
    indices = tuple(range(len(records)))
    signer_parts = partition_indices(records, indices, "signer")
    iid_parts = partition_indices(records, indices, "iid", seed=0)

    assert len(iid_parts) == len(signer_parts), "the control must have the same client count"
    pooled = [i for _, part in iid_parts for i in part]
    assert sorted(pooled) == sorted(indices), "no clip may be lost or duplicated"


def test_iid_partition_actually_mixes_signers():
    records = records_for(signers=(1, 2, 3))
    parts = partition_indices(records, tuple(range(len(records))), "iid", seed=0)
    assert any(len({records[i].signer for i in part}) > 1 for _, part in parts)


def test_iid_partition_is_reproducible_and_seed_dependent():
    records = records_for(signers=(1, 2, 3))
    indices = tuple(range(len(records)))
    assert partition_indices(records, indices, "iid", seed=0) == partition_indices(
        records, indices, "iid", seed=0
    )
    assert partition_indices(records, indices, "iid", seed=0) != partition_indices(
        records, indices, "iid", seed=1
    )


def test_unknown_partition_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown partition mode"):
        partition_indices(records_for(), (0,), "dirichlet")


def test_client_model_config_applies_the_fl_client_block():
    model_cfg = load_config("configs/model.yaml")
    fl_cfg = load_config("configs/fl.yaml")
    cfg = client_model_config(model_cfg, fl_cfg)
    assert cfg["train"]["lr"] == float(fl_cfg["client"]["lr"])
    assert cfg["train"]["epochs"] == int(fl_cfg["client"]["local_epochs"])
    assert cfg["train"]["scheduler"] == "constant", "a within-round cosine is wrong"
    assert model_cfg["train"]["scheduler"] == "cosine", "the centralized config is untouched"


# ------------------------------------------------------------------------------ client


def test_partition_round_trips_through_disk(tmp_path):
    x, y = fake_clips()
    path = write_partition(tmp_path / "c.npz", x, y)
    loaded_x, loaded_y = load_partition(path)
    torch.testing.assert_close(loaded_x, x)
    torch.testing.assert_close(loaded_y, y)


def test_write_partitions_names_files_by_client(tmp_path, monkeypatch):
    records = records_for(signers=(1, 2))
    parts = partition_indices(records, tuple(range(len(records))), "signer")
    monkeypatch.setattr("signadapt.federated.simulation._tensors", lambda *a, **k: fake_clips(n=4))
    specs = write_partitions(records, parts, {}, tmp_path)
    assert [s["signer"] for s in specs] == [1, 2]
    assert all(s["train_path"].endswith(".npz") for s in specs)


def test_client_fit_returns_only_shared_parameters(tiny_cfg, tmp_path):
    """A FedPer client must not put its head on the wire, whatever else it does."""
    path = write_partition(tmp_path / "c.npz", *fake_clips())
    seed_everything(0)
    client = SignerClient(
        model_cfg=tiny_cfg,
        train_path=path,
        val_path=None,
        signer=7,
        share_prefixes=FEDPER,
        device=torch.device("cpu"),
        local_epochs=1,
    )
    incoming = get_parameters(build_model(tiny_cfg), FEDPER)
    outgoing, n, metrics = client.fit(incoming, {"server_round": 1})

    assert len(outgoing) == len(shared_keys(build_model(tiny_cfg), FEDPER))
    assert n == 8, "FedAvg weights by this count, so it must be the real dataset size"
    assert metrics["signer"] == 7


def test_client_fit_changes_the_encoder(tiny_cfg, tmp_path):
    path = write_partition(tmp_path / "c.npz", *fake_clips())
    seed_everything(0)
    client = SignerClient(
        model_cfg=tiny_cfg,
        train_path=path,
        val_path=None,
        signer=1,
        share_prefixes=FEDAVG,
        device=torch.device("cpu"),
        local_epochs=2,
    )
    incoming = client.get_parameters({})
    outgoing, _, _ = client.fit(incoming, {"server_round": 1})
    assert any(not np.allclose(a, b) for a, b in zip(incoming, outgoing, strict=True))


def test_client_evaluate_reports_local_accuracy(tiny_cfg, tmp_path):
    path = write_partition(tmp_path / "c.npz", *fake_clips())
    seed_everything(0)
    client = SignerClient(
        model_cfg=tiny_cfg,
        train_path=path,
        val_path=None,
        signer=2,
        share_prefixes=FEDAVG,
        device=torch.device("cpu"),
    )
    loss, n, metrics = client.evaluate(client.get_parameters({}), {})
    assert n == 8
    assert 0.0 <= metrics["top1"] <= 1.0
    assert loss > 0


def test_client_rejects_a_prefix_that_matches_nothing(tiny_cfg, tmp_path):
    path = write_partition(tmp_path / "c.npz", *fake_clips())
    with pytest.raises(ValueError, match="nothing would be aggregated"):
        SignerClient(
            model_cfg=tiny_cfg,
            train_path=path,
            val_path=None,
            signer=1,
            share_prefixes=("backbone.",),
            device=torch.device("cpu"),
        )


def test_encoder_and_head_prefixes_are_the_ones_the_config_names():
    fl_cfg = load_config("configs/fl.yaml")
    assert tuple(fl_cfg["strategy"]["aggregate_prefixes"]) == (ENCODER_PREFIX,)
    assert tuple(fl_cfg["strategy"]["private_prefixes"]) == (HEAD_PREFIX,)


# ------------------------------------------------- evaluating a global model without a head


def test_as_fedavg_shares_everything_whatever_the_config_says():
    from signadapt.federated.simulation import as_fedavg

    fl_cfg = load_config("configs/fl.yaml")
    assert fl_cfg["strategy"]["name"] == "fedper", "the config default describes phase 4"
    forced = as_fedavg(fl_cfg)
    assert forced["strategy"]["name"] == "fedavg"
    assert tuple(forced["strategy"]["aggregate_prefixes"]) == (ENCODER_PREFIX, HEAD_PREFIX)
    assert fl_cfg["strategy"]["name"] == "fedper", "the original config is not mutated"


def test_running_fedper_through_the_global_evaluation_path_is_refused():
    """Under FedPer nothing ever trains the server's head, so it would report chance."""
    from signadapt.federated.simulation import run_federated

    with pytest.raises(ValueError, match="untrained head"):
        run_federated(
            "signer",
            model_cfg=load_config("configs/model.yaml"),
            data_cfg=load_config("configs/data.yaml"),
            fl_cfg=load_config("configs/fl.yaml"),
        )
