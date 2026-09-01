"""Single-machine Flower simulation: one client per signer (PLAN.md sections 6, 7 and 8).

Two runs live here and they answer different questions.

``iid-check`` is the correctness gate from PLAN.md section 8, week 4. It partitions *the same
clips as E2* uniformly at random across the same number of clients, so the federation sees an
IID split and should land within a few points of centralized training on that data. Until
that check passes, a federated result is uninterpretable: a broken aggregation loop and a
genuine negative result look identical from the outside.

``fedavg`` is the real thing -- one client per signer, which is not IID at all, since a
client's entire dataset is one person's signing. It produces the shared encoder that phase 4
personalizes, so the global parameters are checkpointed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

from signadapt.data.dataset import (
    ClipRecord,
    KeypointDataset,
    Split,
    load_records,
    make_splits,
)
from signadapt.federated.client import make_client_fn, write_partition
from signadapt.federated.parameters import get_parameters, set_parameters, shared_keys
from signadapt.federated.strategy import build_strategy
from signadapt.models.model import HEAD_PREFIX, build_model
from signadapt.train.loop import evaluate_tensors, resolve_device, stack_dataset
from signadapt.utils.config import apply_overrides, config_fingerprint, load_config
from signadapt.utils.metrics import format_mean_std, mean_std
from signadapt.utils.results import ResultsLogger, load_results
from signadapt.utils.seeding import seed_everything, temporary_seed

PARTITION_MODES = ("signer", "iid")

# What a FedAvg run transmits: everything. Named here because both the correctness check and
# the E4 pretraining run must override configs/fl.yaml, whose default strategy is fedper.
FEDAVG_PREFIXES = ("encoder.", "head.")


def as_fedavg(fl_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the FL config forced to plain FedAvg.

    Args:
        fl_cfg: Loaded ``configs/fl.yaml``.

    Returns:
        The same config with ``strategy.name`` set to ``fedavg`` and every parameter shared.
    """
    return {
        **fl_cfg,
        "strategy": {
            **fl_cfg["strategy"],
            "name": "fedavg",
            "aggregate_prefixes": list(FEDAVG_PREFIXES),
        },
    }


def client_model_config(model_cfg: dict[str, Any], fl_cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``configs/fl.yaml``'s ``client`` block onto the model config's ``train`` block.

    Without this the clients would silently train with ``configs/model.yaml``'s centralized
    hyperparameters and the ``client:`` block would be documentation rather than
    configuration. The local schedule is forced to constant -- see
    :func:`signadapt.train.loop.build_schedule` for why a within-round cosine is wrong.

    Args:
        model_cfg: Loaded model config.
        fl_cfg: Loaded FL config.

    Returns:
        A copy of ``model_cfg`` with the client hyperparameters applied.
    """
    client = fl_cfg["client"]
    cfg = {**model_cfg, "train": {**model_cfg["train"]}}
    cfg["train"] |= {
        "batch_size": int(client["batch_size"]),
        "lr": float(client["lr"]),
        "weight_decay": float(client["weight_decay"]),
        "epochs": int(client["local_epochs"]),
        "scheduler": "constant",
        "early_stopping_patience": 0,
    }
    return cfg


def _tensors(
    records: list[ClipRecord], indices: tuple[int, ...], data_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a set of record indices into stacked tensors."""
    return stack_dataset(KeypointDataset(records, indices, data_cfg))


def partition_indices(
    records: list[ClipRecord],
    indices: tuple[int, ...],
    mode: str,
    *,
    seed: int = 0,
) -> list[tuple[int, tuple[int, ...]]]:
    """Split a set of clips across clients.

    Args:
        records: All records.
        indices: The training indices to divide up.
        mode: ``"signer"`` gives one client per signer -- the real, non-IID setting.
            ``"iid"`` shuffles the same clips into the same number of equal shards, which is
            the control condition for the correctness check.
        seed: Seed for the IID shuffle.

    Returns:
        ``(client_id, indices)`` pairs. Under ``"signer"`` the client id *is* the signer id,
        which keeps the per-client metrics attributable to a person.

    Raises:
        ValueError: On an unknown mode.
    """
    if mode not in PARTITION_MODES:
        raise ValueError(f"unknown partition mode {mode!r}, expected one of {PARTITION_MODES}")

    by_signer: dict[int, list[int]] = {}
    for index in indices:
        by_signer.setdefault(records[index].signer, []).append(index)

    if mode == "signer":
        return [(signer, tuple(sorted(v))) for signer, v in sorted(by_signer.items())]

    # Same number of clients and the same clips, only the assignment changes -- so any gap
    # against centralized training is attributable to federation, not to the data.
    n_clients = len(by_signer)
    with temporary_seed(seed):
        order = np.random.permutation(list(indices))
    shards = np.array_split(order, n_clients)
    return [(i, tuple(sorted(int(v) for v in shard))) for i, shard in enumerate(shards)]


def write_partitions(
    records: list[ClipRecord],
    parts: list[tuple[int, tuple[int, ...]]],
    data_cfg: dict[str, Any],
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Materialize each client's clips to a ``.npz`` the Ray actors can read.

    Args:
        records: All records.
        parts: Output of :func:`partition_indices`.
        data_cfg: Loaded data config.
        out_dir: Directory for the partition files.

    Returns:
        One spec per client: ``{"signer", "train_path", "n"}``.
    """
    specs = []
    for client_id, indices in parts:
        x, y = _tensors(records, indices, data_cfg)
        path = write_partition(out_dir / f"client{client_id:02d}.npz", x, y)
        specs.append({"signer": client_id, "train_path": path, "n": int(y.numel())})
    return specs


def run_simulation_round_trip(
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    specs: list[dict[str, Any]],
    val_data: tuple[torch.Tensor, torch.Tensor] | None,
    seed: int,
    on_round: Any = None,
) -> dict[str, Any]:
    """Run one Flower simulation and return its history and chosen global parameters.

    With ``val_data``, model selection is server-side and centralized: the global parameters
    are scored on the held-out validation signer after every round and the best round wins.
    Selecting on the clients' own partitions would instead select for fitting the training
    signers, which is the opposite of what E2 and E5 are about.

    ``val_data=None`` disables selection and returns the final round. That is the setting
    phase 4 pretrains under, and it is a fairness requirement rather than a shortcut: a
    FedPer server holds no head, so it *cannot* score a global model on a validation signer.
    Letting FedAvg pick its best round while FedPer must take its last one would hand E4 an
    advantage that has nothing to do with the method being compared.

    Args:
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded ``configs/fl.yaml``.
        specs: Client specs from :func:`write_partitions`.
        val_data: ``(X, y)`` for server-side validation, or ``None`` to take the final round.
        seed: Seed for the shared initial parameters and the clients.
        on_round: Called with each round's record.

    Returns:
        ``{"rounds", "best_round", "best_val_top1", "best_arrays", "keys", "prefixes"}``.
    """
    strategy_cfg = fl_cfg["strategy"]
    prefixes = tuple(strategy_cfg["aggregate_prefixes"])
    private = tuple(strategy_cfg.get("private_prefixes", ()))
    server_cfg, client_cfg, sim_cfg = fl_cfg["server"], fl_cfg["client"], fl_cfg["simulation"]

    device = resolve_device(model_cfg["train"].get("device", "auto"))
    seed_everything(seed)
    reference = build_model(model_cfg)
    keys = shared_keys(reference, prefixes)
    initial = get_parameters(reference, prefixes)

    state: dict[str, Any] = {"best_val": float("nan"), "best_round": -1, "best_arrays": initial}
    server_model = build_model(model_cfg).to(device)

    def evaluate_fn(
        server_round: int, arrays: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        del config
        set_parameters(server_model, arrays, prefixes)
        result = evaluate_tensors(server_model, val_data, model_cfg, device=device)
        best = state["best_val"]
        if np.isnan(best) or result.top1 > best:
            state.update(
                best_val=result.top1,
                best_round=server_round,
                best_arrays=[np.array(a, copy=True) for a in arrays],
            )
        return result.loss, {"val_top1": result.top1, "val_top5": result.top5}

    strategy = build_strategy(
        strategy_cfg["name"],
        aggregate_prefixes=prefixes,
        private_prefixes=private,
        on_round=on_round,
        fraction_fit=float(server_cfg["fraction_fit"]),
        fraction_evaluate=float(server_cfg.get("fraction_evaluate", 0.0)),
        min_fit_clients=min(int(server_cfg["min_fit_clients"]), len(specs)),
        min_evaluate_clients=min(int(server_cfg.get("min_fit_clients", 2)), len(specs)),
        min_available_clients=len(specs),
        initial_parameters=ndarrays_to_parameters(initial),
        evaluate_fn=evaluate_fn if val_data is not None else None,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": int(client_cfg["local_epochs"]),
        },
    )

    client_fn = make_client_fn(
        model_cfg=client_model_config(model_cfg, fl_cfg),
        partitions=specs,
        share_prefixes=prefixes,
        device=str(sim_cfg.get("client_device", "cpu")),
        seed=seed,
        local_epochs=int(client_cfg["local_epochs"]),
    )

    run_simulation(
        server_app=ServerApp(
            server_fn=lambda context: _components(context, strategy, int(server_cfg["num_rounds"]))
        ),
        client_app=ClientApp(client_fn=client_fn),
        num_supernodes=len(specs),
        backend_config={"client_resources": dict(sim_cfg["client_resources"])},
    )

    final = (
        parameters_to_ndarrays(strategy.latest_parameters)
        if strategy.latest_parameters is not None
        else state["best_arrays"]
    )
    chosen = state["best_arrays"] if val_data is not None else final
    return {
        "rounds": strategy.rounds,
        "best_round": state["best_round"] if val_data is not None else len(strategy.rounds),
        "best_val_top1": state["best_val"],
        "best_arrays": chosen,
        "final_arrays": final,
        "keys": list(keys),
        "prefixes": list(prefixes),
    }


def federated_pretrain(
    records: list[ClipRecord],
    fold: Split,
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    seed: int = 0,
    work_dir: str = "data/cache/partitions",
    cache_dir: str | None = "data/checkpoints/pretrain",
) -> dict[str, Any]:
    """Federate over one LOSO fold's training signers and return the shared parameters.

    The held-out signer's clips are never given to any client, which is what makes the
    personalized result in phase 4 mean anything -- an encoder that had already seen the
    "new" signer would be measuring memory, not adaptation.

    No validation-based round selection is done: see :func:`run_simulation_round_trip` for
    why taking the final round is the fair rule when FedAvg and FedPer are being compared.

    Args:
        records: All records.
        fold: The LOSO fold; only ``fold.train`` is used.
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded FL config, already forced to the intended strategy.
        seed: Seed.
        work_dir: Where client partitions are materialized.
        cache_dir: Where finished pretrainings are cached, or ``None`` to disable. A sweep
            reuses one pretraining across every k, and a re-run reuses it across invocations;
            without the cache the k-sweep would repeat a five-minute simulation five times
            for results that cannot differ.

    Returns:
        ``{"state": {key: tensor}, "prefixes": [...], "n_rounds": int, "clients": [...],
        "cached": bool}``.
    """
    strategy_name = fl_cfg["strategy"]["name"]
    tag = f"{strategy_name}_{fold.name}_seed{seed}"
    cache_path = Path(cache_dir) / f"{tag}.pt" if cache_dir else None
    # A cache keyed only by (strategy, fold, seed) would happily hand a 50-round experiment
    # the 2-round artefact a smoke test left behind: same filename, same shapes, silently
    # wrong numbers. The fingerprint covers everything that changes what the file contains.
    fingerprint = config_fingerprint(
        model_cfg["encoder"], model_cfg["head"], fl_cfg["server"], fl_cfg["client"],
        fl_cfg["strategy"], fold.signers["train"], seed,
    )
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            return {**payload, "cached": True}
        print(f"[pretrain] {tag}: cached under a different config, recomputing")

    parts = partition_indices(records, fold.train, "signer")
    specs = write_partitions(records, parts, data_cfg, Path(work_dir) / tag)
    outcome = run_simulation_round_trip(
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        fl_cfg=fl_cfg,
        specs=specs,
        val_data=None,
        seed=seed,
    )
    payload = {
        "state": {
            key: torch.from_numpy(np.asarray(array))
            for key, array in zip(outcome["keys"], outcome["best_arrays"], strict=True)
        },
        "prefixes": outcome["prefixes"],
        "n_rounds": len(outcome["rounds"]),
        "clients": [int(s["signer"]) for s in specs],
        "strategy": strategy_name,
        "fold": fold.name,
        "seed": seed,
        "fingerprint": fingerprint,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return {**payload, "cached": False}


def _components(context: Context, strategy: Any, num_rounds: int) -> ServerAppComponents:
    """Hand Flower the pre-built strategy so it stays readable after the run."""
    del context
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


def run_federated(
    mode: str,
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    seed: int = 0,
    results_dir: str = "results",
    checkpoint_dir: str = "data/checkpoints",
    work_dir: str = "data/cache/partitions",
) -> dict[str, Any]:
    """Run one federated experiment end to end and write its result JSON.

    Args:
        mode: ``"signer"`` (the real federation) or ``"iid"`` (the correctness control).
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded FL config.
        seed: Seed.
        results_dir: Where the result JSON goes.
        checkpoint_dir: Where the best global parameters go, for phase 4 to personalize.
        work_dir: Where client partitions are materialized.

    Returns:
        The metrics dict written to the result file.
    """
    strategy_name = fl_cfg["strategy"]["name"]
    experiment = "fedavg-iid-check" if mode == "iid" else f"{strategy_name}-pretrain"

    # This function scores the *global* model on a held-out signer. That only means anything
    # when the head travels with the encoder: under FedPer the server's head is the initial
    # random one, nothing ever trains it, and the run would report a confident-looking
    # chance-level number (about 1/64 on LSA64) instead of failing. E5 is evaluated after
    # per-client personalization, which is phase 4 -- see signadapt.personalize.adapt.
    if not any(HEAD_PREFIX.startswith(p) for p in fl_cfg["strategy"]["aggregate_prefixes"]):
        raise ValueError(
            f"strategy {strategy_name!r} does not aggregate {HEAD_PREFIX!r}, so the global "
            "model has an untrained head and its centralized accuracy is meaningless. "
            "Run this with FedAvg (see as_fedavg), or evaluate FedPer through the phase-4 "
            "personalization sweep."
        )

    data_cfg = {**data_cfg, "splits": {**data_cfg["splits"], "mode": "signer_independent"}}
    seed_everything(seed)
    records = load_records(
        data_cfg["dataset"]["cache_dir"],
        exclude_signers=data_cfg["splits"].get("exclude_signers", ()),
    )
    split = make_splits(records, data_cfg, seed=seed)

    parts = partition_indices(records, split.train, mode, seed=seed)
    out_dir = Path(work_dir) / f"{mode}-seed{seed}"
    specs = write_partitions(records, parts, data_cfg, out_dir)
    val_data = _tensors(records, split.val, data_cfg)
    test_data = _tensors(records, split.test, data_cfg)

    description = {
        "iid": "Correctness check (PLAN.md section 8, week 4): FedAvg over an IID partition "
        "of the E2 training clips. Must land within checks.iid_tolerance of "
        "centralized E2, otherwise the federated loop is broken.",
        "signer": f"{strategy_name} over one client per training signer -- the non-IID "
        "setting the thesis is about. Produces the shared encoder that phase 4 "
        "personalizes.",
    }[mode]

    with ResultsLogger(
        experiment,
        config={"model": model_cfg, "data": data_cfg, "fl": fl_cfg},
        seed=seed,
        results_dir=results_dir,
        description=description,
        extra={
            "partition": {
                "mode": mode,
                "n_clients": len(specs),
                "sizes": {str(s["signer"]): s["n"] for s in specs},
            },
            "split": {"name": split.name, "sizes": split.sizes(), "signers": split.signers},
        },
    ) as log:
        outcome = run_simulation_round_trip(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=fl_cfg,
            specs=specs,
            val_data=val_data,
            seed=seed,
            on_round=lambda record: log.log_record(**record),
        )

        device = resolve_device(model_cfg["train"].get("device", "auto"))
        model = build_model(model_cfg).to(device)
        set_parameters(model, outcome["best_arrays"], outcome["prefixes"])
        test = evaluate_tensors(
            model, test_data, model_cfg, device=device, records=records, indices=split.test
        )

        checkpoint = Path(checkpoint_dir) / f"{experiment}_seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "experiment": experiment,
                "seed": seed,
                "prefixes": outcome["prefixes"],
                "state": dict(
                    zip(
                        outcome["keys"],
                        (torch.from_numpy(np.asarray(a)) for a in outcome["best_arrays"]),
                        strict=True,
                    )
                ),
            },
            checkpoint,
        )

        metrics = test.to_dict() | {
            "best_round": outcome["best_round"],
            "best_val_top1": outcome["best_val_top1"],
            "n_rounds": len(outcome["rounds"]),
            "n_shared_tensors": len(outcome["keys"]),
            "checkpoint": str(checkpoint),
        }
        log.set_metrics(**metrics)

    print(
        f"[{experiment}] seed {seed}: test top-1 {test.top1:.4f}  top-5 {test.top5:.4f}  "
        f"(best round {outcome['best_round']}/{len(outcome['rounds'])}, "
        f"val {outcome['best_val_top1']:.4f}, {len(specs)} clients)"
    )
    return metrics


def check_against_centralized(
    metrics: dict[str, Any], fl_cfg: dict[str, Any], *, results_dir: str = "results"
) -> bool:
    """Compare an IID federated run against the centralized E2 baseline.

    Args:
        metrics: The IID run's metrics.
        fl_cfg: Loaded FL config, for ``checks.iid_tolerance``.
        results_dir: Where to look for the E2 results.

    Returns:
        ``True`` if the difference is within tolerance, or if there is no E2 result to
        compare against (in which case the caller is told to run ``make train`` first).
    """
    tolerance = float(fl_cfg["checks"]["iid_tolerance"])
    baseline = [r["metrics"]["top1"] for r in load_results(results_dir, experiment="E2")]
    if not baseline:
        print("[iid-check] no E2 result to compare against; run `make train` first.")
        return True

    summary = mean_std(baseline)
    delta = metrics["top1"] - summary["mean"]
    verdict = "PASS" if abs(delta) <= tolerance else "FAIL"
    print(
        f"[iid-check] {verdict}  FedAvg-IID {metrics['top1'] * 100:.1f} % vs "
        f"centralized E2 {format_mean_std(summary)} "
        f"(delta {delta * 100:+.1f} points, tolerance +/-{tolerance * 100:.0f})"
    )
    return abs(delta) <= tolerance


def main(argv: list[str] | None = None) -> int:
    """Run the federated experiments from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; non-zero when the IID correctness check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fl.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "iid-check", "fedavg"],
        help="'all' runs the IID correctness check first and stops if it fails",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="shorthand for --seeds SEED")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an fl-config key",
    )
    args = parser.parse_args(argv)

    fl_cfg = apply_overrides(load_config(args.config), args.overrides)
    model_cfg = load_config(args.model_config)
    data_cfg = load_config(args.data_config)
    seeds = args.seeds or ([args.seed] if args.seed is not None else [0])

    if args.experiment in ("all", "iid-check"):
        # The check is a property of the loop, not of the seed, so one run settles it.
        metrics = run_federated(
            "iid",
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=as_fedavg(fl_cfg),
            seed=seeds[0],
            results_dir=args.results_dir,
        )
        if not check_against_centralized(metrics, fl_cfg, results_dir=args.results_dir):
            print(
                "[iid-check] the federated loop does not reproduce centralized training on "
                "IID data.\n            Fix it before reporting any federated number: a "
                "broken loop and a negative result look the same.",
            )
            return 1
        if args.experiment == "iid-check":
            return 0

    # E4's shared model is trained with FedAvg regardless of what configs/fl.yaml names as its
    # default strategy -- that default (fedper) describes the phase-4 E5 run, whose global
    # model has no trained head and cannot be scored the way this path scores one.
    runs = [
        run_federated(
            "signer",
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=as_fedavg(fl_cfg),
            seed=s,
            results_dir=args.results_dir,
        )
        for s in seeds
    ]
    summary = mean_std([r["top1"] for r in runs])
    print(f"[fedavg] over {len(runs)} seeds: top-1 {format_mean_std(summary)}")
    print(json.dumps({"checkpoints": [r["checkpoint"] for r in runs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
