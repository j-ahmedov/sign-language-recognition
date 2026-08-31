"""Flower ``NumPyClient``, one instance per signer (PLAN.md sections 4, 6 and 7).

A client owns one signer's clips and never shares them. What it does share is set by
``share_prefixes``: ``("encoder.", "head.")`` is FedAvg, ``("encoder.",)`` is FedPer, and the
difference between E4 and E5 is exactly that tuple.

Clients run inside Ray actor processes, so two things are handled here rather than left to
the caller. Partitions are read from ``.npy`` files with a small per-process cache, because
re-normalizing a signer's 320 clips on every round of every client would dominate the run
time; and torch is pinned to one thread, because Ray hands each actor one core and letting
ten actors each spawn ten threads makes the simulation slower than running it serially.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import Context

from signadapt.federated.parameters import get_parameters, set_parameters, shared_keys
from signadapt.models.model import build_model
from signadapt.train.loop import evaluate_tensors, train_model

# At most two partitions per actor: one client's data is ~38 MB at T=64, and an unbounded
# cache would let each of ten actors accumulate all ten partitions.
_CACHE_SIZE = 2


@lru_cache(maxsize=_CACHE_SIZE)
def load_partition(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one client's stacked clips, caching the result per worker process.

    Args:
        path: Path to the ``.npz`` written by :func:`signadapt.federated.simulation.
            write_partition`.

    Returns:
        ``(X, y)`` tensors.
    """
    with np.load(path) as payload:
        return torch.from_numpy(payload["x"]), torch.from_numpy(payload["y"]).long()


class SignerClient(NumPyClient):
    """One federated client: a single signer's data, plus a local copy of the model.

    Attributes:
        signer: The signer this client represents, for the round's metrics.
        share_prefixes: Which state-dict prefixes this client transmits.
    """

    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        train_path: str,
        val_path: str | None,
        signer: int,
        share_prefixes: Sequence[str],
        device: torch.device,
        seed: int = 0,
        local_epochs: int = 1,
    ) -> None:
        """Build a client.

        Args:
            model_cfg: Loaded ``configs/model.yaml``, with ``train`` already reconciled
                against ``configs/fl.yaml``'s ``client`` block by the simulation.
            train_path: ``.npz`` holding this signer's training clips.
            val_path: ``.npz`` for local evaluation, or ``None`` to evaluate on the training
                clips (which is what FedAvg's distributed evaluation does when no local
                holdout exists).
            signer: Signer id.
            share_prefixes: State-dict prefixes to transmit.
            device: Where this client trains; CPU inside Ray actors.
            seed: Seed for local shuffling and augmentation.
            local_epochs: Local epochs per round.
        """
        self.signer = signer
        self.share_prefixes = tuple(share_prefixes)
        self._model_cfg = model_cfg
        self._train_path = train_path
        self._val_path = val_path
        self._device = device
        self._seed = seed
        self._local_epochs = local_epochs
        self._model = build_model(model_cfg)
        # Fail here, on construction, rather than mid-round: a prefix that matches nothing
        # would make every round a silent no-op.
        shared_keys(self._model, self.share_prefixes)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Return this client's transmittable parameters.

        Args:
            config: Flower's per-call config; unused.

        Returns:
            Arrays in :func:`~signadapt.federated.parameters.shared_keys` order.
        """
        del config
        return get_parameters(self._model, self.share_prefixes)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Run local training for one round.

        Args:
            parameters: The global parameters for this round.
            config: Per-round config from the strategy; ``local_epochs`` and ``server_round``
                are read if present.

        Returns:
            ``(updated_parameters, n_train_examples, metrics)``. The example count is what
            FedAvg weights the average by, so it must be the real local dataset size.
        """
        set_parameters(self._model, parameters, self.share_prefixes)
        x, y = load_partition(self._train_path)
        server_round = int(config.get("server_round", 0))

        outcome = train_model(
            self._model,
            (x, y),
            None,
            self._model_cfg,
            device=self._device,
            # Vary the shuffle and augmentation stream per round, but reproducibly: without
            # the round in the seed every round would replay the same batch order.
            seed=self._seed + 1000 * server_round + self.signer,
            epochs=int(config.get("local_epochs", self._local_epochs)),
        )
        last = outcome.history[-1] if outcome.history else {}
        return (
            self.get_parameters({}),
            int(y.numel()),
            {
                "signer": self.signer,
                "train_loss": float(last.get("train_loss", float("nan"))),
                "train_top1": float(last.get("train_top1", float("nan"))),
                "seconds": outcome.seconds,
            },
        )

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the global parameters on this client's local data.

        Args:
            parameters: The global parameters for this round.
            config: Per-round config; unused.

        Returns:
            ``(loss, n_examples, metrics)``.
        """
        del config
        set_parameters(self._model, parameters, self.share_prefixes)
        x, y = load_partition(self._val_path or self._train_path)
        result = evaluate_tensors(self._model, (x, y), self._model_cfg, device=self._device)
        return result.loss, int(y.numel()), {"top1": result.top1, "signer": self.signer}


def make_client_fn(
    *,
    model_cfg: dict[str, Any],
    partitions: Sequence[dict[str, Any]],
    share_prefixes: Sequence[str],
    device: str = "cpu",
    seed: int = 0,
    local_epochs: int = 1,
    torch_threads: int = 1,
) -> Callable[[Context], Client]:
    """Build the ``client_fn`` that Flower calls to instantiate a client.

    Args:
        model_cfg: Loaded model config.
        partitions: One entry per client, as produced by the simulation; each has
            ``signer``, ``train_path`` and optionally ``val_path``.
        share_prefixes: State-dict prefixes clients transmit.
        device: Client device. CPU by default -- ten Ray actors contending for one MPS
            queue is slower than ten actors on ten cores, and MPS is not partitionable the
            way ``client_resources`` assumes.
        seed: Base seed.
        local_epochs: Local epochs per round.
        torch_threads: Threads per actor; 1 avoids oversubscribing the cores Ray allocated.

    Returns:
        A callable suitable for :class:`flwr.client.ClientApp`.
    """
    specs = list(partitions)

    def client_fn(context: Context) -> Client:
        torch.set_num_threads(torch_threads)
        partition_id = int(context.node_config["partition-id"])
        spec = specs[partition_id]
        return SignerClient(
            model_cfg=model_cfg,
            train_path=spec["train_path"],
            val_path=spec.get("val_path"),
            signer=int(spec["signer"]),
            share_prefixes=share_prefixes,
            device=torch.device(device),
            seed=seed,
            local_epochs=local_epochs,
        ).to_client()

    return client_fn


def write_partition(path: str | Path, x: torch.Tensor, y: torch.Tensor) -> str:
    """Write one client's stacked clips to disk for the Ray actors to read.

    Args:
        path: Destination ``.npz``.
        x: ``(N, T, L, 4)`` clips.
        y: ``(N,)`` labels.

    Returns:
        The path as a string, which is what travels into the actor.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, x=x.numpy(), y=y.numpy())
    return str(destination)
