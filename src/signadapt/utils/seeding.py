"""Deterministic seeding for every source of randomness in the project.

Reproducibility rule for SignAdapt: a run is identified by ``(config, seed)`` and nothing
else. Any number reported in the thesis must come back identical when the same pair is
re-run on the same machine.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "SeedState",
    "seed_everything",
    "seed_worker",
    "torch_generator",
    "temporary_seed",
]


@dataclass(frozen=True)
class SeedState:
    """What ``seed_everything`` actually managed to configure.

    Attributes:
        seed: The base seed applied to python, numpy and torch.
        deterministic_algorithms: Whether torch was put into deterministic-algorithm mode.
        note: Human-readable reason when strict determinism could not be enabled.
    """

    seed: int
    deterministic_algorithms: bool
    note: str = ""


def seed_everything(seed: int, *, strict: bool = True) -> SeedState:
    """Seed python, numpy and torch, and pin cudnn/torch to deterministic kernels.

    Args:
        seed: Base seed. Also exported as ``PYTHONHASHSEED`` for child processes.
        strict: If True, ask torch for deterministic algorithms. Some kernels have no
            deterministic implementation on the MPS backend; when that is the case the
            request is downgraded rather than raised, and the reason is recorded in the
            returned :class:`SeedState` so it can be logged with the results.

    Returns:
        The :class:`SeedState` describing what was configured.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if not strict:
        return SeedState(seed=seed, deterministic_algorithms=False, note="strict=False")

    # CUBLAS needs this set before the first matmul for deterministic reductions; harmless
    # on machines without CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover - backend dependent
        return SeedState(
            seed=seed, deterministic_algorithms=False, note=f"{type(exc).__name__}: {exc}"
        )
    return SeedState(seed=seed, deterministic_algorithms=True)


def seed_worker(worker_id: int) -> None:
    """Seed a ``torch.utils.data.DataLoader`` worker process.

    Passed as ``worker_init_fn``. Without this, every worker inherits the same numpy seed
    and augmentation draws repeat across workers.

    Args:
        worker_id: Index of the worker, supplied by torch.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int) -> torch.Generator:
    """Build the generator that fixes ``DataLoader`` shuffling order.

    Args:
        seed: Seed for the generator.

    Returns:
        A CPU :class:`torch.Generator` seeded with ``seed``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Run a block under a fixed seed, then restore the previous RNG states.

    Used for things that must be identical regardless of how much randomness was consumed
    before them -- most importantly the k-shot support-set sampling in the personalization
    sweep, where run ``k=5`` must contain the same first three examples as run ``k=3``.

    Args:
        seed: Seed applied inside the block.

    Yields:
        None.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
