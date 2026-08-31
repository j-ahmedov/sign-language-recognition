"""Converting between a model's parameters and Flower's ``list[np.ndarray]`` payload.

Not listed in PLAN.md section 7's layout; it sits between ``client.py`` and ``strategy.py``
because both ends of a federated round have to agree, exactly, on which tensors travel and
in what order.

The whole FedPer claim in PLAN.md section 6 -- "the private head never leaves the device" --
is a claim about *this* boundary. So the payload is defined by a **name prefix filter over
the model's state dict**, never by an index into a flat parameter list: an index-based split
still type-checks and still runs after the architecture changes, it just silently starts
transmitting the wrong tensors. ``tests/test_fedper.py`` asserts the filter directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn

Prefixes = Sequence[str]


def shared_keys(model: nn.Module, prefixes: Prefixes) -> tuple[str, ...]:
    """List the state-dict keys that a round transmits, in a stable order.

    Args:
        model: The model.
        prefixes: Name prefixes to include, e.g. ``("encoder.",)`` for FedPer or
            ``("encoder.", "head.")`` for FedAvg.

    Returns:
        Matching keys in ``state_dict`` order, which is fixed by module registration order
        and is therefore identical on the server and on every client.

    Raises:
        ValueError: If no key matches. An empty payload would make every round a no-op that
            still reports plausible-looking losses.
    """
    keys = tuple(k for k in model.state_dict() if any(k.startswith(p) for p in prefixes))
    if not keys:
        raise ValueError(f"no parameters match {list(prefixes)}; nothing would be aggregated")
    return keys


def get_parameters(model: nn.Module, prefixes: Prefixes) -> list[np.ndarray]:
    """Extract the transmittable parameters as numpy arrays.

    Args:
        model: The model.
        prefixes: Name prefixes to include.

    Returns:
        One array per key of :func:`shared_keys`, in that order. The arrays are copies:
        ``Tensor.numpy()`` on a CPU tensor shares storage with the model, so without the copy
        a captured payload would keep changing as the model trains -- and a client comparing
        "what I received" against "what I am sending" would find them identical.
    """
    state = model.state_dict()
    return [state[k].detach().cpu().numpy().copy() for k in shared_keys(model, prefixes)]


def set_parameters(model: nn.Module, arrays: Sequence[np.ndarray], prefixes: Prefixes) -> None:
    """Load transmitted parameters back into a model, leaving everything else untouched.

    Args:
        model: The model to update in place.
        arrays: Arrays in :func:`shared_keys` order.
        prefixes: The same prefixes the arrays were produced with.

    Raises:
        ValueError: If the count or a shape disagrees with the model. A mismatch here means
            client and server disagree about the payload, and loading it anyway would train
            on transposed or misrouted tensors.
    """
    keys = shared_keys(model, prefixes)
    if len(arrays) != len(keys):
        raise ValueError(f"payload has {len(arrays)} arrays, model expects {len(keys)}")

    state = model.state_dict()
    update = {}
    for key, array in zip(keys, arrays, strict=True):
        tensor = torch.as_tensor(array, dtype=state[key].dtype)
        if tuple(tensor.shape) != tuple(state[key].shape):
            raise ValueError(
                f"{key}: payload shape {tuple(tensor.shape)} != model {tuple(state[key].shape)}"
            )
        update[key] = tensor
    model.load_state_dict(update, strict=False)


def assert_excludes(keys: Iterable[str], forbidden: Prefixes) -> None:
    """Raise if any key belongs to a group that must never be transmitted.

    This is the runtime counterpart of ``tests/test_fedper.py``: the test proves the filter
    is right for the configured model, and this call keeps a mis-specified config from
    turning FedPer into FedAvg without anyone noticing.

    Args:
        keys: The keys about to be transmitted.
        forbidden: Prefixes that must not appear.

    Raises:
        ValueError: If a forbidden key is present.
    """
    leaked = [k for k in keys if any(k.startswith(p) for p in forbidden)]
    if leaked:
        raise ValueError(f"private parameters would be transmitted: {leaked}")
