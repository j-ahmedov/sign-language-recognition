"""The training loop shared by every experiment in PLAN.md section 6.

Not listed in PLAN.md section 7's layout. It is factored out because E1/E2/E6 (centralized),
E3 (local-only) and the per-client update inside the Flower simulation are the *same* loop
with different data and different frozen parameter groups. Duplicating it three times is how
E3 and E5 end up incomparable for a reason nobody notices -- a different learning-rate
schedule rather than a different method.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from signadapt.data.augment import augment_batch
from signadapt.data.dataset import ClipRecord, KeypointDataset
from signadapt.train.evaluate import EvalResult, evaluate
from signadapt.utils.seeding import torch_generator


def resolve_device(name: str = "auto") -> torch.device:
    """Pick the torch device.

    Args:
        name: ``"auto"``, or any explicit torch device string.

    Returns:
        ``mps`` on this M4 when available, else ``cuda``, else ``cpu``.
    """
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def stack_dataset(dataset: KeypointDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a :class:`~signadapt.data.dataset.KeypointDataset` into two tensors.

    Normalization is deterministic and the whole of LSA64 is 377 MB at T=64, so it is done
    once up front rather than per epoch. Everything downstream then works on tensors, which
    also removes the dataloader-worker cost on a machine with no spare cores.

    Args:
        dataset: The view to materialize.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(N, T, L, 4)`` and ``y`` of shape ``(N,)``. Row
        ``i`` corresponds to ``dataset.indices[i]``.
    """
    if len(dataset) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    xs, ys = zip(*(dataset[i] for i in range(len(dataset))), strict=True)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap two tensors in a seeded :class:`~torch.utils.data.DataLoader`.

    Args:
        x: Inputs.
        y: Targets.
        batch_size: Batch size.
        shuffle: Shuffle each epoch. Evaluation loaders must pass ``False`` so that the
            prediction order still matches the record order the breakdowns rely on.
        seed: Seed for the shuffle generator.
        drop_last: Drop a final short batch.

    Returns:
        The loader. ``num_workers`` is 0 by design -- the data is already in RAM.
    """
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=torch_generator(seed) if shuffle else None,
    )


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split trainable parameters into decayed and non-decayed groups.

    Biases, LayerNorm gains and the positional embedding are excluded from weight decay:
    decaying them pulls a learned position or a normalization scale toward zero, which is a
    different thing from regularizing a weight matrix.

    Args:
        model: The model.
        weight_decay: Decay applied to the matrix group.

    Returns:
        Optimizer parameter groups; parameters with ``requires_grad=False`` are omitted so
        that a frozen encoder contributes no optimizer state.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim < 2 or name.endswith("pos_embedding") else decay).append(
            parameter
        )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_schedule(spec: dict[str, Any], epochs: int) -> Callable[[int], float]:
    """Pick the learning-rate schedule named in a ``train`` config block.

    ``constant`` exists for federated clients. A client runs only ``local_epochs`` epochs per
    round, so a within-round cosine would decay the learning rate to near zero inside every
    round and then reset it -- the schedule would describe the client's two epochs rather
    than the run's fifty rounds, and the clients would never train at the configured rate at
    all. Any decay across a federated run belongs to the server, not to the local step.

    Args:
        spec: A ``train`` block; reads ``scheduler`` and ``warmup_epochs``.
        epochs: Total epochs this schedule covers.

    Returns:
        A function from epoch index to learning-rate multiplier.

    Raises:
        ValueError: On an unknown scheduler name.
    """
    name = str(spec.get("scheduler", "cosine"))
    if name == "constant":
        return lambda epoch: 1.0
    if name == "cosine":
        return cosine_schedule(epochs, int(spec.get("warmup_epochs", 0)))
    raise ValueError(f"unknown train.scheduler: {name!r}")


def cosine_schedule(epochs: int, warmup_epochs: int) -> Callable[[int], float]:
    """Build the learning-rate multiplier: linear warmup, then cosine decay to zero.

    Args:
        epochs: Total epochs.
        warmup_epochs: Epochs of linear warmup, clipped to ``epochs``.

    Returns:
        A function from epoch index to multiplier.
    """
    warmup = min(max(warmup_epochs, 0), epochs)

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        if epochs == warmup:
            return 1.0
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())

    return factor


@dataclass
class TrainOutcome:
    """What a training run produced.

    Attributes:
        history: One dict per epoch, suitable for ``ResultsLogger.log_record``.
        best_epoch: Epoch with the highest validation top-1, or the last epoch when there
            was no validation set.
        best_val: The validation top-1 at ``best_epoch``; ``nan`` without a validation set.
        best_state: A CPU copy of the model state dict at ``best_epoch``.
        epochs_run: How many epochs actually ran, which is fewer than requested when early
            stopping fired.
        seconds: Wall-clock training time.
    """

    history: list[dict[str, Any]]
    best_epoch: int
    best_val: float
    best_state: dict[str, torch.Tensor]
    epochs_run: int
    seconds: float


def train_model(
    model: nn.Module,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor] | None,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    seed: int = 0,
    val_records: Sequence[ClipRecord] | None = None,
    val_indices: Sequence[int] | None = None,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    epochs: int | None = None,
    augment: bool | None = None,
) -> TrainOutcome:
    """Train a model and return its history and best checkpoint.

    Args:
        model: Model to train in place; the returned ``best_state`` is a separate copy.
        train_data: ``(X, y)`` from :func:`stack_dataset`.
        val_data: ``(X, y)`` for validation, or ``None`` to train for a fixed number of
            epochs with no early stopping.
        cfg: Loaded ``configs/model.yaml``; the ``train`` and ``augment`` blocks are used.
        device: Where to train.
        seed: Seed for shuffling and augmentation.
        val_records: All records, for the per-signer breakdown of the final evaluation.
        val_indices: Record indices the validation set serves, in order.
        on_epoch: Called with each epoch's record, e.g. ``ResultsLogger.log_record``.
        epochs: Override ``train.epochs``.
        augment: Override ``augment.enabled``.

    Returns:
        The :class:`TrainOutcome`.
    """
    spec = cfg["train"]
    aug_cfg = dict(cfg.get("augment", {}))
    if augment is not None:
        aug_cfg["enabled"] = augment
    n_epochs = int(epochs if epochs is not None else spec["epochs"])
    patience = int(spec.get("early_stopping_patience", 0))

    x_train, y_train = train_data
    train_loader = make_loader(
        x_train,
        y_train,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        seed=seed,
        # A batch of one would make the model's own statistics degenerate; with k=1 support
        # sets in phase 4 that is a real possibility, so drop it only when it is not the
        # whole epoch.
        drop_last=len(y_train) > int(spec["batch_size"]),
    )
    val_loader = (
        make_loader(*val_data, batch_size=int(spec["batch_size"]), shuffle=False)
        if val_data is not None and val_data[1].numel() > 0
        else None
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        param_groups(model, float(spec.get("weight_decay", 0.0))), lr=float(spec["lr"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_schedule(spec, n_epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(spec.get("label_smoothing", 0.0)))
    grad_clip = float(spec.get("grad_clip", 0.0))
    generator = torch_generator(seed + 1)

    history: list[dict[str, Any]] = []
    best_val, best_epoch, best_state = float("-inf"), -1, copy.deepcopy(model.state_dict())
    since_improved = 0
    started = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for x, y in train_loader:
            x = augment_batch(x, aug_cfg, generator).to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_seen += y.numel()
        scheduler.step()

        record: dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(1, total_seen),
            "train_top1": total_correct / max(1, total_seen),
        }
        if val_loader is not None:
            val = evaluate(model, val_loader, device)
            record |= {"val_loss": val.loss, "val_top1": val.top1, "val_top5": val.top5}
            if val.top1 > best_val:
                best_val, best_epoch, since_improved = val.top1, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since_improved += 1
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(record)
        if on_epoch is not None:
            on_epoch(record)
        if val_loader is not None and patience > 0 and since_improved >= patience:
            break

    return TrainOutcome(
        history=history,
        best_epoch=best_epoch,
        best_val=best_val if val_loader is not None else float("nan"),
        best_state=best_state,
        epochs_run=len(history),
        seconds=time.time() - started,
    )


def evaluate_tensors(
    model: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    records: Sequence[ClipRecord] | None = None,
    indices: Sequence[int] | None = None,
) -> EvalResult:
    """Evaluate a model on stacked tensors, with breakdowns when records are supplied.

    Args:
        model: The model.
        data: ``(X, y)`` from :func:`stack_dataset`.
        cfg: Loaded model config, for the batch size.
        device: Where to run.
        records: All records.
        indices: Record indices this set serves, in order.

    Returns:
        The :class:`~signadapt.train.evaluate.EvalResult`.
    """
    # Move the model here rather than relying on the caller: an evaluation-only path (E3 at
    # k=0 evaluates an untrained model without ever calling train_model) would otherwise hit
    # a device mismatch that has nothing to do with the experiment.
    model.to(device)
    loader = make_loader(*data, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    return evaluate(model, loader, device, records=records, indices=indices)
