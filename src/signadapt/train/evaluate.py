"""Evaluation: top-1 / top-5 and the per-signer breakdowns PLAN.md section 6 asks for."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from signadapt.data.dataset import ClipRecord
from signadapt.utils.metrics import mean_std, per_group_accuracy, topk_correct


@dataclass(frozen=True)
class EvalResult:
    """Everything one evaluation pass produced.

    Attributes:
        top1: Overall top-1 accuracy.
        top5: Overall top-5 accuracy.
        loss: Mean cross-entropy over the set.
        n: Number of clips evaluated.
        per_signer: Top-1 accuracy keyed by signer id.
        per_participant: Top-1 accuracy keyed by participant id, which separates LSA64's
            signer 10 into ``"10a"`` and ``"10b"`` (README, phase-1 caveat 2).
        per_handedness: Top-1 accuracy for one- and two-handed signs, which is where the
            glove-induced detection loss concentrates (README, phase-1 caveat 1).
    """

    top1: float
    top5: float
    loss: float
    n: int
    per_signer: dict[str, float] = field(default_factory=dict)
    per_participant: dict[str, float] = field(default_factory=dict)
    per_handedness: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view, with the across-signer spread precomputed.

        Returns:
            A dict suitable for :meth:`signadapt.utils.results.ResultsLogger.set_metrics`.
        """
        return {
            "top1": self.top1,
            "top5": self.top5,
            "loss": self.loss,
            "n": self.n,
            "per_signer": self.per_signer,
            "per_participant": self.per_participant,
            "per_handedness": self.per_handedness,
            "across_signers": mean_std(list(self.per_signer.values())),
        }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader and collect logits and targets.

    Args:
        model: The model; set to eval mode and left that way.
        loader: Yields ``(x, y)``. Must not shuffle, or the returned order will not line up
            with the record order used for the per-signer breakdown.
        device: Where to run.

    Returns:
        ``(logits, targets)`` on the CPU.
    """
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        targets.append(y.cpu())
    if not logits:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(logits), torch.cat(targets)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    records: Sequence[ClipRecord] | None = None,
    indices: Sequence[int] | None = None,
) -> EvalResult:
    """Evaluate a model and, when given the records, break the result down by signer.

    Args:
        model: The model.
        loader: An unshuffled loader over the evaluation set.
        device: Where to run.
        records: All clip records, for the breakdowns. Omit to get overall numbers only.
        indices: The record indices the loader serves, **in loader order**.

    Returns:
        The populated :class:`EvalResult`.

    Raises:
        ValueError: If ``indices`` does not line up with the number of predictions. That
            misalignment would attribute one signer's errors to another, so it must not pass
            silently.
    """
    from signadapt.data.dataset import is_two_handed  # local: keeps the import graph shallow

    logits, targets = predict(model, loader, device)
    if targets.numel() == 0:
        return EvalResult(top1=float("nan"), top5=float("nan"), loss=float("nan"), n=0)

    correct1 = topk_correct(logits, targets, 1)
    result = EvalResult(
        top1=correct1.float().mean().item(),
        top5=topk_correct(logits, targets, 5).float().mean().item(),
        loss=nn.functional.cross_entropy(logits, targets).item(),
        n=int(targets.numel()),
    )
    if records is None or indices is None:
        return result

    if len(indices) != correct1.numel():
        raise ValueError(f"{len(indices)} indices for {correct1.numel()} predictions")
    chosen = [records[i] for i in indices]
    return EvalResult(
        top1=result.top1,
        top5=result.top5,
        loss=result.loss,
        n=result.n,
        per_signer=per_group_accuracy(correct1, [r.signer for r in chosen]),
        per_participant=per_group_accuracy(correct1, [r.participant for r in chosen]),
        per_handedness=per_group_accuracy(
            correct1, ["two_handed" if is_two_handed(r.label) else "one_handed" for r in chosen]
        ),
    )
