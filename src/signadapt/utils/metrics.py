"""Classification metrics and per-group breakdowns.

PLAN.md section 6 asks for top-1 and top-5 reported as mean +/- std *across signers*, not
only across seeds -- "inter-signer variance is a finding in itself" -- so the per-group
helpers here are part of the result, not a debugging convenience.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Return a boolean vector marking the samples whose true class is in the top ``k``.

    Args:
        logits: ``(B, n_classes)`` scores.
        targets: ``(B,)`` class indices.
        k: How many predictions count as a hit. Clipped to the number of classes, so
            top-5 on a 4-class problem degrades to top-4 rather than raising.

    Returns:
        ``(B,)`` bool tensor.
    """
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    return (top == targets.unsqueeze(1)).any(dim=1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Top-``k`` accuracy over a batch.

    Args:
        logits: ``(B, n_classes)`` scores.
        targets: ``(B,)`` class indices.
        k: Cut-off.

    Returns:
        Accuracy in ``[0, 1]``; ``nan`` for an empty batch, which keeps an empty group from
        silently contributing a 0 to a mean.
    """
    if targets.numel() == 0:
        return float("nan")
    return topk_correct(logits, targets, k).float().mean().item()


def per_group_accuracy(
    correct: torch.Tensor,
    groups: Sequence[object],
) -> dict[str, float]:
    """Break a correctness vector down by an arbitrary grouping key.

    Args:
        correct: ``(B,)`` bool tensor from :func:`topk_correct`.
        groups: One key per sample, e.g. signer ids or participant ids.

    Returns:
        ``{str(key): accuracy}``, sorted by key.

    Raises:
        ValueError: If the lengths disagree -- a misaligned breakdown would attribute one
            signer's errors to another.
    """
    if len(groups) != correct.numel():
        raise ValueError(f"{len(groups)} group keys for {correct.numel()} predictions")
    totals: dict[str, list[int]] = {}
    flags = correct.tolist()
    for key, hit in zip(groups, flags, strict=True):
        bucket = totals.setdefault(str(key), [0, 0])
        bucket[0] += int(hit)
        bucket[1] += 1
    return {key: hits / n for key, (hits, n) in sorted(totals.items())}


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Summarize a set of values as mean, sample std, min, max and count.

    The std is the sample (n-1) standard deviation, which is the right one for "mean +/- std
    across 3 seeds" or "across 10 signers": those are samples, not the population. It is
    ``nan`` for a single value rather than 0, so a one-run summary cannot be mistaken for a
    zero-variance one.

    Args:
        values: The numbers to summarize.

    Returns:
        ``{"mean", "std", "min", "max", "n"}``.
    """
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    mean = sum(clean) / len(clean)
    if len(clean) < 2:
        return {"mean": mean, "std": float("nan"), "min": mean, "max": mean, "n": len(clean)}
    variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def format_mean_std(summary: dict[str, float], *, percent: bool = True) -> str:
    """Render a :func:`mean_std` summary for a console line or a table cell.

    Args:
        summary: Output of :func:`mean_std`.
        percent: Scale by 100 and append ``%``.

    Returns:
        e.g. ``"93.4 +/- 1.2 %"``.
    """
    scale, unit = (100.0, " %") if percent else (1.0, "")
    if summary["n"] == 0:
        return "n/a"
    if math.isnan(summary["std"]):
        return f"{summary['mean'] * scale:.1f}{unit}"
    return f"{summary['mean'] * scale:.1f} +/- {summary['std'] * scale:.1f}{unit}"
