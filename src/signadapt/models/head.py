"""Private classifier head (PLAN.md section 4).

This is the part that never leaves the device in FedPer (PLAN.md section 6, E5). It is kept
in its own module, and its parameters under their own ``head.`` prefix, so that the
aggregation code can address encoder and head parameter groups independently -- see
:mod:`signadapt.models.model` and ``tests/test_fedper.py``.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    """A single linear layer from the clip embedding to class logits.

    Deliberately the simplest thing that works: the thesis question is whether a *shared
    encoder* helps a new signer, so any capacity in the head would confound the answer --
    a strong head could recover accuracy that the encoder did not provide.

    Attributes:
        in_dim: Embedding width this head consumes.
        n_classes: Number of output classes.
    """

    def __init__(self, *, in_dim: int = 128, n_classes: int = 64, dropout: float = 0.0) -> None:
        """Build the head.

        Args:
            in_dim: Width of the encoder embedding.
            n_classes: Size of the vocabulary.
            dropout: Dropout applied to the embedding before the linear layer.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Map embeddings to logits.

        Args:
            embedding: ``(B, in_dim)`` clip embeddings.

        Returns:
            ``(B, n_classes)`` logits.
        """
        return self.linear(self.dropout(embedding))

    def reset_parameters(self) -> None:
        """Re-initialize the head in place.

        Used when a held-out signer starts personalization from a fresh head rather than
        inheriting the training signers' classifier (PLAN.md section 6, E5).
        """
        self.linear.reset_parameters()


def build_head(cfg: dict) -> LinearHead:
    """Construct the head described by the ``head`` block of ``configs/model.yaml``.

    Args:
        cfg: Loaded model config (the whole file, not just the ``head`` block).

    Returns:
        The configured head.

    Raises:
        ValueError: On an unknown ``head.type``. ``prototypical`` is deferred to phase 4,
            where the k-shot sweep is what makes it worth having.
    """
    spec = dict(cfg["head"])
    kind = spec.pop("type", "linear")
    if kind != "linear":
        raise ValueError(f"unknown or not-yet-implemented head.type: {kind!r}")
    return LinearHead(**spec)
