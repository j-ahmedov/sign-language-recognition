"""Encoder + head assembly with independently addressable parameter groups.

The whole federated design in PLAN.md section 6 rests on being able to say "these tensors
are shared, those are private" without ambiguity. That is enforced structurally here: the
model has exactly two submodules, named ``encoder`` and ``head``, so every parameter name
in :meth:`SignAdaptModel.state_dict` starts with one of the two prefixes below and the
partition is total. ``tests/test_fedper.py`` asserts it, and the FedPer strategy in phase 4
aggregates by prefix rather than by an index into a flat parameter list -- an index-based
split silently mis-partitions the moment the architecture changes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from signadapt.models.encoder import TemporalTransformerEncoder, build_encoder
from signadapt.models.head import LinearHead, build_head

ENCODER_PREFIX = "encoder."
HEAD_PREFIX = "head."


class SignAdaptModel(nn.Module):
    """A shared encoder followed by a private classifier head.

    Attributes:
        encoder: The federated part.
        head: The part that never leaves the device.
    """

    def __init__(self, encoder: TemporalTransformerEncoder, head: LinearHead) -> None:
        """Assemble the model.

        Args:
            encoder: Clip encoder.
            head: Classifier head.

        Raises:
            ValueError: If the head does not consume what the encoder produces.
        """
        super().__init__()
        if encoder.out_dim != head.in_dim:
            raise ValueError(
                f"encoder emits {encoder.out_dim}-d embeddings but head expects {head.in_dim}"
            )
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of clips.

        Args:
            x: ``(B, T, L, C)`` normalized keypoints.

        Returns:
            ``(B, n_classes)`` logits.
        """
        return self.head(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return clip embeddings without classifying them.

        Args:
            x: ``(B, T, L, C)`` normalized keypoints.

        Returns:
            ``(B, d_model)`` embeddings.
        """
        return self.encoder(x)

    # ------------------------------------------------------------------ parameter groups

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters that federated averaging is allowed to touch."""
        return self.encoder.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the parameters that must stay on the client."""
        return self.head.parameters()

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the shared parameters, keyed by their full ``encoder.``-prefixed names.

        Full names are kept rather than stripped so that a payload can be re-loaded with
        ``load_state_dict(..., strict=False)`` and so a mis-routed head tensor is visible
        as a name, not as a silent shape match.

        Returns:
            The encoder's entries of the model state dict.
        """
        return group_state_dict(self, ENCODER_PREFIX)

    def head_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the private parameters, keyed by their full ``head.``-prefixed names."""
        return group_state_dict(self, HEAD_PREFIX)

    def load_encoder_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load shared parameters, leaving the head untouched.

        Args:
            state: A mapping as returned by :meth:`encoder_state_dict`.

        Raises:
            ValueError: If ``state`` contains anything that is not an encoder parameter.
                Refusing is the point -- quietly ignoring a stray ``head.`` key is exactly
                how a private parameter would end up being overwritten by the server.
        """
        stray = [k for k in state if not k.startswith(ENCODER_PREFIX)]
        if stray:
            raise ValueError(f"not encoder parameters: {stray}")
        self.load_state_dict(state, strict=False)

    def n_parameters(self) -> dict[str, int]:
        """Count trainable parameters per group.

        Returns:
            ``{"encoder": ..., "head": ..., "total": ...}``.
        """
        counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            "head": sum(p.numel() for p in self.head.parameters() if p.requires_grad),
        }
        counts["total"] = counts["encoder"] + counts["head"]
        return counts

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Enable or disable gradients for the shared parameters.

        Args:
            frozen: ``True`` trains the head alone -- the E5 personalization setting.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)


def group_state_dict(model: nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    """Select the entries of ``model.state_dict()`` whose key starts with ``prefix``.

    Args:
        model: Any module.
        prefix: Key prefix, e.g. ``"encoder."``.

    Returns:
        A detached, cloned sub-state-dict; cloning keeps a captured payload from mutating
        when training continues.
    """
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(prefix)}


def build_model(cfg: dict[str, Any]) -> SignAdaptModel:
    """Construct the model described by ``configs/model.yaml``.

    Args:
        cfg: Loaded model config.

    Returns:
        The assembled model.
    """
    return SignAdaptModel(build_encoder(cfg), build_head(cfg))
