"""Shared temporal transformer encoder (PLAN.md section 4).

This is the part of the model that is aggregated across clients in the federated setting
(PLAN.md section 6, E4/E5). It maps a fixed-length normalized keypoint clip to a single
``d_model``-dimensional embedding and knows nothing about classes, so the same encoder
serves every client regardless of the vocabulary its private head was trained on.

Input layout is ``(B, T, L, C)`` with ``C = 4`` -- ``[x, y, z, valid]``. The validity
channel is a real input feature, not bookkeeping: after mid-shoulder anchoring, the origin
is a perfectly plausible hand position, so ``(0, 0, 0)`` alone cannot express "no hand
detected here". On LSA64 that distinction covers a lot of frames -- both hands are present
in only 51.2 % of the frames of two-handed signs (see ``results/phase1-extraction_*.json``).
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalTransformerEncoder(nn.Module):
    """Flatten landmarks per frame, project to ``d_model``, then self-attend over time.

    The architecture is deliberately small (PLAN.md section 4 budgets 0.5-2 M parameters):
    at ``d_model=128`` with 4 layers this is about 0.6 M, which keeps a federated round's
    communication cost a footnote and leaves headroom on a 16 GB machine.

    Attributes:
        d_model: Width of the embedding this encoder produces.
        max_len: Longest clip the positional embedding covers.
        pooling: How frame embeddings are reduced to a clip embedding.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
        pooling: str = "mean",
    ) -> None:
        """Build the encoder.

        Args:
            input_dim: Features per frame, i.e. ``n_landmarks * n_channels``.
            d_model: Embedding width.
            n_layers: Number of transformer encoder layers.
            n_heads: Attention heads per layer.
            ff_dim: Width of the position-wise feed-forward block.
            dropout: Dropout used in attention, feed-forward and after the input projection.
            max_len: Maximum sequence length, i.e. T.
            pooling: ``"mean"`` averages frame embeddings, ``"cls"`` prepends a learned token.

        Raises:
            ValueError: On an unknown pooling mode.
        """
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

        self.d_model = d_model
        self.max_len = max_len
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Learned rather than sinusoidal: clips are resampled to exactly T frames, so
        # position here means "fraction of the way through the sign", a quantity with only
        # `max_len` possible values that the model may as well learn directly.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len + int(pooling == "cls"), d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # norm_first: pre-norm blocks train stably without a learning-rate warmup babysitter,
        # which matters because the same architecture is later trained on a single signer's
        # k=1 support set, where any instability would be indistinguishable from a result.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is a no-op with norm_first=True and only emits a warning;
        # every clip is exactly T frames anyway, so there is no padding to skip.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

    @property
    def out_dim(self) -> int:
        """Return the width of the embedding, i.e. the head's expected input size."""
        return self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of clips.

        Args:
            x: ``(B, T, L, C)`` or ``(B, T, L * C)`` normalized keypoints.

        Returns:
            ``(B, d_model)`` clip embeddings.

        Raises:
            ValueError: If T exceeds ``max_len``.
        """
        if x.dim() == 4:
            x = x.flatten(2)
        n_frames = x.shape[1]
        if n_frames > self.max_len:
            raise ValueError(f"clip has {n_frames} frames, max_len is {self.max_len}")

        h = self.input_dropout(self.input_norm(self.input_proj(x)))
        if self.cls_token is not None:
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embedding[:, : h.shape[1]]

        h = self.transformer(h)
        pooled = h[:, 0] if self.pooling == "cls" else h.mean(dim=1)
        return self.out_norm(pooled)


def build_encoder(cfg: dict) -> TemporalTransformerEncoder:
    """Construct the encoder described by the ``encoder`` block of ``configs/model.yaml``.

    Args:
        cfg: Loaded model config (the whole file, not just the ``encoder`` block).

    Returns:
        The configured encoder.

    Raises:
        ValueError: On an unknown ``encoder.type``.
    """
    spec = dict(cfg["encoder"])
    kind = spec.pop("type", "temporal_transformer")
    if kind != "temporal_transformer":
        raise ValueError(f"unknown encoder.type: {kind!r}")
    return TemporalTransformerEncoder(**spec)
