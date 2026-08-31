"""Training-time augmentation of normalized keypoints.

Not listed in PLAN.md section 7's layout; it lives here because it operates on the same
``(T, 115, 4)`` representation as :mod:`signadapt.data.normalize` and is meaningless without
it. ``configs/model.yaml`` already budgeted the ``augment`` block, so this is where that
block is spent.

Every transform obeys one invariant: **augmentation may destroy a landmark, never create
one.** A landmark whose validity flag is 0 carries coordinates ``(0, 0, 0)`` that mean
"absent", and a translation would quietly turn that into a real-looking position at
``(tx, ty, 0)``. Each geometric step is therefore followed by re-applying the validity mask.
"""

from __future__ import annotations

from typing import Any

import torch


def _uniform(
    shape: tuple[int, ...], low: float, high: float, generator: torch.Generator
) -> torch.Tensor:
    """Sample uniformly on the CPU, where the generator lives.

    MPS tensors cannot be filled from a CPU generator, and a per-device generator would make
    a run's augmentation stream depend on the accelerator it happened to run on. The sampled
    tensors are tiny (one value per clip), so drawing on the CPU costs nothing.
    """
    return torch.rand(shape, generator=generator) * (high - low) + low


def augment_batch(
    batch: torch.Tensor,
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the configured augmentations to a batch of normalized clips.

    Args:
        batch: ``(B, T, L, 4)`` normalized keypoints, channels ``[x, y, z, valid]``.
        cfg: The ``augment`` block of ``configs/model.yaml``.
        generator: CPU generator; seeding it is what makes a training run reproducible.

    Returns:
        A new ``(B, T, L, 4)`` tensor. The validity channel is only ever turned off.
    """
    if not cfg.get("enabled", False):
        return batch

    device = batch.device
    n_clips, n_frames, n_landmarks, _ = batch.shape
    out = batch.clone()
    coords, valid = out[..., :3], out[..., 3:]

    rotate_deg = float(cfg.get("rotate_deg", 0.0))
    if rotate_deg > 0:
        # In-plane rotation about the mid-shoulder origin. Only x and y rotate: MediaPipe's z
        # is weakly calibrated and already de-weighted (configs/data.yaml), so mixing it into
        # a 3-D rotation would inject its noise into the two axes that are trustworthy.
        angle = torch.deg2rad(_uniform((n_clips,), -rotate_deg, rotate_deg, generator)).to(device)
        cos, sin = torch.cos(angle).view(-1, 1, 1), torch.sin(angle).view(-1, 1, 1)
        x, y = coords[..., 0].clone(), coords[..., 1].clone()
        coords[..., 0] = cos * x - sin * y
        coords[..., 1] = sin * x + cos * y

    scale_jitter = float(cfg.get("scale_jitter", 0.0))
    if scale_jitter > 0:
        scale = _uniform((n_clips, 1, 1, 1), 1 - scale_jitter, 1 + scale_jitter, generator)
        coords *= scale.to(device)

    translate = float(cfg.get("translate", 0.0))
    if translate > 0:
        shift = _uniform((n_clips, 1, 1, 2), -translate, translate, generator).to(device)
        coords[..., :2] += shift

    dropout_landmarks = float(cfg.get("dropout_landmarks", 0.0))
    if dropout_landmarks > 0:
        # Dropped for the whole clip, not per frame: a landmark that MediaPipe fails on tends
        # to fail for a stretch of frames (a hand leaves the frame, a glove saturates), so
        # per-frame noise would simulate a failure mode the data does not have.
        keep = (
            (torch.rand((n_clips, 1, n_landmarks, 1), generator=generator) >= dropout_landmarks)
            .float()
            .to(device)
        )
        valid *= keep

    time_mask = int(cfg.get("time_mask", 0))
    if time_mask > 0 and n_frames > time_mask:
        span = torch.randint(1, time_mask + 1, (n_clips,), generator=generator)
        start = (torch.rand((n_clips,), generator=generator) * (n_frames - span)).long()
        frames = torch.arange(n_frames).view(1, -1)
        masked = (frames >= start.view(-1, 1)) & (frames < (start + span).view(-1, 1))
        valid *= (~masked).float().view(n_clips, n_frames, 1, 1).to(device)

    # Re-assert the invariant: anything marked invalid, whether it arrived that way or was
    # dropped above, carries zero coordinates.
    out[..., :3] = coords * valid
    return out
