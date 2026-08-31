"""Anchoring, scaling, handedness mirroring, missing-hand masking, temporal resampling.

Input is always the raw cache layout produced by :mod:`signadapt.data.keypoints`:
``(T, 115, 4)`` float32, channels ``[x, y, z, valid]``, with missing landmarks stored as
NaN and ``valid == 0``.

The pipeline, in order (PLAN.md section 4):

1. **Aspect correction.** MediaPipe returns ``x`` normalized by frame *width* and ``y`` by
   frame *height*. On LSA64's 16:9 frames one x-unit is 1.78 y-units of real distance, so
   any distance computed before correcting this -- shoulder width above all -- is wrong.
   ``x`` is multiplied by ``width / height`` to make the units isotropic.
2. **Anchor** on the mid-shoulder point, per frame.
3. **Scale** by shoulder width, so a signer who sits closer to the camera looks the same as
   one who sits further away. Removing this cue matters here specifically: camera distance
   is a per-signer constant, i.e. exactly the kind of nuisance correlation that would let
   the model identify the signer instead of the sign.
4. **De-weight z**, which MediaPipe estimates far less reliably than x and y.
5. **Missing data** stays explicit. A missing hand keeps ``valid == 0``; it is never
   silently written as 0, because after anchoring, 0 *is* the mid-shoulder point -- a
   perfectly plausible hand position. See :func:`fill_missing`.
6. **Optional mirroring**, which flips x *and* swaps the left/right landmark identities.
7. **Temporal resampling** to a fixed ``T`` by linear interpolation over frame index.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from signadapt.data.keypoints import (
    FACE_MIRROR_PAIRS,
    LEFT_SHOULDER,
    N_LANDMARKS,
    POSE_MIRROR_PAIRS,
    RIGHT_SHOULDER,
    SLICES,
)

__all__ = [
    "MissingHandPolicy",
    "correct_aspect",
    "fill_missing",
    "group_presence",
    "interpolate_missing",
    "mirror",
    "normalize_clip",
    "normalize_geometry",
    "resample_time",
]

#: Accepted values of ``normalization.missing_hand`` in configs/data.yaml.
MissingHandPolicy = str
_POLICIES = ("mask", "interpolate", "error")

_EPS = 1e-6


def _check_layout(clip: np.ndarray) -> None:
    """Validate the array layout, raising with a useful message when it is wrong."""
    if clip.ndim != 3 or clip.shape[1] != N_LANDMARKS or clip.shape[2] != 4:
        raise ValueError(
            f"expected (T, {N_LANDMARKS}, 4) [x, y, z, valid], got {clip.shape}. "
            "This function consumes the raw cache written by signadapt.data.keypoints."
        )


def correct_aspect(clip: np.ndarray, aspect: float) -> np.ndarray:
    """Make x and y comparable by rescaling x with the frame aspect ratio.

    Args:
        clip: ``(T, 115, 4)`` array in MediaPipe's per-axis normalized coordinates.
        aspect: ``frame_width / frame_height`` of the source video.

    Returns:
        A new array with ``x`` (and ``z``, which shares x's units) multiplied by ``aspect``.
    """
    _check_layout(clip)
    out = clip.copy()
    out[..., 0] *= aspect
    out[..., 2] *= aspect
    return out


def group_presence(clip: np.ndarray) -> dict[str, np.ndarray]:
    """Report, per frame, which landmark groups were detected.

    Args:
        clip: ``(T, 115, 4)`` array.

    Returns:
        A mapping from group name to a boolean ``(T,)`` array that is True when *every*
        landmark of the group is valid in that frame. MediaPipe emits a hand as all-or-
        nothing, so this is exact for hands rather than a heuristic.
    """
    _check_layout(clip)
    return {name: clip[:, rows, 3].min(axis=1) > 0.5 for name, rows in SLICES.items()}


def interpolate_missing(clip: np.ndarray, group: str) -> np.ndarray:
    """Fill frames where a group vanished by interpolating between frames where it exists.

    Used for the ``interpolate`` missing-hand policy: a hand that MediaPipe drops for two or
    three frames in the middle of a sign is a detector failure, not an absent hand, and
    interpolating across it is more faithful than masking it out. Frames before the first
    and after the last detection are left missing -- extrapolating a hand that was never
    seen would be inventing data.

    Args:
        clip: ``(T, 115, 4)`` array; modified copy is returned.
        group: Key of :data:`signadapt.data.keypoints.SLICES`.

    Returns:
        A new array with interior gaps of that group filled and ``valid`` set to 1 there.
    """
    _check_layout(clip)
    out = clip.copy()
    rows = SLICES[group]
    present = group_presence(clip)[group]
    times = np.flatnonzero(present)
    if times.size == 0 or times.size == clip.shape[0]:
        return out

    interior = np.arange(times[0], times[-1] + 1)
    gaps = interior[~present[interior]]
    if gaps.size == 0:
        return out

    for axis in range(3):
        known = out[times, rows, axis]  # (n_known, n_group)
        out[gaps, rows, axis] = np.stack(
            [np.interp(gaps, times, known[:, j]) for j in range(known.shape[1])], axis=1
        )
    out[gaps, rows, 3] = 1.0
    return out


def normalize_geometry(
    clip: np.ndarray,
    *,
    aspect: float,
    z_weight: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-correct, anchor on mid-shoulder, scale by shoulder width and de-weight z.

    The anchor and scale are computed per frame from the pose landmarks. Frames where the
    shoulders were not detected borrow the nearest frame's anchor and scale by linear
    interpolation, because dropping them would silently shorten the clip.

    Args:
        clip: ``(T, 115, 4)`` raw cache array.
        aspect: ``frame_width / frame_height``.
        z_weight: Multiplier applied to z; MediaPipe's depth is weakly calibrated.

    Returns:
        A tuple ``(normalized, anchorable)`` where ``normalized`` is ``(T, 115, 4)`` and
        ``anchorable`` is a ``(T,)`` boolean array that is True for frames whose shoulders
        were directly observed.

    Raises:
        ValueError: If no frame in the clip has both shoulders, so no anchor exists at all.
    """
    out = correct_aspect(clip, aspect)
    shoulders = out[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :]
    anchorable = shoulders[:, :, 3].min(axis=1) > 0.5
    if not anchorable.any():
        raise ValueError("no frame has both shoulders detected; clip cannot be anchored")

    centre = shoulders[:, :, :3].mean(axis=1)  # (T, 3)
    delta = shoulders[:, 0, :3] - shoulders[:, 1, :3]
    width = np.linalg.norm(delta[:, :2], axis=1)  # xy only: z is not trustworthy

    # Carry the anchor across frames where the shoulders were lost.
    times = np.flatnonzero(anchorable)
    if times.size < out.shape[0]:
        missing = np.flatnonzero(~anchorable)
        for axis in range(3):
            centre[missing, axis] = np.interp(missing, times, centre[times, axis])
        width[missing] = np.interp(missing, times, width[times])

    scale = np.maximum(width, _EPS)[:, None, None]
    out[..., :3] = (out[..., :3] - centre[:, None, :]) / scale
    out[..., 2] *= z_weight
    return out, anchorable


def mirror(clip: np.ndarray) -> np.ndarray:
    """Mirror a clip left/right, swapping landmark identities as well as flipping x.

    Negating x alone would leave a mirrored left hand sitting in the *left*-hand rows, which
    is not what a right-handed signer's data looks like -- the swap is the point of the
    operation, the sign flip is the easy half.

    Args:
        clip: ``(T, 115, 4)`` array, normalized or raw.

    Returns:
        A new mirrored array.
    """
    _check_layout(clip)
    out = clip.copy()
    out[..., 0] *= -1.0

    left, right = SLICES["left_hand"], SLICES["right_hand"]
    out[:, left, :], out[:, right, :] = clip[:, right, :].copy(), clip[:, left, :].copy()
    out[:, left, 0] *= -1.0
    out[:, right, 0] *= -1.0

    pose = SLICES["pose"].start
    for a, b in POSE_MIRROR_PAIRS:
        out[:, pose + a, :], out[:, pose + b, :] = (
            out[:, pose + b, :].copy(),
            out[:, pose + a, :].copy(),
        )
    face = SLICES["face"].start
    for a, b in FACE_MIRROR_PAIRS:
        out[:, face + a, :], out[:, face + b, :] = (
            out[:, face + b, :].copy(),
            out[:, face + a, :].copy(),
        )
    return out


def resample_time(clip: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample a clip to a fixed number of frames by linear interpolation.

    Interpolation is validity-aware: an output frame blends its two neighbours only where
    both are valid, falls back to whichever neighbour is valid, and stays missing when
    neither is. A blended frame is marked valid, which is a deliberate and documented
    approximation -- with T=64 out of the 90-190 source frames of an LSA64 clip the two
    neighbours are adjacent frames, ~16 ms apart.

    Args:
        clip: ``(T_in, 115, 4)`` array.
        n_frames: Output length T.

    Returns:
        A ``(n_frames, 115, 4)`` float32 array.

    Raises:
        ValueError: If the clip has no frames.
    """
    _check_layout(clip)
    t_in = clip.shape[0]
    if t_in == 0:
        raise ValueError("cannot resample an empty clip")
    if t_in == 1:
        return np.repeat(clip, n_frames, axis=0).astype(np.float32)

    positions = np.linspace(0.0, t_in - 1, n_frames)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, t_in - 1)
    weight = (positions - lower)[:, None, None]

    low, high = clip[lower], clip[upper]
    low_ok = low[..., 3:4] > 0.5
    high_ok = high[..., 3:4] > 0.5

    blended = np.where(
        low_ok & high_ok,
        low * (1.0 - weight) + high * weight,
        np.where(low_ok, low, np.where(high_ok, high, np.nan)),
    )
    blended[..., 3] = (low_ok | high_ok)[..., 0].astype(np.float32)
    return blended.astype(np.float32)


def fill_missing(clip: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Replace NaN coordinates with a constant, keeping the validity channel intact.

    This is the last step before the tensor reaches the model, and it is only safe because
    channel 3 still says which entries were substituted: the model sees ``(0, 0, 0, 0)`` and
    can tell it apart from a real landmark at ``(0, 0, 0, 1)``. Call it late, never early.

    Args:
        clip: ``(T, 115, 4)`` array.
        value: Substitute for missing coordinates.

    Returns:
        A new array with no NaNs in channels 0..2.
    """
    _check_layout(clip)
    out = clip.copy()
    coords = out[..., :3]
    out[..., :3] = np.where(np.isnan(coords), value, coords)
    return out


def normalize_clip(
    clip: np.ndarray,
    *,
    aspect: float,
    n_frames: int = 64,
    z_weight: float = 0.25,
    missing_hand: MissingHandPolicy = "mask",
    do_mirror: bool = False,
    fill: bool = True,
) -> np.ndarray:
    """Run the full normalization pipeline on one raw clip.

    Args:
        clip: ``(T_in, 115, 4)`` raw cache array.
        aspect: ``frame_width / frame_height`` of the source video.
        n_frames: Output length T.
        z_weight: Multiplier applied to z.
        missing_hand: ``"mask"`` keeps gaps explicit, ``"interpolate"`` fills interior gaps
            first, ``"error"`` raises when a hand is missing anywhere.
        do_mirror: Apply handedness mirroring.
        fill: Replace remaining NaNs with 0 (see :func:`fill_missing`).

    Returns:
        A ``(n_frames, 115, 4)`` float32 array.

    Raises:
        ValueError: On an unknown policy, or under ``"error"`` when a hand is missing.
    """
    if missing_hand not in _POLICIES:
        raise ValueError(f"missing_hand must be one of {_POLICIES}, got {missing_hand!r}")

    out = clip
    if missing_hand == "interpolate":
        for group in ("left_hand", "right_hand"):
            out = interpolate_missing(out, group)
    elif missing_hand == "error":
        presence = group_presence(out)
        for group in ("left_hand", "right_hand"):
            if not presence[group].all():
                missing = int((~presence[group]).sum())
                raise ValueError(f"{group} missing in {missing}/{out.shape[0]} frames")

    out, _ = normalize_geometry(out, aspect=aspect, z_weight=z_weight)
    if do_mirror:
        out = mirror(out)
    out = resample_time(out, n_frames)
    return fill_missing(out) if fill else out


def config_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate ``configs/data.yaml`` into :func:`normalize_clip` keyword arguments.

    Args:
        cfg: Loaded data config.

    Returns:
        Keyword arguments for :func:`normalize_clip`, excluding ``aspect`` and ``do_mirror``
        which are per-clip decisions.
    """
    norm = cfg.get("normalization", {})
    return {
        "n_frames": cfg.get("temporal", {}).get("n_frames", 64),
        "z_weight": norm.get("z_weight", 0.25),
        "missing_hand": norm.get("missing_hand", "mask"),
    }
