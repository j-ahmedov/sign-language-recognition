"""Normalization tests: anchoring, scaling, mirroring, explicit missing-hand handling.

The behaviour these tests pin down is what makes the signer-independent comparison fair: if
normalization leaked camera distance or frame aspect into the features, the encoder could
identify the signer without recognising the sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from signadapt.data.keypoints import (
    FACE_MIRROR_PAIRS,
    FACE_SUBSET,
    LEFT_SHOULDER,
    N_FACE,
    N_LANDMARKS,
    POSE_MIRROR_PAIRS,
    RIGHT_SHOULDER,
    SLICES,
)
from signadapt.data.normalize import (
    correct_aspect,
    fill_missing,
    group_presence,
    interpolate_missing,
    mirror,
    normalize_clip,
    normalize_geometry,
    resample_time,
)

T = 12


def make_clip(t=T, *, seed=0, shoulder_half_width=0.1, centre=(0.5, 0.4, 0.0)):
    """A synthetic clip with everything detected and shoulders at a known place."""
    rng = np.random.default_rng(seed)
    clip = rng.uniform(0.2, 0.8, size=(t, N_LANDMARKS, 4)).astype(np.float32)
    clip[..., 3] = 1.0
    clip[:, RIGHT_SHOULDER, :3] = np.array(centre) - np.array([shoulder_half_width, 0, 0])
    clip[:, LEFT_SHOULDER, :3] = np.array(centre) + np.array([shoulder_half_width, 0, 0])
    return clip


def drop_group(clip, group, frames=None):
    """Mark a landmark group as undetected, the way keypoints.py stores it."""
    out = clip.copy()
    rows = SLICES[group]
    sel = slice(None) if frames is None else frames
    out[sel, rows, :3] = np.nan
    out[sel, rows, 3] = 0.0
    return out


# ------------------------------------------------------------------------ face subset


def test_face_subset_is_forty_unique_valid_indices():
    assert len(FACE_SUBSET) == N_FACE == 40
    assert len(set(FACE_SUBSET)) == 40
    assert all(0 <= i < 468 for i in FACE_SUBSET)


def test_landmark_layout_is_contiguous_and_complete():
    covered = []
    for rows in SLICES.values():
        covered.extend(range(rows.start, rows.stop))
    assert sorted(covered) == list(range(N_LANDMARKS)) == list(range(115))


def test_mirror_pairs_are_involutive_and_within_range():
    for pairs, size in ((POSE_MIRROR_PAIRS, 33), (FACE_MIRROR_PAIRS, N_FACE)):
        flat = [i for pair in pairs for i in pair]
        assert len(flat) == len(set(flat)), "a landmark appears in two mirror pairs"
        assert all(0 <= i < size for i in flat)


# ------------------------------------------------------------------- aspect and scaling


def test_correct_aspect_scales_x_not_y():
    clip = make_clip()
    out = correct_aspect(clip, 16 / 9)
    assert np.allclose(out[..., 0], clip[..., 0] * 16 / 9)
    assert np.allclose(out[..., 1], clip[..., 1])
    assert np.allclose(out[..., 3], clip[..., 3])


def test_anchor_puts_mid_shoulder_at_the_origin():
    clip = make_clip()
    out, _ = normalize_geometry(clip, aspect=1.0)
    mid = out[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :2].mean(axis=1)
    assert np.allclose(mid, 0.0, atol=1e-5)


def test_scale_makes_shoulder_width_one():
    clip = make_clip(shoulder_half_width=0.07)
    out, _ = normalize_geometry(clip, aspect=1.0)
    width = np.linalg.norm(out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=1)
    assert np.allclose(width, 1.0, atol=1e-5)


def test_normalization_is_invariant_to_camera_distance_and_position():
    """Two recordings of the same pose at different distances must normalize identically."""
    near = make_clip(seed=1, shoulder_half_width=0.10, centre=(0.5, 0.4, 0.0))
    far = near.copy()
    far[..., :3] = (near[..., :3] - np.array([0.5, 0.4, 0.0])) * 0.5 + np.array([0.3, 0.6, 0.0])

    a, _ = normalize_geometry(near, aspect=1.0)
    b, _ = normalize_geometry(far, aspect=1.0)
    assert np.allclose(a[..., :2], b[..., :2], atol=1e-4)


def test_z_is_de_weighted():
    clip = make_clip()
    strong, _ = normalize_geometry(clip, aspect=1.0, z_weight=1.0)
    weak, _ = normalize_geometry(clip, aspect=1.0, z_weight=0.25)
    assert np.allclose(weak[..., 2], strong[..., 2] * 0.25, atol=1e-6)


def test_geometry_raises_when_no_frame_has_shoulders():
    clip = drop_group(make_clip(), "pose")
    with pytest.raises(ValueError, match="cannot be anchored"):
        normalize_geometry(clip, aspect=1.0)


def test_anchor_is_carried_across_frames_that_lost_the_pose():
    clip = make_clip()
    clip = drop_group(clip, "pose", frames=slice(4, 7))
    out, anchorable = normalize_geometry(clip, aspect=1.0)
    assert anchorable.sum() == T - 3
    assert out.shape[0] == T, "frames without a pose must not be dropped"
    assert not np.isnan(out[:, SLICES["face"], :3]).any(), "other groups stay usable"


# --------------------------------------------------------------------- missing hands


def test_missing_hand_is_nan_not_zero():
    """A zero-filled hand sits exactly on the mid-shoulder anchor -- a plausible position."""
    clip = drop_group(make_clip(), "left_hand")
    out, _ = normalize_geometry(clip, aspect=1.0)
    left = out[:, SLICES["left_hand"], :]
    assert np.isnan(left[..., :3]).all()
    assert (left[..., 3] == 0).all()
    assert not (left[..., :3] == 0).any()


def test_group_presence_reports_missing_hands():
    clip = drop_group(make_clip(), "right_hand", frames=slice(0, 5))
    presence = group_presence(clip)
    assert presence["right_hand"].sum() == T - 5
    assert presence["left_hand"].all()
    assert presence["pose"].all()


def test_fill_missing_keeps_the_validity_flag():
    clip = drop_group(make_clip(), "left_hand")
    filled = fill_missing(clip)
    left = filled[:, SLICES["left_hand"], :]
    assert not np.isnan(filled).any()
    assert (left[..., :3] == 0).all()
    assert (left[..., 3] == 0).all(), "the flag is what makes the zeros unambiguous"


def test_interpolate_fills_interior_gaps_only():
    clip = make_clip()
    clip = drop_group(clip, "left_hand", frames=slice(5, 7))
    clip = drop_group(clip, "left_hand", frames=slice(0, 2))  # leading gap: no extrapolation

    out = interpolate_missing(clip, "left_hand")
    presence = group_presence(out)["left_hand"]
    assert presence[5:7].all(), "interior gap should be interpolated"
    assert not presence[0:2].any(), "leading gap must not be extrapolated"


def test_interpolated_values_lie_between_the_neighbours():
    clip = make_clip()
    row = SLICES["left_hand"].start
    clip[:, row, 0] = np.linspace(0.0, 1.0, T)
    gap = slice(4, 6)
    clip = drop_group(clip, "left_hand", frames=gap)

    out = interpolate_missing(clip, "left_hand")
    assert np.allclose(out[gap, row, 0], np.linspace(0.0, 1.0, T)[gap], atol=1e-5)


def test_error_policy_refuses_a_clip_with_a_missing_hand():
    clip = drop_group(make_clip(), "left_hand")
    with pytest.raises(ValueError, match="left_hand missing"):
        normalize_clip(clip, aspect=1.0, missing_hand="error")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="missing_hand must be"):
        normalize_clip(make_clip(), aspect=1.0, missing_hand="zero_fill")


# ------------------------------------------------------------------------- mirroring


def test_mirror_is_its_own_inverse():
    clip = make_clip(seed=3)
    assert np.allclose(mirror(mirror(clip)), clip, equal_nan=True)


def test_mirror_swaps_the_hand_blocks():
    clip = make_clip(seed=4)
    out = mirror(clip)
    left, right = SLICES["left_hand"], SLICES["right_hand"]
    assert np.allclose(out[:, left, 1:], clip[:, right, 1:])
    assert np.allclose(out[:, left, 0], -clip[:, right, 0])


def test_mirror_swaps_shoulder_identities():
    clip = make_clip(seed=5)
    out = mirror(clip)
    assert np.allclose(out[:, LEFT_SHOULDER, 1:], clip[:, RIGHT_SHOULDER, 1:])


def test_mirror_moves_a_one_handed_sign_to_the_other_hand():
    """The point of the operation: a right-handed clip becomes a left-handed one."""
    clip = drop_group(make_clip(seed=6), "left_hand")
    out = mirror(clip)
    presence = group_presence(out)
    assert presence["left_hand"].all()
    assert not presence["right_hand"].any()


def test_mirror_preserves_shoulder_width():
    clip = make_clip(seed=7)
    out = mirror(clip)
    before = np.linalg.norm(clip[:, LEFT_SHOULDER, :2] - clip[:, RIGHT_SHOULDER, :2], axis=1)
    after = np.linalg.norm(out[:, LEFT_SHOULDER, :2] - out[:, RIGHT_SHOULDER, :2], axis=1)
    assert np.allclose(before, after)


# --------------------------------------------------------------- temporal resampling


@pytest.mark.parametrize("n_frames", [1, 8, 64, 200])
def test_resample_returns_the_requested_length(n_frames):
    assert resample_time(make_clip(t=37), n_frames).shape == (n_frames, N_LANDMARKS, 4)


def test_resample_preserves_the_endpoints():
    clip = make_clip(t=30, seed=8)
    out = resample_time(clip, 64)
    assert np.allclose(out[0], clip[0], atol=1e-5)
    assert np.allclose(out[-1], clip[-1], atol=1e-5)


def test_resample_of_a_linear_ramp_is_linear():
    clip = make_clip(t=32)
    clip[:, 40, 0] = np.linspace(0.0, 1.0, 32)
    out = resample_time(clip, 64)
    assert np.allclose(out[:, 40, 0], np.linspace(0.0, 1.0, 64), atol=1e-5)


def test_resample_keeps_a_fully_missing_group_missing():
    clip = drop_group(make_clip(t=20), "left_hand")
    out = resample_time(clip, 64)
    assert np.isnan(out[:, SLICES["left_hand"], :3]).all()
    assert (out[:, SLICES["left_hand"], 3] == 0).all()


def test_resample_of_a_single_frame_repeats_it():
    clip = make_clip(t=1)
    out = resample_time(clip, 64)
    assert np.allclose(out, np.repeat(clip, 64, axis=0))


def test_resample_rejects_an_empty_clip():
    with pytest.raises(ValueError, match="empty clip"):
        resample_time(np.zeros((0, N_LANDMARKS, 4), np.float32), 64)


# --------------------------------------------------------------------- end to end


def test_normalize_clip_end_to_end():
    clip = drop_group(make_clip(t=97, seed=9), "left_hand", frames=slice(10, 20))
    out = normalize_clip(clip, aspect=16 / 9, n_frames=64)

    assert out.shape == (64, N_LANDMARKS, 4)
    assert out.dtype == np.float32
    assert not np.isnan(out).any(), "the model must never receive NaN"
    assert set(np.unique(out[..., 3])) <= {0.0, 1.0}


def test_normalize_clip_rejects_the_wrong_layout():
    with pytest.raises(ValueError, match=r"expected \(T, 115, 4\)"):
        normalize_clip(np.zeros((10, 115, 3), np.float32), aspect=1.0)


def test_normalize_clip_mirroring_changes_the_output():
    clip = make_clip(t=40, seed=10)
    plain = normalize_clip(clip, aspect=1.0)
    flipped = normalize_clip(clip, aspect=1.0, do_mirror=True)
    assert not np.allclose(plain, flipped)
    # Mirroring is horizontal only: after anchoring, y is untouched by the flip itself,
    # though landmark identities move, so compare the sorted y values instead.
    assert np.allclose(np.sort(plain[..., 1], axis=1), np.sort(flipped[..., 1], axis=1), atol=1e-5)
