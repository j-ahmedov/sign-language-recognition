"""Render cached keypoints back over the source video for visual verification.

Extraction quality is not something to take on trust from a detection percentage: a detector
can report 100 % confidence while tracking the wrong hand. This renders a side-by-side video
-- source frames with the raw landmarks drawn on them, next to the *normalized* skeleton the
model will actually see -- so both the extraction and the normalization can be checked by
eye. It is a debugging tool, not part of the training path.

Not listed in PLAN.md section 7's layout; added here because phase 1 requires a visual
sanity check and this is the smallest place to put it.

Usage:
    python -m signadapt.data.overlay --clips 001_001_001 002_004_003 --out figures/overlays
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from signadapt.data.dataset import parse_clip_id
from signadapt.data.keypoints import LEFT_SHOULDER, RIGHT_SHOULDER, SLICES
from signadapt.data.normalize import config_kwargs, group_presence, normalize_clip
from signadapt.utils.config import load_config

__all__ = ["HAND_EDGES", "POSE_EDGES", "render_overlay"]

#: MediaPipe's 21-point hand skeleton.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)  # fmt: skip

#: Upper-body pose edges. The legs are irrelevant to signing and only add clutter.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (0, 11), (0, 12),
)  # fmt: skip

_COLOURS = {
    "pose": (255, 200, 0),
    "left_hand": (0, 220, 60),
    "right_hand": (60, 120, 255),
    "face": (200, 200, 200),
}


def _draw_group(
    canvas: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    colour: tuple[int, int, int],
    *,
    radius: int = 3,
) -> None:
    """Draw one landmark group's points and edges onto a canvas in pixel coordinates."""
    for a, b in edges:
        if a < len(points) and b < len(points) and valid[a] and valid[b]:
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), colour, 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(canvas, tuple(point), radius, colour, -1, cv2.LINE_AA)


def _to_pixels(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized (0..1) coordinates to integer pixel coordinates."""
    scaled = xy * np.array([width, height], dtype=np.float32)
    return np.nan_to_num(scaled, nan=-1e4).astype(np.int32)


def _render_normalized_panel(
    frame_norm: np.ndarray, size: tuple[int, int], caption: str
) -> np.ndarray:
    """Draw the normalized skeleton on a blank canvas, in normalized units.

    Args:
        frame_norm: One frame, ``(115, 4)``, already anchored and scaled.
        size: ``(width, height)`` of the panel.
        caption: Text drawn at the top.

    Returns:
        A BGR image.
    """
    width, height = size
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    # Normalized units are shoulder-widths around the mid-shoulder origin; show +-2.5 of them.
    span = 2.5
    xy = frame_norm[:, :2].copy()
    px = np.stack(
        [
            (xy[:, 0] / (2 * span) + 0.5) * width,
            (xy[:, 1] / (2 * span) + 0.5) * height,
        ],
        axis=1,
    )
    px = np.nan_to_num(px, nan=-1e4).astype(np.int32)
    valid = frame_norm[:, 3] > 0.5

    cv2.line(canvas, (0, height // 2), (width, height // 2), (60, 60, 60), 1)
    cv2.line(canvas, (width // 2, 0), (width // 2, height), (60, 60, 60), 1)

    for name, rows in SLICES.items():
        edges = POSE_EDGES if name == "pose" else HAND_EDGES if name.endswith("hand") else ()
        _draw_group(
            canvas,
            px[rows] if name != "pose" else px,
            valid[rows] if name != "pose" else valid,
            edges,
            _COLOURS[name],
            radius=2 if name == "face" else 3,
        )
    cv2.putText(canvas, caption, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return canvas


def render_overlay(
    video_path: Path,
    cache_path: Path,
    out_path: Path,
    cfg: dict[str, Any],
    *,
    panel_width: int = 640,
) -> dict[str, Any]:
    """Render a side-by-side verification video for one clip.

    Args:
        video_path: Source ``.mp4``.
        cache_path: Cached ``.npy`` written by :mod:`signadapt.data.keypoints`.
        out_path: Destination ``.mp4``.
        cfg: Loaded ``configs/data.yaml``.
        panel_width: Width of each of the two panels.

    Returns:
        A summary dict with detection rates and the output path.

    Raises:
        RuntimeError: If the source video cannot be opened.
    """
    raw = np.load(cache_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    aspect = src_w / src_h

    normalized = normalize_clip(raw, aspect=aspect, fill=False, **config_kwargs(cfg))
    presence = group_presence(raw)

    panel_h = int(round(panel_width / aspect))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_width * 2, panel_h)
    )

    index = 0
    while index < raw.shape[0]:
        ok, frame = capture.read()
        if not ok:
            break
        left = cv2.resize(frame, (panel_width, panel_h))
        xy = _to_pixels(raw[index, :, :2], panel_width, panel_h)
        valid = raw[index, :, 3] > 0.5

        _draw_group(left, xy, valid, POSE_EDGES, _COLOURS["pose"])
        for name in ("left_hand", "right_hand"):
            rows = SLICES[name]
            _draw_group(left, xy[rows], valid[rows], HAND_EDGES, _COLOURS[name])
        face = SLICES["face"]
        _draw_group(left, xy[face], valid[face], (), _COLOURS["face"], radius=2)

        status = " ".join(
            f"{name.replace('_', '')}:{'y' if presence[name][index] else 'N'}"
            for name in ("pose", "left_hand", "right_hand", "face")
        )
        cv2.putText(
            left,
            f"raw f{index:3d} {status}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        norm_index = int(round(index / max(raw.shape[0] - 1, 1) * (normalized.shape[0] - 1)))
        right = _render_normalized_panel(
            normalized[norm_index],
            (panel_width, panel_h),
            f"normalized t{norm_index:2d}/{normalized.shape[0]}  (1 unit = shoulder width)",
        )
        writer.write(np.hstack([left, right]))
        index += 1

    capture.release()
    writer.release()

    width = np.linalg.norm(raw[:, LEFT_SHOULDER, :2] - raw[:, RIGHT_SHOULDER, :2], axis=1)
    return {
        "clip": video_path.stem,
        "out": str(out_path),
        "frames": int(raw.shape[0]),
        "detected": {k: float(v.mean()) for k, v in presence.items()},
        "shoulder_width_px_mean": float(np.nanmean(width) * src_w),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--clips", nargs="+", required=True, help="clip ids, e.g. 001_001_001")
    parser.add_argument("--out", default="figures/overlays")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    cache_dir = Path(cfg["dataset"]["cache_dir"])

    # Deliberately not going through the manifest: this is a debugging tool that has to work
    # while a long extraction run is still in flight, and the video itself supplies the
    # aspect ratio the manifest would otherwise provide.
    for clip_id in args.clips:
        parse_clip_id(clip_id)  # fail early on a typo
        cache_path = cache_dir / f"{clip_id}.npy"
        if not cache_path.is_file():
            print(f"[overlay] {clip_id}: no cached keypoints at {cache_path}, skipping")
            continue
        matches = list(raw_dir.rglob(f"{clip_id}.mp4"))
        if not matches:
            print(f"[overlay] {clip_id}: source video not found under {raw_dir}, skipping")
            continue
        summary = render_overlay(
            matches[0], cache_path, Path(args.out) / f"{clip_id}_overlay.mp4", cfg
        )
        rates = " ".join(f"{k}={v:.0%}" for k, v in summary["detected"].items())
        print(
            f"[overlay] {clip_id}: {summary['frames']:3d} frames  {rates}  "
            f"shoulders={summary['shoulder_width_px_mean']:.0f}px  -> {summary['out']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
