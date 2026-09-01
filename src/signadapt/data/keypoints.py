"""MediaPipe Holistic extraction to a cached ``.npy`` per clip.

Extraction is a one-time cost (PLAN.md section 10): decode video, run the MediaPipe Holistic
Landmarker, keep 115 landmarks per frame, cache, never re-extract.

Cache format
------------
One ``.npy`` per clip, shape ``(T_original, 115, 4)``, dtype float32, channels
``[x, y, z, valid]``:

* ``x``, ``y`` are normalized to the frame (0..1); ``z`` is MediaPipe's weakly calibrated
  depth, roughly in the same units as ``x``.
* ``valid`` is 1.0 when the landmark was actually detected and 0.0 when it was not, and a
  missing landmark's ``xyz`` is stored as **NaN**, never as zero. A zero-filled hand sits at
  the top-left corner of the frame and is indistinguishable from a real hand there; NaN
  propagates loudly and forces every downstream consumer to make an explicit decision
  (see :mod:`signadapt.data.normalize`).

Landmark layout (row index within the 115)
------------------------------------------
=========  ======  ==========================================
rows       count   group
=========  ======  ==========================================
0..32      33      pose
33..53     21      left hand
54..74     21      right hand
75..114    40      face subset (see :data:`FACE_SUBSET`)
=========  ======  ==========================================

Usage:
    python -m signadapt.data.keypoints --config configs/data.yaml
    python -m signadapt.data.keypoints --config configs/data.yaml --limit 5 --workers 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from signadapt.utils.config import load_config

__all__ = [
    "FACE_SUBSET",
    "N_LANDMARKS",
    "SLICES",
    "ClipStats",
    "ensure_model",
    "extract_clip",
    "extract_dataset",
    "frame_from_result",
    "open_landmarker",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_PATH = Path("data/mediapipe/holistic_landmarker.task")

N_POSE, N_HAND, N_FACE = 33, 21, 40
N_LANDMARKS = N_POSE + 2 * N_HAND + N_FACE  # 115

#: Row ranges of each group inside the 115-landmark array.
SLICES: dict[str, slice] = {
    "pose": slice(0, N_POSE),
    "left_hand": slice(N_POSE, N_POSE + N_HAND),
    "right_hand": slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND),
    "face": slice(N_POSE + 2 * N_HAND, N_LANDMARKS),
}

# ---------------------------------------------------------------------------------------
# Face subset: 40 of MediaPipe's 468 face landmarks.
#
# Why subset at all. The full FaceMesh is 468 points -- four times the pose and both hands
# combined. Feeding all of them would (a) let face *geometry*, which is essentially a
# signer fingerprint, dominate a 345-dimensional input and work directly against the
# signer-independence this thesis measures, (b) cost fps in the live demo for information the
# task does not need, and (c) waste capacity in a deliberately small (<2 M) encoder.
#
# Why these 40. Every index below is taken from MediaPipe's own canonical contour groups
# (``FaceLandmarksConnections``), not hand-picked off a mesh render, and covers the regions
# that carry the non-manual channel. Per PLAN.md section 2 these are used *as features*;
# no claim is made about modelling non-manual grammar.
#
#   lips (16)     mouth aperture and shape. Mouthings and mouth gestures are the
#                 highest-bandwidth non-manual cue, and several LSA64 signs differ mainly
#                 in mouth configuration. 8 outer + 8 inner contour points.
#   eyes (8)      eye aperture: squint and wide-eye are productive markers. 4 per eye
#                 (both corners + upper and lower lid centre) is the minimum that measures
#                 opening independently of head scale.
#   brows (6)     brow raise/furrow marks question and topic constructions in many sign
#                 languages. 3 per brow (inner, mid, outer) captures raise and slant.
#   oval (8)      coarse head pose and tilt. The signing space is defined relative to the
#                 head, so head orientation is a frame of reference for the hands.
#   nose (2)      tip and bridge: a stable head-centre reference, and the contact target for
#                 the face-touching signs in the vocabulary.
#
# Excluded on purpose: irises (identity-bearing, needs refine_landmarks, no task value),
# cheeks and the dense tesselation (pure surface geometry).
# ---------------------------------------------------------------------------------------
_LIPS_OUTER = (61, 291, 0, 17, 37, 267, 84, 314)
_LIPS_INNER = (78, 308, 13, 14, 81, 311, 178, 402)
_EYE_RIGHT = (33, 133, 159, 145)
_EYE_LEFT = (263, 362, 386, 374)
_BROW_RIGHT = (70, 105, 107)
_BROW_LEFT = (300, 334, 336)
_OVAL = (10, 152, 234, 454, 132, 361, 58, 288)
_NOSE = (1, 168)

FACE_SUBSET: tuple[int, ...] = (
    *_LIPS_OUTER,
    *_LIPS_INNER,
    *_EYE_RIGHT,
    *_EYE_LEFT,
    *_BROW_RIGHT,
    *_BROW_LEFT,
    *_OVAL,
    *_NOSE,
)

assert len(FACE_SUBSET) == N_FACE, f"face subset must be {N_FACE} points, got {len(FACE_SUBSET)}"
assert len(set(FACE_SUBSET)) == N_FACE, "face subset contains duplicate indices"
assert max(FACE_SUBSET) < 468, "face subset index outside the FaceMesh topology"

# --- mirror topology -------------------------------------------------------------------
# Handedness mirroring (configs/data.yaml, normalization.mirror_left_handed) is not just a
# sign flip on x: the left/right *landmark identities* have to be swapped too, or a mirrored
# left elbow lands in the right elbow's slot.

#: MediaPipe pose landmark index pairs that swap under mirroring. Index 0 (nose) is its own
#: mirror and is omitted, as are the paired indices' reverse entries.
POSE_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)  # fmt: skip

#: Mirror pairs *within* the 40-point face subset, as row offsets into the face block.
#: The subset above is deliberately ordered so that these pairs are contiguous and checkable
#: by eye: 61<->291 lip corners, 33-block <-> 263-block eyes, 70-block <-> 300-block brows,
#: 234<->454 cheeks, and so on. Midline points (0, 17, 13, 14, 10, 152, 1, 168) are omitted
#: because they map to themselves.
FACE_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (4, 5),
    (6, 7),  # outer lips: corners, upper mids, lower mids
    (8, 9),
    (12, 13),
    (14, 15),  # inner lips
    (16, 20),
    (17, 21),
    (18, 22),
    (19, 23),  # eyes, point for point
    (24, 27),
    (25, 28),
    (26, 29),  # brows
    (32, 33),
    (34, 35),
    (36, 37),  # face oval: cheeks, jaw, lower jaw
)

#: MediaPipe pose indices of the shoulders, used as the normalization anchor and scale.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


@dataclass
class ClipStats:
    """Per-clip extraction quality, written to the cache manifest.

    Attributes:
        clip_id: Stem of the source video, e.g. ``"001_001_001"``.
        path: Cache file path relative to the cache directory.
        n_frames: Frames actually extracted.
        fps: Source frame rate.
        width: Source frame width.
        height: Source frame height.
        detected: Fraction of frames in which each landmark group was detected.
        seconds: Wall-clock extraction time.
    """

    clip_id: str
    path: str
    n_frames: int
    fps: float
    width: int
    height: int
    detected: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "clip_id": self.clip_id,
            "path": self.path,
            "n_frames": self.n_frames,
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "detected": {k: round(v, 4) for k, v in self.detected.items()},
            "seconds": round(self.seconds, 3),
        }


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the Holistic Landmarker task bundle if it is not cached yet.

    Args:
        path: Destination for the ``.task`` bundle.

    Returns:
        The path to the bundle.
    """
    if path.is_file() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[model] downloading holistic_landmarker.task -> {path}")
    urllib.request.urlretrieve(MODEL_URL, path)  # noqa: S310 - fixed https google endpoint
    return path


def _landmark_list(group: Any) -> list[Any]:
    """Normalize a result field to a flat list of landmarks.

    The Holistic task returns one person, but the field is a list; older/newer builds have
    wrapped it in an extra level. Accept both rather than depend on the build.

    Args:
        group: A ``HolisticLandmarkerResult`` landmark field.

    Returns:
        A flat list of landmark objects, empty when nothing was detected.
    """
    if not group:
        return []
    first = group[0]
    if isinstance(first, (list, tuple)):
        return list(first)
    return list(group)


def _fill(
    dest: np.ndarray, rows: slice, landmarks: list[Any], indices: tuple[int, ...] | None
) -> bool:
    """Write one landmark group into a frame buffer.

    Args:
        dest: Frame buffer of shape ``(115, 4)``; missing rows are left as NaN/0.
        rows: Destination row range.
        landmarks: Detected landmarks for this group.
        indices: Subset of ``landmarks`` to keep, or None to keep all in order.

    Returns:
        True if the group was detected and written.
    """
    expected = rows.stop - rows.start
    wanted = indices if indices is not None else range(expected)
    if len(landmarks) < (max(wanted) + 1 if indices else expected):
        return False
    for out_row, src in zip(range(rows.start, rows.stop), wanted, strict=True):
        lm = landmarks[src]
        dest[out_row, 0] = lm.x
        dest[out_row, 1] = lm.y
        dest[out_row, 2] = lm.z
        dest[out_row, 3] = 1.0
    return True


def frame_from_result(result: Any) -> tuple[np.ndarray, dict[str, bool]]:
    """Turn one MediaPipe holistic result into the ``(115, 4)`` buffer the model expects.

    Offline extraction and the live demo both go through here on purpose. A second copy of
    this mapping is the most dangerous kind of duplication in the project: if the demo
    assembled its landmark rows in a different order, or marked validity differently, every
    live prediction would be wrong and nothing would look broken -- the model would happily
    return a confident label for scrambled input.

    Args:
        result: A ``HolisticLandmarkerResult``.

    Returns:
        ``(buffer, detected)`` where ``buffer`` is ``(115, 4)`` float32 with coordinates in
        channels 0..2 and a validity flag in channel 3, and ``detected`` says which groups
        were found. Undetected rows are NaN with validity 0, never a silent zero -- the model
        is trained to read that flag (see ``configs/model.yaml``).
    """
    buffer = np.full((N_LANDMARKS, 4), np.nan, dtype=np.float32)
    buffer[:, 3] = 0.0
    groups = {
        "pose": (_landmark_list(result.pose_landmarks), None),
        "left_hand": (_landmark_list(result.left_hand_landmarks), None),
        "right_hand": (_landmark_list(result.right_hand_landmarks), None),
        "face": (_landmark_list(result.face_landmarks), FACE_SUBSET),
    }
    detected = {
        name: _fill(buffer, SLICES[name], landmarks, subset)
        for name, (landmarks, subset) in groups.items()
    }
    return buffer, detected


def open_landmarker(cfg: dict[str, Any]) -> Any:
    """Create a HolisticLandmarker in VIDEO mode from the extraction config.

    Args:
        cfg: Loaded ``configs/data.yaml``.

    Returns:
        An open landmarker; the caller must close it.
    """
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HolisticLandmarker,
        HolisticLandmarkerOptions,
        RunningMode,
    )

    extraction = cfg.get("extraction", {})
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=extraction.get("min_detection_confidence", 0.5),
        min_pose_landmarks_confidence=extraction.get("min_tracking_confidence", 0.5),
        min_hand_landmarks_confidence=extraction.get("min_tracking_confidence", 0.5),
        min_face_detection_confidence=extraction.get("min_detection_confidence", 0.5),
        min_face_landmarks_confidence=extraction.get("min_tracking_confidence", 0.5),
    )
    return HolisticLandmarker.create_from_options(options)


def extract_clip(
    video_path: Path,
    cfg: dict[str, Any],
    *,
    frame_stride: int = 1,
    resize_width: int | None = 640,
) -> tuple[np.ndarray, ClipStats]:
    """Extract landmarks for one video.

    A fresh landmarker is created per clip. That is not just tidiness: in VIDEO running mode
    MediaPipe tracks across calls, so a reused instance would (a) reject the next clip's
    timestamps for not increasing, and (b) seed the first frames of one clip with the
    tracking ROI left over from the end of the previous one -- a subtle cross-clip
    contamination that would be invisible in the output.

    Args:
        video_path: Source video.
        cfg: Loaded ``configs/data.yaml``, used for the detector thresholds.
        frame_stride: Keep every n-th frame.
        resize_width: Downscale frames to this width before detection; None keeps native
            resolution. LSA64 is 1080p and decoding dominates the cost.

    Returns:
        A tuple of the ``(T, 115, 4)`` float32 array and its :class:`ClipStats`.

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    import mediapipe as mp_lib

    started = time.time()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    hits = dict.fromkeys(SLICES, 0)
    index = 0
    landmarker = open_landmarker(cfg)
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if index % frame_stride:
                index += 1
                continue
            if resize_width and width > resize_width:
                scale = resize_width / width
                frame_bgr = cv2.resize(frame_bgr, (resize_width, int(round(height * scale))))

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(round(1000.0 * index / fps))
            result = landmarker.detect_for_video(image, timestamp_ms)

            buffer, detected = frame_from_result(result)
            for name, found in detected.items():
                hits[name] += int(found)
            frames.append(buffer)
            index += 1
    finally:
        landmarker.close()
        capture.release()
    array = np.stack(frames) if frames else np.full((0, N_LANDMARKS, 4), np.nan, dtype=np.float32)
    n = max(len(frames), 1)
    stats = ClipStats(
        clip_id=video_path.stem,
        path="",
        n_frames=len(frames),
        fps=fps,
        width=width,
        height=height,
        detected={k: v / n for k, v in hits.items()},
        seconds=time.time() - started,
    )
    return array, stats


_WORKER_STATE: dict[str, Any] = {}


def _worker_init(cfg: dict[str, Any]) -> None:
    """Store the config in the worker process; landmarkers are per clip, not per worker."""
    _WORKER_STATE["cfg"] = cfg


def _worker_extract(job: tuple[str, str]) -> dict[str, Any] | None:
    """Extract one clip inside a worker process.

    Args:
        job: ``(video_path, cache_path)``.

    Returns:
        The clip's stats dict, or None when extraction failed.
    """
    video_path, cache_path = Path(job[0]), Path(job[1])
    cfg = _WORKER_STATE["cfg"]
    try:
        array, stats = extract_clip(
            video_path,
            cfg,
            frame_stride=cfg.get("extraction", {}).get("frame_stride", 1),
            resize_width=cfg.get("extraction", {}).get("resize_width", 640),
        )
    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill a 3200-clip run
        print(f"[extract] FAILED {video_path.name}: {type(exc).__name__}: {exc}")
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    stats.path = cache_path.name
    return stats.to_dict()


def extract_dataset(
    cfg: dict[str, Any],
    *,
    limit: int | None = None,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract and cache keypoints for every clip in the configured dataset.

    Args:
        cfg: Loaded ``configs/data.yaml``.
        limit: Process at most this many clips (for smoke tests).
        workers: Process count; defaults to ``extraction.n_workers``.
        force: Re-extract clips that are already cached.

    Returns:
        The manifest dictionary that was written to ``<cache_dir>/manifest.json``.

    Raises:
        FileNotFoundError: If the raw video directory holds no mp4 files.
    """
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    cache_dir = Path(cfg["dataset"]["cache_dir"])
    videos = sorted(raw_dir.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(
            f"no .mp4 under {raw_dir} -- run `python -m signadapt.data.download` first"
        )
    if limit:
        videos = videos[:limit]

    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(v), str(cache_dir / f"{v.stem}.npy"))
        for v in videos
        if force or not (cache_dir / f"{v.stem}.npy").is_file()
    ]
    print(f"[extract] {len(videos)} clips found, {len(jobs)} to extract")

    n_workers = workers if workers is not None else cfg.get("extraction", {}).get("n_workers", 4)
    n_workers = max(1, min(n_workers, len(jobs) or 1))
    started = time.time()
    records: list[dict[str, Any]] = []

    manifest_path = cache_dir / "manifest.json"
    manifest: dict[str, Any] = {"clips": {}}
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text())

    def flush() -> None:
        """Write the manifest for the clips finished so far.

        Called periodically, not only at the end: a 3200-clip run takes half an hour, and a
        manifest that only appears on completion makes every partial result unusable.
        """
        for done in records:
            manifest["clips"][done["clip_id"]] = done
        manifest["dataset"] = cfg["dataset"]["name"]
        manifest["n_clips"] = len(manifest["clips"])
        manifest["landmarks"] = {
            "layout": {k: [v.start, v.stop] for k, v in SLICES.items()},
            "face_subset": list(FACE_SUBSET),
            "channels": ["x", "y", "z", "valid"],
        }
        manifest["extraction"] = cfg.get("extraction", {})
        tmp = manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(manifest_path)

    def consume(iterator: Any, total: int) -> None:
        for done, record in enumerate(tqdm(iterator, total=total, desc="extract", unit="clip"), 1):
            if record:
                records.append(record)
            if done % 200 == 0:
                flush()

    if jobs and n_workers == 1:
        _worker_init(cfg)
        consume((_worker_extract(job) for job in jobs), len(jobs))
    elif jobs:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_worker_init, initargs=(cfg,)) as pool:
            consume(pool.imap_unordered(_worker_extract, jobs), len(jobs))
    flush()

    elapsed = time.time() - started
    if records:
        print(
            f"[extract] {len(records)} clips in {elapsed:.1f}s "
            f"({elapsed / len(records):.2f}s per clip)"
        )
    print(f"[extract] manifest: {manifest_path} ({manifest['n_clips']} clips)")
    return manifest


def extraction_quality(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize landmark detection rates over an extracted dataset.

    Detection rate is a dataset property worth reporting rather than a log line: on LSA64 it
    varies systematically with handshape and with signer, which is a confound the thesis has
    to disclose when it attributes inter-signer variance to signing style.

    Args:
        manifest: The manifest written by :func:`extract_dataset`.

    Returns:
        Overall, per-signer and per-class detection statistics.
    """
    from signadapt.data.dataset import is_two_handed, parse_clip_id

    clips = manifest["clips"]
    groups = list(SLICES)
    overall = {g: float(np.mean([c["detected"][g] for c in clips.values()])) for g in groups}

    def required_hands(clip_id: str, clip: dict[str, Any]) -> float:
        """Fraction of frames in which the hands this sign actually uses were detected.

        Taking the *best* hand would flatter two-handed signs, where a clip with only the
        dominant hand tracked is missing half the sign. For a two-handed sign the measure is
        therefore the minimum of the two, not the maximum.
        """
        label, _, _ = parse_clip_id(clip_id)
        left, right = clip["detected"]["left_hand"], clip["detected"]["right_hand"]
        return min(left, right) if is_two_handed(label) else max(left, right)

    required = {cid: required_hands(cid, c) for cid, c in clips.items()}
    by_handedness: dict[str, dict[str, float]] = {}
    for name, wanted in (("one_handed", False), ("two_handed", True)):
        subset = {
            cid: value
            for cid, value in required.items()
            if is_two_handed(parse_clip_id(cid)[0]) is wanted
        }
        by_handedness[name] = {
            "n_clips": len(subset),
            "required_hands_present": float(np.mean(list(subset.values()))),
            "clips_never_complete": int(sum(v == 0.0 for v in subset.values())),
            "clips_below_50pct": int(sum(v < 0.5 for v in subset.values())),
        }

    per_signer: dict[str, list[float]] = {}
    per_class: dict[str, list[float]] = {}
    for clip_id, value in required.items():
        sign, signer = clip_id.split("_")[0], clip_id.split("_")[1]
        per_signer.setdefault(signer, []).append(value)
        per_class.setdefault(sign, []).append(value)

    signer_means = {k: float(np.mean(v)) for k, v in sorted(per_signer.items())}
    return {
        "n_clips": len(clips),
        "mean_detection": overall,
        "required_hands_present_mean": float(np.mean(list(required.values()))),
        "clips_never_complete": int(sum(v == 0.0 for v in required.values())),
        "clips_below_50pct": int(sum(v < 0.5 for v in required.values())),
        "by_handedness": by_handedness,
        "required_hands_by_signer": signer_means,
        # Spread across signers is the number that matters for the thesis: it bounds how
        # much of the inter-signer accuracy variance is data quality rather than signing.
        "signer_spread": float(max(signer_means.values()) - min(signer_means.values())),
        "required_hands_by_class": {k: float(np.mean(v)) for k, v in sorted(per_class.items())},
        "frames": {
            "min": int(min(c["n_frames"] for c in clips.values())),
            "max": int(max(c["n_frames"] for c in clips.values())),
            "mean": float(np.mean([c["n_frames"] for c in clips.values()])),
        },
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    configured = cfg["landmarks"]["face"]["n_landmarks"]
    if configured != N_FACE:
        raise ValueError(
            f"configs/data.yaml says {configured} face landmarks but keypoints.py "
            f"defines {N_FACE}; the two must agree"
        )
    manifest = extract_dataset(cfg, limit=args.limit, workers=args.workers, force=args.force)

    from signadapt.utils.results import ResultsLogger

    quality = extraction_quality(manifest)
    with ResultsLogger(
        "phase1-extraction",
        config=cfg,
        tag=cfg["dataset"]["name"],
        description="MediaPipe landmark detection rates over the extracted dataset.",
        results_dir=args.results_dir,
    ) as log:
        log.set_metrics(**quality)
    one, two = quality["by_handedness"]["one_handed"], quality["by_handedness"]["two_handed"]
    print(
        f"[quality] required hands present in {quality['required_hands_present_mean']:.1%} "
        f"of frames  (one-handed {one['required_hands_present']:.1%}, "
        f"two-handed {two['required_hands_present']:.1%})"
    )
    print(
        f"[quality] clips never complete: {quality['clips_never_complete']} "
        f"({one['clips_never_complete']} one-handed, {two['clips_never_complete']} two-handed); "
        f"per-signer spread {quality['signer_spread']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
