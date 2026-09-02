"""Webcam -> keypoints -> prediction, with end-to-end latency logging (PLAN.md sections 4, 8).

This module is the evidence for RQ4: what it costs to run the system on the machine it would
actually run on. The accuracy questions were settled in phases 2 to 5 on held-out signers;
what is measured here is frame rate and per-stage latency, and the target is PLAN.md section
8's ">= 15 fps with measured latency".

**The demo is not evidence about recognition quality.** It serves a model trained on seven
LSA64 signers performing 64 Argentinian signs in a fixed studio setup. Someone signing at a
laptop is a new signer, in a new room, at a new camera angle, and -- unless they happen to
know LSA64 -- performing something outside the label set entirely. The model has no "no sign"
class: it must return one of 64 labels for whatever it is shown, so a confident-looking
caption over an unrecognized gesture is the expected behaviour, not a malfunction. PLAN.md
section 3 is explicit that own recordings are illustrative only, and the overlay says so on
screen so a screenshot cannot be mistaken for a result.

Three things keep the live path honest:

* **One definition of a landmark frame.** Both this module and offline extraction call
  :func:`~signadapt.data.keypoints.frame_from_result`, and both normalize through
  :func:`~signadapt.data.normalize.normalize_clip` with the same config. A demo that built
  its input slightly differently from training would mispredict silently.
* **A correctness gate, ``--verify``.** It streams held-out clips through this exact live
  path and compares the result against the cached offline pipeline. Nothing about a broken
  live path is visible in a demo video, which is why it is checked rather than watched.
* **A rolling window rather than a whole clip.** Training clips are trimmed to one sign;
  a live stream is not, so the model sees the last :data:`WINDOW_SECONDS` of video. That is
  a real distribution shift and ``--verify`` measures what it costs.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from signadapt.data.dataset import SIGN_NAMES
from signadapt.data.keypoints import N_LANDMARKS, frame_from_result, open_landmarker
from signadapt.data.normalize import config_kwargs, normalize_clip
from signadapt.demo.virtualcam import open_sink, render_overlay
from signadapt.models.model import build_model
from signadapt.train.loop import resolve_device
from signadapt.utils.config import apply_overrides, load_config
from signadapt.utils.results import ResultsLogger

__all__ = [
    "WINDOW_SECONDS",
    "ClipBuffer",
    "FrameSource",
    "LandmarkStream",
    "Prediction",
    "Recognizer",
    "StageClock",
    "main",
    "run_demo",
    "verify_pipeline",
]

#: Seconds of video the model is shown at each prediction.
#:
#: LSA64's clips are trimmed to a single sign and last 1.30 s at the median, 1.97 s at the
#: 90th percentile (measured, ``data/cache/lsa64/manifest.json``). The window has to cover
#: about one sign: too short and the model sees a fragment, too long and the sign is diluted
#: by whatever preceded it. 1.6 s sits between the median and p90. Note that the window is
#: kept in *seconds* and resampled to T frames, not in frames -- LSA64 recorded at 59.94 fps
#: and a webcam typically delivers 30, so a frame-count window would show the model half a
#: sign on the machine the demo actually runs on.
WINDOW_SECONDS = 1.6

#: Served by default: the FedAvg model from phase 3, trained federatedly on signers 1-7 with
#: one client per signer. It is a real artefact of the thesis rather than a demo-only model,
#: and it has a trained head, which FedPer's encoder-only checkpoints do not.
DEFAULT_CHECKPOINT = Path("data/checkpoints/fedavg-pretrain_seed0.pt")


@dataclass(frozen=True)
class Prediction:
    """One model output, after smoothing and thresholding.

    Attributes:
        top: ``(sign name, probability)`` pairs in descending order.
        confident: Whether the leading probability cleared the caption threshold.
    """

    top: tuple[tuple[str, float], ...]
    confident: bool

    @property
    def label(self) -> str:
        """Return the caption, or an ellipsis when nothing cleared the threshold."""
        return self.top[0][0] if (self.top and self.confident) else "..."


@dataclass
class StageClock:
    """Collects per-stage timings so the results file can report percentiles, not an average.

    A mean frame time hides exactly what matters in a live pipeline: the occasional slow
    frame that a viewer perceives as a stutter. p95 is what a reader should judge, so every
    sample is kept and summarized at the end.

    Attributes:
        samples: Milliseconds per stage, in arrival order.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Time a block and record it under ``stage``.

        Args:
            stage: Stage name, e.g. ``"landmarks"``.

        Yields:
            None.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.samples.setdefault(stage, []).append(1000.0 * (time.perf_counter() - started))

    def latest(self) -> dict[str, float]:
        """Return the most recent sample of each stage, in milliseconds, for the HUD."""
        return {f"{stage}_ms": values[-1] for stage, values in self.samples.items() if values}

    def summary(self) -> dict[str, dict[str, float]]:
        """Summarize every stage.

        Returns:
            ``{stage: {"mean", "p50", "p95", "max", "n"}}`` in milliseconds.
        """
        out: dict[str, dict[str, float]] = {}
        for stage, values in self.samples.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            out[stage] = {
                "mean": float(array.mean()),
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)),
                "max": float(array.max()),
                "n": int(array.size),
            }
        return out


class FrameSource:
    """A webcam or a video file, behind one interface.

    A file source is what makes this pipeline measurable without a camera: the same code runs
    in a test and on a laptop, and the latency it reports is the pipeline's rather than the
    camera driver's.
    """

    def __init__(self, spec: str, *, width: int | None = None, height: int | None = None) -> None:
        """Open a source.

        Args:
            spec: ``"webcam"``, ``"webcam:N"`` for camera index N, or a path to a video file.
            width: Requested capture width; cameras may ignore it.
            height: Requested capture height.

        Raises:
            RuntimeError: If the source cannot be opened, with the likely cause.
        """
        self.spec = spec
        self.is_live = spec.startswith("webcam")
        if self.is_live:
            index = int(spec.split(":", 1)[1]) if ":" in spec else 0
            self._capture = cv2.VideoCapture(index)
            if width:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self._capture.isOpened():
                raise RuntimeError(
                    f"cannot open camera {index}. On macOS the terminal needs camera "
                    "permission: System Settings > Privacy & Security > Camera. Pass "
                    "--source <video.mp4> to run the pipeline without a camera."
                )
        else:
            if not Path(spec).is_file():
                raise RuntimeError(f"no such video file: {spec}")
            self._capture = cv2.VideoCapture(spec)
            if not self._capture.isOpened():
                raise RuntimeError(f"cannot open video: {spec}")

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def aspect(self) -> float:
        """Return width / height, which normalization needs to undo the pixel aspect."""
        return self.width / max(self.height, 1)

    def read(self) -> np.ndarray | None:
        """Read one BGR frame.

        Returns:
            The frame, or None at end of stream.
        """
        ok, frame = self._capture.read()
        return frame if ok else None

    def close(self) -> None:
        """Release the device or file."""
        self._capture.release()


class LandmarkStream:
    """A persistent MediaPipe holistic landmarker over one continuous session.

    Offline extraction builds a fresh landmarker per clip, deliberately, so that tracking
    state cannot leak from one clip into the next. Here the opposite is correct: a live
    session *is* one continuous video, and MediaPipe's frame-to-frame tracking is most of
    why it is fast enough to matter. That is a real difference between the live and offline
    paths, and it is the one difference ``--verify`` cannot cover.
    """

    def __init__(self, data_cfg: dict[str, Any], *, resize_width: int | None = 640) -> None:
        """Open a landmarker.

        Args:
            data_cfg: Loaded ``configs/data.yaml``.
            resize_width: Downscale frames to this width before detection. 640 is what
                offline extraction used on 1080p source, so the demo sees the same scale.
        """
        self._landmarker = open_landmarker(data_cfg)
        self.resize_width = resize_width

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Detect landmarks in one frame.

        Args:
            frame_bgr: BGR frame.
            timestamp_ms: Monotonically increasing timestamp; MediaPipe rejects repeats.

        Returns:
            A ``(115, 4)`` float32 buffer, NaN where nothing was detected.
        """
        import mediapipe as mp_lib

        height, width = frame_bgr.shape[:2]
        if self.resize_width and width > self.resize_width:
            scale = self.resize_width / width
            frame_bgr = cv2.resize(frame_bgr, (self.resize_width, int(round(height * scale))))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        buffer, _ = frame_from_result(result)
        return buffer

    def close(self) -> None:
        """Release the landmarker."""
        self._landmarker.close()


class ClipBuffer:
    """The last :data:`WINDOW_SECONDS` of landmark frames, ready to normalize.

    The window is held by timestamp rather than by frame count so that the model is shown the
    same *duration* of signing whatever frame rate the camera delivers.
    """

    def __init__(self, *, seconds: float = WINDOW_SECONDS, min_frames: int = 12) -> None:
        """Create an empty buffer.

        Args:
            seconds: Length of the window.
            min_frames: Frames required before the buffer will produce a clip. The shortest
                LSA64 clip is 14 frames; below roughly that there is not enough motion for a
                prediction to mean anything, and the demo shows its waiting state instead.
        """
        self.seconds = seconds
        self.min_frames = min_frames
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def append(self, timestamp_s: float, frame: np.ndarray) -> None:
        """Add one landmark frame and drop everything older than the window.

        Args:
            timestamp_s: Capture time in seconds.
            frame: A ``(115, 4)`` buffer.
        """
        self._frames.append((timestamp_s, frame))
        cutoff = timestamp_s - self.seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def __len__(self) -> int:
        """Return the number of frames currently in the window."""
        return len(self._frames)

    @property
    def ready(self) -> bool:
        """Return whether there are enough frames to predict from."""
        return len(self._frames) >= self.min_frames

    def raw(self) -> np.ndarray:
        """Return the window as a ``(T_in, 115, 4)`` array in capture order."""
        if not self._frames:
            return np.full((0, N_LANDMARKS, 4), np.nan, dtype=np.float32)
        return np.stack([frame for _, frame in self._frames])

    def clear(self) -> None:
        """Drop every frame, e.g. between clips in verification."""
        self._frames.clear()


class Recognizer:
    """Loads a checkpoint and turns a normalized clip into class probabilities."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        model_cfg: dict[str, Any],
        device: torch.device,
        class_names: tuple[str, ...] = SIGN_NAMES,
    ) -> None:
        """Load a served model.

        Args:
            checkpoint: A ``.pt`` written by the federated or centralized training code.
            model_cfg: Loaded ``configs/model.yaml``; must match the checkpoint.
            device: Where inference runs.
            class_names: Label names, indexed by class id.

        Raises:
            FileNotFoundError: If the checkpoint does not exist, naming how to produce one.
            ValueError: If the checkpoint holds only an encoder, which cannot classify.
        """
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"no checkpoint at {path}. `make federated` writes "
                f"{DEFAULT_CHECKPOINT}, which is the model this demo serves by default."
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state", payload)
        if not any(key.startswith("head.") for key in state):
            raise ValueError(
                f"{path} holds no head, so it cannot produce a label. FedPer checkpoints are "
                "encoder-only by design: a head belongs to one signer and never leaves them "
                "(see signadapt.federated.client). Serve a FedAvg or centralized checkpoint."
            )
        self.name = f"{payload.get('experiment', path.stem)} seed {payload.get('seed', '?')}"
        self.class_names = class_names
        self.device = device
        self._model = build_model(model_cfg)
        self._model.load_state_dict(state)
        self._model.to(device).eval()
        self.n_parameters = self._model.n_parameters()

    @torch.inference_mode()
    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """Score one normalized clip.

        Args:
            clip: A ``(T, 115, 4)`` normalized array.

        Returns:
            A ``(n_classes,)`` float array of probabilities.
        """
        x = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))[None].to(self.device)
        logits = self._model(x)
        return torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()

    def top_k(self, probs: np.ndarray, k: int = 4) -> tuple[tuple[str, float], ...]:
        """Return the k highest-probability classes as ``(name, probability)``.

        Args:
            probs: Probability vector.
            k: How many to return.

        Returns:
            Descending by probability.
        """
        order = np.argsort(probs)[::-1][:k]
        return tuple((self.class_names[int(i)], float(probs[int(i)])) for i in order)


# ------------------------------------------------------------------ the live loop


def _notice(checkpoint_meta: dict[str, Any]) -> str:
    """Build the standing on-screen caveat.

    Args:
        checkpoint_meta: What the served checkpoint records about itself.

    Returns:
        A single line naming the vocabulary and the training signers.
    """
    signers = checkpoint_meta.get("train_signers")
    who = f"signers {min(signers)}-{max(signers)}" if signers else "held-out signers"
    # ASCII only: the overlay is drawn with OpenCV's Hershey fonts, which have no glyph for
    # an em dash and substitute a question mark.
    return f"LSA64 64 signs, trained on {who} | illustrative, not validated"


def run_demo(
    *,
    source: FrameSource,
    stream: LandmarkStream,
    recognizer: Recognizer,
    sink: Any,
    data_cfg: dict[str, Any],
    max_frames: int | None = None,
    max_seconds: float | None = None,
    window_seconds: float = WINDOW_SECONDS,
    min_confidence: float = 0.5,
    smoothing: float = 0.6,
    predict_every: int = 1,
    notice: str = "",
) -> dict[str, Any]:
    """Run the pipeline until the source ends or a budget is reached.

    Args:
        source: Frames in.
        stream: Landmark detector.
        recognizer: Served model.
        sink: Where rendered frames go.
        data_cfg: Loaded ``configs/data.yaml``, for the normalization parameters.
        max_frames: Stop after this many frames, or None for the whole source.
        max_seconds: Stop after this much wall-clock time, or None.
        window_seconds: Length of the rolling window.
        min_confidence: Probability the top class must reach before it is captioned.
        smoothing: Exponential weight on the previous probability vector, in [0, 1). Raw
            per-frame predictions flicker between neighbouring classes several times a
            second, which reads as a broken demo even when the top class is stable.
        predict_every: Run the model every n-th frame; the overlay keeps the last result
            in between. 1 measures the full cost, which is what RQ4 asks for.
        notice: The standing caveat drawn on every frame.

    Returns:
        A metrics dictionary: frame rate, per-stage latency percentiles, and what was served.
    """
    clock = StageClock()
    buffer = ClipBuffer(seconds=window_seconds)
    normalize_kwargs = config_kwargs(data_cfg)
    smoothed: np.ndarray | None = None
    prediction = Prediction(top=(), confident=False)
    frame_times: list[float] = []
    n_predictions = 0

    started = time.perf_counter()
    index = 0
    try:
        while True:
            if max_frames is not None and index >= max_frames:
                break
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                break
            frame_started = time.perf_counter()

            with clock.time("capture"):
                frame = source.read()
            if frame is None:
                break

            now = time.perf_counter() - started
            with clock.time("landmarks"):
                # A live camera has no frame index, so drive MediaPipe from the wall clock;
                # it only requires the timestamp to increase.
                landmarks = stream(frame, int(round(now * 1000)))
            buffer.append(now, landmarks)

            if buffer.ready and index % predict_every == 0:
                with clock.time("normalize"):
                    clip = normalize_clip(buffer.raw(), aspect=source.aspect, **normalize_kwargs)
                with clock.time("model"):
                    probs = recognizer(clip)
                smoothed = (
                    probs if smoothed is None else smoothing * smoothed + (1.0 - smoothing) * probs
                )
                top = recognizer.top_k(smoothed)
                prediction = Prediction(top=top, confident=top[0][1] >= min_confidence)
                n_predictions += 1

            telemetry = clock.latest()
            telemetry["latency_ms"] = 1000.0 * (time.perf_counter() - frame_started)
            telemetry["fps"] = 1000.0 / np.mean(frame_times[-30:]) if frame_times else float("nan")
            with clock.time("render"):
                canvas = render_overlay(
                    frame,
                    labels=list(prediction.top),
                    telemetry=telemetry,
                    model_name=recognizer.name,
                    notice=notice,
                    confident=prediction.confident,
                )
                sink.send(canvas)

            frame_times.append(1000.0 * (time.perf_counter() - frame_started))
            index += 1
            if sink.should_stop():
                break
    except KeyboardInterrupt:
        pass

    elapsed = time.perf_counter() - started
    per_frame = np.asarray(frame_times, dtype=np.float64)
    fps = float(len(frame_times) / elapsed) if elapsed > 0 else float("nan")
    return {
        "n_frames": len(frame_times),
        "n_predictions": n_predictions,
        "seconds": elapsed,
        "fps": fps,
        # The sustained rate is what a viewer experiences; the per-frame budget is what a
        # p95 frame cost would allow if nothing else ran. They differ when a stage blocks.
        "fps_from_p50_frame": float(1000.0 / np.percentile(per_frame, 50))
        if per_frame.size
        else float("nan"),
        "frame_ms": {
            "mean": float(per_frame.mean()) if per_frame.size else float("nan"),
            "p50": float(np.percentile(per_frame, 50)) if per_frame.size else float("nan"),
            "p95": float(np.percentile(per_frame, 95)) if per_frame.size else float("nan"),
            "max": float(per_frame.max()) if per_frame.size else float("nan"),
        },
        "stages_ms": clock.summary(),
        "meets_target": bool(fps >= 15.0),
        "target_fps": 15.0,
        "window_seconds": window_seconds,
        "min_confidence": min_confidence,
        "smoothing": smoothing,
        "predict_every": predict_every,
        "source": {
            "spec": source.spec,
            "live": source.is_live,
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
        },
        "model": {
            "checkpoint": recognizer.name,
            "device": str(recognizer.device),
            "parameters": recognizer.n_parameters,
        },
    }


# ------------------------------------------------------------------ the correctness gate


def verify_pipeline(
    *,
    recognizer: Recognizer,
    data_cfg: dict[str, Any],
    clip_ids: list[str],
    window_seconds: float = WINDOW_SECONDS,
) -> dict[str, Any]:
    """Check that the live path predicts what the offline pipeline predicts.

    A demo can look perfect and be wrong: if the live path assembled landmarks in a different
    order, or normalized with a different anchor, the model would still emit a confident
    label on every frame and the video would look exactly the same. So each clip is run three
    ways and the results are compared.

    ``cached`` is the offline pipeline reading the ``.npy`` written in phase 1. ``live_full``
    re-decodes the video and streams it through this module's :class:`LandmarkStream` and
    :class:`ClipBuffer`, using the whole clip -- it should agree with ``cached``, and a
    disagreement is a bug in the live path. ``live_window`` uses only the last
    ``window_seconds``, which is what the demo actually sees; it is expected to be somewhat
    worse, and by how much is a finding rather than a fault.

    Args:
        recognizer: Served model.
        data_cfg: Loaded ``configs/data.yaml``.
        clip_ids: Clips to check, e.g. ``["005_009_002"]``. Use held-out signers.
        window_seconds: Window for the ``live_window`` condition.

    Returns:
        Per-clip records and the aggregate agreement and accuracy.

    Raises:
        FileNotFoundError: If a clip's video or cache entry is missing.
    """
    raw_dir = Path(data_cfg["dataset"]["raw_dir"])
    cache_dir = Path(data_cfg["dataset"]["cache_dir"])
    normalize_kwargs = config_kwargs(data_cfg)
    records: list[dict[str, Any]] = []

    for clip_id in clip_ids:
        matches = list(raw_dir.rglob(f"{clip_id}.mp4"))
        cache_path = cache_dir / f"{clip_id}.npy"
        if not matches or not cache_path.is_file():
            raise FileNotFoundError(
                f"{clip_id}: need both {raw_dir}/**/{clip_id}.mp4 and {cache_path}. "
                "Run `make data` first."
            )
        label = int(clip_id.split("_")[0]) - 1

        source = FrameSource(str(matches[0]))
        # A fresh landmarker per clip, matching offline extraction: unrelated clips must not
        # share tracking state. A real session is one continuous video and does share it,
        # which is the one difference this gate cannot check.
        stream = LandmarkStream(data_cfg)
        buffer = ClipBuffer(seconds=window_seconds)
        frames: list[np.ndarray] = []
        try:
            index = 0
            while (frame := source.read()) is not None:
                timestamp_s = index / source.fps
                landmarks = stream(frame, int(round(1000.0 * timestamp_s)))
                frames.append(landmarks)
                buffer.append(timestamp_s, landmarks)
                index += 1
        finally:
            stream.close()
            source.close()

        conditions = {
            "cached": np.load(cache_path),
            "live_full": np.stack(frames),
            "live_window": buffer.raw(),
        }
        row: dict[str, Any] = {"clip_id": clip_id, "label": label, "n_frames": len(frames)}
        for name, clip in conditions.items():
            probs = recognizer(normalize_clip(clip, aspect=source.aspect, **normalize_kwargs))
            row[name] = {
                "pred": int(np.argmax(probs)),
                "prob": float(probs.max()),
                "correct": bool(int(np.argmax(probs)) == label),
                "n_frames": int(clip.shape[0]),
            }
        records.append(row)

    def _rate(key: str, field_name: str = "correct") -> float:
        return float(np.mean([r[key][field_name] for r in records]))

    agreement = float(np.mean([r["live_full"]["pred"] == r["cached"]["pred"] for r in records]))
    # Most LSA64 clips are shorter than the window, so for them the two live conditions are
    # the same array and agree trivially. Reporting the window's cost over all clips would
    # therefore understate it by however many clips it never truncated -- so the count is
    # reported with it, and the cost is also given over the clips it actually bit on.
    truncated = [r for r in records if r["live_window"]["n_frames"] < r["live_full"]["n_frames"]]
    cost_on_truncated = (
        100.0
        * float(
            np.mean([r["live_full"]["correct"] for r in truncated])
            - np.mean([r["live_window"]["correct"] for r in truncated])
        )
        if truncated
        else float("nan")
    )
    return {
        "n_clips": len(records),
        "top1": {name: _rate(name) for name in ("cached", "live_full", "live_window")},
        "live_matches_cached": agreement,
        "n_truncated_by_window": len(truncated),
        "window_cost_points": 100.0 * (_rate("live_full") - _rate("live_window")),
        "window_cost_points_on_truncated": cost_on_truncated,
        "passed": bool(agreement >= 0.95),
        "records": records,
    }


# ------------------------------------------------------------------ the RQ4 benchmark


def build_benchmark_video(
    data_cfg: dict[str, Any],
    *,
    signer: int | None = None,
    n_clips: int = 24,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    cache_dir: str | Path = "data/cache/demo",
) -> Path:
    """Assemble one continuous webcam-like video from LSA64 clips, for the latency benchmark.

    RQ4 asks what the pipeline costs on this machine, and a number that cannot be regenerated
    from a checkout is not a result. Every LSA64 clip is under 3.4 s, which is too short to
    measure a sustained frame rate and never exercises the tracking that a real session
    depends on, so the benchmark needs a longer stream -- and it has to be built by the code,
    deterministically, rather than by hand.

    The output deliberately looks like a laptop webcam rather than like LSA64: 1280x720 at 30
    fps, not 1920x1080 at 59.94. Capture and decode are part of the frame budget, and
    measuring them at the dataset's resolution would report a cost the demo never pays.

    Args:
        data_cfg: Loaded ``configs/data.yaml``.
        signer: Which signer to draw clips from; defaults to the first configured test
            signer, so the benchmark never shows the served model a signer it trained on.
        n_clips: How many clips to concatenate.
        width: Output width.
        height: Output height.
        fps: Output frame rate; source frames are dropped evenly to reach it.
        cache_dir: Where the built video is kept.

    Returns:
        Path to the video, built on first use and reused afterwards.

    Raises:
        FileNotFoundError: If the raw clips are missing.
    """
    signer = signer if signer is not None else int(data_cfg["splits"]["test_signers"][0])
    destination = Path(cache_dir) / f"bench_{width}x{height}_{fps:.0f}fps_s{signer}_{n_clips}.mp4"
    if destination.is_file():
        return destination

    raw_dir = Path(data_cfg["dataset"]["raw_dir"])
    sources = sorted(raw_dir.rglob(f"*_{signer:03d}_*.mp4"))[:n_clips]
    if len(sources) < n_clips:
        raise FileNotFoundError(
            f"need {n_clips} clips for signer {signer} under {raw_dir}, found "
            f"{len(sources)}. Run `make data` first."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        for path in sources:
            capture = cv2.VideoCapture(str(path))
            source_fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
            keep_every = source_fps / fps
            budget = 0.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                budget += 1.0
                if budget >= keep_every:
                    budget -= keep_every
                    writer.write(cv2.resize(frame, (width, height)))
            capture.release()
    finally:
        writer.release()
    return destination


# ------------------------------------------------------------------ entry point


def _held_out_clip_ids(data_cfg: dict[str, Any], n: int) -> list[str]:
    """Pick verification clips from signers the served model never saw.

    Args:
        data_cfg: Loaded ``configs/data.yaml``.
        n: How many clips to return.

    Returns:
        Clip ids spread across classes, from the configured test signers. Deterministic:
        the stride is fixed, so the gate checks the same clips on every run.
    """
    signers = data_cfg["splits"]["test_signers"]
    cache_dir = Path(data_cfg["dataset"]["cache_dir"])
    candidates = sorted(
        path.stem for path in cache_dir.glob("*.npy") if int(path.stem.split("_")[1]) in signers
    )
    if not candidates:
        return []
    stride = max(1, len(candidates) // n)
    return candidates[::stride][:n]


def main(argv: list[str] | None = None) -> int:
    """Run the live demo, or its correctness gate, from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; 1 if the verification gate fails or the source cannot be opened.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--source",
        default="webcam",
        help="'webcam', 'webcam:N', or a path to a video file",
    )
    parser.add_argument(
        "--sink",
        default="virtualcam",
        choices=("virtualcam", "window", "file", "null"),
        help="virtualcam publishes to Zoom/Meet; null measures the pipeline alone",
    )
    parser.add_argument("--out", default="figures/demo.mp4", help="destination for --sink file")
    parser.add_argument("--device", default="cpu", help="cpu | mps | auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument("--predict-every", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument(
        "--verify",
        type=int,
        nargs="?",
        const=20,
        default=None,
        metavar="N",
        help="check the live path against the offline pipeline on N held-out clips",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="measure the pipeline on a reproducible offline stream instead of a camera",
    )
    parser.add_argument("--bench-clips", type=int, default=24)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="K=V")
    args = parser.parse_args(argv)

    model_cfg = apply_overrides(load_config(args.config), args.overrides)
    data_cfg = load_config(args.data_config)
    device = resolve_device(args.device)
    recognizer = Recognizer(args.checkpoint, model_cfg=model_cfg, device=device)

    config_snapshot = {"model": model_cfg, "data": data_cfg}

    if args.verify is not None:
        clip_ids = _held_out_clip_ids(data_cfg, args.verify)
        if not clip_ids:
            print("[verify] no cached clips for the configured test signers; run `make data`")
            return 1
        with ResultsLogger(
            "demo-verify",
            config=config_snapshot,
            seed=0,
            description=(
                "Checks that the live demo path reproduces the offline pipeline on held-out "
                "clips, and measures what the rolling window costs against a trimmed clip."
            ),
            results_dir=args.results_dir,
        ) as log:
            report = verify_pipeline(
                recognizer=recognizer,
                data_cfg=data_cfg,
                clip_ids=clip_ids,
                window_seconds=args.window_seconds,
            )
            for row in report.pop("records"):
                log.log_record(**row)
            log.set_metrics(**report)
        verdict = "PASS" if report["passed"] else "FAIL"
        print(
            f"\n[verify] {verdict}  live path reproduces the offline pipeline on "
            f"{report['live_matches_cached']:.0%} of {report['n_clips']} held-out clips"
        )
        for name in ("cached", "live_full", "live_window"):
            print(f"  {name:12s} top-1 {report['top1'][name]:.1%}")
        print(
            f"  the {args.window_seconds:.1f}s rolling window truncated "
            f"{report['n_truncated_by_window']}/{report['n_clips']} clips and costs "
            f"{report['window_cost_points']:+.1f} points overall, "
            f"{report['window_cost_points_on_truncated']:+.1f} on the clips it truncated"
        )
        return 0 if report["passed"] else 1

    if args.bench:
        # The benchmark measures the pipeline, so frames go nowhere: a window or an mp4
        # would put display cost into the RQ4 number.
        args.source = str(build_benchmark_video(data_cfg, n_clips=args.bench_clips))
        args.sink = "null"

    try:
        source = FrameSource(args.source)
    except RuntimeError as error:
        print(f"[demo] {error}")
        return 1

    stream = LandmarkStream(data_cfg, resize_width=args.resize_width)
    sink = open_sink(
        args.sink,
        size=(source.width, source.height),
        fps=source.fps,
        path=args.out,
        title="SignAdapt",
    )
    print(
        f"[demo] {args.source} {source.width}x{source.height} -> {args.sink} | "
        f"{recognizer.name} on {device} | ctrl-c to stop"
    )
    try:
        with ResultsLogger(
            "demo",
            config=config_snapshot,
            seed=0,
            description=(
                "RQ4: end-to-end latency and frame rate of the live pipeline on this "
                "machine. Not evidence about recognition quality -- see the module docstring."
            ),
            results_dir=args.results_dir,
        ) as log:
            metrics = run_demo(
                source=source,
                stream=stream,
                recognizer=recognizer,
                sink=sink,
                data_cfg=data_cfg,
                max_frames=args.max_frames,
                max_seconds=args.max_seconds,
                window_seconds=args.window_seconds,
                min_confidence=args.min_confidence,
                smoothing=args.smoothing,
                predict_every=args.predict_every,
                notice=_notice({"train_signers": data_cfg["splits"]["train_signers"]}),
            )
            log.set_metrics(**metrics)
    finally:
        sink.close()
        stream.close()
        source.close()

    print(
        f"\n[demo] {metrics['n_frames']} frames in {metrics['seconds']:.1f}s = "
        f"{metrics['fps']:.1f} fps "
        f"({'meets' if metrics['meets_target'] else 'BELOW'} the 15 fps target)"
    )
    print(
        f"  frame  p50 {metrics['frame_ms']['p50']:.1f} ms   "
        f"p95 {metrics['frame_ms']['p95']:.1f} ms"
    )
    for stage, values in metrics["stages_ms"].items():
        print(f"  {stage:10s} p50 {values['p50']:6.1f} ms   p95 {values['p95']:6.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
