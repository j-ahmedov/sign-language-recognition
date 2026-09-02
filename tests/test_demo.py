"""Phase 6: the live demo pipeline (PLAN.md sections 4, 7 and 8, week 6).

A demo is the one part of this project that looks convincing whether or not it is correct: a
caption appears over a video either way. So the checks here are about the things a viewer
cannot see -- that the live path builds the model's input exactly as training did, that the
verification gate draws only on signers the served model never saw, and that a FedPer
checkpoint is refused rather than served with a random head.

Nothing here needs a camera, a virtual camera, or MediaPipe. Frames are synthesized and
landmarks are supplied directly, which is what the split between ``realtime`` and
``virtualcam`` exists to make possible.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from signadapt.data.keypoints import N_LANDMARKS
from signadapt.data.normalize import config_kwargs, normalize_clip
from signadapt.demo import virtualcam
from signadapt.demo.realtime import (
    WINDOW_SECONDS,
    ClipBuffer,
    FrameSource,
    Prediction,
    Recognizer,
    StageClock,
    _held_out_clip_ids,
    _notice,
)
from signadapt.models.model import build_model
from signadapt.utils.config import load_config


@pytest.fixture
def data_cfg():
    return load_config("configs/data.yaml")


@pytest.fixture
def tiny_model_cfg():
    cfg = load_config("configs/model.yaml")
    cfg["encoder"] |= {"d_model": 16, "n_layers": 1, "n_heads": 2, "ff_dim": 16, "max_len": 8}
    cfg["head"] |= {"in_dim": 16, "n_classes": 5}
    return cfg


def _landmark_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.normal(size=(N_LANDMARKS, 4)).astype(np.float32)
    frame[:, 3] = 1.0
    return frame


def _write_video(path: Path, *, n_frames: int = 12, size: tuple[int, int] = (64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), i * 8 % 255, dtype=np.uint8))
    writer.release()
    return path


# ------------------------------------------------------------------ the rolling window


def test_clip_buffer_drops_by_time_not_by_frame_count():
    """The window is a duration, so it must hold different frame counts at different rates."""
    fast, slow = ClipBuffer(seconds=1.0), ClipBuffer(seconds=1.0)
    for i in range(120):
        fast.append(i / 60.0, _landmark_frame(i))
    for i in range(60):
        slow.append(i / 30.0, _landmark_frame(i))
    assert len(fast) == pytest.approx(60, abs=2)
    assert len(slow) == pytest.approx(30, abs=2)


def test_clip_buffer_keeps_the_newest_frames_in_order():
    buffer = ClipBuffer(seconds=0.5)
    frames = [_landmark_frame(i) for i in range(40)]
    for i, frame in enumerate(frames):
        buffer.append(i / 30.0, frame)
    window = buffer.raw()
    assert np.array_equal(window, np.stack(frames[-window.shape[0] :]))


def test_clip_buffer_waits_before_predicting():
    buffer = ClipBuffer(seconds=2.0, min_frames=12)
    for i in range(11):
        buffer.append(i / 30.0, _landmark_frame(i))
        assert not buffer.ready
    buffer.append(11 / 30.0, _landmark_frame(11))
    assert buffer.ready


def test_the_live_window_normalizes_to_exactly_what_training_would_produce(data_cfg):
    """The cardinal check: the demo must not build the model's input its own way.

    A live path that assembled or normalized frames differently would still emit confident
    labels, and the demo video would look identical. Here the same frames go through the
    buffer and through the offline call directly, and the tensors must be bitwise equal.
    """
    frames = [_landmark_frame(i) for i in range(40)]
    buffer = ClipBuffer(seconds=10.0)
    for i, frame in enumerate(frames):
        buffer.append(i / 30.0, frame)

    kwargs = config_kwargs(data_cfg)
    live = normalize_clip(buffer.raw(), aspect=16 / 9, **kwargs)
    offline = normalize_clip(np.stack(frames), aspect=16 / 9, **kwargs)
    assert np.array_equal(live, offline)


def test_window_default_covers_a_typical_lsa64_sign():
    """LSA64's median clip is 1.30 s and its p90 is 1.97 s; the window sits between them."""
    assert 1.30 <= WINDOW_SECONDS <= 1.97


# ------------------------------------------------------------------ timing


def test_stage_clock_reports_percentiles_not_just_a_mean():
    clock = StageClock()
    for _ in range(20):
        with clock.time("model"):
            pass
    summary = clock.summary()["model"]
    assert summary["n"] == 20
    assert summary["p50"] <= summary["p95"] <= summary["max"]
    assert set(clock.latest()) == {"model_ms"}


def test_stage_clock_records_a_stage_that_raised():
    clock = StageClock()
    with pytest.raises(ValueError), clock.time("model"):
        raise ValueError("boom")
    assert clock.summary()["model"]["n"] == 1


# ------------------------------------------------------------------ sources


def test_frame_source_reads_a_video_file(tmp_path):
    source = FrameSource(str(_write_video(tmp_path / "clip.mp4")))
    try:
        assert not source.is_live
        assert source.aspect == pytest.approx(64 / 48)
        # Not `iter(source.read, None)`: that compares with ==, and an ndarray == None is
        # an elementwise array, not a bool.
        count = 0
        while source.read() is not None:
            count += 1
        assert count == 12
    finally:
        source.close()


def test_frame_source_names_the_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="no such video file"):
        FrameSource(str(tmp_path / "absent.mp4"))


# ------------------------------------------------------------------ the served model


def _save_checkpoint(path: Path, cfg: dict, *, encoder_only: bool) -> Path:
    state = build_model(cfg).state_dict()
    if encoder_only:
        state = {k: v for k, v in state.items() if k.startswith("encoder.")}
    torch.save({"experiment": "test", "seed": 0, "state": state}, path)
    return path


def test_recognizer_refuses_an_encoder_only_checkpoint(tmp_path, tiny_model_cfg):
    """FedPer's checkpoints have no head on purpose; serving one would invent a classifier.

    ``build_model`` would happily supply a randomly initialized head, and the demo would then
    caption every frame with a uniformly random sign at high confidence after smoothing.
    """
    path = _save_checkpoint(tmp_path / "fedper.pt", tiny_model_cfg, encoder_only=True)
    with pytest.raises(ValueError, match="holds no head"):
        Recognizer(path, model_cfg=tiny_model_cfg, device=torch.device("cpu"))


def test_recognizer_names_the_command_that_makes_a_checkpoint(tmp_path, tiny_model_cfg):
    with pytest.raises(FileNotFoundError, match="make federated"):
        Recognizer(tmp_path / "absent.pt", model_cfg=tiny_model_cfg, device=torch.device("cpu"))


def test_recognizer_scores_a_clip_and_ranks_it(tmp_path, tiny_model_cfg):
    path = _save_checkpoint(tmp_path / "fedavg.pt", tiny_model_cfg, encoder_only=False)
    names = ("a", "b", "c", "d", "e")
    recognizer = Recognizer(
        path, model_cfg=tiny_model_cfg, device=torch.device("cpu"), class_names=names
    )
    clip = np.zeros((8, N_LANDMARKS, 4), dtype=np.float32)
    probs = recognizer(clip)
    assert probs.shape == (5,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    top = recognizer.top_k(probs, k=3)
    assert [name for name, _ in top] == [names[i] for i in np.argsort(probs)[::-1][:3]]
    assert [p for _, p in top] == sorted((p for _, p in top), reverse=True)


# ------------------------------------------------------------------ the verification gate


def test_verification_clips_come_only_from_held_out_signers(data_cfg):
    """The demo's gate is still a signer-leakage surface: it must not score training signers."""
    held_out = set(data_cfg["splits"]["test_signers"])
    clip_ids = _held_out_clip_ids(data_cfg, 30)
    if not clip_ids:  # pragma: no cover - only when the cache is absent
        pytest.skip("no extracted keypoint cache")
    assert {int(cid.split("_")[1]) for cid in clip_ids} <= held_out
    assert len(clip_ids) == len(set(clip_ids))


def test_verification_clip_choice_is_deterministic(data_cfg):
    if not _held_out_clip_ids(data_cfg, 5):  # pragma: no cover
        pytest.skip("no extracted keypoint cache")
    assert _held_out_clip_ids(data_cfg, 12) == _held_out_clip_ids(data_cfg, 12)


# ------------------------------------------------------------------ the overlay


def _frame() -> np.ndarray:
    return np.full((360, 640, 3), 120, dtype=np.uint8)


def test_render_overlay_does_not_modify_its_input():
    frame = _frame()
    before = frame.copy()
    out = virtualcam.render_overlay(
        frame,
        labels=[("Green", 0.9)],
        telemetry={"fps": 30.0, "latency_ms": 14.0},
        model_name="test",
        notice="illustrative",
        confident=True,
    )
    assert np.array_equal(frame, before)
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_render_overlay_handles_the_waiting_state():
    """The first frames of every session have no prediction; that must not raise."""
    out = virtualcam.render_overlay(
        _frame(),
        labels=[],
        telemetry={"fps": float("nan"), "latency_ms": float("nan")},
        model_name="test",
        notice="illustrative",
        confident=False,
    )
    assert out.shape == (360, 640, 3)


def test_right_aligned_text_stays_inside_the_frame():
    """The model name is drawn from the right margin; putText positions the left end."""
    frame = _frame()
    virtualcam._put_right(  # noqa: SLF001
        frame,
        "a considerably longer model name than fits",
        right=frame.shape[1] - 20,
        baseline=30,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale=0.5,
        colour=(255, 255, 255),
    )
    ink = np.argwhere(frame[:, :, 0] != 120)
    assert ink.size > 0, "nothing was drawn"
    assert ink[:, 1].max() <= frame.shape[1] - 20, "text ran past the right margin"
    assert ink[:, 1].min() >= 0


def test_every_string_the_overlay_draws_is_ascii(data_cfg):
    """OpenCV's Hershey fonts draw '?' for anything else, which reached a rendered frame once."""
    caption = Prediction(top=(), confident=False).label
    notice = _notice({"train_signers": data_cfg["splits"]["train_signers"]})
    for text in (caption, notice):
        assert text.isascii(), text


# ------------------------------------------------------------------ sinks


def test_null_sink_counts_and_never_stops():
    sink = virtualcam.open_sink("null")
    sink.send(_frame())
    assert sink.n_frames == 1
    assert not sink.should_stop()
    sink.close()


def test_file_sink_writes_a_playable_video(tmp_path):
    sink = virtualcam.open_sink("file", path=tmp_path / "out.mp4", fps=30.0, size=(640, 360))
    for _ in range(6):
        sink.send(_frame())
    sink.close()
    capture = cv2.VideoCapture(str(tmp_path / "out.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    finally:
        capture.release()


def test_unknown_sink_lists_the_valid_ones():
    with pytest.raises(ValueError, match="virtualcam"):
        virtualcam.open_sink("zoom")
