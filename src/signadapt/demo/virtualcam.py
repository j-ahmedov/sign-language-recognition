"""Caption overlay and output sinks for the live demo (PLAN.md sections 4 and 7).

The demo's last two stages -- draw a caption on the frame, hand the frame to something that
displays it -- are separated from the recognition pipeline here so that both can run without
a camera or a virtual camera device. ``render_overlay`` is a pure function on a BGR array,
and every sink implements the same three methods, so the pipeline can be measured on a video
file in CI and driven by a webcam into Zoom on a laptop with no change to the code between
them.

Four sinks are provided. ``virtualcam`` is the one PLAN.md section 8 asks for: it publishes
frames to a virtual camera device that Zoom, Meet and Teams see as an ordinary webcam, with
no SDK, account or app review on any of them. ``window`` opens a local preview, ``file``
writes an mp4 for a slide deck, and ``null`` discards frames so a benchmark measures the
pipeline rather than the display.

The overlay deliberately shows more than the prediction. A demo that displays only its top
guess invites the reader to judge it as a product; this one keeps the model's identity, its
training signers and its live latency on screen, because what the demo is evidence for is
RQ4 -- the on-device cost -- and not recognition quality on an unseen signer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

__all__ = [
    "FileSink",
    "NullSink",
    "Sink",
    "VirtualCamSink",
    "WindowSink",
    "open_sink",
    "render_overlay",
]

# BGR, matching the figure palette in signadapt.figures so a slide and a chart agree.
_INK = (11, 11, 11)
_SURFACE = (251, 252, 252)
_MUTED = (127, 134, 135)
_ACCENT = (214, 120, 42)  # blue  #2a78d6
_WARM = (52, 104, 235)  # orange #eb6834

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_THIN = cv2.FONT_HERSHEY_SIMPLEX


def _band(frame: np.ndarray, y0: int, y1: int, *, alpha: float = 0.72) -> None:
    """Darken a horizontal band in place so text stays legible over any background.

    Args:
        frame: BGR image, modified in place.
        y0: Top row.
        y1: Bottom row, exclusive.
        alpha: Opacity of the band.
    """
    y0, y1 = max(0, y0), min(frame.shape[0], y1)
    if y1 <= y0:
        return
    region = frame[y0:y1]
    region[:] = cv2.addWeighted(region, 1.0 - alpha, np.zeros_like(region), 0.0, 0.0)


def _put_right(
    frame: np.ndarray,
    text: str,
    *,
    right: int,
    baseline: int,
    font: int,
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    """Draw right-aligned text.

    ``cv2.putText`` positions the *left* end of the string, so passing a right margin as the
    x coordinate silently runs the text off the frame -- which is what the first version of
    this overlay did to the model name. The width has to be measured first.

    Args:
        frame: BGR image, modified in place.
        text: The string; OpenCV's Hershey fonts are ASCII-only, so non-ASCII is dropped.
        right: x coordinate the text should end at.
        baseline: y coordinate of the text baseline.
        font: An OpenCV font constant.
        scale: Font scale.
        colour: BGR colour.
    """
    ascii_text = text.encode("ascii", "replace").decode("ascii")
    (width, _), _ = cv2.getTextSize(ascii_text, font, scale, 1)
    cv2.putText(frame, ascii_text, (right - width, baseline), font, scale, colour, 1, cv2.LINE_AA)


def render_overlay(
    frame: np.ndarray,
    *,
    labels: list[tuple[str, float]],
    telemetry: dict[str, float],
    model_name: str,
    notice: str,
    confident: bool,
) -> np.ndarray:
    """Draw the caption, the runner-up predictions and the latency HUD onto a frame.

    Args:
        frame: BGR image; a copy is drawn on, the input is not modified.
        labels: ``(sign name, probability)`` in descending order; the first is the caption.
            An empty list renders the waiting state, which is what the first two seconds of
            any session look like while the clip buffer fills.
        telemetry: Numbers for the HUD -- ``fps``, ``latency_ms``, and optionally the
            per-stage ``capture_ms``, ``landmarks_ms``, ``normalize_ms``, ``model_ms``.
        model_name: Which checkpoint is being served, shown so a screenshot is traceable.
        notice: The standing caveat, e.g. which signers the model saw. Always drawn.
        confident: Whether the top prediction cleared the threshold. When it has not, the
            caption renders as a dash rather than as the model's best guess: a caption that
            always shows *something* reads as a recognition even when the model is only
            picking the least unlikely of 64 classes it must choose between. The Hershey
            fonts OpenCV ships are ASCII-only, so every string drawn here is ASCII.

    Returns:
        A new BGR image the same size as ``frame``.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    scale = width / 1280.0

    # --- caption band -------------------------------------------------------------
    band_height = int(118 * scale)
    _band(canvas, height - band_height, height)
    # ASCII only: OpenCV's Hershey fonts have no glyph for an em dash and draw "???".
    caption = labels[0][0] if (labels and confident) else "..."
    cv2.putText(
        canvas,
        caption,
        (int(28 * scale), height - int(58 * scale)),
        _FONT,
        1.5 * scale,
        _SURFACE,
        max(1, int(2 * scale)),
        cv2.LINE_AA,
    )
    if labels:
        runners = "   ".join(f"{name} {prob:.0%}" for name, prob in labels[1:4])
        cv2.putText(
            canvas,
            runners,
            (int(30 * scale), height - int(26 * scale)),
            _FONT_THIN,
            0.52 * scale,
            _MUTED,
            1,
            cv2.LINE_AA,
        )
        # A confidence bar for the top class: the number the caption is gated on, shown.
        bar_width = int(260 * scale)
        x0, y0 = width - bar_width - int(28 * scale), height - int(46 * scale)
        cv2.rectangle(canvas, (x0, y0), (x0 + bar_width, y0 + int(8 * scale)), (60, 60, 60), -1)
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + int(bar_width * float(labels[0][1])), y0 + int(8 * scale)),
            _ACCENT if confident else _MUTED,
            -1,
        )

    # --- latency HUD --------------------------------------------------------------
    hud_height = int(64 * scale)
    _band(canvas, 0, hud_height, alpha=0.62)
    fps = telemetry.get("fps", float("nan"))
    cv2.putText(
        canvas,
        f"{fps:4.1f} fps   {telemetry.get('latency_ms', float('nan')):5.1f} ms end-to-end",
        (int(28 * scale), int(28 * scale)),
        _FONT,
        0.62 * scale,
        _SURFACE if fps >= 15.0 else _WARM,
        1,
        cv2.LINE_AA,
    )
    stages = [
        (key, telemetry[key])
        for key in ("capture_ms", "landmarks_ms", "normalize_ms", "model_ms")
        if key in telemetry
    ]
    if stages:
        detail = "  ".join(f"{k.removesuffix('_ms')} {v:.1f}" for k, v in stages)
        cv2.putText(
            canvas,
            detail,
            (int(28 * scale), int(50 * scale)),
            _FONT_THIN,
            0.46 * scale,
            _MUTED,
            1,
            cv2.LINE_AA,
        )
    _put_right(
        canvas,
        model_name,
        right=width - int(28 * scale),
        baseline=int(28 * scale),
        font=_FONT_THIN,
        scale=0.46 * scale,
        colour=_SURFACE,
    )
    _put_right(
        canvas,
        notice,
        right=width - int(28 * scale),
        baseline=int(50 * scale),
        font=_FONT_THIN,
        scale=0.42 * scale,
        colour=_MUTED,
    )
    return canvas


class Sink(Protocol):
    """Somewhere a rendered frame goes: a virtual camera, a window, a file, or nowhere."""

    def send(self, frame: np.ndarray) -> None:
        """Publish one BGR frame."""

    def should_stop(self) -> bool:
        """Return whether the viewer asked to end the session."""

    def close(self) -> None:
        """Release the device or file handle."""


class NullSink:
    """Discards frames. The default for benchmarking, so display cost is excluded."""

    def __init__(self, **_: Any) -> None:
        """Accept and ignore the common sink keyword arguments."""
        self.n_frames = 0

    def send(self, frame: np.ndarray) -> None:
        """Count the frame and drop it.

        Args:
            frame: BGR image.
        """
        del frame
        self.n_frames += 1

    def should_stop(self) -> bool:
        """Never stops on its own."""
        return False

    def close(self) -> None:
        """Nothing to release."""


class WindowSink:
    """Shows frames in a local OpenCV window; q or Esc ends the session."""

    def __init__(self, *, title: str = "SignAdapt", **_: Any) -> None:
        """Open a preview window.

        Args:
            title: Window title.
        """
        self.title = title
        self._stop = False

    def send(self, frame: np.ndarray) -> None:
        """Display one frame and poll the keyboard.

        Args:
            frame: BGR image.
        """
        cv2.imshow(self.title, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            self._stop = True

    def should_stop(self) -> bool:
        """Return whether q or Esc was pressed."""
        return self._stop

    def close(self) -> None:
        """Close the window."""
        cv2.destroyWindow(self.title)


class FileSink:
    """Writes frames to an mp4, for a slide deck or a regression artefact."""

    def __init__(
        self, *, path: str | Path, fps: float = 30.0, size: tuple[int, int], **_: Any
    ) -> None:
        """Open a video writer.

        Args:
            path: Destination file.
            fps: Frame rate written into the container.
            size: ``(width, height)``.

        Raises:
            RuntimeError: If the writer could not be opened.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"cannot open a video writer at {self.path}")

    def send(self, frame: np.ndarray) -> None:
        """Append one frame.

        Args:
            frame: BGR image.
        """
        self._writer.write(frame)

    def should_stop(self) -> bool:
        """Never stops on its own."""
        return False

    def close(self) -> None:
        """Finalize the file."""
        self._writer.release()


class VirtualCamSink:
    """Publishes frames to a virtual camera that Zoom, Meet and Teams see as a webcam.

    pyvirtualcam wants RGB; OpenCV works in BGR, and getting that backwards produces a demo
    where the presenter is blue, so the conversion happens here rather than at the callsite.
    """

    def __init__(self, *, size: tuple[int, int], fps: float = 30.0, **_: Any) -> None:
        """Open the virtual camera device.

        Args:
            size: ``(width, height)``.
            fps: Frame rate advertised to the consuming application.

        Raises:
            RuntimeError: If no virtual camera backend is available, with the macOS fix.
        """
        try:
            import pyvirtualcam
        except ImportError as error:  # pragma: no cover - optional extra
            raise RuntimeError(
                "pyvirtualcam is not installed; run `make setup` or `pip install -e '.[demo]'`"
            ) from error
        try:
            self._cam = pyvirtualcam.Camera(width=size[0], height=size[1], fps=int(fps))
        except Exception as error:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "no virtual camera backend is available. On macOS, install OBS and click "
                "'Start Virtual Camera' once so the device exists; then rerun. "
                f"(pyvirtualcam said: {error})"
            ) from error

    @property
    def device(self) -> str:
        """Return the backing device name, for the results file."""
        return str(self._cam.device)

    def send(self, frame: np.ndarray) -> None:
        """Publish one frame, converting BGR to RGB.

        Args:
            frame: BGR image.
        """
        self._cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def should_stop(self) -> bool:
        """Never stops on its own; the session ends on ctrl-c or the frame budget."""
        return False

    def close(self) -> None:
        """Release the device."""
        self._cam.close()


_SINKS = {
    "null": NullSink,
    "window": WindowSink,
    "file": FileSink,
    "virtualcam": VirtualCamSink,
}


def open_sink(name: str, **kwargs: Any) -> Sink:
    """Construct a sink by name.

    Args:
        name: One of ``null``, ``window``, ``file``, ``virtualcam``.
        **kwargs: Passed to the sink; unrecognized keys are ignored, so one callsite can
            build any of them.

    Returns:
        The sink.

    Raises:
        ValueError: On an unknown name.
    """
    if name not in _SINKS:
        raise ValueError(f"unknown sink {name!r}; expected one of {sorted(_SINKS)}")
    return _SINKS[name](**kwargs)
