"""JSON results logging.

Every experiment writes exactly one JSON file per (experiment, tag, seed) run. Figures are
regenerated from these files and never from numbers typed by hand (PLAN.md section 7), so a
result file has to be self-describing: it carries its own config snapshot, seed, environment
and git commit alongside the metrics.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

__all__ = ["SCHEMA_VERSION", "ResultsLogger", "load_result", "load_results", "environment_info"]

SCHEMA_VERSION = 1


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars and arrays."""

    def default(self, o: Any) -> Any:
        """Convert numpy types to plain python, delegating everything else."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _git_info() -> dict[str, Any]:
    """Return the current commit and dirty flag, or nulls when git is unavailable."""

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def environment_info() -> dict[str, Any]:
    """Capture the machine and library versions a result was produced on.

    Returns:
        A dictionary with python/platform/library versions and git state. Library versions
        are best-effort: a missing optional import is recorded as ``None`` rather than raised.
    """
    versions: dict[str, Any] = {"numpy": np.__version__}
    for name in ("torch", "flwr", "mediapipe", "cv2"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None

    torch_device = None
    try:
        import torch

        if torch.cuda.is_available():
            torch_device = "cuda"
        elif torch.backends.mps.is_available():
            torch_device = "mps"
        else:
            torch_device = "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch_device": torch_device,
        "versions": versions,
        "git": _git_info(),
    }


def _slugify(text: str) -> str:
    """Reduce a string to a filename-safe slug."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()


class ResultsLogger:
    """Accumulates the records and metrics of one run and writes them atomically to JSON.

    Use as a context manager so that the duration and the completion status are recorded
    even when the run raises::

        with ResultsLogger("E2", config=cfg, seed=0) as log:
            for epoch in range(n):
                log.log_record(epoch=epoch, val_acc=acc)
            log.set_metrics(top1=..., top5=...)

    Attributes:
        experiment: Experiment id from PLAN.md section 6, e.g. ``"E2"``.
        seed: The seed the run was executed with.
        path: Destination file, resolved at construction time.
    """

    def __init__(
        self,
        experiment: str,
        *,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        tag: str | None = None,
        description: str = "",
        results_dir: str | Path = "results",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Create a logger and reserve its output path.

        Args:
            experiment: Experiment id, e.g. ``"E1"``, ``"E5"``, ``"sanity"``.
            config: Config snapshot stored verbatim in the result file.
            seed: Seed of the run.
            tag: Optional discriminator, e.g. ``"loso-signer03"`` or ``"k5"``.
            description: One line explaining what the run is, for a reader of the JSON.
            results_dir: Directory the JSON is written to; created if missing.
            extra: Any additional top-level fields to store.
        """
        self.experiment = experiment
        self.seed = seed
        self.tag = tag
        self._description = description
        self._config = config or {}
        self._extra = extra or {}
        self._records: list[dict[str, Any]] = []
        self._metrics: dict[str, Any] = {}
        self._started = time.time()
        self._status = "running"
        self._error: str | None = None

        parts = [_slugify(experiment)]
        if tag:
            parts.append(_slugify(tag))
        if seed is not None:
            parts.append(f"seed{seed}")
        parts.append(datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
        self.dir = Path(results_dir)
        self.path = self.dir / ("_".join(parts) + ".json")

    def log_record(self, **fields: Any) -> None:
        """Append one row to the run's record list (typically one epoch or one round).

        Args:
            **fields: Arbitrary JSON-serializable values, e.g. ``epoch=3, val_acc=0.71``.
        """
        self._records.append(dict(fields))

    def set_metrics(self, **metrics: Any) -> None:
        """Set or update the run's headline metrics.

        Args:
            **metrics: Final values, e.g. ``top1=0.82, top5=0.96``.
        """
        self._metrics.update(metrics)

    def to_dict(self) -> dict[str, Any]:
        """Build the full result document.

        Returns:
            The dictionary that will be serialized to JSON.
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment": self.experiment,
            "tag": self.tag,
            "description": self._description,
            "seed": self.seed,
            "status": self._status,
            "error": self._error,
            "created_utc": datetime.fromtimestamp(self._started, UTC).isoformat(),
            "duration_s": round(time.time() - self._started, 3),
            "environment": environment_info(),
            "config": self._config,
            "metrics": self._metrics,
            "records": self._records,
            **self._extra,
        }

    def save(self) -> Path:
        """Write the result document atomically, leaving the status untouched.

        Safe to call repeatedly as a mid-run checkpoint: the status stays ``"running"``
        until :meth:`finish` or the context manager marks the run terminal, and
        :func:`load_results` skips non-``"ok"`` runs so a half-written run cannot reach a
        figure.

        Returns:
            The path written.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=False, cls=_NumpyEncoder)
            handle.write("\n")
        os.replace(tmp, self.path)
        return self.path

    def finish(self) -> Path:
        """Mark the run as completed successfully and write it.

        For code that does not use the context manager. Prefer ``with ResultsLogger(...)``,
        which also records failures.

        Returns:
            The path written.
        """
        self._status = "ok"
        return self.save()

    def __enter__(self) -> ResultsLogger:
        """Enter the context; nothing is written until exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Record the outcome and save. Never suppresses the exception."""
        if exc is None:
            self._status = "ok"
        else:
            self._status = "failed"
            self._error = f"{exc_type.__name__ if exc_type else 'Error'}: {exc}"
        self.save()
        return False


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a single result JSON file.

    Args:
        path: Path to the file.

    Returns:
        The parsed result document.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results(
    results_dir: str | Path = "results",
    *,
    experiment: str | None = None,
    status: str | None = "ok",
) -> list[dict[str, Any]]:
    """Load every result file in a directory, optionally filtered.

    Args:
        results_dir: Directory to scan (non-recursive, ``*.json``).
        experiment: Keep only results with this experiment id.
        status: Keep only results with this status; pass ``None`` to keep all. Defaults to
            ``"ok"`` so that a crashed run cannot silently end up in a figure.

    Returns:
        Result documents sorted by ``created_utc``.
    """
    docs: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        doc = load_result(path)
        if experiment is not None and doc.get("experiment") != experiment:
            continue
        if status is not None and doc.get("status") != status:
            continue
        doc["_path"] = str(path)
        docs.append(doc)
    return sorted(docs, key=lambda d: d.get("created_utc", ""))
