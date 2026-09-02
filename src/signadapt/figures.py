"""Regenerate every figure in the thesis from the committed ``results/*.json``.

PLAN.md section 7 makes this module the only place a number reaches a chart: results are
written as JSON by the experiments, JSON is committed, and figures are produced *from* the
JSON by ``make figures``. Nothing here contains a measured value as a literal, so "redo that
chart with error bars" is one command rather than one evening -- and a figure can never drift
away from the run it claims to describe.

Two consequences are deliberate:

* **A missing experiment is an error, not a gap in a chart.** Every loader raises when the
  results it needs are absent. A partial adaptation curve that silently omits E4 would be
  read as "E4 was not run" by a reader who cannot tell it apart from "E4 does not appear
  here", so the script refuses to draw it at all.
* **Runs that did not finish are excluded.** :func:`~signadapt.utils.results.load_results`
  keeps only ``status == "ok"``, so a crashed or half-written run cannot reach a figure.

Alongside the images, ``figures/summary.json`` records every number that was drawn, keyed by
figure, with the result files it came from. That file is what the thesis text and the README
tables quote, so prose and figures cannot disagree.

Colours follow a categorical palette validated for colour-vision deficiency (all-pairs
Delta E >= 8 in OKLab, normal-vision >= 15) rather than matplotlib's default cycle, and each
method keeps its hue across every figure -- E5 is the same blue in the curve, the paired
differences and the spread plot.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

# Figures are written to disk, never displayed; select the non-interactive backend before
# pyplot is imported so `make figures` works over ssh and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from signadapt.utils.metrics import mean_std  # noqa: E402
from signadapt.utils.results import load_results  # noqa: E402

__all__ = [
    "METHOD_COLOR",
    "Fold",
    "adaptation_curve",
    "by_k",
    "latency_budget",
    "load_demo",
    "load_folds",
    "matched_budget_check",
    "main",
    "paired_delta",
]

# ------------------------------------------------------------------ palette and style

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#87867f"
GRID = "#e6e5e1"

#: One hue per method, fixed across every figure: colour follows the entity, never its rank
#: or its position in a legend. Validated all-pairs on the light surface -- worst CVD
#: separation 9.2 (aqua/orange, deutan), worst normal-vision separation 16.3 (violet/blue).
METHOD_COLOR = {
    "E3": "#4a3aa7",  # violet -- local-only, the null hypothesis
    "E4": "#eb6834",  # orange -- FedAvg + fine-tune
    "E5": "#2a78d6",  # blue   -- FedPer, the proposed method
    "E6": "#1baf7a",  # aqua   -- centralized pretrain + head
}

#: Diverging pair for signed differences: warm and cool poles that read as opposite, with a
#: neutral midpoint. Never a hue at zero.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"

METHOD_LABEL = {
    "E3": "E3  local-only",
    "E4": "E4  FedAvg + fine-tune",
    "E5": "E5  FedPer (proposed)",
    "E6": "E6  centralized + head",
}

#: Draw order, so a legend and a panel row never disagree about sequence.
METHOD_ORDER = ("E3", "E4", "E5", "E6")

REFERENCE_LABEL = {
    "E1": "E1  centralized, signer-dependent (ceiling)",
    "E2": "E2  centralized, signer-independent",
}


def apply_style() -> None:
    """Set the rcParams every figure in this module shares.

    Thin marks, hairline solid grid one shade off the surface, and text in ink tokens rather
    than in a series colour. Called by each figure function, so importing this module has no
    global side effect on a caller's own plots.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "figure.dpi": 120,
        }
    )


def _despine(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    """Remove the top and right spines and optionally soften the remaining two.

    Args:
        ax: Axes to modify.
        left: Keep the left spine.
        bottom: Keep the bottom spine.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def _declutter(
    values: list[float], *, min_gap: float, limit: tuple[float, float] | None = None
) -> list[float]:
    """Nudge label positions apart so direct labels never overlap.

    Direct labels are required here rather than optional: two of the four method hues sit
    below 3:1 contrast against the surface, and the palette's relief rule says a chart using
    them must carry visible labels. Labels that print on top of each other would not satisfy
    it, and at k=4 three of the four methods land within two points of one another.

    Args:
        values: Desired label positions, in data coordinates, in any order.
        min_gap: Minimum separation to enforce, in the same units.
        limit: Optional ``(low, high)`` the whole set is shifted back into after spreading,
            so labels cannot be pushed off the axes. Shifting moves every label by the same
            amount, which preserves order but detaches a label from its point -- callers that
            pass a limit should draw a leader line.

    Returns:
        Adjusted positions, in the same order as ``values``.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = list(values)
    for position, index in enumerate(order):
        if position == 0:
            continue
        previous = adjusted[order[position - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    if limit is not None:
        low, high = limit
        overflow = max(adjusted) - high
        if overflow > 0:
            adjusted = [v - overflow for v in adjusted]
        underflow = low - min(adjusted)
        if underflow > 0:
            adjusted = [v + underflow for v in adjusted]
    return adjusted


# ------------------------------------------------------------------ loading


@dataclass(frozen=True)
class Fold:
    """One measurement: one method, on one held-out signer, at one k, under one seed.

    This is the unit everything downstream aggregates over. Keeping ``seed`` and ``signer``
    separate rather than pre-averaging is what makes the paired comparisons in
    :func:`paired_delta` possible: two methods are only comparable on folds they share.

    Attributes:
        method: Experiment id, ``"E3"`` to ``"E6"``.
        seed: The seed of the run this came from.
        signer: The held-out signer.
        k: Labelled clips per class the signer supplied.
        top1: Top-1 accuracy on that signer's query set, in [0, 1].
        top5: Top-5 accuracy on the same query set.
    """

    method: str
    seed: int
    signer: int
    k: int
    top1: float
    top5: float

    @property
    def key(self) -> tuple[int, int, int]:
        """Return the ``(seed, signer, k)`` identity a paired comparison matches on."""
        return (self.seed, self.signer, self.k)


def load_folds(
    results_dir: str | Path = "results", *, methods: tuple[str, ...] = METHOD_ORDER
) -> dict[str, tuple[Fold, ...]]:
    """Load every per-fold record of the k-sweep experiments.

    Args:
        results_dir: Directory holding the committed result files.
        methods: Which sweep experiments to admit. The default is the four methods the
            figures draw, which is what keeps a diagnostic run such as E6M out of every
            chart without anyone having to remember to exclude it.

    Returns:
        ``{method: folds}`` for every sweep method found, each tuple sorted by
        ``(seed, signer, k)`` so the output does not depend on filesystem order.

    Raises:
        ValueError: If a method's runs disagree about which k values were swept, which would
            make a curve compare different x-axes against each other.
    """
    collected: dict[str, list[Fold]] = defaultdict(list)
    k_values: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    sources: dict[str, list[str]] = defaultdict(list)

    for doc in load_results(results_dir):
        method = doc.get("experiment", "")
        if method not in methods:
            continue
        seed = int(doc["seed"])
        sources[method].append(doc["_path"])
        k_values[method].add(tuple(sorted(int(k) for k in doc.get("k_values", []))))
        for record in doc["records"]:
            collected[method].append(
                Fold(
                    method=method,
                    seed=seed,
                    signer=int(record["signer"]),
                    k=int(record["k"]),
                    top1=float(record["top1"]),
                    top5=float(record["top5"]),
                )
            )

    for method, sweeps in k_values.items():
        if len(sweeps) > 1:
            raise ValueError(
                f"{method} was run over more than one k grid ({sorted(sweeps)}) in "
                f"{sources[method]}; a curve over them would put different x-axes on one "
                "plot. Re-run the odd one out or move it out of the results directory."
            )
    return {m: tuple(sorted(f, key=lambda x: x.key)) for m, f in collected.items()}


def require_methods(folds: dict[str, tuple[Fold, ...]], needed: tuple[str, ...]) -> None:
    """Raise unless every named method has results.

    Args:
        folds: Loaded folds keyed by method.
        needed: Methods the caller is about to draw.

    Raises:
        FileNotFoundError: If any is missing, naming the make target that produces it.
    """
    missing = [m for m in needed if not folds.get(m)]
    if missing:
        raise FileNotFoundError(
            f"no finished results for {', '.join(missing)}. A figure drawn without them "
            f"would read as 'this method was not measured'. Run `make sweep METHODS=\""
            f'{" ".join(missing)}"` first.'
        )


def load_centralized(results_dir: str | Path = "results") -> dict[str, dict[str, Any]]:
    """Summarize the centralized baselines E1 and E2 across their seeds.

    Args:
        results_dir: Directory holding the committed result files.

    Returns:
        ``{experiment: {"top1": mean_std, "top5": mean_std, "per_signer": {...}, "seeds":
        [...], "sources": [...]}}`` for whichever of E1 and E2 are present.
    """
    summary: dict[str, dict[str, Any]] = {}
    for experiment in ("E1", "E2"):
        docs = load_results(results_dir, experiment=experiment)
        if not docs:
            continue
        per_signer: dict[int, list[float]] = defaultdict(list)
        for doc in docs:
            for signer, value in doc["metrics"].get("per_signer", {}).items():
                per_signer[int(signer)].append(float(value))
        summary[experiment] = {
            "runs": [{"seed": d["seed"], "top1": float(d["metrics"]["top1"])} for d in docs],
            "top1": mean_std([d["metrics"]["top1"] for d in docs]),
            "top5": mean_std([d["metrics"]["top5"] for d in docs]),
            "per_signer": {s: mean_std(v) for s, v in sorted(per_signer.items())},
            "test_signers": sorted(per_signer),
            "seeds": [d["seed"] for d in docs],
            "sources": [d["_path"] for d in docs],
        }
    return summary


def load_federated(results_dir: str | Path = "results") -> dict[str, Any]:
    """Load the phase-3 federated runs: the IID correctness check and the FedAvg pretrainings.

    Args:
        results_dir: Directory holding the committed result files.

    Returns:
        ``{"iid": [...], "signer": [...]}`` where each entry carries the run's per-round
        records and its final metrics. Either list may be empty.
    """
    out: dict[str, Any] = {}
    for key, experiment in (("iid", "fedavg-iid-check"), ("signer", "fedavg-pretrain")):
        out[key] = [
            {
                "seed": doc["seed"],
                "rounds": [int(r["round"]) for r in doc["records"]],
                "train_top1": [float(r["fit"]["train_top1"]) for r in doc["records"]],
                "top1": float(doc["metrics"]["top1"]),
                "n_rounds": int(doc["metrics"]["n_rounds"]),
                "n_shared_tensors": int(doc["metrics"]["n_shared_tensors"]),
                "n_clients": int(doc["partition"]["n_clients"]),
                "tolerance": doc["config"].get("fl", {}).get("checks", {}).get("iid_tolerance"),
                "source": doc["_path"],
            }
            for doc in load_results(results_dir, experiment=experiment)
        ]
    return out


def load_demo(results_dir: str | Path = "results") -> dict[str, Any]:
    """Load the demo latency runs and the demo's correctness gate.

    Runs are split by how the frames were sourced, because the two kinds are not comparable
    and must never share an axis. A benchmark run reads a video file as fast as it can, so
    every millisecond it reports is work. A live run blocks waiting for the next camera frame,
    and that wait is charged to ``capture`` -- on this machine 16.3 ms of a 33.4 ms frame,
    which is not the pipeline being slow but the camera setting the pace. Averaging or
    overwriting one with the other reverses the CPU-vs-MPS comparison in
    :func:`latency_budget`, since a camera-bound CPU row loses to an unthrottled MPS one.

    Args:
        results_dir: Directory holding the committed result files.

    Returns:
        ``{"runs": [...], "live_runs": [...], "verify": {...} | None}``. ``runs`` holds the
        reproducible offline benchmarks, newest last, each carrying the per-stage percentiles
        and the device it ran on; ``live_runs`` holds the camera sessions in the same shape.
    """
    everything = [
        {
            "device": doc["metrics"]["model"]["device"],
            "fps": float(doc["metrics"]["fps"]),
            "n_frames": int(doc["metrics"]["n_frames"]),
            "frame_ms": doc["metrics"]["frame_ms"],
            "stages_ms": doc["metrics"]["stages_ms"],
            "source_fps": float(doc["metrics"]["source"]["fps"]),
            "resolution": (doc["metrics"]["source"]["width"], doc["metrics"]["source"]["height"]),
            "target_fps": float(doc["metrics"]["target_fps"]),
            "live": bool(doc["metrics"]["source"].get("live", False)),
            "source": doc["_path"],
        }
        for doc in load_results(results_dir, experiment="demo")
    ]
    checks = load_results(results_dir, experiment="demo-verify")
    return {
        "runs": [run for run in everything if not run["live"]],
        "live_runs": [run for run in everything if run["live"]],
        "verify": None
        if not checks
        else {k: v for k, v in checks[-1]["metrics"].items() if k != "records"},
    }


def model_config(results_dir: str | Path = "results") -> dict[str, Any]:
    """Return the model config the sweep actually ran under, from a result file.

    Reading it back out of a result rather than from ``configs/model.yaml`` is the point: the
    config on disk can have moved on since the runs, and a communication-cost figure has to
    describe the model that was measured.

    Args:
        results_dir: Directory holding the committed result files.

    Returns:
        The ``config.model`` block of the most recent E5 run.

    Raises:
        FileNotFoundError: If no finished E5 run exists.
    """
    docs = load_results(results_dir, experiment="E5")
    if not docs:
        raise FileNotFoundError("no finished E5 run to read the measured model config from")
    return docs[-1]["config"]["model"]


# ------------------------------------------------------------------ aggregation


def by_k(folds: tuple[Fold, ...], *, metric: str = "top1") -> dict[int, dict[str, float]]:
    """Summarize a method over every fold at each k.

    Pools the ``(signer, seed)`` folds, so ``n = signers x seeds`` and the spread is the
    total variability a deployment sees: a random new signer, trained once. The alternative
    -- average the seeds per signer first, then take the spread over signers -- is reported
    alongside it by :func:`by_k_per_signer`, because it answers a different question. The two
    means agree; the spreads need not be ordered either way, since averaging seeds shrinks
    each point's noise but also cuts ``n``, which the sample (n-1) correction pays for. On
    this data the per-signer spread is the smaller of the two at every k, because seed noise
    is real and the signer effect dominates it.

    Args:
        folds: Folds of one method.
        metric: ``"top1"`` or ``"top5"``.

    Returns:
        ``{k: {"mean", "std", "min", "max", "n"}}``.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for fold in folds:
        grouped[fold.k].append(getattr(fold, metric))
    return {k: mean_std(v) for k, v in sorted(grouped.items())}


def by_k_per_signer(
    folds: tuple[Fold, ...], *, metric: str = "top1"
) -> dict[int, dict[str, float]]:
    """Summarize a method by averaging seeds within a signer, then spreading over signers.

    This is "mean +/- std across signers" in the strict sense of PLAN.md section 6: ``n`` is
    the number of held-out signers, and seed noise has been averaged out of each point rather
    than added to the spread.

    Args:
        folds: Folds of one method.
        metric: ``"top1"`` or ``"top5"``.

    Returns:
        ``{k: {"mean", "std", "min", "max", "n"}}`` with ``n`` equal to the signer count.
    """
    grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for fold in folds:
        grouped[fold.k][fold.signer].append(getattr(fold, metric))
    return {
        k: mean_std([sum(v) / len(v) for v in per_signer.values()])
        for k, per_signer in sorted(grouped.items())
    }


def paired_delta(
    left: tuple[Fold, ...], right: tuple[Fold, ...], *, metric: str = "top1"
) -> dict[int, dict[str, float]]:
    """Compare two methods fold by fold rather than mean against mean.

    Two methods that were run on the same folds share their hard signers and their easy ones,
    so the paired difference removes the between-signer variance that dominates the unpaired
    comparison. On this data that variance is larger than most of the effects being measured,
    which is why the sweep reports both.

    Args:
        left: Folds of the first method.
        right: Folds of the second method; only folds present in both are used.
        metric: ``"top1"`` or ``"top5"``.

    Returns:
        ``{k: {"mean", "std", "n", "n_better", "t"}}`` with differences in accuracy points
        (``left - right``), ``n_better`` counting folds where ``left`` won strictly, and
        ``t`` the one-sample t statistic of the differences. No p-value is reported: with
        ten signers appearing under three seeds the folds are not independent, so a nominal
        p would overstate the evidence.
    """
    index = {fold.key: getattr(fold, metric) for fold in right}
    grouped: dict[int, list[float]] = defaultdict(list)
    for fold in left:
        other = index.get(fold.key)
        if other is not None:
            grouped[fold.k].append(getattr(fold, metric) - other)

    out: dict[int, dict[str, float]] = {}
    for k, diffs in sorted(grouped.items()):
        summary = mean_std(diffs)
        n = summary["n"]
        std = summary["std"]
        t = summary["mean"] / (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
        out[k] = {
            "mean": summary["mean"],
            "std": std,
            "n": n,
            "n_better": sum(1 for d in diffs if d > 0),
            "t": t,
        }
    return out


def crossover(
    curve: dict[int, dict[str, float]], threshold: float
) -> dict[str, float | int | None]:
    """Find the smallest k at which a method's mean reaches a reference level.

    The margin is returned with the k because a crossing can be a dead heat. On this data
    E5 meets E1 at k=3 by 0.00 points -- reporting the k alone would read as "E5 clears the
    ceiling at k=3" when what happened is that the two land on the same number.

    Args:
        curve: A ``by_k``-style mapping.
        threshold: The accuracy to reach, in [0, 1].

    Returns:
        ``{"k": int | None, "margin_points": float | None}``. ``k`` is ``None`` when the
        sweep never gets there, which is a result rather than a missing value: it says the
        method does not reach that line within the k this dataset can supply, and the margin
        is then the shortfall at the largest k.
    """
    for k in sorted(curve):
        if curve[k]["mean"] >= threshold:
            return {"k": k, "margin_points": (curve[k]["mean"] - threshold) * 100}
    last = max(curve)
    return {"k": None, "margin_points": (curve[last]["mean"] - threshold) * 100}


# ------------------------------------------------------------------ figures


def _save(fig: plt.Figure, out_dir: Path, name: str, formats: tuple[str, ...]) -> list[str]:
    """Write one figure in every requested format and close it.

    Args:
        fig: The figure.
        out_dir: Destination directory, created if missing.
        name: Basename without extension.
        formats: Extensions, e.g. ``("png", "pdf")``. PDF is what the thesis embeds; PNG is
            what a reader opens.

    Returns:
        The paths written, as strings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = out_dir / f"{name}.{extension}"
        # PDF carries a CreationDate by default, so an unchanged figure would still show as
        # modified in git on every regeneration -- which makes "make figures is a no-op if
        # nothing changed" untrue, and trains a reader to ignore the diff. Setting it to None
        # omits the key. PNG has no such stamp.
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(path, format=extension, bbox_inches="tight", dpi=200, metadata=metadata)
        written.append(str(path))
    plt.close(fig)
    return written


def adaptation_curve(
    folds: dict[str, tuple[Fold, ...]],
    centralized: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw the adaptation curve: accuracy against k, one line per method.

    This is the figure PLAN.md section 6 calls the money chart. Reading it: a method is worth
    its complexity only where its line sits above E3, the signer training alone, and the k at
    which a line crosses E1 is how much of the signer's own data it takes to recover the
    ceiling.

    Args:
        folds: Loaded folds keyed by method.
        centralized: Output of :func:`load_centralized`.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn, for ``figures/summary.json``.
    """
    require_methods(folds, ("E3", "E4", "E5"))
    apply_style()
    methods = [m for m in METHOD_ORDER if m in folds]
    curves = {m: by_k(folds[m]) for m in methods}
    per_signer_curves = {m: by_k_per_signer(folds[m]) for m in methods}
    ks = sorted({k for c in curves.values() for k in c})

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _despine(ax)

    for experiment in ("E2", "E1"):
        if experiment not in centralized:
            continue
        level = centralized[experiment]["top1"]["mean"] * 100
        # Stops at the last data point rather than spanning the axes: the right margin holds
        # the direct labels, and a rule running through them is noise.
        ax.plot(
            [ks[0] - 0.78, ks[-1] + 0.05],
            [level, level],
            color=INK_MUTED,
            linestyle=(0, (5, 4)),
            linewidth=1.1,
            zorder=1,
        )
        ax.text(
            ks[0] - 0.66,
            level,
            experiment,
            color=INK_MUTED,
            fontsize=8.5,
            va="center",
            ha="left",
            zorder=2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
        )

    ends: list[tuple[str, float]] = []
    for method in methods:
        xs = sorted(curves[method])
        ys = [curves[method][k]["mean"] * 100 for k in xs]
        errs = [curves[method][k]["std"] * 100 for k in xs]
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            color=METHOD_COLOR[method],
            marker="o",
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            capsize=2.5,
            elinewidth=1.0,
            zorder=3,
            label=METHOD_LABEL[method],
        )
        ends.append((method, ys[-1]))

    placed = _declutter([y for _, y in ends], min_gap=3.8, limit=(0.0, 101.0))
    for (method, y_true), y_label in zip(ends, placed, strict=True):
        colour = METHOD_COLOR[method]
        ax.plot(
            [ks[-1] + 0.07, ks[-1] + 0.42],
            [y_true, y_label],
            color=colour,
            linewidth=0.9,
            zorder=2,
        )
        ax.text(
            ks[-1] + 0.5,
            y_label,
            f"{method}  {y_true:.1f}%",
            color=colour,
            fontsize=8.5,
            va="center",
            ha="left",
            fontweight="medium",
        )

    n_folds = curves[methods[0]][ks[0]]["n"]
    ax.set_xlim(ks[0] - 0.78, ks[-1] + 1.6)
    ax.set_ylim(-4, 104)
    ax.set_xticks(ks)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("k  — labelled clips per sign supplied by the held-out signer")
    ax.set_ylabel("top-1 accuracy on the held-out signer (%)")
    ax.set_title(
        "Personalization from k examples, leave-one-signer-out on LSA64",
        color=INK,
        loc="left",
        pad=30,
    )
    ax.text(
        0.0,
        1.018,
        f"mean over {n_folds} folds — every signer held out in turn, under 3 seeds; bars "
        "are ±1 sd across folds\nE1 and E2 are the centralized baselines of the same "
        "architecture",
        transform=ax.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.44))

    reached = {}
    for experiment in ("E1", "E2"):
        if experiment in centralized:
            level = centralized[experiment]["top1"]["mean"]
            reached[experiment] = {m: crossover(curves[m], level) for m in methods}

    return {
        "file": _save(fig, out_dir, "fig1_adaptation_curve", formats),
        "question": "RQ2, RQ3 — PLAN.md section 6 'the money chart'",
        "k_values": ks,
        "n_folds_per_point": n_folds,
        "top1_pooled_over_signers_and_seeds": {m: {k: curves[m][k] for k in ks} for m in methods},
        "top1_seed_averaged_then_across_signers": {
            m: {k: per_signer_curves[m][k] for k in ks} for m in methods
        },
        "reference_levels": {e: centralized[e]["top1"] for e in ("E1", "E2") if e in centralized},
        "smallest_k_reaching_reference": reached,
    }


def generalization_gap(
    folds: dict[str, tuple[Fold, ...]],
    centralized: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw RQ1: the signer-independent generalization gap, and who it falls on.

    The left panel is the gap as PLAN.md defines it, E1 minus E2 over three seeds. The right
    panel exists because that E2 estimate rests on two test signers: the leave-one-signer-out
    sweep evaluates every signer zero-shot, and the spread it exposes is wider than the E1/E2
    comparison can show. Those folds are federatedly pretrained rather than centralized, so
    the panel is labelled as such -- phase 3 measured the two within a point of each other,
    which is what makes the substitution reasonable, not an assumption that they are equal.

    Args:
        folds: Loaded folds keyed by method.
        centralized: Output of :func:`load_centralized`.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn.

    Raises:
        FileNotFoundError: If E1 or E2 is missing.
    """
    missing = [e for e in ("E1", "E2") if e not in centralized]
    if missing:
        raise FileNotFoundError(f"no finished results for {', '.join(missing)}; run `make train`")
    require_methods(folds, ("E4",))
    apply_style()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1.0, 1.5]}
    )
    _despine(left)
    _despine(right)

    # --- left: the gap itself ------------------------------------------------------
    runs = {e: [r["top1"] * 100 for r in centralized[e]["runs"]] for e in ("E1", "E2")}
    positions = {"E1": 0.0, "E2": 1.0}
    for experiment, values in runs.items():
        colour = INK_SOFT
        x = positions[experiment]
        offsets = np.linspace(-0.09, 0.09, len(values)) if len(values) > 1 else np.zeros(1)
        left.scatter(
            x + offsets,
            values,
            s=26,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
        mean = centralized[experiment]["top1"]["mean"] * 100
        left.plot([x - 0.22, x + 0.22], [mean, mean], color=INK, linewidth=2.0, zorder=4)
        left.text(
            x,
            mean + 1.6,
            f"{mean:.1f}%",
            color=INK,
            fontsize=9,
            ha="center",
            va="bottom",
            fontweight="medium",
        )

    gap = (centralized["E1"]["top1"]["mean"] - centralized["E2"]["top1"]["mean"]) * 100
    top = centralized["E1"]["top1"]["mean"] * 100
    bottom = centralized["E2"]["top1"]["mean"] * 100
    left.annotate(
        "",
        xy=(0.5, top),
        xytext=(0.5, bottom),
        arrowprops={"arrowstyle": "<->", "color": NEGATIVE, "linewidth": 1.4},
    )
    left.text(
        0.56,
        (top + bottom) / 2,
        f"{gap:.1f} pts",
        color=NEGATIVE,
        fontsize=9.5,
        va="center",
        ha="left",
        fontweight="medium",
    )
    left.set_xlim(-0.45, 1.45)
    left.set_ylim(bottom - 8, top + 6)
    left.set_xticks([0.0, 1.0])
    left.set_xticklabels(["E1\nsigner-dependent", "E2\nsigner-independent"])
    left.set_ylabel("top-1 accuracy (%)")
    left.set_title("The gap (RQ1)", color=INK, loc="left", pad=18)
    left.text(
        0.0,
        1.02,
        f"one dot per seed (n={len(runs['E1'])}); bar is the mean",
        transform=left.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    left.grid(axis="x", visible=False)

    # --- right: every signer, zero-shot -------------------------------------------
    zero_shot: dict[int, list[float]] = defaultdict(list)
    for fold in folds["E4"]:
        if fold.k == 0:
            zero_shot[fold.signer].append(fold.top1 * 100)
    per_signer = {s: mean_std(v) for s, v in sorted(zero_shot.items())}
    order = sorted(per_signer, key=lambda s: per_signer[s]["mean"])
    ys = np.arange(len(order))
    means = [per_signer[s]["mean"] for s in order]

    for row, signer in enumerate(order):
        values = zero_shot[signer]
        # A range line per signer rather than a bar from zero: the finding is where the
        # signers sit relative to each other, and a zero baseline would compress all ten
        # into the last fifth of the panel.
        right.plot([min(values), max(values)], [row, row], color=GRID, linewidth=2.4, zorder=1)
        right.scatter(
            values,
            np.full(len(values), row),
            s=18,
            color=METHOD_COLOR["E4"],
            alpha=0.45,
            edgecolor="none",
            zorder=2,
        )
    right.scatter(
        means,
        ys,
        s=42,
        color=METHOD_COLOR["E4"],
        edgecolor=SURFACE,
        linewidth=1.4,
        zorder=3,
    )
    overall = mean_std([v for values in zero_shot.values() for v in values])
    right.axvline(overall["mean"], color=INK_MUTED, linestyle=(0, (5, 4)), linewidth=1.1)
    right.text(
        overall["mean"],
        -0.62,
        f" mean {overall['mean']:.1f}%",
        color=INK_MUTED,
        fontsize=8.5,
        va="center",
        ha="left",
    )
    right.set_yticks(ys)
    right.set_yticklabels([f"signer {s}" for s in order])
    right.set_ylim(-1.0, len(order) - 0.3)
    span = max(v for values in zero_shot.values() for v in values) - min(
        v for values in zero_shot.values() for v in values
    )
    right.set_xlim(
        min(v for values in zero_shot.values() for v in values) - span * 0.12,
        max(v for values in zero_shot.values() for v in values) + span * 0.12,
    )
    right.set_xlabel("zero-shot top-1 on that signer (%)")
    right.set_title("Who the gap falls on", color=INK, loc="left", pad=18)
    right.text(
        0.0,
        1.02,
        "each signer held out in turn, k=0, federated pretraining (E4); small dots are seeds",
        transform=right.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    right.grid(axis="y", visible=False)

    fig.tight_layout()
    return {
        "file": _save(fig, out_dir, "fig2_generalization_gap", formats),
        "question": "RQ1 — how large is the signer-independent generalization gap",
        "E1_top1": centralized["E1"]["top1"],
        "E2_top1": centralized["E2"]["top1"],
        "gap_points": gap,
        "E2_test_signers": centralized["E2"]["test_signers"],
        "zero_shot_per_signer_E4_k0": per_signer,
        "zero_shot_overall_E4_k0": overall,
    }


#: The contrasts that answer the research questions, as ``(left, right, what it decides)``.
CONTRASTS = (
    ("E5", "E3", "RQ3: does a federated encoder beat training alone?"),
    ("E4", "E5", "Does keeping the head private cost accuracy?"),
    ("E5", "E6", "What does federating instead of pooling cost?"),
)


def paired_differences(
    folds: dict[str, tuple[Fold, ...]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw the three method contrasts fold by fold.

    Between-signer variance on this dataset is larger than most of the differences being
    measured, so an unpaired comparison of two means hides whichever effect is real. Pairing
    on ``(seed, signer, k)`` removes it: every bar is the mean of differences measured on the
    same folds.

    ``k=0`` is excluded. E5 and E6 have no classifier at all until the signer supplies a
    label, so their zero-shot accuracy is chance by construction and a difference against it
    measures the architecture rather than the method.

    Args:
        folds: Loaded folds keyed by method.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn.
    """
    require_methods(folds, tuple({m for pair in CONTRASTS for m in pair[:2]}))
    apply_style()
    deltas = {
        f"{a}-{b}": {k: v for k, v in paired_delta(folds[a], folds[b]).items() if k > 0}
        for a, b, _ in CONTRASTS
    }
    ks = sorted({k for d in deltas.values() for k in d})
    span = max(
        abs(v["mean"]) * 100 + abs(v["std"]) * 100 / math.sqrt(v["n"])
        for d in deltas.values()
        for v in d.values()
    )

    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(8.6, 3.9), sharey=True)
    for ax, (a, b, question) in zip(axes, CONTRASTS, strict=True):
        _despine(ax)
        name = f"{a}-{b}"
        values = [deltas[name][k]["mean"] * 100 for k in ks]
        sems = [deltas[name][k]["std"] * 100 / math.sqrt(deltas[name][k]["n"]) for k in ks]
        colours = [POSITIVE if v >= 0 else NEGATIVE for v in values]
        ax.bar(ks, values, width=0.62, color=colours, zorder=2, linewidth=0)
        ax.errorbar(
            ks,
            values,
            yerr=sems,
            fmt="none",
            ecolor=INK_SOFT,
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        ax.axhline(0, color=INK_SOFT, linewidth=1.0, zorder=4)
        for k, value, sem in zip(ks, values, sems, strict=True):
            offset = sem + span * 0.05
            ax.text(
                k,
                value + (offset if value >= 0 else -offset),
                f"{value:+.1f}",
                color=INK,
                fontsize=8,
                ha="center",
                va="bottom" if value >= 0 else "top",
            )
        improved = " ".join(f"{deltas[name][k]['n_better']}/{deltas[name][k]['n']}" for k in ks)
        ax.set_title(f"{a} − {b}", color=INK, loc="left", pad=34)
        ax.text(
            0.0, 1.065, question, transform=ax.transAxes, color=INK_SOFT, fontsize=8, va="bottom"
        )
        ax.text(
            0.0,
            1.018,
            f"folds where {a} wins, k={ks[0]}..{ks[-1]}:  {improved}",
            transform=ax.transAxes,
            color=INK_MUTED,
            fontsize=7.5,
            va="bottom",
        )
        ax.set_xticks(ks)
        ax.set_xlabel("k")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("paired difference in top-1 (percentage points)")
    axes[0].set_ylim(-span * 0.55, span * 1.28)
    fig.tight_layout()
    return {
        "file": _save(fig, out_dir, "fig3_paired_differences", formats),
        "question": "RQ3 — the method contrasts, paired on (seed, signer, k)",
        "excluded_k": [0],
        "note": (
            "differences are in accuracy points; error bars are the standard error of the "
            "paired mean. No p-value: ten signers recur under three seeds, so the 30 folds "
            "are not independent and a nominal p would overstate the evidence."
        ),
        "contrasts": {
            f"{a}-{b}": {"question": q, "by_k": deltas[f"{a}-{b}"]} for a, b, q in CONTRASTS
        },
    }


def signer_spread(
    folds: dict[str, tuple[Fold, ...]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw the distribution over held-out signers, one panel per k.

    PLAN.md section 6 asks for inter-signer variance as a result in its own right, not as an
    error bar. A method whose mean is a point lower but whose worst signer is ten points
    better is the one you would deploy, and that ordering is invisible in the curve.

    Args:
        folds: Loaded folds keyed by method.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn.
    """
    require_methods(folds, ("E3", "E4", "E5"))
    apply_style()
    methods = [m for m in METHOD_ORDER if m in folds]
    ks = sorted({f.k for m in methods for f in folds[m] if f.k > 0})

    fig, axes = plt.subplots(1, len(ks), figsize=(9.2, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    spreads: dict[str, dict[int, dict[str, float]]] = {m: {} for m in methods}

    for ax, k in zip(axes, ks, strict=True):
        _despine(ax)
        for column, method in enumerate(methods):
            values = [f.top1 * 100 for f in folds[method] if f.k == k]
            spreads[method][k] = mean_std([v / 100 for v in values])
            # Deterministic spread within the column: a seeded RNG would still make the
            # figure depend on call order, and the dots carry no information in x.
            offsets = np.linspace(-0.22, 0.22, len(values))
            ax.scatter(
                column + offsets,
                values,
                s=13,
                color=METHOD_COLOR[method],
                alpha=0.5,
                edgecolor="none",
                zorder=2,
            )
            mean = float(np.mean(values))
            ax.plot(
                [column - 0.32, column + 0.32],
                [mean, mean],
                color=METHOD_COLOR[method],
                linewidth=2.2,
                solid_capstyle="round",
                zorder=3,
            )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.set_title(f"k = {k}", color=INK, loc="left", pad=8)
        ax.grid(axis="x", visible=False)

    # Seeds averaged within a signer before ranking, so "the hardest signer" is a property
    # of the signer rather than of one unlucky run.
    per_signer: dict[str, dict[int, dict[int, float]]] = {}
    for method in methods:
        grouped: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for fold in folds[method]:
            if fold.k > 0:
                grouped[fold.k][fold.signer].append(fold.top1)
        per_signer[method] = {
            k: {s: sum(v) / len(v) for s, v in sorted(rows.items())}
            for k, rows in sorted(grouped.items())
        }

    axes[0].set_ylabel("top-1 on the held-out signer (%)")
    # Lay the panels out first: figure-level text placed above y=1 is outside tight_layout's
    # accounting, so adding it beforehand leaves a band of empty space under the header.
    fig.tight_layout()
    fig.suptitle(
        "Every held-out signer, not just the mean",
        color=INK,
        x=0.008,
        y=1.085,
        ha="left",
        va="top",
        fontsize=10.5,
    )
    fig.text(
        0.008,
        1.028,
        "one dot per (signer, seed) fold; the bar is the method's mean at that k",
        color=INK_SOFT,
        fontsize=8,
        ha="left",
        va="top",
    )
    return {
        "file": _save(fig, out_dir, "fig4_signer_spread", formats),
        "question": "PLAN.md section 6 — inter-signer variance as a finding",
        "by_method_and_k": spreads,
        "worst_fold": {
            m: {k: min(f.top1 for f in folds[m] if f.k == k) for k in ks} for m in methods
        },
        "range_points_pooled": {
            m: {k: (spreads[m][k]["max"] - spreads[m][k]["min"]) * 100 for k in ks} for m in methods
        },
        "top1_per_signer_seed_averaged": per_signer,
        "range_points_across_signers": {
            m: {
                k: (max(per_signer[m][k].values()) - min(per_signer[m][k].values())) * 100
                for k in ks
            }
            for m in methods
        },
        "hardest_signer": {
            m: {k: min(per_signer[m][k], key=per_signer[m][k].__getitem__) for k in ks}
            for m in methods
        },
    }


def federated_convergence(
    federated: dict[str, Any],
    centralized: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw the phase-3 evidence that the federated simulation is correct.

    The left panel is what the clients were doing: local training accuracy per round, for the
    signer partition that the experiments use and for the IID partition that the correctness
    check uses. The right panel is the check itself. FedAvg over an IID partition is solving
    the same problem as centralized training, so it has to land in the same place; if it does
    not, every federated number afterwards is measuring a bug. PLAN.md section 8 makes this
    the week-4 gate.

    Args:
        federated: Output of :func:`load_federated`.
        centralized: Output of :func:`load_centralized`.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn.

    Raises:
        FileNotFoundError: If either federated run is missing.
    """
    if not federated.get("signer") or not federated.get("iid"):
        raise FileNotFoundError(
            "no finished federated runs; run `make federated` before drawing the convergence figure"
        )
    if "E2" not in centralized:
        raise FileNotFoundError("no finished E2 run to check the IID partition against")
    apply_style()

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.8, 3.8), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    _despine(left)
    _despine(right)

    for key, colour, label in (
        ("signer", METHOD_COLOR["E4"], "one client per signer"),
        ("iid", METHOD_COLOR["E5"], "IID partition (correctness check)"),
    ):
        for index, run in enumerate(federated[key]):
            left.plot(
                run["rounds"],
                [v * 100 for v in run["train_top1"]],
                color=colour,
                linewidth=1.8,
                alpha=1.0 if index == 0 else 0.4,
                label=label if index == 0 else None,
                zorder=3,
            )
    left.set_xlabel("federated round")
    left.set_ylabel("client-side training top-1 (%)")
    left.set_ylim(-4, 104)
    left.set_yticks(range(0, 101, 20))
    left.set_title("Clients converge", color=INK, loc="left", pad=12)
    left.text(
        0.0,
        1.015,
        f"{len(federated['signer'])} seeds, {federated['signer'][0]['n_clients']} clients, "
        f"{federated['signer'][0]['n_rounds']} rounds; faint lines are the other seeds",
        transform=left.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    left.grid(axis="x", visible=False)
    left.legend(loc="lower right")

    iid = federated["iid"][0]
    tolerance = iid["tolerance"]
    reference = centralized["E2"]["top1"]
    delta = (iid["top1"] - reference["mean"]) * 100
    if tolerance is not None:
        right.axhspan(
            (reference["mean"] - tolerance) * 100,
            (reference["mean"] + tolerance) * 100,
            color=GRID,
            zorder=1,
        )
        right.text(
            1.45,
            (reference["mean"] + tolerance) * 100 + 0.12,
            f"±{tolerance * 100:.0f} pts tolerance",
            color=INK_MUTED,
            fontsize=8,
            va="bottom",
            ha="right",
        )
    right.plot(
        [-0.4, 1.4],
        [reference["mean"] * 100] * 2,
        color=INK_MUTED,
        linestyle=(0, (5, 4)),
        linewidth=1.1,
        zorder=2,
    )
    for x, value, colour in (
        (0.0, reference["mean"] * 100, INK_SOFT),
        (1.0, iid["top1"] * 100, METHOD_COLOR["E5"]),
    ):
        right.scatter([x], [value], s=70, color=colour, edgecolor=SURFACE, linewidth=1.6, zorder=4)
        right.text(
            x, value + 0.55, f"{value:.1f}%", color=INK, fontsize=9, ha="center", va="bottom"
        )
    if tolerance is not None:
        margin = tolerance * 100 * 0.45
        right.set_ylim(
            (reference["mean"] - tolerance) * 100 - margin,
            (reference["mean"] + tolerance) * 100 + margin,
        )
    right.set_xlim(-0.55, 1.55)
    right.set_xticks([0.0, 1.0])
    right.set_xticklabels(["centralized E2", "FedAvg, IID"])
    right.set_ylabel("top-1 (%)")
    passed = tolerance is None or abs(delta) <= tolerance * 100
    right.set_title(
        f"Correctness check: {'PASS' if passed else 'FAIL'}", color=INK, loc="left", pad=12
    )
    right.text(
        0.0,
        1.015,
        f"delta {delta:+.1f} pts",
        transform=right.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    right.grid(axis="x", visible=False)

    fig.tight_layout()
    return {
        "file": _save(fig, out_dir, "fig5_federated_convergence", formats),
        "question": "PLAN.md section 8 week 4 — the federated correctness gate",
        "iid_check": {
            "fedavg_iid_top1": iid["top1"],
            "centralized_E2_top1": reference,
            "delta_points": delta,
            "tolerance_points": None if tolerance is None else tolerance * 100,
            "passed": passed,
        },
        "signer_partition_final_top1": mean_std([r["top1"] for r in federated["signer"]]),
        "rounds": federated["signer"][0]["n_rounds"],
        "n_clients": federated["signer"][0]["n_clients"],
    }


def communication_cost(
    measured_model: dict[str, Any],
    federated: dict[str, Any],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw what FedPer actually saves on the wire.

    FedPer is often motivated as cheaper as well as more private. On this architecture it is
    not: the classifier is a single 128 x 64 layer and the encoder is everything else, so
    withholding the head removes about a percent of the payload. The privacy argument stands
    on its own; the bandwidth argument does not, and the figure is drawn so that a reader
    reaches that conclusion from the geometry rather than from a sentence.

    Sizes are computed by instantiating the architecture recorded in the result file, not the
    one currently in ``configs/``, so the figure describes the model that was measured.

    Args:
        measured_model: The ``config.model`` block from a finished run.
        federated: Output of :func:`load_federated`, for the round and client counts.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn.
    """
    # Imported here rather than at module scope so the other five figures do not pay torch's
    # import cost, and so a results-only checkout can still draw them.
    from signadapt.models.model import ENCODER_PREFIX, build_model

    model = build_model(measured_model)
    state = model.state_dict()
    encoder = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    head = {k: v for k, v in state.items() if not k.startswith(ENCODER_PREFIX)}

    def bytes_of(part: dict[str, Any]) -> int:
        return sum(int(v.numel()) * v.element_size() for v in part.values())

    encoder_bytes, head_bytes = bytes_of(encoder), bytes_of(head)
    total = encoder_bytes + head_bytes
    saving = head_bytes / total

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 1.95))
    _despine(ax, left=False)
    ax.set_axisbelow(True)

    scale = 100 / total
    ax.barh(
        [0], [encoder_bytes * scale], height=0.42, color=METHOD_COLOR["E5"], zorder=3, linewidth=0
    )
    # A 2px surface gap between the two fills rather than a border around them.
    ax.barh(
        [0],
        [head_bytes * scale],
        left=encoder_bytes * scale,
        height=0.42,
        color=METHOD_COLOR["E4"],
        zorder=3,
        linewidth=2.0,
        edgecolor=SURFACE,
    )
    ax.text(
        encoder_bytes * scale / 2,
        0,
        f"encoder — federated · {encoder_bytes / 1e6:.2f} MB",
        color=SURFACE,
        fontsize=8.5,
        ha="center",
        va="center",
        fontweight="medium",
    )
    ax.annotate(
        f"head — private · {head_bytes / 1e3:.0f} kB",
        xy=(100, 0.0),
        xytext=(100, 0.42),
        color=METHOD_COLOR["E4"],
        fontsize=8.5,
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": METHOD_COLOR["E4"], "linewidth": 1.0},
    )
    ax.set_xlim(0, 104)
    ax.set_ylim(-0.42, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("share of the per-round payload (%)")
    ax.set_title(
        f"Withholding the head saves {saving * 100:.1f}% of the payload",
        color=INK,
        loc="left",
        pad=22,
    )
    ax.text(
        0.0,
        1.04,
        f"one client, one round: FedAvg sends {total / 1e6:.2f} MB, FedPer sends "
        f"{encoder_bytes / 1e6:.2f} MB. FedPer's case is privacy, not bandwidth.",
        transform=ax.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    ax.grid(axis="y", visible=False)

    rounds = federated["signer"][0]["n_rounds"] if federated.get("signer") else None
    clients = federated["signer"][0]["n_clients"] if federated.get("signer") else None
    return {
        "file": _save(fig, out_dir, "fig6_communication_cost", formats),
        "question": "RQ4 — per-round communication cost",
        "n_parameters": model.n_parameters(),
        "n_tensors": {"encoder": len(encoder), "head": len(head), "all": len(state)},
        "bytes_per_client_per_round": {
            "fedavg": total,
            "fedper": encoder_bytes,
            "saving_fraction": saving,
        },
        "total_uplink_bytes": None
        if rounds is None
        else {"fedavg": total * rounds * clients, "fedper": encoder_bytes * rounds * clients},
        "rounds": rounds,
        "n_clients": clients,
    }


#: Stage order in the latency figure: the order a frame actually passes through them.
STAGE_ORDER = ("capture", "landmarks", "normalize", "model", "render")

STAGE_LABEL = {
    "capture": "capture",
    "landmarks": "keypoints (MediaPipe)",
    "normalize": "normalize",
    "model": "model",
    "render": "caption + render",
}


def latency_budget(
    demo: dict[str, Any],
    out_dir: Path,
    formats: tuple[str, ...],
) -> dict[str, Any]:
    """Draw where a frame's time goes, against the budget a live camera actually allows.

    The headline of a demo is usually its frame rate, which on an unthrottled video file is
    the pipeline's capacity rather than anything a viewer would see: a 30 fps camera caps the
    rate whatever the pipeline can do. What matters is how much of the 33.3 ms between camera
    frames the pipeline spends, so that is what the figure is drawn against.

    Only the offline benchmarks are drawn. A live camera session measures the camera as much
    as the pipeline -- its ``capture`` stage is mostly the blocking wait for the next frame --
    so putting a live row on the same axis as a benchmark row compares a throttled run with an
    unthrottled one. It also reverses the conclusion: a camera-bound CPU run reports a slower
    frame than an unthrottled MPS one, which would make the GPU look like the right choice
    when the benchmark says the opposite. The live session is reported in the summary instead,
    under ``live``, where it answers a different question -- what a viewer actually sees.

    Args:
        demo: Output of :func:`load_demo`.
        out_dir: Destination directory.
        formats: File formats to write.

    Returns:
        The numbers drawn, plus the newest live session if one exists.

    Raises:
        FileNotFoundError: If no offline benchmark run exists.
    """
    if not demo.get("runs"):
        extra = (
            " (the demo results present are all live camera sessions, which are not drawn here)"
            if demo.get("live_runs")
            else ""
        )
        raise FileNotFoundError(
            "no finished demo benchmark; run `make demo-bench` before drawing the latency "
            f"figure{extra}"
        )
    apply_style()
    runs = demo["runs"]
    devices = {}
    for run in runs:  # newest run wins per device
        devices[run["device"]] = run
    order = sorted(devices, key=lambda d: devices[d]["frame_ms"]["p50"])

    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(order) + 1.75))
    _despine(ax, left=False)

    budget_ms = 1000.0 / runs[-1]["source_fps"]
    ax.axvline(budget_ms, color=NEGATIVE, linewidth=1.2, zorder=5)
    ax.text(
        budget_ms + 0.4,
        len(order) - 0.34,
        f"{budget_ms:.1f} ms — one frame at {runs[-1]['source_fps']:.0f} fps",
        color=NEGATIVE,
        fontsize=8.5,
        va="center",
        ha="left",
    )

    # One hue, light to dark, in pipeline order: the stages are an ordered sequence, not
    # unrelated categories, so a categorical palette would be the wrong encoding. Validated
    # as an ordinal ramp -- monotone lightness, every adjacent gap >= 0.06, and the lightest
    # step clears 2:1 against the surface so the first segment does not vanish into it.
    ramp = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
    drawn: dict[str, Any] = {}
    for row, device in enumerate(order):
        run = devices[device]
        left = 0.0
        for stage, colour in zip(STAGE_ORDER, ramp, strict=True):
            if stage not in run["stages_ms"]:
                continue
            width = run["stages_ms"][stage]["p50"]
            ax.barh(
                [row],
                [width],
                left=left,
                height=0.62,
                color=colour,
                zorder=3,
                linewidth=2.0,
                edgecolor=SURFACE,
            )
            if width > budget_ms * 0.07:
                # Ink or surface, chosen by the step's luminance rather than by a hardcoded
                # list, so re-stepping the ramp cannot leave a label unreadable.
                rgb = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
                luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                ax.text(
                    left + width / 2,
                    row,
                    f"{width:.1f}",
                    color=INK if luminance > 0.5 else SURFACE,
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            left += width
        ax.text(
            left + 0.5,
            row,
            f"{run['fps']:.0f} fps",
            color=INK,
            fontsize=8.5,
            va="center",
            ha="left",
            fontweight="medium",
        )
        drawn[device] = {
            "fps": run["fps"],
            "frame_ms": run["frame_ms"],
            "stages_ms": {k: v["p50"] for k, v in run["stages_ms"].items()},
            "budget_used": left / budget_ms,
        }

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour)
        for stage, colour in zip(STAGE_ORDER, ramp, strict=True)
        if stage in runs[-1]["stages_ms"]
    ]
    ax.legend(
        handles,
        [STAGE_LABEL[s] for s in STAGE_ORDER if s in runs[-1]["stages_ms"]],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=len(handles),
        columnspacing=1.2,
        handlelength=1.1,
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"model on {d}" for d in order])
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.set_xlim(0, budget_ms * 1.16)
    ax.set_xlabel("median time per frame (ms)")
    ax.set_title("A frame's time budget, and what is left of it", color=INK, loc="left", pad=26)
    ax.text(
        0.0,
        1.03,
        f"{runs[-1]['resolution'][0]}x{runs[-1]['resolution'][1]} at "
        f"{runs[-1]['source_fps']:.0f} fps; bars are p50 per stage. Anything left of the red "
        "line runs in real time.",
        transform=ax.transAxes,
        color=INK_SOFT,
        fontsize=8,
        va="bottom",
    )
    ax.grid(axis="y", visible=False)

    live = demo.get("live_runs") or []
    return {
        "file": _save(fig, out_dir, "fig7_latency_budget", formats),
        "question": "RQ4 — on-device inference latency",
        "frame_budget_ms": budget_ms,
        "target_fps": runs[-1]["target_fps"],
        "by_device": drawn,
        # Not drawn, and not comparable with `by_device` -- see the docstring. Kept so the
        # README's claim about what a real camera delivers stays regenerable rather than
        # transcribed by hand.
        "live": None
        if not live
        else {
            "fps": live[-1]["fps"],
            "device": live[-1]["device"],
            "n_frames": live[-1]["n_frames"],
            "resolution": list(live[-1]["resolution"]),
            "camera_reported_fps": live[-1]["source_fps"],
            "frame_ms": live[-1]["frame_ms"],
            "stages_ms": {k: v["p50"] for k, v in live[-1]["stages_ms"].items()},
            "capture_share": (
                live[-1]["stages_ms"]["capture"]["p50"] / live[-1]["frame_ms"]["p50"]
            ),
        },
        "verify": demo.get("verify"),
    }


def matched_budget_check(results_dir: str | Path = "results") -> dict[str, Any] | None:
    """Compare E5 against E6 run at E5's own pretraining budget.

    E5 beats E6 at every k >= 2, which is the wrong way round: E6 pools on one machine what
    E5 can only reach through averaging, so it is supposed to be the ceiling. Before that can
    be read as evidence for federation it has to survive the dull explanation, which is that
    E6's early stopping simply gave it less training. E6M is E6 with the budget matched --
    100 epochs, early stopping off, same validation selection -- so the difference between
    ``E5-E6`` and ``E5-E6M`` is what the schedule was worth.

    Not a figure. It is a check on a figure's claim, it uses a method the charts deliberately
    do not know about, and drawing it beside the three real contrasts would imply E6M is a
    fourth method under test rather than the same method run twice.

    Args:
        results_dir: Directory holding the committed result files.

    Returns:
        The two paired contrasts and the budgets behind them, or ``None`` if E6M has not been
        run -- absence is the normal state of a diagnostic, not an error.
    """
    folds = load_folds(results_dir, methods=(*METHOD_ORDER, "E6M"))
    if "E6M" not in folds or "E5" not in folds or "E6" not in folds:
        return None
    return {
        "question": "Is E5 > E6 a property of federation, or of E6's shorter schedule?",
        "note": (
            "E6M is E6 with pretraining matched to the clip presentations the federation "
            "consumes (50 rounds x 2 local epochs = 100 passes over the pooled training "
            "set), early stopping off so the budget is spent, validation selection kept so "
            "only the budget differs. Run with `make sweep-matched`."
        ),
        "n_folds": {name: len(folds[name]) for name in ("E5", "E6", "E6M")},
        "contrasts": {
            "E5-E6": {k: dict(v) for k, v in paired_delta(folds["E5"], folds["E6"]).items()},
            "E5-E6M": {k: dict(v) for k, v in paired_delta(folds["E5"], folds["E6M"]).items()},
            "E6M-E6": {k: dict(v) for k, v in paired_delta(folds["E6M"], folds["E6"]).items()},
        },
        "by_k": {
            name: {k: dict(v) for k, v in by_k(folds[name]).items()} for name in ("E5", "E6", "E6M")
        },
    }


# ------------------------------------------------------------------ entry point


def build_all(
    results_dir: str | Path = "results",
    out_dir: str | Path = "figures",
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> dict[str, Any]:
    """Regenerate every figure and the summary of the numbers they contain.

    Args:
        results_dir: Directory of committed result JSON.
        out_dir: Where the images and ``summary.json`` are written.
        formats: Image formats to write for each figure.

    Returns:
        The summary document, also written to ``<out_dir>/summary.json``.
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    folds = load_folds(results_dir)
    centralized = load_centralized(results_dir)
    federated = load_federated(results_dir)

    summary: dict[str, Any] = {
        "generated_from": str(results_dir),
        "n_result_files": len(list(results_dir.glob("*.json"))),
        "methods_found": sorted(folds),
        "figures": {
            "fig1_adaptation_curve": adaptation_curve(folds, centralized, out_dir, formats),
            "fig2_generalization_gap": generalization_gap(folds, centralized, out_dir, formats),
            "fig3_paired_differences": paired_differences(folds, out_dir, formats),
            "fig4_signer_spread": signer_spread(folds, out_dir, formats),
            "fig5_federated_convergence": federated_convergence(
                federated, centralized, out_dir, formats
            ),
            "fig6_communication_cost": communication_cost(
                model_config(results_dir), federated, out_dir, formats
            ),
            "fig7_latency_budget": latency_budget(load_demo(results_dir), out_dir, formats),
        },
        # Checks on the figures' claims rather than figures. Omitted entirely when the run
        # they need is absent, so `make figures` from a fresh checkout is unaffected.
        "diagnostics": {
            key: value
            for key, value in (("matched_budget", matched_budget_check(results_dir)),)
            if value is not None
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Regenerate the figures from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code; 1 if a figure could not be drawn because its results are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = parser.parse_args(argv)

    try:
        summary = build_all(args.results, args.out, formats=tuple(args.formats))
    except FileNotFoundError as error:
        print(f"[figures] {error}")
        return 1

    print(f"[figures] from {summary['n_result_files']} result files in {args.results}/")
    for name, entry in summary["figures"].items():
        print(f"  {name:28s} -> {', '.join(entry['file'])}")
    print(f"  {'summary':28s} -> {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
