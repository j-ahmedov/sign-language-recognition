"""Centralized baselines: E1 (signer-dependent) and E2 (signer-independent).

E1 is the optimistic ceiling -- every signer appears in training. E2 trains on one set of
signers and tests on another. **E1 - E2 is the signer-independent generalization gap, the
number RQ1 exists to produce and the motivation for everything after it** (PLAN.md section
6), so both are run over the seeds in ``configs/model.yaml`` and reported as mean +/- std.

E6 (centralized pretrain + private head) also belongs in this module per PLAN.md section 7,
but it is a k-shot experiment and is implemented in phase 4 alongside E4 and E5, against the
same k-sweep harness -- running it on a different one would make the comparison meaningless.

Before any of that, :func:`run_sanity` trains on 50 clips and requires near-perfect fit
(PLAN.md section 8, week 3). A pipeline that cannot memorize 50 examples is broken, and no
number it produces afterwards means anything; ``--experiment all`` therefore refuses to run
E1 or E2 if the gate fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from signadapt.data.dataset import (
    ClipRecord,
    KeypointDataset,
    Split,
    load_records,
    make_splits,
)
from signadapt.models.model import build_model
from signadapt.train.loop import evaluate_tensors, resolve_device, stack_dataset, train_model
from signadapt.utils.config import apply_overrides, config_fingerprint, load_config
from signadapt.utils.metrics import format_mean_std, mean_std
from signadapt.utils.results import ResultsLogger
from signadapt.utils.seeding import seed_everything, temporary_seed


def _tensors(
    records: list, indices: tuple[int, ...], data_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one part of a split into tensors."""
    return stack_dataset(KeypointDataset(records, indices, data_cfg))


def run_sanity(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    *,
    seed: int = 0,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Overfit a tiny subset and check the pipeline can memorize it.

    Args:
        model_cfg: Loaded ``configs/model.yaml``.
        data_cfg: Loaded ``configs/data.yaml``.
        seed: Seed for the subset draw and for training.
        results_dir: Where to write the result JSON.

    Returns:
        ``{"train_top1", "target", "passed", "subset_size", "epochs", "seconds"}``.
    """
    spec = model_cfg["sanity"]
    subset_size, target = int(spec["subset_size"]), float(spec["target_accuracy"])
    device = resolve_device(model_cfg["train"].get("device", "auto"))
    seed_everything(seed)

    records = load_records(data_cfg["dataset"]["cache_dir"])
    with temporary_seed(seed):
        chosen = torch.randperm(len(records))[:subset_size].tolist()
    x, y = _tensors(records, tuple(chosen), data_cfg)

    model = build_model(model_cfg)
    with ResultsLogger(
        "sanity",
        config={"model": model_cfg, "data": data_cfg},
        seed=seed,
        results_dir=results_dir,
        description=(
            f"Overfit gate (PLAN.md section 8, week 3): fit {subset_size} clips and require "
            f"train top-1 >= {target:.2f}. Augmentation off -- the question is whether the "
            "pipeline can memorize, not whether it generalizes."
        ),
        extra={"parameters": model.n_parameters(), "device": str(device)},
    ) as log:
        # No validation set and no early stopping: the whole point is to run to convergence
        # on the training data itself.
        outcome = train_model(
            model,
            (x, y),
            None,
            model_cfg,
            device=device,
            seed=seed,
            on_epoch=lambda record: log.log_record(**record),
            epochs=int(spec["epochs"]),
            augment=False,
        )
        final = evaluate_tensors(model, (x, y), model_cfg, device=device)
        summary = {
            "train_top1": final.top1,
            "train_top5": final.top5,
            "target": target,
            "passed": bool(final.top1 >= target),
            "subset_size": subset_size,
            "epochs": outcome.epochs_run,
            "seconds": outcome.seconds,
        }
        log.set_metrics(**summary)

    verdict = "PASS" if summary["passed"] else "FAIL"
    print(
        f"[sanity] {verdict}  top-1 {final.top1:.3f} on {subset_size} clips "
        f"(target {target:.2f}) after {outcome.epochs_run} epochs, {outcome.seconds:.1f}s"
    )
    return summary


def run_centralized(
    experiment: str,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    *,
    seed: int = 0,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Train and evaluate one centralized baseline for one seed.

    Args:
        experiment: ``"E1"`` (signer-dependent) or ``"E2"`` (signer-independent).
        model_cfg: Loaded ``configs/model.yaml``.
        data_cfg: Loaded ``configs/data.yaml``. Its ``splits.mode`` is set from
            ``experiment``, so the caller cannot accidentally run E2 on an E1 split.
        seed: Seed for the run.
        results_dir: Where to write the result JSON.

    Returns:
        The test metrics dict, as written to the result file.

    Raises:
        ValueError: On an unknown experiment id.
    """
    modes = {"E1": "signer_dependent", "E2": "signer_independent"}
    if experiment not in modes:
        raise ValueError(f"expected E1 or E2, got {experiment!r}")

    data_cfg = {**data_cfg, "splits": {**data_cfg["splits"], "mode": modes[experiment]}}
    device = resolve_device(model_cfg["train"].get("device", "auto"))
    seed_everything(seed)

    records = load_records(
        data_cfg["dataset"]["cache_dir"],
        exclude_signers=data_cfg["splits"].get("exclude_signers", ()),
    )
    split = make_splits(records, data_cfg, seed=seed)
    parts = {
        name: _tensors(records, getattr(split, name), data_cfg) for name in ("train", "val", "test")
    }

    model = build_model(model_cfg)
    description = {
        "E1": "Centralized, signer-dependent: every signer in train and test (split by "
        "repetition). The optimistic ceiling, PLAN.md section 6.",
        "E2": "Centralized, signer-independent: train and test signers disjoint. The gap "
        "this thesis is about, PLAN.md section 6 / RQ1.",
    }[experiment]

    with ResultsLogger(
        experiment,
        config={"model": model_cfg, "data": data_cfg},
        seed=seed,
        results_dir=results_dir,
        description=description,
        extra={
            "split": {"name": split.name, "sizes": split.sizes(), "signers": split.signers},
            "parameters": model.n_parameters(),
            "device": str(device),
        },
    ) as log:
        outcome = train_model(
            model,
            parts["train"],
            parts["val"],
            model_cfg,
            device=device,
            seed=seed,
            on_epoch=lambda record: log.log_record(**record),
        )
        # Evaluate the best-validation checkpoint, not the last one: with early stopping the
        # last epoch is by construction a worse model than one seen `patience` epochs ago.
        model.load_state_dict(outcome.best_state)
        model.to(device)
        test = evaluate_tensors(
            model,
            parts["test"],
            model_cfg,
            device=device,
            records=records,
            indices=split.test,
        )
        metrics = test.to_dict() | {
            "best_epoch": outcome.best_epoch,
            "best_val_top1": outcome.best_val,
            "epochs_run": outcome.epochs_run,
            "seconds": outcome.seconds,
        }
        log.set_metrics(**metrics)

    print(
        f"[{experiment}] seed {seed}: test top-1 {test.top1:.4f}  top-5 {test.top5:.4f}  "
        f"(best epoch {outcome.best_epoch}, val {outcome.best_val:.4f}, "
        f"{outcome.epochs_run} epochs, {outcome.seconds:.0f}s)"
    )
    print("          per signer: " + "  ".join(f"{k}={v:.3f}" for k, v in test.per_signer.items()))
    return metrics


def pretrain_centralized(
    records: list[ClipRecord],
    fold: Split,
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    seed: int = 0,
    cache_dir: str | None = "data/checkpoints/pretrain",
) -> dict[str, Any]:
    """Train one model centrally on a LOSO fold's training signers, for E6.

    E6 is the non-federated upper bound for E5 (PLAN.md section 6): the same architecture and
    the same training signers, pooled on one machine. It keeps its natural advantage of
    early-stopping on the fold's validation signer, because being the ceiling is the point --
    the gap between E6 and E5 is what federation costs. That does mean E6's encoder is
    validation-selected while E4's and E5's are the final round, and that difference is part
    of what "upper bound" means here rather than a like-for-like comparison.

    Args:
        records: All records.
        fold: The LOSO fold; ``fold.train`` trains and ``fold.val`` selects.
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        seed: Seed.
        cache_dir: Where finished pretrainings are cached, or ``None`` to disable.

    Returns:
        ``{"state": {key: tensor}, "best_epoch": int, "best_val_top1": float,
        "cached": bool}``.
    """
    tag = f"centralized_{fold.name}_seed{seed}"
    cache_path = Path(cache_dir) / f"{tag}.pt" if cache_dir else None
    fingerprint = config_fingerprint(
        model_cfg["encoder"],
        model_cfg["head"],
        model_cfg["train"],
        model_cfg.get("augment"),
        fold.signers["train"],
        fold.signers["val"],
        seed,
    )
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            return {**payload, "cached": True}
        print(f"[pretrain] {tag}: cached under a different config, recomputing")

    device = resolve_device(model_cfg["train"].get("device", "auto"))
    seed_everything(seed)
    model = build_model(model_cfg)
    outcome = train_model(
        model,
        _tensors(records, fold.train, data_cfg),
        _tensors(records, fold.val, data_cfg),
        model_cfg,
        device=device,
        seed=seed,
    )
    payload = {
        "state": outcome.best_state,
        "best_epoch": outcome.best_epoch,
        "best_val_top1": outcome.best_val,
        "epochs_run": outcome.epochs_run,
        "fold": fold.name,
        "seed": seed,
        "fingerprint": fingerprint,
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return {**payload, "cached": False}


def summarize(name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-seed runs of one experiment.

    Args:
        name: Experiment id, for the printed line.
        runs: The metrics dicts returned by :func:`run_centralized`.

    Returns:
        ``{"top1": mean_std, "top5": mean_std, "per_signer": {signer: mean_std}}``.
    """
    per_signer_keys = sorted({k for r in runs for k in r.get("per_signer", {})})
    summary = {
        "top1": mean_std([r["top1"] for r in runs]),
        "top5": mean_std([r["top5"] for r in runs]),
        "per_signer": {
            key: mean_std([r["per_signer"][key] for r in runs if key in r["per_signer"]])
            for key in per_signer_keys
        },
    }
    print(
        f"[{name}] over {len(runs)} seeds: "
        f"top-1 {format_mean_std(summary['top1'])}, top-5 {format_mean_std(summary['top5'])}"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the centralized experiments from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Non-zero when the sanity gate fails, so ``make train`` stops
        rather than producing numbers from a broken pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "sanity", "E1", "E2"],
        help="'all' runs the sanity gate first and stops if it fails",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="defaults to `seeds` in configs/model.yaml",
    )
    parser.add_argument("--seed", type=int, default=None, help="shorthand for --seeds SEED")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a model-config key, e.g. train.lr=1e-3",
    )
    args = parser.parse_args(argv)

    model_cfg = apply_overrides(load_config(args.config), args.overrides)
    data_cfg = load_config(args.data_config)
    seeds = [args.seed] if args.seed is not None else (args.seeds or model_cfg.get("seeds", [0]))

    if args.experiment in ("all", "sanity"):
        if not run_sanity(model_cfg, data_cfg, seed=seeds[0], results_dir=args.results_dir)[
            "passed"
        ]:
            print(
                "[sanity] gate failed -- the pipeline cannot memorize a 50-clip subset.\n"
                "         Debug it; do not tune hyperparameters and do not report E1/E2.",
                file=sys.stderr,
            )
            return 1
        if args.experiment == "sanity":
            return 0

    wanted = ["E1", "E2"] if args.experiment == "all" else [args.experiment]
    summaries = {
        name: summarize(
            name,
            [
                run_centralized(name, model_cfg, data_cfg, seed=s, results_dir=args.results_dir)
                for s in seeds
            ],
        )
        for name in wanted
    }

    if {"E1", "E2"} <= summaries.keys():
        gap = summaries["E1"]["top1"]["mean"] - summaries["E2"]["top1"]["mean"]
        print(f"\n[RQ1] signer-independent generalization gap E1 - E2 = {gap * 100:.1f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
