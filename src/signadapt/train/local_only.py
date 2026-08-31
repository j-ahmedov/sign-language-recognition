"""E3: a held-out signer trains from scratch on k of their own examples.

This is the null hypothesis for RQ3 -- the "you don't need federated learning" baseline
(PLAN.md section 6). If a new signer can reach usable accuracy from their own handful of
labelled clips alone, federation is complexity without payoff, and saying so is a result.
The baseline is therefore made as strong as it honestly can be: augmentation stays on, and
the model is the same architecture E5 personalizes, so the only difference between E3 and E5
is where the encoder's weights came from.

k=0 is included and is not a training run: with no shared encoder and no local examples there
is nothing to fit, so the point is the accuracy of the freshly initialized model, which is
chance (about 1/64 on LSA64). It anchors the left-hand end of the adaptation curve.
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from signadapt.data.dataset import (
    ClipRecord,
    KeypointDataset,
    feasible_k_values,
    kshot_indices,
    load_records,
    loso_folds,
)
from signadapt.models.model import build_model
from signadapt.train.loop import evaluate_tensors, resolve_device, stack_dataset, train_model
from signadapt.utils.config import apply_overrides, load_config
from signadapt.utils.metrics import format_mean_std, mean_std
from signadapt.utils.results import ResultsLogger
from signadapt.utils.seeding import seed_everything


def _tensors(
    records: list[ClipRecord], indices: tuple[int, ...], data_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a set of record indices into stacked tensors."""
    return stack_dataset(KeypointDataset(records, indices, data_cfg))


def adapt_config(model_cfg: dict[str, Any], fl_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the training config used for local adaptation.

    Args:
        model_cfg: Loaded ``configs/model.yaml``.
        fl_cfg: Loaded ``configs/fl.yaml``; its ``personalization`` block supplies the epoch
            count and learning rate.

    Returns:
        A copy of ``model_cfg`` with ``train`` overridden. Early stopping is disabled because
        there is no validation set to stop on -- a signer with k=1 has one clip per sign and
        spending any of it on validation would change the experiment.
    """
    spec = fl_cfg["personalization"]
    cfg = {**model_cfg, "train": {**model_cfg["train"]}}
    cfg["train"] |= {
        "epochs": int(spec["adapt_epochs"]),
        "lr": float(spec["adapt_lr"]),
        "early_stopping_patience": 0,
        "warmup_epochs": min(
            int(model_cfg["train"].get("warmup_epochs", 0)), int(spec["adapt_epochs"]) // 4
        ),
    }
    return cfg


def run_local_only(
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    seed: int = 0,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Run E3 over every leave-one-signer-out fold and every feasible k, for one seed.

    Args:
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded FL config; supplies the k sweep and the adaptation schedule.
        seed: Seed for the k-shot draw and for training.
        results_dir: Where the result JSON goes.

    Returns:
        ``{"by_k": {k: mean_std across signers}, "k_values": [...], "skipped_k": [...]}``.
    """
    spec = fl_cfg["personalization"]
    query_repetitions = int(spec.get("query_repetitions", 1))
    cfg = adapt_config(model_cfg, fl_cfg)
    device = resolve_device(model_cfg["train"].get("device", "auto"))

    seed_everything(seed)
    records = load_records(
        data_cfg["dataset"]["cache_dir"],
        exclude_signers=data_cfg["splits"].get("exclude_signers", ()),
    )
    k_values, skipped = feasible_k_values(
        records, spec["k_values"], query_repetitions=query_repetitions
    )
    folds = loso_folds(records, seed=seed)

    by_k: dict[int, list[float]] = {k: [] for k in k_values}
    with ResultsLogger(
        "E3",
        config={"model": cfg, "data": data_cfg, "fl": fl_cfg},
        seed=seed,
        results_dir=results_dir,
        description=(
            "Local-only from scratch: each held-out signer trains their own model on k of "
            "their own clips and is evaluated on a query set that is fixed across k. The "
            "null hypothesis for RQ3 (PLAN.md section 6)."
        ),
        extra={
            "k_values": list(k_values),
            "skipped_k": list(skipped),
            "query_repetitions": query_repetitions,
            "n_folds": len(folds),
        },
    ) as log:
        for fold in folds:
            signer = fold.signers["test"][0]
            for k in k_values:
                support, query = kshot_indices(
                    records, fold.test, k, seed=seed, query_repetitions=query_repetitions
                )
                seed_everything(seed * 100 + signer)
                model = build_model(cfg)
                seconds = 0.0
                if k > 0:
                    outcome = train_model(
                        model,
                        _tensors(records, support, data_cfg),
                        None,
                        cfg,
                        device=device,
                        seed=seed * 100 + signer,
                    )
                    seconds = outcome.seconds
                result = evaluate_tensors(
                    model,
                    _tensors(records, query, data_cfg),
                    cfg,
                    device=device,
                    records=records,
                    indices=query,
                )
                by_k[k].append(result.top1)
                log.log_record(
                    k=k,
                    signer=signer,
                    top1=result.top1,
                    top5=result.top5,
                    loss=result.loss,
                    n_support=len(support),
                    n_query=len(query),
                    per_handedness=result.per_handedness,
                    seconds=seconds,
                )
            print(
                f"[E3] seed {seed} signer {signer:2d}: "
                + "  ".join(f"k={k}:{by_k[k][-1]:.3f}" for k in k_values)
            )

        summary = {
            "by_k": {str(k): mean_std(v) for k, v in by_k.items()},
            "k_values": list(k_values),
            "skipped_k": list(skipped),
        }
        log.set_metrics(**summary)

    for k in k_values:
        print(
            f"[E3] seed {seed} k={k}: {format_mean_std(mean_std(by_k[k]))} across "
            f"{len(by_k[k])} signers"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run E3 from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fl.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="shorthand for --seeds SEED")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    fl_cfg = apply_overrides(load_config(args.config), args.overrides)
    model_cfg = load_config(args.model_config)
    data_cfg = load_config(args.data_config)
    seeds = args.seeds or (
        [args.seed] if args.seed is not None else fl_cfg["personalization"]["seeds"]
    )

    runs = [
        run_local_only(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=fl_cfg,
            seed=s,
            results_dir=args.results_dir,
        )
        for s in seeds
    ]
    print(f"\n[E3] over {len(runs)} seeds, mean across signers and seeds:")
    for k in runs[0]["k_values"]:
        pooled = [r["by_k"][str(k)]["mean"] for r in runs]
        print(f"  k={k}: {format_mean_std(mean_std(pooled))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
