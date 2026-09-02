"""The k-shot adaptation sweep: E4, E5 and E6 (PLAN.md section 6).

Three ways for a new signer to end up with a working model, all measured on the same folds,
the same k values and the same fixed query sets, so the only thing that differs is the method:

===  ===============================================================================
E4   FedAvg across the training signers, then fine-tune the **whole** model on k clips.
     The standard FL baseline. Its k=0 point is the only meaningful zero-shot one: the
     global model arrives with a trained classifier already attached.
E5   FedPer across the training signers -- only the encoder is ever transmitted -- then
     train a **private head** on k clips with the encoder frozen. The proposed method.
     Its k=0 point is chance by construction: there is no head until the signer makes one.
E6   Centralized pretraining on the same signers, then the same private head. The
     non-federated ceiling for E5; the gap between them is what federation costs.
E6M  E6 again, with its pretraining budget matched to what E5's federation consumes.
     E5 beats E6 at every k >= 2, which a ceiling should not allow, and the obvious
     alternative explanation is budget rather than method: see ``PRETRAIN_EPOCHS``.
===  ===============================================================================

Every pretraining excludes the held-out signer, which is what makes the numbers mean
adaptation rather than memory, and pretrainings are cached so the sweep runs one per
(method, fold, seed) rather than one per k.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from signadapt.data.dataset import (
    ClipRecord,
    KeypointDataset,
    Split,
    feasible_k_values,
    kshot_indices,
    load_records,
    loso_folds,
)
from signadapt.federated.simulation import as_fedavg, federated_pretrain
from signadapt.models.model import ENCODER_PREFIX, build_model
from signadapt.train.centralized import pretrain_centralized
from signadapt.train.evaluate import EvalResult
from signadapt.train.loop import evaluate_tensors, resolve_device, stack_dataset, train_model
from signadapt.utils.config import apply_overrides, load_config
from signadapt.utils.metrics import format_mean_std, mean_std
from signadapt.utils.results import ResultsLogger
from signadapt.utils.seeding import seed_everything

# pretrain: where the shared weights come from. load: which of them are loaded into the
# adapted model. adapt: which parameter group the k clips are allowed to move.
METHODS: dict[str, dict[str, str]] = {
    "E4": {"pretrain": "fedavg", "load": "all", "adapt": "full"},
    "E5": {"pretrain": "fedper", "load": "encoder", "adapt": "head"},
    "E6": {"pretrain": "centralized", "load": "encoder", "adapt": "head"},
    "E6M": {"pretrain": "centralized", "load": "encoder", "adapt": "head"},
}

#: Methods whose pretraining runs to a fixed epoch budget instead of early-stopping.
#:
#: E5 outperforms E6 by about two points at every k >= 2, which is the wrong way round for a
#: method and its own non-federated ceiling, so the budget has to be ruled out before the
#: result can be attributed to federation. The two budgets are not comparable under a single
#: number, and which one is "matched" decides the answer:
#:
#: * **Sequential updates per model.** A client runs 2 local epochs over one signer's 320
#:   clips per round: 50 x 2 x 10 = 1000 optimizer steps. E6 selects at epoch 23.6 on average
#:   over 2560 clips, or 1888 steps -- so E6 already gets 1.9x *more* than E5 by this measure,
#:   and the confound does not exist.
#: * **Clip presentations.** Every round, all 8 training clients each pass twice over their
#:   own shard, so the federation consumes 2 epochs of the pooled training set per round and
#:   100 over the run, against E6's 23.6. Here E5 gets 4.2x more.
#:
#: E6M matches the second, because it is the only one under which E6 is short-changed. Early
#: stopping is switched off so the full budget is actually spent, while validation selection
#: is kept, so E6M differs from E6 in budget alone rather than in how its checkpoint is
#: chosen. If E6M closes the gap, the E5-over-E6 result is an artefact of the schedule; if it
#: does not, federated averaging is doing something the pooled run does not.
PRETRAIN_EPOCHS: dict[str, int] = {"E6M": 100}

DESCRIPTIONS = {
    "E4": "FedAvg pretraining across the training signers, then fine-tuning the whole model "
    "on k clips of the held-out signer (PLAN.md section 6).",
    "E5": "FedPer: only the encoder is federated, and the held-out signer trains a private "
    "head on k clips with the encoder frozen. The proposed method, RQ3.",
    "E6": "Centralized pretraining on the same signers, then the same private head. The "
    "non-federated upper bound for E5.",
    "E6M": "E6 with its pretraining budget matched to the clip presentations E5's federation "
    "consumes (100 epochs, no early stopping): does the budget explain E5 > E6?",
}


def _tensors(
    records: list[ClipRecord], indices: tuple[int, ...], data_cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a set of record indices into stacked tensors."""
    return stack_dataset(KeypointDataset(records, indices, data_cfg))


def adapt_config(
    model_cfg: dict[str, Any], fl_cfg: dict[str, Any], *, adapt: str
) -> dict[str, Any]:
    """Build the training config for local adaptation.

    Args:
        model_cfg: Loaded model config.
        fl_cfg: Loaded FL config; its ``personalization`` block supplies the schedule.
        adapt: ``"head"`` or ``"full"``.

    Returns:
        A copy of ``model_cfg`` with ``train`` overridden. ``"full"`` uses ``adapt_lr_full``
        rather than ``adapt_lr``: the head starts from scratch and wants a large rate, while
        a whole pretrained encoder at that rate is simply destroyed, and reporting E4 under a
        rate that wrecks it would be a rigged comparison rather than a result.

    Raises:
        ValueError: On an unknown adaptation mode.
    """
    if adapt not in ("head", "full"):
        raise ValueError(f"adapt must be 'head' or 'full', got {adapt!r}")
    spec = fl_cfg["personalization"]
    lr = float(spec["adapt_lr_full"] if adapt == "full" else spec["adapt_lr"])
    cfg = {**model_cfg, "train": {**model_cfg["train"]}}
    cfg["train"] |= {
        "epochs": int(spec["adapt_epochs"]),
        "lr": lr,
        "early_stopping_patience": 0,
        "warmup_epochs": min(
            int(model_cfg["train"].get("warmup_epochs", 0)), int(spec["adapt_epochs"]) // 4
        ),
    }
    return cfg


def pretrained_state(
    method: str,
    records: list[ClipRecord],
    fold: Split,
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    seed: int,
    cache_dir: str | None = "data/checkpoints/pretrain",
) -> dict[str, Any]:
    """Produce (or reuse) the shared weights this method starts a fold from.

    Args:
        method: ``"E4"``, ``"E5"`` or ``"E6"``.
        records: All records.
        fold: The LOSO fold. Only its training signers are ever used here.
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded FL config.
        seed: Seed.
        cache_dir: Where pretrainings are cached, or ``None`` to disable.

    Returns:
        ``{"state": {key: tensor}, "cached": bool, ...}``.

    Raises:
        ValueError: On an unknown method.
    """
    source = METHODS[method]["pretrain"]
    if source == "centralized":
        budget = PRETRAIN_EPOCHS.get(method)
        if budget is not None:
            model_cfg = {
                **model_cfg,
                "train": {**model_cfg["train"], "epochs": budget, "early_stopping_patience": 0},
            }
            # A separate directory, because the tag pretrain_centralized builds is blind to
            # the budget: sharing one would let this run overwrite the early-stopped
            # checkpoints that every committed E6 number was produced from, and the overwrite
            # is silent -- the fingerprint check only prints, then saves over them.
            cache_dir = str(Path(cache_dir) / f"budget{budget}") if cache_dir else None
        return pretrain_centralized(
            records,
            fold,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            seed=seed,
            cache_dir=cache_dir,
        )
    if source == "fedavg":
        return federated_pretrain(
            records,
            fold,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=as_fedavg(fl_cfg),
            seed=seed,
        )
    if source == "fedper":
        # configs/fl.yaml already names fedper and lists encoder-only prefixes; build_strategy
        # re-checks that they exclude every private prefix before a single round runs.
        return federated_pretrain(
            records,
            fold,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            fl_cfg=fl_cfg,
            seed=seed,
            cache_dir=cache_dir,
        )
    raise ValueError(f"unknown pretraining source {source!r}")


def adapt_and_evaluate(
    method: str,
    state: dict[str, torch.Tensor],
    support: tuple[torch.Tensor, torch.Tensor],
    query: tuple[torch.Tensor, torch.Tensor],
    *,
    cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    records: list[ClipRecord] | None = None,
    query_indices: tuple[int, ...] | None = None,
) -> tuple[EvalResult, float]:
    """Adapt a pretrained model to one signer's k clips and score it on their query set.

    Args:
        method: ``"E4"``, ``"E5"`` or ``"E6"``.
        state: The pretrained parameters, keyed by full state-dict name.
        support: ``(X, y)`` for the k clips; may be empty for k=0.
        query: ``(X, y)`` for the fixed evaluation set.
        cfg: Config from :func:`adapt_config`.
        device: Where to run.
        seed: Seed for the fresh head and for training.
        records: All records, for the per-handedness breakdown.
        query_indices: Record indices of the query set, in order.

    Returns:
        ``(result, seconds)``.
    """
    spec = METHODS[method]
    seed_everything(seed)
    model = build_model(cfg)

    # E5 and E6 take only the encoder: the head is the signer's own and starts from scratch.
    # E4 takes both, because the whole point of that baseline is that the classifier arrives
    # already trained on the other signers.
    payload = (
        state
        if spec["load"] == "all"
        else {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    )
    model.load_state_dict(payload, strict=False)
    model.freeze_encoder(spec["adapt"] == "head")

    seconds = 0.0
    if support[1].numel() > 0:
        outcome = train_model(model, support, None, cfg, device=device, seed=seed)
        seconds = outcome.seconds

    result = evaluate_tensors(
        model, query, cfg, device=device, records=records, indices=query_indices
    )
    return result, seconds


def run_sweep(
    method: str,
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fl_cfg: dict[str, Any],
    seed: int = 0,
    results_dir: str = "results",
    cache_dir: str | None = "data/checkpoints/pretrain",
) -> dict[str, Any]:
    """Run one method's full leave-one-signer-out k-sweep for one seed.

    Args:
        method: ``"E4"``, ``"E5"`` or ``"E6"``.
        model_cfg: Loaded model config.
        data_cfg: Loaded data config.
        fl_cfg: Loaded FL config.
        seed: Seed for the k-shot draw, the fresh head and the adaptation.
        results_dir: Where the result JSON goes.
        cache_dir: Where pretrainings are cached, or ``None`` to disable.

    Returns:
        ``{"by_k": {k: mean_std across signers}, "k_values": [...]}``.

    Raises:
        ValueError: On an unknown method.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(METHODS)}")

    spec = fl_cfg["personalization"]
    query_repetitions = int(spec.get("query_repetitions", 1))
    cfg = adapt_config(model_cfg, fl_cfg, adapt=METHODS[method]["adapt"])
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
        method,
        config={"model": cfg, "data": data_cfg, "fl": fl_cfg},
        seed=seed,
        results_dir=results_dir,
        description=DESCRIPTIONS[method],
        extra={
            "method": METHODS[method],
            "k_values": list(k_values),
            "skipped_k": list(skipped),
            "query_repetitions": query_repetitions,
            "n_folds": len(folds),
        },
    ) as log:
        for fold in folds:
            signer = fold.signers["test"][0]
            pretrained = pretrained_state(
                method,
                records,
                fold,
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                fl_cfg=fl_cfg,
                seed=seed,
                cache_dir=cache_dir,
            )
            for k in k_values:
                support, query = kshot_indices(
                    records, fold.test, k, seed=seed, query_repetitions=query_repetitions
                )
                result, seconds = adapt_and_evaluate(
                    method,
                    pretrained["state"],
                    _tensors(records, support, data_cfg),
                    _tensors(records, query, data_cfg),
                    cfg=cfg,
                    device=device,
                    seed=seed * 100 + signer,
                    records=records,
                    query_indices=query,
                )
                by_k[k].append(result.top1)
                log.log_record(
                    k=k,
                    signer=signer,
                    participant=records[query[0]].participant,
                    top1=result.top1,
                    top5=result.top5,
                    loss=result.loss,
                    n_support=len(support),
                    n_query=len(query),
                    per_handedness=result.per_handedness,
                    pretrain_cached=pretrained["cached"],
                    seconds=seconds,
                )
            print(
                f"[{method}] seed {seed} signer {signer:2d}: "
                + "  ".join(f"k={k}:{by_k[k][-1]:.3f}" for k in k_values)
            )
            # Checkpoint the result file after every fold: a sweep is long enough that
            # losing it to a crash at fold 9 would be a real cost.
            log.save()

        summary = {
            "by_k": {str(k): mean_std(v) for k, v in by_k.items()},
            "k_values": list(k_values),
            "skipped_k": list(skipped),
        }
        log.set_metrics(**summary)

    for k in k_values:
        print(
            f"[{method}] seed {seed} k={k}: {format_mean_std(mean_std(by_k[k]))} "
            f"across {len(by_k[k])} signers"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the k-shot adaptation sweep from the command line.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fl.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["E5", "E4", "E6"],
        choices=sorted(METHODS),
        help="E5 first: it is the one under test",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="shorthand for --seeds SEED")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--cache-dir",
        default="data/checkpoints/pretrain",
        help="reuse pretrainings across k and across invocations",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    fl_cfg = apply_overrides(load_config(args.config), args.overrides)
    model_cfg = load_config(args.model_config)
    data_cfg = load_config(args.data_config)
    seeds = args.seeds or (
        [args.seed] if args.seed is not None else fl_cfg["personalization"]["seeds"]
    )

    summaries: dict[str, list[dict[str, Any]]] = {}
    for method in args.methods:
        summaries[method] = [
            run_sweep(
                method,
                model_cfg=model_cfg,
                data_cfg=data_cfg,
                fl_cfg=fl_cfg,
                seed=s,
                results_dir=args.results_dir,
                cache_dir=args.cache_dir,
            )
            for s in seeds
        ]

    print(f"\n[sweep] mean across signers and {len(seeds)} seeds:")
    k_values = summaries[args.methods[0]][0]["k_values"]
    header = "  k   " + "".join(f"{m:>18}" for m in args.methods)
    print(header)
    for k in k_values:
        cells = "".join(
            f"{format_mean_std(mean_std([r['by_k'][str(k)]['mean'] for r in summaries[m]])):>18}"
            for m in args.methods
        )
        print(f"  {k:<4}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
