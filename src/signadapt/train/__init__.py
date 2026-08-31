"""Centralized training, local-only training and evaluation."""

from signadapt.train.evaluate import EvalResult, evaluate, predict
from signadapt.train.loop import (
    TrainOutcome,
    evaluate_tensors,
    make_loader,
    resolve_device,
    stack_dataset,
    train_model,
)

__all__ = [
    "EvalResult",
    "TrainOutcome",
    "evaluate",
    "evaluate_tensors",
    "make_loader",
    "predict",
    "resolve_device",
    "stack_dataset",
    "train_model",
]
