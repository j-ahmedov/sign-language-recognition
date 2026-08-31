"""Flower client, aggregation strategies and the single-machine simulation."""

from signadapt.federated.client import SignerClient, make_client_fn, write_partition
from signadapt.federated.parameters import (
    assert_excludes,
    get_parameters,
    set_parameters,
    shared_keys,
)
from signadapt.federated.strategy import RecordingFedAvg, build_strategy, weighted_average

__all__ = [
    "RecordingFedAvg",
    "SignerClient",
    "assert_excludes",
    "build_strategy",
    "get_parameters",
    "make_client_fn",
    "set_parameters",
    "shared_keys",
    "weighted_average",
    "write_partition",
]
