"""FedAvg, and the encoder-only aggregation that makes it FedPer (PLAN.md section 6).

The difference between E4 (FedAvg) and E5 (FedPer) is **not** a different averaging rule.
Both average the payload they are given, weighted by local dataset size; FedPer simply gives
them a payload with no head in it. Keeping the distinction at the client boundary rather than
inside the aggregation means the private parameters are never serialized, never reach the
server process, and never depend on the server implementing a filter correctly -- which is a
stronger privacy statement than "the server promises not to average them", and it is what
``tests/test_fedper.py`` checks in phase 4.

``build_strategy`` therefore returns the same class for both names and differs only in the
prefixes it reports; the E5 *experiment* -- k-shot personalization of the private head -- is
phase 4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from signadapt.federated.parameters import assert_excludes

STRATEGIES = ("fedavg", "fedper")

# Client metrics that identify the client rather than measure it. Averaging a signer id
# produces a number that looks like a metric and means nothing, so these are recorded per
# round as a list (see RecordingFedAvg.aggregate_fit) and excluded from the weighted mean.
IDENTITY_KEYS = frozenset({"signer"})


def weighted_average(metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
    """Aggregate client metrics weighted by each client's number of examples.

    A plain mean would give a signer with 40 clips the same say as one with 320, which is
    not what "accuracy over the federation" means.

    Args:
        metrics: ``(n_examples, metrics)`` per client, as Flower supplies them.

    Returns:
        The weighted mean of every numeric key present in all clients' metrics.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = set.intersection(*(set(m) for _, m in metrics)) if metrics else set()
    return {
        key: sum(n * float(m[key]) for n, m in metrics) / total
        for key in sorted(keys - IDENTITY_KEYS)
        if all(isinstance(m[key], int | float) for _, m in metrics)
    }


class RecordingFedAvg(FedAvg):
    """FedAvg that keeps the per-round history and the latest global parameters.

    ``flwr.simulation.run_simulation`` returns ``None``, but it runs the ``ServerApp`` in the
    calling process, so a strategy instance is still readable afterwards. Holding the state
    here is what lets the simulation write a self-describing result file and hand phase 4 a
    federated checkpoint to personalize from.

    Attributes:
        rounds: One record per round, in order.
        latest_parameters: The most recent aggregated global parameters.
        aggregate_prefixes: The state-dict prefixes this run aggregates, recorded so the
            result file says what was actually shared.
    """

    def __init__(
        self,
        *,
        aggregate_prefixes: Sequence[str] = ("encoder.", "head."),
        private_prefixes: Sequence[str] = (),
        on_round: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the strategy.

        Args:
            aggregate_prefixes: Prefixes the clients transmit, for the record.
            private_prefixes: Prefixes that must never be transmitted, for the record.
            on_round: Called with each round's record, e.g. ``ResultsLogger.log_record``.
            **kwargs: Passed to :class:`flwr.server.strategy.FedAvg`.
        """
        super().__init__(**kwargs)
        self.aggregate_prefixes = tuple(aggregate_prefixes)
        self.private_prefixes = tuple(private_prefixes)
        self.rounds: list[dict[str, Any]] = []
        self.latest_parameters: Parameters | None = kwargs.get("initial_parameters")
        self._on_round = on_round

    def _record(self, server_round: int, **fields: Any) -> None:
        """Merge fields into this round's record, creating it if needed."""
        for existing in self.rounds:
            if existing["round"] == server_round:
                existing.update(fields)
                if self._on_round is not None:
                    self._on_round(existing)
                return
        entry = {"round": server_round, **fields}
        self.rounds.append(entry)
        if self._on_round is not None:
            self._on_round(entry)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Average the client updates and record what happened this round.

        Args:
            server_round: 1-based round index.
            results: Successful client results.
            failures: Failed clients. A silent failure would shrink the federation without
                changing anything visible, so the count is recorded.

        Returns:
            ``(parameters, metrics)`` as FedAvg produces them.
        """
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_parameters = parameters
        self._record(
            server_round,
            n_clients=len(results),
            n_failures=len(failures),
            n_examples=sum(res.num_examples for _, res in results),
            signers=sorted(int(res.metrics.get("signer", -1)) for _, res in results),
            fit=weighted_average([(res.num_examples, res.metrics) for _, res in results]),
            fit_metrics=dict(metrics),
        )
        return parameters, metrics

    def aggregate_evaluate(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate the clients' local evaluations and record them.

        Args:
            server_round: 1-based round index.
            results: Successful client evaluations.
            failures: Failed clients.

        Returns:
            ``(loss, metrics)`` as FedAvg produces them.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self._record(server_round, distributed_loss=loss, distributed=dict(metrics))
        return loss, metrics


def build_strategy(
    name: str,
    *,
    aggregate_prefixes: Sequence[str],
    private_prefixes: Sequence[str] = (),
    on_round: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> RecordingFedAvg:
    """Construct the strategy named in ``configs/fl.yaml``.

    Args:
        name: ``"fedavg"`` or ``"fedper"``.
        aggregate_prefixes: Prefixes the clients transmit. For ``fedper`` this must exclude
            every private prefix, which is checked here rather than trusted.
        private_prefixes: Prefixes that must never be transmitted.
        on_round: Per-round callback.
        **kwargs: Passed to :class:`flwr.server.strategy.FedAvg`.

    Returns:
        The configured strategy.

    Raises:
        ValueError: On an unknown name, or when a ``fedper`` configuration would in fact
            transmit a private prefix -- that misconfiguration turns E5 back into E4 while
            still labelling the result "fedper".
    """
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}, expected one of {STRATEGIES}")
    if name == "fedper":
        assert_excludes(aggregate_prefixes, private_prefixes)
    return RecordingFedAvg(
        aggregate_prefixes=aggregate_prefixes,
        private_prefixes=private_prefixes,
        on_round=on_round,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        **kwargs,
    )
