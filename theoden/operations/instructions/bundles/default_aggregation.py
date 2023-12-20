from __future__ import annotations

from ....common import Transferable
from ....resources import StateLoader
from ...commands import *
from ..aggregation.aggregate import AggregationAction, Aggregator
from ..bundles import InstructionBundle
from ..distribution import ClosedDistribution
from ..error import DistributionErrorHandler
from ..selection import BinarySelector


class DefaultAggregationBundle(InstructionBundle, Transferable):
    def __init__(
        self,
        aggregator: Aggregator,
        train_command: Command,
        selector: BinarySelector | None = None,
        _no_aggregation: bool = False,
        error_handler: DistributionErrorHandler | None = None,
        set_flag_after_execution: str | list[str] | None = None,
        remove_flag_after_execution: str | list[str] | None = None,
        loader: type[StateLoader] | None = None,
        simultaneous_execution: int = 0,
    ) -> None:
        # The aggregation consists of two steps:
        # 1. Load global model, train and send it to the server (happens on the client)
        train_distribution = ClosedDistribution(
            commands=SequentialCommand(
                [
                    LoadStateDictCommand("model", loader=loader),
                    train_command,
                    SendModelToServerCommand("model", loader=loader),
                ]
            ),
            selector=selector,
            set_flag_after_execution=set_flag_after_execution,
            remove_flag_after_execution=remove_flag_after_execution,
            simultaneous_execution=simultaneous_execution,
            error_handler=error_handler,
        )
        # 2. Aggregate the models on the server (happens on the server)
        aggregation_action = AggregationAction(
            aggregator=aggregator, _no_aggregation=_no_aggregation
        )

        super().__init__([train_distribution, aggregation_action])
