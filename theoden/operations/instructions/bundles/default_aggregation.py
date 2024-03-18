from __future__ import annotations

from ....resources import StateLoader
from ...commands import *
from ..aggregation.aggregate import AggregationAction, Aggregator
from ..bundles import InstructionBundle
from ..distribution import ClosedDistribution
from ..error import DistributionErrorHandler
from ..selection import BinarySelector


class DefaultAggregationBundle(InstructionBundle):
    def __init__(
        self,
        aggregator: Aggregator,
        train_command: Command,
        client_score_command: CalculateClientScoreCommand | None = None,
        selector: BinarySelector | None = None,
        _no_aggregation: bool = False,
        error_handler: DistributionErrorHandler | None = None,
        set_flag_after_execution: str | list[str] | None = None,
        remove_flag_after_execution: str | list[str] | None = None,
        loader: type[StateLoader] | None = None,
        model_keys: list[str] | None = None,
        optimizer_keys: list[str] | None = None,
        simultaneous_execution: int = 0,
        only_grad: bool = False,
    ) -> None:
        """Bundle for the default aggregation.

        Args:
            aggregator (Aggregator): The aggregator to use for aggregation
            train_command (Command): The command to train the model
            selector (BinarySelector | None, optional): The selector to use for the aggregation. Defaults to None.
            _no_aggregation (bool, optional): Whether to skip the aggregation. Defaults to False.
            error_handler (DistributionErrorHandler | None, optional): The error handler to use for the distribution. Defaults to None.
            set_flag_after_execution (str | list[str] | None, optional): The flag to set after execution. Defaults to None.
            remove_flag_after_execution (str | list[str] | None, optional): The flag to remove after execution. Defaults to None.
            loader (type[StateLoader] | None, optional): The loader to use for loading the state. Defaults to None.
            model_keys (list[str] | None, optional): The keys of the models to load. Defaults to None. The default results in ["model"].
            optimizer_keys (list[str] | None, optional): The keys of the optimizers to load. Defaults to None.
            simultaneous_execution (int, optional): The number of simultaneous executions. Defaults to 0.
            only_grad (bool, optional): Whether to only send the parameters that require gradients. Defaults to False.
        """

        # The aggregation consists of two steps:
        # 1. Load global model, train and send it to the server (happens on the client)
        train_distribution = ClosedDistribution(
            commands=SequentialCommand(
                [
                    *[
                        LoadStateDictCommand(key, loader=loader)
                        for key in model_keys or ["model"]
                    ],
                    *[
                        LoadStateDictCommand(key, loader=loader)
                        for key in optimizer_keys or []
                    ],
                    train_command,
                    *[
                        SendModelToServerCommand(key, only_grad=only_grad)
                        for key in model_keys or ["model"]
                    ],
                    *[
                        SendOptimizerToServerCommand(key)
                        for key in optimizer_keys or []
                    ],
                    *([client_score_command] if client_score_command else []),
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
