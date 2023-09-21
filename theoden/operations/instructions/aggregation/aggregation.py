import torch

import io

from .aggregator import Aggregator
from ..instruction import Instruction, NodeCommandStatus
from ..distribute import Distributor
from ..status_handler import StatusHandler, BaseHandler
from ...commands import (
    SendModelToServerCommand,
    SendOptimizerToServerCommand,
    Command,
    SequentialCommand,
)
from ....common import StatusUpdate, Transferable
from ....topology.topology_register import TopologyRegister
from ....resources import ResourceRegister
from ....resources.meta import CheckpointManager, DictCheckpoint
from ....watcher import (
    WatcherPool,
    StatusUpdateNotification,
    AggregationCompletedNotification,
)


class AggregationStatusHandler(StatusHandler, Transferable):
    def handle_status_update(
        self,
        instruction: "AggregateInstruction",
        status_update: StatusUpdate,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> None:
        if (
            status_update.status == NodeCommandStatus.FINISHED.value
            and status_update.contains_files()
        ):
            if status_update.datatype == SendModelToServerCommand.__name__:
                for file_name, resource in status_update.response.get_files().items():
                    instruction.aggregator.process_resource(
                        "model",
                        file_name,
                        status_update.node_uuid,
                        torch.load(io.BytesIO(resource)),
                    )

            elif status_update.datatype == SendOptimizerToServerCommand.__name__:
                for file_name, resource in status_update.response.get_files().items():
                    instruction.aggregator.process_resource(
                        "optimizer",
                        file_name,
                        status_update.node_uuid,
                        torch.load(io.BytesIO(resource)),
                    )


class AggregateInstruction(Instruction, Transferable):
    def __init__(
        self,
        wrapped_object: Command,
        aggregator: Aggregator,
        distributor: Distributor | None = None,
        communication_rounds: int | None = None,
        status_handler: StatusHandler | None = None,
        has_base_handler: bool = True,
        _no_aggregation: bool = False,
        **kwargs,
    ):
        super().__init__(
            wrapped_object=wrapped_object,
            distributor=distributor,
            block=True,
            status_handler=status_handler
            if status_handler
            else [AggregationStatusHandler()],
            has_base_handler=has_base_handler,
            **kwargs,
        )
        self.aggregator = aggregator
        self.communication_rounds = communication_rounds
        self._no_aggregation = _no_aggregation

    def on_init(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ):
        """This method is used to add commands to the wrapped object before the execution starts.

        This is necessary to add the commands that update the resources before the next training round starts.

        Args:
            topology_register (TopologyRegister): The topology register.
            resource_register (ResourceRegister): The resource register.
        """
        if not self._no_aggregation:
            self._add_commands_to_wrapped(
                pre_commands=self.aggregator.distribute_commands(
                    topology_register, resource_register
                ),
                post_commands=self.aggregator.required_resources_for_aggregation(),
            )

    def on_finish(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ) -> None | Instruction:
        # if no aggregation is required, return None. This is mainly the case when training on a single node
        if self._no_aggregation:
            return None

        aggregated_resources = {}

        for resource_type, resources in self.aggregator.resources.items():
            aggregated_resources[resource_type] = {}
            for resource_key, resource in resources.items():
                # get global model from checkpoint manager
                cm = resource_register.gr(
                    "__checkpoints__", assert_type=CheckpointManager
                )

                # aggregate resources using the aggregators aggregate method
                aggregated = self.aggregator.aggregate(
                    resource_type=resource_type,
                    resource_key=resource_key,
                    resources=resource,
                    topology_register=topology_register,
                    resource_register=resource_register,
                )

                # save aggregated resources in checkpoint manager
                cm.register_checkpoint(
                    resource_type=resource_type,
                    resource_key=resource_key,
                    checkpoint_key="__global__",
                    checkpoint=DictCheckpoint(aggregated),
                )

        # remove saved state_dict from aggregator
        self.aggregator.resources.reset()

        resource_register.gr("__watcher__", assert_type=WatcherPool).notify_all(
            AggregationCompletedNotification()
        )
