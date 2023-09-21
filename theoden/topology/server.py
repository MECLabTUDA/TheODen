from __future__ import annotations

import yaml

from ..operations import *
from ..common import StatusUpdate, Transferable
from ..resources import ResourceRegister
from ..resources.meta import CheckpointManager
from .topology_register import TopologyRegister, TopologySchema
from ..networking.rest import RestServerInterface
from ..watcher import (
    WatcherPool,
    StatusUpdateNotification,
    AimMetricCollectorWatcher,
    MetricAggregationWatcher,
    NewBestDetectorWatcher,
    ModelSaverWatcher,
    InitializationNotification,
)


# currently not in use
class OperationManager(Transferable, is_base_type=True):
    def __init__(
        self,
        initial_operations: list[Instruction | InstructionGroup | Stopper],
        opt_in_operations: list[Instruction | InstructionGroup | Stopper],
        running_operations: list[Instruction | InstructionGroup | Stopper],
    ) -> None:
        self.initial_operations = initial_operations
        self.opt_in_operations = opt_in_operations
        self.running_operations = running_operations

    def save_as_dict(self) -> dict[str, list[dict[str, any]]]:
        return {
            "initial_operations": [op.dict() for op in self.initial_operations],
            "opt_in_operations": [op.dict() for op in self.opt_in_operations],
            "running_operations": [op.dict() for op in self.running_operations],
        }

    def save_as_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.save_as_dict(), f)

    @staticmethod
    def load_from_dict(data: dict[str, list[dict[str, any]]]) -> OperationManager:
        return OperationManager(
            initial_operations=[
                Instruction.init_from_dict(op) for op in data["initial_operations"]
            ],
            opt_in_operations=[
                Instruction.init_from_dict(op) for op in data["opt_in_operations"]
            ],
            running_operations=[
                Instruction.init_from_dict(op) for op in data["running_operations"]
            ],
        )

    def process_status_update(self, status_update: StatusUpdate) -> None:
        ...


class Server:
    def __init__(
        self,
        initial_instructions: list[Instruction | InstructionGroup | Stopper] = None,
        initial_nodes: str | None = None,
        topology_schema: TopologySchema | None = None,
        run_name: str | None = None,
        **kwargs,
    ) -> None:
        """A federated learning server.

        Args:
            initial_instructions (list[Instruction | InstructionGroup | Stopper], optional): A list of initial instructions to be executed by the server. Defaults to None.
            initial_nodes (str | None, optional): A string representing the initial nodes to be connected to the server. Defaults to None.
            topology_schema (TopologySchema | None, optional): The topology schema to be used by the server. Defaults to None.
        """

        # Initialize the topology register with the initial nodes and the topology schema. This will hold information about the topology of the federated learning system.
        self.topology_register = TopologyRegister(
            initial_nodes=initial_nodes, schema=topology_schema
        )

        # Initialize the ResourceRegister as an empty dictionary. This will hold all the resources that the server has access to and that are to be shared with the nodes.
        self.resource_register: ResourceRegister = ResourceRegister(
            __checkpoints__=CheckpointManager(),
            __watcher__=WatcherPool(self)
            .add(AimMetricCollectorWatcher())
            .add(MetricAggregationWatcher())
            .add(NewBestDetectorWatcher())
            .add(ModelSaverWatcher(model_key="model"))
            .notify_all(InitializationNotification(run_name=run_name)),
        )

        # Initialize the private resource register as an empty dictionary.
        # This will hold all the resources that the server has access to and that are not to be shared with the nodes.
        self.private_resource_register: ResourceRegister = ResourceRegister()

        # Initialize the history as an empty list. This will hold all the instructions and stopper that have been executed by the server.
        self.history: list[Instruction | InstructionGroup | Stopper] = []

        # Initialize the operations as an empty list or with the initial instructions. This will hold all the instructions and stoppers that are to be executed by the server.
        self.operations: list[Instruction | InstructionGroup | Stopper] = (
            initial_instructions if initial_instructions else []
        )

        # Initialize the communication interface as a RestServerInterface. This will be used to enable communication with the nodes.
        self.communication_interface = RestServerInterface(server=self, **kwargs)

    def process_server_request(self, request: ServerRequest) -> any:
        """Process a server request. This will be called by the communication interface.

        Args:
            request (ServerRequest): The server request to process.

        Returns:
            any: The result of the request.
        """
        return request.execute()

    def process_status_update(self, status_update: StatusUpdate) -> None:
        """Process a status update from a node. This will be called by the communication interface.

        Args:
            status_update (StatusUpdate): The status update to process.
        """

        self.resource_register.watcher.notify_all(
            StatusUpdateNotification(status_update=status_update)
        )

        self.operations[0].process_status_update(
            status_update,
            self.topology_register,
            self.resource_register,
        )
