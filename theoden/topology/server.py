from __future__ import annotations

import ssl
import time

from ..common import (
    ExecutionResponse,
    ForbiddenOperationError,
    StatusUpdate,
    Transferable,
    Transferables,
)
from ..networking.rabbitmq import ServerToMQInterface
from ..networking.rest import RestServerInterface
from ..networking.storage import FileStorage, FileStorageInterface
from ..operations import *
from ..resources import ResourceManager
from ..resources.meta import CheckpointManager
from ..security.operation_protection import OperationBlackList, OperationWhiteList
from ..watcher import *
from .client_status import TimeoutClientStatusObserver
from .topology import Node, NodeStatus, NodeType, Topology

operation_manager_types = list[Instruction | InstructionBundle | Condition] | None


# currently not in use
class OperationManager(Transferable, is_base_type=True):
    def __init__(
        self,
        initial_operations: operation_manager_types = None,
        opt_in_operations: operation_manager_types = None,
        running_operations: operation_manager_types = None,
        constant_conditions: list[Condition] | None = None,
        operation_protection: OperationWhiteList | OperationBlackList | None = None,
    ) -> None:
        """A class that manages the operations of a server.

        Args:
            initial_operations (list[Instruction | InstructionBundle | Condition], optional): The initial operations to be executed by the server. Defaults to None.
            opt_in_operations (list[Instruction | InstructionBundle | Condition], optional): The opt-in operations to be executed by the server. Defaults to None.
            running_operations (list[Instruction | InstructionBundle | Condition], optional): The running operations to be executed by the server. Defaults to None.
            constant_conditions (list[Condition], optional): The constant conditions to be checked by the server. Defaults to None.
        """

        self.operation_protection = operation_protection

        # check if all operations are allowed
        if (
            self.operation_protection is not None
            and not self.operation_protection.allows(
                initial_operations
                + opt_in_operations
                + running_operations
                + constant_conditions
            )
        ):
            raise ForbiddenOperationError(
                f"OperationManager initialization is not allowed"
            )

        self.initial_operations = initial_operations
        self.opt_in_operations = opt_in_operations
        self.running_operations = running_operations
        self.constant_conditions = constant_conditions

    def save_as_dict(self) -> dict[str, list[dict[str, any]]]:
        return {
            "initial_operations": [op.dict() for op in self.initial_operations],
            "opt_in_operations": [op.dict() for op in self.opt_in_operations],
            "running_operations": [op.dict() for op in self.running_operations],
            "constant_conditions": [cond.dict() for cond in self.constant_conditions],
        }

    @staticmethod
    def load_from_dict(data: dict[str, list[dict[str, any]]]) -> OperationManager:
        return Transferables().to_object(data, OperationManager)

    def process_status_update(self, status_update: StatusUpdate) -> None:
        ...


class Server:
    def __init__(
        self,
        initial_instructions: list[Instruction | InstructionBundle | Condition]
        | None = None,
        node_config: str | None = None,
        run_name: str | None = None,
        communication_address: str | None = None,
        communication_port: int | None = None,
        storage_address: str | None = None,
        storage_port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        operation_protection: OperationWhiteList | OperationBlackList | None = None,
        watcher: list[Watcher] | None = None,
        rabbitmq: bool = True,
        ssl_context: ssl.SSLContext | None = None,
        https: bool = False,
        **kwargs,
    ) -> None:
        """A federated learning server.

        Args:
            initial_instructions (list[Instruction | InstructionBundle | Condition], optional): A list of initial instructions to be executed by the server. Defaults to None.
            initial_nodes (str | None, optional): A string representing the initial nodes to be connected to the server. Defaults to None.
            topology_schema (TopologySchema | None, optional): The topology schema to be used by the server. Defaults to None.
            run_name (str | None, optional): The name of the run. Defaults to None.
            storage_address (str | None, optional): The address of the storage. Defaults to None.
            storage_port (int | None, optional): The port of the storage. Defaults to None.
            rabbitmq (bool, optional): Whether to use RabbitMQ for communication. Defaults to True.
            operation_protection (OperationWhiteList | OperationBlackList | None, optional): The operation protection to be used by the server. Defaults to None.
            watcher (list[Watcher] | None, optional): A list of watchers to be used by the server. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the communication interface.
        """

        self.storage_address = storage_address
        self.storage_port = storage_port
        self.operation_protection = operation_protection
        self.rabbitmq = rabbitmq

        # Initialize the storage interface. This will be used to enable communication with the storage.
        storage_interface = FileStorageInterface(
            storage=None if storage_address else FileStorage(node_config=node_config),
            address=self.storage_address,
            port=self.storage_port,
            https=https,
            username=username or "server",
            password=password or "server",
        )

        # Initialize the Resource Manager.
        # This will hold all the resource_manager that the server has access to and that are to be shared with the nodes.
        self.resources: ResourceManager = ResourceManager(
            __checkpoints__=CheckpointManager(),
            __storage__=storage_interface,
            __watcher__=WatcherPool(self)
            .add(AimMetricCollectorWatcher())
            .add(MetricAggregationWatcher())
            .add(NewBestDetectorWatcher())
            .add(ModelSaverWatcher(model_key="model"))
            .add(TheodenConsoleWatcher())
            .add(watcher or [])
            .notify_all(InitializationNotification(run_name=run_name)),
        )

        # Initialize the topology. This will hold information about the topology of the federated learning system.
        server_node = Node(
            node_name="server",
            node_type=NodeType.SERVER,
            status=NodeStatus.ONLINE,
        )
        self.topology = Topology(
            node_config=node_config,
            watcher_pool=self.resources.watcher,
            observer=TimeoutClientStatusObserver(),
            resource_manager=self.resources,
        ).add_node(server_node)

        # Initialize the private resource register as an empty dictionary.
        # This will hold all the resource_manager that the server has access to and that are not to be shared with the nodes.
        self.private_resource_manager: ResourceManager = ResourceManager()

        # Initialize the history as an empty list. This will hold all the instructions and condition that have been executed by the server.
        self.history: list[Instruction | InstructionBundle | Condition] = []

        # Initialize the operations as an empty list or with the initial instructions. This will hold all the instructions and conditions that are to be executed by the server.
        self.operations: list[Instruction | InstructionBundle | Condition] = (
            initial_instructions if initial_instructions else []
        )

        # Initialize the communication interface as a RestServerInterface. This will be used to enable communication with the nodes.
        if rabbitmq:
            self.communication_interface = ServerToMQInterface(
                server=self,
                host=communication_address,
                port=communication_port or 5672,
                username=username,
                password=username,
                ssl_context=ssl_context,
                **kwargs,
            )
        else:
            self.communication_interface = RestServerInterface(
                server=self, node_config=node_config, **kwargs
            )

        # Add the storage interface to the communication interface and initialize it
        self.communication_interface.add_storage_interface(storage_interface)
        self.communication_interface.init()

    def process_server_request(self, request: ServerRequest) -> ExecutionResponse:
        """Process a server request. This will be called by the communication interface.

        Args:
            request (ServerRequest): The server request to process.

        Returns:
            any: The result of the request.

        Raises:
            ForbiddenOperationError: If the operation is not allowed.
        """

        # Update the last active time of the node
        self.topology.get_client_by_name(request.node_name).last_active = time.time()

        if (
            self.operation_protection is not None
            and not self.operation_protection.allows(request)
        ):
            raise ForbiddenOperationError(
                f"ServerRequest {type(request).__name__} is not allowed"
            )

        # Notify all watchers about the request
        self.resources.watcher.notify_all(ServerRequestNotification(request=request))

        # Set the server of the request to this server and execute it
        execution_response = request.set_server(self).execute()
        return execution_response or ExecutionResponse()

    def process_status_update(self, status_update: StatusUpdate) -> None:
        """Process a status update from a node. This will be called by the communication interface.

        Args:
            status_update (StatusUpdate): The status update to process.
        """

        # Update the last active time of the node
        node_name = status_update.node_name
        self.topology.get_client_by_name(node_name).last_active = time.time()

        # Process the status update with the first operation in the operations list
        self.operations[0].handle_status_update(
            status_update, self.topology, self.resources
        )
