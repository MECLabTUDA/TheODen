from __future__ import annotations

import ssl
import time

from ..common import ExecutionResponse, ForbiddenOperationError, StatusUpdate
from ..datasets import *
from ..models import *
from ..networking.rabbitmq import ServerToMQInterface
from ..networking.rest import RestServerInterface
from ..networking.storage import FileStorage, FileStorageInterface
from ..operations import *
from ..resources import *
from ..resources import ResourceManager
from ..resources.meta import CheckpointManager
from ..security.operation_protection import OperationBlackList, OperationWhiteList
from ..watcher import *
from .client_status import TimeoutClientStatusObserver
from .manager import OperationManager
from .topology import Node, NodeStatus, NodeType, Topology

import logging
logger = logging.getLogger(__name__)

class Server:
    def __init__(
        self,
        initial_instructions: (
            list[Instruction | InstructionBundle | Condition] | None
        ) = None,
        permanent_conditions: list[Condition] | None = None,
        open_distribution: OpenDistribution | None = None,
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
        use_aim: bool = False,
        start_console: bool = True,
        choosing_metric: str | None = None,
        metric_type: str = None,
        lower_is_better: bool = False,
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
            .add(AimMetricCollectorWatcher() if use_aim else [])
            .add(MetricAggregationWatcher())
            .add(NewBestDetectorWatcher(metric=choosing_metric, metric_type=metric_type, lower_is_better=lower_is_better))
            .add(BestModelSaverWatcher(model_key="model", listen_to=choosing_metric)) #TODO maybe "model" should not be hardcoded
            .add(TheodenConsoleWatcher() if start_console else [])
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
            observer=TimeoutClientStatusObserver(20),
            resource_manager=self.resources,
        ).add_node(server_node)

        # Initialize the operation manager. This will be used to execute the operations.
        self.operation_manager = OperationManager(
            operations=initial_instructions or [],
            operation_protection=self.operation_protection,
            open_distribution=open_distribution,
            constant_conditions=permanent_conditions or [],
        )

        # Initialize the communication interface as a RestServerInterface. This will be used to enable communication with the nodes.
        if rabbitmq:
            self.communication_interface = ServerToMQInterface(
                server=self,
                host=communication_address,
                port=communication_port or 5672,
                username=username,
                password=password,
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

        if isinstance(request, PullCommandRequest):
            logger.debug(f"Processing server request: {request}")
        else:
            logger.info(f"Processing server request: {request}")

        # Update the last active time of the client
        self.topology.get_client_by_name(request.client_name).last_active = time.time()

        if (
            self.operation_protection is not None
            and not self.operation_protection.allows(request)
        ):
            logger.error(f"ServerRequest {type(request).__name__} is not allowed")
            raise ForbiddenOperationError(
                f"ServerRequest {type(request).__name__} is not allowed"
            )

        # Notify all watchers about the request
        self.resources.watcher.notify_all(ServerRequestNotification(request=request))

        # Set the server of the request to this server and execute it
        execution_response = request.set_server(self).execute()

        if execution_response.data:
            logger.info(f"Execution response to {request.client_name}: {execution_response.__str__()}")
        else:
            logger.debug(f"Execution response to {request.client_name}: {execution_response.__str__()}")

        return execution_response or ExecutionResponse()

    def process_status_update(self, status_update: StatusUpdate) -> None:
        """Process a status update from a client. This will be called by the communication interface.

        Args:
            status_update (StatusUpdate): The status update to process.
        """

        logger.info(f"Processing status update: {status_update}")

        # Update the last active time of the client
        # ToDo set onloine if offline
        client_name = status_update.client_name
        self.topology.get_client_by_name(client_name).last_active = time.time()

        # Process the status update with the first operation in the operations list
        self.operation_manager.process_status_update(
            status_update, self.topology, self.resources
        )
