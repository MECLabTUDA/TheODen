# import torch.multiprocessing as mp
import asyncio
import ssl
from multiprocessing import Event, Manager, Process

import requests

from ..common import (
    ForbiddenOperationError,
    ServerRequestError,
    StatusUpdate,
    Transferables,
    UnauthorizedError,
)
from ..networking.rabbitmq import ClientToMQInterface
from ..networking.rest import RestClientInterface
from ..networking.storage import FileStorageInterface
from ..operations import Command, ServerRequest
from ..resources.resource import ResourceManager
from ..security.operation_protection import OperationBlackList, OperationWhiteList

import logging
logger = logging.getLogger(__name__)

class Client:
    def __init__(
        self,
        communication_address: str = "localhost",
        communication_port: int | None = None,
        resource_address: str | None = None,
        resource_port: int | None = None,
        username: str = "dummy",
        password: str = "dummy",
        ping_interval: float = 1.0,
        rabbitmq: bool = True,
        ssl: bool = False,
        ssl_context: ssl.SSLContext | None = None,
        operation_protection: OperationWhiteList | OperationBlackList | None = None,
        initial_commands: list[Command] | None = None,
    ) -> None:
        """A federated learning client.

        Args:
            communication_address (str): The address of the server.
            communication_port (int, optional): The port of the server. Defaults to None.
            resource_address (str, optional): The address of the resource_manager server. Defaults to None.
            resource_port (int, optional): The port of the resource_manager server. Defaults to None.
            username (str, optional): The username of the client. Defaults to "dummy".
            password (str, optional): The password of the client. Defaults to "dummy".
            ping_interval (float, optional): The interval at which the client will ping the server. Defaults to 1.0.
            rabbitmq (bool, optional): Whether to use RabbitMQ for communication. Defaults to True.
            operation_protection (OperationWhiteList | OperationBlackList | None, optional): A list of operations that the client is allowed to execute. Defaults to None.
            initial_commands (list[Command] | None, optional): A list of commands to be executed by the client. Defaults to None.
        """

        # Initialize the command queue and resource register as empty dictionaries
        self.ping_interval = ping_interval
        self.operation_protection = operation_protection

        # Initialize the command queue as a list. This will hold all the commands that the client has to execute.
        self.command_queue: list[dict] = Manager().list(
            [] if initial_commands is None else initial_commands
        )

        # Initialize the resource register. This will hold all the resource_manager that are required for the commands.
        self.resources: ResourceManager = ResourceManager(
            __storage__=FileStorageInterface(
                username=username,
                password=password,
                address=resource_address or communication_address,
                port=resource_port or 8000,
                https=ssl,
            ),
            device="cuda",
        )

        # Initialize the network interface
        if rabbitmq:
            self.network_interface = ClientToMQInterface(
                self.command_queue,
                host=communication_address,
                port=communication_port or 5672,
                username=username,
                password=password,
                ping_interval=ping_interval,
                ssl_context=ssl_context,
            )
        else:
            self.network_interface = RestClientInterface(
                command_queue=self.command_queue,
                address=communication_address,
                port=communication_port or 8000,
                username=username,
                password=password,
                ping_interval=ping_interval,
                https=ssl,
            )
        # Add the storage interface to the network interface
        self.network_interface.add_storage_interface(self.resources.storage)

        # Initialize the processes as an empty list
        self.processes: list[Process] = []

        self.stop_event = Event()

    def _run_client(self, fn, *args):
        # Run an async function in an event loop
        asyncio.run(fn(*args))

    def start(self):
        """Start the client.

        This method will register the client with the server and start the command queue and serverrequests queue.
        """

        # Create two separate processes to execute command queue and serverrequests queue
        # target is the function to be called by the process
        # args is the arguments to be passed to the function

        try:
            command_process = Process(
                target=self._run_client,
                args=(self.start_command_queue,),
            )
            request_process = Process(
                target=self._run_client,
                args=(self.network_interface.start_request_loop, self.stop_event),
            )

            # Append the processes to the processes list
            self.processes.extend([command_process, request_process])

            # Start the processes
            command_process.start()
            request_process.start()
            self.network_interface.start()

            command_process.join()
            request_process.join()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        # Set the stop event
        self.stop_event.set()

        # Reset the processes list
        self.processes = []

    async def start_command_queue(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Pop the first command from the command queue
                command_json = self.command_queue.pop(0)

                # Convert the command from json to a Command object
                command: Command = Transferables().to_object(command_json, Command)

                # Check if the command and all subcommands are allowed
                if self.operation_protection is not None:
                    # get all commands inside the command tree
                    command_tree = command.get_command_tree()

                    # check if all commands are allowed
                    for command in command_tree:
                        if not self.operation_protection.allows(command):
                            raise ForbiddenOperationError(
                                f"Command {command.__name__} is not allowed"
                            )

                logger.info(f"Executing command {command.__class__.__name__}")

                # Execute the command on the client
                command.set_client(self)()

            # If the command queue is empty, do nothing
            except IndexError:
                # sleep for 0.1 seconds
                await asyncio.sleep(0.1)
                continue

    def send_status_update(self, status_update: StatusUpdate) -> requests.Response:
        logger.info(f"Sending status update: {status_update.__str__()}")
        try:
            return self.network_interface.send_status_update(status_update)
        except UnauthorizedError:
            logger.error(f"Unauthorized access: Could not authenticate with server {e}")
            raise RuntimeError("Could not authenticate with server")
        except ServerRequestError as e:
            logger.error(f"Could not send status update: {e}")
            raise RuntimeError(f"Could not send status update: {e}")
        except Exception as e:
            logger.error(f"Could not send status update: {e}")
            raise RuntimeError(f"Could not send status update: {e}")

    def send_server_request(self, request: "ServerRequest") -> requests.Response:
        logger.debug(f"Sending server request: {request.__str__()}")
        return self.network_interface.send_server_request(request)
