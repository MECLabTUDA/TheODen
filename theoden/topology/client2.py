# import torch.multiprocessing as mp
import asyncio
import logging
import ssl
from multiprocessing import Manager, Process

import requests

from ..common import (
    ClientConfigurationError,
    ForbiddenOperationError,
    ServerRequestError,
    StatusUpdate,
    Transferables,
    UnauthorizedError,
)
from ..networking.interface import ClientInterface
from ..networking.rabbitmq import ClientToMQInterface
from ..networking.rest import RestClientInterface
from ..networking.storage import FileStorageInterface
from ..operations import Command, ServerRequest
from ..resources.resource import ResourceManager
from ..security.operation_protection import OperationBlackList, OperationWhiteList
from ..watcher import Watcher, WatcherPool


class Client:
    def __init__(
        self,
        network_interface: ClientInterface | None = None,
        filestorage_interface: FileStorageInterface | None = None,
        operation_protection: OperationWhiteList | OperationBlackList | None = None,
        initial_commands: list[Command] | None = None,
        watcher: list[Watcher] | None = None,
    ) -> None:
        """A federated learning client.

        Args:
            network_interface (ClientInterface): The network interface for the client.
            filestorage_interface (FileStorageInterface): The file storage interface for the client.
            operation_protection (OperationWhiteList | OperationBlackList | None, optional): A list of operations that the client is allowed to execute. Defaults to None.
            initial_commands (list[Command] | None, optional): A list of commands to be executed by the client. Defaults to None.
        """

        if network_interface is None and initial_commands is None:
            raise ClientConfigurationError(
                "Either network_interface or initial_commands must be provided."
            )

        self.operation_protection = operation_protection

        # Initialize the command queue as a list. This will hold all the commands that the client has to execute.
        self.command_queue: list[dict] = Manager().list(
            [] if initial_commands is None else initial_commands
        )

        # Initialize the resource register. This will hold all the resource_manager that are required for the commands.
        self.resources: ResourceManager = ResourceManager(
            device="cuda",
            __storage__=filestorage_interface,
            __watcher__=WatcherPool(self).add(watcher or []),
        )

        self.network_interface = network_interface
        # Add the storage interface to the network interface
        self.network_interface.add_storage_interface(self.resources.storage)

        # Initialize the processes as an empty list
        self.processes: list[Process] = []

    def _run_client(self, fn):
        # Run an async function in an event loop
        asyncio.run(fn())

    def start(self):
        """Start the client.

        This method will register the client with the server and start the command queue and serverrequests queue.
        """

        # Create two separate processes to execute command queue and serverrequests queue
        # target is the function to be called by the process
        # args is the arguments to be passed to the function

        try:
            command_process = Process(
                target=self._run_client, args=(self.start_command_queue,)
            )
            self.processes.append(command_process)

            # If no network interface is provided, only start the command queue
            if self.network_interface is not None:
                request_process = Process(
                    target=self._run_client,
                    args=(self.network_interface.start_request_loop,),
                )
                self.processes.append(request_process)

            # Start the processes
            command_process.start()

            # If no network interface is provided, only start the command queue
            if self.network_interface is not None:
                request_process.start()
                self.network_interface.start()

            command_process.join()
            request_process.join()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        # Terminate and join all processes in the processes list
        for p in self.processes:
            p.terminate()
            p.join()

        # Reset the processes list
        self.processes = []

    async def start_command_queue(self) -> None:
        while True:
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

                logging.debug(f"Executing command {command.__class__.__name__}")

                # Execute the command on the client
                command.set_client(self)()

            # If the command queue is empty, do nothing
            except IndexError:
                # sleep for 0.1 seconds
                await asyncio.sleep(0.1)
                continue

    def send_status_update(self, status_update: StatusUpdate) -> requests.Response:
        try:
            return self.network_interface.send_status_update(status_update)
        except UnauthorizedError:
            raise RuntimeError("Could not authenticate with server")
        except ServerRequestError as e:
            raise RuntimeError(f"Could not send status update: {e}")
        except Exception as e:
            raise RuntimeError(f"Could not send status update: {e}")

    def send_server_request(self, request: "ServerRequest") -> requests.Response:
        return self.network_interface.send_server_request(request)
