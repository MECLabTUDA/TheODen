# import torch.multiprocessing as mp
from multiprocessing import Process, Manager
from typing import TYPE_CHECKING
import requests
import asyncio

from ..common import (
    Transferables,
    StatusUpdate,
    ServerRequestError,
    UnauthorizedError,
    ForbiddenCommandError,
)
from ..operations import PullCommandRequest, Command
from ..resources.resource import ResourceRegister
from ..networking.rest import RestNodeInterface
from ..security import CommandWhiteList, CommandBlackList

if TYPE_CHECKING:
    from ..operations import ServerRequest


class Node:
    # Define a dictionary to hold waiting times for pop operation
    # Each key represents a waiting level, and each value represents the corresponding time to wait
    _WAITING_TIME: dict[int, float] = {0: 0.001, 1: 0.01, 2: 0.05, 3: 0.1, 4: 0.250}

    def __init__(
        self,
        address: str = "localhost",
        port: int = 8000,
        username: str = "dummy",
        password: str = "dummy",
        ping_interval: float = 1.0,
        command_protection: CommandWhiteList | CommandBlackList | None = None,
    ) -> None:
        """A federated learning node.

        Args:
            address (str, optional): The address of the server. Defaults to "localhost".
            port (int, optional): The port of the server. Defaults to 8000.
            username (str, optional): The username to use for authentication. Defaults to "dummy".
            password (str, optional): The password to use for authentication. Defaults to "dummy".
            ping_interval (float, optional): The interval at which to ping the server for new commands. Defaults to 1.0.
            command_protection (CommandWhiteList | CommandBlackList | None, optional): A whitelist or blacklist of commands to allow or disallow. Defaults to None.
        """

        # Initialize the command queue and resource register as empty dictionaries
        self.uuid: str | None = None
        self.command_queue: list[dict] = Manager().list([])
        self.resource_register: ResourceRegister = ResourceRegister()
        self.resource_register["device"] = "cuda"
        self.shutdown: bool = False
        self.ping_interval = ping_interval
        self.command_protection = command_protection

        # try:

        self.network_interface = RestNodeInterface(
            address=address, port=port, username=username, password=password
        )

        # except ServerRequestError as e:
        #     raise RuntimeError(f"Could not connect to server: {e}")

        # Initialize the processes as an empty list
        self.processes: list[Process] = []

    def _run_node(self, fn):
        # Run an async function in an event loop
        asyncio.run(fn())

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

    def start(self):
        """Start the node.

        This method will register the node with the server and start the command queue and serverrequests queue.
        """

        # response = self.send_server_request(RegisterRequest())
        # self.uuid = response.json()["uuid"]

        # if self.uuid is None:
        #     raise RuntimeError("Could not register with server")

        # Create two separate processes to execute command queue and serverrequests queue
        # target is the function to be called by the process
        # args is the arguments to be passed to the function

        try:
            p1 = Process(target=self._run_node, args=(self.start_command_queue,))
            p2 = Process(target=self._run_node, args=(self.start_request_loop,))

            # Append the processes to the processes list
            self.processes.extend([p1, p2])

            # Start the processes
            p1.start()
            p2.start()

            p1.join()
            p2.join()

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
        # Initialize the waiting time as 0
        waiting_level: int = 0

        # Continue to loop until shutdown is True
        while not self.shutdown:
            try:
                # Pop the first command from the command queue
                command_json = self.command_queue.pop(0)

                command: Command = Transferables().to_object(
                    command_json, Command, node=self
                )

                # Check if the command and all subcommands are allowed
                if self.command_protection is not None:
                    # get all commands inside the command tree
                    command_tree = command.get_command_tree()

                    # check if all commands are allowed
                    for command in command_tree:
                        if not self.command_protection.allows(command):
                            raise ForbiddenCommandError(
                                f"Command {command.__name__} is not allowed"
                            )

                # Execute the command
                command.execute()

                # Reset the waiting level
                waiting_level = 0

            # If the command queue is empty, wait for a certain amount of time before trying again
            except IndexError:
                await asyncio.sleep(self._WAITING_TIME[waiting_level])
                waiting_level = min(4, waiting_level + 1)

    async def start_request_loop(self) -> None:
        # Continue to loop until shutdown is True
        while not self.shutdown:
            # Call the _pull method to get a serverrequests from the server
            self._pull()

            # Wait for 1 second before trying again
            await asyncio.sleep(self.ping_interval)

    def _process_command(self, command: dict) -> bool:
        # TODO: Implement command processing logic
        # Return True if the command was successfully processed, False otherwise
        if not command:
            return False
        return True

    def _pull(self):
        # Make a GET serverrequests to the server to get the next command
        try:
            response = self.send_server_request(PullCommandRequest(self.uuid))

            # If the response contains a command, parse it into a CommandModel object and add it to the command queue
            if response.status_code == 200:
                if self._process_command(response.json()):
                    # print(len(self.command_queue))
                    self.command_queue.append(response.json())
        except ServerRequestError as e:
            return
        except UnauthorizedError as e:
            print("Unauthorized")
            # self.network_interface.request_token()
