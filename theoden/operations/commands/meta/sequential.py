from typing import List, Optional

from theoden.resources import ResourceManager
from theoden.topology import Topology

from ....common import ExecutionResponse, Transferable
from .. import Command


class SequentialCommand(Command, Transferable):
    def __init__(
        self, commands: list[Command], *, uuid: Optional[str] = None, **kwargs
    ) -> None:
        """A command that executes a list of commands sequentially.

        Args:
            commands (List[Command]): The commands to execute sequentially.
            client (Optional["Node"], optional): The client to execute the command on. Defaults to None.
            uuid (Optional[str], optional): The uuid of the command. Defaults to None.
        """
        self.commands = commands
        super().__init__(uuid=uuid, **kwargs)

    def set_client(self, client: "Client") -> Command:
        self.client = client
        # set client to the commands
        for c in self.commands:
            c.set_client(client)
        return self

    def execute(self) -> ExecutionResponse | None:
        for c in self.commands:
            c()
        return None

    def client_specific_modification(
        self, distribution_table: "DistributionStatusTable", client_name: str
    ) -> Command:
        for c in self.commands:
            c.client_specific_modification(
                distribution_table=distribution_table,
                client_name=client_name,
            )
        return self

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_clients: list[str],
    ):
        for c in self.commands:
            c.on_init_server_side(topology, resource_manager, selected_clients)

    def __add__(self, other: Command) -> Command:
        """
        Adds two Command objects together.

        Args:
            other (Command): The Command object to add to the current Command object.

        Returns:
            The current Command object with the other Command object added to it.
        """
        return SequentialCommand([*self.commands, other])
