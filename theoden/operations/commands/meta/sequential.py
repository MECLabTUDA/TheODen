from typing import Any, List, Optional

from theoden.resources import ResourceManager
from theoden.topology import Topology

from .. import Command
from ....common import Transferable, ExecutionResponse


class SequentialCommand(Command, Transferable):
    def __init__(
        self, commands: List[Command], *, uuid: Optional[str] = None, **kwargs
    ) -> None:
        """A command that executes a list of commands sequentially.

        Args:
            commands (List[Command]): The commands to execute sequentially.
            node (Optional["Node"], optional): The node to execute the command on. Defaults to None.
            uuid (Optional[str], optional): The uuid of the command. Defaults to None.
        """
        self.commands = commands
        super().__init__(uuid=uuid, **kwargs)

    def set_node(self, node: "Node") -> Command:
        self.node = node
        # set node to the commands
        for c in self.commands:
            c.set_node(node)
        return self

    def execute(self) -> ExecutionResponse | None:
        for c in self.commands:
            c()
        return None

    def node_specific_modification(
        self, distribution_table: "DistributionStatusTable", node_name: str
    ) -> Command:
        for c in self.commands:
            c.node_specific_modification(
                distribution_table=distribution_table,
                node_name=node_name,
            )
        return self

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_nodes: list[str],
    ):
        for c in self.commands:
            c.on_init_server_side(topology, resource_manager, selected_nodes)

    def __add__(self, other: Command) -> Command:
        """
        Adds two Command objects together.

        Args:
            other (Command): The Command object to add to the current Command object.

        Returns:
            The current Command object with the other Command object added to it.
        """
        return SequentialCommand([*self.commands, other])
