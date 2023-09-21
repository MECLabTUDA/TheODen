from typing import Any, List, Optional

from .. import Command
from ....common import Transferable


class SequentialCommand(Command, Transferable):
    def __init__(
        self,
        commands: List[Command],
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs
    ) -> None:
        """A command that executes a list of commands sequentially.

        Args:
            commands (List[Command]): The commands to execute sequentially.
            node (Optional["Node"], optional): The node to execute the command on. Defaults to None.
            uuid (Optional[str], optional): The uuid of the command. Defaults to None.
        """
        self.commands = commands
        super().__init__(node=node, uuid=uuid, **kwargs)

    def set_node(self, node: "Node"):
        self.node = node
        # set node to the commands
        for c in self.commands:
            c.set_node(node)

    def execute(self) -> Any:
        for c in self.commands:
            c.execute()

    def node_specific_modification(
        self, status_register: dict[str, "StatusTable"], node_uuid: str
    ) -> Command:
        for c in self.commands:
            c.node_specific_modification(
                {
                    cmd.uuid: status_register[cmd.uuid]
                    for cmd in list(c.get_command_tree().values())
                },
                node_uuid=node_uuid,
            )
        return self
