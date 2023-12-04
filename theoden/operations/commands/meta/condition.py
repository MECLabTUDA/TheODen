from ....resources import ResourceManager
from ....topology import Topology
from .. import Command
from ... import CommandExecutionStatus, Condition
from ....common import Transferable, ExecutionResponse, StatusUpdate


class ConditionalCommand(Command, Transferable):
    def __init__(
        self,
        command: Command,
        condition: Condition,
        *,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        """A command that executes a list of commands sequentially.

        Args:
            command (Command): The command to execute if the condition is met.
            condition (callable): The condition that needs to be met.
            uuid (Optional[str], optional): The uuid of the command. Defaults to None.
        """
        self.command = command
        self.condition = condition
        super().__init__(uuid=uuid, **kwargs)

    def set_node(self, node: "Node") -> Command:
        self.node = node
        self.command.set_node(node)
        return self

    def execute(self) -> ExecutionResponse | None:
        if self.condition.resolved(resource_manager=self.node.resources):
            self.command()
        else:
            self.node.send_status_update(
                StatusUpdate(
                    command_uuid=self.command.uuid,
                    status=CommandExecutionStatus.EXCLUDED.value,
                    datatype=type(self.command).__name__,
                )
            )
        return None

    def node_specific_modification(
        self, distribution_table: "DistributionStatusTable", node_name: str
    ) -> Command:
        self.command.node_specific_modification(
            distribution_table=distribution_table, node_name=node_name
        )
        return self

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_nodes: list[str],
    ):
        self.command.on_init_server_side(topology, resource_manager, selected_nodes)
