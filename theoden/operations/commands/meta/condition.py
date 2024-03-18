from ....common import ExecutionResponse, StatusUpdate, Transferable
from ....resources import ResourceManager
from ....topology import Topology
from ... import CommandExecutionStatus, Condition
from .. import Command


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

    def set_client(self, client: "Client") -> Command:
        self.client = client
        self.command.set_client(client)
        return self

    def execute(self) -> ExecutionResponse | None:
        if self.condition.resolved(resource_manager=self.client.resources):
            self.command()
        else:
            self.client.send_status_update(
                StatusUpdate(
                    command_uuid=self.command.uuid,
                    status=CommandExecutionStatus.EXCLUDED.value,
                    datatype=type(self.command).__name__,
                )
            )
        return None

    def client_specific_modification(
        self, distribution_table: "DistributionStatusTable", client_name: str
    ) -> Command:
        self.command.client_specific_modification(
            distribution_table=distribution_table, client_name=client_name
        )
        return self

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_clients: list[str],
    ):
        self.command.on_init_server_side(topology, resource_manager, selected_clients)
