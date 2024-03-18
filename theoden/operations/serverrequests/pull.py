from ...common import ExecutionResponse, Transferable
from .request import ServerRequest


class PullCommandRequest(ServerRequest, Transferable):
    """A request to pull a command from the server.

    This request will pull a command from the server and return it as a dictionary.
    It is the main request used by the clients communicating with the server and the tool to distribute commands to the clients.
    """

    def __init__(self, uuid: None | str = None, **kwargs):
        """A request to pull a command from the server.

        Args:
            uuid (None | str, optional): The uuid of the request. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)

    def execute(self) -> ExecutionResponse | None:
        """Executes the request and returns, if available, the command to be executed by the client.

        Returns:
            dict: The command to be executed by the client.
        """

        cmd = self.server.operation_manager.get_command(
            client_name=self.client_name,
            topology=self.server.topology,
            resource_manager=self.server.resources,
        )

        return (
            ExecutionResponse(data=cmd.dict())
            if cmd is not None
            else ExecutionResponse(data={})
        )
