from ...common import ExecutionResponse, Transferable
from .request import ServerRequest


class LoginRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_online(self.node_name)
        return None


class LogoutRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_offline(self.node_name)
        return None
