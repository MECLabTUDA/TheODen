from .request import ServerRequest
from ...common import Transferable, ExecutionResponse


class LoginRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_online(self.node_name)
        return None


class LogoutRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_offline(self.node_name)
        return None
