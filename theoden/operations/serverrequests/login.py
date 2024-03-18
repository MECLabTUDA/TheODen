from ...common import ExecutionResponse, Transferable
from .request import ServerRequest


class LoginRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_online(self.client_name)
        return None


class LogoutRequest(ServerRequest, Transferable):
    def execute(self) -> ExecutionResponse | None:
        self.server.topology.set_offline(self.client_name)
        return None
