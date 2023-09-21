from uuid import uuid4
from typing import Optional

from .request import ServerRequest
from ...common import Transferable, UnauthorizedError
from ...topology import TopologyType, NodeStatus


class LoginRequest(ServerRequest, Transferable):
    def __init__(
        self,
        uuid: str | None = None,
        server: Optional["Server"] = None,
        node_uuid: str | None = None,
        **kwargs
    ):
        super().__init__(uuid=uuid, server=server, node_uuid=node_uuid, **kwargs)

    def execute(self) -> None:
        self.server.topology_register.nodes[
            self.node_uuid
        ].status = NodeStatus.CONNECTED


class LogoutRequest(LoginRequest, Transferable):
    def execute(self) -> None:
        self.server.topology_register.nodes[
            self.node_uuid
        ].status = NodeStatus.DISCONNECTED
