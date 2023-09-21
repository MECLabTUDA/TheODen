from uuid import uuid4
from typing import Optional

from .request import ServerRequest
from ...common import Transferable, UnauthorizedError
from ...topology import TopologyType


class RegisterRequest(ServerRequest, Transferable):
    def __init__(
        self,
        topology_uuid: str | None = None,
        topology_type: str = "node",
        uuid: str | None = None,
        server: Optional["Server"] = None,
        **kwargs
    ):
        super().__init__(uuid=uuid, server=server)
        self.topology_uuid = topology_uuid
        self.kwargs = kwargs
        self.topology_type = TopologyType(topology_type)

    def execute(self) -> dict:
        # set uuid if not set
        uuid = str(uuid4()) if self.topology_uuid is None else self.topology_uuid
        uuid = self.server.topology_register.register(
            topology_type=self.topology_type, uuid=uuid, **self.kwargs
        )
        if not uuid:
            raise UnauthorizedError()

        # register node in topology register and return uuid
        return {"uuid": uuid}
