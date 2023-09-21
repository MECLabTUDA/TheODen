from typing import TYPE_CHECKING, Optional

from uuid import uuid4
from abc import ABC, abstractmethod

from ...common import Transferable
from ...topology import TopologyRegister
from ...resources import ResourceRegister

if TYPE_CHECKING:
    from theoden.topology.server import Server


class ServerRequest(ABC, Transferable, is_base_type=True):
    def __init__(
        self,
        uuid: None | str = None,
        server: Optional["Server"] = None,
        node_uuid: str | None = None,
        **kwargs
    ):
        """A request to the server.

        Args:
            uuid (None | str, optional): The uuid of the request. Defaults to None.
            server (Optional["Server"], optional): The server to pull the command from. Defaults to None. It will be set by the server when the request is processed.
            node_uuid (str | None, optional): The uuid of the node that sent the request. Defaults to None. It will be set by the server when the request is processed.
        """
        self.uuid = uuid
        if self.uuid is None:
            self.uuid = str(uuid4())
        self.server = server
        self.node_uuid = node_uuid

    @property
    def server_rr(self) -> ResourceRegister:
        """The server's resource register.

        Returns:
            ResourceRegister: The server's resource register.
        """
        return self.server.resource_register

    @property
    def server_tr(self) -> TopologyRegister:
        """The server's topology register.

        Returns:
            TopologyRegister: The server's topology register.
        """
        return self.server.topology_register

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Please implement this method")
