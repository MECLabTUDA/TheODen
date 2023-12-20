from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4

from ...common import ExecutionResponse, Transferable
from ...resources import ResourceManager
from ...topology import Topology

if TYPE_CHECKING:
    from theoden.topology.server import Server


class ServerRequest(ABC, Transferable, is_base_type=True):
    def __init__(
        self,
        uuid: None | str = None,
        node_name: str | None = None,
        **kwargs,
    ):
        """A request to the server.

        Args:
            uuid (None | str, optional): The uuid of the request. Defaults to None.
            server (Optional["Server"], optional): The server to pull the command from. Defaults to None. It will be set by the server when the request is processed.
            node_name (str | None, optional): The uuid of the node that sent the request. Defaults to None. It will be set by the server when the request is processed.
        """
        self.uuid = uuid
        if self.uuid is None:
            self.uuid = str(uuid4())
            self.add_initialization_parameter(uuid=self.uuid)
        self.server: "Server" | None = None
        self.node_name = node_name

    def set_server(self, server: "Server") -> ServerRequest:
        """Sets the server.

        Args:
            server ("Server"): The server to set.

        Returns:
            ServerRequest: The request.
        """
        self.server = server
        return self

    @property
    def server_rr(self) -> ResourceManager:
        """The server's resource register.

        Returns:
            ResourceManager: The server's resource register.
        """
        return self.server.resources

    @property
    def server_tr(self) -> Topology:
        """The server's topology register.

        Returns:
            Topology: The server's topology register.
        """
        return self.server.topology

    @abstractmethod
    def execute(self) -> ExecutionResponse | None:
        raise NotImplementedError("Please implement this method")
