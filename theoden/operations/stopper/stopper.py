from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from theoden.topology.server import Server
from ...common import Transferable


class Stopper(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def resolved(self, server: Server) -> bool:
        """Returns True if the stopper is resolved.

        Args:
            server (Server): The server.

        Returns:
            bool: True if the stopper is resolved.
        """
        raise NotImplementedError("Please implement this method")
