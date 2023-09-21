from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..common import StatusUpdate

if TYPE_CHECKING:
    from ..operations import ServerRequest


class NodeInterface(ABC):
    @abstractmethod
    def send_status_update(self, status_update: StatusUpdate) -> any:
        pass

    @abstractmethod
    def send_server_request(self, request: "ServerRequest") -> any:
        pass
