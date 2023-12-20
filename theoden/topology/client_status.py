import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import Topology


class ClientStatusObserver(ABC):
    @abstractmethod
    def observe(self, topology: "Topology"):
        raise NotImplementedError("ClientStatusObserver.observe() is not implemented")


class TimeoutClientStatusObserver(ClientStatusObserver):
    def __init__(self, timeout: float = 3.0, sleep: float = 1.0) -> None:
        self.timeout = timeout
        self.sleep = sleep

    def observe(self, topology: "Topology"):
        from .topology import NodeStatus

        while True:
            for node in topology.clients:
                if node.status == NodeStatus.ONLINE:
                    if node.last_active + self.timeout < time.time():
                        topology.set_offline(node.name)
                        logging.warning(f"Client {node.name} timed out")

            time.sleep(self.sleep)
