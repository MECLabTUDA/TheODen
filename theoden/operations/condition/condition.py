from abc import ABC, abstractmethod

from ...common import Transferable
from ...resources import ResourceManager
from ...topology import Topology


class Condition(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        """Returns True if the condition is resolved.

        Args:
            topology (Topology): The topology.
            resource_manager (ResourceManager): The resource register.

        Returns:
            bool: True if the condition is resolved.
        """
        raise NotImplementedError("Please implement this method")
