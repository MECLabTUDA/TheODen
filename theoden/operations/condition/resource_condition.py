from typing import Any

from .condition import Condition
from ...common import Transferable
from ...topology import Topology
from ...resources import ResourceManager


class HasResourceCondition(Condition, Transferable):
    def __init__(
        self, resource_name: str, of_type: type[Transferable] | None = None
    ) -> None:
        """A condition that requires a certain resource to be present in the resource register.

        Args:
            resource_name (str): The name of the resource.
        """
        super().__init__()
        self.resource_name = resource_name
        self.of_type = of_type

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        """Returns True if the resource is present in the resource register.

        Args:
            topology (Topology): The topology.
            resource_manager (ResourceManager): The resource register.

        Returns:
            bool: True if the resource is present in the resource register.
        """
        try:
            resource_manager.gr(self.resource_name, self.of_type or Any)
        except KeyError:
            return False
        else:
            return True
