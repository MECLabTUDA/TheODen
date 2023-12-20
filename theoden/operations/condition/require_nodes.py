from ...common import Transferable
from ...resources import ResourceManager
from ...topology import Topology
from .condition import Condition


class RequireNumberOfNodesCondition(Condition, Transferable):
    def __init__(self, number_of_nodes: int) -> None:
        """A condition that requires a certain number of nodes to be connected to the server.

        Args:
            number_of_nodes (int): The number of nodes required.

        Raises:
            AssertionError: If the number of nodes is less than or equal to 0.
        """
        super().__init__()
        assert number_of_nodes > 0
        self.number_of_nodes = number_of_nodes

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        """Returns True if the number of nodes in the topology register
        is equal or greater than the number of nodes required.

        Args:
            topology (Topology): The topology.
            resource_manager (ResourceManager): The resource register.

        Returns:
            bool: True if the number of nodes in the topology register
            is equal or greater than the number of nodes required.
        """
        return topology.num_connected_clients >= self.number_of_nodes

    def __add__(self, other) -> "InstructionBundle":
        from ..instructions import InstructionBundle

        return InstructionBundle(
            [RequireNumberOfNodesCondition(self.number_of_nodes), other]
        )


class RequirePercentageOfNodesToBeConnectedCondition(Condition, Transferable):
    def __init__(self, percentage: float) -> None:
        """A condition that requires a certain percentage of nodes to be connected to the server.

        Args:
            percentage (float): The percentage of nodes required.

        Raises:
            AssertionError: If the percentage is less than or equal to 0 or greater than 1.
        """
        super().__init__()
        assert 0 < percentage <= 1
        self.percentage = percentage

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        return topology.fraction_connected_clients >= self.percentage


class RequireAllNodesToBeConnectedCondition(
    RequirePercentageOfNodesToBeConnectedCondition, Transferable
):
    def __init__(self) -> None:
        """A condition that requires all nodes to be connected to the server.

        This is a special case of the RequirePercentageOfNodesToBeConnectedCondition with a percentage of 1.0.
        """
        super().__init__(percentage=1.0)


class RequireNumberOfNodesOrRequireAllIfSchemaHasAuthModeCondition(
    Condition, Transferable
):
    def __init__(self, number_of_nodes: int = 2) -> None:
        """A condition that requires a certain number of nodes to be connected to the server,
         or all nodes if the topology schema has an auth mode.

        This is a special case of the RequireNumberOfNodesCondition and the RequireAllNodesToBeConnectedCondition.

        Args:
            number_of_nodes (int, optional): The number of nodes required. Defaults to 2.

        Raises:
            AssertionError: If the number of nodes is less than or equal to 0.
        """
        super().__init__()
        self.number_of_nodes = number_of_nodes
        self.require_all_nodes_to_be_connected = RequireAllNodesToBeConnectedCondition()
        self.require_number_of_nodes_to_be_connected = RequireNumberOfNodesCondition(
            self.number_of_nodes
        )

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        return self.require_number_of_nodes_to_be_connected.resolved(
            topology=topology, resource_manager=resource_manager
        )
