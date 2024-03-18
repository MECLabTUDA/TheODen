from ...resources import ResourceManager
from ...topology import Topology
from .condition import Condition


class RequireNumberOfClientsCondition(Condition):
    def __init__(self, number_of_clients: int) -> None:
        """A condition that requires a certain number of clients to be connected to the server.

        Args:
            number_of_clients (int): The number of clients required.

        Raises:
            AssertionError: If the number of clients is less than or equal to 0.
        """
        super().__init__()
        assert number_of_clients > 0
        self.number_of_clients = number_of_clients

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        """Returns True if the number of clients in the topology register
        is equal or greater than the number of clients required.

        Args:
            topology (Topology): The topology.
            resource_manager (ResourceManager): The resource register.

        Returns:
            bool: True if the number of clients in the topology register
            is equal or greater than the number of clients required.
        """
        return topology.num_connected_clients >= self.number_of_clients


class RequireClientsCondition(Condition):
    def __init__(self, nodes: list[str]) -> None:
        """A condition that requires a certain number of clients to be connected to the server.

        Args:
            number_of_clients (int): The number of clients required.

        Raises:
            AssertionError: If the number of clients is less than or equal to 0.
        """
        super().__init__()
        self.nodes = nodes

    def resolved(
        self, resource_manager: ResourceManager, topology: Topology | None = None
    ) -> bool:
        """Returns True the nodes are connected to the server.

        Args:
            topology (Topology): The topology.
            resource_manager (ResourceManager): The resource register.

        Returns:
            bool: True if the nodes are connected to the server.
        """
        return all([node in topology.online_clients(True) for node in self.nodes])


class RequirePercentageOfClientsCondition(Condition):
    def __init__(self, percentage: float) -> None:
        """A condition that requires a certain percentage of clients to be connected to the server.

        Args:
            percentage (float): The percentage of clients required.

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


class RequireAllClientsCondition(RequirePercentageOfClientsCondition):
    def __init__(self) -> None:
        """A condition that requires all clients to be connected to the server.

        This is a special case of the RequirePercentageOfNodesToBeConnectedCondition with a percentage of 1.0.
        """
        super().__init__(percentage=1.0)
