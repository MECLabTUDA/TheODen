from .stopper import Stopper
from ...common import Transferable
from ...topology.server import Server


class RequireNumberOfNodesStopper(Stopper, Transferable):
    def __init__(self, number_of_nodes: int) -> None:
        """A stopper that requires a certain number of nodes to be connected to the server.

        Args:
            number_of_nodes (int): The number of nodes required.

        Raises:
            AssertionError: If the number of nodes is less than or equal to 0.
        """
        super().__init__()
        assert number_of_nodes > 0
        self.number_of_nodes = number_of_nodes

    def resolved(self, server: Server) -> bool:
        """Returns True if the number of nodes in the topology register is equal or greater than the number of nodes required.

        Args:
            server (Server): The server.

        Returns:
            bool: True if the number of nodes in the topology register is equal or greater than the number of nodes required.
        """
        return (
            len(server.topology_register.get_connected_nodes()) >= self.number_of_nodes
        )


class RequirePercentageOfNodesToBeConnectedStopper(Stopper, Transferable):
    def __init__(self, percentage: float) -> None:
        """A stopper that requires a certain percentage of nodes to be connected to the server.

        Args:
            percentage (float): The percentage of nodes required.

        Raises:
            AssertionError: If the percentage is less than or equal to 0 or greater than 1.
        """
        super().__init__()
        assert percentage > 0 and percentage <= 1
        self.percentage = percentage

    def resolved(self, server: Server) -> bool:
        return (
            len(server.topology_register.get_connected_nodes())
            / len(server.topology_register.get_all_nodes())
            >= self.percentage
        )


class RequireAllNodesToBeConnectedStopper(
    RequirePercentageOfNodesToBeConnectedStopper, Transferable
):
    def __init__(self) -> None:
        """A stopper that requires all nodes to be connected to the server.

        This is a special case of the RequirePercentageOfNodesToBeConnectedStopper with a percentage of 1.0.
        """
        super().__init__(percentage=1.0)


class RequireNumberOfNodesOrRequireAllIfSchemaHasAuthMode(Stopper, Transferable):
    def __init__(self, number_of_nodes: int = 2) -> None:
        """A stopper that requires a certain number of nodes to be connected to the server, or all nodes if the topology schema has an auth mode.

        This is a special case of the RequireNumberOfNodesStopper and the RequireAllNodesToBeConnectedStopper.

        Args:
            number_of_nodes (int, optional): The number of nodes required. Defaults to 2.

        Raises:
            AssertionError: If the number of nodes is less than or equal to 0.
        """
        super().__init__()
        self.number_of_nodes = number_of_nodes
        self.require_all_nodes_to_be_connected = RequireAllNodesToBeConnectedStopper()
        self.require_number_of_nodes_to_be_connected = RequireNumberOfNodesStopper(
            self.number_of_nodes
        )

    def resolved(self, server: Server) -> bool:
        if server.topology_register.schema.auth_mode:
            return self.require_all_nodes_to_be_connected.resolved(server)
        else:
            return self.require_number_of_nodes_to_be_connected.resolved(server)
