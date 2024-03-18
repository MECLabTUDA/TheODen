# Conditions

Conditions are essential for pausing command execution until specific criteria are met. For instance, they are used to wait for a certain number of clients to proceed.

They can also be used in ConditionalCommands to execute a command only if a certain condition is met.

## Current Conditions

| Condition Name                      | Description                                                                                            | Link                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| `RequireNumberOfClientsCondition`     | This condition waits for a certain number of clients to proceed                                        | [link](../theoden/operations/condition/require_nodes.py)      |
| `RequireClientsCondition`             | This condition waits for specific clients to proceed                                                   | [link](../theoden/operations/condition/require_nodes.py)      |
| `RequirePercentageOfClientsCondition` | This condition waits for a certain percentage of clients to proceed                                    | [link](../theoden/operations/condition/require_nodes.py)      |
| `RequireAllClientsCondition`          | This condition waits for all clients to proceed                                                        | [link](../theoden/operations/condition/require_nodes.py)      |
| `HasResourceCondition`              | This condition is met if a certain resource is set in a resource_manager. Used for CondtionalCommands. | [link](../theoden/operations/condition/resource_condition.py) |


# Create a condition
A Condition must implement the abstract function `resolved` which returns a boolean value. The condition is considered resolved if the function returns True.

```python
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
```

As an example please refer to the tutorial [here](./TUTORIAL.md) to create a new condition.
