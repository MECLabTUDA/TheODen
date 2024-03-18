# Actions

An Action is the equivalant of a command for the server. It is a functionality that is being executed on the server.
Like Distributions they can have a successor that is being added to the operation manager after finishing.

## Current Actions
Currently, there are two actions that can be used to control the flow of the framework.

| Action              | Description                                                                                                                                                                                                      | Link                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `InitModelAction`   | This Action initializes the global model. You can select different ways ton initialize the model my selecting an Intializer (currently `ServerInitializer`, `FileInitializer` and `SelectRandomOneInitializer`). | [link](../theoden/operations/instructions/initialization.py)        |
| `AggregationAction` | This Action aggregates the models of the clients. It uses the client checkpoint Manager and an Aggregator class. More details on the aggregation can be found [here](./AGGREGATION.md).                          | [link](../theoden/operations/instructions/aggregation/aggregate.py) |

## Create an Action
To create a new action, one must define a class inheriting from the Action class. 
The function `perform` must be implemented. This function is called when the action is executed and should be implemented by the user.

```python
    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        """Performs the action. This method is called when the action is executed 
        and should be implemented by the user.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.
        """
```

As an example please refer to the tutorial [here](./TUTORIAL.md) to create a new action.