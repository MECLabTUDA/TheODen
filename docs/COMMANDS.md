# Commands
A client is basically a programm that executes the commands that it receives from the server. Therefore it is the most important Components for a client.

Below is a list of all commands that are implemented in the framework. The commands are grouped into three categories: **Action Commands**, **Helper Commands** and **Meta Commands**.

The name of each command is usually self-explanatory, as one command performs one specific action on the client. The Meta commands can be used to build commands that include multiple or different commands.

| Command Name                           | Function                                                                                                                                                           | Link                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| `TrainRoundCommand`                    | Train the model for one epoch                                                                                                                                      | [link](../theoden/operations/commands/action/train.py)              |
| `ValidateEpochCommand`                 | Validate the model for one epoch                                                                                                                                   | [link](../theoden/operations/commands/action/validate.py)           |
| `CalculateClientScoreCommand`          | Calculate the client score                                                                                                                                         | [link](../theoden/operations/commands/action/client_score.py)       |
| `SendModelToServerCommand`             | Send the model to the server                                                                                                                                       | [link](../theoden/operations/commands/action/send_model.py)         |
| `SendOptimizerToServerCommand`         | Send the optimizer to the server                                                                                                                                   | [link](../theoden/operations/commands/action/send_optimizer.py)     |
| `SendCheckpointToServerCommand`        | Send a resource to the server                                                                                                                                      | [link](../theoden/operations/commands/action/send_resource.py)      |
| `InspectLabelDistributionCommand`      | Inspect the label distribution                                                                                                                                     | [link](../theoden/operations/commands/inspect/distribution.py)      |
| `LoadDatasetCommand`                   | Load a dataset into the resource register of a client                                                                                                              | [link](../theoden/operations/commands/resource/load_dataset.py)     |
| `InitModelCommand`                     | Random initialization of a model on the client                                                                                                                     | [link](../theoden/operations/commands/resource/model.py)            |
| `SetOptimizerCommand`                  | Set the optimizer on the client                                                                                                                                    | [link](../theoden/operations/commands/resource/optimizer.py)        |
| `PlotSamplesCommand`                   | Plot samples from a dataset                                                                                                                                        | [link](../theoden/operations/commands/resource/plot_samples.py)     |
| `SetAugmentationCommand`               | Set the augmentation on the client (same for all clients). The dataset will be wrapped by an AugmentationDataset the performs the augmentation during dataloading. | [link](../theoden/operations/commands/resource/set_augmentation.py) |
| `SetClientSpecificAugmentationCommand` | Set the augmentation on the client (different for each client)                                                                                                     | [link](../theoden/operations/commands/resource/set_augmentation.py) |
| `SetDataLoaderCommand`                 | Set the dataloader on the client                                                                                                                                   | [link](../theoden/operations/commands/resource/set_dataloader.py)   |
| `SetDataSamplerCommand`                | Set the datasampler on the client                                                                                                                                  | [link](../theoden/operations/commands/resource/set_datasampler.py)  |
| `SetLossesCommand`                     | Set the losses on the client                                                                                                                                       | [link](../theoden/operations/commands/resource/set_loss.py)         |
| `SetPartitionCommand`                  | Set the partitions for the different clients. This will distribute the dataset between clients (e.g. all clients get specific patients).                           | [link](../theoden/operations/commands/resource/set_partition.py)    |
| `SetLocalPartitionCommand`             | Set the local partition for the client. This will split the local dataset of a client into different datasets (e.g. train/val/test).                               | [link](../theoden/operations/commands/resource/set_partition.py)    |
| `SetResourceCommand`                   | Set a resource on the client. Usually the set commands inherit from this class.                                                                                    | [link](../theoden/operations/commands/resource/set_resource.py)     |
| `SetLRSchedulerCommand`                | Set the learning rate scheduler on the client                                                                                                                      | [link](../theoden/operations/commands/resource/set_scheduler.py)    |
| `StorageCommand`                       | Request a resource from the storage server                                                                                                                         | [link](../theoden/operations/commands/resource/storage_command.py)  |
| `LoadStateDictCommand`                 | Load a state dict from the storage server into a model. This uses StorageCommand.                                                                                  | [link](../theoden/operations/commands/resource/storage_command.py)  |
| `LoadOptimizerStateDictCommand`        | Load a state dict from the storage server into an optimizer. This uses StorageCommand.                                                                             | [link](../theoden/operations/commands/resource/storage_command.py)  |
| `WrapDatasetCommand`                   | Wrap a dataset with a wrapper. This is used to wrap a dataset with an augmentation.                                                                                | [link](../theoden/operations/commands/resource/wrap_dataset.py)     |



## Helper Commands
| Command Name               | Function                                                                                                                                   |                               Link                               |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------| :--------------------------------------------------------------: |
| `RepeatNTimesCommand`      | Repeat a command N times                                                                                                                   |      [link](../theoden/operations/commands/collections/repeat.py)       |
| `TrainValNTimesCommand`    | Train and validate N times. This command will create a SequentialCommand that gets the multiple train and val rounds as a list of commands. | [link](../theoden/operations/commands/collections/train_val.py) |
| `PrintResourceKeysCommand` | Print the keys and datatypes of the client resource manager. Can be used for debugging                                                     |    [link](../theoden/operations/commands/resource/helper.py)     |
| `ClearResourcesCommand`    | Clear all resources of the client                                                                                                          |    [link](../theoden/operations/commands/resource/helper.py)     |
| `ExitRunCommand`              | Exits the clients and afterward the server                                                                                                 | [link](../theoden/operations/commands/resource/helper.py)         |

## Meta Commands
| Command Name         | Function                                                                                                                                                       | Link                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `SequentialCommand`  | Gets a list of commands as parameter and executes these sequentially. This is the main command if you want to distribute more than one command to the clients. | [link](../theoden/operations/commands/meta/sequential.py) |
| `ConditionalCommand` | Run commands on the client if a **condition on the client** is fulfilled                                                                                       | [link](../theoden/operations/commands/meta/condition.py)  |

# Creating a new Command

## Command Structure

```python	
class Command(Transferable, is_base_type=True):
    """
    The Command interface declares a method for executing a command.
    """

    def __init__(
        self,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        self.uuid = uuid
        self.client: "Client" | None = None

    def init_uuid(self) -> Command:
        """
        Initializes a UUID for the current Command object and recursively sets UUIDs for all its subcommands.

        Returns:
            The current Command object with the UUID and any updated initialization parameters.
        """
        # This method only needs modification if the command has subcommands.
        

    def set_client(self, client: "Client") -> Command:
        """
        Sets the client attribute of the Command object.

        Args:
            client (Node): The client to set as the attribute.

        Returns:
            The current Command object with the client attribute set.
        """
        # This method only needs modification if the command has subcommands.
        

    @property
    def client_rm(self) -> ResourceManager:
        """
        Returns the ResourceManager of the client attribute of the Command object.

        Returns:
            The ResourceManager of the client attribute of the Command object.
        """
        # Just a helper method to get the resource manager of the client.
        

    def get_command_tree(self, flatten: bool = True) -> dict[str, Command]:
        """
        Returns a nested dictionary representing the Command object and all its subcommands.

        Args:
            flatten (bool): Whether to flatten the dictionary or keep subcommands nested. Default is True.

        Returns:
            A dictionary representing the Command object and all its subcommands.
        """
        # This method only needs modification if the command has subcommands that are not stored in lists, dicts or tuples.
        # This should be avoided if possible.

    def on_init_server_side(self, topology: Topology, resource_manager: ResourceManager, selected_clients: list[str]) -> None:
        """This method is called on the server when the command is initialized.
        It is used to modify resources or the command at the time of initialization.This can be used to add information
        to the command that is only available at runtime.

        Args:
            topology (Topology): The topology register of the instruction.
            resource_manager (ResourceManager): The resource register of the instruction.
            selected_clients (list[str]): The list of clients that are selected for the instruction.
        """
        # If you have to perform an action on the server at the time of initialization, you can do it here.
        # This can be necessary if you want to add information to the command that is only available at runtime.
        # As an example the StorageCommand will upload the resource (e.g. the updated global model)
        # to the storage server at the time of initialization.

    def client_specific_modification(
        self, distribution_table: "DistributionStatusTable", client_name: str
    ) -> Command:
        """
        This method is called on the server to modify the command to be executed on the clients. It is used to add client specific information to the command after initialization.
        This is necessary because the command is initialized on the server but might need information that is only available during runtime.

        Args:
            status_register (dict): The dictionary of all the commands in the instruction and their status.
            client_name (str): The uuid of the client the command is executed on.

        Returns:
            The modified command.
        """
        # If you have to modify the command with information that is only available at runtime, you can do it here.
        # As an example the SetPartitionCommand is modified to only contain the partition of the client it is executed on.

    @abstractmethod
    def execute(self) -> ExecutionResponse | None:
        """Abstract execute command that is called to perform the actions specific to the command

        Returns:
            ExecutionResponse | None: The response of the execution.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        # This is most important method of the command. It is called to execute the command on the client.
        # In this method you should perform all actions that are specific to the command.

    def __call__(self, *args, **kwargs) -> ExecutionResponse | None:
        """This method is called when the command is executed. It is used to send status updates to the server before and after the execution of the command.

        Returns:
            ExecutionResponse | None: The response of the execution.
        """
        # Don't modify this method. It is used to send status updates to the server before and after the execution of the command.

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        client_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ) -> None:
        """
        This method is called after the execution of the command is finished.

        Args:
            topology (Topology): The topology register of the instruction.
            resource_manager (ResourceManager): The resource register of the instruction.
            client_name (str): The uuid of the client the command is executed on.
            execution_response (ExecutionResponse): The response of the execution.
            instruction_uuid (str): The uuid of the instruction.
        """
        # Modify this method if the server should perform an action after the execution of the command on one of the clients.
        # As an example the send model will be saved in a checkpoint manager after it is sent to the server.

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        """
        This method is called on the server to check if all clients have finished executing the command.

        Args:
            topology (Topology): The topology register of the client.
            resource_manager (ResourceManager): The resource register of the client.
            instruction_uuid (str): The uuid of the instruction.
        """
        # Modify this method if the server should perform an action after all clients have finished executing the command.
        # As an example the StorageCommand removes the resource from the storage server after all clients downloaded it.
```

As an example please refer to the tutorial [here](./TUTORIAL.md) to create a new command.

# Overwriting and Abstract Commands

Abstract commands are commands that are not implemented by the framework. They act as a placeholder for a command that will be implemented by each client individually. This allows each client to have full control over all processes that are executed on the client side, as every client can implement their own version of the command.

```python
from theoden import Transferable 

class ABCClassImplementation(Transferable, implements=ABCClass):
    # implementation
```

As an example please refer to the tutorial [here](./TUTORIAL.md) to overwrite an abstract command.