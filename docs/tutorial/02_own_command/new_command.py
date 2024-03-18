from theoden import ExecutionResponse
from theoden.operations import Command
from theoden.resources import TorchModel, ResourceManager
from theoden.topology import Topology


class NumberOfModelParametersCommand(Command):
    """Calculate the number of parameters in the model and print it."""

    def execute(self) -> ExecutionResponse | None:
        """Calculate the number of parameters in the model and print it.

        Returns:
            ExecutionResponse: response with the number of parameters
        """

        model = self.client_rm.gr("model", assert_type=TorchModel)

        num_params = sum(p.numel() for p in model.model.parameters())

        print(f"Number of model parameters: {num_params}")

        return ExecutionResponse(data={"num_params": num_params})


class CalculateSumOfModelParametersCommand(NumberOfModelParametersCommand):
    """
    Calculate the sum of model parameters from all clients. If the command is distributed to the clients,
    set the sum of parameters to 0. Whenever a client finishes, add the number of parameters to the sum. When all clients
    finish, print the sum of parameters.

    The execute method is inherited from NumberOfModelParametersCommand.
    """

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_clients: list[str],
    ) -> None:
        resource_manager.sr("sum_of_params", 0)

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        client_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ) -> None:
        sum_of_params = resource_manager.gr("sum_of_params")
        sum_of_params += execution_response.data["num_params"]
        resource_manager.sr("sum_of_params", sum_of_params)

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        sum_of_params = resource_manager.gr("sum_of_params")
        print(f"Sum of model parameters: {sum_of_params}")
