from __future__ import annotations


import torch

from ....common import ClientScoreResponse, ExecutionResponse, Transferable
from ....resources import ResourceManager, SampleDataset
from ....topology import Topology
from ..command import Command

import logging
logger = logging.getLogger(__name__)

class ClientScore(Transferable, is_base_type=True):
    def calculate_client_score(
        self, command: CalculateClientScoreCommand
    ) -> float | int:
        """Calculate the client score

        Args:
            command (CalculateClientScoreCommand): The command to calculate the score

        Returns:
            float | int: The score
        """
        raise NotImplementedError("ClientScore is an abstract class")


class DatasetLengthScore(ClientScore):
    def __init__(self, dataset_key: str) -> None:
        """Calculate the score based on the length of the dataset

        Args:
            dataset_key (str): The key of the dataset
        """
        self.dataset_key = dataset_key

    def calculate_client_score(
        self, command: CalculateClientScoreCommand
    ) -> float | int:
        return len(command.client_rm.gr(self.dataset_key, SampleDataset))


class ResourceScore(ClientScore):
    def __init__(self, resource_key: str) -> None:
        """Calculate the score based on the length of the dataset

        Args:
            resource_key (str): The key of the resource
        """
        self.resource_key = resource_key

    def calculate_client_score(
        self, command: CalculateClientScoreCommand
    ) -> float | int:
        return command.client_rm.gr(self.resource_key, float | int)


class CalculateClientScoreCommand(Command):
    def __init__(
        self,
        client_score: ClientScore,
        softmax: bool = False,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Calculate the client score

        Args:
            client_score (ClientScore): The client score to calculate
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.client_score = client_score
        self.softmax = softmax

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_clients: list[str],
    ) -> None:
        # reset the scores
        for client in topology.clients:
            # delete property if exists
            if type(self.client_score).__name__ in client.data:
                del client.data[type(self.client_score).__name__]

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        client_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        """Set the client score on the topology

        Args:
            topology (Topology): The topology of the network
            resource_manager (ResourceManager): The resource manager of the server
            client_name (str): The name of the client
            execution_response (ExecutionResponse): The response from the client
            instruction_uuid (str): The uuid of the instruction
        """
        topology.nodes[client_name].data[
            execution_response.get_data()["score_type"]
        ] = execution_response.get_data()["score"]

        logger.warning(
            f"Client {client_name} has a score of {execution_response.get_data()['score']} ({execution_response.get_data()['score_type']})"
        )

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        if self.softmax:
            # get all weights for clients that have a weight
            weights = {
                client.name: client.data[type(self.client_score).__name__]
                for client in topology.clients
                if type(self.client_score).__name__ in client.data
            }

            weights_list = torch.tensor(list(weights.values()))
            softmax = torch.softmax(weights_list, dim=0)
            weights = {
                client: score.item() for client, score in zip(weights.keys(), softmax)
            }

            # set the weights
            for client in topology.clients:
                if client.name in weights:
                    print("setting weight", client.name, weights[client.name])
                    client.data[type(self.client_score).__name__] = weights[client.name]

    def execute(self) -> ClientScoreResponse:
        return ClientScoreResponse(
            score=self.client_score.calculate_client_score(self),
            score_type=type(self.client_score).__name__,
        )
