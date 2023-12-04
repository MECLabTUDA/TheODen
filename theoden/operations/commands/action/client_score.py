from __future__ import annotations

from theoden.common import ExecutionResponse
from theoden.resources import ResourceManager
from theoden.topology import Topology
from ..command import Command
from ....common import ClientScoreResponse, Transferable
from ....resources import SampleDataset


class ClientScore(Transferable, is_base_type=True):
    def calculate_client_score(
        self, command: CalculateClientScoreCommand
    ) -> float | int:
        raise NotImplementedError("ClientScore is an abstract class")


class DatasetLengthScore(ClientScore):
    def __init__(self, dataset_key: str) -> None:
        self.dataset_key = dataset_key

    def calculate_client_score(
        self, command: CalculateClientScoreCommand
    ) -> float | int:
        return len(command.node_rm.gr(self.dataset_key, SampleDataset))


class CalculateClientScoreCommand(Command):
    def __init__(
        self,
        client_score: ClientScore,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.client_score = client_score

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        node_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        topology.nodes[node_name].data[
            execution_response.get_data()["score_type"]
        ] = execution_response.get_data()["score"]

        print(
            f"Client {node_name} has a score of {execution_response.get_data()['score']} ({execution_response.get_data()['score_type']})"
        )

    def execute(self) -> ClientScoreResponse:
        return ClientScoreResponse(
            score=self.client_score.calculate_client_score(self),
            score_type=type(self.client_score).__name__,
        )
