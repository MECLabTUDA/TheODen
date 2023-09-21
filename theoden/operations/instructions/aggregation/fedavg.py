import torch

from .aggregator import Aggregator
from ....common import Transferable
from ....topology.topology_register import TopologyRegister
from ....resources.resource import ResourceRegister
from ....common.utils import create_sorted_lists


class FedAvgAggregator(Aggregator, Transferable):
    def __init__(
        self,
        model_key: str | list[str] = "@all",
        optimizer_key: str | list[str] | None = None,
        client_score: str = "uniform",
    ):
        super().__init__(model_key, optimizer_key, client_score=client_score)

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, any],
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ):
        dict_of_weights = {
            v: 1 / len(topology_register.get_connected_nodes())
            for v in topology_register.get_connected_nodes()
        }

        new_dict = self._recursive_averaging(
            *create_sorted_lists(resources, dict_of_weights)
        )

        return new_dict
