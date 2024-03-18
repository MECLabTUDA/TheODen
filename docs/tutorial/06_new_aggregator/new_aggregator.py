import torch

from theoden.operations import Aggregator
from theoden.resources import ResourceManager
from theoden.resources.meta import DictCheckpoint
from theoden.topology import Topology

import numpy as np


class SelectRandomOneAggregator(Aggregator):
    def __init__(self, client_score: type | None = None, **kwargs):
        super().__init__(client_score, **kwargs)

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> dict[str, torch.Tensor]:
        # select single client
        selected_client_name = np.random.choice(list(resources.keys()))

        print(f"Selected client: {selected_client_name}")

        # get the resource of the selected client
        selected_client_resource = resources[selected_client_name].to(dict)

        # return the resource of the selected client as new global model
        return selected_client_resource
