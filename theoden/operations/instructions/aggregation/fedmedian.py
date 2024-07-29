import torch

from ....resources import ResourceManager
from ....resources.meta import DictCheckpoint
from ....topology import Topology
from .aggregate import Aggregator


class MedianAggregator(Aggregator):
    """Aggregator that uses the median to aggregate the resources."""

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> dict[str, torch.Tensor]:

        new_global_model = {}

        # get model keys
        keys = resources[list(resources.keys())[0]].to(dict).keys()

        for parameter_key in keys:
            new_global_model[parameter_key] = torch.median(
                torch.stack(
                    [
                        resource.to(dict)[parameter_key]
                        for resource in resources.values()
                    ]
                ),
                dim=0,
            ).values

        return new_global_model
