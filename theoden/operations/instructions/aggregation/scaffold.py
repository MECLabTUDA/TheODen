import torch
from torch._tensor import Tensor

from ....resources import ResourceManager
from ....resources.meta import DictCheckpoint
from ....topology import Topology
from .aggregate import Aggregator


class ScaffoldAggregator(Aggregator):
    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> dict[str, Tensor]:
        ...
