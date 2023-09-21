from typing import Optional

import matplotlib.pyplot as plt

from ....common import Transferable, ExecutionResponse
from ..command import Command


class PlotSamplesCommand(Command, Transferable):
    def __init__(
        self,
        num_samples: int = 5,
        dataset_key: str = "dataset:train",
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.num_samples = num_samples
        self.dataset_key = dataset_key

    def execute(self) -> ExecutionResponse | None:
        self.node_rr.gr(self.dataset_key).sample(self.num_samples)
        plt.show()
