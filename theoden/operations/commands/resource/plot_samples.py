import matplotlib.pyplot as plt

from ....common import ExecutionResponse, Transferable
from ..command import Command


class PlotSamplesCommand(Command, Transferable):
    def __init__(
        self,
        num_samples: int = 5,
        dataset_key: str = "dataset:train",
        *,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.num_samples = num_samples
        self.dataset_key = dataset_key

    def execute(self) -> ExecutionResponse | None:
        self.node_rm.gr(self.dataset_key).sample(self.num_samples)
        plt.show()
