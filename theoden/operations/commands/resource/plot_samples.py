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
        """Plot samples from the dataset

        Args:
            num_samples (int, optional): The number of samples to plot. Defaults to 5.
            dataset_key (str, optional): The key of the dataset to plot samples from. Defaults to "dataset:train".
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.num_samples = num_samples
        self.dataset_key = dataset_key

    def execute(self) -> ExecutionResponse | None:
        self.client_rm.gr(self.dataset_key).sample(self.num_samples)
        plt.show()
