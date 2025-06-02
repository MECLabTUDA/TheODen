

import torch
from tqdm import tqdm

from ....common import ExecutionResponse, Transferable
from ....resources.data.dataset import SampleDataset
from ..command import Command

import logging
logger = logging.getLogger(__name__)


class InspectLabelDistributionCommand(Command, Transferable):
    def __init__(
        self,
        dataset: str = "dataset:train",
        label: str = "class_label",
        use_sampler: bool = False,
        num_classes: int | None = 2,
        start_at: int = 0,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Inspect the label distribution of a dataset

        Args:
            dataset (str, optional): The key of the dataset to inspect. Defaults to "dataset:train".
            label (str, optional): The label to inspect. Defaults to "class_label".
            use_sampler (bool, optional): Whether to use the sampler. Defaults to False.
            num_classes (int | None, optional): The number of classes. Defaults to 2.
            start_at (int, optional): The starting class. Defaults to 0.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.dataset = dataset
        self.use_sampler = use_sampler
        self.label = label
        self.num_classes = num_classes
        self.start_at = start_at

    def execute(self) -> ExecutionResponse | None:
        dataset = self.client_rm.gr(key=self.dataset, assert_type=SampleDataset)

        labels = torch.zeros(self.num_classes)

        for sample in tqdm(
            dataset.get_dataloader(batch_size=1, shuffle=False),
            desc="Inspecting label distribution",
        ):
            labels += torch.bincount(
                sample[self.label].reshape(-1) - self.start_at,
                minlength=self.num_classes,
            )

        for c, v in enumerate(list(labels / torch.sum(labels))):
            logger.info(f"Class {c + self.start_at}: {v:.2f}")
