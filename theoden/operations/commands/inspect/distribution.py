from tqdm import tqdm
import torch

from collections import Counter

from typing import Optional
from ....common import Transferable, ExecutionResponse
from ..command import Command
from ....resources.data.dataset import SampleDataset


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
        super().__init__(uuid=uuid, **kwargs)
        self.dataset = dataset
        self.use_sampler = use_sampler
        self.label = label
        self.num_classes = num_classes
        self.start_at = start_at

    def execute(self) -> ExecutionResponse | None:
        dataset = self.node_rm.gr(key=self.dataset, assert_type=SampleDataset)

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
            print(f"Class {c + self.start_at}: {v:.2f}")
