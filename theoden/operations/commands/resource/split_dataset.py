from typing import Optional, Any

from theoden.operations.commands.command import Command
from theoden.common import Transferable
from theoden.resources.data import LocalDatasetSplit, SampleDataset


class SplitDatasetLocallyCommand(Command, Transferable):
    def __init__(
        self,
        split: LocalDatasetSplit,
        resource: str = "dataset",
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.split = split
        self.resource = resource

    def execute(self) -> Any:
        assert isinstance(
            self.node.resource_register[self.resource],
            SampleDataset,
        ), "Split can only be applied to a `SampleDataset`"

        # apply split to the dataset
        splitted_datasets = self.split.split(self.node.resource_register[self.resource])

        # register splitted dataset int he format `original_name:split` e.g. `BCSS:test`
        for split_name, split_set in splitted_datasets.items():
            self.node.resource_register[f"{self.resource}:{split_name}"] = split_set
