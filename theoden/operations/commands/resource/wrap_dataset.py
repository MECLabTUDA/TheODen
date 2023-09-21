from typing import Type

from ..command import Command
from ....resources import SampleDataset


class WrapDatasetCommand(Command):
    def __init__(
        self,
        dataset: str,
        wrapper: Type[SampleDataset],
        *,
        node=None,
        uuid=None,
        **kwargs
    ):
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.dataset = dataset
        self.kwargs = kwargs
        self.wrapper = wrapper

    def execute(self) -> None:
        wrapped_dataset = self.wrapper(
            dataset=self.node_rr.gr(self.dataset, SampleDataset), **self.kwargs
        ).init_after_deserialization()
        self.node_rr.sr(key=self.dataset, resource=wrapped_dataset)
