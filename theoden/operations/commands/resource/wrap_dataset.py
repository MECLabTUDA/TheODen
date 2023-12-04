from ..command import Command
from ....resources import SampleDataset


class WrapDatasetCommand(Command):
    def __init__(
        self,
        dataset: str,
        wrapper: type[SampleDataset],
        *,
        uuid: str | None = None,
        **kwargs
    ):
        super().__init__(uuid=uuid, **kwargs)
        self.dataset = dataset
        self.kwargs = kwargs
        self.wrapper = wrapper

    def execute(self) -> None:
        wrapped_dataset = self.wrapper(
            dataset=self.node_rm.gr(self.dataset, SampleDataset), **self.kwargs
        ).init_after_deserialization()
        self.node_rm.sr(key=self.dataset, resource=wrapped_dataset)
