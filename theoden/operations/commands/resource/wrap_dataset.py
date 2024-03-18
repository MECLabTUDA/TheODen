from ....resources import SampleDataset
from ..command import Command


class WrapDatasetCommand(Command):
    def __init__(
        self,
        dataset: str,
        wrapper: type[SampleDataset],
        *,
        uuid: str | None = None,
        **kwargs
    ):
        """Wrap a dataset with a wrapper dataset

        Args:
            dataset (str): The key of the dataset to wrap
            wrapper (type[SampleDataset]): The wrapper dataset
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.dataset = dataset
        self.kwargs = kwargs
        self.wrapper = wrapper

    def execute(self) -> None:
        wrapped_dataset = self.wrapper(
            dataset=self.client_rm.gr(self.dataset, SampleDataset), **self.kwargs
        ).init_after_deserialization()
        self.client_rm.sr(key=self.dataset, resource=wrapped_dataset)
        return None
