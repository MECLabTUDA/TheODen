from ....operations.commands.resource import SetResourceCommand
from ....common import Transferable
from ....resources.data import SampleDataset


class LoadDatasetCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        dataset: SampleDataset | dict[str, SampleDataset],
        dataset_key: str = "dataset",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=dataset_key,
            resource=dataset,
            overwrite=overwrite,
            # if resource_manager is a dict and unpack_dict is True, then unpack the dict and register each key-value pair
            unpack_dict=isinstance(dataset, dict),
            uuid=uuid,
            **kwargs,
        )
