from ....common import Transferable
from ....operations.commands.resource import SetResourceCommand
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
        """Load a dataset on the client

        Args:
            dataset (SampleDataset | dict[str, SampleDataset]): The dataset(s) to load
            dataset_key (str, optional): The resource key of the dataset. Defaults to "dataset".
            overwrite (bool, optional): Whether to overwrite the existing dataset. Defaults to True.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(
            key=dataset_key,
            resource=dataset,
            overwrite=overwrite,
            # if resource is a dict and unpack_dict is True, then unpack the dict and register each key-value pair
            unpack_dict=isinstance(dataset, dict),
            uuid=uuid,
            **kwargs,
        )
