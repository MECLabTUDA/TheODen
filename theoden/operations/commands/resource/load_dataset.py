from typing import Optional, Union, Dict, Any

from theoden.operations.commands.resource import SetResourceCommand
from theoden.common import Transferable
from theoden.resources.data import SampleDataset


class LoadDatasetCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        dataset: Union[SampleDataset, Dict[str, SampleDataset]],
        dataset_key: str = "dataset",
        overwrite: bool = True,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=dataset_key,
            resource=dataset,
            overwrite=overwrite,
            unpack_dict=isinstance(
                dataset, dict
            ),  # if resources is a dict and unpack_dict is True, then unpack the dict and register each key-value pair
            node=node,
            uuid=uuid,
            **kwargs,
        )
