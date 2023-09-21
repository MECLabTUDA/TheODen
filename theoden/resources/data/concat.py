from torch.utils.data import ConcatDataset

from theoden.common.transferables import Transferable

from ...common import Transferable
from .dataset import WrapperSampleDataset, SampleDataset
from .metadata_set import MetadataWrapperDataset


class ConcatSampleDataset(WrapperSampleDataset, Transferable):
    def __init__(
        self, datasets: list[SampleDataset] | dict[int | str, SampleDataset]
    ) -> None:
        """Concatenates multiple datasets into one dataset.

        Args:
            datasets (list[SampleDataset] | dict[int | str, SampleDataset]): datasets to concatenate. If dict, the key will be used as metadata for the samples.

        Raises:
            TypeError: if any of the datasets is not a SampleDataset
        """

        super().__init__()
        if isinstance(datasets, list):
            self.datasets = {i: ds for i, ds in enumerate(datasets)}

        for name, ds in datasets.items():
            if not isinstance(ds, SampleDataset):
                raise TypeError(
                    f"Expected SampleDataset, got {type(ds).__name__} with name/index {name}"
                )

        self.datasets = datasets

    def init_after_deserialization(self) -> Transferable:
        self.dataset = ConcatDataset(
            [
                MetadataWrapperDataset(dataset=ds, metadata={"dataset": name})
                for name, ds in self.datasets.items()
            ]
        )
        return self
