from .sample import Sample
from ...common import Transferable
from .dataset import SampleDataset


class MetadataWrapperDataset(SampleDataset, Transferable):
    def __init__(self, dataset: SampleDataset, metadata: dict[str, any]) -> None:
        super().__init__()
        self.dataset = dataset
        self.metadata = metadata

    def __getitem__(self, index: int) -> Sample:
        sample = self.dataset.__getitem__(index)
        sample.metadata.update(self.metadata)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)
