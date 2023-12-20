import numpy as np

from ...common import Transferable
from .dataset import SampleDataset
from .sample import Sample


class Mapping(Transferable, is_base_type=True):
    """Base class for mapping labels of a sample according to a mapping scheme."""

    def __call__(self, sample: Sample) -> Sample:
        """Maps the labels of a sample according to a mapping scheme.

        Args:
            sample (Sample): The sample to map.

        Returns:
            Sample: The mapped sample.
        """
        raise NotImplementedError("Please implement this method")


class SegmentationMapping(Mapping, Transferable):
    def __init__(
        self,
        map_classes: list[tuple[list[int], int]] | None = None,
        shift: int = 0,
        except_map: tuple[list[int], int] | None = None,
        mask_key: str = "segmentation_mask",
    ) -> None:
        self.map_classes = map_classes
        self.shift = shift
        self.except_map = except_map
        self.mask_key = mask_key

    def __call__(self, sample: Sample) -> Sample:
        sample_mask = sample[self.mask_key]

        shifted_mask = sample_mask + self.shift

        if self.map_classes is not None:
            for classes, new_class in self.map_classes:
                for class_ in classes:
                    shifted_mask[shifted_mask == class_] = new_class

        if self.except_map is not None:
            # map all except the except classes to the new value
            except_classes, new_class = self.except_map
            # if a class is not in except_classes, map it to new_class
            shifted_mask[np.isin(shifted_mask, except_classes, invert=True)] = new_class

        sample[self.mask_key] = shifted_mask

        return sample


class MappingDataset(SampleDataset, Transferable):
    def __init__(self, dataset: SampleDataset, mapping: Mapping) -> None:
        super().__init__()
        self.dataset = dataset
        self.mapping = mapping

    def __getitem__(self, index) -> Sample:
        return self.mapping(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)
