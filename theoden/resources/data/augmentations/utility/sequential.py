from ..... import Transferable
from ... import Sample
from .. import Augmentation


class SequentialAugmentation(Augmentation, Transferable):
    def __init__(self, augmentations: list[Augmentation]):
        self.augmentations = augmentations

    def _augment(self, sample: Sample) -> Sample:
        for augmentation in self.augmentations:
            sample = augmentation(sample)
        return sample
