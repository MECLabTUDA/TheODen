import random

import torch

from ....common import Transferable
from ..sample import Sample
from .augmentation import Augmentation


class RandomFlippingAugmentation(Augmentation, Transferable):
    def __init__(self, p: float = 0.5, labels: list[str] | None = None) -> None:
        super().__init__()
        self.p = p
        self.labels = labels if labels is not None else ["segmentation_mask"]

    def _augment(self, sample: Sample) -> Sample:
        axis = []

        if random.random() < self.p:
            axis.append(-1)
        if random.random() < self.p:
            axis.append(-2)

        if len(axis) > 0:
            img = torch.flip(sample["image"], axis)
            sample["image"] = img
            for label in self.labels:
                # check if label is in sample
                if label in sample:
                    new = torch.flip(sample[label], axis)
                    sample[label] = new

        return sample
