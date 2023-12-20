import numpy as np
from torchvision.transforms import Resize

from ....common import Transferable
from ..sample import Sample
from .augmentation import Augmentation


class ResizeAugmentation(Augmentation, Transferable):
    def __init__(
        self, size: tuple[int, int] | int, labels: list[str] | None = None
    ) -> None:
        super().__init__()
        self.size = size
        self.labels = labels if labels is not None else ["segmentation_mask"]
        self.resize = Resize(size, antialias=True)

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.resize(sample["image"])
        for label in self.labels:
            # check if label is in sample
            if label in sample:
                # add axis if shape is 2D
                if len(sample[label].shape) == 2:
                    sample[label] = self.resize(sample[label].unsqueeze(0)).squeeze(0)
                else:
                    sample[label] = self.resize(sample[label])

        return sample
