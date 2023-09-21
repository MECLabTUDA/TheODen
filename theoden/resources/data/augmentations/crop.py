import numpy as np

from .augmentation import Augmentation
from ....common import Transferable
from ..sample import Sample


class CroppingAugmentation(Augmentation, Transferable):
    def __init__(
        self, size: tuple[int, int] | int, labels: list[str] | None = None
    ) -> None:
        super().__init__()
        self.size = size
        self.labels = labels if labels is not None else ["segmentation_mask"]

    def _augment(self, sample: Sample) -> Sample:
        image = sample["image"]

        if isinstance(self.size, int):
            size = (self.size, self.size)
        else:
            size = self.size

        if image.shape[1] < size[0] or image.shape[2] < size[1]:
            raise ValueError(
                f"Image size ({image.shape}) is smaller than crop size ({size})"
            )

        x = (
            np.random.randint(0, image.shape[1] - size[0])
            if size[0] < image.shape[1]
            else 0
        )
        y = (
            np.random.randint(0, image.shape[2] - size[1])
            if size[1] < image.shape[2]
            else 0
        )

        sample["image"] = image[..., x : x + size[0], y : y + size[1]]
        for label in self.labels:
            # check if label is in sample
            if label in sample:
                new = sample[label][..., x : x + size[0], y : y + size[1]]
                sample[label] = new

        return sample
