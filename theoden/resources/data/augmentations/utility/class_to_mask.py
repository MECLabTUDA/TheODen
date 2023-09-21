import torch

from .. import Augmentation
from ... import Sample
from ..... import Transferable


class ClassToMaskAugmentation(Augmentation, Transferable):
    def _augment(self, sample: Sample) -> Sample:
        class_label: torch.Tensor = sample["class_label"]
        _, height, width = sample["image"].shape
        sample["segmentation_mask"] = torch.zeros((height, width)) + class_label
        return sample
