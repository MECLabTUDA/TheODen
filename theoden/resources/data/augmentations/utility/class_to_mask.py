import torch

from ..... import Transferable
from ... import Sample
from .. import Augmentation


class ClassToMaskAugmentation(Augmentation, Transferable):
    def _augment(self, sample: Sample) -> Sample:
        class_label: torch.Tensor = sample["class_label"]
        _, height, width = sample["image"].shape
        sample["segmentation_mask"] = torch.zeros((height, width)) + class_label
        return sample
