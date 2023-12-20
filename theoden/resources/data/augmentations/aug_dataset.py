import random

import numpy as np
import torch

from ....common import Transferable
from .. import Sample, SampleDataset
from . import Augmentation, IdentityAugmentation


class AugmentationDataset(SampleDataset, Transferable):
    def __init__(
        self,
        dataset: SampleDataset,
        augmentation: Augmentation | None = None,
        seed: int | None = None,
    ) -> None:
        """SampleDataset that applies an augmentation to each sample.

        Args:
            dataset (SampleDataset): Dataset to augment
            augmentation (Augmentation, optional): Augmentation to apply. Defaults to None.
            seed (int, optional): Seed to use for pseudo-random augmentations. Defaults to None.
        """

        super().__init__()
        self.dataset = dataset
        self.set_augmentation(augmentation)
        if seed == None:
            self.pseudo_random = False
        else:
            self.init_seeds(seed)

    def init_seeds(self, seed: int) -> None:
        self.pseudo_random = True
        np.random.seed(seed)
        self.seeds = (np.random.random(len(self)) * 10000).astype(np.int64)
        np.random.seed()

    def set_augmentation(self, augmentation: Augmentation | None = None) -> None:
        if augmentation is None:
            self.augmentation = IdentityAugmentation()
            return
        assert isinstance(augmentation, Augmentation)
        self.augmentation = augmentation

    def __getitem__(self, index: int) -> Sample:
        if self.pseudo_random:
            random.seed(self.seeds[index])
            np.random.seed(self.seeds[index])
            torch.manual_seed(self.seeds[index])
        return self.augmentation(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)
