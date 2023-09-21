from .. import Augmentation
from ... import Sample
from ..... import Transferable


class IdentityAugmentation(Augmentation, Transferable):
    def _augment(self, sample: Sample) -> Sample:
        return sample
