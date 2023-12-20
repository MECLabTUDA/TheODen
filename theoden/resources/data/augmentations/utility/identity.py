from ..... import Transferable
from ... import Sample
from .. import Augmentation


class IdentityAugmentation(Augmentation, Transferable):
    def _augment(self, sample: Sample) -> Sample:
        return sample
