from tiatoolbox.tools.stainaugment import StainAugmentor

from ....common import Transferable
from ..sample import Sample
from .augmentation import Augmentation


class StainAugmentation(Augmentation, Transferable):
    def __init__(
        self,
        method: str = "vahadane",
        sigma1: float = 0.4,
        sigma2: float = 0.2,
        augment_background: bool = False,
        always_apply=False,
        p=0.5,
    ) -> None:
        super().__init__()
        self.augmentor = StainAugmentor(
            method=method,
            sigma1=sigma1,
            sigma2=sigma2,
            augment_background=augment_background,
            always_apply=always_apply,
            p=p,
        )

    def _augment(self, sample: Sample) -> Sample:
        try:
            self.augmentor.fit(self._transform_to_numpy(sample["image"]))
            sample["image"] = self._transform_to_tensor(self.augmentor.augment())
        except:
            pass

        return sample
