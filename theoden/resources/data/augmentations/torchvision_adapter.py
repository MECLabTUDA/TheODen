import torchvision.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
from torchvision.transforms import Compose as Compose
from ....common import Transferable, Transferables
from .augmentation import Augmentation
from .. import Sample
from .augmentation import Augmentation

all_transforms = transforms.transforms.__all__

all_auto = autoaugment.__all__

# Get all the attributes of torchvision.transforms
attrs = dir(transforms)


class TorchVisionTheodenAdapter(Transferable, is_base_type=True):
    ...


# Filter out the non-transforms attributes
transform_attrs = [attr for attr in attrs if not attr.startswith("_")]

# Iterate over all the transform attributes and print their names
for attr in transform_attrs:
    # Get the transform class by name
    transform_class = getattr(transforms, attr)

    # Check if the class is a subclass of torchvision.transforms.Transform
    if transform_class.__name__ in all_transforms:
        # mod = type(transform_class.__name__, (TorchVisionTheodenAdapter,), {})
        Transferable.make_transferable(
            transform_class, base_type=TorchVisionTheodenAdapter
        )

for attr in all_auto:
    # Get the transform class by name
    transform_class = getattr(autoaugment, attr)

    # Check if the class is a subclass of torchvision.transforms.Transform
    if transform_class.__name__ in all_auto:
        Transferable.make_transferable(
            transform_class, base_type=TorchVisionTheodenAdapter
        )


class TVAugmentation(Augmentation, Transferable):
    def __init__(self, transform: TorchVisionTheodenAdapter) -> None:
        self.transform = transform

    def _augment(self, sample: Sample) -> Sample:
        sample["image"] = self.transform(sample["image"])
        return sample


class TVCompose(TVAugmentation, Transferable):
    def __init__(self, transforms: list[TorchVisionTheodenAdapter]) -> None:
        super().__init__(transform=Compose(transforms))
