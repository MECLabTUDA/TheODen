import torch
from torchvision.transforms import Compose, ToTensor

from ..resources.data import DatasetAdapter, Sample
from ..common import GlobalContext, Transferable


class MedMNISTAdapter(DatasetAdapter):
    def __init__(
        self,
        dataset_class: type,
        root: str,
        split: str = "test",
        transform=None,
        download: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_class(
                split=split,
                as_rgb=True,
                transform=ToTensor() if transform == None else transform,
                download=download,
                root=root,
            ),
            f"{dataset_class.__name__}_{split}",
            **kwargs,
        )

    def __getitem__(self, index) -> Sample:
        image, label = self.dataset[index]
        return Sample({"image": image, "class_label": torch.tensor(label[0])})


class PathMNIST_Adapted(MedMNISTAdapter, Transferable):
    def __init__(
        self, split: str = "test", transform=None, download: bool = False, **kwargs
    ) -> None:
        from medmnist import PathMNIST

        super().__init__(
            PathMNIST,
            split=split,
            transform=transform,
            root=GlobalContext().get_dataset_path("pathmnist"),
            download=download,
            **kwargs,
        )
