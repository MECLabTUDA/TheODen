import torch
from torchvision.transforms import Compose, ToTensor

from theoden.resources.data.sample import Sample

from ..resources.data import DatasetAdapter, Sample
from ..common import GlobalContext, Transferable


class Camelyon17_Adapted(DatasetAdapter, Transferable):
    def __init__(self, split_scheme: str = "official", **kwargs) -> None:
        from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

        super().__init__(
            Camelyon17Dataset(
                root_dir=GlobalContext().get_dataset_path("camelyon17"),
                download=True,
                split_scheme=split_scheme,
            ),
            "Camelyon17",
            **kwargs
        )

        self.to_tensor = ToTensor()

    def __getitem__(self, index: int) -> Sample:
        image, class_label, meta = self.dataset[index]
        return Sample(
            {"image": self.to_tensor(image), "class_label": class_label},
            metadata={
                "institute": meta[0].item(),
                "wsi": meta[1].item(),
                "3": meta[2].item(),
                "4": meta[3].item(),
            },
        )
