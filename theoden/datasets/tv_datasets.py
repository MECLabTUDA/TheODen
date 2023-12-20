import torch
from torchvision.datasets import (  # SVHN,; FashionMNIST,; LSUN,; STL10,; KMNIST,; Places365,; ImageNet,
    CIFAR10,
    CIFAR100,
    MNIST,
    CelebA,
    VisionDataset,
)
from torchvision.transforms import ToTensor

from ..common import GlobalContext, Transferable
from ..resources.data.dataset import DatasetAdapter
from ..resources.data.sample import Sample


class TorchvisionAdapter(DatasetAdapter):
    def __init__(self, dataset: VisionDataset, split: str, **kwargs) -> None:
        super().__init__(dataset, name=f"{type(dataset).__name__}_{split}", **kwargs)

    def get_sample(self, index: int) -> Sample:
        image, label = self.dataset[index]
        return Sample({"image": image, "class_label": torch.tensor(label)})


class CIFAR10_Adapted(
    TorchvisionAdapter,
    Transferable,
    description="""`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.""",
):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            CIFAR10(
                root=GlobalContext().get_dataset_path("cifar10", parameter_path=root),
                train=train,
                transform=ToTensor(),
                download=False,
            ),
            split="train" if train else "test",
            **kwargs,
        )


class CIFAR100_Adapted(
    TorchvisionAdapter,
    Transferable,
    description="""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.""",
):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            CIFAR100(
                root=GlobalContext().get_dataset_path("cifar100", parameter_path=root),
                train=train,
                transform=ToTensor(),
                download=True,
            ),
            split="train" if train else "test",
            **kwargs,
        )


class MNIST_Adapted(TorchvisionAdapter, Transferable):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            MNIST(
                root=GlobalContext().get_dataset_path("mnist", parameter_path=root),
                train=train,
                transform=ToTensor(),
                download=True,
            ),
            split="train" if train else "test",
            **kwargs,
        )


class CelebADataset(DatasetAdapter, Transferable):
    def __init__(
        self,
        root: str | None = None,
        split: str = "train",
        target_type: list[str] | str = "attr",
        **kwargs,
    ):
        super().__init__(
            CelebA(
                root=GlobalContext().get_dataset_path("celeba", parameter_path=root),
                download=True,
                split=split,
                target_type=target_type,
                transform=ToTensor(),
            ),
            name=f"celeba_{split}",
            **kwargs,
        )

    def get_sample(self, index: int) -> Sample:
        image, label = self.dataset[index]
        return Sample({"image": image, "class_label": torch.tensor(label)})
