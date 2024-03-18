from typing_extensions import Self

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
    def __init__(self, dataset_name: str, split: str, **kwargs) -> None:
        super().__init__(None, name=f"{dataset_name}_{split}", **kwargs)

    def get_sample(self, index: int) -> Sample:
        image, label = self.dataset[index]
        return Sample({"image": image, "class_label": torch.tensor(label)})


class CIFAR10_Adapted(
    TorchvisionAdapter,
    description="""`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.""",
):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        download: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            type(CIFAR10).__name__,
            split="train" if train else "test",
            **kwargs,
        )
        self.root = root
        self.train = train
        self.download = download

    def init_after_deserialization(self) -> Self:
        self.dataset = CIFAR10(
            root=GlobalContext().get_dataset_path("cifar10", parameter_path=self.root),
            train=self.train,
            transform=ToTensor(),
            download=self.download,
        )
        return self


class CIFAR100_Adapted(
    TorchvisionAdapter,
    description="""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.""",
):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        download: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            type(CIFAR100).__name__,
            split="train" if train else "test",
            **kwargs,
        )
        self.root = root
        self.train = train
        self.download = download

    def init_after_deserialization(self) -> Self:
        self.dataset = CIFAR100(
            root=GlobalContext().get_dataset_path("cifar100", parameter_path=self.root),
            train=self.train,
            transform=ToTensor(),
            download=self.download,
        )
        return self


class MNIST_Adapted(TorchvisionAdapter):
    def __init__(
        self,
        root: str | None = None,
        train: bool = False,
        download: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            split="train" if train else "test",
            **kwargs,
        )
        self.root = root
        self.train = train
        self.download = download

    def init_after_deserialization(self) -> Self:
        self.dataset = (
            MNIST(
                root=GlobalContext().get_dataset_path(
                    "mnist", parameter_path=self.root
                ),
                train=self.train,
                transform=ToTensor(),
                download=self.download,
            ),
        )
        return self


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
