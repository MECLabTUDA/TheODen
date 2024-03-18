from typing import TYPE_CHECKING

from ....common import Transferable
from ....resources import Augmentation, AugmentationDataset
from .. import Command
from .wrap_dataset import WrapDatasetCommand

if TYPE_CHECKING:
    from ....operations import DistributionStatusTable


class SetAugmentationCommand(WrapDatasetCommand, Transferable):
    def __init__(
        self,
        augmentation: Augmentation,
        key: str = "dataset:train",
        mode: str = "replace",
        seed: int | None = None,
        *,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        """Set the same augmentation on all clients

        Args:
            augmentation (Augmentation): The augmentation to set
            key (str, optional): The resource key of the dataset. Defaults to "dataset:train".
            mode (str, optional): The mode of the augmentation. Defaults to "replace".
            seed (int | None, optional): The seed for the augmentation. Defaults to None.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(
            dataset=key,
            wrapper=AugmentationDataset,
            uuid=uuid,
            augmentation=augmentation,
            seed=seed,
            **kwargs
        )
        self.mode = mode


class SetClientSpecificAugmentationCommand(Command, Transferable):
    def __init__(
        self,
        augmentations: list[Augmentation],
        key: str = "dataset:train",
        mode: str = "replace",
        seed: int | None = None,
        *,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        """Set different augmentations on different clients

        Args:
            augmentations (list[Augmentation]): The augmentations to set
            key (str, optional): The resource key of the dataset. Defaults to "dataset:train".
            mode (str, optional): The mode of the augmentation. Defaults to "replace".
            seed (int | None, optional): The seed for the augmentation. Defaults to None.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.augmentations = augmentations
        self.key = key
        self.mode = mode
        self.seed = seed

    def client_specific_modification(
        self, distribution_table: "DistributionStatusTable", client_name: str
    ) -> Command:
        included = distribution_table.selected
        key = sorted(included).index(client_name)

        return SetAugmentationCommand(
            augmentation=self.augmentations[key],
            key=self.key,
            mode=self.mode,
            seed=self.seed,
            uuid=self.uuid,
        )
