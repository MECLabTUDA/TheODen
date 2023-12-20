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
        super().__init__(
            dataset=key,
            wrapper=AugmentationDataset,
            uuid=uuid,
            augmentation=augmentation,
            seed=seed,
            **kwargs
        )
        self.mode = mode


class SetNodeSpecificAugmentationCommand(Command, Transferable):
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
        super().__init__(uuid=uuid, **kwargs)
        self.augmentations = augmentations
        self.key = key
        self.mode = mode
        self.seed = seed

    def node_specific_modification(
        self, distribution_table: "DistributionStatusTable", node_name: str
    ) -> Command:
        included = distribution_table.selected
        key = sorted(included).index(node_name)

        return SetAugmentationCommand(
            augmentation=self.augmentations[key],
            key=self.key,
            mode=self.mode,
            seed=self.seed,
            uuid=self.uuid,
        )
