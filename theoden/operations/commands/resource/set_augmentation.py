from typing import Optional

from .wrap_dataset import WrapDatasetCommand
from ....common import Transferable
from ....resources import Augmentation, AugmentationDataset
from .. import Command


class SetAugmentationCommand(WrapDatasetCommand, Transferable):
    def __init__(
        self,
        augmentation: Augmentation,
        key: str = "dataset:train",
        mode: str = "replace",
        seed: Optional[int] = None,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(
            dataset=key,
            wrapper=AugmentationDataset,
            node=node,
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
        seed: Optional[int] = None,
        *,
        node: Optional["Node"] = None,
        uuid: str | None = None,
        **kwargs
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.augmentations = augmentations
        self.key = key
        self.mode = mode
        self.seed = seed

    def node_specific_modification(
        self, status_register: dict[str, "StatusTable"], node_uuid: str
    ) -> Command:
        table = status_register[self.uuid]
        included = table.get_included()
        key = sorted(included).index(node_uuid)

        return SetAugmentationCommand(
            augmentation=self.augmentations[key],
            key=self.key,
            mode=self.mode,
            seed=self.seed,
            uuid=self.uuid,
        )
