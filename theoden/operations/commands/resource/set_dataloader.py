from typing import Optional, Union, Dict, Any, List

from theoden.operations.commands.command import Command
from theoden.common import Transferable


class SetDataLoaderCommand(Command, Transferable):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: Union[int, None] = 10,
        shuffle: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split: str = "train",
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.additional_kwargs = kwargs

    def execute(self) -> Any:
        self.node.resource_register[
            f"dataloader:{self.split}"
        ] = self.node.resource_register[f"dataset:{self.split}"].get_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            **self.additional_kwargs,
        )
