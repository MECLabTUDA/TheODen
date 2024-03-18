from ....common import ExecutionResponse, Transferable
from ..command import Command


class SetDataLoaderCommand(Command, Transferable):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int | None = 10,
        shuffle: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        split: str = "train",
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the dataloader on the client

        Args:
            batch_size (int, optional): The batch size. Defaults to 16.
            num_workers (int | None, optional): The number of workers. Defaults to 10.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
            pin_memory (bool, optional): Whether to pin the memory. Defaults to False.
            persistent_workers (bool, optional): Whether to use persistent workers. Defaults to False.
            split (str, optional): The split to set the dataloader for. Defaults to "train".
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.additional_kwargs = kwargs

    def execute(self) -> ExecutionResponse | None:
        self.client.resources[f"dataloader:{self.split}"] = self.client.resources[
            f"dataset:{self.split}"
        ].get_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            **self.additional_kwargs,
        )
        return None
