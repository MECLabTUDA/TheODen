from typing import Optional

from ....common import (
    ExecutionResponse,
    NotImplementedAbstractCommandError,
    Transferable,
)
from ..command import Command


class AbstractCommand(Command, Transferable):
    def __init__(
        self, *, node: Optional["Node"] = None, uuid: str | None = None, **kwargs
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.kwargs = kwargs

    def execute(self) -> ExecutionResponse | None:
        raise NotImplementedAbstractCommandError(
            f"Abstract command {self.__class__.__name__} not implemented."
        )


class ABCLoadDatasetCommand(AbstractCommand, Transferable):
    """
    The ABCLoadDataset class is an abstract command
    """


class ABCTrainEpochCommand(AbstractCommand, Transferable):
    """
    The ABCTrainEpochCommand class is an abstract command
    """


class ABCInitModelCommand(AbstractCommand, Transferable):
    """
    The ABCInitModelCommand class is an abstract command
    """
