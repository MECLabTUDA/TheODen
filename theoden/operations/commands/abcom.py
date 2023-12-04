from ...common import (
    ExecutionResponse,
    NotImplementedAbstractCommandError,
    Transferable,
)
from .command import Command


class AbstractCommand(Command, Transferable):
    def __init__(self, *, uuid: str | None = None, **kwargs) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.kwargs = kwargs

    def execute(self) -> ExecutionResponse | None:
        raise NotImplementedAbstractCommandError(
            f"Abstract command {self.__class__.__name__} not implemented."
        )


class ABCLoadDatasetCommand(AbstractCommand, Transferable):
    """
    The ABCLoadDataset class is an abstract command
    """


class ABCTrainRoundCommand(AbstractCommand, Transferable):
    """
    The ABCTrainRoundCommand class is an abstract command
    """

    def __init__(
        self,
        communication_round: int | None = None,
        num_epochs: int | None = None,
        num_steps: int | None = None,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.communication_round = communication_round
        self.num_epochs = num_epochs
        self.num_steps = num_steps


class ABCInitModelCommand(AbstractCommand, Transferable):
    """
    The ABCInitModelCommand class is an abstract command
    """
