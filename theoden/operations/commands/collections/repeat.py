from copy import deepcopy

from theoden.operations.commands.command import Command
from theoden.operations.commands.meta.sequential import SequentialCommand


class RepeatNTimesCommand(SequentialCommand, return_super_class_dict=True):
    def __init__(self, base_command: Command, n: int, **kwargs) -> None:
        """The RepeatNTimesCommand is a convenience class for creating a sequence of commands that repeats a given command n times.

        Args:
            base_command (Command): The command to repeat.
            n (int): The number of times to repeat the command.

        Raises:
            ValueError: If n is not positive.
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        super().__init__([deepcopy(base_command) for _ in range(n)], **kwargs)
