from ..operations import Command


class CommandWhiteList:
    def __init__(self, *commands: Command) -> None:
        """A whitelist of commands to allow

        Args:
            *commands (Command): A list of commands to allow
        """
        self.commands = commands

    def allows(self, item: type[Command] | list[Command] | Command | str) -> bool:
        """Check if a command is allowed

        Args:
            item (type[Command] | list[Command] | Command | str): The command(s) to check

        Returns:
            bool: True if the command is allowed, False otherwise
        """

        if isinstance(item, str):
            return any([item == command.__name__ for command in self.commands])
        elif isinstance(item, list):
            return all([self.allows(type(command)) for command in item])
        elif isinstance(item, type):
            return item in self.commands
        else:
            return type(Command) in self.commands


class CommandBlackList(CommandWhiteList):
    def __init__(self, *commands: Command) -> None:
        """A blacklist of commands to disallow

        Args:
            *commands (Command): A list of commands to disallow
        """
        super().__init__(*commands)

    def allows(self, item: type[Command] | list[Command] | Command | str) -> bool:
        return super().allows(item) is False
