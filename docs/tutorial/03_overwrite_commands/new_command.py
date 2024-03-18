from theoden.operations import Command, PrintResourceKeysCommand, ABCInitModelCommand


class IDoSomethingDifferentCommand(Command, implements=PrintResourceKeysCommand):
    def execute(self) -> None:
        print("I do something different")
        return None


class IDontInitTheModelCommand(Command, implements=ABCInitModelCommand):
    def execute(self) -> None:
        print("I don't init the model")
        return None
