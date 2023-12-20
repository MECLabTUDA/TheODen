from ..operations import (
    Command,
    Condition,
    Instruction,
    InstructionBundle,
    ServerRequest,
)

operations_types = Instruction | InstructionBundle | Condition | ServerRequest | Command


class OperationWhiteList:
    def __init__(self, *operations: operations_types) -> None:
        """A whitelist of operations to allow

        Args:
            *operations (operations_types): A list of operations to allow
        """
        self.operations = operations

    def allows(
        self,
        item: type[operations_types] | list[operations_types] | operations_types | str,
    ) -> bool:
        """Check if an operation is allowed

        Args:
            item (type[operations_types] | list[operations_types] | operations_types | str): The operation to check

        Returns:
            bool: True if the operation is allowed, False otherwise
        """

        if isinstance(item, str):
            return any([item == operation.__name__ for operation in self.operations])
        elif isinstance(item, list):
            return all([self.allows(type(operation)) for operation in item])
        elif isinstance(item, type):
            return item in self.operations
        else:
            return type(item) in self.operations


class OperationBlackList(OperationWhiteList):
    def __init__(self, *operations: operations_types) -> None:
        """A blacklist of operations to disallow

        Args:
            *operations (operations_types): A list of operations to disallow
        """
        super().__init__(*operations)

    def allows(
        self,
        item: type[operations_types] | list[operations_types] | operations_types | str,
    ) -> bool:
        return super().allows(item) is False
