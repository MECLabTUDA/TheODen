from ...common import ExecutionResponse, Transferable
from ..condition import Condition
from ..instructions import (
    Action,
    Distribution,
    Instruction,
    InstructionBundle,
    InstructionStatus,
)
from .request import ServerRequest


class PullCommandRequest(ServerRequest, Transferable):
    """A request to pull a command from the server.

    This request will pull a command from the server and return it as a dictionary.
    It is the main request used by the nodes communicating with the server and the tool to distribute commands to the nodes.
    """

    def __init__(self, uuid: None | str = None, **kwargs):
        """A request to pull a command from the server.

        Args:
            uuid (None | str, optional): The uuid of the request. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)

    @property
    def operation_head(self) -> Instruction | InstructionBundle | Condition | None:
        """Returns the operation head, which is the first operation in the operations list.
        If the operation head is an instruction set, unpack it and add it to the operations list as separate instructions.

        Returns:
            Instruction | InstructionBundle | Condition | None: The operation head. If there are no operations, returns None.
        """
        if self.server.operations and isinstance(
            self.server.operations[0], InstructionBundle
        ):
            # if the next operation is an instruction set, we need to unpack it and add it to the operations list as separate instructions
            self.server.operations = (
                self.server.operations[0].instructions + self.server.operations[1:]
            )
        return self.server.operations[0] if self.server.operations else None

    def execute(self) -> ExecutionResponse | None:
        """Executes the request and returns, if available, the command to be executed by the node.

        Returns:
            dict: The command to be executed by the node.
        """
        # If there are no operations, return an empty dictionary
        if not self.operation_head:
            return ExecutionResponse(data={})

        # If the operation head is a condition, check if it is resolved. If it is, pop it from the operations list and continue
        while isinstance(self.operation_head, Condition):
            # If the condition is not resolved, return an empty dictionary
            if not self.operation_head.resolved(
                topology=self.server.topology,
                resource_manager=self.server.resources,
            ):
                return ExecutionResponse(data={})
            # If the condition is resolved, pop it from the operations list and continue
            else:
                self.server.history.append(self.server.operations.pop(0))

        if isinstance(self.operation_head, Instruction):
            if self.operation_head.status is InstructionStatus.COMPLETED:
                # If the instruction is completed, check if it has a successor. If it does, add it to the operations list
                successor = self.operation_head.successor
                self.server.history.append(self.server.operations.pop(0))
                if successor is not None and isinstance(successor, list):
                    self.server.operations = successor + self.server.operations

            # Infer the command from the instruction and return it as a dictionary
            if not self.operation_head:
                return ExecutionResponse(data={})

        # If the operation head is an instruction, check if it is completed. If it is, pop it from the operations list and continue
        if isinstance(self.operation_head, Distribution):
            command = self.operation_head.infer_command(
                node_name=self.node_name,
                topology=self.server.topology,
                resource_manager=self.server.resources,
            )
            return ExecutionResponse(data=command.dict() if command is not None else {})

        elif isinstance(self.operation_head, Action):
            if self.operation_head.status is InstructionStatus.CREATED:
                self.operation_head(
                    topology=self.server.topology,
                    resource_manager=self.server.resources,
                )
            return ExecutionResponse(data={})
