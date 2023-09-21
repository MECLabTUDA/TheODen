from typing import TYPE_CHECKING, Optional

from .request import ServerRequest
from ...common import Transferable
from ..instructions import Instruction, InstructionStatus, InstructionGroup
from ..stopper import Stopper

if TYPE_CHECKING:
    from theoden.topology.server import Server


class PullCommandRequest(ServerRequest, Transferable):
    """A request to pull a command from the server.

    This request will pull a command from the server and return it as a dictionary.
    It is the main request used by the nodes communicating with the server and the tool to distribute commands to the nodes.
    """

    def __init__(
        self, uuid: None | str = None, server: Optional["Server"] = None, **kwargs
    ):
        """A request to pull a command from the server.

        Args:
            uuid (None | str, optional): The uuid of the request. Defaults to None.
            server (Optional["Server"], optional): The server to pull the command from. Defaults to None.
        """
        super().__init__(uuid, server, **kwargs)

    @property
    def operation_head(self) -> Instruction | InstructionGroup | Stopper | None:
        """Returns the operation head, which is the first operation in the operations list.
        If the operation head is an instruction set, unpack it and add it to the operations list as separate instructions.

        Returns:
            Instruction | InstructionGroup | Stopper | None: The operation head. If there are no operations, returns None.
        """
        if self.server.operations and isinstance(
            self.server.operations[0], InstructionGroup
        ):
            # if the next operation is an instruction set, we need to unpack it and add it to the operations list as separate instructions
            self.server.operations = (
                self.server.operations[0].instructions + self.server.operations[1:]
            )
        return self.server.operations[0] if self.server.operations else None

    def execute(self) -> dict:
        """Executes the request and returns, if available, the command to be executed by the node.

        Returns:
            dict: The command to be executed by the node.
        """
        # If there are no operations, return an empty dictionary
        if not self.operation_head:
            return {}

        # If the operation head is a stopper, check if it is resolved. If it is, pop it from the operations list and continue
        while isinstance(self.operation_head, Stopper):
            # If the stopper is not resolved, return an empty dictionary
            if not self.operation_head.resolved(self.server):
                return {}
            # If the stopper is resolved, pop it from the operations list and continue
            else:
                self.server.history.append(self.server.operations.pop(0))

        # If the operation head is an instruction, check if it is completed. If it is, pop it from the operations list and continue
        if isinstance(self.operation_head, Instruction):
            if self.operation_head.instruction_status is InstructionStatus.COMPLETED:
                # If the instruction is completed, check if it has a successor. If it does, add it to the operations list
                successor = self.operation_head.successor
                self.server.history.append(self.server.operations.pop(0))
                if successor is not None and isinstance(successor, list):
                    self.server.operations = successor + self.server.operations

            # Infer the command from the instruction and return it as a dictionary

            if not self.operation_head:
                return {}
            return self.operation_head.infer_command(
                node_uuid=self.node_uuid,
                topology_register=self.server.topology_register,
                resource_register=self.server.resource_register,
                as_dict=True,
            )
