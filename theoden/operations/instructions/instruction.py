from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING
from uuid import uuid4

from ...common import Transferable
from ...resources import ResourceManager
from ...resources.resource import is_instance_of_type_hint
from ...topology import Topology

if TYPE_CHECKING:
    from .bundles import InstructionBundle


class InstructionStatus(IntEnum):
    # Enum values for Instruction status
    CREATED = auto()  # The Instruction object has been created
    BOOTING = auto()  # The Instruction object is initializing
    EXECUTION = auto()  # The Instruction object is executing
    EXECUTION_FINISHED = auto()  # The Instruction object has finished executing
    COMPLETED = auto()  # The Instruction object has completed its execution cycle


class Instruction(Transferable, is_base_type=True):
    def __init__(
        self,
        predecessor: Instruction | None = None,
        remove_instruction_resource_entry: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.successor: list[Instruction] = []
        self.predecessor: Instruction | None = predecessor
        self.remove_instruction_resource_entry = remove_instruction_resource_entry
        self.status = InstructionStatus.CREATED

        self.uuid: str | None = None
        self.on_finish_hooks: list[callable] = []

    def _set_uuid(self):
        self.uuid = str(uuid4())

    def on_finish(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | list[Instruction] | None:
        return

    def register_on_finish_hook(self, hook: callable) -> Instruction:
        """Register a hook to be called when the instruction finishes

        This may be necessary if the successor instruction requires data from the current instruction

        Args:
            hook (callable): The hook to be called

        Returns:
            Instruction: The instruction itself
        """
        self.on_finish_hooks.append(hook)
        return self

    def _on_finish(self, topology: Topology, resource_manager: ResourceManager):
        """Internal method for finishing the instruction.

        This method is called when all commands have a status of FINISHED or EXCLUDED. It calls the on_finish method of the subclass and sets the instruction status to COMPLETED.
        Also, the successor s are determined by calling the on_finish method of the subclass and the on_finish_hooks. They are then added to the successor list of the instruction.
        Finally, the instruction status is set to COMPLETED and the resource_manager are freed up.

        Args:
            topology (Topology): The topology register of the server
            resource_manager (ResourceManager): The resource register of the server
        """

        # Set the instruction status to EXECUTION_FINISHED
        self.status = InstructionStatus.EXECUTION_FINISHED

        # Call the on_finish method of the subclass
        successor = self.on_finish(topology, resource_manager)

        # if successor is not a list, convert it to a list
        if successor is not None and (
            isinstance(successor, Instruction)
            or is_instance_of_type_hint(successor, list[Instruction])
        ):
            if not isinstance(successor, list):
                assert isinstance(
                    successor, Instruction
                ), "successor must be an Instruction, not {}".format(
                    type(successor).__name__
                )
                successor = [successor]
            self.successor = self.successor + successor

        # Call the on_finish_hooks and add the successors to the successor list
        for hook in self.on_finish_hooks:
            successor = hook(self, topology, resource_manager)

            if successor is not None:
                if not isinstance(successor, list):
                    assert isinstance(
                        successor, Instruction
                    ), "successor must be an Instruction, not {}".format(
                        type(successor).__name__
                    )
                    successor = [successor]
                self.successor = self.successor + successor

        # If the instruction status is still EXECUTION_FINISHED, set it to COMPLETED
        if self.status == InstructionStatus.EXECUTION_FINISHED:
            self.status = InstructionStatus.COMPLETED

        # Free up resource_manager after the instruction has finished to save memory
        if self.remove_instruction_resource_entry and self.uuid in resource_manager:
            resource_manager.rm(self.uuid, assert_type=ResourceManager)
            if self.predecessor is not None:
                resource_manager.rm(self.predecessor.uuid, assert_type=ResourceManager)

    def __add__(self, other: Instruction | "InstructionBundle") -> "InstructionBundle":
        """Overload the + operator to create an InstructionBundle

        Args:
            other (Instruction | InstructionBundle): The other Instruction or InstructionBundle

        Returns:
            InstructionBundle: The InstructionBundle containing the two instructions
        """
        from .bundles import InstructionBundle

        return InstructionBundle([self, other])
