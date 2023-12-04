from .instruction import Instruction, InstructionStatus
from ...common import Transferable
from ...resources import ResourceManager
from ...topology import Topology


class Action(Instruction, Transferable, is_base_type=True):
    def __init__(
        self,
        predecessor: Instruction | None = None,
        remove_instruction_resource_entry: bool = True,
        **kwargs
    ) -> None:
        """An action that can be executed by the instruction manager.

        Args:
            remove_instruction_resource_entry (bool, optional): Whether to remove the resource entry for the instruction after the action is finished. Defaults to True.
        """

        super().__init__(
            predecessor=predecessor,
            remove_instruction_resource_entry=remove_instruction_resource_entry,
            **kwargs
        )
        self._set_uuid()

    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        """Performs the action. This method is called when the action is executed and should be implemented by the user.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.
        """
        print(type(self))
        raise NotImplementedError(
            "The perform method of an Action must be implemented."
        )

    def __call__(self, topology: Topology, resource_manager: ResourceManager) -> None:
        """Performs the action. This is the method that is called by the instruction manager.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.
        """
        self.status = InstructionStatus.EXECUTION
        successor = self.perform(topology, resource_manager)
        if successor is not None:
            self.successor.append(successor)
        self._on_finish(topology, resource_manager)
