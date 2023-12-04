from __future__ import annotations

from ..instruction import Instruction
from ...condition import Condition
from ....common import Transferable
from ..action import Action
from ..distribution import Distribution


class InstructionBundle(Transferable, is_base_type=True):
    def __init__(
        self, instructions: list[Distribution | Action | Condition | InstructionBundle]
    ) -> None:
        assert len(instructions) > 0
        assert all(
            [
                isinstance(
                    instruction, InstructionBundle | Condition | Action | Distribution
                )
                for instruction in instructions
            ]
        ), "All instructions in an instruction set must be of type Distribution, Action, Condition or InstructionBundle."
        self.instructions = instructions

    def __add__(
        self, other: Instruction | InstructionBundle | Condition
    ) -> InstructionBundle:
        """Overload the + operator to create an InstructionBundle

        Args:
            other (Instruction | InstructionBundle | Condition): The other Instruction or InstructionBundle

        Returns:
            InstructionBundle: The InstructionBundle containing the two instructions
        """
        return InstructionBundle([*self.instructions, other])
