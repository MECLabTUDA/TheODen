from __future__ import annotations

from .instruction import Instruction
from ..stopper import Stopper
from ...common import Transferable


class InstructionGroup(Transferable, is_base_type=True):
    def __init__(
        self, instructions: list[Instruction | InstructionGroup | Stopper]
    ) -> None:
        assert len(instructions) > 0
        assert all(
            [
                isinstance(instruction, Instruction)
                or isinstance(instruction, InstructionGroup)
                or isinstance(instruction, Stopper)
                for instruction in instructions
            ]
        ), "All instructions in an instruction set must be of type Instruction, InstructionGroup or Stopper."
        self.instructions = instructions
