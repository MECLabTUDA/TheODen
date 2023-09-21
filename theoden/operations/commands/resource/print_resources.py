from typing import Any, Optional
import time

from .. import Command
from ....common import ExecutionResponse, Transferable


class PrintResourcesCommand(Command, Transferable):
    def execute(self) -> ExecutionResponse | None:
        print(self.node_rr)
        return None
