import json

from .. import Command
from ....common import ExecutionResponse, Transferable
from ....resources import ResourceManager


class PrintResourceKeysCommand(Command, Transferable):
    """Print all resource keys on the node"""

    def execute(self) -> ExecutionResponse | None:
        print("Resource keys:")
        print(json.dumps(self.node_rm.get_key_type_dict(), indent=3))
        return None


class ClearResourcesCommand(Command, Transferable):
    """Clear all resources on the node"""

    def execute(self) -> ExecutionResponse | None:
        self.node_rm.clear()
        return None
