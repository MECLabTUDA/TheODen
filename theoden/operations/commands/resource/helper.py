import json
import logging
import os
import signal

from theoden.resources.resource import ResourceManager
from theoden.topology.topology import Topology

from ....common import ExecutionResponse, Transferable
from .. import Command


class PrintResourceKeysCommand(Command, Transferable):
    """Print all resource keys on the client"""

    def execute(self) -> ExecutionResponse | None:
        logging.info("Resource keys:")
        logging.info(json.dumps(self.client_rm.get_key_type_dict(), indent=3))
        return None


class ClearResourcesCommand(Command, Transferable):
    """Clear all resources on the client"""

    def execute(self) -> ExecutionResponse | None:
        self.client_rm.clear()
        return None


class ExitRunCommand(Command, Transferable):
    """Exit the client and server"""

    def execute(self) -> ExecutionResponse | None:
        self.client.stop()

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        os.kill(os.getpid(), signal.SIGINT)
