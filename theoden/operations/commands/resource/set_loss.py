from typing import Optional

from theoden.operations.commands.resource import SetResourceCommand
from theoden.common import ExecutionResponse, Transferable
from theoden.resources import Loss, ResourceRegister
from theoden.topology import TopologyRegister


class SetLossesCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        losses: list[Loss],
        key: str = "losses",
        overwrite: bool = True,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            resource=losses,
            overwrite=overwrite,
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = list[Loss]

    def on_client_finish_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        node_uuid: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        resource_register.sr(key=self.key, resource=self.resource)
