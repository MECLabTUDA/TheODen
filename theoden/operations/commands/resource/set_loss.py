from ....common import Transferable
from ....resources import Loss, ResourceManager
from ....topology import Topology
from . import SetResourceCommand


class SetLossesCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        losses: list[Loss],
        key: str = "losses",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            resource=losses,
            overwrite=overwrite,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = list[Loss]

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        resource_manager.sr(key=self.key, resource=self.resource)
