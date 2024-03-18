from ....common import Transferable
from ....resources import Loss, ResourceManager
from ....topology import Topology
from . import SetResourceCommand


class SetLossesCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        losses: list[Loss] | Loss,
        key: str = "losses",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the losses on the client

        Args:
            losses (list[Loss] | Loss): The loss(es) to set
            key (str, optional): The resource key of the losses. Defaults to "losses".
            overwrite (bool, optional): Whether to overwrite the existing losses. Defaults to True.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(
            key=key,
            resource=losses if isinstance(losses, list) else [losses],
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
        # set the losses on the server side
        resource_manager.sr(key=self.key, resource=self.resource)
