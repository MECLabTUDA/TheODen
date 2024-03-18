from ....common import Transferable
from ....resources import Model, Optimizer, Optimizer_, ResourceManager
from ....topology import Topology
from ....watcher import ParameterNotification
from . import SetResourceCommand


class SetOptimizerCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        optimizer: Optimizer_,
        key: str = "optimizer",
        model_key: str = "model",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the optimizer on the client

        Args:
            optimizer (Optimizer_): The optimizer to set
            key (str, optional): The resource key of the optimizer. Defaults to "optimizer".
            model_key (str, optional): The resource key of the model. Defaults to "model".
            overwrite (bool, optional): Whether to overwrite the existing optimizer. Defaults to True.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(
            key=key,
            resource=optimizer,
            overwrite=overwrite,
            uuid=uuid,
            **kwargs,
        )
        self.model_key = model_key
        self.assert_type = Optimizer

    def modify_resource(self, resource: Optimizer_) -> Optimizer:
        resource = resource.build([self.client_rm.gr(self.model_key, Model).module()])
        return resource

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        resource_manager.watcher.notify_all(
            notification=ParameterNotification(
                params={
                    "optimizer": self.resource.opti_class.__name__,
                    **self.resource.opti_args,
                }
            ),
        )
