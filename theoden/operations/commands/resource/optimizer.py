from ....common import Transferable
from ....resources import Model, Optimizer, Optimizer_, ResourceManager
from ....topology import Topology
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
        resource = resource.build([self.node_rm.gr(self.model_key, Model).module()])
        return resource

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        from ....watcher import ParameterNotification

        resource_manager.watcher.notify_all(
            notification=ParameterNotification(
                params={
                    "optimizer": self.resource.opti_class.__name__,
                    **self.resource.opti_args,
                }
            ),
        )
