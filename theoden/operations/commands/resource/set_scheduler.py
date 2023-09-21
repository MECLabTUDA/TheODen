from typing import Optional

from theoden.operations.commands.resource import SetResourceCommand
from theoden.common import Transferable
from theoden.resources.training import LRScheduler, Scheduler, Optimizer


class SetLRSchedulerCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        scheduler: Scheduler,
        key: str = "scheduler",
        optimizer_key: str = "optimizer",
        overwrite: bool = True,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            resource=scheduler,
            overwrite=overwrite,
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.optimizer_key = optimizer_key
        self.assert_type = LRScheduler

    def modify_resource(self, resource: Scheduler) -> LRScheduler:
        resource = resource.build(
            self.node_rr.gr(self.optimizer_key, assert_type=Optimizer)
        )
        return resource
