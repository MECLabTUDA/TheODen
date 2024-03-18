from ....common import Transferable
from ....resources.training import LRScheduler, Optimizer, Scheduler
from . import SetResourceCommand


class SetLRSchedulerCommand(SetResourceCommand, Transferable):
    """Set the learning rate scheduler on the client"""

    def __init__(
        self,
        scheduler: Scheduler,
        key: str = "scheduler",
        optimizer_key: str = "optimizer",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Set the learning rate scheduler on the client

        Args:
            scheduler (Scheduler): The scheduler to set
            key (str, optional): The resource key of the scheduler. Defaults to "scheduler".
            optimizer_key (str, optional): The resource key of the optimizer. Defaults to "optimizer".
            overwrite (bool, optional): Whether to overwrite the existing scheduler. Defaults to True.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """

        super().__init__(
            key=key, resource=scheduler, overwrite=overwrite, uuid=uuid, **kwargs
        )
        self.optimizer_key = optimizer_key
        self.assert_type = LRScheduler

    def modify_resource(self, resource: Scheduler) -> LRScheduler:
        """Build the scheduler

        Args:
            resource (Scheduler): The scheduler to build (TheODen resource)

        Returns:
            LRScheduler: The built scheduler (torch.optim.lr_scheduler._LRScheduler)
        """
        resource = resource.build(
            self.client_rm.gr(self.optimizer_key, assert_type=Optimizer)
        )
        return resource
