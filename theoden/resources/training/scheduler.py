import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
import numpy as np

from ...common import Transferable


# Get all scheduler classe
# Transferable.make_transferable(lr_scheduler.LRScheduler, is_base_type=True)

attrs = dir(lr_scheduler)

# Filter out the non-transforms attributes
transform_attrs = [attr for attr in attrs if not attr.startswith("_")]

# Iterate over all the transform attributes and print their names
for attr in transform_attrs:
    # Get the transform class by name
    scheduler_class = getattr(lr_scheduler, attr)

    if type(scheduler_class) is type and issubclass(
        scheduler_class, lr_scheduler.LRScheduler
    ):
        Transferable.make_transferable(
            scheduler_class, base_type=lr_scheduler.LRScheduler
        )


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class Scheduler(Transferable, is_base_type=True):
    def __init__(self, scheduler: type[lr_scheduler._LRScheduler], **kwargs) -> None:
        """Create a scheduler wrapper.

        Args:
            scheduler (lr_scheduler._LRScheduler): scheduler class
            **kwargs: other arguments for the scheduler
        """

        self.scheduler = scheduler
        self.kwargs = kwargs

    def build(self, optimizer) -> lr_scheduler._LRScheduler:
        """Build the scheduler.

        Args:
            optimizer (Optimizer): the optimizer to be scheduled

        Returns:
            lr_scheduler._LRScheduler: the built scheduler
        """

        return self.scheduler(optimizer, **self.kwargs)


class MultiStepLRScheduler(Scheduler, Transferable):
    def __init__(self, milestones: list[int], **kwargs) -> None:
        """Create a MultiStepLR scheduler wrapper.

        Args:
            milestones (list[int]): list of epoch indices. Must be increasing.
            **kwargs: other arguments for MultiStepLR
        """

        super().__init__(lr_scheduler.MultiStepLR, milestones=milestones, **kwargs)


class CosineAnnealingLRScheduler(Scheduler, Transferable):
    def __init__(self, num_epochs: int, **kwargs) -> None:
        """Create a CosineAnnealingLR scheduler wrapper.

        Args:
            num_epochs (int): number of epochs
            **kwargs: other arguments for CosineAnnealingLR
        """

        super().__init__(LambdaLR, **kwargs)
        self.num_epochs = num_epochs

    def build(self, optimizer) -> lr_scheduler._LRScheduler:
        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step, self.num_epochs, 1, 1e-6 / 0.1
            ),
        )
