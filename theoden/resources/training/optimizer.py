from typing import List, Optional, Type

import torch
from torch.optim import SGD, Adam, Optimizer

from ...common import Transferable

attrs = dir(torch.optim)

# Filter out the non-transforms attributes
transform_attrs = [attr for attr in attrs if not attr.startswith("_")]

# Iterate over all the transform attributes and print their names
for attr in transform_attrs:
    # Get the transform class by name
    scheduler_class = getattr(torch.optim, attr)

    if type(scheduler_class) is type and issubclass(scheduler_class, Optimizer):
        Transferable.make_transferable(scheduler_class, base_type=Optimizer)


class Optimizer_(Transferable, is_base_type=True):
    def __init__(
        self,
        opti_class: Type[Optimizer],
        exclude_layers: Optional[List[str]] = None,
        include_layers: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Create an optimizer wrapper.

        Args:
            opti_class (Type[Optimizer]): optimizer class
            exclude_layers (Optional[List[str]], optional): layers to exclude. Defaults to None.
            include_layers (Optional[List[str]], optional): layers to include. Defaults to None.

        Raises:
            AssertionError: if both include and exclude layers are specified
        """

        self.opti_class = opti_class
        self.opti_args = kwargs
        self.exclude_layers = exclude_layers
        self.include_layers = include_layers
        assert not (
            self.include_layers != None and self.exclude_layers != None
        ), "Either specifically include or exclude layers, not both"

    def build(self, modules: List[torch.nn.Module]) -> Optimizer:
        """Build the optimizer.

        Args:
            modules (List[torch.nn.Module]): list of modules to be included in the optimizer

        Returns:
            Optimizer: the built optimizer
        """

        parameters = []
        for module in modules:
            for name, parameter in module.named_parameters():
                if self.exclude_layers != None:
                    if name not in self.exclude_layers:
                        parameters.append(parameter)
                elif self.include_layers != None:
                    if name in self.include_layers:
                        parameters.append(parameter)
                else:
                    parameters.append(parameter)

        return self.opti_class(
            parameters,
            **self.opti_args,
        )


class AdamOptimizer(Optimizer_, Transferable):
    def __init__(
        self,
        lr: float,
        **kwargs,
    ) -> None:
        """Create an Adam optimizer wrapper.

        Args:
            lr (float): learning rate
            **kwargs: other arguments for Adam
        """
        super().__init__(
            Adam,
            lr=lr,
            **kwargs,
        )


class SGDOptimizer(Optimizer_, Transferable):
    def __init__(
        self,
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        **kwargs,
    ) -> None:
        """Create an SGD optimizer wrapper.

        Args:
            lr (float): learning rate
            momentum (float, optional): momentum. Defaults to 0.
            weight_decay (float, optional): weight decay. Defaults to 0.
            nesterov (bool, optional): nesterov momentum. Defaults to False.
            **kwargs: other arguments for SGD
        """

        super().__init__(
            SGD,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs,
        )
