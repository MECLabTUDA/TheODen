import torch

import io

from .aggregator import Aggregator
from ....common import Transferable
from ....topology.topology_register import TopologyRegister
from ....resources.resource import ResourceRegister
from ....resources.meta import DictCheckpoint
from ....common.utils import create_sorted_lists


class ServerOptimizer(Transferable, is_base_type=True):
    def __init__(self):
        self.momentum = None

    def _calculate_momentum(
        self,
        averaged_pseudo_gradient: any,
        beta: float,
        resource_register: ResourceRegister,
    ) -> any:
        new_momentum = {}

        # iterate over all weights
        for i, weight in enumerate(averaged_pseudo_gradient):
            # calculate new momentum term
            if self.momentum is None:
                new_momentum[weight] = (1 - beta) * averaged_pseudo_gradient[weight]
            else:
                new_momentum[weight] = (
                    beta * self.momentum[weight]
                    + (1 - beta) * averaged_pseudo_gradient[weight]
                )

        # set new momentum term
        self.momentum = new_momentum

        return self.momentum

    def step(
        self,
        averaged_pseudo_gradient: any,
        global_model: any,
        resource_register: ResourceRegister,
    ):
        ...


class FedAdamServerOptimizer(ServerOptimizer, Transferable):
    def __init__(self, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999)):
        super().__init__()
        self.lr = lr
        self.betas = betas

    def step(
        self,
        averaged_pseudo_gradient: any,
        global_model: any,
        resource_register: ResourceRegister,
    ) -> any:
        # calculate momentum
        momentum = self._calculate_momentum(
            averaged_pseudo_gradient, self.betas[0], resource_register
        )

        return averaged_pseudo_gradient


def calculate_min_max_mean_of_state_dict(
    state_dict: dict[str, any]
) -> tuple[float, float, float]:
    vals = []
    for key in state_dict:
        vals.append(state_dict[key].flatten())
    test = torch.cat(vals)
    return torch.min(test).item(), torch.max(test).item(), torch.mean(test).item()


class FedSGDServerOptimizer(ServerOptimizer, Transferable):
    def __init__(self, lr: float = 1.0, beta: float = 0.9) -> None:
        super().__init__()
        self.beta = beta
        self.lr = lr

    def step(
        self,
        averaged_pseudo_gradient: any,
        global_model: any,
        resource_register: ResourceRegister,
    ):
        # calculate momentum
        momentum = self._calculate_momentum(
            averaged_pseudo_gradient, self.beta, resource_register
        )

        new_global_model = {}

        for weight in averaged_pseudo_gradient:
            new_global_model[weight] = (
                global_model[weight].cpu() + self.lr * momentum[weight]
            )

        return new_global_model


class FedOptAggregator(Aggregator, Transferable):
    def __init__(
        self,
        server_optimizer: ServerOptimizer | None = None,
        model_key: str | list[str] = "@all",
        optimizer_key: str | None = None,
        client_score: type | None = None,
    ):
        super().__init__(model_key, optimizer_key, client_score=client_score)
        self.server_optimizer = server_optimizer

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> any:
        cm = self._get_checkpoint_manager(resource_register=resource_register)

        global_model = cm.get_checkpoint(
            resource_type=resource_type,
            resource_key=resource_key,
            checkpoint_key="__global__",
        ).to(dict)

        # get pseudo gradient
        pseudo_gradients = self._calculate_pseudo_gradients(resources, global_model)

        dict_of_weights = self._get_weights(
            topology_register=topology_register, resource_register=resource_register
        )

        # average pseudo gradient
        averaged_pseudo_gradient = self._recursive_averaging(
            *create_sorted_lists(pseudo_gradients, dict_of_weights)
        )

        # apply server optimizer
        new_model = self.server_optimizer.step(
            averaged_pseudo_gradient=averaged_pseudo_gradient,
            global_model=global_model,
            resource_register=resource_register,
        )

        # return new model
        return new_model
