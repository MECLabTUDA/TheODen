import torch

from .aggregate import Aggregator
from ....common import Transferable, AggregationError
from ....topology.topology import Topology
from ....resources.resource import ResourceManager
from ....resources.meta import DictCheckpoint
from ....common.utils import create_sorted_lists


class ServerOptimizer(Transferable, is_base_type=True):
    def __init__(self):
        self.momentum = None

    def _calculate_momentum(
        self,
        averaged_pseudo_gradient: any,
        beta: float,
        resource_manager: ResourceManager,
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
        resource_manager: ResourceManager,
    ):
        raise NotImplementedError(
            "The step method of a ServerOptimizer must be implemented."
        )


class FedAdamServerOptimizer(ServerOptimizer, Transferable):
    def __init__(self, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999)):
        super().__init__()
        self.lr = lr
        self.betas = betas

    def step(
        self,
        averaged_pseudo_gradient: any,
        global_model: any,
        resource_manager: ResourceManager,
    ) -> any:
        # calculate momentum
        momentum = self._calculate_momentum(
            averaged_pseudo_gradient, self.betas[0], resource_manager
        )

        # TODO: calculate second momentum

        return averaged_pseudo_gradient


class FedSGDServerOptimizer(ServerOptimizer, Transferable):
    def __init__(self, lr: float = 1.0, beta: float = 0.9) -> None:
        super().__init__()
        self.beta = beta
        self.lr = lr

    def step(
        self,
        averaged_pseudo_gradient: any,
        global_model: any,
        resource_manager: ResourceManager,
    ):
        # calculate momentum
        momentum = self._calculate_momentum(
            averaged_pseudo_gradient, self.beta, resource_manager
        )

        new_global_model = {}

        for weight in averaged_pseudo_gradient:
            new_global_model[weight] = global_model[weight] + self.lr * momentum[weight]

        return new_global_model


class FedOptAggregator(Aggregator, Transferable):
    def __init__(
        self,
        server_optimizer: ServerOptimizer | None = None,
        client_score: type | None = None,
    ):
        super().__init__(client_score=client_score)
        self.server_optimizer = server_optimizer

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> any:
        cm = resource_manager.checkpoint_manager

        try:
            global_model = cm.get_checkpoint(
                resource_type=resource_type,
                resource_key=resource_key,
                checkpoint_key="__global__",
            ).to(dict)
        except KeyError:
            raise AggregationError(
                f"Could not find global resource for resource type `{resource_type}` and resource key `{resource_key}`. Make sure to initialize the global resource before starting the aggregation."
            )

        # get pseudo gradient
        pseudo_gradients = self._calculate_pseudo_gradients(resources, global_model)

        dict_of_weights = self._get_weights(
            topology=topology, resource_manager=resource_manager
        )

        # average pseudo gradient
        averaged_pseudo_gradient = self._recursive_averaging(
            *create_sorted_lists(pseudo_gradients, dict_of_weights)
        )

        # apply server optimizer
        new_model = self.server_optimizer.step(
            averaged_pseudo_gradient=averaged_pseudo_gradient,
            global_model=global_model,
            resource_manager=resource_manager,
        )

        # return new model
        return new_model


class FedAvgAggregator(FedSGDServerOptimizer, Transferable):
    def __init__(self, lr: float = 1.0):
        super().__init__(lr=lr, beta=0)
