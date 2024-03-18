import torch

from ....common import AggregationError, Transferable
from ....common.utils import create_sorted_lists
from ....resources.meta import DictCheckpoint
from ....resources.resource import ResourceManager
from ....topology.topology import Topology
from .aggregate import Aggregator


class ServerOptimizer(Transferable, is_base_type=True):
    def __init__(self):
        """A server optimizer."""
        self.momentum: dict = {}

    def _calculate_momentum(
        self, averaged_pseudo_gradient: dict[str, torch.Tensor], beta: float
    ) -> dict[str, torch.Tensor]:
        """Calculate the momentum term and update the momentum dictionary.

        Args:
            averaged_pseudo_gradient (dict[str, torch.Tensor]): The averaged pseudo gradient.
            beta (float): The beta parameter.

        Returns:
            dict[str, torch.Tensor]: The momentum term.
        """

        # iterate over all weights
        for weight in averaged_pseudo_gradient:
            # calculate new momentum term
            if weight not in self.momentum:
                self.momentum[weight] = torch.zeros_like(
                    averaged_pseudo_gradient[weight]
                )
            self.momentum[weight] = (
                beta * self.momentum[weight]
                + (1 - beta) * averaged_pseudo_gradient[weight]
            )
        return self.momentum

    def step(
        self,
        averaged_pseudo_gradient: dict[str, torch.Tensor],
        global_model: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Perform a step of the server optimizer.

        Args:
            averaged_pseudo_gradient (dict[str, torch.Tensor]): The averaged pseudo gradient of the last round.
            global_model (dict[str, torch.Tensor]): The current global model dict.

        Returns:
            dict[str, torch.Tensor]: The new global model dict.
        """
        raise NotImplementedError(
            "The step method of a ServerOptimizer must be implemented."
        )


class AdaptiveFedOptServerOptimizer(ServerOptimizer):
    def __init__(self, beta: float = 0.9, lr: float = 0.1):
        """An adaptive server optimizer.

        Args:
            beta (float, optional): The momentum beta parameter. Defaults to 0.9.
            lr (float, optional): The server learning rate. Defaults to 0.1.
        """
        super().__init__()
        self.velocity: dict = {}
        self.beta = beta
        self.lr = lr

    def update_velocity(
        self, old_velocity: torch.Tensor, averaged_pseudo_gradient: torch.Tensor
    ) -> torch.Tensor:
        """Update the velocity term for a specific weight.

        Args:
            old_velocity (torch.Tensor): The old velocity term.
            averaged_pseudo_gradient (torch.Tensor): The averaged pseudo gradient.

        Returns:
            torch.Tensor: The new velocity term.
        """
        raise NotImplementedError(
            "The velocity method of a AdaptiveFedOptServerOptimizer must be implemented."
        )

    def _calculate_velocity(
        self, averaged_pseudo_gradient: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Calculate the velocity term and update the velocity dictionary.

        Args:
            averaged_pseudo_gradient (dict[str, torch.Tensor]): The averaged pseudo gradient.

        Returns:
            dict[str, torch.Tensor]: The velocity term.
        """

        # iterate over all weights
        for weight in averaged_pseudo_gradient:
            if weight not in self.velocity:
                self.velocity[weight] = torch.zeros_like(
                    averaged_pseudo_gradient[weight]
                )
            self.velocity[weight] = self.update_velocity(
                self.velocity[weight], averaged_pseudo_gradient[weight]
            )

        return self.velocity

    def step(
        self,
        averaged_pseudo_gradient: dict[str, torch.Tensor],
        global_model: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        momentum = self._calculate_momentum(averaged_pseudo_gradient, self.beta)

        velocity = self._calculate_velocity(averaged_pseudo_gradient)

        new_global_model = {}
        for weight in averaged_pseudo_gradient:
            new_global_model[weight] = global_model[weight] + self.lr * momentum[
                weight
            ] / (torch.sqrt(velocity[weight]) + self.tau)

        return new_global_model


class FedAdamServerOptimizer(AdaptiveFedOptServerOptimizer):
    def __init__(
        self,
        beta: float = 0.9,
        beta_2: float = 0.99,
        lr: float = 0.1,
        tau: float = 1e-9,
    ):
        """FedAdam Server Optimizer.

        Args:
            beta (float, optional): The momentum beta parameter. Defaults to 0.9.
            beta_2 (float, optional): The velocity beta_2 parameter. Defaults to 0.99.
            lr (float, optional): The server learning rate. Defaults to 0.1.
            tau (float, optional): The adaptability tau parameter. Defaults to 1e-9.
        """
        super().__init__(beta, lr)
        self.beta_2 = beta_2
        self.tau = tau

    def update_velocity(
        self, old_velocity: torch.Tensor, averaged_pseudo_gradient: torch.Tensor
    ) -> torch.Tensor:
        # vt = β2 * vt−1 + (1 − β2) * ∆^2
        return self.beta_2 * old_velocity + (1 - self.beta_2) * torch.multiply(
            averaged_pseudo_gradient, averaged_pseudo_gradient
        )


class FedAdagradServerOptimizer(AdaptiveFedOptServerOptimizer):
    """FedAdagrad Server Optimizer."""

    def update_velocity(
        self, old_velocity: torch.Tensor, averaged_pseudo_gradient: torch.Tensor
    ) -> torch.Tensor:
        # vt = vt−1 + ∆^2
        return old_velocity + torch.multiply(
            averaged_pseudo_gradient, averaged_pseudo_gradient
        )


class FedYogiServerOptimizer(AdaptiveFedOptServerOptimizer):
    def __init__(self, beta: float = 0.9, beta_2: float = 0.99, lr: float = 0.1):
        """FedYogi Server Optimizer.

        Args:
            beta (float, optional): The beta parameter. Defaults to 0.9.
            beta_2 (float, optional): The beta_2 parameter. Defaults to 0.99.
            lr (float, optional): The learning rate. Defaults to 0.1.
        """
        super().__init__(beta=beta, lr=lr)
        self.beta_2 = beta_2

    def update_velocity(
        self, old_velocity: torch.Tensor, averaged_pseudo_gradient: torch.Tensor
    ) -> torch.Tensor:
        # vt = vt−1 − (1 − β2) * ∆^2 * sign(vt−1 − ∆^2)
        delta2 = torch.multiply(averaged_pseudo_gradient, averaged_pseudo_gradient)
        return old_velocity - (1 - self.beta_2) * torch.multiply(delta2) * torch.sign(
            old_velocity - torch.multiply(delta2)
        )


class FedSGDServerOptimizer(ServerOptimizer, Transferable):
    def __init__(self, lr: float = 0.1, beta: float = 0.9) -> None:
        super().__init__()
        self.beta = beta
        self.lr = lr

    def step(
        self,
        averaged_pseudo_gradient: dict[str, torch.Tensor],
        global_model: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        momentum = self._calculate_momentum(averaged_pseudo_gradient, self.beta)

        new_global_model = {}

        for weight in averaged_pseudo_gradient:
            new_global_model[weight] = global_model[weight] + self.lr * momentum[weight]

        return new_global_model


class FedOptAggregator(Aggregator, Transferable):
    def __init__(
        self, server_optimizer: ServerOptimizer, client_score: type | None = None
    ):
        """FedOpt Aggregator.

        Args:
            server_optimizer (ServerOptimizer): The server optimizer to use.
            client_score (type | None, optional): The client score to use for weighting. Defaults to None.
        """
        super().__init__(client_score=client_score)
        self.server_optimizer = server_optimizer

    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> dict[str, torch.Tensor]:
        global_model = self._get_global_model(
            resource_type=resource_type,
            resource_key=resource_key,
            resource_manager=resource_manager,
        )

        # get pseudo gradient
        pseudo_gradients = self._calculate_pseudo_gradients(resources, global_model)

        dict_of_weights = self._get_weights(
            topology=topology, resource_manager=resource_manager
        )

        print(dict_of_weights)

        # average pseudo gradient
        averaged_pseudo_gradient = self._recursive_averaging(
            *create_sorted_lists(pseudo_gradients, dict_of_weights)
        )

        # apply server optimizer
        new_model = self.server_optimizer.step(
            averaged_pseudo_gradient=averaged_pseudo_gradient, global_model=global_model
        )

        # return new model
        return new_model


class FedAvgAggregator(FedOptAggregator, Transferable):
    def __init__(self, lr: float = 1.0, client_score: type | None = None):
        """FedAvg Aggregator.

        Args:
            lr (float, optional): Learning rate. Defaults to 1.0.
            client_score (type | None, optional): The client score to use for weighting. Defaults to None.
        """
        super().__init__(
            FedSGDServerOptimizer(lr=lr, beta=0), client_score=client_score
        )
