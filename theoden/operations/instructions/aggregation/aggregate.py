import numpy as np
import torch

from abc import ABC, abstractmethod

from ....topology import Topology
from ....resources import ResourceManager
from ....resources.meta import CheckpointManager, DictCheckpoint
from ...commands import Command, SendModelToServerCommand, SendOptimizerToServerCommand
from ....common import Transferable
from .. import Action, Instruction
from ....watcher import AggregationCompletedNotification


class Aggregator(ABC, Transferable, is_base_type=True):
    def __init__(
        self,
        model_key: str | list[str] = "@all",
        optimizer_key: str | None = None,
        client_score: type | None = None,
        **kwargs,
    ):
        self.model_key = model_key
        self.optimizer_key = optimizer_key
        self.client_score = client_score

    def distribute_commands(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> list[Command]:
        """Returns a list of commands that are required to distribute the models.

        This method is called on the server side before the aggregation is started. All commands that load resources
        that should be updated after a communication round should be returned here.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            list[Command]: A list of commands that are required to distribute the models.
        """

        return resource_manager.checkpoint_manager.get_global_checkpoints_commands(
            of_resource_type=["model", "optimizer"] if self.optimizer_key else ["model"]
        )

    def required_resources_for_aggregation(self) -> list[Command]:
        """Returns a list of commands that are required to aggregate the models.

        This method is called on the server after the aggregation is finished. All resources that are required for the
        aggregation should be returned here.
        The server will then request these resources from the nodes in the next round. The resources will be added to
        the resource register of the instruction.
        Default implementation is to request the model from the nodes and also the optimizer if one is specified. This
        method should be overwritten if other resources are required.

        Returns:
            list[Command]: A list of commands that are required to aggregate the models.
        """
        return [SendModelToServerCommand(resource_key=self.model_key)] + (
            []
            if not self.optimizer_key
            else [SendOptimizerToServerCommand(resource_key=self.optimizer_key)]
        )

    def _calculate_pseudo_gradients(
        self, resources: dict[str, DictCheckpoint], global_model: dict[str, any]
    ) -> dict[str, dict[str, any]]:
        # create pseudo gradient dict
        pseudo_gradients = {}

        # for each local model
        for resource_key, resource in resources.items():
            # pseudo gradient dict
            pseudo_gradients[resource_key] = {}

            # for each local model's parameter
            for parameter_key, parameter in resource.to(dict).items():
                # get global model's parameter
                global_parameter = global_model[parameter_key]

                # calculate pseudo gradient
                pseudo_gradients[resource_key][parameter_key] = (
                    parameter - global_parameter
                )

        # return pseudo gradient
        return pseudo_gradients

    def _recursive_averaging(self, state_dicts, weights):
        if isinstance(state_dicts[0], dict):
            result_dict = {}
            for key in state_dicts[0]:
                nested_state_dicts = [state_dict[key] for state_dict in state_dicts]
                result_dict[key] = self._recursive_averaging(
                    nested_state_dicts, weights
                )
            return result_dict

        elif isinstance(state_dicts[0], torch.Tensor | np.ndarray):
            total_weight = sum(weights)
            total_value = sum(
                weight * value for weight, value in zip(weights, state_dicts)
            )
            return total_value / total_weight

        elif state_dicts[0] is None:
            return None
        else:
            raise TypeError(
                f"Invalid data type. Only nested dictionaries or torch tensors are supported, got {type(state_dicts[0])}"
            )

    def _get_weights(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> dict[str, float]:
        # get weights from topology register
        if self.client_score is None:
            scores = {node.name: 1 for node in topology.online_clients()}
        else:
            try:
                scores = {
                    node.name: node.data[self.client_score.__name__]
                    for node in topology.online_clients(False)
                }
            except KeyError:
                raise KeyError(
                    f"Client score {self.client_score.__name__} not found in topology register. "
                    f"Please check that the score is requested."
                )

        # normalize weights
        total_score = sum(scores.values())
        weights = {
            node_name: score / total_score for node_name, score in scores.items()
        }
        return weights

    @abstractmethod
    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> any:
        raise NotImplementedError(
            "Please implement this method for your aggregation method."
        )


class AggregationAction(Action, Transferable):
    def __init__(
        self,
        aggregator: Aggregator,
        _no_aggregation: bool = False,
        predecessor: Instruction | None = None,
        remove_instruction_resource_entry: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(predecessor, remove_instruction_resource_entry, **kwargs)
        self.aggregator = aggregator
        self._no_aggregation = _no_aggregation

    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        """Aggregates the models and optimizers on the server.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            Instruction | None: The instruction that should be executed after the aggregation.
        """

        # if no aggregation is required, return None. This is mainly the case when training on a single node
        if self._no_aggregation:
            return None

        # use the client checkpoints to aggregate the models / optimizers
        client_checkpoints = resource_manager.client_checkpoints

        for resource_type, resources in client_checkpoints.items():
            for resource_key, resource in resources.items():
                # get global model from checkpoint manager

                cm = resource_manager.checkpoint_manager

                # aggregate resource_manager using the aggregators aggregate method
                aggregated = self.aggregator.aggregate(
                    resource_type=resource_type,
                    resource_key=resource_key,
                    resources=resource,
                    topology=topology,
                    resource_manager=resource_manager,
                )

                # save aggregated resource_manager in checkpoint manager
                cm.register_checkpoint(
                    resource_type=resource_type,
                    resource_key=resource_key,
                    checkpoint_key="__global__",
                    checkpoint=DictCheckpoint(aggregated),
                )

        # reset the client checkpoints
        client_checkpoints.reset()

        # inform the watcher that the aggregation is completed
        resource_manager.watcher.notify_all(AggregationCompletedNotification())
