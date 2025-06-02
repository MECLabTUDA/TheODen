from abc import ABC, abstractmethod

import numpy as np
import torch

from ....common import AggregationError, Transferable
from ....resources import ResourceManager
from ....resources.meta import DictCheckpoint
from ....topology import Topology
from ....watcher import AggregationCompletedNotification
from .. import Action, Instruction

import logging
logger = logging.getLogger(__name__)

class Aggregator(ABC, Transferable, is_base_type=True):
    def __init__(self, client_score: type | None = None, **kwargs):
        self.client_score = client_score

    def _get_global_model(
        self, resource_type: str, resource_key: str, resource_manager: ResourceManager
    ) -> dict[str, any]:
        """Get the global model from the checkpoint manager.

        Args:
            resource_type (str): The type of the resource.
            resource_key (str): The key of the resource.
            resource_manager (ResourceManager): The resource manager.

        Returns:
            dict[str, any]: The global model.
        """
        try:
            return resource_manager.checkpoint_manager.get_checkpoint(
                resource_type=resource_type,
                resource_key=resource_key,
                checkpoint_key="__global__",
            ).to(dict)
        except KeyError:
            raise AggregationError(
                f"Could not find global resource for resource type `{resource_type}` and resource key `{resource_key}`. Make sure to initialize the global resource before starting the aggregation."
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
                pseudo_gradients[resource_key][parameter_key] = (
                    parameter - global_model[parameter_key]
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
            scores = {client.name: 1 for client in topology.online_clients()}
        else:
            try:
                scores = {
                    client.name: client.data[self.client_score.__name__]
                    for client in topology.online_clients(False)
                }
            except KeyError:
                raise KeyError(
                    f"Client score {self.client_score.__name__} not found in topology register. "
                    f"Please check that the score is requested."
                )

        # normalize weights
        total_score = sum(scores.values())
        weights = {
            client_name: score / total_score for client_name, score in scores.items()
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
    ) -> dict[str, torch.Tensor]:
        """Aggregates the resources.

        Args:
            resource_type (str): The type of the resource.
            resource_key (str): The key of the resource.
            resources (dict[str, DictCheckpoint]): The resources to aggregate.
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            dict[str, torch.Tensor]: The aggregated resource.
        """
        raise NotImplementedError(
            "Please implement this method for your aggregation method."
        )


class AggregationAction(Action):
    def __init__(
        self,
        aggregator: Aggregator,
        _no_aggregation: bool = False,
        predecessor: Instruction | None = None,
        remove_instruction_resource_entry: bool = True,
        **kwargs,
    ) -> None:
        """Aggregation action. Aggregates the models and optimizers on the server.

        Args:
            aggregator (Aggregator): The aggregator to use.
            _no_aggregation (bool, optional): Whether to skip the aggregation. Defaults to False.
            predecessor (Instruction | None, optional): The predecessor instruction. Defaults to None.
            remove_instruction_resource_entry (bool, optional): Whether to remove the resource entry from the instruction. Defaults to True.
        """
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

        # if no aggregation is required, return None. This is mainly the case when training on a single client
        if self._no_aggregation:
            return None

        # use the client checkpoints to aggregate the models / optimizers
        client_checkpoints = resource_manager.client_checkpoints

        for resource_type, resources in client_checkpoints.items():
            for resource_key, resource in resources.items():
                # get global model from checkpoint manager

                cm = resource_manager.checkpoint_manager

                if len(resource) < 2:
                    logger.warning(
                        f"Only one client checkpoint found for resource {resource_type} {resource_key}."
                    )

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
