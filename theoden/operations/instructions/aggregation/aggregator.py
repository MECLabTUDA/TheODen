import torch

from abc import ABC, abstractmethod

from ....topology import TopologyRegister
from ....resources import ResourceRegister
from ....resources.meta import CheckpointManager, DictCheckpoint
from ...commands import (
    Command,
    SendModelToServerCommand,
    SendOptimizerToServerCommand,
)
from ....common import Transferable


class Aggregator(ABC, Transferable, is_base_type=True):
    def __init__(
        self,
        model_key: str | list[str] = "@all",
        optimizer_key: str | None = None,
        client_score: type | None = None,
        **kwargs,
    ):
        self.models_dicts: dict[str, dict] = {}
        self.model_key = model_key
        self.optimizer_key = optimizer_key
        self.resources = CheckpointManager()
        self.client_score = client_score

    def process_resource(
        self, resource_type: str, resource_name: str, node_uuid: str, resource: any
    ) -> None:
        """Process a resource that is received from a node.

        Args:
            resource_type (str): The type of the resource.
            resource_name (str): The name of the resource.
            node_uuid (str): The uuid of the node that sent the resource.
            resource (any): The resource that was sent.
        """

        self.resources.register_checkpoint(
            resource_type=resource_type,
            resource_key=resource_name,
            checkpoint_key=node_uuid,
            checkpoint=DictCheckpoint(resource),
        )

    def distribute_commands(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ) -> list[Command]:
        """Returns a list of commands that are required to distribute the models.

        This method is called on the server side before the aggregation is started. All commands that load resources
        that should be updated after a communication round should be returned here.

        Args:
            topology_register (TopologyRegister): The topology register.
            resource_register (ResourceRegister): The resource register.

        Returns:
            list[Command]: A list of commands that are required to distribute the models.
        """

        return self._get_checkpoint_manager(
            resource_register
        ).get_global_checkpoints_commands(
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
        self,
        resources: dict[str, DictCheckpoint],
        global_model: dict[str, any],
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
                    parameter.cpu() - global_parameter.cpu()
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

        elif isinstance(state_dicts[0], torch.Tensor):
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
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ) -> dict[str, float]:
        # get weights from topology register
        if self.client_score is None:
            scores = {
                node.uuid: 1 for node in topology_register.get_connected_nodes(False)
            }
        else:
            try:
                scores = {
                    node.uuid: node[self.client_score.__name__]
                    for node in topology_register.get_connected_nodes(False)
                }
            except KeyError:
                raise KeyError(
                    f"Client score {self.client_score.__name__} not found in topology register. "
                    f"Please check that the score is requested."
                )

        # normalize weights
        total_score = sum(scores.values())
        weights = {
            node_uuid: score / total_score for node_uuid, score in scores.items()
        }
        return weights

    def _get_checkpoint_manager(
        self, resource_register: ResourceRegister
    ) -> CheckpointManager:
        # get checkpoint manager from resource register
        try:
            cm = resource_register.gr("__checkpoints__", assert_type=CheckpointManager)
        except KeyError:
            raise KeyError(
                f"Checkpoint manager not found in resource register. "
                f"Please check that the checkpoint manager is requested."
            )
        return cm

    @abstractmethod
    def aggregate(
        self,
        resource_type: str,
        resource_key: str,
        resources: dict[str, DictCheckpoint],
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> any:
        raise NotImplementedError(
            "Please implement this method for your aggregation method."
        )
