import torch

from typing import Any, Optional
import io
from pathlib import Path


from ....common import Transferable, GlobalContext, ExecutionResponse
from ....topology import TopologyRegister
from ....resources import ResourceRegister, Model
from .. import Command
from . import SetResourceCommand
from ...serverrequests import GetResourceCheckpointRequest


class InitModelCommand(SetResourceCommand, Transferable):
    """Command to initialize a model on a node.

    This command is used to initialize a model on a node. It is used in the following way:

    ```python
    from theoden.operations.commands.resource import InitModelCommand
    from theoden.model_registry import Model

    model = Model("timm", {"model_name": "resnet18"})})
    command = InitModelCommand(model)
    command.execute()
    ```
    """

    def __init__(
        self,
        model: Model,
        model_key: str = "model",
        overwrite: bool = True,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize a model on a node.

        Args:
            model (Model): The model to initialize.
            model_key (str, optional): The key to use to register the model. Defaults to "model".
            overwrite (bool, optional): Whether to overwrite the model if it already exists. Defaults to True.
            node (Optional["Node"], optional): The node to initialize the model on. Defaults to None.
            uuid (Optional[str], optional): The uuid of the command. Defaults to None.

        Raises:
            ValueError: If the model is not of type Model.
        """

        super().__init__(
            key=model_key,
            resource=model,
            overwrite=overwrite,
            node=node,
            uuid=uuid,
            **kwargs
        )
        self.assert_type = Model

    def modify_resource(self, resource: Model) -> Model:
        """Parse the model and put it on the correct device.

        Args:
            resource (Model): The model to parse.

        Returns:
            Model: The parsed model.
        """
        return resource.parse_to(self.node_rr.gr("device", str))

    def all_clients_finished_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        instruction_uuid: str,
    ) -> None:
        # initialize a global model
        resource_register.sr(self.key, self.resource)


class LoadStateDictCommand(Command, Transferable):
    def __init__(
        self,
        resource_key: str,
        checkpoint_key: str = "__global__",
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.resource_key = resource_key
        self.checkpoint_key = checkpoint_key

    def execute(self) -> ExecutionResponse | None:
        # request state dict from server
        response = self.node.send_server_request(
            GetResourceCheckpointRequest(
                resource_type="model",
                resource_key=self.resource_key,
                checkpoint_key=self.checkpoint_key,
            )
        )
        # load state dict into model
        sd = torch.load(io.BytesIO(response.content))
        self.node_rr.gr(self.resource_key, assert_type=Model).load_state_dict(sd)
