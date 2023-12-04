from ....common import Transferable
from ....topology import Topology
from ....resources import ResourceManager, Model
from . import SetResourceCommand


class InitModelCommand(SetResourceCommand, Transferable):
    """Command to initialize a model on a node.

    This command is used to initialize a model on a node. It is used in the following way:

    ```python
    from theoden.operations.commands.resource import InitModelCommand
    from theoden.model_registry import Model

    model = Model("timm", {"model_name": "resnet18"})})
    command = InitModelCommand(model)
    command()
    ```
    """

    def __init__(
        self,
        model: Model,
        model_key: str = "model",
        overwrite: bool = True,
        *,
        uuid: str | None = None,
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
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = Model

    def modify_resource(self, resource: Model) -> Model:
        """Parse the model and put it on the correct device.

        Args:
            resource (Model): The model to parse.

        Returns:
            Model: The parsed model.
        """
        return resource.parse_to(self.node_rm.gr("device", str))

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        # initialize a global model
        resource_manager.sr(self.key, self.resource)
