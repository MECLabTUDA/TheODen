from ....common import ExecutionResponse, ResourceResponse
from ....resources import (
    Model,
    NumpyStateLoader,
    Optimizer,
    ResourceManager,
    StateLoader,
)
from ....resources.meta import DictCheckpoint
from ....topology import Topology
from .. import Command


class SendCheckpointToServerCommand(Command):
    """Sends a checkpoint to the server."""

    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        loader: type[StateLoader] | None = None,
        *,
        assert_type: type = bytes,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Sends a checkpoint to the server.

        Args:
            resource_key (str | list[str], optional): The key of the resource that should be sent to the server. Defaults to "@all".
            reset_checkpoints (bool, optional): Whether to reset the checkpoints on the client after the resource has been sent. Defaults to True.
            assert_type (type, optional): The type of the resource that should be sent to the server. Defaults to bytes.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """
        super().__init__(uuid=uuid, **kwargs)

        if isinstance(resource_key, str):
            self.resource_key = [resource_key]
        else:
            self.resource_key = resource_key

        self.assert_type = assert_type
        self.state_loader = loader or NumpyStateLoader

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        client_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        """Add checkpoint to the client checkpoint manager (this is on the server).

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.
            client_name (str): The name of the client that executed the command.
            execution_response (ExecutionResponse): The execution response of the command.
            instruction_uuid (str): The uuid of the instruction that executed the command.
        """

        # get the client checkpoint manager
        cm = resource_manager.client_checkpoints

        # add the checkpoints to the client checkpoint manager
        for key, resource in execution_response.get_files().items():
            cm.register_checkpoint(
                resource_type=self.resource_type,
                resource_key=key,
                checkpoint_key=client_name,
                checkpoint=DictCheckpoint(state_dict=self.state_loader.load(resource)),
            )

    def _to_dict(self, resource: any) -> dict:
        """Converts the resource to a dictionary.

        Args:
            resource (any): The resource that should be converted.

        Returns:
            dict: The converted resource.
        """
        return resource

    def execute(self) -> ResourceResponse | None:
        resource_manager = {}

        # if resource_key is @all, send all models
        if self.resource_key == ["@all"]:
            _resource_manager = self.client_rm.gr_of_type(self.assert_type)
        else:
            _resource_manager = {
                key: self.client_rm.gr(key, assert_type=self.assert_type)
                for key in self.resource_key
            }

        # save all models to a buffer
        for key in _resource_manager:
            resource_manager[key] = self.state_loader.save(
                self._to_dict(_resource_manager[key]),
            )

        return ResourceResponse(
            resource_type=self.resource_type,
            resource_manager=resource_manager,
        )


class SendModelToServerCommand(SendCheckpointToServerCommand):
    """Sends a model checkpoint to the server."""

    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        only_grad: bool = False,
        cpu: bool = False,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            resource_key,
            assert_type=bytes,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = Model
        self.resource_type = "model"
        self.only_grad = only_grad
        self.cpu = cpu

    def _to_dict(self, resource: Model) -> dict:
        """Converts the model to the state dict.

        Args:
            resource (Model): The model that should be converted.

        Returns:
            dict: The converted model.
        """

        if self.only_grad:
            sd = resource.get_grad_params_state_dict()

        else:
            sd = resource.get_state_dict()

        if self.cpu:
            for k in sd:
                sd[k] = sd[k].cpu()
        return sd


class SendOptimizerToServerCommand(SendCheckpointToServerCommand):
    """Sends an optimizer checkpoint to the server."""

    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            resource_key,
            assert_type=bytes,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = Optimizer
        self.resource_type = "optimizer"

    def _to_dict(self, resource: Optimizer) -> dict:
        """Converts the optimizer to the state dict.

        Args:
            resource (Optimizer): The optimizer that should be converted.

        Returns:
            dict: The converted optimizer.
        """
        return resource.state_dict()["state"]
