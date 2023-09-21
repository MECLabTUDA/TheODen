import torch

from typing import Any, Optional
import io

from .. import Command
from ....common import Transferable, ExecutionResponse, GlobalContext, ResourceResponse
from ....topology import TopologyRegister
from ....resources import ResourceRegister, Model, Optimizer


class SendResourceToServerCommand(Command):
    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        *,
        assert_type: type = bytes,
        node: Optional["Node"] = None,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)

        if isinstance(resource_key, str):
            self.resource_key = [resource_key]
        else:
            self.resource_key = resource_key

        self.assert_type = assert_type

    def on_client_finish_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        node_uuid: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        # add the model to the resource register of the instruction
        for key, resource in execution_response.get_files().items():
            resource_register.sr(
                key=f"{instruction_uuid}:{key}:{node_uuid}",
                resource=resource,
                assert_type=bytes,
                overwrite=True,
            )

    def _to_resource(self, resource: any) -> any:
        return resource

    def execute(self) -> ResourceResponse | None:
        resources = {}

        # if resource_key is @all, send all models
        if self.resource_key == ["@all"]:
            _resources = self.node_rr.gr_of_type(self.assert_type)
        else:
            _resources = {
                key: self.node_rr.gr(key, assert_type=self.assert_type)
                for key in self.resource_key
            }

        # save all models to a buffer
        for key in _resources:
            buffer = io.BytesIO()
            torch.save(self._to_resource(_resources[key]), buffer)
            resources[key] = buffer.getvalue()

        return ResourceResponse(
            resource_type=self.resource_type,
            resources=resources,
        )


class SendModelToServerCommand(SendResourceToServerCommand, Transferable):
    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        *,
        node: Any | None = None,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            resource_key,
            assert_type=bytes,
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = Model
        self.resource_type = "model"

    def _to_resource(self, resource: Model) -> any:
        return resource.get_state_dict()


class SendOptimizerToServerCommand(SendResourceToServerCommand, Transferable):
    def __init__(
        self,
        resource_key: str | list[str] = "@all",
        *,
        node: Any | None = None,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            resource_key,
            assert_type=bytes,
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.assert_type = Optimizer
        self.resource_type = "optimizer"

    def _to_resource(self, resource: Optimizer) -> any:
        return resource.state_dict()["state"]
