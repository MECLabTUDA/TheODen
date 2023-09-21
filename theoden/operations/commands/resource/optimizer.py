import torch

from typing import Optional, Union, Dict, Any, List
import io

from theoden.topology import TopologyRegister

from .. import Command
from . import SetResourceCommand
from theoden.common import Transferable, ExecutionResponse
from theoden.resources import Optimizer_, Optimizer, Model, ResourceRegister
from ...serverrequests import GetResourceCheckpointRequest


class SetOptimizerCommand(SetResourceCommand, Transferable):
    def __init__(
        self,
        optimizer: Optimizer_,
        key: str = "optimizer",
        model_key: str = "model",
        overwrite: bool = True,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            key=key,
            resource=optimizer,
            overwrite=overwrite,
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.model_key = model_key
        self.assert_type = Optimizer

    def modify_resource(self, resource: Optimizer_) -> Optimizer:
        resource = resource.build([self.node_rr.gr(self.model_key, Model).module()])
        return resource

    def all_clients_finished_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        instruction_uuid: str,
    ) -> None:
        from ....watcher import ParameterNotification

        resource_register.watcher.notify_all(
            notification=ParameterNotification(
                params={
                    "optimizer": self.resource.opti_class.__name__,
                    **self.resource.opti_args,
                }
            ),
        )


class LoadOptimizerStateDictCommand(Command, Transferable):
    def __init__(
        self,
        resource_key: str,
        checkpoint_key: str = "__global__",
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            node=node,
            uuid=uuid,
            **kwargs,
        )
        self.resource_key = resource_key
        self.checkpoint_key = checkpoint_key

    def execute(self) -> ExecutionResponse | None:
        # request state dict from server
        response = self.node.send_server_request(
            GetResourceCheckpointRequest(
                resource_type="optimizer",
                resource_key=self.resource_key,
                checkpoint_key=self.checkpoint_key,
            )
        )
        # load state dict into model
        if response.content != b"null":
            # print(response.content, response.content != b"null")
            sd = torch.load(io.BytesIO(response.content))
            _sd = self.node_rr.gr("optimizer", assert_type=Optimizer).state_dict()
            _sd["state"] = sd
            self.node_rr.gr("optimizer", assert_type=Optimizer).load_state_dict(_sd)
