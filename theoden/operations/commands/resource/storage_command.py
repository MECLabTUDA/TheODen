import torch

import io

from ....common import Transferable, ExecutionResponse
from ....topology import Topology
from ....resources import ResourceManager, Model, Optimizer
from .. import Command
from ....resources import StateLoader, NumpyStateLoader


class StorageCommand(Command, Transferable):
    def __init__(
        self, storage_uuids: dict | None = None, *, uuid: str | None = None, **kwargs
    ) -> None:
        super().__init__(uuid=uuid, **kwargs)
        self.storage_uuids = storage_uuids

    def _get_files_server_side(
        self, resource_manager: ResourceManager
    ) -> dict[str, bytes]:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def _get_files_client_side(
        self, resource_manager: ResourceManager
    ) -> dict[str, bytes]:
        """Download the resources from the storage server

        Args:
            resource_manager (ResourceManager): The resource register of the node.

        Returns:
            dict[str, bytes]: The downloaded files.
        """

        fsi = resource_manager.storage
        files = {}
        for key, storage_uuid in self.storage_uuids.items():
            files[key] = fsi.load_resource(storage_uuid)
        return files

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_nodes: list[str],
    ):
        if self.storage_uuids is None:
            self.storage_uuids = resource_manager.storage.upload_resources(
                self._get_files_server_side(resource_manager)
            )
            self.add_initialization_parameter(storage_uuids=self.storage_uuids)

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        fsi = resource_manager.storage
        for key, storage_uuid in self.storage_uuids.items():
            fsi.remove_resource(storage_uuid)


class LoadStateDictCommand(StorageCommand, Transferable):
    def __init__(
        self,
        resource_key: str,
        checkpoint_key: str = "__global__",
        storage_uuids: dict | None = None,
        loader: type[StateLoader] | None = None,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(storage_uuids=storage_uuids, uuid=uuid, **kwargs)
        self.resource_key = resource_key
        self.checkpoint_key = checkpoint_key
        self.loader = loader or NumpyStateLoader

    def _get_files_server_side(
        self, resource_manager: ResourceManager
    ) -> dict[str, bytes]:
        return {
            self.resource_key: self.loader().save(
                resource_manager.checkpoint_manager.get_checkpoint(
                    "model", self.resource_key, self.checkpoint_key
                ).to(dict)
            )
        }

    def execute(self) -> ExecutionResponse | None:
        # request state dict from server
        model = self._get_files_client_side(self.node_rm)
        # load state dict into model
        sd = self.loader.load(model[self.resource_key])
        self.node_rm.gr(self.resource_key, assert_type=Model).load_state_dict(sd)
        return None


class LoadOptimizerStateDictCommand(StorageCommand, Transferable):
    def __init__(
        self,
        resource_key: str,
        checkpoint_key: str = "__global__",
        storage_uuids: dict | None = None,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(storage_uuids=storage_uuids, uuid=uuid, **kwargs)
        self.resource_key = resource_key
        self.checkpoint_key = checkpoint_key

    def _get_files_server_side(
        self, resource_manager: ResourceManager
    ) -> dict[str, bytes]:
        return {
            self.resource_key: resource_manager.checkpoint_manager.get_checkpoint(
                "optimizer", self.resource_key, self.checkpoint_key
            ).to(bytes)
        }

    def execute(self) -> ExecutionResponse | None:
        # request state dict from server
        model = self._get_files_client_side(self.node_rm)
        # load state dict into model
        sd = torch.load(io.BytesIO(model[self.resource_key]))
        self.node_rm.gr(self.resource_key, assert_type=Optimizer).load_state_dict(sd)
        return None
