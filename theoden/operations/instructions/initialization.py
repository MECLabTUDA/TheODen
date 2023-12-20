from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch

from ...common import Transferable
from ...resources import Model, NumpyStateLoader, ResourceManager, StateLoader
from ...resources.meta import DictCheckpoint, ModelCheckpoints
from ...topology import Topology
from ..commands import SendModelToServerCommand
from .action import Action
from .distribution import ClosedDistribution
from .instruction import Instruction
from .selection import NSelector


class Initializer(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def initialize(
        self,
        action: InitGlobalModelAction,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> Instruction | list[Instruction] | None:
        pass


class ServerInitializer(Initializer, Transferable):
    def initialize(
        self,
        action: InitGlobalModelAction,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> Instruction | list[Instruction] | None:
        # init the model on the server
        state_dict = (
            resource_manager.gr(action.model_key, assert_type=Model)
            .parse_to("cpu")
            .get_state_dict()
        )
        # create a checkpoint
        checkpoint = DictCheckpoint(state_dict)
        # register the checkpoint
        resource_manager.checkpoint_manager.register_checkpoint(
            resource_type="model",
            resource_key=action.model_key,
            checkpoint_key="__global__",
            checkpoint=checkpoint,
            create_type_if_not_exists=ModelCheckpoints,
        )
        return None


class FileInitializer(Initializer, Transferable):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    def initialize(
        self,
        action: InitGlobalModelAction,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> Instruction | list[Instruction] | None:
        # load the state dict from the file
        state_dict = torch.load(self.file_path)
        # create a checkpoint
        checkpoint = DictCheckpoint(state_dict)
        # register the checkpoint
        resource_manager.checkpoint_manager.register_checkpoint(
            resource_type="model",
            resource_key=action.model_key,
            checkpoint_key="__global__",
            checkpoint=checkpoint,
            create_type_if_not_exists=ModelCheckpoints,
        )
        return None


class SelectRandomOneInitializer(Initializer, Transferable):
    """Select the state dict of a random client and distribute it to all clients."""

    def __init__(
        self,
        loader: type[StateLoader] | None = None,
    ):
        super().__init__()
        self.loader = loader if loader else NumpyStateLoader()

    def initialize(
        self,
        action: InitGlobalModelAction,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> Instruction | list[Instruction] | None:
        """Select the state dict of a random client and distribute it to all clients.

        Args:
            instruction (Instruction): The ModelInitializationInstruction.
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            Instruction: The instruction to distribute the model to the clients.
        """

        # Step 1: Select a random client and get the state dict
        _get_state_dict_of_one = ClosedDistribution(
            SendModelToServerCommand(action.model_key, loader=self.loader),
            selector=NSelector(1),
        )

        def _get_state_dict_of_one_hook(
            i: Instruction,
            tr: Topology,
            rr: ResourceManager,
        ) -> None:
            # register the checkpoint
            try:
                rr.checkpoint_manager.register_checkpoint(
                    resource_type="model",
                    resource_key=action.model_key,
                    checkpoint_key="__global__",
                    checkpoint=rr.client_checkpoints.get_checkpoint(
                        resource_type="model",
                        resource_key=action.model_key,
                        checkpoint_key=i.dist_table.selected[0],
                    ),
                    create_type_if_not_exists=ModelCheckpoints,
                )
            except IndexError as e:
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Could not register the global model checkpoint. The client checkpoint with key {i.dist_table.selected[0]} does not exist."
                    f"This might be due to the fact that the client did not send the model to the server. or that an error occured during the transfer."
                )
                raise e

        # Step 2: register an on_finish hook to copy set the global model to the returned state dict of the client
        _get_state_dict_of_one.register_on_finish_hook(_get_state_dict_of_one_hook)

        return _get_state_dict_of_one


class InitGlobalModelAction(Action, Transferable):
    """Action to initialize the model on the clients.

    This Action will send the model to the clients and initialize it.
    """

    def __init__(
        self,
        initializer: Initializer | None = None,
        model_key: str = "model",
        remove_instruction_resource_registry: bool = True,
    ) -> None:
        """Initialize the model on the clients.

        Args:
            initializer (Initializer, optional): The initializer to use. Defaults to None.
            model_key (str, optional): The key of the model to initialize. Defaults to "model".
            remove_instruction_resource_registry (bool, optional): Whether to remove the resource registry of the instruction after the execution. Defaults to True.
        """
        super().__init__(
            remove_instruction_resource_entry=remove_instruction_resource_registry
        )
        self.initializer = initializer if initializer else SelectRandomOneInitializer()
        self.model_key = model_key

    def perform(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | None:
        """Set the state dict of all clients to a unified state dict.

        Return a successor instruction, that will be executed after the model is initialized.
        This Instruction will select a state dict and distribute it to all clients.

        Args:
            topology (Topology): The topology register.
            resource_manager (ResourceManager): The resource register.

        Returns:
            Instruction: The successor instruction.
        """
        return self.initializer.initialize(self, topology, resource_manager)
