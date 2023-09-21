from __future__ import annotations
import torch

from abc import ABC, abstractmethod
from pathlib import Path
import io

from theoden.operations.instructions.distribute import Distributor, NDistributor
from theoden.operations.instructions.status_handler import StatusHandler
from theoden.operations.instructions.instruction import Instruction
from theoden.resources import ResourceRegister
from theoden.topology import TopologyRegister
from ...common import Transferable, GlobalContext
from ...topology import TopologyRegister
from ...resources import ResourceRegister, Model
from ..commands import InitModelCommand, SendModelToServerCommand, LoadStateDictCommand
from .instruction import Instruction
from ...resources.meta import (
    CheckpointManager,
    ModelCheckpoints,
    DictCheckpoint,
    BytesCheckpoint,
)


class Initializer(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def initialize(
        self,
        instruction: ModelInitializationInstruction,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> Instruction:
        pass


class ServerInitializer(Initializer, Transferable):
    def __init__(self) -> None:
        super().__init__()

    def initialize(
        self,
        instruction: ModelInitializationInstruction,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> Instruction | list[Instruction] | None:
        # init the model on the server
        state_dict = (
            resource_register.gr(instruction.model_key, assert_type=Model)
            .parse_to("cpu")
            .get_state_dict()
        )
        # create a checkpoint
        checkpoint = DictCheckpoint(state_dict)
        # register the checkpoint
        resource_register.gr(
            "__checkpoints__", assert_type=CheckpointManager
        ).register_checkpoint(
            resource_type="model",
            resource_key=instruction.model_key,
            checkpoint_key="__global__",
            checkpoint=checkpoint,
            create_type_if_not_exists=ModelCheckpoints,
        )
        return None


class SelectRandomOneInitializer(Initializer, Transferable):
    """Select the state dict of a random client and distribute it to all clients."""

    def initialize(
        self,
        instruction: ModelInitializationInstruction,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> Instruction | list[Instruction] | None:
        """Select the state dict of a random client and distribute it to all clients.

        Args:
            instruction (Instruction): The ModelInitializationInstruction.
            topology_register (TopologyRegister): The topology register.
            resource_register (ResourceRegister): The resource register.

        Returns:
            Instruction: The instruction to distribute the model to the clients.
        """

        # Step 1: Select a random client and get the state dict
        _get_state_dict_of_one = Instruction(
            SendModelToServerCommand(instruction.model_key), distributor=NDistributor(1)
        )

        # Step 2: register an on_finish hook to copy set the global model to the returned state dict of the client
        _get_state_dict_of_one.register_on_finish_hook(
            lambda i, tr, rr: rr.sr(
                f"__checkpoints__:model:{instruction.model_key}:__global__",
                BytesCheckpoint(
                    rr.gr(
                        f"{i.uuid}:{instruction.model_key}:{i.selected[0]}",
                    )
                ),
                return_resource=False,
            )
        )
        return _get_state_dict_of_one


class FileInitializer(Initializer, Transferable):
    def __init__(self, file_path: str) -> None:
        super().__init__()
        self.file_path = file_path

    def initialize(
        self,
        instruction: ModelInitializationInstruction,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> Instruction:
        # load the state dict from the file
        state_dict = torch.load(self.file_path)
        # create a checkpoint
        checkpoint = DictCheckpoint(state_dict)
        # register the checkpoint
        resource_register.gr(
            "__checkpoints__", assert_type=CheckpointManager
        ).register_checkpoint(
            resource_type="model",
            resource_key=instruction.model_key,
            checkpoint_key="__global__",
            checkpoint=checkpoint,
            create_type_if_not_exists=ModelCheckpoints,
        )
        return None


class ModelInitializationInstruction(Instruction, Transferable):
    """Instruction to initialize the model on the clients.

    This instruction will send the model to the clients and initialize it.
    """

    def __init__(
        self,
        model: Model,
        model_key: str = "model",
        initializer: Initializer | None = None,
        distributor: Distributor | None = None,
        status_handler: list[StatusHandler] | None = None,
        has_base_handler: bool = True,
        block: bool = True,
        remove_instruction_resource_registry: bool = True,
    ) -> None:
        """Initialize the model on the clients.

        Args:
            model (Model): The model to initialize on the clients.
            initializer (Initializer, optional): The initializer to use. Defaults to None.
        """
        super().__init__(
            InitModelCommand(model, model_key=model_key),
            distributor,
            status_handler,
            has_base_handler,
            block,
            remove_instruction_resource_registry,
        )
        self.initializer = initializer if initializer else SelectRandomOneInitializer()
        self.model_key = model_key

    def on_finish(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ) -> Instruction | list[Instruction] | None:
        """Set the state dict of all clients to a unified state dict.

        Return a successor instruction, that will be executed after the model is initialized.
        This Instruction will select a state dict and distribute it to all clients.

        Args:
            topology_register (TopologyRegister): The topology register.
            resource_register (ResourceRegister): The resource register.

        Returns:
            Instruction: The successor instruction.
        """
        return self.initializer.initialize(self, topology_register, resource_register)
