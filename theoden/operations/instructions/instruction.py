from __future__ import annotations

from enum import IntEnum, auto
from copy import deepcopy
from uuid import uuid4
from typing import Literal


from ..status import NodeCommandStatus
from .distribute import Distributor, AllDistributor
from ...common import Transferable, StatusUpdate
from ...topology import TopologyRegister, TopologyType
from ...resources import ResourceRegister
from .status_handler import StatusHandler, BaseHandler
from ..commands import Command, SequentialCommand
from ...resources.resource import is_instance_of_type_hint


class StatusTable:
    def __init__(self, keys: dict[str, bool]):
        self.table: dict[str, NodeCommandStatus] = {}
        self.history: dict[str, list[NodeCommandStatus]] = {}
        for key, included in keys.items():
            if included:
                self.table[key] = NodeCommandStatus.UNREQUESTED
            else:
                self.table[key] = NodeCommandStatus.EXCLUDED
            self.history[key] = [self.table[key]]

    def set_status(self, key: str, status: NodeCommandStatus):
        if key not in self.table:
            raise KeyError(f"Key '{key}' is not in the status table")

        prev_status = self.table[key]
        if prev_status != status:
            self.history[key].append(status)

        self.table[key] = status

    def get_status(self, key: str) -> NodeCommandStatus:
        if key not in self.table:
            raise KeyError(f"Key '{key}' is not in the status table")

        return self.table[key]

    def get_history(self, key: str) -> list[NodeCommandStatus]:
        if key not in self.history:
            raise KeyError(f"Key '{key}' is not in the status table")

        return self.history[key]

    def get_included(self) -> list[str]:
        return [k for k, v in self.table.items() if v != NodeCommandStatus.EXCLUDED]

    def is_finished_or_excluded(self) -> bool:
        for status in self.table.values():
            if (
                status != NodeCommandStatus.FINISHED
                and status != NodeCommandStatus.EXCLUDED
            ):
                return False
        return True

    def get_number_of_active(self) -> int:
        count = 0
        for status in self.table.values():
            if (
                status == NodeCommandStatus.SEND
                or status == NodeCommandStatus.STARTED
                or status == NodeCommandStatus.WAIT_FOR_RESPONSE
            ):
                count += 1
        return count

    def __repr__(self) -> str:
        return f"StatusTable({self.table})"


class InstructionStatus(IntEnum):
    # Enum values for Instruction status
    CREATED = auto()  # The Instruction object has been created
    BOOTING = auto()  # The Instruction object is initializing
    EXECUTION = auto()  # The Instruction object is executing
    EXECUTION_FINISHED = auto()  # The Instruction object has finished executing
    COMPLETED = auto()  # The Instruction object has completed its execution cycle


class Instruction(Transferable, is_base_type=True):
    def __init__(
        self,
        wrapped_object: Command,
        distributor: Distributor | None = None,
        status_handler: list[StatusHandler] | None = None,
        has_base_handler: bool = True,
        block: bool = True,
        remove_instruction_resource_registry: bool = True,
        simultaneous_execution: int = 0,
        **kwargs,
    ) -> None:
        """Instruction class is a wrapper around a Command.
        It is responsible for distributing the Command to the nodes in the topology and handling the status updates from the nodes.

        Args:
            wrapped_object (Command): The Command object to be wrapped
            distributor (Distributor, optional): The Distributor object to be used for distributing the Command. Defaults to None.
            status_handler (list[StatusHandler], optional): The StatusHandler objects to be used for handling the status updates from the nodes. Defaults to None.
            has_base_handler (bool, optional): Whether to include the BaseHandler in the list of StatusHandler objects. Defaults to True.
            block (bool, optional): Whether to block the execution of the Instruction until it is completed. Defaults to True.
            remove_instruction_resource_registry (bool, optional): Whether to remove the resources registered by the Instruction from the ResourceRegister. Defaults to True.
            simultaneous_execution (int, optional): The execution order of the Instruction. Defines how many node will execute the command simultaneously. Defaults to 0 (all nodes).

        Raises:
            AssertionError: If the wrapped object is not a Command
        """

        # Ensure that the wrapped object is a Command or Instruction
        assert isinstance(wrapped_object, Command), "Wrapped object must be Command"

        # Initialize instance variables
        self.wrapped_object = wrapped_object
        self.block = block
        self.command_level_status: dict[str, StatusTable] = {}
        self.instruction_status = InstructionStatus.CREATED
        self.successor: list[Instruction] = []
        self.remove_instruction_resource_registry = remove_instruction_resource_registry
        self.simultaneous_execution = simultaneous_execution

        if distributor is None:
            self.distributor = AllDistributor()
        else:
            self.distributor = distributor
            assert isinstance(self.distributor, Distributor)

        if status_handler is None:
            self.status_handler: list[StatusHandler] = (
                [] if not has_base_handler else [BaseHandler()]
            )
        else:
            self.status_handler = status_handler
            assert isinstance(self.status_handler, list)
            for handler in self.status_handler:
                assert isinstance(handler, StatusHandler)

            if has_base_handler:
                # Put BaseHandler as the first handler in the list of status handlers
                self.status_handler.insert(0, BaseHandler())

        self.uuid: str | None = None
        self.on_init_hooks: list[callable] = []
        self.on_finish_hooks: list[callable] = []

    def _init_table(self, topology_register: TopologyRegister) -> dict[str, bool]:
        """Helper method to create a status table for each node in the topology register

        Args:
            topology_register (TopologyRegister): The topology register to use for creating the status table

        Returns:
            dict[str, bool]: A dictionary mapping each node in the topology register to a boolean value indicating if it is selected
        """
        self.selected = self.distributor.select_nodes(topology_register)
        return {key: key in self.selected for key in topology_register.get_all_nodes()}

    # Placeholder methods to be overridden by subclasses
    def on_init(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ):
        pass

    def _on_init(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ):
        """Internal method for initializing the instruction

        Args:
            topology_register (TopologyRegister): The topology register of the server
            resource_register (ResourceRegister): The resource register of the server
        """

        # set uuid
        self.uuid = str(uuid4())

        # Set the instruction status to BOOTING
        self.instruction_status = InstructionStatus.BOOTING

        # Call the on_init method of the subclass
        self.on_init(topology_register, resource_register)

        # Create a status table for the current node
        self.node_level_status = StatusTable(self._init_table(topology_register))

        # If the wrapped object is a Command, initialize its UUID and create a status table for each node in the
        # topology register
        if isinstance(self.wrapped_object, Command):
            self.wrapped_object.init_uuid()
            for command_uuid in self.wrapped_object.get_command_tree().keys():
                self.command_level_status[command_uuid] = deepcopy(
                    self.node_level_status
                )

        # Set the instruction status to EXECUTION
        self.instruction_status = InstructionStatus.EXECUTION

    def register_on_init_hook(self, hook: callable):
        self.on_init_hooks.append(hook)

    def infer_command(
        self,
        node_uuid: str,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        as_dict: bool = True,
    ) -> dict | None:
        """Infer the command to send to a node.

        This method is called by the server if a node requests a command. It returns the command to send to the node.
        If the instruction is in the EXECUTION state, it returns the command to send to the node.
        Otherwise, it returns None or an empty dict depending on the value of `as_dict`.
        if the instruction is in the CREATED state, it calls the _on_init method to initialize the instruction.

        Args:
            node_uuid (str): The UUID of the node to infer the command for
            topology_register (TopologyRegister): The topology register of the server
            resource_register (ResourceRegister): The resource register of the server
            as_dict (bool, optional): Whether to return the command as a dictionary. Defaults to True.

        Returns:
            dict | None: The command to send to the node if the instruction is in the EXECUTION state, None or an empty dict otherwise
        """

        # If the instruction has just been created, initialize it
        if self.instruction_status is InstructionStatus.CREATED:
            self._on_init(
                topology_register=topology_register, resource_register=resource_register
            )

        # If the instruction is still booting up, return an empty dictionary or None
        elif self.instruction_status is InstructionStatus.BOOTING:
            return {} if as_dict else None

        # If the instruction is currently executing
        elif self.instruction_status is InstructionStatus.EXECUTION:
            # Check if the command with the given UUID has been requested from all nodes
            # print(self.node_level_status.get_number_of_active())
            if self.node_level_status.table[
                node_uuid
            ] is NodeCommandStatus.UNREQUESTED and not (
                self.simultaneous_execution > 0
                and self.node_level_status.get_number_of_active()
                >= self.simultaneous_execution
            ):
                # Mark the command as requested from all nodes
                for command_uuid, table in self.command_level_status.items():
                    table.table[node_uuid] = NodeCommandStatus.SEND
                self.node_level_status.table[node_uuid] = NodeCommandStatus.SEND

                # If the wrapped object is an instruction, recursively call `infer_command`
                if isinstance(self.wrapped_object, Instruction):
                    return self.wrapped_object.infer_command(
                        node_uuid, topology_register, as_dict
                    )

                # Otherwise, return the wrapped object in JSON format if `as_dict` is True, or the object itself
                else:
                    return (
                        self.wrapped_object.node_specific_modification(
                            self.command_level_status, node_uuid=node_uuid
                        ).dict()
                        if as_dict
                        else self.wrapped_object.node_specific_modification(
                            self.command_level_status, node_uuid=node_uuid
                        )
                    )

        # If the instruction is already completed, return an empty dictionary or None
        return {} if as_dict else None

    def handle_response(
        self,
        status_update: StatusUpdate,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> None:
        """Handle a status update from a node by using the status handlers. This method will be called inside the process_status_update method.

        Args:
            status_update (StatusUpdate): The status update to handle
            topology_register (TopologyRegister): The topology register of the server
            resource_register (ResourceRegister): The resource register of the server
        """

        for handler in self.status_handler:
            handler.handle_status_update(
                self, status_update, topology_register, resource_register
            )

    def _check_all_finished_or_excluded(self) -> bool:
        """Internal method to check if all commands have a status of FINISHED or EXCLUDED

        Returns:
            bool: True if all commands have a status of FINISHED or EXCLUDED, False otherwise
        """
        # Iterate over each status table for each command
        for value in self.command_level_status.values():
            # Iterate over the status of each node command in the table

            if not value.is_finished_or_excluded():
                return False

        # If all commands have a status of FINISHED or EXCLUDED, return True
        return True

    def process_status_update(
        self,
        status_update: StatusUpdate,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> None:
        """Process a status update from a node.

        This method is called by the server when it receives a status update from a node. It calls the handle_response method to update the status of the command.
        If all commands have a status of FINISHED or EXCLUDED, it calls the _on_finish method to finish the instruction.

        Args:
            status_update (StatusUpdate): The status update to process
            topology_register (TopologyRegister): The topology register of the server
            resource_register (ResourceRegister): The resource register of the server
        """

        # Call the handle_response method to update the status of the command
        self.handle_response(status_update, topology_register, resource_register)

        # If all commands have a status of FINISHED or EXCLUDED, call the _on_finish method
        if self._check_all_finished_or_excluded():
            self._on_finish(topology_register, resource_register)

    def on_finish(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ) -> Instruction | list[Instruction] | None:
        return

    def register_on_finish_hook(self, hook: callable) -> Instruction:
        """Register a hook to be called when the instruction finishes

        This may be necessary if the successor instruction requires data from the current instruction

        Args:
            hook (callable): The hook to be called

        Returns:
            Instruction: The instruction itself
        """
        self.on_finish_hooks.append(hook)
        return self

    # Internal method for finishing the instruction
    def _on_finish(
        self, topology_register: TopologyRegister, resource_register: ResourceRegister
    ):
        """Internal method for finishing the instruction.

        This method is called when all commands have a status of FINISHED or EXCLUDED. It calls the on_finish method of the subclass and sets the instruction status to COMPLETED.
        Also, the successor s are determined by calling the on_finish method of the subclass and the on_finish_hooks. They are then added to the successor list of the instruction.
        Finally the instruction status is set to COMPLETED and the resources are freed up.

        Args:
            topology_register (TopologyRegister): The topology register of the server
            resource_register (ResourceRegister): The resource register of the server
        """

        # Set the instruction status to EXECUTION_FINISHED
        self.instruction_status = InstructionStatus.EXECUTION_FINISHED

        # Call the on_finish method of the subclass
        successor = self.on_finish(topology_register, resource_register)

        # if successor is not a list, convert it to a list
        if successor is not None and (
            isinstance(successor, Instruction)
            or is_instance_of_type_hint(successor, list[Instruction])
        ):
            if not isinstance(successor, list):
                assert isinstance(
                    successor, Instruction
                ), "successor must be an Instruction, not {}".format(
                    type(successor).__name__
                )
                successor = [successor]
            self.successor = self.successor + successor

        # Call the on_finish_hooks and add the successors to the successor list
        for hook in self.on_finish_hooks:
            successor = hook(self, topology_register, resource_register)

            if successor is not None:
                if not isinstance(successor, list):
                    assert isinstance(
                        successor, Instruction
                    ), "successor must be an Instruction, not {}".format(
                        type(successor).__name__
                    )
                    successor = [successor]
                self.successor = self.successor + successor

        # If the instruction status is still EXECUTION_FINISHED, set it to COMPLETED
        if self.instruction_status == InstructionStatus.EXECUTION_FINISHED:
            self.instruction_status = InstructionStatus.COMPLETED

        # Free up resources after the instruction has finished to save memory
        if self.remove_instruction_resource_registry and self.uuid in resource_register:
            resource_register.rm(self.uuid, assert_type=ResourceRegister)

    def _add_commands_to_wrapped(
        self, pre_commands: list[Command], post_commands: list[Command]
    ):
        """Internal method to add commands to the wrapped object

        Args:
            pre_commands (list[Command]): The commands to add before the wrapped object
            post_commands (list[Command]): The commands to add after the wrapped object
        """
        # Add the pre_commands to the wrapped object
        self.wrapped_object = SequentialCommand(
            pre_commands + [self.wrapped_object] + post_commands
        )
