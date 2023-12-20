from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from ...common import StatusUpdate, Transferable, to_list
from ...resources import ResourceManager
from ...topology import NodeStatus, Topology
from ...watcher import CommandFinishedNotification, StatusUpdateNotification
from .. import CommandDistributionStatus
from ..commands import Command, SequentialCommand
from .error import DistributionErrorHandler
from .instruction import Instruction, InstructionStatus
from .selection import AllSelector, Selector


class DistributionStatusTable(dict[str, dict[str, CommandDistributionStatus] | None]):
    """A table to keep track of the status of commands distributed to nodes.
    ```
    Structure:
    {
        node1: {
            command1: CommandDistributionStatus, # first command is the main command
            command2: CommandDistributionStatus, # next commands are subcommands
            ...
        },
        node2: {
            command1: CommandDistributionStatus,
            command2: CommandDistributionStatus,
            ...
        },
        node3: None, # None means that the node is not selected for the instruction
    }
    ```
    """

    def add_node(self, node_name: str, command_uuids: list[str] | None = None) -> None:
        """Adds a node to the table.

        Args:
            node_name (str): The name of the node.
            command_uuids (list[str], optional): The UUIDs of the commands to add. Defaults to None.
        """

        self[node_name] = (
            {
                command_uuid: CommandDistributionStatus.UNREQUESTED
                for command_uuid in command_uuids
            }
            if command_uuids is not None
            else None
        )

    @property
    def nodes(self) -> list[str]:
        """Returns a list of all nodes.

        Returns:
            list[str]: A list of all nodes.
        """
        return list(self.keys())

    def get_main_command_uuid(self, node_name: str) -> str | None:
        """Returns the UUID of the main command on a node.

        Args:
            node_name (str): The name of the node.

        Returns:
            str | None: The UUID of the main command on a node. Returns None if the node is not part of the instruction.
        """
        return list(self[node_name].keys())[0] if node_name in self.selected else None

    def set_for_all_node_commands(
        self, node_name: str, status: CommandDistributionStatus
    ) -> None:
        """Sets the status of all commands on a node.

        Args:
            node_name (str): The name of the node.
            status (CommandDistributionStatus): The new status.
        """
        if self[node_name] is None:
            logging.warning(
                f"Attempted to set status of all commands on node {node_name}, but the node is not part of the instruction."
            )
            return
        for command_uuid in self[node_name].keys():
            self[node_name][command_uuid] = status

    """The following properties are used to get information about the selection status of the nodes and commands.
    The selection status of a node gives information about whether the node is selected for the instruction."""

    @property
    def selection(self) -> dict[str, bool]:
        """Returns a dictionary of the selection status of all nodes.

        Returns:
            dict[str, bool]: A dictionary of the selection status of all nodes.
        """
        return {node_name: self[node_name] is not None for node_name in self}

    @property
    def selected(self) -> list[str]:
        """Returns a list of all active nodes.

        Returns:
            list[str]: A list of all active nodes.
        """
        return [node_name for node_name in self if self[node_name] is not None]

    @property
    def unselected(self) -> list[str]:
        """Returns a list of all inactive nodes.

        Returns:
            list[str]: A list of all inactive nodes.
        """
        return [node_name for node_name in self if self[node_name] is None]

    def node_selected(self, node_name: str) -> bool:
        """Returns whether a node is selected.

        Args:
            node_name (str): The name of the node.

        Returns:
            bool: Whether a node is selected.
        """
        return node_name in self.selected

    """The following properties are used to get information about the status of each command on each active node.
    It gives information about whether the command has been requested, is currently executing, has finished executing, or has failed."""

    def node_has_commands_of_status(
        self,
        node_name: str,
        status: CommandDistributionStatus | list[CommandDistributionStatus],
        check_main_command: bool = False,
        return_for_unselected: bool = True,
    ) -> bool:
        """Returns whether all commands on a node have a certain status.

        Args:
            node_name (str): The name of the node.
            status (CommandDistributionStatus): The status to check for.
            check_main_command (bool, optional): Whether to only check the main command. Defaults to False.
            return_for_unselected (bool, optional): Whether to return True if the node is not selected. Defaults to True.

        Returns:
            bool: Whether all commands on a node have a certain status.
        """

        if not self.node_selected(node_name):
            return return_for_unselected

        if check_main_command:
            return self[node_name][self.get_main_command_uuid(node_name)] in to_list(
                status
            )

        return all(
            [node_status in to_list(status) for node_status in self[node_name].values()]
        )

    @property
    def active(self) -> list[str]:
        return self.nodes_with_main_command_status(
            [
                CommandDistributionStatus.SEND,
                CommandDistributionStatus.STARTED,
            ]
        )

    def node_finished(self, node_name: str) -> bool:
        """Returns whether all commands on a node have finished.

        Args:
            node_name (str): The name of the node.

        Returns:
            bool: Whether all commands on a node have finished.
        """
        return self.node_has_commands_of_status(
            node_name=node_name,
            status=CommandDistributionStatus.FINISHED,
            check_main_command=False,
            return_for_unselected=True,
        )

    def node_finished_failed_excluded(self, node_name: str) -> bool:
        """Returns whether all commands on a node have finished or failed or are excluded.

        Args:
            node_name (str): The name of the node.

        Returns:
            bool: Whether all commands on a node have finished or failed or are excluded.
        """
        return self.node_has_commands_of_status(
            node_name,
            [
                CommandDistributionStatus.FINISHED,
                CommandDistributionStatus.FAILED,
                CommandDistributionStatus.EXCLUDED,
            ],
            check_main_command=False,
            return_for_unselected=True,
        )

    def finished_or_failed_or_excluded(self) -> bool:
        """Returns whether all commands on all nodes have finished or failed or are excluded.

        Returns:
            bool: Whether all commands on all nodes have finished or failed or are excluded.
        """
        return all(
            [
                self.node_finished_failed_excluded(node_name)
                for node_name in self.selected
            ]
        )

    def nodes_with_main_command_status(
        self, status: CommandDistributionStatus | list[CommandDistributionStatus]
    ) -> list[str]:
        """Returns a list of nodes that have a certain status for their main command.

        Args:
            status (CommandDistributionStatus): The status to check for.

        Returns:
            list[str]: A list of nodes that have a certain status for their main command.
        """
        return [
            node_name
            for node_name in self.selected
            if self[node_name][self.get_main_command_uuid(node_name)] in to_list(status)
        ]

    def command_finished_or_failed(self, command_uuid: str) -> bool:
        """Returns whether all commands on all nodes have finished or failed.

        Args:
            command_uuid (str): The UUID of the command.

        Returns:
            bool: Whether all commands on all nodes have finished or failed.
        """
        command_status = []
        for node_name in self.selected:
            try:
                command_status.append(self[node_name][command_uuid])
            except KeyError:
                pass
        return all(
            [
                status
                in [
                    CommandDistributionStatus.FINISHED,
                    CommandDistributionStatus.FAILED,
                ]
                for status in command_status
            ]
        )

    def __repr__(self) -> str:
        """Returns a string representation of the table."""
        vis = {}
        for node_name, commands in self.items():
            vis[node_name] = {} if commands is not None else None
            if commands is not None:
                for command_uuid, status in commands.items():
                    vis[node_name][command_uuid] = status.name

        return json.dumps(vis, indent=3, ensure_ascii=False)


class StatusUpdateHandler(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def handle_status_update(
        self,
        distribution: Distribution,
        status_update: StatusUpdate,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        """Handles a status update for a command in an instruction.

        Args:
            instruction (Instruction): The instruction to handle the status update for.
            status_update (StatusUpdate): The status update to handle.
            topology (Topology): The server topology register.
            resource_manager (dict): The server resource register.
        """
        pass


class Distribution(Instruction, Transferable, is_base_type=True):
    def __init__(
        self,
        commands: list[Command],
        status_handler: list[StatusUpdateHandler] | None = None,
        remove_instruction_resource_entry: bool = True,
        error_handler: DistributionErrorHandler | None = None,
        set_flag_after_execution: str | list[str] | None = None,
        simultaneous_execution: int = 0,
        predecessor: Instruction | None = None,
        remove_flag_after_execution: str | list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            predecessor=predecessor,
            remove_instruction_resource_entry=remove_instruction_resource_entry,
            **kwargs,
        )
        self.commands = commands
        self.successor: list[Distribution] = []
        self.remove_instruction_resource_entry = remove_instruction_resource_entry
        self.simultaneous_execution = simultaneous_execution
        self.status_handler: list[StatusUpdateHandler] = status_handler or []
        self.error_handler = error_handler
        self.set_flag_after_execution = (
            to_list(set_flag_after_execution) if set_flag_after_execution else []
        )
        self.remove_flag_after_execution = (
            to_list(remove_flag_after_execution) if remove_flag_after_execution else []
        )

        self.dist_table: DistributionStatusTable = DistributionStatusTable()

        self.uuid: str | None = None
        self.on_init_hooks: list[callable] = []
        self.on_finish_hooks: list[callable] = []

    def get_command_by_uuid(self, uuid: str) -> Command | None:
        for command in self.commands:
            for sub_command in command.get_command_tree().values():
                if sub_command.uuid == uuid:
                    return sub_command
        return None

    def init_distribution_table(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

    def on_init(self, topology: Topology, resource_manager: ResourceManager):
        pass

    def register_on_init_hook(self, hook: callable) -> Distribution:
        self.on_init_hooks.append(hook)
        return self

    def _on_init(self, topology: Topology, resource_manager: ResourceManager):
        """Internal method for initializing the instruction

        Args:
            topology (Topology): The topology register of the server
            resource_manager (ResourceManager): The resource register of the server
        """

        # set uuid
        self._set_uuid()

        # Set the instruction status to BOOTING
        self.status = InstructionStatus.BOOTING

        # Call the on_init method of the subclass
        self.on_init(topology, resource_manager)

        # execute init hooks
        for hook in self.on_init_hooks:
            hook(self, topology, resource_manager)

        # set uuids of commands
        for command in self.commands:
            command.init_uuid()

        # initialize the lifecycle manager
        self.init_distribution_table(topology, resource_manager)

        # If the instruction is in the COMPLETED state, return
        if self.status is InstructionStatus.COMPLETED:
            return

        topology.add_lifecycle(self)

        for command in self.commands:
            for sub_commands in command.get_command_tree(True).values():
                sub_commands.on_init_server_side(
                    topology, resource_manager, self.dist_table.selected
                )

        self.status = InstructionStatus.EXECUTION

    def infer_command(
        self, node_name: str, topology: Topology, resource_manager: ResourceManager
    ) -> Command | None:
        """Infer the command to send to a node.

        This method is called by the server if a node requests a command. It returns the command to send to the node.

        Args:
            node_name (str): The UUID of the node to infer the command for
            topology (Topology): The topology register of the server
            resource_manager (ResourceManager): The resource register of the server

        Returns:
            Command | None: The command to send to the node if the instruction is in the EXECUTION state, None otherwise
        """

        if self.status is InstructionStatus.CREATED:
            # if the instruction is in the CREATED state, initialize it
            self._on_init(topology=topology, resource_manager=resource_manager)

        elif self.status is InstructionStatus.EXECUTION:
            # check if the node already received its command
            if self.dist_table.node_has_commands_of_status(
                node_name,
                CommandDistributionStatus.UNREQUESTED,
                check_main_command=True,
                return_for_unselected=False,
            ):
                # check if simultaneous execution is enabled and if there are already too many active nodes
                if not (
                    self.simultaneous_execution > 0
                    and len(self.dist_table.active) >= self.simultaneous_execution
                ):
                    # set the status of commands of the node to SEND
                    self.dist_table.set_for_all_node_commands(
                        node_name, CommandDistributionStatus.SEND
                    )

                    # get the command uuid for the node
                    command_uuid = self.dist_table.get_main_command_uuid(node_name)

                    # get the command using the uuid
                    command = self.get_command_by_uuid(command_uuid)

                    # return the command
                    return command.node_specific_modification(
                        self.dist_table, node_name=node_name
                    )

        return None

    def handle_status_update(
        self,
        status_update: StatusUpdate,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        """Function to handle status updates

        Args:
            status_update (StatusUpdate): The status update to handle.
        """

        for handler in self.status_handler:
            handler.handle_status_update(
                self, status_update, topology, resource_manager
            )

        status = CommandDistributionStatus(status_update.status)
        node_name = status_update.node_name

        # if the node is excluded, ignore the status update
        if not self.dist_table.node_selected(node_name):
            logging.warning(
                f"Received status update for command {status_update.command_uuid} on node {node_name} that is excluded from the distribution."
            )
            return

        # if the execution of the command failed, handle the failure
        if status == CommandDistributionStatus.FAILED:
            # TODO: handle failure
            pass

        try:
            # set the status of the command
            self.dist_table[node_name][status_update.command_uuid] = status

            node_name = status_update.node_name
            command_uuid = status_update.command_uuid

            # if the command is finished, call the on_client_finish_server_side method of the command
            if status == CommandDistributionStatus.FINISHED:
                self.get_command_by_uuid(command_uuid).on_client_finish_server_side(
                    topology=topology,
                    resource_manager=resource_manager,
                    node_name=node_name,
                    execution_response=status_update.response,
                    instruction_uuid=self.uuid,
                )

            # Notify all watchers about the status update
            resource_manager.watcher.notify_all(
                StatusUpdateNotification(status_update=status_update)
            )

            # if a node is finished, add and remove flags
            if self.dist_table.node_finished(node_name):
                for flag in self.set_flag_after_execution:
                    topology.set_flag_of_nodes([node_name], flag)
                for flag in self.remove_flag_after_execution:
                    topology.remove_flag_of_nodes([node_name], flag)

            if self.dist_table.command_finished_or_failed(command_uuid):
                command = self.get_command_by_uuid(command_uuid)
                command.all_clients_finished_server_side(
                    topology=topology,
                    resource_manager=resource_manager,
                    instruction_uuid=self.uuid,
                )

                resource_manager.watcher.notify_all(
                    notification=CommandFinishedNotification(
                        command_uuid=status_update.command_uuid
                    )
                )

        except KeyError:
            logging.warning(
                f"Received status update for command {status_update.command_uuid} on node {node_name} that is not part of the instruction."
            )

    def handle_topology_change(
        self, node_name: str, topology: Topology, resource_manager: ResourceManager
    ) -> None:
        """Function to handle topology changes

        Args:
            node_name (str): The name of the node that changed.
        """
        raise NotImplementedError(
            "This method must be implemented by the subclass of InstructionLifecycle"
        )

    def on_finish(
        self, topology: Topology, resource_manager: ResourceManager
    ) -> Instruction | list[Instruction] | None:
        topology.remove_lifecycle(self)
        return super().on_finish(topology, resource_manager)

    def _add_commands_to_wrapped(
        self, pre_commands: list[Command], post_commands: list[Command]
    ):
        """Internal method to add commands to the wrapped object

        Args:
            pre_commands (list[Command]): The commands to add before the wrapped object
            post_commands (list[Command]): The commands to add after the wrapped object
        """
        # Add the pre_commands to the wrapped object
        self.commands = [
            SequentialCommand(pre_commands + [cmd] + post_commands)
            for cmd in self.commands
        ]


class OpenDistribution(Distribution):
    def __init__(
        self,
        command: Command,
        status_handler: list[StatusUpdateHandler] | None = None,
        error_handler: DistributionErrorHandler | None = None,
        set_flag_after_execution: str | list[str] | None = None,
        remove_flag_after_execution: str | list[str] | None = None,
        simultaneous_execution: int = 0,
        predecessor: Instruction | None = None,
        remove_instruction_resource_entry: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            commands=[command],
            set_flag_after_execution=set_flag_after_execution,
            remove_flag_after_execution=remove_flag_after_execution,
            status_handler=status_handler,
            error_handler=error_handler,
            predecessor=predecessor,
            remove_instruction_resource_entry=remove_instruction_resource_entry,
            simultaneous_execution=simultaneous_execution,
            **kwargs,
        )

    def init_distribution_table(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        # select all current nodes (if new nodes are added, they will be added to the instruction)
        current_nodes = topology.online_clients(True)
        # open lifecycles can only have one command
        command = self.commands[0]
        # add the nodes to the distribution table
        for node_name in current_nodes:
            self.dist_table.add_node(node_name, command_uuids=[command.uuid])

    def handle_topology_change(
        self, node_name: str, topology: Topology, resource_manager: ResourceManager
    ) -> None:
        # in an open lifecycle, new nodes are added to the instruction
        if node_name not in self.dist_table:
            self.dist_table.add_node(node_name, command_uuids=[self.commands[0].uuid])
        # if a node is removed from the topology, remove it from the instruction
        if topology.get_client_by_name(node_name).status == NodeStatus.OFFLINE:
            del self.dist_table[node_name]


class ClosedDistribution(Distribution):
    def __init__(
        self,
        commands: list[Command] | Command,
        selector: Selector | None = None,
        status_handler: list[StatusUpdateHandler] | None = None,
        error_handler: DistributionErrorHandler | None = None,
        set_flag_after_execution: str | list[str] | None = None,
        remove_flag_after_execution: str | list[str] | None = None,
        predecessor: Instruction | None = None,
        simultaneous_execution: int = 0,
        remove_instruction_resource_entry: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            commands=to_list(commands),
            set_flag_after_execution=set_flag_after_execution,
            remove_flag_after_execution=remove_flag_after_execution,
            status_handler=status_handler,
            error_handler=error_handler,
            remove_instruction_resource_entry=remove_instruction_resource_entry,
            predecessor=predecessor,
            simultaneous_execution=simultaneous_execution,
            **kwargs,
        )
        self.selector = selector or AllSelector()

    def _check_for_finish(self, topology: Topology, resource_manager: ResourceManager):
        # if all nodes are finished, call the on_finish method
        if self.dist_table.finished_or_failed_or_excluded():
            self._on_finish(topology, resource_manager)

    def init_distribution_table(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        # use the selector to select nodes for the instruction
        selection = self.selector.selection(topology, self.commands)

        for node_name, command_uuid in selection.items():
            # if a node is not selected, add it to the distribution table with the excluded status
            if command_uuid is None:
                self.dist_table.add_node(node_name)
            else:
                # get the command uuids of the selected command
                command = self.get_command_by_uuid(command_uuid)
                uuids = list(command.get_command_tree(True).keys())
                self.dist_table.add_node(node_name, command_uuids=uuids)

        # if no nodes are selected, set the status to COMPLETED
        self._check_for_finish(topology, resource_manager)

    def handle_status_update(
        self,
        status_update: StatusUpdate,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        super().handle_status_update(
            status_update=status_update,
            topology=topology,
            resource_manager=resource_manager,
        )

        self._check_for_finish(topology, resource_manager)

    def handle_topology_change(
        self, node_name: str, topology: Topology, resource_manager: ResourceManager
    ) -> None:
        # in a fixed lifecycle, new nodes are not added to the instruction
        if node_name not in self.dist_table:
            return
        # if a node is removed from the topology, remove it from the instruction
        if topology.get_client_by_name(node_name).status == NodeStatus.OFFLINE:
            self.dist_table[node_name] = None
            self._check_for_finish(topology, resource_manager)
