from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4
from abc import ABC, abstractmethod

from ...topology import Topology
from ...common import Transferable, StatusUpdate, ExecutionResponse
from ...resources import ResourceManager
from ..status import CommandExecutionStatus

if TYPE_CHECKING:
    from ...topology.node import Node
    from .. import DistributionStatusTable


class Command(Transferable, is_base_type=True):
    """
    The Command interface declares a method for executing a command.
    """

    def __init__(
        self,
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        self.uuid = uuid
        self.node: "Node" | None = None

    def init_uuid(self) -> Command:
        """
        Initializes a UUID for the current Command object and recursively sets UUIDs for all its subcommands.

        Returns:
            The current Command object with the UUID and any updated initialization parameters.
        """
        # Generate a UUID for the current Command object
        self.uuid = str(uuid4())

        # Update the initialization parameters with the UUID, if they exist
        self.add_initialization_parameter(uuid=self.uuid, _overwrite=True)

        # Recursively call `init_uuid()` on all subcommands
        for subcommand in self.get_command_tree(flatten=False)["subcommands"]:
            subcommand_object = list(subcommand["main"].values())[0]
            subcommand_object.init_uuid()

        # Return the current Command object with the UUID and any updated initialization parameters
        return self

    def set_node(self, node: "Node") -> Command:
        """
        Sets the node attribute of the Command object.

        Args:
            node (Node): The node to set as the attribute.

        Returns:
            The current Command object with the node attribute set.
        """
        # Set the node attribute of the Command object
        self.node = node

        # Return the current Command object with the node attribute set
        return self

    @property
    def node_rm(self) -> ResourceManager:
        """
        Returns the ResourceManager of the node attribute of the Command object.

        Returns:
            The ResourceManager of the node attribute of the Command object.
        """
        return self.node.resources

    def get_command_tree(self, flatten: bool = True) -> dict[str, Command]:
        """
        Returns a nested dictionary representing the Command object and all its subcommands.

        Args:
            flatten (bool): Whether to flatten the dictionary or keep subcommands nested. Default is True.

        Returns:
            A dictionary representing the Command object and all its subcommands.
        """
        # Initialize the dictionary with the current Command object as the main command
        commands = {"main": {self.uuid: self}}
        subcommands = []

        # Iterate through all attributes of the current object
        for _, attribute in self.__dict__.items():
            # If the attribute is a Command object, recursively call `get_command_tree()` on it and append the result
            # to `subcommands`
            if isinstance(attribute, Command):
                subcommands.append(attribute.get_command_tree(flatten))
            # If the attribute is a list, iterate through its elements and append any Command objects to `subcommands`
            elif isinstance(attribute, list):
                for attr_of_list in attribute:
                    if isinstance(attr_of_list, Command):
                        subcommands.append(attr_of_list.get_command_tree(flatten))
            # If the attribute is a dict, iterate through its values and append any Command objects to `subcommands`
            elif isinstance(attribute, dict):
                for attr_of_dict in attribute.values():
                    if isinstance(attr_of_dict, Command):
                        subcommands.append(attr_of_dict.get_command_tree(flatten))

        # If `flatten` is True, flatten the dictionary by merging all subcommands into the main dictionary
        if flatten:
            commands = commands["main"]
            for subcommand in subcommands:
                commands.update(subcommand)
        # Otherwise, keep subcommands nested under the "subcommands" key
        else:
            commands["subcommands"] = subcommands

        # Return the dictionary representing the Command object and all its subcommands
        return commands

    def on_init_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        selected_nodes: list[str],
    ) -> None:
        pass

    def node_specific_modification(
        self, distribution_table: "DistributionStatusTable", node_name: str
    ) -> Command:
        """
        This method is called on the server to modify the command to be executed on the clients. It is used to add node specific information to the command after initialization.
        This is necessary because the command is initialized on the server but might need information that is only available during runtime.

        Args:
            status_register (dict): The dictionary of all the commands in the instruction and their status.
            node_name (str): The uuid of the node the command is executed on.

        Returns:
            The modified command.
        """
        return self

    @abstractmethod
    def execute(self) -> ExecutionResponse | None:
        """Abstract execute command that is called to perform the actions specific to the command

        Returns:
            ExecutionResponse | None: The response of the execution.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Please Implement this method")

    def __call__(self, *args, **kwargs):
        self.node.send_status_update(
            StatusUpdate(
                node_name=None,
                command_uuid=self.uuid,
                status=CommandExecutionStatus.STARTED,
                datatype=type(self).__name__,
            )
        )

        try:
            # Call the original function
            result: ExecutionResponse | None = self.execute()
        except Exception as e:
            # Send a "failed" status update to the server if an exception occurs
            self.node.send_status_update(
                StatusUpdate(
                    node_name=None,
                    command_uuid=self.uuid,
                    status=CommandExecutionStatus.FAILED,
                    datatype=type(self).__name__,
                )
            )
            raise e
        else:
            # Send a "finished" status update to the server if the function completes successfully
            self.node.send_status_update(
                StatusUpdate(
                    node_name=None,
                    command_uuid=self.uuid,
                    status=CommandExecutionStatus.FINISHED,
                    datatype=type(self).__name__,
                    response=result,
                )
            )
            return result

    def on_client_finish_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        node_name: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ) -> None:
        """
        This method is called after the execution of the command is finished.

        Args:
            topology (Topology): The topology register of the instruction.
            resource_manager (ResourceManager): The resource register of the instruction.
            node_name (str): The uuid of the node the command is executed on.
            execution_response (ExecutionResponse): The response of the execution.
            instruction_uuid (str): The uuid of the instruction.
        """
        pass

    def all_clients_finished_server_side(
        self,
        topology: Topology,
        resource_manager: ResourceManager,
        instruction_uuid: str,
    ) -> None:
        """
        This method is called on the server to check if all clients have finished executing the command.

        Args:
            topology (Topology): The topology register of the node.
            resource_manager (ResourceManager): The resource register of the node.
            instruction_uuid (str): The uuid of the instruction.
        """
        pass

    def __add__(self, other: Command) -> Command:
        """
        Adds two Command objects together.

        Args:
            other (Command): The Command object to add to the current Command object.

        Returns:
            The current Command object with the other Command object added to it.
        """
        from . import SequentialCommand

        return SequentialCommand([self, other])
