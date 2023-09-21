from __future__ import annotations

from typing import Any, Optional, Dict, TYPE_CHECKING
from uuid import uuid4
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from ...topology.node import Node
    from .. import StatusTable
from ...topology import TopologyRegister
from ..decorators import return_execution_status_and_result
from ...common import ExecutionResponse, Transferable
from ...resources import ResourceRegister


class CommandMeta(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        """
        Creates a new CommandMeta object.

        Args:
            name (str): The name of the class to create.
            bases (tuple): The base classes of the class to create.
            attrs (dict): The attributes of the class to create.

        Returns:
            The new CommandMeta object.
        """
        # Wrap the `execute` method with the `return_execution_status()` function
        if "execute" in attrs and not hasattr(attrs["execute"], "__wrapped__"):
            attrs["execute"] = return_execution_status_and_result(attrs["execute"])

        for meth_name, meth in cls.__class__.__dict__.items():
            if getattr(meth, "__isabstractmethod__", False):
                raise TypeError(
                    f"Can't create new class {name} with no abstract classmethod {meth_name} redefined in the metaclass"
                )

        # Create the new CommandMeta object
        return super().__new__(cls, name, bases, attrs, **kwargs)


class Command(Transferable, is_base_type=True, metaclass=CommandMeta):
    """
    The Command interface declares a method for executing a command.
    """

    def __init__(
        self,
        *,
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.set_node(node)
        self.uuid = uuid

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
    def node_rr(self) -> ResourceRegister:
        """
        Returns the ResourceRegister of the node attribute of the Command object.

        Returns:
            The ResourceRegister of the node attribute of the Command object.
        """
        return self.node.resource_register

    def get_command_tree(self, flatten: bool = True) -> Dict[str, Command]:
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

    @abstractmethod
    def execute(self) -> ExecutionResponse | None:
        """Abstract execute command that is called to perform the actions specific to the command

        Returns
        -------
        Any
            Different returns possible

        Raises
        ------
        NotImplementedError
            This method has to be implemented in order to have a functioning command
        """
        raise NotImplementedError("Please Implement this method")

    def node_specific_modification(
        self, status_register: dict[str, "StatusTable"], node_uuid: str
    ) -> Command:
        """
        This method is called on the server to modify the command to be executed on the clients. It is used to add node specific information to the command after initialization.
        This is necessary because the command is initialized on the server but might need information that is only available during runtime.

        Args:
            status_register (dict): The dictionary of all the commands in the instruction and their status.
            node_uuid (str): The uuid of the node the command is executed on.

        Returns:
            The modified command.
        """
        return self

    def on_client_finish_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        node_uuid: str,
        execution_response: ExecutionResponse,
        instruction_uuid: str,
    ):
        """
        This method is called after the execution of the command is finished.

        Args:
            topology_register (TopologyRegister): The topology register of the instruction.
            resource_register (ResourceRegister): The resource register of the instruction.
            node_uuid (str): The uuid of the node the command is executed on.
            execution_response (ExecutionResponse): The response of the execution.
            instruction_uuid (str): The uuid of the instruction.
        """
        pass

    def all_clients_finished_server_side(
        self,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
        instruction_uuid: str,
    ) -> None:
        """
        This method is called on the server to check if all clients have finished executing the command.

        Args:
            status_register (dict): The dictionary of all the commands in the instruction and their status.

        Returns:
            True if all clients have finished executing the command, False otherwise.
        """
        pass
