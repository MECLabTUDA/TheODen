from __future__ import annotations

import threading

from ..common import *
from ..operations import *
from ..resources.resource import ResourceManager
from ..security.operation_protection import OperationBlackList, OperationWhiteList
from .topology import Topology

import logging
logger = logging.getLogger(__name__)

class ActionThread(threading.Thread):
    def __init__(
        self, action: Action, resource_manager: ResourceManager, topology: Topology
    ):
        """A thread that executes an action.

        Args:
            action (Action): The action to be executed.
            resource_manager (ResourceManager): The resource manager of the server.
            topology (Topology): The topology of the federated learning system.
        """

        super().__init__()
        self.action = action
        self.resource_manager = resource_manager
        self.topology = topology

    def run(self) -> None:
        self.action(self.topology, self.resource_manager)


class OperationManager(Transferable, is_base_type=True):
    def __init__(
        self,
        open_distribution: OpenDistribution | None = None,
        operations: (
            list[Action | ClosedDistribution | InstructionBundle | Condition] | None
        ) = None,
        constant_conditions: list[Condition] | None = None,
        operation_protection: OperationWhiteList | OperationBlackList | None = None,
    ) -> None:
        """A class that manages the operations of a server.

        Args:
            open_distribution (OpenDistribution, optional): The open distribution to be used by the server. Defaults to None.
            operations (list[Action | ClosedDistribution | InstructionBundle | Condition], optional): The operations to be executed by the server. Defaults to None.
            constant_conditions (list[Condition], optional): The constant conditions to be checked by the server. Defaults to None.
        """

        # Initialize the open distribution as None or with the given open distribution
        self.open_distribution = open_distribution

        # Initialize the operations as an empty list or with the given operations
        self.operations: list[
            Action | ClosedDistribution | InstructionBundle | Condition
        ] = (operations if operations else [])

        self.operation_protection = operation_protection

        # check if all operations are allowed
        if (
            self.operation_protection is not None
            and not self.operation_protection.allows(
                self.operations + [self.open_distribution] + constant_conditions
            )
        ):
            raise ForbiddenOperationError(
                f"OperationManager initialization is not allowed"
            )

        self.constant_conditions = constant_conditions
        self.action_thread: ActionThread | None = None
        self.action_queue: list[Action] = []

    def process_status_update(
        self,
        status_update: StatusUpdate,
        topology: Topology,
        resource_manager: ResourceManager,
    ) -> None:
        if self.open_distribution is not None:
            # check if the status update is for the open distribution
            if (
                self.open_distribution.get_command_by_uuid(status_update.command_uuid)
                is not None
            ):
                self.open_distribution.handle_status_update(
                    status_update, topology, resource_manager
                )
                return

        # check if the status update is for a running operation
        self.operations[0].handle_status_update(
            status_update, topology, resource_manager
        )

    @property
    def operation_head(self) -> Instruction | InstructionBundle | Condition | None:
        """Returns the operation head, which is the first operation in the operations list.
        If the operation head is an instruction set, unpack it and add it to the operations list as separate instructions.

        Returns:
            Instruction | InstructionBundle | Condition | None: The operation head. If there are no operations, returns None.
        """
        if self.operations and isinstance(self.operations[0], InstructionBundle):
            # if the next operation is an instruction set, we need to unpack it and add it to the operations list as separate instructions
            self.operations = self.operations[0].instructions + self.operations[1:]
        return self.operations[0] if self.operations else None

    def get_command(
        self, client_name: str, topology: Topology, resource_manager: ResourceManager
    ) -> Command | None:
        """Returns the command to be executed by the client.

        Args:
            client_name (str): The name of the client.
            topology (Topology): The topology of the federated learning system.
            resource_manager (ResourceManager): The resource manager of the server.

        Returns:
            Command | None: The command to be executed by the client.
        """

        """Handle Constant Conditions"""
        # Check if all constant conditions are resolved. If they are not, return an empty dictionary
        for condition in self.constant_conditions:
            if not condition.resolved(
                topology=topology, resource_manager=resource_manager
            ):
                return None

        """Handle Action Thread"""
        if self.action_thread is not None:
            if self.action_thread.is_alive():
                return None
            else:
                self.action_thread = None

        """Handle Open Distribution"""
        if self.open_distribution is not None:
            # check if there is an uncompleted command for the open distribution
            if self.open_distribution.client_started_but_unfinished(client_name):
                return None

            try:
                command = self.open_distribution.infer_command(
                    client_name, topology=topology, resource_manager=resource_manager
                )

            except TooManyCommandsExecutingException as e:
                logger.debug("Too many commands executing exception")
                return None

            except NoCommandException as e:
                #print(f"No command exception {e.message}")
                command = None

            # If the open distribution returns a command, return it
            if command is not None:
                return command

        """Handle Conditions"""
        # If the operation head is a condition, check if it is resolved. If it is, pop it from the operations list and continue
        while isinstance(self.operation_head, Condition):
            # If the condition is not resolved, return an empty dictionary
            if not self.operation_head.resolved(
                topology=topology, resource_manager=resource_manager
            ):
                return None
            # If the condition is resolved, pop it from the operations list and continue
            else:
                self.operations.pop(0)

        """Handle Instructions"""
        if isinstance(self.operation_head, Instruction):
            if self.operation_head.status is InstructionStatus.COMPLETED:
                # If the instruction is completed, check if it has a successor. If it does, add it to the operations list
                successor = self.operation_head.successor
                self.operations.pop(0)
                if successor is not None and isinstance(successor, list):
                    self.operations = successor + self.operations

            # Infer the command from the instruction and return it as a dictionary
            if not self.operation_head:
                return None

        """Handle Distributions"""
        # If the operation head is an instruction, check if it is completed. If it is, pop it from the operations list and continue
        if isinstance(self.operation_head, ClosedDistribution):
            try:
                command = self.operation_head.infer_command(
                    client_name=client_name,
                    topology=topology,
                    resource_manager=resource_manager,
                )

            except (NoCommandException, TooManyCommandsExecutingException) as e:
                return None

            return command

        # Handle Actions
        elif isinstance(self.operation_head, Action):
            if self.operation_head.status is InstructionStatus.CREATED:
                self.action_thread = ActionThread(
                    action=self.operation_head,
                    resource_manager=resource_manager,
                    topology=topology,
                )
                self.action_thread.start()

            return None
