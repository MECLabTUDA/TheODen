from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...topology import TopologyRegister
from ...common import Transferable, StatusUpdate
from ..status import NodeCommandStatus
from ...resources.resource import ResourceRegister
from ..commands.command import Command
from ...watcher import CommandFinishedNotification


if TYPE_CHECKING:
    from .instruction import Instruction


class StatusHandler(ABC, Transferable, is_base_type=True):
    @abstractmethod
    def handle_status_update(
        self,
        instruction: "Instruction",
        status_update: StatusUpdate,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> None:
        """Handles a status update for a command in an instruction.

        Args:
            instruction (Instruction): The instruction to handle the status update for.
            status_update (StatusUpdate): The status update to handle.
            topology_register (TopologyRegister): The server topology register.
            resource_register (dict): The server resource register.
        """
        pass


class BaseHandler(StatusHandler, Transferable):
    def handle_status_update(
        self,
        instruction: "Instruction",
        status_update: StatusUpdate,
        topology_register: TopologyRegister,
        resource_register: ResourceRegister,
    ) -> None:
        # update the status table

        instruction.command_level_status[status_update.command_uuid].table[
            status_update.node_uuid
        ] = NodeCommandStatus(status_update.status)

        # check if command is finished
        if (
            instruction.command_level_status[status_update.command_uuid].table[
                status_update.node_uuid
            ]
            is NodeCommandStatus.FINISHED
        ):
            # if command is finished, check if the wrapped object is a command
            if isinstance(instruction.wrapped_object, Command):
                # if the wrapped object is a command, call the on_finish method
                instruction.wrapped_object.get_command_tree(True)[
                    status_update.command_uuid
                ].on_client_finish_server_side(
                    topology_register=topology_register,
                    resource_register=resource_register,
                    node_uuid=status_update.node_uuid,
                    execution_response=status_update.response,
                    instruction_uuid=instruction.uuid,
                )

        if all(
            [
                status is NodeCommandStatus.FINISHED
                for status in instruction.command_level_status[
                    status_update.command_uuid
                ].table.values()
            ]
        ):
            instruction.wrapped_object.get_command_tree(True)[
                status_update.command_uuid
            ].all_clients_finished_server_side(
                topology_register=topology_register,
                resource_register=resource_register,
                instruction_uuid=instruction.uuid,
            )

            resource_register.watcher.notify_all(
                notification=CommandFinishedNotification(
                    command_uuid=status_update.command_uuid
                )
            )

        # check if all commands on a node are finished
        if all(
            [
                status.table[status_update.node_uuid] is NodeCommandStatus.FINISHED
                for status in instruction.command_level_status.values()
            ]
        ):
            instruction.node_level_status.table[
                status_update.node_uuid
            ] = NodeCommandStatus.FINISHED
