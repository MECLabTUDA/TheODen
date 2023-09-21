from typing import TYPE_CHECKING


from ..operations.status import NodeCommandStatus
from ..common.typing import ExecutionResponse, StatusUpdate

if TYPE_CHECKING:
    from . import Command


def _send_status(
    command: "Command",
    status: NodeCommandStatus,
    block: bool = False,
    command_response: ExecutionResponse | None = None,
):
    # If block is False, prepare to send status update
    if not block:
        # Send status update to node
        command.node.send_status_update(
            StatusUpdate(
                node_uuid=command.node.uuid,
                command_uuid=command.uuid,
                status=status._value_,
                datatype=type(command).__name__,
                response=command_response,
            )
        )


def _return_execution_status(func, with_returned_value: bool):
    """
    A decorator that sends status updates to the parent node for a command's execution.

    Args:
        func (function): The function to decorate.

    Returns:
        The decorated function.
    """

    def wrapper(*args, **kwargs):
        # Get the command object from the first argument
        command: "Command" = args[0]

        # Get the `block_status` flag from the kwargs
        block_status = kwargs.get("block_status", False)

        # Send a "started" status update to the server
        _send_status(command, NodeCommandStatus.STARTED, block_status)

        try:
            # Call the original function
            result: ExecutionResponse | None = func(*args, **kwargs)
        except Exception as e:
            # Send a "failed" status update to the server if an exception occurs
            _send_status(command, NodeCommandStatus.FAILED, block_status)
            raise e
        else:
            # Send a "finished" status update to the server if the function completes successfully
            _send_status(
                command,
                NodeCommandStatus.FINISHED,
                block_status,
                command_response=result if with_returned_value else None,
            )
            return result

    # Set the `__wrapped__` attribute to True to indicate that the function has been decorated
    wrapper.__wrapped__ = True

    return wrapper


def return_execution_status(func):
    return _return_execution_status(func, False)


def return_execution_status_and_result(func):
    return _return_execution_status(func, True)
