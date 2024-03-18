from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from ..common import (
    ExecutionResponse,
    ServerRequestError,
    StatusUpdate,
    UnauthorizedError,
)
from ..operations import PullCommandRequest, ServerRequest


class ClientInterface(ABC):
    def __init__(self, command_queue: list[dict], ping_interval: float = 1.0):
        self.command_queue = command_queue
        self.ping_interval = ping_interval

    @abstractmethod
    def send_status_update(self, status_update: StatusUpdate) -> None:
        pass

    @abstractmethod
    def send_server_request(self, request: ServerRequest) -> ExecutionResponse:
        pass

    async def start_request_loop(self, stop_event) -> None:
        while not stop_event.is_set():
            # Wait for 1 second before making the next _pull() call
            await asyncio.sleep(self.ping_interval)

            # Call the _pull method to get a server request from the server
            self._pull()

    def _pull(self):
        """Pulls a command from the server and adds it to the command queue."""
        # Make a GET serverrequests to the server to get the next command
        try:
            response = self.send_server_request(PullCommandRequest())

            # If the response contains a command, parse it into a CommandModel object and add it to the command queue
            if response.data:
                self.command_queue.append(response.get_data())
        except ServerRequestError as e:
            return
        except UnauthorizedError as e:
            logging.error("UnauthorizedError: %s", e)
