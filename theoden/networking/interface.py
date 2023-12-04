from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio

from ..common import (
    StatusUpdate,
    ServerRequestError,
    UnauthorizedError,
    ExecutionResponse,
)

from ..operations import ServerRequest, PullCommandRequest


class NodeInterface(ABC):
    @abstractmethod
    def send_status_update(self, status_update: StatusUpdate) -> any:
        pass

    @abstractmethod
    def send_server_request(self, request: ServerRequest) -> ExecutionResponse:
        pass

    async def start_request_loop(self) -> None:
        while True:
            # Wait for 1 second before making the next _pull() call
            await asyncio.sleep(self.ping_interval)

            # Call the _pull method to get a server request from the server
            self._pull()

    def _pull(self):
        # Make a GET serverrequests to the server to get the next command
        try:
            response = self.send_server_request(PullCommandRequest())

            # If the response contains a command, parse it into a CommandModel object and add it to the command queue
            if response.data:
                self.command_queue.append(response.get_data())
        except ServerRequestError as e:
            return
        except UnauthorizedError as e:
            print("Unauthorized")
