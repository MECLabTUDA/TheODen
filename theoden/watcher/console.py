from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedOK
from uvicorn import Config, Server
from pydantic import BaseModel

import time
from queue import SimpleQueue
import threading
import asyncio

from ..common import Transferable
from .metric_collector import MetricCollectionWatcher, Watcher
from .notifications import (
    InitializationNotification,
    TopologyChangeNotification,
    StatusUpdateNotification,
)

app = FastAPI()

# Keep track of connected WebSocket clients
connected_clients = set()

# Queue for communication between threads
message_queue = SimpleQueue()


async def send_message_to_clients(message):
    for client in connected_clients:
        await client.send_json(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        while websocket.application_state != WebSocketState.DISCONNECTED:
            # Send a message to all connected clients
            while not message_queue.empty():
                message = message_queue.get()
                await send_message_to_clients(message)

            # Sleep for a short duration to avoid high CPU usage
            await asyncio.sleep(0.2)

    except asyncio.CancelledError:
        pass
    except ConnectionClosedOK:
        pass
    finally:
        connected_clients.remove(websocket)


async def run_server():
    config = Config(app, host="localhost", port=3791, loop="asyncio")
    server = Server(config)
    await server.serve()


def start_server_in_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server())


def send_message_to_websocket(message):
    if len(connected_clients) == 0:
        return
    message_queue.put(message)


async def main():
    # Start the server in a separate thread
    thread = threading.Thread(target=start_server_in_thread, daemon=True)
    thread.start()


class StatusRow(BaseModel):
    node_name: str
    uuid: str
    type: str
    status: str
    expanded: bool = True
    subcommands: list[StatusRow] | None = None


class TheodenConsoleWatcher(MetricCollectionWatcher, Transferable):
    """Watcher to collect metrics from the framework and save them to Aim"""

    def __init__(self) -> None:
        super().__init__(
            notification_of_interest={
                TopologyChangeNotification: self._show_topology,
                StatusUpdateNotification: self._show_status,
                InitializationNotification: self._init,
            }
        )
        self.last_update_time = None
        self.wait_for = 0.0

    def _init(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.server_task = asyncio.run(main())

    def _show_topology(
        self, notification: TopologyChangeNotification, origin: Watcher | None = None
    ) -> None:
        msg = {
            "type": "topology_update",
            "update": [
                {
                    "node_name": node.name,
                    "status": node.status.name,
                    "flags": node.flags,
                }
                for node in notification.topology.clients
            ],
        }
        send_message_to_websocket(msg)

    def _show_status(
        self, notification: StatusUpdateNotification, origin: Watcher | None = None
    ) -> None:
        if (
            self.last_update_time is None
            or time.time() - self.last_update_time > self.wait_for
        ):
            dist = self.base_topology.operations[0]
            dist_table = dist.dist_table

            update = []

            for node_name, commands in dist_table.items():
                # convert commands dict into tuples
                if not commands:
                    continue
                commands = [
                    (dist.get_command_by_uuid(k), v) for k, v in commands.items()
                ]

                update.append(
                    StatusRow(
                        node_name=node_name,
                        uuid=commands[0][0].uuid,
                        type=type(commands[0][0]).__name__,
                        status=commands[0][1].name,
                        subcommands=[
                            StatusRow(
                                node_name=node_name,
                                uuid=command.uuid,
                                type=type(command).__name__,
                                status=status.name,
                            )
                            for command, status in commands[1:]
                        ],
                    ).dict()
                )

            self.last_update_time = time.time()
            send_message_to_websocket({"type": "status_update", "update": update})
