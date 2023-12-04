from __future__ import annotations

import pika


from typing import TYPE_CHECKING
import json
import asyncio
from functools import partial

from ..common import (
    Transferables,
    StatusUpdate,
    TransmissionStatusUpdate,
    ExecutionResponse,
    Transferable,
)
from .interface import NodeInterface
from .storage import FileStorageInterface
from ..operations import ServerRequest


if TYPE_CHECKING:
    from ..topology.server import Server


def transform_dict(data: dict, request_form: str) -> bytes:
    return json.dumps({"message_type": request_form, "data": data}).encode()


class ServerRequestResponse(Transferable, is_base_type=True):
    def __init__(self, request_uuid: str, response: dict) -> None:
        self.request_uuid = request_uuid
        self.response = ExecutionResponse(**response)


class ServerToMQInterface:
    def __init__(
        self,
        server: "Server",
        host: str,
        port: int,
        username: str,
        password: str,
        purge_queues: bool = True,
        **kwargs,
    ):
        self.server = server
        self.host = host
        self.port = port
        self.purge_queues = purge_queues

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=pika.PlainCredentials(username=username, password=password),
                heartbeat=0,
                virtual_host="theoden",
            )
        )
        self.channel = self.connection.channel()

        # get all clients from topology
        self.clients = self.server.topology.client_names

        for client in self.clients:
            self.channel.queue_declare(queue=f"client_queue_{client}")
            self.channel.queue_declare(queue=f"server_queue_{client}")

            self.channel.basic_consume(
                queue=f"server_queue_{client}",
                on_message_callback=partial(self._on_response, client_name=client),
                auto_ack=True,
            )

        if self.purge_queues:
            self.remove_all_messages()

    def init(self):
        self.channel.start_consuming()

    def _start(self):
        asyncio.run(self.start())

    async def start(self):
        print("Server started consuming")
        self.channel.start_consuming()

    def add_storage_interface(self, storage_interface: FileStorageInterface) -> None:
        print("Added storage interface")
        self.storage_interface = storage_interface

    def remove_all_messages(self):
        for client in self.clients:
            self.channel.queue_purge(queue=f"server_queue_{client}")
            self.channel.queue_purge(queue=f"client_queue_{client}")

    def _on_response(self, ch, method, props, body, client_name: str):
        """Callback function for when a response is received from the server.

        Args:
            ch (pika.channel.Channel): The channel.
            method (pika.spec.Basic.Deliver): The method.
            props (pika.spec.BasicProperties): The properties.
            body (bytes): The body.
        """

        # Get the response
        response = json.loads(body)

        from ..topology.topology import NodeStatus, Node, NodeType

        # check if serverrequest or status update
        try:
            node = self.server.topology.get_client_by_name(client_name)
            if node.status == NodeStatus.OFFLINE:
                self.server.topology.set_online(node.name)
        except KeyError:
            node = Node(
                node_name=client_name,
                node_type=NodeType.CLIENT,
                status=NodeStatus.ONLINE,
            )
            self.server.topology.add_node(node)
            node = self.server.topology.get_client_by_name(client_name)

        # try:
        if response["message_type"] == "ServerRequest":
            # check if client is registered

            # get message datatype
            request = (
                Transferables()
                .to_object(
                    response["data"],
                    "ServerRequest",
                    node_name=node.name,
                )
                .set_server(self.server)
            )

            response = ServerRequestResponse(
                request_uuid=request.uuid,
                response=self.server.process_server_request(request).dict(),
            )

            self.channel.basic_publish(
                exchange="",
                routing_key=f"client_queue_{client_name}",
                body=transform_dict(
                    response.dict(),
                    "ServerRequestResponse",
                ),
                properties=pika.BasicProperties(delivery_mode=2),
            )

        elif response["message_type"] == "StatusUpdate":
            # print("Received response:", response)

            status_update = TransmissionStatusUpdate(**response["data"])

            status_update.node_name = node.name

            downloaded_resource_manager = {}

            if status_update.contains_files():
                for file_name, file_uuid in status_update.response.files.items():
                    response = self.storage_interface.load_resource(file_uuid)
                    downloaded_resource_manager[file_name] = response

                    # remove file from storage
                    self.storage_interface.remove_resource(file_uuid)

            status_update = status_update.refill(downloaded_resource_manager)
            self.server.process_status_update(status_update)

        # except Exception as e:
        #     print("Error:", e)


class ClientToMQInterface(NodeInterface):
    def __init__(
        self,
        command_queue: list[dict],
        host: str,
        port: int,
        username: str,
        password: str,
        ping_interval: int = 1,
    ):
        self.command_queue = command_queue
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ping_interval = ping_interval
        self.node_name = username

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=pika.PlainCredentials(username=username, password=password),
                heartbeat=0,
                virtual_host="theoden",
            )
        )
        self.channel = self.connection.channel()

        # Define the response queue (client-specific)
        self.client_queue_name = "client_queue_" + self.node_name
        self.server_queue_name = "server_queue_" + self.node_name

        self.channel.queue_declare(queue=self.client_queue_name)
        self.channel.queue_declare(queue=self.server_queue_name)
        self.channel.basic_consume(
            queue=self.client_queue_name,
            on_message_callback=self._on_response,
            auto_ack=True,
        )

        # Purge the queues to remove any old messages
        self.channel.queue_purge(queue=self.client_queue_name)
        self.channel.queue_purge(queue=self.server_queue_name)

    def start(self):
        self.channel.start_consuming()

    def _pull(self):
        # Make a GET serverrequests to the server to get the next command
        from ..operations import PullCommandRequest

        self.send_server_request(PullCommandRequest())

    def add_storage_interface(self, storage_interface: FileStorageInterface) -> None:
        self.storage_interface = storage_interface

    def _on_response(self, ch, method, props, body):
        """Callback function for when a response is received from the server.

        Args:
            ch (pika.channel.Channel): The channel.
            method (pika.spec.Basic.Deliver): The method.
            props (pika.spec.BasicProperties): The properties.
            body (bytes): The body.
        """

        # Get the response
        response = json.loads(body)

        # print("Received response:", response)

        # get message datatype
        message_type = response["message_type"]

        if message_type == "ServerRequestResponse":
            # Get the request uuid
            response = Transferables().to_object(
                response["data"],
                ServerRequestResponse,
            )

            # Get the request
            # request = self._get_request_by_uuid(request_uuid)

            # Get the response
            if response.response.data:
                self.command_queue.append(response.response.data)

    def send_server_request(self, request: ServerRequest) -> any:
        self.channel.basic_publish(
            exchange="",
            routing_key=self.server_queue_name,
            body=transform_dict(request.dict(), "ServerRequest"),
        )

    def send_status_update(self, status_update: StatusUpdate) -> any:
        resource_uuids = {}

        if status_update.contains_files():
            resource_uuids.update(
                self.storage_interface.upload_resources(
                    status_update.response.get_files(),
                )
            )

        self.channel.basic_publish(
            exchange="",
            routing_key=self.server_queue_name,
            body=transform_dict(
                status_update.unload(resource_uuids).dict(), "StatusUpdate"
            ),
        )
