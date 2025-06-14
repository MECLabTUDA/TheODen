from __future__ import annotations

import asyncio
import json
import ssl
from functools import partial
from typing import TYPE_CHECKING

import pika

from ..common import (
    ExecutionResponse,
    StatusUpdate,
    Transferable,
    Transferables,
    TransmissionStatusUpdate,
)
from ..operations import PullCommandRequest, ServerRequest
from .interface import ClientInterface
from .storage import FileStorageInterface

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
        ssl_context: ssl.SSLContext | None = None,
        **kwargs,
    ):
        self.storage_interface = None
        self.server = server
        self.host = host
        self.port = port
        self.purge_queues = purge_queues
        self.ssl_context = ssl_context

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=pika.PlainCredentials(username=username, password=password),
                heartbeat=10,
                ssl_options=pika.SSLOptions(ssl_context, server_hostname=self.host)
                if ssl_context
                else None,
            )
        )
        self.channel = self.connection.channel()

        # get all clients from topology
        self.clients = self.server.topology.client_names

        for client in self.clients:
            self.channel.exchange_declare(
                exchange=f"{client}_exchange",
                exchange_type="direct",
                durable=True,
            )

            self.channel.queue_declare(queue=f"{client}_client_queue")
            self.channel.queue_declare(queue=f"{client}_server_queue")

            self.channel.queue_bind(
                exchange=f"{client}_exchange", queue=f"{client}_client_queue"
            )

            self.channel.queue_bind(
                exchange=f"{client}_exchange", queue=f"{client}_server_queue"
            )

            # Purge the queues to remove any old messages
            if self.purge_queues:
                self.channel.queue_purge(queue=f"{client}_client_queue")
                self.channel.queue_purge(queue=f"{client}_server_queue")

            self.channel.basic_consume(
                queue=f"{client}_server_queue",
                on_message_callback=partial(self._on_response, client_name=client),
                auto_ack=True,
            )

    def init(self):
        self.channel.start_consuming()

    def _start(self):
        asyncio.run(self.start())

    async def start(self):
        self.channel.start_consuming()

    def add_storage_interface(self, storage_interface: FileStorageInterface) -> None:
        self.storage_interface = storage_interface

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

        from ..topology.topology import Node, NodeStatus, NodeType

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
            # get message datatype
            request = (
                Transferables()
                .to_object(
                    response["data"],
                    "ServerRequest",
                    client_name=node.name,
                )
                .set_server(self.server)
            )

            server_response = self.server.process_server_request(request).dict()

            response = ServerRequestResponse(
                request_uuid=request.uuid,
                response=server_response,
            )

            self.channel.basic_publish(
                exchange=f"{client_name}_exchange",
                routing_key=f"{client_name}_client_queue",
                body=transform_dict(
                    response.dict(),
                    "ServerRequestResponse",
                ),
                properties=pika.BasicProperties(delivery_mode=2),
            )

        elif response["message_type"] == "StatusUpdate":
            status_update = TransmissionStatusUpdate(**response["data"])

            status_update.client_name = node.name

            downloaded_resource_manager = {}

            if status_update.contains_files():
                for file_name, file_uuid in status_update.response.files.items():
                    response = self.storage_interface.load_resource(file_uuid)
                    downloaded_resource_manager[file_name] = response

                    # remove file from storage
                    self.storage_interface.remove_resource(file_uuid)

            status_update = status_update.refill(downloaded_resource_manager)
            self.server.process_status_update(status_update)


class ClientToMQInterface(ClientInterface):
    def __init__(
        self,
        command_queue: list[dict],
        host: str,
        port: int,
        username: str,
        password: str,
        ping_interval: int = 1,
        ssl_context: ssl.SSLContext | None = None,
    ):
        super().__init__(command_queue=command_queue, ping_interval=ping_interval)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client_name = username
        self.ssl_context = ssl_context
        self.storage_interface = None

        self.client_queue_name = f"{self.username}_client_queue"
        self.server_queue_name = f"{self.username}_server_queue"

        # as two threads can send requests simultaneously, we need two channels (one for each thread)
        self.request_channel = self._build_connection(with_consume=True)    # ask for tasks and get tasks
        self.execute_channel = self._build_connection()                     # send status updates

    def _build_connection(self, with_consume: bool = False) -> pika.channel.Channel:
        """Builds a connection to the server.

        Args:
            with_consume (bool, optional): Whether to consume messages from the server. Defaults to False.

        Returns:
            pika.channel.Channel: The channel.
        """

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=pika.PlainCredentials(
                    username=self.username, password=self.password
                ),
                heartbeat=10 if not with_consume else 0, #this only works status updates queue. The other one sends from another thread
                ssl_options=pika.SSLOptions(self.ssl_context, server_hostname=self.host)
                if self.ssl_context
                else None,
            )
        )
        channel = connection.channel()

        channel.exchange_declare(
            exchange=f"{self.client_name}_exchange",
            exchange_type="direct",
            durable=True,
        )

        channel.queue_declare(queue=self.client_queue_name)
        channel.queue_declare(queue=self.server_queue_name)

        channel.queue_bind(
            exchange=f"{self.client_name}_exchange", queue=self.client_queue_name
        )

        if with_consume:
            channel.basic_consume(
                queue=self.client_queue_name,
                on_message_callback=self._on_response,
                auto_ack=True,
            )

        # Purge the queues to remove any old messages
        channel.queue_purge(queue=self.client_queue_name)
        channel.queue_purge(queue=self.server_queue_name)

        return channel

    def start(self):
        self.request_channel.start_consuming()

    def _pull(self):
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

        # get message datatype
        message_type = response["message_type"]

        if message_type == "ServerRequestResponse":
            # Get the request uuid
            response = Transferables().to_object(
                response["data"],
                ServerRequestResponse,
            )

            # if the response contains a new command, add it to the command queue
            if response.response.data:
                self.command_queue.append(response.response.data)

    def send_server_request(self, request: ServerRequest) -> None:
        """Publishes a server request to the server queue.

        Args:
            request (ServerRequest): The server request to send.
        """

        self.request_channel.basic_publish(
            exchange=f"{self.client_name}_exchange",
            routing_key=self.server_queue_name,
            body=transform_dict(request.dict(), "ServerRequest"),
        )

    def send_status_update(self, status_update: StatusUpdate) -> None:
        """Sends a status update to the server.

        Args:
            status_update (StatusUpdate): The status update to send.
        """

        resource_uuids = {}

        # Upload all files in the status update
        if status_update.contains_files():
            resource_uuids.update(
                self.storage_interface.upload_resources(
                    status_update.response.get_files(),
                )
            )

        # publish status update in the server queue
        self.execute_channel.basic_publish(
            exchange=f"{self.client_name}_exchange",
            routing_key=self.server_queue_name,
            body=transform_dict(
                status_update.unload(resource_uuids).dict(), "StatusUpdate"
            ),
        )
