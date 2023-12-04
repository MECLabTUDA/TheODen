import uvicorn
import logging
from getpass import getpass


from .common import GlobalContext
from .operations import *
from .resources import *
from .datasets import *
from .models import *
from .topology import *
from theoden.networking import FileStorage

from .topology.server import Server
from .topology.node import Node


def start_node(
    communication_address: str = "localhost",
    communication_port: int | None = None,
    resource_address: str | None = None,
    resource_port: int | None = None,
    username: str = "dummy",
    password: str = "dummy",
    global_context: str = "global_context.yaml",
    ping_interval: float = 0.2,
    rabbitmq: bool = False,
    ssl: bool = False,
):
    GlobalContext().load_from_yaml(global_context)
    if username != "dummy" and password == "dummy":
        password = getpass("Password: ")
    node = Node(
        communication_address=communication_address,
        communication_port=communication_port,
        resource_address=resource_address,
        resource_port=resource_port,
        username=username,
        password=password,
        ping_interval=ping_interval,
        rabbitmq=rabbitmq,
        ssl=ssl,
    )
    node.start()


def start_storage(
    config: str | None = None,
    host: str = "localhost",
    port: int = 8000,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
):
    uvicorn.run(
        FileStorage(node_config=config),
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


def start_server(
    instructions: list[Condition | Instruction | InstructionBundle] | InstructionBundle,
    run_name: str,
    global_context: str = "global_context.yaml",
    communication_address: str | None = None,
    communication_port: int | None = None,
    storage_address: str | None = None,
    storage_port: int | None = None,
    username: str | None = None,
    password: str | None = None,
    config: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
    rabbitmq: bool = False,
):
    GlobalContext().load_from_yaml(global_context)

    logging.basicConfig(level=logging.WARNING)
    e = Server(
        debug=True,
        initial_instructions=instructions
        if isinstance(instructions, list)
        else [instructions],
        run_name=run_name,
        storage_address=storage_address,
        storage_port=storage_port,
        username=username,
        password=password,
        communication_address=communication_address,
        communication_port=communication_port,
        node_config=config,
        rabbitmq=rabbitmq,
    )
    if not e.rabbitmq:
        uvicorn.run(
            e.communication_interface,
            host=communication_address or "localhost",
            port=communication_port or 8000,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
