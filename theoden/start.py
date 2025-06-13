import ssl
from getpass import getpass

import uvicorn

from .common import GlobalContext
from .datasets import *
from .models import *
from .networking import FileStorage
from .operations import *
from .resources import *
from .topology import *
from .topology.client import Client
from .topology.server import Server
from .watcher import Watcher

import logging
logger = logging.getLogger(__name__)

def start_client(
    communication_address: str = "localhost",
    communication_port: int | None = None,
    resource_address: str | None = None,
    resource_port: int | None = None,
    username: str = "dummy",
    password: str = "dummy",
    global_context: str | None = None,
    ping_interval: float = 0.2,
    rabbitmq: bool = False,
    ssl: bool = False,
    ssl_context: ssl.SSLContext | None = None,
    initial_commands: list[Command] | None = None,
    dataset_paths: dict[str, str] | None = None,
):
    """Start a client node.

    Args:
        communication_address (str): The address of the communication interface.
        communication_port (int): The port of the communication interface.
        resource_address (str): The address of the resource interface.
        resource_port (int): The port of the resource interface.
        username (str): The username of the client.
        password (str): The password of the client.
        global_context (str): The path to the global context YAML file.
        ping_interval (float): The interval at which the client should ping the server.
        rabbitmq (bool): Whether to use RabbitMQ as the communication interface.
        ssl (bool): Whether to use SSL for the communication interface.
        ssl_context (ssl.SSLContext): The SSL context to use for the communication interface.
    """
    # Load the global context if it is provided
    if global_context is not None:
        GlobalContext().load_from_yaml(global_context)

    # Set the dataset paths if they are provided
    if dataset_paths is not None:
        for dataset, path in dataset_paths.items():
            GlobalContext()["datasets"] = {**GlobalContext()["datasets"], dataset: path}

    # If the username is not "dummy" and the password is "dummy", prompt the user for the password
    if username != "dummy" and password == "dummy":
        password = getpass("Password: ")

    # Start the client
    client = Client(
        communication_address=communication_address,
        communication_port=communication_port,
        resource_address=resource_address,
        resource_port=resource_port,
        username=username,
        password=password,
        ping_interval=ping_interval,
        rabbitmq=rabbitmq,
        ssl=ssl,
        ssl_context=ssl_context,
        initial_commands=initial_commands,
    )
    client.start()


def start_storage(
    config: str | None = None,
    host: str = "localhost",
    port: int = 8000,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
):
    """Start a file storage node.

    Args:
        config (str, optional): The path to the node config YAML file. Defaults to None.
        host (str, optional): The host to bind to. Defaults to "localhost".
        port (int, optional): The port to bind to. Defaults to 8000.
        ssl_keyfile (str, optional): The path to the SSL key file. Defaults to None.
        ssl_certfile (str, optional): The path to the SSL certificate file. Defaults to None.
    """
    from uvicorn.config import LOGGING_CONFIG

    logging_config = LOGGING_CONFIG.copy()

    logging_config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    logging_config["formatters"]["access"]["fmt"] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    logging_config["formatters"]["default"]["datefmt"] = '%Y-%m-%d %H:%M:%S'
    logging_config["formatters"]["access"]["datefmt"] = '%Y-%m-%d %H:%M:%S'


    uvicorn.run(
        FileStorage(node_config=config),
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_config=logging_config
    )


def start_server(
    instructions: list[Condition | Instruction | InstructionBundle] | InstructionBundle,
    run_name: str,
    permanent_conditions: list[Condition] | None = None,
    open_distribution: OpenDistribution | None = None,
    global_context: str | None = None,
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
    ssl_context: ssl.SSLContext | None = None,
    https: bool = False,
    uvicorn_kwargs: dict = {},
    exit_on_finish: bool = False,
    use_aim: bool = False,
    watcher: list[Watcher] | None = None,
    choosing_metric: str | None = None,
    lower_is_better: bool = False,
    metric_type: str = None,
    **kwargs,
):
    """Start a server node.

    Args:
        instructions (list[Condition | Instruction | InstructionBundle] | InstructionBundle): The instructions to execute.
        run_name (str): The name of the run.
        permanent_conditions (list[Condition], optional): The permanent conditions to execute. Defaults to None.
        open_distribution (OpenDistribution, optional): The open distribution to use. Defaults to None.
        global_context (str, optional): The path to the global context YAML file. Defaults to "global_context.yaml".
        communication_address (str, optional): The address of the communication interface. Defaults to None.
        communication_port (int, optional): The port of the communication interface. Defaults to None.
        storage_address (str, optional): The address of the storage interface. Defaults to None.
        storage_port (int, optional): The port of the storage interface. Defaults to None.
        username (str, optional): The username of the server. Defaults to None.
        password (str, optional): The password of the server. Defaults to None.
        config (str, optional): The path to the node config YAML file. Defaults to None.
        ssl_keyfile (str, optional): The path to the SSL key file. Defaults to None.
        ssl_certfile (str, optional): The path to the SSL certificate file. Defaults to None.
        rabbitmq (bool, optional): Whether to use RabbitMQ as the communication interface. Defaults to False.
        ssl_context (ssl.SSLContext, optional): The SSL context to use for the communication interface. Defaults to None.
        https (bool, optional): Whether to use HTTPS for the communication interface. Defaults to False.
        uvicorn_kwargs (dict, optional): Additional keyword arguments for uvicorn. Defaults to {}.
        exit_on_finish (bool, optional): Whether to exit the server after finishing. Defaults to False.
        use_aim (bool, optional): Whether to use AIM for logging. Defaults to False.
    """

    if global_context is not None:
        GlobalContext().load_from_yaml(global_context)

    logging.basicConfig(level=logging.WARNING)
    e = Server(
        debug=True,
        initial_instructions=(
            instructions if isinstance(instructions, list) else [instructions]
        )
        + ([Exit()] if exit_on_finish else []),
        permanent_conditions=permanent_conditions,
        open_distribution=open_distribution,
        run_name=run_name,
        storage_address=storage_address,
        storage_port=storage_port,
        username=username,
        password=password,
        communication_address=communication_address,
        communication_port=communication_port,
        node_config=config,
        rabbitmq=rabbitmq,
        ssl_context=ssl_context,
        https=https,
        use_aim=use_aim,
        watcher=watcher,
        choosing_metric=choosing_metric,
        lower_is_better=lower_is_better,
        metric_type=metric_type,
        **kwargs,
    )

    if not e.rabbitmq:
        if ssl_context is not None:
            logger.warning("SSL context is provided, but uvicorn will not use")
        if https:
            logger.warning("HTTPS is enabled, but uvicorn will not use it")

        uvicorn.run(
            e.communication_interface,
            host=communication_address or "localhost",
            port=communication_port or 8000,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            **uvicorn_kwargs,
        )
