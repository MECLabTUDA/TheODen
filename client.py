import logging

import typer

from theoden import start_client

import ssl as ssl_connector

def main(
    communication_address: str = "localhost",
    communication_port: int | None = None,
    resource_address: str | None = None,
    resource_port: int | None = None,
    username: str = "dummy",
    password: str = "dummy",
    global_context: str | None = None,
    rabbitmq: bool = False,
    ssl: bool = False,
    ping_interval: float = 0.4,
):
    logging.basicConfig(level=logging.INFO)

    start_client(
        communication_address=communication_address,
        communication_port=communication_port,
        resource_address=resource_address,
        resource_port=resource_port,
        username=username,
        password=password,
        ping_interval=ping_interval,
        rabbitmq=rabbitmq,
        ssl=ssl,
        global_context=global_context,
        ssl_context=ssl_connector.create_default_context() if ssl else None,
    )


if __name__ == "__main__":
    typer.run(main)
