import logging

import typer

from theoden import start_client


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
    )


if __name__ == "__main__":
    typer.run(main)
