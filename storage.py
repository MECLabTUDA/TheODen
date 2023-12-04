import typer
from theoden import start_storage


def main(
    config: str | None = None,
    host: str = "localhost",
    port: int = 8000,
    ssl_keyfile: str | None = None,
    ssl_certfile: str | None = None,
):
    start_storage(
        config=config,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == "__main__":
    typer.run(main)
