from ..commands import ExitRunCommand
from .distribution import ClosedDistribution


class Exit(ClosedDistribution):
    """Exit the server and all clients"""

    def __init__(self) -> None:
        super().__init__(ExitRunCommand())
