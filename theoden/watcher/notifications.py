from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.typing import StatusUpdate
    from ..operations import ServerRequest
    from ..topology import Topology


class WatcherNotification:
    """Base class for all notifications."""

    pass


@dataclass
class InitializationNotification(WatcherNotification):
    run_name: str | None = None


@dataclass
class StatusUpdateNotification(WatcherNotification):
    """Notification for a status update."""

    status_update: "StatusUpdate"


@dataclass
class ServerRequestNotification(WatcherNotification):
    """Notification for a server request."""

    request: "ServerRequest"


@dataclass
class TopologyChangeNotification(WatcherNotification):
    """Notification for a server request."""

    topology: "Topology"


@dataclass
class NewBestModelNotification(WatcherNotification):
    """Notification for a new best model."""

    metric: str
    split: str
    comm_round: int | None = None


@dataclass
class AggregationCompletedNotification(WatcherNotification):
    """Notification that aggregation has completed."""

    comm_round: int | None = None


@dataclass
class ParameterNotification(WatcherNotification):
    """Notification for a new best model."""

    params: dict[str, float | int | str]
    comm_round: int | None = None


@dataclass
class CommandFinishedNotification(WatcherNotification):
    """Notification for a new best model."""

    command_uuid: str


@dataclass
class MetricNotification(WatcherNotification):
    """Notification for a new best model."""

    metrics: dict[str, float]
    metric_type: str
    comm_round: int | None = None
    epoch: int | None = None
    client_name: str | None = None
    is_aggregate: bool = False
    command_uuid: str | None = None
