from pydantic import BaseModel

from ..common.typing import StatusUpdate
from ..operations import ServerRequest
from ..topology import Topology


class WatcherNotification(BaseModel):
    """Base class for all notifications."""

    pass


class InitializationNotification(WatcherNotification):
    run_name: str | None = None


class StatusUpdateNotification(WatcherNotification):
    """Notification for a status update."""

    status_update: StatusUpdate


class ServerRequestNotification(WatcherNotification):
    """Notification for a server request."""

    request: ServerRequest

    class Config:
        arbitrary_types_allowed = True


class TopologyChangeNotification(WatcherNotification):
    """Notification for a server request."""

    topology: Topology

    class Config:
        arbitrary_types_allowed = True


class NewBestModelNotification(WatcherNotification):
    """Notification for a new best model."""

    metric: str
    split: str
    comm_round: int | None = None


class AggregationCompletedNotification(WatcherNotification):
    """Notification that aggregation has completed."""

    comm_round: int | None = None


class ParameterNotification(WatcherNotification):
    """Notification for a new best model."""

    params: dict[str, float | int | str]
    comm_round: int | None = None


class CommandFinishedNotification(WatcherNotification):
    """Notification for a new best model."""

    command_uuid: str


class MetricNotification(WatcherNotification):
    """Notification for a new best model."""

    metrics: dict[str, float]
    metric_type: str
    comm_round: int | None = None
    epoch: int | None = None
    node_name: str | None = None
    is_aggregate: bool = False
    command_uuid: str | None = None
