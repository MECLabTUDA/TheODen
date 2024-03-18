from .notifications import (
    MetricNotification,
    StatusUpdateNotification,
    WatcherNotification,
)
from .watcher import Watcher


class MetricCollectionWatcher(Watcher):
    """Watcher to collect metrics from the framework"""

    def __init__(
        self,
        notification_of_interest: (
            dict[type[WatcherNotification], callable] | None
        ) = None,
    ) -> None:
        if notification_of_interest is None:
            notification_of_interest = {
                StatusUpdateNotification: self._handle_status_update,
                MetricNotification: self._handle_metric,
            }
        else:
            if StatusUpdateNotification not in notification_of_interest:
                notification_of_interest[
                    StatusUpdateNotification
                ] = self._handle_status_update
            if MetricNotification not in notification_of_interest:
                notification_of_interest[MetricNotification] = self._handle_metric

        super().__init__(notification_of_interest, self._process_other_notification)

    def _process_metrics(self, metric: MetricNotification) -> None:
        """Function to save the metrics"""
        pass

    def _process_other_notification(
        self, notification: WatcherNotification, origin: Watcher | None = None
    ) -> None:
        """Function to process other notifications"""
        pass

    def _handle_status_update(
        self, notification: StatusUpdateNotification, origin: Watcher | None = None
    ) -> None:
        # Get the status update
        status_update = notification.status_update
        # Check if the status update is a metric response
        if status_update.response and status_update.response.response_type == "metric":
            # Save the metrics
            self._process_metrics(
                MetricNotification(
                    metrics=status_update.response.get_data()["metrics"],
                    comm_round=status_update.response.get_data().get(
                        "comm_round", None
                    ),
                    epoch=status_update.response.get_data().get("epoch", None),
                    metric_type=status_update.response.get_data()["metric_type"],
                    client_name=status_update.client_name,
                    command_uuid=status_update.command_uuid,
                )
            )

    def _handle_metric(
        self, notification: MetricNotification, origin: Watcher | None = None
    ) -> None:
        # check if this object is the origin of the notification
        if origin is not self:
            self._process_metrics(notification)
