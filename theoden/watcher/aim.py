from aim import Run

from ..common import Transferable
from .metric_collector import MetricCollectionWatcher, Watcher
from .notifications import (
    MetricNotification,
    ParameterNotification,
    InitializationNotification,
)


class AimMetricCollectorWatcher(MetricCollectionWatcher, Transferable):
    """Watcher to collect metrics from the framework and save them to Aim"""

    def __init__(self) -> None:
        super().__init__(
            notification_of_interest={
                ParameterNotification: self._process_parameter,
                InitializationNotification: self._init,
            }
        )

    def _init(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.run = Run(experiment=notification.run_name)

    def _process_metrics(self, metric: MetricNotification) -> None:
        for metric_name, metric_value in metric.metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError("Metric names must be strings")
            if not isinstance(metric_value, (float | int)):
                raise ValueError("Metric values must be floats or int")

            self.run.track(
                metric_value,
                name=f"{metric_name}_{metric.metric_type}",
                step=metric.comm_round,
                context={
                    "node_uuid"
                    if not metric.is_aggregate
                    else "aggregate": metric.node_uuid
                },
            )

    def _process_parameter(
        self, notification: ParameterNotification, origin: Watcher | None = None
    ) -> None:
        for param_name, param_value in notification.params.items():
            if not isinstance(param_name, str):
                raise ValueError("Parameter names must be strings")
            if not isinstance(param_value, (float | int | str)):
                raise ValueError("Parameter values must be string, floats or int")

        self.run["hparams"] = notification.params
        # (
        #     self.run["hparams"] if "hparams" in self.run else {}
        # ) | {param_name: param_value}
