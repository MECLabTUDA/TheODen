from typing import Literal

from ..common import Transferable
from .metric_collector import MetricCollectionWatcher, Watcher
from .notifications import MetricNotification, NewBestModelNotification
from ..resources import Loss


class NewBestDetectorWatcher(MetricCollectionWatcher, Transferable):
    def __init__(
        self,
        metric: str | None = None,
        metric_type: Literal["aggregate", "single"] = "aggregate",
        lower_is_better: bool = False,
    ) -> None:
        self.metric_type = metric_type
        self.metric = metric
        self.lower_is_better = lower_is_better

        self.best_score: dict[str, float] = {}

        super().__init__()

    def _process_metrics(self, metric: MetricNotification) -> None:
        if self.metric is None:
            losses = self.pool.base_topology.resource_register.gr(
                "losses", assert_type=list[Loss]
            )
            choosing = Loss.get_choosing_criterion(losses)
            self.metric = choosing.display_name()
            self.lower_is_better = not choosing.higher_better

        if metric.is_aggregate and self.metric_type == "single":
            return

        if not metric.is_aggregate and self.metric_type == "aggregate":
            return

        if self.metric not in metric.metrics:
            return

        is_new_best = False

        if metric.metric_type not in self.best_score:
            is_new_best = True
        elif self.lower_is_better:
            if metric.metrics[self.metric] < self.best_score[metric.metric_type]:
                is_new_best = True
        else:
            if metric.metrics[self.metric] > self.best_score[metric.metric_type]:
                is_new_best = True

        if is_new_best:
            self.best_score[metric.metric_type] = metric.metrics[self.metric]
            self.pool.notify_all(
                notification=NewBestModelNotification(
                    metric=self.metric,
                    split=metric.metric_type,
                    comm_round=metric.comm_round,
                ),
                origin=self,
            )
            print(f"Current best metric (Round {metric.comm_round}): {self.best_score}")
