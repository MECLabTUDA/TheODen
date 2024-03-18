from itertools import chain

from .metric_collector import MetricCollectionWatcher, Watcher
from .notifications import CommandFinishedNotification, MetricNotification


class MetricAggregationWatcher(MetricCollectionWatcher):
    def __init__(
        self, aggregation_method: str = "mean", clear_after_aggregation: bool = True
    ) -> None:
        super().__init__({CommandFinishedNotification: self._handle_command_finished})
        self.aggregation_method = aggregation_method
        self.clear_after_aggregation = clear_after_aggregation
        self._metrics: dict[str, list[MetricNotification]] = {}

    def _handle_command_finished(
        self, notification: CommandFinishedNotification, origin: Watcher | None = None
    ) -> None:
        if notification.command_uuid not in self._metrics:
            return

        # get the metrics
        metrics = self._metrics[notification.command_uuid]

        # loop over different comm rounds
        for comm_round in set(m.comm_round for m in metrics):
            # loop over different epochs
            for epoch in set(m.epoch for m in metrics):
                # loop over different metric types
                for metric_type in set(m.metric_type for m in metrics):
                    # get all the relevant metrics based on the comm round, epoch and metric type
                    relevant_metrics = [
                        m
                        for m in metrics
                        if m.comm_round == comm_round
                        and m.epoch == epoch
                        and m.metric_type == metric_type
                    ]

                    # check if there are any metrics
                    if len(relevant_metrics) == 0:
                        continue

                    # get all the metric names
                    metric_names = set(
                        chain.from_iterable(
                            list(m.metrics.keys()) for m in relevant_metrics
                        )
                    )

                    # continue if there is a metric with score None
                    if any(
                        m.metrics.get(metric_name) is None
                        for m in relevant_metrics
                        for metric_name in metric_names
                    ):
                        continue

                    # aggregate the metrics
                    if self.aggregation_method == "mean":
                        aggregated_metric = {
                            metric_name: sum(
                                m.metrics[metric_name] for m in relevant_metrics
                            )
                            / len(relevant_metrics)
                            for metric_name in metric_names
                        }
                    elif self.aggregation_method == "median":
                        raise NotImplementedError(
                            "Median aggregation is not implemented yet"
                        )

                    # notify the pool of the aggregated metric
                    self.pool.notify_of_type(
                        notification=MetricNotification(
                            metrics=aggregated_metric,
                            comm_round=comm_round,
                            epoch=epoch,
                            metric_type=metric_type,
                            client_name=self.aggregation_method,
                            is_aggregate=True,
                            command_uuid=notification.command_uuid,
                        ),
                        of_type=MetricCollectionWatcher,
                        origin=self,
                    )
        # remove the metric
        del self._metrics[notification.command_uuid]

    def _process_metrics(self, metric: MetricNotification) -> None:
        # check if the metric is an aggregate, if so, ignore it
        if metric.is_aggregate:
            return

        if metric.command_uuid not in self._metrics:
            self._metrics[metric.command_uuid] = []

        # append the new metric
        self._metrics[metric.command_uuid].append(metric)
