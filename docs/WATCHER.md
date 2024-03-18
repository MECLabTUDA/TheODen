# Watcher observe the training

Watcher are the main tool for observing the training. They can be used to collect metrics, aggregate them, save the best model, etc. The watcher are implemented as a observer pattern. They are notified by the server about the current status and can react to it. The watcher can also send notifications to the other watchers to trigger a specific action.

Withing the server or client code, classes can send notifications to the WatcherPool. The WatcherPool will then notify all watcher about the new notification.

Each watcher has a specific set of notifications it is interested in. They are defined in the constructor of the watcher. For each notification a function is defined that will be called if the notification is received.

As an example, the MetricCollectionWatcher is interested in the `MetricNotification` and `StatusUpdateNotification`. If the MetricCollectionWatcher receives a `MetricNotification` it will collect the metric and if it receives a `StatusUpdateNotification` it will collect the metric:

```python
class MetricCollectionWatcher(Watcher):
    """Watcher to collect metrics from the framework"""

    def __init__(
        self,
        notification_of_interest: (
            dict[type[WatcherNotification], callable] | None
        ) = None,
    ) -> None:
        # ...
            notification_of_interest = {
                StatusUpdateNotification: self._handle_status_update,
                MetricNotification: self._handle_metric,
            }
        # ...

        super().__init__(notification_of_interest, self._process_other_notification)
```

## Current Notifications

| Notification                       | Description                                                                                                                                |
| ---------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------|
| `InitializationNotification`       | Notify the start of an experiment                                                                                                          |
| `StatusUpdateNotification`         | Notify about a Status Update received from of of te clients                                                                                |
| `ServerRequestNotification`        | Notify about a Serverrequest received from of of te clients                                                                                |
| `TopologyChangeNotification`       | Notify about a change in the topology (e.g. a client logged in or off or got timed out)                                                    |
| `NewBestModelNotification`         | Notify that the current model is the new best model according to a given decision metric (either directly set or using the loss functions) |
| `AggregationCompletedNotification` | Notify that the aggregation of an Aggregator is finished                                                                                   |
| `ParameterNotification`            | Notify about set  training parameters                                                                                                      |
| `CommandFinishedNotification`      | Notify about the end of a command. Using this information e.g. the metric aggregation is being triggered as now all responses are send     |
| `MetricNotification`               | Notify about a new metric either from a client or from another watcher (e.g. the MetricAggregationWatcher)                                 |

## Current Watcher

| Watcher                     | Description                                                                                                                                                                             | Link                                            |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `MetricCollectionWatcher`   | Collects the metrics from the other Watcher or the clients Status Updates                                                                                                               | [link](../theoden/watcher/metric_collector.py)  |
| `MetricAggregationWatcher`  | Aggregates the metrics of the clients (e.g. mean) and send it as a new metric                                                                                                           | [link](../theoden/watcher/metric_aggregator.py) |
| `NewBestDetectorWatcher`    | Detects if the current model is the new best model and triggers a NewBestModelNotification if its the case                                                                              | [link](../theoden/watcher/best_detector.py)     |
| `BestModelSaverWatcher`     | Saves a model, if it the new best model                                                                                                                                                 | [link](../theoden/watcher/model_saver.py)       |
| `AimMetricCollectorWatcher` | The watcher will create an Aim run and los the metrics and parameters                                                                                                                   | [link](../theoden/watcher/aim.py)               |
| `TheodenConsoleWatcher`     | This Watcher will create an rest interface to open a websocket to the Theoden Console. The console will display the status of the current topology and the current command distribution | [link](../theoden/watcher/console.py)           |




## Create a watcher

To implement a watcher, a class must be created that inherits from the Watcher class. The class must implement the `__init__` function and the functions that are called when a notification is received.

As an example, the `AimMetricCollectorWatcher` is a watcher that collects metrics and parameters and saves them to Aim. The watcher is interested in the `ParameterNotification`, `InitializationNotification`, `TopologyChangeNotification` and `MetricNotification`. If the watcher receives a `ParameterNotification` it will log the parameters, if it receives a `InitializationNotification` it will create a new run in Aim, if it receives a `TopologyChangeNotification` it will log the topology and if it receives a `MetricNotification` it will log the metric:
    
```python

class AimMetricCollectorWatcher(MetricCollectionWatcher):
    """Watcher to collect metrics from the framework and save them to Aim"""

    def __init__(self) -> None:
        super().__init__(
            notification_of_interest={
                ParameterNotification: self._process_parameter,
                InitializationNotification: self._init,
                TopologyChangeNotification: self._show_topology,
            }
        )

    def _init(
        self, notification: InitializationNotification, origin: Watcher | None = None
    ) -> None:
        self.run = Run(experiment=notification.run_name)

    def _show_topology(
        self, notification: TopologyChangeNotification, origin: Watcher | None = None
    ) -> None:
        logging.info("Topology change notification received")

    def _process_metrics(self, metric: MetricNotification) -> None:
        for metric_name, metric_value in metric.metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError("Metric names must be strings")
            if metric_value is None:
                continue
            if not isinstance(metric_value, (float | int)):
                raise ValueError("Metric values must be floats or int")

            self.run.track(
                metric_value,
                name=f"{metric_name}_{metric.metric_type}",
                step=metric.comm_round,
                context={
                    (
                        "client_name" if not metric.is_aggregate else "aggregate"
                    ): metric.client_name
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
```

As an example please refer to the tutorial [here](./TUTORIAL.md) to create a new watcher.

## Notify the WatcherPool
If you have access to the resource manager, you can notify the WatcherPool by calling the `notify_all` function. The function takes a WatcherNotification as an argument. The WatcherPool will then notify all watcher about the new notification.

E.g. when the server gets a serverrequest, it will notify the WatcherPool:

```python
self.resources.watcher.notify_all(ServerRequestNotification(request=request))
```