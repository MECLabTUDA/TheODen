# Aggregation
![Aggregation](imgs/aggregation.png "Aggregation")
Aggregation is a combination of client commands and an Aggregation Action as shown in the figure above.
## Current Aggregators

| Aggregator                  | Description                                                                     | Link                                     |
| --------------------------- |---------------------------------------------------------------------------------|------------------------------------------|
| `FedAvgAggregator`          | Aggregates the weights of the clients (e.g. mean) and send it as a new metric   | [link](../theoden/aggregator/mean.py)    |
| `MedianAggregator`          | Aggregates the weights of the clients (e.g. median) and send it as a new metric | [link](../theoden/aggregator/median.py)  |
| `FedAdamServerOptimizer`    | Aggregates the weights using Adam Optimizer on the server.                      | [link](../theoden/aggregator/max.py)     |
| `FedAdagradServerOptimizer` | Aggregates the weights using AdaGrad Optimizer on the server.                   | [link](../theoden/aggregator/min.py)     |
| `FedYogiServerOptimizer`    | Aggregates the weights using Yogi Optimizer on the server.                      | [link](../theoden/aggregator/sum.py)     |

# Create an aggregator
   
Please refer to the tutorial [here](./TUTORIAL.md) to create a new aggregator.
