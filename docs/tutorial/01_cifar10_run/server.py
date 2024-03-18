from theoden import start_server

from theoden.operations import *
from theoden.datasets import CIFAR10_Adapted
from theoden.resources import *
from theoden.models import TimmModel


if __name__ == "__main__":
    instructions = [
        ClosedDistribution(
            SequentialCommand(
                [
                    LoadDatasetCommand(CIFAR10_Adapted("/home/jstieber/data")),
                    SetPartitionCommand(
                        partition_function=IndexPartition(),
                        balancing_function=EqualBalancing(),
                    ),
                    SetLocalPartitionCommand(
                        partition_function=IndexPartition(),
                        balancing_function=PercentageBalancing(
                            {"train": 0.8, "val": 0.2}
                        ),
                    ),
                    SetLossesCommand([CELoss(), AccuracyLoss(choosing_criterion=True)]),
                    InitModelCommand(
                        TimmModel(
                            "resnet18", num_classes=10, pretrained=False, in_chans=3
                        )
                    ),
                    SetOptimizerCommand(AdamOptimizer(lr=0.0001)),
                    PrintResourceKeysCommand(),
                ]
            )
        ),
        InitGlobalModelAction(initializer=SelectRandomOneInitializer()),
        MultiRoundTrainingInstructionBundle(
            n_rounds=10,
            steps_per_round=100,
            train_batch_size=32,
            aggregator=FedAvgAggregator(),
            final_validation=False,
        ),
    ]

    start_server(
        run_name="01_cifar10_run",
        instructions=instructions,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        exit_on_finish=True,
        global_context="../tutorial_context.yaml",
    )
