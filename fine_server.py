import logging

from torchvision import transforms

from theoden import start_server
from theoden.datasets import *
from theoden.models import *
from theoden.operations import *
from theoden.resources import *
from theoden.watcher import BestModelSaverWatcher

if __name__ == "__main__":
    """
    Basic example of a federated learning server.
    As the server is the main module to handle and orchestrate the training,"""

    open_distribution = DefaultTrainingInitOpenDistribution(
        # we use the SMPModel with a resnet34 encoder and 10 classes
        pre_commands=[
            InitModelCommand(SMPModel(architecture="unet", classes=10)),
            SelectTorchEncoderOfModelCommand("model", "encoder"),
            WrapModelCommand("encoder", SimCLRWrapperModel, "simclr"),
        ],
        # we use the BCSSDataset where we exclude patches with mainly background and only use the first 4 classes
        dataset=ExclusionDataset(
            dataset=MappingDataset(
                mapping=SegmentationMapping(shift=-1),
                dataset=SemiCOLDataset(patch_size=320, overlap=0.2),
            ),
            exclusion=SegmentationExclusion([(-1, 0.9)]),
        ),
        # we will split the dataset globally by cases
        global_partition_function=MetadataPartition("institute"),
        global_balancing_function=EqualBalancing(),
        # we will split the dataset locally by case into train, val and test
        local_partition_function=MetadataPartition("case"),
        local_balancing_function=PercentageBalancing(
            {"train": 0.6, "val": 0.3, "test": 0.1}
        ),
        # we will use a color jitter augmentation for training
        # we use the cross entropy loss for training and also display the dice
        losses=[
            CELoss(),
            DisplayDiceLoss(10, ignore_index=-1, choosing_criterion=True),
        ],
        # we use the Adam optimizer with a learning rate of 0.0001
        optimizer=SGDOptimizer(0.1),
        # we use the CosineAnnealingLRScheduler with a T_max of 1000
        lr_scheduler=CosineAnnealingLRScheduler(15000),
        model_key="model",
        simultaneous_execution=1,
    )

    instructions = [
        # The condition is used to wait for a certain number of clients to join the federation before starting training.
        InitGlobalModelAction(
            FileInitializer(
                "/home/jstieber/models/theoden/pretraining_simclr_weak/simclr_best_train.pt"
            ),
            "simclr",
        ),
        ClosedDistribution(LoadStateDictCommand("simclr", strict=True)),
        InitGlobalModelAction(),
        # The multi-round training instruction group is used to specify the training loop.
        # It specifies the number of rounds, steps per round, the batch sizes, the aggregator and the distributor.
        MultiRoundTrainingInstructionBundle(
            # we train for 1000 rounds with 5 steps per round
            n_rounds=10,
            steps_per_round=2,
            # we use a batch size of 12 for training and 128 for validation
            train_batch_size=24,
            validation_batch_size=128,
            # we use the FedAvgAggregator with the DatasetLengthScore as client score
            aggregator=FedAvgAggregator(),
            # we validate on the val split of the dataset after every 10 rounds
            validate_every_n_rounds=400,
            start_n_rounds_without_validation=0,
            label_key="segmentation_mask",
            model_key="model",
            # if you encounter cuda memory issues, you can reduce the number of simultaneous executions to 1
            simultaneous_execution=2,
        ),
    ]

    logging.basicConfig(level=logging.INFO)

    # start the server
    start_server(
        instructions=instructions,
        open_distribution=open_distribution,
        permanent_conditions=[RequireNumberOfClientsCondition(2)],
        run_name="fine_tuning_simclr_2000",
        global_context="global_context.yaml",
        config="topology.yaml",
        use_aim=True,
        exit_on_finish=True,
    )
