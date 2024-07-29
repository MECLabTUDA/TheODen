import logging

from torchvision import transforms

from theoden import start_server
from theoden.datasets import *
from theoden.models import *
from theoden.operations import *
from theoden.resources import *

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
        dataset=SemiCOLWeaklyDataset(patch_size=320),
        # we will split the dataset globally by cases
        global_partition_function=MetadataPartition("institute"),
        global_balancing_function=EqualBalancing(),
        # we will split the dataset locally by case into train, val and test
        local_partition_function=MetadataPartition("tif"),
        local_balancing_function=PercentageBalancing({"train": 0.85, "val": 0.15}),
        # we will use a color jitter augmentation for training
        train_augmentation=[SimCLRTransformWrapper(input_size=320, normalize=None)],
        val_augmentation=[SimCLRTransformWrapper(input_size=320, normalize=None)],
        # test_augmentation=[SimCLRTransformWrapper(input_size=320, normalize=None)],
        # we use the cross entropy loss for training and also display the dice
        losses=[NTXentLossWrapper()],
        # we use the Adam optimizer with a learning rate of 0.0001
        optimizer=SGDOptimizer(0.1),
        # we use the CosineAnnealingLRScheduler with a T_max of 1000
        lr_scheduler=CosineAnnealingLRScheduler(1200),
        model_key="simclr",
        simultaneous_execution=1,
    )

    instructions = [
        # The condition is used to wait for a certain number of clients to join the federation before starting training.
        InitGlobalModelAction(),
        # The multi-round training instruction group is used to specify the training loop.
        # It specifies the number of rounds, steps per round, the batch sizes, the aggregator and the distributor.
        MultiRoundTrainingInstructionBundle(
            # we train for 1000 rounds with 5 steps per round
            n_rounds=1000,
            steps_per_round=10,
            # we use a batch size of 12 for training and 128 for validation
            train_batch_size=24,
            validation_batch_size=128,
            # we use the FedAvgAggregator with the DatasetLengthScore as client score
            aggregator=FedAvgAggregator(),
            # we validate on the val split of the dataset after every 10 rounds
            validate_every_n_rounds=100,
            start_n_rounds_without_validation=0,
            label_key="segmentation_mask",
            model_key="simclr",
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
        run_name="pretraining_simclr_weak",
        global_context="global_context.yaml",
        config="topology.yaml",
        use_aim=True,
    )
