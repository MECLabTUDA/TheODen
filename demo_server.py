from torchvision import transforms

from theoden.common import GlobalContext
from theoden.operations import *
from theoden.resources import *
from theoden.datasets import *
from theoden.models import *
from theoden import start_server

"""
Basic example of a federated learning server. 
As the server is the main module to handle and orchestrate the training, 
"""


# set up global context. This is a singleton object that is used to store global variables like paths to datasets.
GlobalContext().load_from_yaml("demo_context.yaml")

initial_instructions = [
    # The condition is used to wait for a certain number of nodes to join the federation before starting training.
    RequireNumberOfNodesCondition(3),
    # The default training instruction group is a class that allows specifying many relevant training options.
    # This includes the dataset, the splits, augmentations, losses, optimizer, scheduler and the model.
    DefaultTrainingInitInstructionBundle(
        # we use the SMPModel with a resnet34 encoder and 10 classes
        model=SMPModel(architecture="unet", classes=4, encoder_name="resnet34"),
        # we use the BCSSDataset where we exclude patches with mainly background and only use the first 4 classes
        dataset=ExclusionDataset(
            dataset=MappingDataset(
                mapping=SegmentationMapping(
                    except_map=([-1, 0, 1, 2, 3], -1), shift=-1
                ),
                dataset=BCSSDataset(patch_size=320, overlap=0.2),
            ),
            exclusion=SegmentationExclusion([(-1, 0.9)]),
        ),
        # we will split the dataset globally by cases
        global_partition_function=MetadataPartition("case"),
        global_balancing_function=EqualBalancing(),
        # we will split the dataset locally by case into train, val and test
        local_partition_function=MetadataPartition("case"),
        local_balancing_function=PercentageBalancing(
            {"train": 0.7, "val": 0.15, "test": 0.15}
        ),
        # we will use a color jitter augmentation for training
        train_augmentation=[
            RandomFlippingAugmentation(),
            TVAugmentation(
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                )
            ),
        ],
        # we use the cross entropy loss for training and also display the dice
        losses=[
            CELoss(ignore_index=-1),
            DisplayDiceLoss(4, choosing_criterion=True, ignore_index=-1),
        ],
        # we use the Adam optimizer with a learning rate of 0.0001
        optimizer=AdamOptimizer(0.0001),
        # we use the CosineAnnealingLRScheduler with a T_max of 1000
        lr_scheduler=CosineAnnealingLRScheduler(1000),
        simultaneous_execution=1,
    ),
    # For the client score we use the DatasetLengthScore which is the default score for FedAvg.
    ClosedDistribution(
        CalculateClientScoreCommand(DatasetLengthScore("dataset:train"))
    ),
    # The multi-round training instruction group is used to specify the training loop.
    # It specifies the number of rounds, steps per round, the batch sizes, the aggregator and the distributor.
    MultiRoundTrainingInstructionBundle(
        # we train for 1000 rounds with 5 steps per round
        n_rounds=1000,
        steps_per_round=5,
        # we use a batch size of 12 for training and 128 for validation
        train_batch_size=12,
        validation_batch_size=128,
        # we use the FedAvgAggregator with the DatasetLengthScore as client score
        aggregator=FedOptAggregator(
            server_optimizer=FedAvgAggregator(), client_score=DatasetLengthScore
        ),
        # we validate on the val split of the dataset after every 10 rounds
        validate_every_n_rounds=10,
        label_key="segmentation_mask",
        # if you encounter cuda memory issues, you can reduce the number of simultaneous executions to 1
        simultaneous_execution=2,
    ),
]

# start the server
start_server(
    instructions=initial_instructions,
    run_name="theoden_demo",
    rabbitmq=False,
)
