from .. import (
    Initializer,
    InitGlobalModelAction,
    ClosedDistribution,
)
from .bundle import InstructionBundle
from ...commands import *
from ....resources import *
from ....common import Transferable


class DefaultTrainingInitInstructionBundle(InstructionBundle, Transferable):
    def __init__(
        self,
        model: Model,
        dataset: SampleDataset | dict[str, SampleDataset],
        losses: list[Loss],
        optimizer: Optimizer_,
        initializer: Initializer | None = None,
        train_augmentation: Augmentation | list[Augmentation] | None = None,
        val_augmentation: Augmentation | list[Augmentation] | None = None,
        test_augmentation: Augmentation | list[Augmentation] | None = None,
        lr_scheduler: Scheduler | None = None,
        global_partition_function: Partition | dict[str, Partition] | None = None,
        global_balancing_function: BalancingDistribution
        | dict[str, BalancingDistribution]
        | None = None,
        global_partition_seed: int = 42,
        local_partition_base_dataset: str = "dataset",
        local_partition_function: Partition | dict[str, Partition] | None = None,
        local_balancing_function: BalancingDistribution
        | dict[str, BalancingDistribution]
        | None = None,
        local_partition_seed: int = 42,
        datasampler: DataSampler | None = None,
        clipper: GradientClipper | None = None,
        train_key: str = "dataset:train",
        val_key: str = "dataset:val",
        test_key: str = "dataset:test",
        sample: bool = False,
        simultaneous_execution: int = 0,
    ) -> None:
        """The DefaultTrainingInitInstructionBundle is a convenience class for creating a sequence of instructions that initialize a model for training.

        Args:
            model (Model): The model to train.
            dataset (SampleDataset | dict[str, SampleDataset]): The dataset to use for training. If a dict is passed, it must contain the keys "train", "val" and "test".
            losses (list[Loss]): The losses to use for training.
            optimizer (Optimizer): The optimizer to use for training.
            initializer (Initializer, optional): The initializer to use for initializing the model. Defaults to None.
            train_augmentation (Augmentation, optional): The augmentation to use for training. Defaults to None.
            val_augmentation (Augmentation, optional): The augmentation to use for validation. Defaults to None.
            test_augmentation (Augmentation, optional): The augmentation to use for testing. Defaults to None.
            lr_scheduler (LRScheduler, optional): The learning rate scheduler to use for training. Defaults to None.
            partition (Partition, optional): The partition to use for training. Defaults to None.
            train_key (str, optional): The key of the training dataset in the dataset dict. Defaults to "dataset:train".
            val_key (str, optional): The key of the validation dataset in the dataset dict. Defaults to "dataset:val".
            test_key (str, optional): The key of the testing dataset in the dataset dict. Defaults to "dataset:test".
        """

        # create the commands for the sequential command
        commands = [
            LoadDatasetCommand(dataset=dataset),
            InitModelCommand(model=model),
            SetLossesCommand(losses=losses),
            SetOptimizerCommand(optimizer=optimizer),
        ]

        if lr_scheduler:
            commands.append(SetLRSchedulerCommand(scheduler=lr_scheduler))

        if clipper:
            commands.append(
                SetResourceCommand(
                    key="clipper", resource=clipper, assert_type=GradientClipper
                )
            )

        if global_partition_function or global_balancing_function:
            if isinstance(global_partition_function, Partition) and isinstance(
                global_balancing_function, BalancingDistribution
            ):
                commands.append(
                    SetPartitionCommand(
                        partition_function=global_partition_function,
                        balancing_function=global_balancing_function,
                        seed=global_partition_seed,
                    )
                )
            elif isinstance(global_partition_function, dict) and isinstance(
                global_balancing_function, dict
            ):
                # check that the keys of the dict are the same as the keys of the dataset dict
                if not set(global_partition_function.keys()) == set(
                    global_balancing_function.keys()
                ):
                    raise ValueError(
                        "The keys of the global_partition_function and the global_balancing_function must be the same."
                    )

                if not set(global_partition_function.keys()) == set(dataset.keys()):
                    raise ValueError(
                        "The keys of the global_partition_function and the dataset dict must be the same."
                    )

                for key in global_partition_function:
                    commands.append(
                        SetPartitionCommand(
                            key=f"{local_partition_base_dataset}:{key}",
                            partition_function=global_partition_function[key],
                            balancing_function=global_balancing_function[key],
                            seed=local_partition_seed,
                        )
                    )
            else:
                raise ValueError(
                    "The local_partition_function and the local_balancing_function must either both be of type Partition and BalancingDistribution or both be of type dict[str, Partition] and dict[str, BalancingDistribution]."
                )

        if local_partition_function or local_balancing_function:
            if isinstance(local_partition_function, Partition) and isinstance(
                local_balancing_function, BalancingDistribution
            ):
                commands.append(
                    SetLocalPartitionCommand(
                        base_dataset=local_partition_base_dataset,
                        partition_function=local_partition_function,
                        balancing_function=local_balancing_function,
                        seed=local_partition_seed,
                    )
                )
            elif isinstance(local_partition_function, dict) and isinstance(
                local_balancing_function, dict
            ):
                # check that the keys of the dict are the same as the keys of the dataset dict
                if not set(local_partition_function.keys()) == set(
                    local_balancing_function.keys()
                ):
                    raise ValueError(
                        "The keys of the local_partition_function and the local_balancing_function must be the same."
                    )

                if not set(local_partition_function.keys()) == set(dataset.keys()):
                    raise ValueError(
                        "The keys of the local_partition_function and the dataset dict must be the same."
                    )

                for key in local_partition_function:
                    commands.append(
                        SetLocalPartitionCommand(
                            base_dataset=f"{local_partition_base_dataset}:{key}",
                            partition_function=local_partition_function[key],
                            balancing_function=local_balancing_function[key],
                            seed=local_partition_seed,
                        )
                    )
            else:
                raise ValueError(
                    "The local_partition_function and the local_balancing_function must either both be of type Partition and BalancingDistribution or both be of type dict[str, Partition] and dict[str, BalancingDistribution]."
                )

        if datasampler:
            commands.append(SetDataSamplerCommand(datasampler=datasampler))

        if train_augmentation:
            commands.append(
                SetAugmentationCommand(
                    augmentation=train_augmentation
                    if isinstance(train_augmentation, Augmentation)
                    else SequentialAugmentation(augmentations=train_augmentation),
                    key=train_key,
                )
            )
        if val_augmentation:
            commands.append(
                SetAugmentationCommand(
                    augmentation=val_augmentation
                    if isinstance(val_augmentation, Augmentation)
                    else SequentialAugmentation(augmentations=val_augmentation),
                    key=val_key,
                )
            )
        if test_augmentation:
            commands.append(
                SetAugmentationCommand(
                    augmentation=test_augmentation
                    if isinstance(test_augmentation, Augmentation)
                    else SequentialAugmentation(augmentations=test_augmentation),
                    key=test_key,
                )
            )

        if sample:
            commands.append(PlotSamplesCommand(10))

        super().__init__(
            [
                ClosedDistribution(
                    SequentialCommand(commands=commands),
                    simultaneous_execution=simultaneous_execution,
                ),
                InitGlobalModelAction(initializer=initializer),
            ]
        )
