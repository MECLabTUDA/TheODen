from ....common import Transferable
from ....resources import StateLoader
from ...commands import (
    CalculateClientScoreCommand,
    LoadStateDictCommand,
    SequentialCommand,
    TrainValNTimesCommand,
    ValidateEpochCommand,
)
from ..aggregation.aggregate import Aggregator
from ..distribution import ClosedDistribution
from ..instruction import Instruction
from ..selection import BinarySelector
from .bundle import InstructionBundle
from .default_aggregation import DefaultAggregationBundle


class MultiRoundTrainingInstructionBundle(InstructionBundle, Transferable):
    def __init__(
        self,
        n_rounds: int,
        aggregator: Aggregator,
        epochs_per_round: int | None = None,
        steps_per_round: int | None = None,
        selector: BinarySelector | None = None,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        num_workers: int = 6,
        pre_round_operations: list[Instruction | InstructionBundle] | None = None,
        post_round_operations: list[Instruction | InstructionBundle] | None = None,
        client_score_command: CalculateClientScoreCommand | None = None,
        final_validation: bool = True,
        val_split: str = "val",
        start_at_round: int = 0,
        start_with_val: bool = True,
        end_with_val: bool = False,
        validate_every_n_rounds: int | None = None,
        start_n_rounds_without_validation: int = 0,
        model_key: str = "model",
        label_key: str = "class_label",
        loader: type[StateLoader] | None = None,
        only_grad: bool = False,
        simultaneous_execution: int = 0,
    ) -> None:
        """The MultiRoundTrainingInstructionBundle is a convenience class for creating a sequence of instructions that train a model for a given number of rounds.

        Args:
            n_rounds (int): The number of rounds to train the model for.
            aggregator (Aggregator): The aggregator to use for aggregating the model updates.
            epochs_per_round (int, optional): The number of epochs to train the model for in each round. Defaults to None.
            steps_per_round (int, optional): The number of steps to train the model for in each round. Defaults to None.
            distributor (Distributor, optional): The distributor to use for selecting the clients to train on. Defaults to None.
            pre_round_operations (list[Instruction | InstructionBundle], optional): A list of instructions to execute before each round. Defaults to None.
            post_round_operations (list[Instruction | InstructionBundle], optional): A list of instructions to execute after each round. Defaults to None.
            final_validation (bool, optional): Whether to perform a final validation on the test set after training. Defaults to True.
            val_split (str, optional): The split to use for validation. Defaults to "val".
            start_at_round (int, optional): The round to start at. Defaults to 0.
            start_with_val (bool, optional): Whether to start with a validation step. Defaults to False.
            end_with_val (bool, optional): Whether to end with a validation step. Defaults to False.
            validate_every_n_rounds (int, optional): Whether to validate after every n rounds. Defaults to None.
            start_n_rounds_without_validation (int, optional): The number of rounds to start without validation. Defaults to 0.
            label_key (str, optional): The key of the label in the dataset. Defaults to "class_label".
            simultaneous_execution (int, optional): The number of client that simultaneous executions of a command. Defaults to 0 (all clients).

        Raises:
            ValueError: If epochs_per_round and steps_per_round are both None or both not None.
        """
        super().__init__(
            [
                item
                for sublist in [
                    [
                        *(pre_round_operations if pre_round_operations else []),
                        DefaultAggregationBundle(
                            selector=selector,
                            train_command=TrainValNTimesCommand(
                                n_epochs=epochs_per_round,
                                n_steps=steps_per_round,
                                val_split=val_split,
                                communication_round=start_at_round + i + 1,
                                start_with_val=start_with_val,
                                end_with_val=end_with_val,
                                model_key=model_key,
                                label_key=label_key,
                                train_batch_size=train_batch_size,
                                validation_batch_size=validation_batch_size,
                                num_workers=num_workers,
                                validate=(
                                    i % validate_every_n_rounds
                                    == validate_every_n_rounds - 1
                                    if validate_every_n_rounds
                                    else True
                                )
                                and i >= start_n_rounds_without_validation,
                            ),
                            client_score_command=client_score_command,
                            aggregator=aggregator,
                            simultaneous_execution=simultaneous_execution,
                            loader=loader,
                            only_grad=only_grad,
                            model_keys=[model_key],
                        ),
                        *(post_round_operations if post_round_operations else []),
                    ]
                    for i in range(n_rounds)
                ]
                for item in sublist
            ]
            + (
                [
                    ClosedDistribution(
                        SequentialCommand(
                            [
                                LoadStateDictCommand(
                                    "model",
                                    checkpoint_key=(
                                        "model_best_val"
                                        if not start_with_val
                                        else "model_best_agg_val"
                                    ),
                                    loader=loader,
                                ),
                                ValidateEpochCommand(
                                    0,
                                    split="test",
                                    label_key=label_key,
                                    batch_size=validation_batch_size,
                                    num_workers=num_workers,
                                ),
                            ]
                        ),
                        simultaneous_execution=simultaneous_execution,
                    )
                ]
                if final_validation
                else []
            )
        )
        print(final_validation)
        print(self.instructions[-1].commands[0].commands)
