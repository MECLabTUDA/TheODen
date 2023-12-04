from ....common import Transferable
from ..meta.sequential import SequentialCommand
from ..action import TrainRoundCommand, ValidateEpochCommand


class TrainValNTimesCommand(
    SequentialCommand, Transferable, return_super_class_dict=True
):
    def __init__(
        self,
        n_epochs: int | None = None,
        n_steps: int | None = None,
        communication_round: int | None = None,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        num_workers: int = 6,
        val_split: str = "val",
        start_with_val: bool = True,
        end_with_val: bool = True,
        label_key: str = "class_label",
        validate: bool = True,
        **kwargs,
    ) -> None:
        """The TrainValNTimesCommand is a convenience class for creating a sequence of commands that trains a model for a given number of epochs.

        Args:
            n_epochs (int): The number of epochs to train the model for. Defaults to None.
            n_steps (int, optional): The number of steps to train the model for. Defaults to None.
            communication_round (int, optional): The communication round to use for logging. Defaults to None.
            train_batch_size (int, optional): The batch size to use for training. Defaults to 32.
            validation_batch_size (int, optional): The batch size to use for validation. Defaults to 32.
            num_workers (int, optional): The number of workers to use for data loading. Defaults to 6.
            val_split (str, optional): The split to use for validation. Defaults to "val".
            start_with_val (bool, optional): Whether to start with a validation step. Defaults to True.
            label_key (str, optional): The key to use for the label. Defaults to "class_label".

        Raises:
            ValueError: If neither n_epochs nor n_steps is specified.
        """

        # create a list of commands that train and validate the model for n_epochs

        cmds = [
            TrainRoundCommand(
                communication_round=communication_round,
                num_epochs=n_epochs,
                num_steps=n_steps,
                label_key=label_key,
                batch_size=train_batch_size,
                num_workers=num_workers,
            ),
        ]
        if validate and end_with_val:
            cmds.append(
                ValidateEpochCommand(
                    communication_round=communication_round,
                    split=val_split,
                    label_key=label_key,
                    batch_size=validation_batch_size,
                    num_workers=num_workers,
                )
            )

        commands = [SequentialCommand(cmds)]

        # add validation at the end if start_with_val is True
        if start_with_val and validate:
            commands = [
                ValidateEpochCommand(
                    communication_round=communication_round,
                    split=val_split,
                    metric_prefix="agg_",
                    label_key=label_key,
                    batch_size=validation_batch_size,
                    num_workers=num_workers,
                )
            ] + commands
        super().__init__(commands, **kwargs)
