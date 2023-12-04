import torch
import tqdm

from theoden.resources import (
    Loss,
    SampleDataset,
    LRScheduler,
    Optimizer,
    Model,
    DataSampler,
    GradientClipper,
)
from theoden.operations.commands import Command
from theoden.common import Transferable, MetricResponse


class TrainRoundCommand(Command, Transferable):
    """Command to train a model on a dataset split."""

    def __init__(
        self,
        *,
        communication_round: int | None = None,
        num_epochs: int | None = None,
        num_steps: int | None = None,
        model_key: str = "model",
        label_key: str = "class_label",
        batch_size: int = 32,
        num_workers: int = 6,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Command to train a model on a dataset.

        Args:
            communication_round (int | None, optional): The communication round of the command. Defaults to None.
            num_epochs (int | None, optional): The number of epochs to train. Defaults to None.
            num_steps (int | None, optional): The number of steps to train. Defaults to None.
            model_key (str, optional): The key of the model to use for training. Defaults to "model".
            label_key (str, optional): The key of the label to use for training. Defaults to "class_label".
            batch_size (int, optional): The batch size to use for training. Defaults to 32.
            num_workers (int, optional): The number of workers to use for training. Defaults to 6.
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """

        super().__init__(uuid=uuid, **kwargs)
        self.communication_round = communication_round
        self.num_epochs = num_epochs
        self.num_steps = num_steps

        self.model_key = model_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers

        # check that exactly one of num_epochs and num_steps is set
        if (num_epochs is None and num_steps is None) or (
            num_epochs is not None and num_steps is not None
        ):
            raise ValueError(
                "Exactly one of num_epochs and num_steps must be set, but both are None or not None."
            )

    def execute(self) -> MetricResponse:
        # gather all required resource_manager from the node
        model = self.node_rm.gr(self.model_key, Model)
        scheduler = self.node_rm.gr("scheduler", LRScheduler)
        losses = self.node_rm.gr("losses", list[Loss])
        device = self.node_rm.gr("device", str)
        optimizer = self.node_rm.gr("optimizer", Optimizer)
        dataset = self.node_rm.gr("dataset:train", SampleDataset)

        clipper = self.node_rm.gr("clipper", assert_type=GradientClipper, default=None)

        sampler = self.node_rm.gr(
            "dataset:train_sampler", assert_type=DataSampler, default=None
        )

        dataloader = dataset.get_dataloader(
            shuffle=sampler is None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler.sampler(dataset.get_dataset_chain(True))
            if sampler is not None
            else None,
        )

        # add number of epochs and number of steps to one total step
        total_steps = (
            self.num_epochs * len(dataloader)
            if self.num_steps is None
            else self.num_steps
        )

        pbar = tqdm.tqdm(total=total_steps, desc=f"Round {self.communication_round}")

        # faster training
        torch.backends.cudnn.benchmark = True

        model.module().train()
        model.to(device)

        # reset losses for new epoch
        Loss.reset(losses)

        current_step = 0

        while current_step < total_steps:
            # iterate over data
            for batch in dataloader:
                batch.to(device)

                output = model.training_call(batch, label_key=self.label_key)[
                    "_prediction"
                ]

                # append current batch to losses
                for loss in losses:
                    loss.append_batch_prediction(
                        batch,
                        output,
                        "_label" if "_label" in batch else batch[self.label_key],
                        self.communication_round,
                        False,
                    )

                # display mean of losses as tqdm postfix
                post = Loss.create_dict(losses)
                post["lr"] = scheduler.get_last_lr()[0]
                pbar.set_postfix(post)

                optimizer.zero_grad()

                # create combined loss based on losses
                loss = Loss.create_combined_loss(losses)

                # clip gradients
                if clipper is not None:
                    clipper.clip(model.module())

                loss.backward()

                optimizer.step()

                pbar.update(1)
                current_step += 1

                if current_step >= total_steps:
                    break

        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        return MetricResponse(
            metrics=Loss.create_dict(losses),
            comm_round=self.communication_round,
            metric_type="train",
        )
