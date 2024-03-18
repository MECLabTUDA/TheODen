import torch
import tqdm

from ....common import MetricResponse
from ....resources import (
    DataSampler,
    GradientClipper,
    Loss,
    LRScheduler,
    Optimizer,
    SampleDataset,
    TorchModel,
)
from .. import Command


class TrainRoundCommand(Command):
    """Command to train a model on a dataset split."""

    def __init__(
        self,
        *,
        communication_round: int | None = None,
        num_epochs: int | None = None,
        num_steps: int | None = None,
        model_key: str = "model",
        optimizer_key: str = "optimizer",
        losses_key: str = "losses",
        scheduler_key: str = "scheduler",
        label_key: str = "class_label",
        clipper_key: str = "clipper",
        dataset_key: str = "dataset:train",
        datasampler_key: str = "dataset:train_sampler",
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
        self.optimizer_key = optimizer_key
        self.losses_key = losses_key
        self.scheduler_key = scheduler_key
        self.label_key = label_key
        self.dataset_key = dataset_key
        self.clipper_key = clipper_key
        self.datasampler_key = datasampler_key
        self.batch_size = batch_size
        self.num_workers = num_workers

        # check that exactly one of num_epochs and num_steps is set
        if (num_epochs is None and num_steps is None) or (
            num_epochs is not None and num_steps is not None
        ):
            raise ValueError(
                "Exactly one of num_epochs and num_steps must be set, but both are None or not None."
            )

    def _load_resources(
        self,
    ) -> tuple[
        TorchModel,
        LRScheduler,
        list[Loss],
        str,
        Optimizer,
        SampleDataset,
        GradientClipper | None,
        DataSampler | None,
    ]:
        return (
            self.client_rm.gr(self.model_key, TorchModel),
            self.client_rm.gr(self.scheduler_key, LRScheduler, default=None),
            self.client_rm.gr(self.losses_key, list[Loss]),
            self.client_rm.gr("device", str),
            self.client_rm.gr(self.optimizer_key, Optimizer),
            self.client_rm.gr(self.dataset_key, SampleDataset),
            self.client_rm.gr(self.clipper_key, GradientClipper, default=None),
            self.client_rm.gr(self.datasampler_key, DataSampler, default=None),
        )

    def execute(self) -> MetricResponse:
        # gather all required resource_manager from the client
        (
            model,
            scheduler,
            losses,
            device,
            optimizer,
            dataset,
            clipper,
            sampler,
        ) = self._load_resources()

        dataloader = dataset.get_dataloader(
            shuffle=sampler is None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=(
                sampler.sampler(dataset.get_dataset_chain(True))
                if sampler is not None
                else None
            ),
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

                loss = model.calc_loss(
                    batch=batch,
                    losses=losses,
                    label_key=self.label_key,
                    train=True,
                    communication_round=self.communication_round,
                )

                # display mean of losses as tqdm postfix
                post = Loss.create_dict(losses)
                if scheduler is not None:
                    post["lr"] = scheduler.get_last_lr()[0]
                pbar.set_postfix(post)

                optimizer.zero_grad()

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
