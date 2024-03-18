import torch
import tqdm

from ....common import MetricResponse
from ....resources import Loss, SampleDataset, TorchModel
from ...commands import Command


class ValidateEpochCommand(Command):
    """Command to validate a model on a dataset split."""

    def __init__(
        self,
        communication_round: int | None = None,
        batch_size: int = 32,
        num_workers: int = 6,
        split: str = "val",
        model_key: str = "model",
        metric_prefix: str = "",
        metric_postfix: str = "",
        label_key: str = "class_label",
        *,
        uuid: str | None = None,
        **kwargs,
    ) -> None:
        """Command to validate a model on a dataset.

        Args:
            communication_round (int | None, optional): The communication round of the command. Defaults to None.
            batch_size (int, optional): The batch size to use for validation. Defaults to 32.
            num_workers (int, optional): The number of workers to use for validation. Defaults to 6.
            split (str, optional): The split to use for validation. Defaults to "val".
            model_key (str, optional): The key of the model to use for validation. Defaults to "model".
            metric_prefix (str, optional): The prefix to use for the metric. Defaults to "".
            metric_postfix (str, optional): The postfix to use for the metric. Defaults to "".
            label_key (str, optional): The key of the label to use for validation. Defaults to "class_label".
            uuid (str | None, optional): The uuid of the command. Defaults to None.
        """

        super().__init__(uuid=uuid, **kwargs)
        self.split = split
        self.model_key = model_key
        self.communication_round = communication_round
        self.metric_prefix = metric_prefix
        self.metric_postfix = metric_postfix
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers

    def execute(self) -> MetricResponse:
        """Validate a model on a dataset.

        Returns:
            MetricResponse: A MetricResponse containing the metrics of the validation.
        """

        with torch.no_grad():
            # gather all required resource_manager from the client
            model = self.client_rm.gr(self.model_key, TorchModel)
            losses = self.client_rm.gr("losses", list[Loss])
            device = self.client_rm.gr("device", str)
            dataset = self.client_rm.gr(f"dataset:{self.split}", SampleDataset)
            dataloader = dataset.get_dataloader(
                shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers
            )

            model.module().eval()

            # reset losses for new epoch
            Loss.reset(losses)

            t = tqdm.tqdm(dataloader, desc=f"Validation")

            # iterate over data
            for batch in t:
                batch.to(device)

                model.calc_loss(
                    batch=batch,
                    losses=losses,
                    label_key=self.label_key,
                    train=False,
                    communication_round=self.communication_round,
                )
                t.set_postfix(Loss.create_dict(losses))

        return MetricResponse(
            metrics=Loss.create_dict(losses),
            metric_type=f"{self.metric_prefix}{self.split}{self.metric_postfix}",
            comm_round=self.communication_round,
        )
