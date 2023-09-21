import torch
import tqdm

from typing import Tuple, List, Type, Optional

from ...commands import Command
from ....common import Transferable, ExecutionResponse, MetricResponse
from ....resources import SampleDataset, Loss, Model


class ValidateEpochCommand(Command, Transferable):
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
        node: Optional["Node"] = None,
        uuid: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(node=node, uuid=uuid, **kwargs)
        self.split = split
        self.model_key = model_key
        self.communication_round = communication_round
        self.metric_prefix = metric_prefix
        self.metric_postfix = metric_postfix
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers

    def execute(self) -> MetricResponse:
        with torch.no_grad():
            # gather all required resources from the node
            model = self.node_rr.gr(self.model_key, Model)
            losses = self.node_rr.gr("losses", list[Loss])
            device = self.node_rr.gr("device", str)
            dataset = self.node_rr.gr(f"dataset:{self.split}", SampleDataset)
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

                output = model.eval_call(batch)["_prediction"]

                # append current batch to losses
                for loss in losses:
                    loss.append_batch_prediction(batch, output, self.label_key, 0, True)

                # display mean of losses as tqdm postfix
                post = Loss.create_dict(losses)
                t.set_postfix(post)

                loss = Loss.create_combined_loss(losses)

        return MetricResponse(
            metrics=Loss.create_dict(losses),
            metric_type=f"{self.metric_prefix}{self.split}{self.metric_postfix}",
            comm_round=self.communication_round,
        )
