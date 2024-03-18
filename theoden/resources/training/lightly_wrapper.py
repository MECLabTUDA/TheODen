import numpy as np
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules import heads
from lightly.transforms import SimCLRTransform
from PIL import Image
from torch import Tensor

from theoden.resources.data.sample import Sample

from ..data import Augmentation
from ..data.sample import Batch
from .loss import Loss
from .model import Model, WrappedTorchModel


# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, x):
        features = self.backbone(x)[-1].flatten(start_dim=1)
        z = self.projection_head(features)
        return z


class SimCLRWrapperModel(WrappedTorchModel):
    def __init__(
        self,
        wrapped_model: Model,
        input_dim: int = 51200,
        hidden_dim: int = 512,
        output_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(wrapped_model, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def parse(self) -> Model:
        self.model = SimCLR(
            backbone=self.wrapped_model.module(),
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )
        return self

    def calc_loss(
        self,
        batch: Batch,
        losses: list[Loss],
        label_key: str,
        train: bool,
        communication_round: int,
    ) -> Tensor:
        if train:
            view1 = self.training_call(batch, modality="image")["_prediction"]
            view2 = self.training_call(batch, modality="_lightly_augmentation")[
                "_prediction"
            ]
        else:
            view1 = self.eval_call(batch, modality="image")["_prediction"]
            view2 = self.eval_call(batch, modality="_lightly_augmentation")[
                "_prediction"
            ]

        losses[0].append_batch_prediction(
            batch, [view1, view2], label_key, communication_round, not train
        )

        return Loss.create_combined_loss(losses)


class LightlyLossWrapper(Loss):
    def __init__(
        self,
        lightly_loss,
        train: bool = True,
        choosing_criterion: bool = True,
        factor: float = 1,
    ) -> None:
        super().__init__(train, choosing_criterion, factor)
        self.lightly_loss = lightly_loss

    def display_name(self) -> str:
        return f"{self.lightly_loss.__class__.__name__}"

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        self.set_epoch_loss(self.lightly_loss(prediction[0], prediction[1]))
        self.sum += self.get_epoch_loss().detach().cpu()
        self.num_total += 1

    def get(self) -> float | dict[str, float]:
        return self.sum.item() / self.num_total

    def reset(self) -> None:
        self.sum = 0
        self.num_total = 0


class NTXentLossWrapper(LightlyLossWrapper):
    def __init__(self, temperature: float = 0.5, **kwargs) -> None:
        super().__init__(NTXentLoss(temperature=temperature), **kwargs)


class LightlyAugmentationWrapper(Augmentation):
    def __init__(self, lightly_augmentation) -> None:
        self.lightly_augmentation = lightly_augmentation

    def _augment(self, sample: Sample) -> Sample:
        numpy_img = self._transform_to_numpy(sample["image"]).astype(np.uint8)

        pillow_img = Image.fromarray(numpy_img)

        augmented = self.lightly_augmentation(pillow_img)

        sample["_lightly_augmentation"] = torch.from_numpy(np.array(augmented[1]))
        sample["image"] = torch.from_numpy(np.array(augmented[0]))

        return sample


class SimCLRTransformWrapper(LightlyAugmentationWrapper):
    def __init__(self, **kwargs) -> None:
        super().__init__(SimCLRTransform(**kwargs))
