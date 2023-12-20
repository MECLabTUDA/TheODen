from __future__ import annotations

from torch import nn

from ..common import Transferable
from ..resources.training.model import TorchModel


class TimmModel(TorchModel, Transferable):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_chans: int,
        modify_first_layer: bool = True,
        **params,
    ) -> None:
        """Model from timm library

        Args:
            model_name (str): model name from timm library
            num_classes (int): number of classes
            in_chans (int): number of input channels
            modify_first_layer (bool, optional): modify first layer to match smaller image size. Defaults to True.
            **params: additional parameters

        Examples:
            >>> TimmModel("resnet18", 10, 3)
        """
        self.params = {
            "model_name": model_name,
            "num_classes": num_classes,
            "in_chans": in_chans,
            **params,
        }
        self.modify_first_layer = modify_first_layer

    def parse(self) -> TorchModel:
        import timm

        self.model = timm.create_model(**self.params)
        if self.modify_first_layer:
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.model.maxpool = nn.Identity()
        return self


class SMPModel(TorchModel, Transferable):
    def __init__(
        self,
        architecture: str,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        **params,
    ) -> None:
        """Model from segmentation_models_pytorch library

        Args:
            architecture (str): architecture name
            encoder_name (str, optional): encoder name. Defaults to "resnet34".
            encoder_weights (str | None, optional): encoder weights. Defaults to "imagenet".
            in_channels (int, optional): number of input channels. Defaults to 3.
            classes (int, optional): number of classes. Defaults to 1.
            **params: additional parameters

        Examples:
            >>> SMPModel("unet", "resnet18", "imagenet", 3, 10)
        """
        self.params = {
            "arch": architecture,
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": classes,
            **params,
        }

    def parse(self) -> TorchModel:
        import segmentation_models_pytorch as smp

        self.model = smp.create_model(**self.params)
        return self
