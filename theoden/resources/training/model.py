from __future__ import annotations

from torch import nn

from theoden.resources.data import Batch

from ...common import Transferable
from ..data import Batch


class Model(Transferable, is_base_type=True):
    def __init__(self, **kwargs) -> None:
        pass

    def parse(self) -> Model:
        raise NotImplementedError("Please implement this method in a subclass.")

    def to(self, device: str) -> Model:
        raise NotImplementedError("Please implement this method in a subclass.")

    def parse_to(self, device: str) -> Model:
        return self.parse().to(device)

    def get_state_dict(self) -> dict:
        raise NotImplementedError("Please implement this method in a subclass.")

    def load_state_dict(self, state_dict: dict) -> Model:
        raise NotImplementedError("Please implement this method in a subclass.")

    def module(self) -> nn.Module:
        raise NotImplementedError("Please implement this method in a subclass.")

    def training_call(
        self, batch: Batch, label_key: str, modality: str = "image"
    ) -> Batch:
        raise NotImplementedError("Please implement this method in a subclass.")

    def eval_call(self, batch: Batch, modality: str = "image") -> Batch:
        raise NotImplementedError("Please implement this method in a subclass.")


class TorchModel(Model, Transferable):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: nn.Module

    def to(self, device: str) -> TorchModel:
        """Move the model to the specified device.

        Args:
            device (str): device name

        Returns:
            TorchModel: self
        """

        self.model.to(device)
        return self

    def get_state_dict(self) -> dict:
        """Get the state dict of the model.

        Returns:
            dict: state dict
        """

        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> TorchModel:
        """Load the state dict of the model.

        Args:
            state_dict (dict): state dict

        Returns:
            TorchModel: model with loaded state dict
        """
        self.model.load_state_dict(state_dict)
        return self

    def module(self) -> nn.Module:
        return self.model

    def training_call(
        self, batch: Batch, label_key: str | None = None, modality: str = "image"
    ) -> Batch:
        if label_key is not None:
            batch["_label"] = batch[label_key]
        return self.eval_call(batch, modality)

    def eval_call(self, batch: Batch, modality: str = "image") -> Batch:
        prediction = self.model(batch[modality])
        batch["_prediction"] = prediction
        return batch


# class TFModel(Model):
#     def __init__(self, **kwargs) -> None:
#         super().__init__(**kwargs)

#     def to(self, device: str) -> TFModel:
#         raise NotImplementedError("Please implement this method in a subclass.")

#     def get_state_dict(self) -> dict:
#         raise NotImplementedError("Please implement this method in a subclass.")

#     def load_state_dict(self, state_dict: dict) -> TFModel:
#         raise NotImplementedError("Please implement this method in a subclass.")

#     def module(self) -> tf.keras.Model:
#         return self.model
