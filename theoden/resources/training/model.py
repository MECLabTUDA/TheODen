from __future__ import annotations

from torch import Tensor, nn
from typing_extensions import Self

from ...common import Transferable
from ..data import Batch
from .loss import Loss


class Model(Transferable, is_base_type=True):
    def __init__(self, **kwargs) -> None:
        pass

    def parse(self) -> Model:
        raise NotImplementedError("Please implement this method in a subclass.")

    def to(self, device: str) -> Self:
        raise NotImplementedError("Please implement this method in a subclass.")

    def parse_to(self, device: str) -> Self:
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

    def set_model(self, model: nn.Module) -> Self:
        self.model = model
        return self

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

    def get_grad_params_state_dict(self) -> dict:
        grad_params_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params_state_dict[name] = param.data
        return grad_params_state_dict

    def load_state_dict(self, state_dict: dict, **kwargs) -> Self:
        """Load the state dict of the model.

        Args:
            state_dict (dict): state dict

        Returns:
            TorchModel: model with loaded state dict
        """
        self.model.load_state_dict(state_dict, **kwargs)
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

    def calc_loss(
        self,
        batch: Batch,
        losses: list[Loss],
        label_key: str,
        train: bool,
        communication_round: int,
    ) -> Tensor:
        if train:
            output = self.training_call(batch, label_key=label_key)["_prediction"]
        else:
            output = self.eval_call(batch)["_prediction"]

        # append current batch to losses
        for loss in losses:
            loss.append_batch_prediction(
                batch,
                output,
                "_label" if "_label" in batch else label_key,
                communication_round,
                test=not train,
            )

        return Loss.create_combined_loss(losses)


class WrappedTorchModel(TorchModel):
    def __init__(self, wrapped_model: Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wrapped_model = wrapped_model
