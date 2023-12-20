from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

# from segmentation_models_pytorch.losses import DiceLoss as SMPDiceLoss
from pytorch_toolbelt.losses import DiceLoss
from torch.nn import CrossEntropyLoss, Module
from torchmetrics import Dice
from torchmetrics.classification import MulticlassConfusionMatrix

from ...common import Transferable
from ...resources.data.sample import Batch
from ..data.sample import Batch


class Loss(ABC, Transferable, is_base_type=True):
    def __init__(
        self, train: bool = False, choosing_criterion: bool = False, factor: float = 1.0
    ) -> None:
        """Abstract implementation of a loss. Losses can be used to evaluate the performance of a model. They are also used for model training.

        Args:
            train (bool, optional): Whether the loss is used for training. Defaults to False.
            choosing_criterion (bool, optional): Whether the loss is used as a choosing criterion. Defaults to False.
            factor (float, optional): The factor to multiply the loss with. Defaults to 1.0.
        """
        self.factor = factor
        self.choosing_criterion = choosing_criterion
        self.train = train
        self.reset()
        self.higher_better = False

    @abstractmethod
    def display_name(self) -> str:
        """Get the display name for the loss

        Returns:
            str: name of the loss"""
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        """Append a prediction for one batch to the loss.

        This method is called by the trainer after each inference step. Here, the loss should be calculated and stored.

        Args:
            batch (Batch): The batch that was used for inference.
            prediction (list[torch.Tensor]): The prediction that was made by the model.
            label_key (str): The key of the label in the batch.
            epoch (int): The current epoch.
            test (bool, optional): Whether the loss is calculated during testing. Defaults to False.
        """
        raise NotImplementedError("Please Implement this method")

    def set_epoch_loss(self, value: torch.Tensor | float) -> None:
        """This methods sets the loss for the current epoch.

        Args:
            value (torch.Tensor | float): The loss for the current epoch.
        """
        self.epoch_loss = (
            value
            if (self.train or not isinstance(value, torch.Tensor))
            else value.detach().cpu()
        )

    def get_epoch_loss(self) -> torch.Tensor | float:
        """This methods returns the loss for the current epoch.

        Returns:
            torch.Tensor | float: The loss for the current epoch.
        """
        return self.epoch_loss

    @abstractmethod
    def get(self) -> float:
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def reset(losses: list[Loss]) -> None:
        """This method resets a list of losses. It will be called by the trainer before each epoch.

        Args:
            losses (list[Loss]): The losses to reset.
        """
        for l in losses:
            l.reset()

    @staticmethod
    def create_dict(
        losses: list[Loss], post: str = "", pre: str = ""
    ) -> dict[str, float]:
        """This method creates a dictionary from a list of losses. It is mainly used for visualization of the losses.

        Args:
            losses (list[Loss]): The losses to create the dictionary from.
            post (str, optional): A string to append to the loss name. Defaults to "".
            pre (str, optional): A string to prepend to the loss name. Defaults to "".

        Returns:
            dict[str, float]: The dictionary containing name: loss pairs.
        """
        d = {}
        for l in losses:
            d[f"{pre}{l.display_name()}{post}"] = l.get()
        return d

    @staticmethod
    def create_combined_loss(losses: list[Loss]) -> torch.Tensor | float:
        """This method creates a combined loss from a list of losses by summing them up with their factors.

        Args:
            losses (list[Loss]): The losses to combine.

        Returns:
            torch.Tensor | float: The combined loss.
        """
        loss = 0
        for l in losses:
            if l.train:
                loss += l.factor * l.get_epoch_loss()
        return loss

    @staticmethod
    def get_choosing_criterion(losses: list) -> Loss:
        possible = []
        for l in losses:
            if l.choosing_criterion:
                possible.append(l)
        assert len(possible) == 1, "Exactly one Loss needs must be choosing critereon"
        return possible[0]


class ComposedLoss(Loss, Transferable):
    def __init__(
        self,
        losses: list[Loss],
        name: str,
        train: bool = False,
        choosing_criterion: bool = False,
        factor: float = 1,
    ) -> None:
        super().__init__(train, choosing_criterion, factor)
        self.losses = losses
        self.name = name

    def display_name(self) -> str:
        return self.name

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        for l in self.losses:
            l.append_batch_prediction(batch, prediction, label_key, epoch, test)


class AccuracyLoss(Loss, Transferable):
    def __init__(self, choosing_criterion: bool = False, factor: float = 1.0) -> None:
        super().__init__(False, choosing_criterion, factor)
        self.higher_better = True

    def display_name(self) -> str:
        return "Acc"

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        self.set_epoch_loss(
            torch.max(prediction, dim=1)[1].eq(batch[label_key]).sum().item()
        )
        if self.exp_moving == -1:
            self.exp_moving = self.get_epoch_loss() / batch["image"].shape[0]

        self.exp_moving = 0.8 * self.exp_moving + 0.2 * (
            self.get_epoch_loss() / batch["image"].shape[0]
        )
        self.num_correct += self.get_epoch_loss()
        self.num_total += batch["image"].shape[0]

    def get(self) -> float:
        return self.num_correct / self.num_total

    def reset(self) -> None:
        self.exp_moving = -1
        self.num_correct = 0
        self.num_total = 0


class CELoss(Loss, Transferable):
    def __init__(
        self,
        weight: list[float] | None = None,
        ignore_index: int = -100,
        train: bool = True,
        choosing_criterion: bool = False,
        factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(train, choosing_criterion, factor)
        self.cross_entropy = CrossEntropyLoss(
            weight if weight == None else torch.tensor(weight).cuda(),
            ignore_index=ignore_index,
            **kwargs,
        )

    def display_name(self) -> str:
        return "CE"

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        self.set_epoch_loss(self.cross_entropy(prediction, batch[label_key].long()))
        # if self.exp_moving == 0:
        #     self.exp_moving = self.get_epoch_loss()
        # self.exp_moving = 0.8 * self.exp_moving + 0.2 * self.get_epoch_loss()
        self.sum += self.get_epoch_loss().detach().cpu()
        self.num_total += 1

    def get(self) -> float:
        # return self.exp_moving.item()
        return self.sum.item() / self.num_total

    def reset(self) -> None:
        # self.exp_moving = 0
        self.sum = 0
        self.num_total = 0


class MulticlassDiceLoss(Loss, Transferable):
    def __init__(
        self,
        ignore_index: int = -100,
        train: bool = True,
        choosing_criterion: bool = False,
        factor: float = 1.0,
    ) -> None:
        super().__init__(train, choosing_criterion, factor)

        self.dice_loss = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def display_name(self) -> str:
        return "DL"

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        self.set_epoch_loss(self.dice_loss(prediction, batch[label_key].long()))
        self.sum += self.get_epoch_loss().detach().cpu()
        self.num_total += 1

    def get(self) -> float:
        return self.sum.item() / self.num_total

    def reset(self) -> None:
        self.sum = 0
        self.num_total = 0


class MultiClassSegmentationMetric(MulticlassConfusionMatrix):
    def iou(
        self,
        average: bool = True,
        class_id: int = None,
    ) -> torch.Tensor:
        conf_matrix = self.compute()
        if class_id is not None:
            return (
                conf_matrix[class_id, class_id].float()
                / (
                    torch.sum(conf_matrix[:, class_id]).float()
                    + torch.sum(conf_matrix[class_id, :]).float()
                    - conf_matrix[class_id, class_id]
                ).float()
            )
        else:
            class_wise = conf_matrix.diag().float() / (
                torch.sum(conf_matrix, 0).float()
                + torch.sum(conf_matrix, 1).float()
                - conf_matrix.diag().float()
            )
            return class_wise if not average else class_wise.nanmean()

    def dice(
        self,
        average: bool = True,
        class_id: int = None,
    ) -> torch.Tensor:
        conf_matrix = self.compute()

        if class_id is not None:
            return (2 * conf_matrix[class_id, class_id].float()) / (
                torch.sum(conf_matrix[:, class_id]).float()
                + torch.sum(conf_matrix[class_id, :]).float()
            )
        else:
            class_wise = (
                2
                * conf_matrix.diag().float()
                / (
                    torch.sum(conf_matrix, 0).float()
                    + torch.sum(conf_matrix, 1).float()
                )
            )
            return class_wise if not average else class_wise.nanmean()


class DisplayDiceLoss(Loss, Transferable):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        train: bool = False,
        choosing_criterion: bool = False,
        factor: float = 1.0,
    ) -> None:
        self.dice_loss = MultiClassSegmentationMetric(
            num_classes=num_classes, ignore_index=ignore_index
        ).to("cuda")
        super().__init__(train, choosing_criterion, factor)
        self.higher_better = True

    def display_name(self) -> str:
        return "Dice"

    def append_batch_prediction(
        self,
        batch: Batch,
        prediction: list[torch.Tensor],
        label_key: str,
        epoch: int,
        test: bool = False,
    ) -> None:
        self.dice_loss.update(prediction.clone(), batch[label_key].long())
        self.set_epoch_loss(self.dice_loss.dice(average=True))

    def get(self) -> float:
        return self.dice_loss.dice(average=True).item()

    def reset(self) -> None:
        print(self.dice_loss.dice(False))
        self.dice_loss.reset()


class ClasswiseDiceLoss(DisplayDiceLoss, Transferable):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        train: bool = False,
        choosing_criterion: bool = False,
        factor: float = 1.0,
    ) -> None:
        super().__init__(train, choosing_criterion, factor)
        self.dice_loss = Dice(ignore_index=num_classes - 1, average=None).to("cuda")
        self.ignore_index = ignore_index
        self.new_ignore = num_classes - 1

    def display_name(self) -> str:
        return "CWDice"

    def get(self) -> float:
        return self.sum / self.num_total

    def reset(self) -> None:
        self.sum = torch.zeros(self.num_classes)
        self.num_total = 0
