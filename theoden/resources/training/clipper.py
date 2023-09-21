import torch

from ...common import Transferable


class GradientClipper(Transferable, is_base_type=True):
    def __init__(self, clip_value: float) -> None:
        self.clip_value = clip_value

    def clip(self, model: torch.nn.Module) -> None:
        """Clip the gradients of the model.

        Args:
            model (nn.Module): model
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
