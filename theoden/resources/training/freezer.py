from torch import nn

from ...common import Transferable


class LayerFreezer(Transferable, is_base_type=True):
    def __init__(self, freeze_layers: list[str]) -> None:
        """Create a layer freezer.

        Args:
            freeze_layers (list[str]): layers to freeze
        """
        self.freeze_layers = freeze_layers

    def freeze(self, model: nn.Module) -> None:
        """Freeze the layers of the model.

        Args:
            model (nn.Module): model
        """
        for name, param in model.named_parameters():
            if name in self.freeze_layers:
                param.requires_grad = False

    def unfreeze(self, model: nn.Module) -> None:
        """Unfreeze the layers of the model.

        Args:
            model (nn.Module): model
        """
        for name, param in model.named_parameters():
            if name in self.freeze_layers:
                param.requires_grad = True
