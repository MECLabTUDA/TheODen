from .tv_datasets import CIFAR10_Adapted, CIFAR100_Adapted, MNIST_Adapted, CelebADataset
from .bcss import BCSSDataset
from .medmnist import PathMNIST_Adapted
from .camelyon17 import Camelyon17_Adapted
from .shared.wsi_loader import (
    AdaptedTiledWSIDataset,
    WSIDataset,
    TiledWSIDataset,
    Tiling,
)
from .semicol import SemiCOLDataset, SemiCOLWeaklyDataset
from .crc import CRCDataset
