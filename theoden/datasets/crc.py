import torch
from torchvision.transforms import Compose, ToTensor
import cv2

import os


from ..resources.data import DatasetAdapter, Sample, SampleDataset
from ..common import GlobalContext, Transferable


class CRC(SampleDataset, Transferable):
    def __init__(self, norm: bool = True, val: bool = False):
        if val:
            self.root_dir = GlobalContext().get_dataset_path("crc_val")
        else:
            self.root_dir = GlobalContext().get_dataset_path(
                "crc" if norm else "crc_nonorm"
            )
        self.images, self.labels = self._read_img_names()

    def _read_img_names(self) -> tuple[list[str], list[str]]:
        X = []
        y = []
        for root, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                img = os.path.join(root, filename)
                lbl = root.split("/")[-1]
                X.append(img)
                y.append(lbl)

        return X, y

    @staticmethod
    def _get_label(lbl):
        lbl_map = {
            "ADI": 0,
            "BACK": 1,
            "DEB": 2,
            "LYM": 3,
            "MUC": 4,
            "MUS": 5,
            "NORM": 6,
            "STR": 7,
            "TUM": 8,
        }
        return lbl_map[lbl]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        img = cv2.imread(self.images[index])
        lbl = self._get_label(self.labels[index])

        return Sample(
            {
                "image": torch.from_numpy(img).permute(2, 0, 1) / 255.0,
                "class_label": torch.tensor(lbl),
            }
        )


class CRCDataset(DatasetAdapter, Transferable):
    def __init__(self, norm: bool = True, val: bool = False, **kwargs) -> None:
        super().__init__(
            CRC(norm=norm, val=val), f"CRC{'_nonorm' if not norm else ''}", **kwargs
        )

    def get_sample(self, index: int) -> Sample:
        return self.dataset[index]
