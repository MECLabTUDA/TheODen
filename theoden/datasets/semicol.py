from pathlib import Path

from numpy import ndarray
from typing_extensions import Self

from ..common import GlobalContext
from ..resources import SampleDataset
from .shared.wsi_loader import (
    AdaptedTiledWSIDataset,
    TifReader,
    TiledWSIDataset,
    Tiling,
    WSIDataset,
)


class SemiCOLDataset(AdaptedTiledWSIDataset):
    def __init__(
        self,
        *,
        folder: str | None = None,
        patch_size: int | tuple[int, int] = 256,
        overlap: float | tuple[float, float] = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            WSIDataset(
                GlobalContext().get_dataset_path(
                    "semicol", default_to_none=True, parameter_path=folder
                ),
                image_mask_split_mode="leaf",
                image_folder_name="image",
                mask_folder_name="mask",
                depth_metadata=["institute", "case"],
            ),
            tiling_strategy=Tiling(patch_size=patch_size, overlap=overlap),
            name="SemiCOL",
            **kwargs,
        )


class WeaklyWSIDataset(WSIDataset):
    def __init__(self, base_folder: str) -> None:
        super().__init__(base_folder=base_folder, image_mask_split_mode="leaf")

    def init_after_deserialization(self) -> Self:
        for clinic_scanner in self.base_folder.iterdir():
            if clinic_scanner.is_dir():
                for label in clinic_scanner.iterdir():
                    if label.is_dir():
                        for slide in label.iterdir():
                            if slide.is_file():
                                self.files.append(
                                    {
                                        "image": [
                                            clinic_scanner.name,
                                            label.name,
                                            slide.name,
                                        ],
                                        "meta": {
                                            "institute": clinic_scanner.name,
                                            "label": label.name,
                                            "tif": slide.name,
                                        },
                                    }
                                )

        return self


class SemiCOLWeaklyDataset(AdaptedTiledWSIDataset):
    def __init__(
        self,
        folder: str | None = None,
        patch_size: int | tuple[int, int] = 256,
        overlap: float | tuple[float, float] = 0.1,
    ) -> None:
        super().__init__(
            WeaklyWSIDataset(
                GlobalContext().get_dataset_path(
                    "semicol_weak", default_to_none=True, parameter_path=folder
                ),
            ),
            tiling_strategy=Tiling(patch_size=patch_size, overlap=overlap),
            name="SemiCOLWeakly",
        )
