from ..common import GlobalContext, Transferable
from .shared.wsi_loader import AdaptedTiledWSIDataset, Tiling, WSIDataset


class SemiCOLDataset(AdaptedTiledWSIDataset, Transferable):
    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int] = 256,
        overlap: float | tuple[float, float] = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            WSIDataset(
                GlobalContext().get_dataset_path("semicol"),
                image_mask_split_mode="leaf",
                image_folder_name="image",
                mask_folder_name="mask",
                depth_metadata=["institute", "case"],
            ),
            tiling_strategy=Tiling(patch_size=patch_size, overlap=overlap),
            name="SemiCOL",
            **kwargs,
        )
