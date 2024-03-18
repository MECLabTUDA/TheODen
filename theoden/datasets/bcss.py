from ..common import GlobalContext, Transferable
from .shared.wsi_loader import AdaptedTiledWSIDataset, Tiling, WSIDataset


class BCSSDataset(AdaptedTiledWSIDataset, Transferable):
    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int] = 256,
        overlap: float | tuple[float, float] = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            WSIDataset(
                GlobalContext().get_dataset_path("bcss", default_to_none=True),
                depth_metadata=["case"],
            ),
            tiling_strategy=Tiling(patch_size=patch_size, overlap=overlap),
            name="BCSS",
            **kwargs,
        )
