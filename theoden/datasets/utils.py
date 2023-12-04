import zarr
import torch
import warnings
import rasterio
import tifffile

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
import tifffile as tiff

import random


def read_tif_region(
    file: str,
    from_x: int | None = None,
    to_x: int | None = None,
    from_y: int | None = None,
    to_y: int | None = None,
):
    if from_x == None:
        store = tiff.imread(file, aszarr=True)
        out_labels_slide = zarr.open(store, mode="r")
        return out_labels_slide

    mmapped_image = tifffile.memmap(file)
    return mmapped_image[from_y:to_y, from_x:to_x].squeeze()


def random_crop(img, mask, height, width):
    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)
    img = img[:, y : y + height, x : x + width]
    mask = mask[y : y + height, x : x + width]
    return (img, mask)


def to_tensor(array, mask: bool = False):
    if not mask:
        return torch.from_numpy(array).permute(2, 0, 1)
    else:
        return torch.from_numpy(array)
