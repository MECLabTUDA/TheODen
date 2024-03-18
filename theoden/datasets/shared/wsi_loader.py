import re
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from typing_extensions import Self

from ...common import Transferable
from ...resources.data import DatasetAdapter, Sample


class WSIDataset(Dataset, Transferable, is_base_type=True):
    def __init__(
        self,
        base_folder: str,
        depth_metadata: list[str] | None = None,
        image_folder_name: str = "images",
        mask_folder_name: str | None = "masks",
        image_mask_split_mode: Literal["root", "leaf"] = "root",
    ):
        """A dataset for whole slide images.

        If the image_mask_split_mode is "root", it is expected that in the base folder there is a folder for the images
        and the masks.
        If the image_mask_split_mode is "leaf", it is expected that in the leaf pre-leaf folder there is a folder for
        the images and the masks.

        Args:
            base_folder (str): The base folder of the dataset
            depth_metadata (list[str], optional): The metadata to use for the depth of the dataset. Defaults to None.
            image_folder_name (str, optional): The name of the folder containing the images. Defaults to "images".
            mask_folder_name (str, optional): The name of the folder containing the masks. Defaults to "masks".
            image_mask_split_mode (Literal["root", "leaf"], optional): The mode to use for splitting the image and mask folders. Defaults to "root".

        Examples:
            If the image_mask_split_mode is "root", the folder structure should look like this:
            base_folder
            ├── images
            │   ├── institute1
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   └── image3.png
            │   └── institute2
            │       └── ...
            └── masks
                ├── institute1
                │   ├── image1.png
                │   ├── image2.png
                │   └── image3.png
                └── institute2
                    └── ...


            If the image_mask_split_mode is "leaf", the folder structure should look like this:
            base_folder
            ├── institute1
            │   ├── images
            │   │   ├── image1.png
            │   │   ├── image2.png
            │   │   └── image3.png
            │   └── masks
            │       ├── image1.png
            │       ├── image2.png
            │       └── image3.png
            ├── institute2
            │   ├── ...
        """
        self.base_folder = (
            Path(base_folder) if isinstance(base_folder, str) else base_folder
        )
        self.image_folder_name = image_folder_name
        self.mask_folder_name = mask_folder_name
        self.depth_metadata = depth_metadata
        self.image_mask_split_mode = image_mask_split_mode

        self.files = []

    def init_after_deserialization(self) -> Self:
        # find all files. Start at the image folder if the image_mask_split_mode is "root" else start at the base folder
        self.files_ = self._recursive_find_files(
            (self.base_folder / self.image_folder_name)
            if self.image_mask_split_mode == "root"
            else self.base_folder
        )

        for file in self.files_:
            self.files.append(
                {
                    "image": file[0],
                    "mask": file[1],
                    "meta": {
                        key: val
                        for key, val in zip(
                            (
                                self.depth_metadata
                                if self.depth_metadata
                                else [] + ["wsi"]
                            ),
                            file[2],
                        )
                    },
                }
            )
        return self

    def _recursive_find_files(
        self,
        folder: str,
        extension: str | None = None,
        depth_folder: list[str] | None = None,
    ) -> list[tuple[str, str, list[str]]]:
        """Recursively finds all files in a folder and its subfolders

        Args:
            folder (str): The folder to search
            extension (str, optional): The extension of the files to search for. Defaults to None.

        Returns:
            list[tuple[str, str, list[str]]]: A list of tuples containing the image path, the mask path and the depth folder
        """
        files = []
        if depth_folder is None:
            depth_folder = []

        for file in Path(folder).iterdir():
            if file.is_file() and (extension is None or file.suffix == extension):
                if self.image_mask_split_mode == "root":
                    mask_path = Path(
                        self.base_folder,
                        self.mask_folder_name,
                        *depth_folder,
                    )
                    # find the one file with the image name in the name in the mask folder
                    for _mask_path in mask_path.iterdir():
                        if re.match(
                            f"{re.escape(file.stem)}.*{file.suffix}", _mask_path.name
                        ):
                            files.append(
                                (
                                    [self.image_folder_name, *depth_folder, file.name],
                                    [
                                        self.mask_folder_name,
                                        *depth_folder,
                                        _mask_path.name,
                                    ],
                                    depth_folder + [file.stem],
                                )
                            )
                            break

                elif self.image_mask_split_mode == "leaf":
                    mask_path = Path(
                        self.base_folder,
                        *depth_folder[:-1],
                        self.mask_folder_name,
                    )
                    # find the one file with the image name in the name in the mask folder
                    for _mask_path in mask_path.iterdir():
                        if re.match(
                            f"{re.escape(file.stem)}.*{file.suffix}", _mask_path.name
                        ):
                            files.append(
                                (
                                    [*depth_folder, file.name],
                                    [
                                        *depth_folder[:-1],
                                        self.mask_folder_name,
                                        _mask_path.name,
                                    ],
                                    depth_folder[:-1] + [file.stem],
                                )
                            )
                            break

            elif file.is_dir():
                if file.name == self.mask_folder_name:
                    continue
                # if the folder is not the mask folder, recursively search the folder
                files.extend(
                    self._recursive_find_files(
                        file.as_posix(), extension, depth_folder + [file.name]
                    )
                )

        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        pair = self.files[index]
        image_path = Path(self.base_folder, *pair["image"])

        # add the size of the image to the metadata
        return pair | (
            {"wsi_size": tifffile.TiffFile(image_path).pages[0].shape}
            if image_path.suffix in [".tif", ".tiff"]
            else {"wsi_size": Image.open(image_path).size}
        )

    def save_as_tif(self, folder: str, force_overwrite: bool = True) -> None:
        """Saves the dataset as tif files

        Args:
            folder (str): The folder to save the dataset to
            force_overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        """

        # check if folder exists
        if Path(folder).exists():
            if not force_overwrite:
                raise Exception(
                    f"The folder {folder} already exists. Set force_overwrite to True to overwrite the files."
                )

        for file in tqdm.tqdm(self.files, desc="Saving as tif files"):
            image_path = Path(self.base_folder, *file["image"])
            mask_path = Path(self.base_folder, *file["mask"])

            # open the image and mask
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # create the paths for the tif files
            image_path = Path(folder, *file["image"])
            mask_path = Path(folder, *file["mask"])

            # create the folder structure
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            # save the image and mask as tif files
            image.save(image_path.with_suffix(".tif"))
            mask.save(mask_path.with_suffix(".tif"))


class Tiling:
    def __init__(
        self,
        patch_size: int | tuple[int, int],
        overlap: float | tuple[float, float] = 0.0,
    ):
        self.patch_size = patch_size
        self.overlap = overlap

    def get_tiles(self, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        return Tiling.tile_image(image_size, self.patch_size, self.overlap)

    @staticmethod
    def tile_image(
        image_size: tuple[int, int],
        patch_size: int | tuple[int, int],
        overlap: float | tuple[float, float] = 0.0,
    ) -> list[tuple[int, int, int, int]]:
        """Tiles an image into patches

        Args:
            image_size (tuple[int, int]): The size of the image
            patch_size (int | tuple[int, int]): The size of the patches
            overlap (float | tuple[float, float], optional): The overlap between the patches. Defaults to 0.0.

        Returns:
            list[tuple[int, int, int, int]]: A list of tuples containing the coordinates of the patches

        Examples:
            >>> tile_image((100, 100), 50, 0.5)
            [(0, 0, 50, 50), (25, 0, 75, 50), (50, 0, 100, 50), (0, 25, 50, 75), (25, 25, 75, 75), (50, 25, 100, 75), (0, 50, 50, 100), (25, 50, 75, 100), (50, 50, 100, 100)]
        """

        image_height, image_width, *_ = image_size

        if isinstance(overlap, float):
            overlap = (overlap, overlap)

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        patch_height, patch_width = patch_size
        overlap_height, overlap_width = overlap

        # Calculate the overlap in pixels
        overlap_height = int(patch_height * overlap_height)
        overlap_width = int(patch_width * overlap_width)

        # Calculate the number of patches
        num_patches_vertical = (
            int(
                np.ceil((image_height - patch_height) / (patch_height - overlap_height))
            )
            + 1
        )
        num_patches_horizontal = (
            int(np.ceil((image_width - patch_width) / (patch_width - overlap_width)))
            + 1
        )

        patches = []

        for i in range(num_patches_vertical):
            for j in range(num_patches_horizontal):
                # Calculate the starting coordinates of the patch
                start_x = j * (patch_width - overlap_width)
                start_y = i * (patch_height - overlap_height)

                # Calculate the ending coordinates of the patch
                end_x = min(start_x + patch_width, image_width)
                end_y = min(start_y + patch_height, image_height)

                # Adjust the dimensions for the edge patches
                if i == num_patches_vertical - 1:
                    end_y = image_height
                    start_y = end_y - patch_height

                if j == num_patches_horizontal - 1:
                    end_x = image_width
                    start_x = end_x - patch_width

                # Append the patch coordinates to the patches list
                patches.append((int(start_x), int(start_y), int(end_x), int(end_y)))

        return patches


class Exclusion:
    def __init__(
        self,
        class_amount: list[tuple[int, float]] | None = None,
        contains_classes: list[int] | None = None,
        min_num_classes: int | None = None,
        max_num_classes: int | None = None,
    ) -> None:
        self.class_amount = class_amount
        self.contains_classes = contains_classes
        self.min_num_classes = min_num_classes
        self.max_num_classes = max_num_classes

    def exclude(self, patch: np.ndarray) -> bool:
        if self.class_amount is not None:
            for class_, amount in self.class_amount:
                # check if percentage of class_ is higher than amount
                if np.sum(patch == class_) / patch.size > amount:
                    return True

        if self.contains_classes is not None:
            # check if patch contains any of the classes
            if np.any(
                [np.sum(patch == class_) > 0 for class_ in self.contains_classes]
            ):
                return True

        if self.min_num_classes is not None:
            # check if patch contains at least min_num_classes
            if (
                np.sum(
                    [np.sum(patch == class_) > 0 for class_ in self.contains_classes]
                )
                < self.min_num_classes
            ):
                return True

        if self.max_num_classes is not None:
            # check if patch contains at most max_num_classes
            if (
                np.sum(
                    [np.sum(patch == class_) > 0 for class_ in self.contains_classes]
                )
                > self.max_num_classes
            ):
                return True

        return False


class Mapping:
    def __init__(
        self,
        map_classes: list[tuple[list[int], int]] | None = None,
        shift: int = 0,
        except_map: tuple[list[int], int] | None = None,
    ) -> None:
        self.map_classes = map_classes
        self.shift = shift
        self.except_map = except_map

    def apply_mapping(self, patch_mask: np.ndarray) -> np.ndarray:
        shifted_mask = patch_mask + self.shift

        if self.map_classes is not None:
            for classes, new_class in self.map_classes:
                for class_ in classes:
                    shifted_mask[shifted_mask == class_] = new_class

        if self.except_map is not None:
            # map all except the except classes to the new value
            except_classes, new_class = self.except_map
            # if a class is not in except_classes, map it to new_class
            shifted_mask[np.isin(shifted_mask, except_classes, invert=True)] = new_class

        return shifted_mask


class TifReader:
    def read_patch(
        self, file: str | Path, patch: tuple[int, int, int, int]
    ) -> np.ndarray:
        mmapped_image = tifffile.memmap(file)
        start_x, start_y, end_x, end_y = patch
        return mmapped_image[start_y:end_y, start_x:end_x].squeeze()


class TiledWSIDataset(Dataset, Transferable, is_base_type=True):
    def __init__(
        self,
        wsi_dataset: WSIDataset,
        tiling_strategy: Tiling,
        exclusion_strategy: Exclusion | None = None,
        mapping_strategy: Mapping | None = None,
        tif_reader: TifReader | None = None,
    ) -> None:
        self.wsi_dataset = wsi_dataset
        self.tiling_strategy = tiling_strategy
        self.exclusion_strategy = exclusion_strategy
        self.mapping_strategy = mapping_strategy
        self.tif_reader = tif_reader
        self.tiles = []

    def init_after_deserialization(self) -> Transferable:
        self.wsi_dataset = self.wsi_dataset.init_after_deserialization()
        for index, sample in enumerate(tqdm.tqdm(self.wsi_dataset, desc="Tiling")):
            tiles = self.tiling_strategy.get_tiles(sample["wsi_size"])

            for tile in tiles:
                self.tiles.append(
                    {
                        "base_index": index,
                        "patch": tile,
                    }
                )

        if self.tif_reader is None:
            self.tif_reader = TifReader()
        else:
            self.tif_reader = self.tif_reader

        if self.exclusion_strategy is not None:
            for tile in tqdm.tqdm(self.tiles, desc="Excluding tiles"):
                sample = self.wsi_dataset[tile["base_index"]]
                mask_path = Path(self.wsi_dataset.base_folder, *sample["mask"])
                mask_patch = self.tif_reader.read_patch(mask_path, tile["patch"])

                if self.exclusion_strategy.exclude(mask_patch):
                    self.tiles.remove(tile)
        return self

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, Image.Image | np.ndarray | dict]:
        tile = self.tiles[index]
        sample = self.wsi_dataset[tile["base_index"]]

        image_path = Path(self.wsi_dataset.base_folder, *sample["image"])
        if "mask" in sample:
            mask_path = Path(self.wsi_dataset.base_folder, *sample["mask"])
            mask_patch = self.tif_reader.read_patch(mask_path, tile["patch"])
        image_patch = self.tif_reader.read_patch(image_path, tile["patch"])

        if self.mapping_strategy is not None:
            mask_patch = self.mapping_strategy.apply_mapping(mask_patch)

        return {
            "image": image_patch,
            "meta": sample["meta"],
        } | ({"mask": mask_patch} if "mask" in sample else {})


class AdaptedTiledWSIDataset(DatasetAdapter, Transferable):
    def __init__(
        self,
        wsi_dataset: WSIDataset,
        tiling_strategy: Tiling,
        exclusion_strategy: Exclusion | None = None,
        mapping_strategy: Mapping | None = None,
        name: str | None = None,
        tif_reader: TifReader | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            TiledWSIDataset(
                wsi_dataset=wsi_dataset,
                tiling_strategy=tiling_strategy,
                exclusion_strategy=exclusion_strategy,
                mapping_strategy=mapping_strategy,
                tif_reader=tif_reader,
            ),
            name,
            **kwargs,
        )
        self.to_tensor = ToTensor()

    def get_sample(self, index: int) -> Sample:
        sample = self.dataset[index]

        return Sample(
            data=(
                {
                    "image": self.to_tensor(sample["image"]),
                }
                | (
                    {"segmentation_mask": torch.from_numpy(sample["mask"]).long()}
                    if "mask" in sample
                    else {}
                )
            ),
            metadata=sample["meta"] | {"dataset": self.name},
        )

    def save_as_tif(self, folder: str, force_overwrite: bool = True) -> None:
        self.dataset.wsi_dataset.save_as_tif(folder, force_overwrite)

    def init_after_deserialization(self) -> Self:
        self.dataset = self.dataset.init_after_deserialization()
        return self
