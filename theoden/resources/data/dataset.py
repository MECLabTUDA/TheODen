from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset

from ...common import GlobalContext, Transferable
from ...common.utils import hash_dict
from .sample import Batch, Sample, sample_collate

if TYPE_CHECKING:
    from .concat import ConcatSampleDataset
    from .exclusion import Exclusion, ExclusionDataset


class SampleDataset(Dataset, ABC, Transferable, is_base_type=True):
    def load_fingerprint(self, folder: str) -> dict:
        fingerprint_hash = self.initialization_hash()
        path = Path(folder) / f"{fingerprint_hash}.json"
        if not path.exists():
            raise FileNotFoundError(f"Could not find fingerprint {fingerprint_hash}.")
        with open(path, "r") as f:
            fingerprint = json.load(f)
        return fingerprint

    def save_fingerprint(self, folder: str, additional_fields: dict):
        fingerprint_hash = self.initialization_hash()
        fingerprint = self.fingerprint()
        path = Path(folder) / f"{fingerprint_hash}.json"
        with open(path, "w") as f:
            json.dump(fingerprint | additional_fields, f)

    def fingerprint(self) -> dict:
        return {"hash": self.initialization_hash(), "dict": self.dict()}

    def find_base_name(self, default: str | None = None) -> str | None:
        """Returns the name of the base dataset.

        Args:
            default (str | None, optional): The default name to return if no base dataset is found. Defaults to None.

        Returns:
            str | None: The name of the base dataset.
        """
        return getattr(self.get_dataset_chain()[-1], "name", default)

    def get_dataset_chain(
        self, return_base: bool = False
    ) -> list[SampleDataset] | SampleDataset:
        """Returns the chain of datasets that lead to the current dataset.

        Args:
            return_base (bool, optional): If True, returns the base dataset. Defaults to False.

        Returns:
            list[SampleDataset] | SampleDataset: The chain of datasets that lead to the current dataset.
        """

        # the `name` attribute is used to identify the base dataset
        if hasattr(self, "name"):
            return [self] if not return_base else self

        # iterate over all attributes
        for _, attribute in self.__dict__.items():
            # if the attribute is a SampleDataset, return the chain
            if isinstance(attribute, SampleDataset):
                if return_base:
                    return attribute.get_dataset_chain(return_base=True)
                return [self] + attribute.get_dataset_chain()
        # if no dataset is found, return the current dataset
        return [self] if not return_base else self

    def get_dataloader(
        self,
        *,
        batch_size: int = 16,
        num_workers: int | None = 10,
        shuffle: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **dataloader_kwargs,
    ) -> DataLoader:
        """Returns a dataloader for the dataset

        Args:
            batch_size (int, optional): Number of samples in a batch. Defaults to 16.
            num_workers (int | None, optional): Number of workers for the dataloader. Defaults to 10.
            shuffle (bool, optional): If True, shuffles the dataset. Defaults to False.
            pin_memory (bool, optional): If True, copies the data to CUDA pinned memory. Defaults to False.
            persistent_workers (bool, optional): If True, keeps the workers alive between data loading. Defaults to False.

        Returns:
            DataLoader: The dataloader for the dataset.
        """

        return DataLoader(
            self,
            collate_fn=sample_collate,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            persistent_workers=persistent_workers,
            **dataloader_kwargs,
        )

    def sample_batch(self, batch_size: int = 12) -> Batch:
        """Returns a random batch of samples from the dataset

        Args:
            batch_size (int, optional): Number of samples in the batch. Defaults to 12.
        """
        ids = np.random.choice(range(len(self)), batch_size, replace=False)
        return Batch.init_from_samples([self[i] for i in ids])

    def as_single_batch(self) -> Batch:
        """Returns the whole dataset as a single batch

        Returns:
            Batch: The whole dataset as a single batch
        """
        return Batch.init_from_samples([self[i] for i in range(len(self))])

    def save_partition_indices(self, indices: dict[str, list[int]], partition: str):
        path = Path(GlobalContext()["partition_folder"])

        if not path.exists():
            path.mkdir(parents=True)

        file_path = path / f"{self.find_base_name()}-{partition}.json"
        # safe as json file
        with open(file_path.as_posix(), "w") as f:
            json.dump(indices, f)

    def load_partition_indices(self, partition: str) -> dict[str, list[int]]:
        # create path
        path = (
            Path(GlobalContext()["partition_folder"])
            / f"{self.find_base_name()}-{partition}.json"
        )

        # check if file exists. If not throw error
        if not path.exists():
            raise FileNotFoundError(
                f"Partition indices for dataset {self.find_base_name()} not found. "
                f"Please create them first by calling `save_partition_indices`."
            )

        # load indices
        with open(path.as_posix(), "r") as f:
            indices = json.load(f)

        return indices

    def sample(
        self,
        n: int = 8,
        *,
        show: bool = True,
        random: bool = True,
        vmin: int = 0,
        vmax: int = 5,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Plots samples of the dataset including the assigned labels

        Parameters
        ----------
        n : int, optional
            number of samples, by default 8
        show : bool, optional
            show the samples, by default True
        random : bool, optional
            if True, randomly select samples, by default True
        vmin : int, optional
            minimum label number (required for plotting masks), by default 0
        vmax : int, optional
            maximum label number (required for plotting masks), by default 5

        Returns
        -------
        matplotlib.figure.Figure
            figure containing the samples
        """
        if random:
            ids = np.random.choice(range(len(self)), n, replace=False)
        else:
            ids = [i + 50 for i in range(n)]
        data = [self[i] for i in ids]

        keys = data[0].keys()
        mask_overlap = set(keys).intersection(
            {"_lightly_augmentation", "segmentation_mask"}
        )
        n_masks = len(mask_overlap)

        if show:
            f, axes = plt.subplots(1 + n_masks, n, figsize=(n * 4, (n_masks + 1) * 4))

            # expand dims if no mask is found
            if n_masks == 0:
                axes = np.expand_dims(axes, axis=0)

            axes[0, 0].set_ylabel("image", rotation=90, size=20)
            axes[0, 0].yaxis.set_label_coords(-0.1, 0.5)
            for i, sample in enumerate(data):
                axes[0, i].imshow(sample["image"].numpy().transpose((1, 2, 0)))
                axes[0, i].set_yticks([])
                axes[0, i].set_xticks([])
                j = 0
                for k, v in sample.items():
                    if k in mask_overlap:
                        axes[j + 1, i].imshow(
                            (
                                v.numpy()
                                if len(v.shape) == 2
                                else v.numpy().transpose((1, 2, 0))
                            ),
                            vmin=vmin,
                            vmax=vmax,
                        )
                        axes[j + 1, i].set_yticks([])
                        axes[j + 1, i].set_xticks([])
                        if i == 0:
                            axes[j + 1, i].set_ylabel(k, rotation=90, size=20)
                            axes[j + 1, i].yaxis.set_label_coords(-0.1, 0.5)
                        j += 1
                    elif k == "class_label":
                        axes[0, i].set_title(
                            f"Class {int(v.item())}",
                            fontsize=25,
                        )

            return f

    def __getitem__(self, index: int) -> Sample:
        """Returns a sample from the dataset

        Args:
            index (int): index of the sample

        Returns:
            Sample: sample from the dataset
        """
        raise NotImplementedError("Please Implement this method")

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: length of the dataset
        """
        raise NotImplementedError("Please Implement this method")


class DatasetAdapter(SampleDataset):
    def __init__(self, dataset, name: str | None = None, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset
        self.metadata_args = kwargs
        self.name = name

    def _add_metadata_args(self, sample: Sample) -> Sample:
        """Adds the metadata arguments to the sample

        Args:
            sample (Sample): sample to which the metadata should be added

        Returns:
            Sample: sample with added metadata
        """
        sample.metadata.update(self.metadata_args)
        return sample

    def __getitem__(self, index: int) -> Sample:
        """Returns a sample from the dataset and adds the metadata

        Args:
            index (int): index of the sample

        Returns:
            Sample: sample from the dataset
        """
        sample = self.get_sample(index)
        return self._add_metadata_args(sample)

    def get_sample(self, index: int) -> Sample:
        """Returns a sample from the dataset

        Args:
            index (int): index of the sample

        Returns:
            Sample: sample from the dataset
        """
        raise NotImplementedError("Please Implement this method")

    def __len__(self) -> int:
        return len(self.dataset)

    def __sub__(self, other: Exclusion) -> ExclusionDataset:
        from .exclusion import ExclusionDataset

        return ExclusionDataset(dataset=self, exclusion=other)


class WrapperSampleDataset(SampleDataset):
    dataset: SampleDataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[index]
