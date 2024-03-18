import random
from abc import ABC, abstractmethod

import torch
import tqdm

from ....common import Transferable
from .. import SampleDataset


class Partition(ABC, Transferable, is_base_type=True):
    def __init__(
        self, exclude: list[str] | None = None, include: list[str] | None = None
    ):
        super().__init__()
        if exclude is not None and include is not None:
            raise ValueError("Cannot specify both exclude and include")

        self.exclude = exclude
        self.include = include

    def _apply_include_exclude(
        self, indices: dict[str, list[int]]
    ) -> dict[str, list[int]]:
        if self.exclude is not None:
            for key in self.exclude:
                indices.pop(key, None)
        elif self.include is not None:
            for key in list(indices.keys()):
                if key not in self.include:
                    indices.pop(key, None)
        return indices

    @abstractmethod
    def _name(self) -> str:
        """Returns the name of the partition.

        Returns:
            str: The name of the partition.
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def __call__(
        self, dataset: SampleDataset, force_overwrite: bool = False, **kwargs
    ) -> dict[str, list[int]]:
        """Returns a dictionary of partition_name: list of indices in the partition.

        Args:
            dataset (SampleDataset): The dataset to partition.
            force_overwrite (bool, optional): Whether to overwrite the partition if it already exists. Defaults to False.

        Returns:
            dict[str, list[int]]: A dictionary of partition_name: list of indices in the partition.
        """
        raise NotImplementedError("Please implement this method")


class IndexPartition(Partition, Transferable):
    def _name(self) -> str:
        return "IP"

    def __call__(
        self, dataset: SampleDataset, force_overwrite: bool = False, **kwargs
    ) -> dict[str, list[int]]:
        return {i: [i] for i in range(len(dataset))}


class MetadataPartition(Partition):
    def __init__(
        self,
        metadata_key: str,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ):
        super().__init__(exclude, include)
        self.metadata_key = metadata_key

    def _name(self) -> str:
        return f"MP_{self.metadata_key}"

    def __call__(
        self, dataset: SampleDataset, force_overwrite: bool = False, **kwargs
    ) -> dict[str, list[int]]:
        if not force_overwrite:
            try:
                indices = dataset.load_partition_indices(self._name())
                return indices
            except FileNotFoundError:
                pass
        indices = {}
        for i, sample in enumerate(
            tqdm.tqdm(dataset.get_dataloader(batch_size=1), desc="Partitioning dataset")
        ):
            metadata = sample.metadata[self.metadata_key][0]
            if str(metadata) not in indices:
                indices[str(metadata)] = []
            indices[str(metadata)].append(i)

        dataset.save_partition_indices(indices, self._name())
        return self._apply_include_exclude(indices)


class DataPartition(Partition):
    def __init__(
        self,
        data_key: str,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ):
        super().__init__(exclude, include)
        self.data_key = data_key

    def _name(self) -> str:
        return f"DP_{self.data_key}"

    def __call__(
        self, dataset: SampleDataset, force_overwrite: bool = False, **kwargs
    ) -> dict[str, list[int]]:
        if not force_overwrite:
            try:
                indices = dataset.load_partition_indices(self._name())
                return indices
            except FileNotFoundError:
                pass

        indices = {}
        for i, sample in enumerate(tqdm.tqdm(dataset, desc="Partitioning dataset")):
            data = sample[self.data_key]

            # if the data is a tensor, convert it to a scalar
            if isinstance(data, torch.Tensor) and len(data.shape) == 0:
                data = data.item()

            # if the data is a list, convert it to a string
            if str(data) not in indices:
                indices[str(data)] = []
            indices[str(data)].append(i)

        dataset.save_partition_indices(indices, self._name())

        return self._apply_include_exclude(indices)


class ClassLabelPartition(DataPartition):
    def __init__(
        self,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ):
        super().__init__("class_label", exclude, include)
