from __future__ import annotations

import torch
from torch.utils.data.dataloader import default_collate

from .metadata import Metadata, MetadataBatch, init_sample_metadata


def init_augmentation(sample: Sample) -> Sample:
    sample.metadata = init_sample_metadata(sample.metadata)
    return sample


class Sample(dict):
    def __init__(
        self,
        data: dict[str, any] = None,
        metadata: Metadata | MetadataBatch | dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # create metadata object
        if not metadata:
            self.metadata = Metadata()
        else:
            if isinstance(metadata, Metadata | MetadataBatch):
                self.metadata = metadata
            elif isinstance(metadata, dict):
                self.metadata = Metadata(metadata)
            else:
                raise TypeError("Metadata should be dictionary or Metadata object")

                # update parent class defaultdictionary with the data dictionary
        if isinstance(data, dict):
            self.update(data)
        self.update(kwargs)

    def set_metadata(self, key: str, value: any, comment: str | None = None) -> None:
        self.metadata[key] = value
        if isinstance(comment, str):
            self.metadata.add_comment(key, comment)

    def to(self, device: str) -> Sample:
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self[k] = v.to(device)
        return self

    def print_data_types(self) -> None:
        print("\033[1mKey: Data Type\033[0m")
        for key, value in self.items():
            print(
                f"\033[1m{key}\033[0m: {type(value).__module__ + '.' + type(value).__name__}"
            )

    def __contains__(self, item: str) -> bool:
        return item in self.keys()

    def __repr__(self) -> str:
        # datatypes for each key, metadata
        types = {k: type(v) for k, v in self.items()}
        return f"Sample({types}, {self.metadata})"


class Batch(Sample):
    def __init__(
        self,
        data: dict[str, any] = None,
        metadata: MetadataBatch | None = None,
        **kwargs,
    ) -> None:
        super().__init__(data, metadata, **kwargs)
        assert isinstance(self.metadata, MetadataBatch)
        self.metadata: MetadataBatch = self.metadata

    @staticmethod
    def init_from_samples(samples: list[Sample]) -> Batch:
        meta = MetadataBatch([])
        for sample in samples:
            meta.batch.append(sample.metadata)
        return Batch(default_collate(samples), meta)


def sample_collate(d: list[Sample]) -> Batch:
    """Collation function for torch dataset, that can handle Sample as datatype

    Args:
        d (list[Sample]): List of samples

    Returns:
        Batch: Batch of samples
    """
    return Batch.init_from_samples(d)
