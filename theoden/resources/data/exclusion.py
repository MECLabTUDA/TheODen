import numpy as np
import torch

from ...common import GlobalContext, Transferable
from . import SampleDataset
from .sample import Sample
from .subset import SubsetDataset


class Exclusion(Transferable, is_base_type=True):
    def ex(self, sample: Sample) -> bool:
        raise NotImplementedError("Exclusion is an abstract class")


class SegmentationExclusion(Exclusion, Transferable):
    def __init__(
        self,
        class_amount: list[tuple[int, float]] | None = None,
        contains_classes: list[int] | None = None,
        min_num_classes: int | None = None,
        max_num_classes: int | None = None,
        label_key: str = "segmentation_mask",
    ) -> None:
        self.class_amount = class_amount
        self.contains_classes = contains_classes
        self.min_num_classes = min_num_classes
        self.max_num_classes = max_num_classes
        self.label_key = label_key

    def ex(self, sample: Sample) -> bool:
        patch: torch.Tensor = sample[self.label_key]

        if self.class_amount is not None:
            for class_, amount in self.class_amount:
                # check if percentage of class_ is higher than amount
                if (torch.sum(patch == class_) / torch.numel(patch)) > amount:
                    return True

        if self.contains_classes is not None:
            # check if patch contains any of the classes
            if torch.sum(
                [torch.sum(patch == class_) > 0 for class_ in self.contains_classes]
            ):
                return True

        if self.min_num_classes is not None:
            # check if patch contains at least min_num_classes
            if (
                torch.sum(
                    [torch.sum(patch == class_) > 0 for class_ in self.contains_classes]
                )
                < self.min_num_classes
            ):
                return True

        if self.max_num_classes is not None:
            # check if patch contains at most max_num_classes
            if (
                torch.sum(
                    [torch.sum(patch == class_) > 0 for class_ in self.contains_classes]
                )
                > self.max_num_classes
            ):
                return True

        return False


class ClassificationExclusion(Exclusion, Transferable):
    def __init__(
        self,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
        label_key: str = "class_label",
    ):
        super().__init__()
        self.include = include
        self.exclude = exclude
        self.label_key = label_key

        if self.include is None and self.exclude is None:
            raise ValueError("Either include or exclude must be specified")

    def ex(self, sample: Sample) -> bool:
        return (
            self.include is not None
            and sample[self.label_key] not in self.include
            or self.exclude is not None
            and sample[self.label_key] in self.exclude
        )


class MetadataExclusion(Exclusion, Transferable):
    def __init__(
        self,
        metadata_key: str,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ):
        super().__init__()
        self.metadata_key = metadata_key
        self.include = include
        self.exclude = exclude

        if self.include is None and self.exclude is None:
            raise ValueError("Either include or exclude must be specified")

    def ex(self, sample: Sample) -> bool:
        return (
            self.include is not None
            and sample.metadata[self.metadata_key] not in self.include
            or self.exclude is not None
            and sample.metadata[self.metadata_key] in self.exclude
        )


class ExclusionDataset(SubsetDataset, Transferable):
    def __init__(self, dataset: SampleDataset, exclusion: Exclusion, **kwargs):
        super().__init__(dataset=dataset, indices=None)
        self.exclusion = exclusion
        self.force = kwargs.get("force", False)
        self.kwargs = kwargs
        self.name = self.initialization_hash()

    def init_after_deserialization(self) -> Transferable:
        self.indices = self.get_indices(
            folder=GlobalContext()["partition_folder"],
            mask_fn=self.exclusion.ex,
            show_progress=True,
            description="Excluding samples",
            force=self.force,
            invert=True,
        )
        print("INDICES", self.indices)
        return self

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def initialization_hash(self, exclude_keys: list[str] | None = None) -> str:
        return super().initialization_hash(
            exclude_keys=list(self.kwargs.keys()) + (exclude_keys or [])
        )
