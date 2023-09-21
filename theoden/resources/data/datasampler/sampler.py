from torch.utils.data import Sampler, BatchSampler, WeightedRandomSampler
import torch
import tqdm

from collections import Counter
from pathlib import Path
import json

from ....common import Transferable, GlobalContext
from ..dataset import SampleDataset


class DataSampler(Transferable, is_base_type=True):
    def __init__(
        self,
        overwrite: bool = False,
        **kwargs,
    ):
        self.overwrite = overwrite

    def _load_existing(self, dataset: SampleDataset) -> list[float] | None:
        """Loads existing weights from the dataset

        Args:
            dataset (SampleDataset): dataset

        Returns:
            list[float] | None: weights or None if no weights exist
        """

        dataset_hash = dataset.initialization_hash()

        return ...

    def sampler(self, dataset: SampleDataset, **kwargs) -> Sampler:
        raise NotImplementedError()


class UniformClassDataSampler(DataSampler, Transferable):
    def __init__(
        self,
        num_worker: int = 10,
        label: str = "class_label",
        **kwargs,
    ):
        """Samples batches with uniform class distribution

        Args:
            batch_size (int): batch size
            drop_last (bool, optional): drop last batch if smaller than batch_size. Defaults to False.
            num_worker (int, optional): number of workers for dataloader (only for computing the weights and not for training/validation). Defaults to 10.
            label (str, optional): label to use for computing the weights. Defaults to "class_label".
        """
        super().__init__(**kwargs)
        self.num_worker = num_worker
        self.label = label

    def _compute_weights(self, dataset: SampleDataset) -> list[float]:
        # check if weights already exist
        existing = (
            Path(GlobalContext()["partition_folder"])
            / f"{dataset.initialization_hash()}_weights.json"
        )
        if existing.exists():
            with open(existing, "r") as f:
                return json.load(f)

        labels = []

        for sample in tqdm.tqdm(
            dataset.get_dataloader(batch_size=1, num_workers=self.num_worker),
            desc="Computing weights",
        ):
            labels.append(sample[self.label].item())

        count = Counter(labels)

        # sum count
        sum_count = sum(count.values())

        # get weight per class
        weight_per_class = {c: count[c] / sum_count for c in count}

        # set sample weights to have uniform probability per class
        inverted = {c: 1 / t for c, t in weight_per_class.items()}
        sum_inverted = sum(inverted.values())
        balanced = {c: t / sum_inverted for c, t in inverted.items()}

        # return weights per sample
        sample_weights = [balanced[label] for label in labels]

        # save weights to dataset

        path = Path(GlobalContext()["partition_folder"])
        path.mkdir(parents=True, exist_ok=True)

        with open(path / f"{dataset.initialization_hash()}_weights.json", "w") as f:
            json.dump(sample_weights, f)

        return sample_weights

    def sampler(self, dataset: SampleDataset, **kwargs) -> Sampler:
        return WeightedRandomSampler(
            weights=self._compute_weights(dataset),
            num_samples=len(dataset),
            replacement=True,
        )


class MultiClassSampler(DataSampler, Transferable):
    def __init__(
        self,
        num_classes: int,
        ignore_classes: list[int] | None = None,
        start_at: int = 0,
        num_worker: int = 10,
        balancing: str = "weighted",
        label: str = "segmentation_mask",
        temperature: float | None = None,
        force: bool = False,
        **kwargs,
    ):
        """Samples batches with uniform class distribution

        Args:
            batch_size (int): batch size
            num_classes (int): number of classes
            drop_last (bool, optional): drop last batch if smaller than batch_size. Defaults to False.
            ignore_classes (list[int] | None, optional): classes to ignore. Defaults to None.
            num_worker (int, optional): number of workers for dataloader (only for computing the weights and not for training/validation). Defaults to 10.
            label (str, optional): label to use for computing the weights. Defaults to "segmentation_mask".
        """
        super().__init__(**kwargs)
        self.num_worker = num_worker
        self.label = label
        self.num_classes = num_classes
        self.ignore_classes = ignore_classes
        self.start_at = start_at
        self.balancing = balancing
        self.temperature = temperature if temperature is not None else 1 / num_classes
        self.force = force

    def _compute_weights(self, dataset: SampleDataset) -> list[float]:
        existing = (
            Path(GlobalContext()["partition_folder"])
            / f"{dataset.initialization_hash()}_weights.json"
        )
        if existing.exists() and not self.force:
            with open(existing, "r") as f:
                return json.load(f)

        labels = torch.zeros(self.num_classes, dtype=torch.int64)

        for sample in tqdm.tqdm(
            dataset.get_dataloader(batch_size=24, num_workers=self.num_worker),
            desc="Computing weights",
        ):
            labels += torch.bincount(
                sample[self.label].reshape(-1) - self.start_at,
                minlength=self.num_classes,
            )

        labels_without_ignore = labels.clone()
        if self.ignore_classes is not None:
            for c in self.ignore_classes:
                labels_without_ignore[c - self.start_at] = 0

        # sum count
        sum_count = sum(labels_without_ignore)

        # get weight per class
        weight_per_class = labels / sum_count

        print("Dist:", weight_per_class)

        # set sample weights to have uniform probability per class
        inverted = 1 / (weight_per_class + self.temperature)

        if self.ignore_classes is not None:
            for c in self.ignore_classes:
                inverted[c - self.start_at] = 0

        sum_inverted = sum(inverted)

        balanced = inverted / sum_inverted

        print("Weights:", balanced)

        weights = []

        for sample in tqdm.tqdm(
            dataset.get_dataloader(batch_size=1, num_workers=self.num_worker),
            desc="Applying weights",
        ):
            sample_dist = torch.bincount(
                sample[self.label].reshape(-1) - self.start_at,
                minlength=self.num_classes,
            )

            weight = sum(balanced * sample_dist)
            weights.append(weight.item())

        path = Path(GlobalContext()["partition_folder"])
        path.mkdir(parents=True, exist_ok=True)

        with open(path / f"{dataset.initialization_hash()}_weights.json", "w") as f:
            json.dump(weights, f)

        # return weights per sample
        return weights

    def sampler(self, dataset: SampleDataset, **kwargs) -> Sampler:
        return WeightedRandomSampler(
            weights=self._compute_weights(dataset),
            num_samples=len(dataset),
            replacement=True,
        )
